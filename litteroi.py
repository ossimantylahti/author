#!/usr/bin/env python3
"""MP3 transcription CLI using the OpenAI Audio API.

Examples:
    python3 litteroi.py interview.mp3
    python3 litteroi.py interview.mp3 /start=30 /stop=95
    python3 litteroi.py interview.mp3 --start 30 --stop 95
    python3 litteroi.py interview.mp3 --srt
    python3 litteroi.py interview.mp3 --diarize
    python3 litteroi.py interview.mp3 --diarize --speaker A=GUEST --speaker B=HOST

Notes:
- The script reads OPENAI_API_KEY from the environment.
- If /start or /stop is used, the script trims the MP3 locally with ffmpeg first.
- The transcript is written automatically to a .txt file next to the source MP3.
- Optionally, the script can also create an .srt subtitle file.
- Optionally, the script can request speaker diarization and label speakers in the output.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Any, Optional

from openai import OpenAI

MODEL = "gpt-4o-transcribe"
DIARIZE_MODEL = "gpt-4o-transcribe-diarize"
DEFAULT_LANGUAGE = "fi"
ALLOWED_EXTENSIONS = {".mp3"}
DEFAULT_MAX_FILE_MB = 20.0
DEFAULT_CHUNK_SECONDS = 900.0
DEFAULT_SPEAKER_COUNT = 2
DEFAULT_CONTEXT_CHARS = 1200

client = OpenAI()  # uses OPENAI_API_KEY from the environment


@dataclass
class ChunkInfo:
    path: str
    index: int
    total: int
    start_offset: float
    duration: float


def clean_text(s: str) -> str:
    return s.encode("utf-8", errors="ignore").decode("utf-8")


def normalise_legacy_args(argv: list[str]) -> list[str]:
    """Accept legacy /start= and /stop= syntax in addition to argparse style flags."""
    normalised: list[str] = []
    for arg in argv:
        if arg.startswith("/start="):
            normalised.extend(["--start", arg.split("=", 1)[1]])
        elif arg.startswith("/stop="):
            normalised.extend(["--stop", arg.split("=", 1)[1]])
        else:
            normalised.append(arg)
    return normalised


def parse_speaker_mapping(values: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(
                f"Invalid --speaker value '{value}'. Use the form A=GUEST or B=HOST."
            )
        key, label = value.split("=", 1)
        key = key.strip()
        label = label.strip()
        if not key or not label:
            raise ValueError(
                f"Invalid --speaker value '{value}'. Use the form A=GUEST or B=HOST."
            )
        mapping[key] = label
    return mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe an MP3 file and print only the recognised speech."
    )
    parser.add_argument("file", help="Input audio file (.mp3)")
    parser.add_argument("--start", type=float, default=None, help="Start time in seconds")
    parser.add_argument("--stop", type=float, default=None, help="Stop time in seconds")
    parser.add_argument(
        "--language",
        default=DEFAULT_LANGUAGE,
        help=f"Language hint for transcription (default: {DEFAULT_LANGUAGE})",
    )
    parser.add_argument(
        "--model",
        default=MODEL,
        help=f"Transcription model (default: {MODEL})",
    )
    parser.add_argument(
        "--diarize",
        dest="diarize",
        action="store_true",
        default=True,
        help="Use the diarization model and label speakers in the output. This is enabled by default.",
    )
    parser.add_argument(
        "--no-diarize",
        dest="diarize",
        action="store_false",
        help="Disable speaker diarization and use plain transcription.",
    )
    parser.add_argument(
        "--speaker",
        action="append",
        default=[],
        help="Rename diarized speakers, e.g. --speaker A=GUEST --speaker B=HOST",
    )
    parser.add_argument(
        "--speaker-count",
        type=int,
        default=DEFAULT_SPEAKER_COUNT,
        help=(
            "Expected number of speakers for diarization normalisation "
            f"(default: {DEFAULT_SPEAKER_COUNT}, suitable for phone calls)."
        ),
    )
    parser.add_argument(
        "--context-chars",
        type=int,
        default=DEFAULT_CONTEXT_CHARS,
        help=(
            "Number of previous transcript characters sent as context for later chunks "
            f"(default: {DEFAULT_CONTEXT_CHARS})."
        ),
    )
    parser.add_argument(
        "--srt",
        action="store_true",
        help="Also write an .srt subtitle file.",
    )
    parser.add_argument(
        "--max-file-mb",
        type=float,
        default=DEFAULT_MAX_FILE_MB,
        help=f"Maximum MP3 size before chunking (default: {DEFAULT_MAX_FILE_MB:g} MB).",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=DEFAULT_CHUNK_SECONDS,
        help=f"Chunk duration in seconds when chunking is needed (default: {DEFAULT_CHUNK_SECONDS:g}).",
    )
    args = parser.parse_args(normalise_legacy_args(sys.argv[1:]))
    args.speaker_map = parse_speaker_mapping(args.speaker)
    return args


def validate_args(
    path: str,
    start: Optional[float],
    stop: Optional[float],
    diarize: bool,
    model: str,
    max_file_mb: float,
    chunk_seconds: float,
) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    _, ext = os.path.splitext(path.lower())
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError("Only .mp3 files are supported.")

    if start is not None and start < 0:
        raise ValueError("--start must be 0 or greater.")

    if stop is not None and stop < 0:
        raise ValueError("--stop must be 0 or greater.")

    if start is not None and stop is not None and stop <= start:
        raise ValueError("--stop must be greater than --start.")

    if diarize and model != MODEL:
        raise ValueError(
            "--diarize selects the diarization model automatically, so do not combine it with a custom --model."
        )
    if max_file_mb <= 0:
        raise ValueError("--max-file-mb must be greater than 0.")
    if chunk_seconds <= 0:
        raise ValueError("--chunk-seconds must be greater than 0.")


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def build_trimmed_mp3(source_path: str, start: Optional[float], stop: Optional[float]) -> str:
    """Return path to a temporary trimmed MP3 if time range is requested."""
    if start is None and stop is None:
        return source_path

    if not ffmpeg_available():
        raise RuntimeError(
            "ffmpeg is required when using --start/--stop (or /start=/ /stop=)."
        )

    temp_dir = tempfile.mkdtemp(prefix="litteroi_")
    output_path = os.path.join(temp_dir, "segment.mp3")

    cmd = ["ffmpeg", "-y"]

    if start is not None:
        cmd.extend(["-ss", str(start)])

    cmd.extend(["-i", source_path])

    if start is not None and stop is not None:
        duration = stop - start
        cmd.extend(["-t", str(duration)])
    elif stop is not None:
        cmd.extend(["-t", str(stop)])

    cmd.extend([
        "-vn",
        "-acodec",
        "libmp3lame",
        "-q:a",
        "2",
        output_path,
    ])

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.strip()}")

    return output_path


def get_audio_duration_seconds(path: str) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed while reading duration: {result.stderr.strip()}")
    try:
        return max(0.0, float(result.stdout.strip()))
    except ValueError as exc:
        raise RuntimeError("Could not parse duration from ffprobe output.") from exc


def split_mp3_to_chunks(source_path: str, chunk_seconds: float, temp_dir: str) -> list[ChunkInfo]:
    output_pattern = os.path.join(temp_dir, "chunk_%04d.mp3")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        source_path,
        "-f",
        "segment",
        "-segment_time",
        str(chunk_seconds),
        "-c",
        "copy",
        output_pattern,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg chunking failed: {result.stderr.strip()}")

    chunk_paths = sorted(
        os.path.join(temp_dir, name)
        for name in os.listdir(temp_dir)
        if name.startswith("chunk_") and name.endswith(".mp3")
    )
    if not chunk_paths:
        raise RuntimeError("ffmpeg did not produce any MP3 chunks.")

    chunk_infos: list[ChunkInfo] = []
    cumulative_offset = 0.0
    total = len(chunk_paths)
    for idx, chunk_path in enumerate(chunk_paths, start=1):
        duration = get_audio_duration_seconds(chunk_path)
        chunk_infos.append(
            ChunkInfo(
                path=chunk_path,
                index=idx,
                total=total,
                start_offset=cumulative_offset,
                duration=duration,
            )
        )
        cumulative_offset += duration
    return chunk_infos


def add_offset_to_segments(segments: list[dict[str, Any]], offset_seconds: float) -> list[dict[str, Any]]:
    adjusted: list[dict[str, Any]] = []
    for segment in segments:
        item = dict(segment)
        try:
            item["start"] = float(item.get("start", 0.0)) + offset_seconds
        except (TypeError, ValueError):
            item["start"] = offset_seconds
        try:
            item["end"] = float(item.get("end", 0.0)) + offset_seconds
        except (TypeError, ValueError):
            item["end"] = offset_seconds
        adjusted.append(item)
    return adjusted


def normalise_chunk_speakers(
    segments: list[dict[str, Any]],
    speaker_count: int,
    chunk_index: int,
) -> list[dict[str, Any]]:
    """Map per-chunk diarization speaker labels to stable call-level labels.

    The OpenAI diarization response may restart speaker labels for each chunk.
    For phone calls, the usual case is two speakers, so this function normalises
    the first distinct local speaker in a chunk to A, the second to B, and so on.

    This is intentionally conservative: it does not claim to recognise voices
    acoustically across chunks. It makes long, chunked phone-call output more
    readable and predictable, while preserving the original local label.
    """
    if speaker_count <= 0:
        raise ValueError("--speaker-count must be greater than 0.")

    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    local_to_global: dict[str, str] = {}
    next_index = 0
    normalised: list[dict[str, Any]] = []

    for segment in segments:
        item = dict(segment)
        raw_speaker = item.get("speaker")
        local_label = str(raw_speaker).strip() if raw_speaker is not None else ""

        if not local_label:
            local_label = "UNKNOWN"

        if local_label not in local_to_global:
            if next_index < len(labels):
                global_label = labels[next_index]
            else:
                global_label = f"Speaker {next_index + 1}"
            local_to_global[local_label] = global_label
            next_index += 1

        item["original_speaker"] = local_label
        item["speaker"] = local_to_global[local_label]
        item["chunk_index"] = chunk_index

        if next_index > speaker_count:
            item["speaker_warning"] = (
                f"More than {speaker_count} speakers detected in chunk {chunk_index}."
            )

        normalised.append(item)

    return normalised


def merge_adjacent_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge adjacent diarized segments from the same speaker for cleaner output."""
    merged: list[dict[str, Any]] = []

    for segment in merge_adjacent_segments(segments):
        text = segment.get("text", "").strip()
        if not text:
            continue

        speaker = segment.get("speaker")
        current = dict(segment)

        if merged and merged[-1].get("speaker") == speaker:
            previous_text = merged[-1].get("text", "").strip()
            merged[-1]["text"] = f"{previous_text} {text}".strip()
            merged[-1]["end"] = current.get("end", merged[-1].get("end"))
            continue

        merged.append(current)

    return merged


def get_output_stem(source_path: str, start: Optional[float], stop: Optional[float]) -> str:
    base, _ = os.path.splitext(source_path)
    if start is None and stop is None:
        return base

    start_part = "start" if start is None else f"{start:g}"
    stop_part = "end" if stop is None else f"{stop:g}"
    return f"{base}_{start_part}-{stop_part}"


def get_output_txt_path(source_path: str, start: Optional[float], stop: Optional[float]) -> str:
    return f"{get_output_stem(source_path, start, stop)}.txt"


def get_output_srt_path(source_path: str, start: Optional[float], stop: Optional[float]) -> str:
    return f"{get_output_stem(source_path, start, stop)}.srt"


def write_text_file(output_path: str, content: str) -> None:
    with open(output_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)
        if content and not content.endswith("\n"):
            handle.write("\n")


def object_to_dict(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, list):
        return [object_to_dict(item) for item in value]
    if isinstance(value, dict):
        return {key: object_to_dict(item) for key, item in value.items()}
    return value


def extract_text(response: Any) -> str:
    if isinstance(response, str):
        return clean_text(response).strip()

    text = getattr(response, "text", None)
    if isinstance(text, str):
        return clean_text(text).strip()

    payload = object_to_dict(response)
    if isinstance(payload, dict) and isinstance(payload.get("text"), str):
        return clean_text(payload["text"]).strip()

    return clean_text(str(response)).strip()


def extract_segments(response: Any) -> list[dict[str, Any]]:
    payload = object_to_dict(response)
    segments = []
    if isinstance(payload, dict):
        segments = payload.get("segments") or []
    if not isinstance(segments, list):
        return []

    cleaned: list[dict[str, Any]] = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        cleaned.append(
            {
                "start": segment.get("start"),
                "end": segment.get("end"),
                "text": clean_text(str(segment.get("text", ""))).strip(),
                "speaker": segment.get("speaker"),
            }
        )
    return cleaned


def choose_model(diarize: bool, requested_model: str) -> str:
    return DIARIZE_MODEL if diarize else requested_model


def transcribe_mp3(
    path: str,
    model: str,
    language: str,
    diarize: bool,
    want_srt: bool,
    prompt: Optional[str] = None,
) -> Any:
    response_format = "text"
    request_kwargs: dict[str, Any] = {}

    if prompt:
        request_kwargs["prompt"] = prompt

    if diarize:
        response_format = "diarized_json"
        request_kwargs["chunking_strategy"] = "auto"
    elif want_srt:
        response_format = "verbose_json"
        request_kwargs["timestamp_granularities"] = ["segment"]

    with open(path, "rb") as audio_file:
        return client.audio.transcriptions.create(
            model=model,
            file=audio_file,
            language=language,
            response_format=response_format,
            **request_kwargs,
        )


def format_seconds_for_srt(seconds: Any) -> str:
    try:
        value = float(seconds)
    except (TypeError, ValueError):
        value = 0.0

    if value < 0:
        value = 0.0

    total_ms = int(round(value * 1000))
    hours = total_ms // 3_600_000
    remainder = total_ms % 3_600_000
    minutes = remainder // 60_000
    remainder %= 60_000
    secs = remainder // 1000
    millis = remainder % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def build_default_speaker_map(
    segments: list[dict[str, Any]],
    user_map: dict[str, str],
) -> dict[str, str]:
    resolved = dict(user_map)
    next_index = 1

    for segment in segments:
        raw_speaker = segment.get("speaker")
        label = str(raw_speaker).strip() if raw_speaker is not None else ""
        if not label:
            continue
        if label in resolved:
            continue
        resolved[label] = f"Speaker {next_index}"
        next_index += 1

    return resolved


def remap_speaker(speaker: Any, speaker_map: dict[str, str]) -> str:
    label = str(speaker).strip() if speaker is not None else ""
    if not label:
        return "Speaker"
    return speaker_map.get(label, label)


def build_plain_text(text: str) -> str:
    return text.strip()


def build_chunk_context_prompt(
    previous_segments: list[dict[str, Any]],
    speaker_map: dict[str, str],
    context_chars: int,
) -> Optional[str]:
    """Build a compact speaker/context hint for a later chunk.

    This does not provide biometric speaker recognition. It gives the
    transcription model recent textual continuity and expected speaker labels,
    which is often useful for two-person phone calls split into chunks.
    """
    if context_chars <= 0 or not previous_segments:
        return None

    recent_lines: list[str] = []
    for segment in merge_adjacent_segments(previous_segments[-12:]):
        text = segment.get("text", "").strip()
        if not text:
            continue
        speaker = remap_speaker(segment.get("speaker"), speaker_map)
        recent_lines.append(f"{speaker}: {text}")

    recent_text = "\n".join(recent_lines).strip()
    if not recent_text:
        return None

    if len(recent_text) > context_chars:
        recent_text = recent_text[-context_chars:]

    known_speakers = ", ".join(sorted(set(speaker_map.values()))) or "A, B"

    return (
        "This is a Finnish phone-call recording split into chunks. "
        "There are normally two speakers. Keep speaker labels consistent with "
        f"the previous context when possible. Known speaker labels: {known_speakers}. "
        "Recent previous transcript context:\n"
        f"{recent_text}"
    )


def build_diarized_text(segments: list[dict[str, Any]], speaker_map: dict[str, str]) -> str:
    lines: list[str] = []
    for segment in merge_adjacent_segments(segments):
        text = segment.get("text", "").strip()
        if not text:
            continue
        speaker = remap_speaker(segment.get("speaker"), speaker_map)
        lines.append(f"{speaker}: {text}")
    return "\n".join(lines).strip()


def build_srt(
    segments: list[dict[str, Any]],
    speaker_map: dict[str, str],
    include_speakers: bool,
    start_index: int = 1,
) -> str:
    blocks: list[str] = []
    index = start_index

    for segment in segments:
        text = segment.get("text", "").strip()
        if not text:
            continue

        start = format_seconds_for_srt(segment.get("start"))
        end = format_seconds_for_srt(segment.get("end"))

        if include_speakers:
            speaker = remap_speaker(segment.get("speaker"), speaker_map)
            body = f"{speaker}: {text}"
        else:
            body = text

        blocks.append(f"{index}\n{start} --> {end}\n{body}")
        index += 1

    return "\n\n".join(blocks).strip()


def main() -> None:
    try:
        args = parse_args()
        if args.speaker_count <= 0:
            raise ValueError("--speaker-count must be greater than 0.")
        if args.context_chars < 0:
            raise ValueError("--context-chars must be 0 or greater.")
        validate_args(
            args.file, args.start, args.stop, args.diarize, args.model, args.max_file_mb, args.chunk_seconds
        )

        actual_model = choose_model(args.diarize, args.model)
        path_to_send = build_trimmed_mp3(args.file, args.start, args.stop)
        file_size_bytes = os.path.getsize(path_to_send)
        file_size_mb = file_size_bytes / (1024 * 1024)
        size_limit_bytes = int(args.max_file_mb * 1024 * 1024)
        print(
            f"Input file size: {file_size_mb:.2f} MB, chunking threshold: {args.max_file_mb:.2f} MB",
            file=sys.stderr,
        )

        if shutil.which("ffprobe") is None:
            raise RuntimeError(
                "ffprobe is required for reading audio duration. Install ffmpeg tools and try again."
            )

        audio_duration_seconds = get_audio_duration_seconds(path_to_send)
        should_chunk = (
            file_size_bytes > size_limit_bytes
            or audio_duration_seconds > args.chunk_seconds
        )

        print(
            (
                f"Input duration: {audio_duration_seconds:.2f}s, "
                f"chunk duration threshold: {args.chunk_seconds:.2f}s"
            ),
            file=sys.stderr,
        )

        texts: list[str] = []
        segments: list[dict[str, Any]] = []
        speaker_map = {
            chr(ord("A") + i): f"Speaker {i + 1}"
            for i in range(min(args.speaker_count, 26))
        }
        speaker_map.update(args.speaker_map)
        if not should_chunk:
            response = transcribe_mp3(
                path_to_send,
                actual_model,
                args.language,
                args.diarize,
                args.srt,
                prompt=None,
            )
            texts.append(extract_text(response))
            segments = extract_segments(response)
            if args.diarize:
                segments = normalise_chunk_speakers(
                    segments,
                    args.speaker_count,
                    1,
                )
            print("Transcription completed without chunking.", file=sys.stderr)
        else:
            if shutil.which("ffmpeg") is None:
                raise RuntimeError(
                    "ffmpeg is required for chunking files. Install ffmpeg and try again."
                )
            print("Starting chunking with ffmpeg.", file=sys.stderr)
            with tempfile.TemporaryDirectory(prefix="litteroi_chunks_") as chunk_dir:
                chunks = split_mp3_to_chunks(path_to_send, args.chunk_seconds, chunk_dir)
                print(f"Chunking complete. Created {len(chunks)} chunks.", file=sys.stderr)
                for chunk in chunks:
                    print(
                        (
                            f"Transcribing chunk {chunk.index}/{chunk.total} "
                            f"(duration {chunk.duration:.2f}s, start offset {chunk.start_offset:.2f}s)"
                        ),
                        file=sys.stderr,
                    )
                    try:
                        context_prompt = None
                        if args.diarize and chunk.index > 1:
                            context_prompt = build_chunk_context_prompt(
                                segments,
                                speaker_map,
                                args.context_chars,
                            )

                        response = transcribe_mp3(
                            chunk.path,
                            actual_model,
                            args.language,
                            args.diarize,
                            args.srt,
                            prompt=context_prompt,
                        )
                    except Exception as exc:
                        raise RuntimeError(
                            f"Transcription failed for chunk {chunk.index}/{chunk.total}: {chunk.path}"
                        ) from exc
                    texts.append(extract_text(response))
                    chunk_segments = extract_segments(response)
                    if args.diarize:
                        chunk_segments = normalise_chunk_speakers(
                            chunk_segments,
                            args.speaker_count,
                            chunk.index,
                        )
                    segments.extend(add_offset_to_segments(chunk_segments, chunk.start_offset))
                    print(f"Chunk {chunk.index}/{chunk.total} transcribed successfully.", file=sys.stderr)

        text = "\n".join(part for part in texts if part.strip()).strip()
        speaker_map = build_default_speaker_map(segments, speaker_map)

        if args.diarize:
            transcript = build_diarized_text(segments, speaker_map)
            if not transcript:
                transcript = text
        else:
            transcript = build_plain_text(text)

        output_txt = get_output_txt_path(args.file, args.start, args.stop)
        print("Starting transcript merge and write.", file=sys.stderr)
        write_text_file(output_txt, transcript)
        print(transcript)

        print(f"\nSaved transcript to: {output_txt}", file=sys.stderr)

        if args.srt:
            if not segments:
                raise RuntimeError(
                    "The API response did not contain timestamped segments, so .srt could not be generated."
                )
            output_srt = get_output_srt_path(args.file, args.start, args.stop)
            srt_content = build_srt(segments, speaker_map, include_speakers=args.diarize)
            write_text_file(output_srt, srt_content)
            print(f"Saved subtitles to: {output_srt}", file=sys.stderr)
        print("Merging complete.", file=sys.stderr)

    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
