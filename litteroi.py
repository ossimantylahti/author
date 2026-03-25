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
from typing import Any, Optional

from openai import OpenAI

MODEL = "gpt-4o-transcribe"
DIARIZE_MODEL = "gpt-4o-transcribe-diarize"
DEFAULT_LANGUAGE = "fi"
ALLOWED_EXTENSIONS = {".mp3"}

client = OpenAI()  # uses OPENAI_API_KEY from the environment


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
        action="store_true",
        help="Use the diarization model and label speakers in the output.",
    )
    parser.add_argument(
        "--speaker",
        action="append",
        default=[],
        help="Rename diarized speakers, e.g. --speaker A=GUEST --speaker B=HOST",
    )
    parser.add_argument(
        "--srt",
        action="store_true",
        help="Also write an .srt subtitle file.",
    )
    args = parser.parse_args(normalise_legacy_args(sys.argv[1:]))
    args.speaker_map = parse_speaker_mapping(args.speaker)
    return args


def validate_args(path: str, start: Optional[float], stop: Optional[float], diarize: bool, model: str) -> None:
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


def transcribe_mp3(path: str, model: str, language: str, diarize: bool, want_srt: bool) -> Any:
    response_format = "text"
    request_kwargs: dict[str, Any] = {}

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


def build_diarized_text(segments: list[dict[str, Any]], speaker_map: dict[str, str]) -> str:
    lines: list[str] = []
    for segment in segments:
        text = segment.get("text", "").strip()
        if not text:
            continue
        speaker = remap_speaker(segment.get("speaker"), speaker_map)
        lines.append(f"{speaker}: {text}")
    return "\n".join(lines).strip()


def build_srt(segments: list[dict[str, Any]], speaker_map: dict[str, str], include_speakers: bool) -> str:
    blocks: list[str] = []
    index = 1

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
        validate_args(args.file, args.start, args.stop, args.diarize, args.model)

        actual_model = choose_model(args.diarize, args.model)
        path_to_send = build_trimmed_mp3(args.file, args.start, args.stop)
        response = transcribe_mp3(
            path_to_send,
            actual_model,
            args.language,
            args.diarize,
            args.srt,
        )

        text = extract_text(response)
        segments = extract_segments(response)
        speaker_map = build_default_speaker_map(segments, args.speaker_map)

        if args.diarize:
            transcript = build_diarized_text(segments, speaker_map)
            if not transcript:
                transcript = text
        else:
            transcript = build_plain_text(text)

        output_txt = get_output_txt_path(args.file, args.start, args.stop)
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

    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
