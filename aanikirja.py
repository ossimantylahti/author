#!/usr/bin/env python3
"""ElevenLabs audiobook generator from SSML/text file."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from urllib import error, request
import shutil
import random

API_BASE = "https://api.elevenlabs.io/v1"
OUTPUT_FORMAT = "mp3_44100_128"
DEFAULT_NARRATORS_FILE = "prompt_narrators.txt"
MAX_TEXT_LEN = 10000
DEFAULT_CHUNK_LIMIT = 9500
NARRATOR_NAME = "Kertoja"
MODEL_MAP = {"v2": "eleven_multilingual_v2", "v3": "eleven_v3"}


def load_narrators(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Kertojatiedoston pitää olla JSON-objekti.")
    return {str(k): str(v) for k, v in data.items()}


def strip_speak_wrappers(text: str) -> str:
    t = text.strip()
    t = re.sub(r"^\s*<speak[^>]*>", "", t, flags=re.IGNORECASE)
    t = re.sub(r"</speak>\s*$", "", t, flags=re.IGNORECASE)
    return t.strip()


def extract_speaker_segments(body: str) -> list[tuple[str, str]]:
    pattern = re.compile(r'<voice\s+[^>]*name="([^"]+)"[^>]*>.*?</voice>', re.IGNORECASE | re.DOTALL)
    segments: list[tuple[str, str]] = []
    pos = 0
    for m in pattern.finditer(body):
        before = body[pos:m.start()].strip()
        if before:
            segments.append((NARRATOR_NAME, before))
        segments.append((m.group(1).strip() or NARRATOR_NAME, m.group(0).strip()))
        pos = m.end()
    tail = body[pos:].strip()
    if tail:
        segments.append((NARRATOR_NAME, tail))
    return segments


def force_split_text(text: str, chunk_limit: int) -> list[str]:
    parts: list[str] = []
    remaining = text.strip()
    while len(remaining) > chunk_limit:
        cut = remaining.rfind(" ", 0, chunk_limit)
        if cut <= 0:
            cut = chunk_limit
        parts.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    if remaining:
        parts.append(remaining)
    return parts


def split_large_text(text: str, chunk_limit: int) -> list[str]:
    if len(text) <= chunk_limit:
        return [text]

    pieces: list[str] = []
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    current = ""

    for block in blocks:
        cand = f"{current}\n\n{block}".strip() if current else block
        if len(cand) <= chunk_limit:
            current = cand
            continue

        if current:
            pieces.append(current)
            current = ""

        if len(block) <= chunk_limit:
            current = block
            continue

        part = ""
        for sentence in re.split(r"(?<=[.!?])\s+", block):
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(sentence) > chunk_limit:
                if part:
                    pieces.append(part)
                    part = ""
                pieces.extend(force_split_text(sentence, chunk_limit))
                continue

            cand2 = f"{part} {sentence}".strip() if part else sentence
            if len(cand2) <= chunk_limit:
                part = cand2
            else:
                if part:
                    pieces.append(part)
                part = sentence

        if part:
            current = part

    if current:
        pieces.append(current)

    return pieces




def split_voice_segment(segment_text: str, chunk_limit: int) -> list[str]:
    m = re.match(r'^<voice\s+([^>]*)>(.*)</voice>$', segment_text.strip(), flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return split_large_text(segment_text, chunk_limit)

    inner = m.group(2).strip()
    return [part for part in split_large_text(inner, chunk_limit) if part.strip()]


def merge_mp3_parts(out_dir: Path, merged_path: Path) -> None:
    part_files = sorted(out_dir.glob('*.mp3'))
    if not part_files:
        raise RuntimeError('Ei mp3-osia yhdistettäväksi.')

    list_file = out_dir / 'concat_list.txt'
    lines = [f"file '{f.resolve()}'" for f in part_files]
    list_file.write_text("\n".join(lines) + "\n", encoding='utf-8')

    cmd = [
        'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
        '-i', str(list_file), '-c', 'copy', str(merged_path)
    ]
    subprocess.run(cmd, check=True)


def is_pause_only_ssml(text: str) -> bool:
    stripped = re.sub(r"<break\b[^>]*/>", "", text, flags=re.IGNORECASE)
    stripped = re.sub(r"<[^>]+>", "", stripped)
    return not stripped.strip()


def split_ssml_chunks(text: str, chunk_limit: int) -> list[tuple[str, str]]:
    body = strip_speak_wrappers(text)
    segments = extract_speaker_segments(body)
    chunks: list[tuple[str, str]] = []
    pending_pause = ""
    pending_speaker = NARRATOR_NAME

    for speaker, segment_text in segments:
        pending_speaker = speaker
        for piece in split_voice_segment(segment_text, chunk_limit):
            piece = piece.strip()
            if not piece:
                continue

            if is_pause_only_ssml(piece):
                pending_pause = f"{pending_pause}\n{piece}".strip() if pending_pause else piece
                continue

            if pending_pause:
                piece = f"{pending_pause}\n{piece}"
                pending_pause = ""

            chunk = f"<speak>\n{piece}\n</speak>"
            if len(chunk) > MAX_TEXT_LEN:
                raise ValueError(f"Chunk liian pitkä 11labsille (>{MAX_TEXT_LEN})")
            chunks.append((speaker, chunk))

    if pending_pause and chunks:
        speaker, last_chunk = chunks[-1]
        chunks[-1] = (speaker, last_chunk.replace("\n</speak>", f"\n{pending_pause}\n</speak>"))

    return chunks




def choose_voice_id(speaker: str, narrators: dict[str, str], default_voice_id: str) -> str:
    direct = narrators.get(speaker)
    if direct:
        return direct

    if speaker.lower().startswith("chat"):
        chat_voices = [vid for name, vid in narrators.items() if name.lower().startswith("chat")]
        if chat_voices:
            return random.choice(chat_voices)

    return default_voice_id

def synthesize_one(api_key: str, voice_id: str, text: str, out_path: Path, model_id: str, stability: float, similarity_boost: float, style: float, use_speaker_boost: bool) -> None:
    url = f"{API_BASE}/text-to-speech/{voice_id}/stream"
    headers = {"xi-api-key": api_key, "accept": "audio/mpeg", "content-type": "application/json"}
    payload = {
        "text": text,
        "model_id": model_id,
        "output_format": OUTPUT_FORMAT,
        "enable_ssml_parsing": True,
        "voice_settings": {
            "stability": stability,
            "similarity_boost": similarity_boost,
            "style": style,
            "use_speaker_boost": use_speaker_boost,
        },
    }
    req = request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    try:
        with request.urlopen(req, timeout=180) as response:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(response.read())
    except error.HTTPError as e:
        details = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code}: {details}") from e


def render_silence(out_path: Path, seconds: float) -> None:
    cmd = [
        "ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
        "-t", f"{seconds:.2f}", "-q:a", "9", "-acodec", "libmp3lame", str(out_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main() -> int:
    parser = argparse.ArgumentParser(description="ElevenLabs audiobook generator")
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--out-dir", default="audio_parts", help="Hakemisto osa-mp3 tiedostoille")
    parser.add_argument("--narrators-file", default=DEFAULT_NARRATORS_FILE)
    parser.add_argument("--voice-name", default="Kertoja")
    parser.add_argument("--voice-id", default=None)
    parser.add_argument("--model", choices=["v2", "v3"], default="v2")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--chunk-limit", type=int, default=DEFAULT_CHUNK_LIMIT)
    parser.add_argument("--stability", type=float, default=0.45)
    parser.add_argument("--similarity-boost", type=float, default=0.75)
    parser.add_argument("--style", type=float, default=0.15)
    parser.add_argument("--no-speaker-boost", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-merge", action="store_true", help="Älä yhdistä osia lopuksi")
    parser.add_argument("--merged-file", default="audiobook.mp3", help="Yhdistetyn mp3:n tiedostonimi")
    args = parser.parse_args()

    content = Path(args.input_file).read_text(encoding="utf-8").strip()
    narrators = load_narrators(Path(args.narrators_file))
    voice_id = args.voice_id or narrators.get(args.voice_name)
    if not voice_id:
        print(f"Virhe: hahmoa '{args.voice_name}' ei löydy tiedostosta {args.narrators_file}", file=sys.stderr)
        return 2

    model_id = args.model_id or MODEL_MAP[args.model]
    chunks = split_ssml_chunks(content, args.chunk_limit)
    print(f"Voice: {args.voice_name} -> {voice_id}")
    print(f"Model: {args.model} ({model_id})")
    print(f"Chunkit: {len(chunks)} kpl")

    if args.dry_run:
        return 0

    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        print("Virhe: ELEVENLABS_API_KEY puuttuu ympäristöstä.", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir)
    script_dir = Path(args.input_file).resolve().parent
    part_no = 1
    for speaker, chunk in chunks:
        audio_notif = re.search(r'<audio\s+[^>]*src="notification"[^>]*/>', chunk, flags=re.IGNORECASE)
        if audio_notif:
            notif_src = script_dir / "notification.mp3"
            if notif_src.exists():
                out_path = out_dir / f"{part_no:04d}.mp3"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(notif_src, out_path)
                print(f"[{part_no}] -> {out_path} (notification.mp3)")
                part_no += 1

            # Yleinen chat-rakenne: notification + break + varsinainen ääneen luettava teksti.
            # Poistetaan notification sekä heti perässä oleva break, mutta säilytetään muu puhe.
            chunk_after_notif = re.sub(
                r'<audio\s+[^>]*src="notification"[^>]*/>\s*',
                '',
                chunk,
                count=1,
                flags=re.IGNORECASE,
            )
            break_m = re.match(r'\s*<break\s+[^>]*time="([0-9.]+)s"[^>]*/>\s*', chunk_after_notif, flags=re.IGNORECASE)
            if break_m:
                pause_s = float(break_m.group(1))
                out_path = out_dir / f"{part_no:04d}.mp3"
                render_silence(out_path, pause_s)
                print(f"[{part_no}] -> {out_path} (silence {pause_s:.2f}s)")
                part_no += 1
                chunk_after_notif = re.sub(r'^\s*<break\s+[^>]*time="[0-9.]+s"[^>]*/>\s*', '', chunk_after_notif, count=1, flags=re.IGNORECASE)

            chunk = chunk_after_notif.strip()
            if not chunk or is_pause_only_ssml(chunk):
                continue

        i = part_no
        chunk_voice_id = choose_voice_id(speaker, narrators, voice_id)
        out_path = out_dir / f"{i:04d}.mp3"
        print(f"[{i}] -> {out_path} ({len(chunk)} merkkiä) speaker={speaker}")
        print("--- 11labs request debug ---")
        print(f"voice_id={chunk_voice_id} model_id={model_id} output_format={OUTPUT_FORMAT} enable_ssml_parsing=True")
        print(f"voice_settings={{stability: {args.stability}, similarity_boost: {args.similarity_boost}, style: {args.style}, use_speaker_boost: {not args.no_speaker_boost}}}")
        print("text:")
        print(chunk)
        print("--- /11labs request debug ---")
        synthesize_one(api_key, chunk_voice_id, chunk, out_path, model_id, args.stability, args.similarity_boost, args.style, not args.no_speaker_boost)
        part_no += 1

    if not args.no_merge:
        merged_path = Path(args.merged_file)
        print(f"Yhdistetään osat tiedostoon: {merged_path}")
        merge_mp3_parts(out_dir, merged_path)

    print("Valmis.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
