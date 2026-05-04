#!/usr/bin/env python3
"""ElevenLabs audiobook generator from SSML/text file.

- Reads synthesis content from a file (--input-file)
- Lets user choose model family (--model v2|v3) or explicit --model-id
- Produces one MP3 output file
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from urllib import error, request

API_BASE = "https://api.elevenlabs.io/v1"
OUTPUT_FORMAT = "mp3_44100_128"

MODEL_MAP = {
    "v2": "eleven_multilingual_v2",
    "v3": "eleven_v3",
}


def synthesize(
    api_key: str,
    voice_id: str,
    text: str,
    out_path: Path,
    model_id: str,
    stability: float,
    similarity_boost: float,
    style: float,
    use_speaker_boost: bool,
) -> None:
    url = f"{API_BASE}/text-to-speech/{voice_id}/stream"
    headers = {
        "xi-api-key": api_key,
        "accept": "audio/mpeg",
        "content-type": "application/json",
    }
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

    req = request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=180) as response:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("wb") as f:
                while True:
                    chunk = response.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
    except error.HTTPError as e:
        details = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code}: {details}") from e


def main() -> int:
    parser = argparse.ArgumentParser(description="ElevenLabs audiobook generator")
    parser.add_argument("--input-file", required=True, help="Path to SSML/text content file")
    parser.add_argument("--out", default="final_merged.mp3", help="Output mp3 file path")
    parser.add_argument("--voice-id", default=os.getenv("ELEVENLABS_VOICE_ID", "Dkbbg7k9Ir9TNzn5GYLp"))
    parser.add_argument("--model", choices=["v2", "v3"], default="v2", help="ElevenLabs model family")
    parser.add_argument("--model-id", default=None, help="Optional explicit model id (overrides --model)")
    parser.add_argument("--stability", type=float, default=0.45)
    parser.add_argument("--similarity-boost", type=float, default=0.75)
    parser.add_argument("--style", type=float, default=0.15)
    parser.add_argument("--no-speaker-boost", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Only validate input and print settings")
    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Virhe: syötetiedostoa ei löydy: {input_path}", file=sys.stderr)
        return 2

    content = input_path.read_text(encoding="utf-8").strip()
    if not content:
        print("Virhe: syötetiedosto on tyhjä.", file=sys.stderr)
        return 2

    model_id = args.model_id or MODEL_MAP[args.model]

    print(f"Input: {input_path}")
    print(f"Output: {args.out}")
    print(f"Voice ID: {args.voice_id}")
    print(f"Model: {args.model} ({model_id})")

    if args.dry_run:
        return 0

    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        print("Virhe: ELEVENLABS_API_KEY puuttuu ympäristöstä.", file=sys.stderr)
        return 2

    synthesize(
        api_key=api_key,
        voice_id=args.voice_id,
        text=content,
        out_path=Path(args.out),
        model_id=model_id,
        stability=args.stability,
        similarity_boost=args.similarity_boost,
        style=args.style,
        use_speaker_boost=not args.no_speaker_boost,
    )
    print("Valmis.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
