#!/usr/bin/env python3
"""ElevenLabs audiobook generator from SSML/text file."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from urllib import error, request

API_BASE = "https://api.elevenlabs.io/v1"
OUTPUT_FORMAT = "mp3_44100_128"
DEFAULT_NARRATORS_FILE = "prompt_narrators.txt"

MODEL_MAP = {
    "v2": "eleven_multilingual_v2",
    "v3": "eleven_v3",
}


def load_narrators(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Kertojatiedoston pitää olla JSON-objekti.")
    return {str(k): str(v) for k, v in data.items()}


def synthesize(api_key: str, voice_id: str, text: str, out_path: Path, model_id: str, stability: float, similarity_boost: float, style: float, use_speaker_boost: bool) -> None:
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


def main() -> int:
    parser = argparse.ArgumentParser(description="ElevenLabs audiobook generator")
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--out", default="final_merged.mp3")
    parser.add_argument("--narrators-file", default=DEFAULT_NARRATORS_FILE)
    parser.add_argument("--voice-name", default="Kertoja", help="Hahmon nimi narrators-JSON:stä")
    parser.add_argument("--voice-id", default=None, help="Yliaja voice-id suoraan")
    parser.add_argument("--model", choices=["v2", "v3"], default="v2")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--stability", type=float, default=0.45)
    parser.add_argument("--similarity-boost", type=float, default=0.75)
    parser.add_argument("--style", type=float, default=0.15)
    parser.add_argument("--no-speaker-boost", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    content = Path(args.input_file).read_text(encoding="utf-8").strip()
    if not content:
        print("Virhe: syötetiedosto on tyhjä.", file=sys.stderr)
        return 2

    narrators = load_narrators(Path(args.narrators_file))
    if args.voice_id:
        voice_id = args.voice_id
    else:
        voice_id = narrators.get(args.voice_name)
        if not voice_id:
            print(f"Virhe: hahmoa '{args.voice_name}' ei löydy tiedostosta {args.narrators_file}", file=sys.stderr)
            return 2

    model_id = args.model_id or MODEL_MAP[args.model]
    print(f"Voice: {args.voice_name} -> {voice_id}")
    print(f"Model: {args.model} ({model_id})")

    if args.dry_run:
        return 0

    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        print("Virhe: ELEVENLABS_API_KEY puuttuu ympäristöstä.", file=sys.stderr)
        return 2

    synthesize(api_key, voice_id, content, Path(args.out), model_id, args.stability, args.similarity_boost, args.style, not args.no_speaker_boost)
    print("Valmis.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
