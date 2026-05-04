#!/usr/bin/env python3
"""ElevenLabs v2 audiobook segment generator.

- Generates one MP3 per dialogue line: 0001.mp3, 0002.mp3, ...
- Uses one voice per API call (works around multi-speaker limitations).
- Tests pauses + SSML emotion/prosody controls.
- Merges all generated parts into a single MP3 with ffmpeg.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import json
from urllib import error, request

API_BASE = "https://api.elevenlabs.io/v1"
OUTPUT_FORMAT = "mp3_44100_128"

# You can override these via environment variables if needed.
VOICES: Dict[str, str] = {
    "Kertoja": os.getenv("ELEVENLABS_VOICE_KERTOJA", "Dkbbg7k9Ir9TNzn5GYLp"),
    "Miksu": os.getenv("ELEVENLABS_VOICE_MIKSU", "6n4YmXLiuP4C7cZqYOJl"),
    "Elias": os.getenv("ELEVENLABS_VOICE_ELIAS", "IKne3meq5aSn9XLyUdCD"),
}

# (name, ssml_text)
SCENE: List[Tuple[str, str]] = [
    (
        "Kertoja",
        """
<speak>
Vastapäätä istuva Miksu löi kätensä yhteen ja Elias havahtui mietteistään.
Miksu oli esittänyt hänelle liikeideansa: Miksu alkaisi hänen managerikseen
ja kaupallistajakseen.
<break time=\"1.0s\"/>
</speak>
""".strip(),
    ),
    (
        "Miksu",
        """
<speak>
<prosody rate=\"104%\" pitch=\"+2st\">Tajuatko sä Elias minkälainen kohu tästä on syntymässä?</prosody>
<break time=\"0.3s\"/>
<prosody rate=\"106%\">Sun kanavan tilaajaluvut ovat nousseet jo lähes kolmeensataan</prosody>
ja Vuosaari-keikan lopusta on klippailtu jo ainakin kymmenelle YouTube-julkaisulle oma osuutensa.
</speak>
""".strip(),
    ),
    (
        "Elias",
        """
<speak>
<prosody pitch=\"+3st\" rate=\"95%\">Ai mitä?</prosody>
</speak>
""".strip(),
    ),
    (
        "Miksu",
        """
<speak>
Aistivastevideota.
<break time=\"0.2s\"/>
Kuvankauniit naiset pulikoivat kahluualtaissa ja kuiskailevat mikrofoniin lempeällä äänellä helliä sanoja.
<prosody rate=\"96%\" pitch=\"-1st\">Mutta älä siitä välitä, keskitytään nyt suhun.</prosody>
Oliko siinä Vuosaari-videolla ihan oikeasti ruumis, vai trollasitko vain?
</speak>
""".strip(),
    ),
    (
        "Kertoja",
        """
<speak>
Elias ei vastannut heti. Hän katsoi Miksua suoraan silmiin.
<break time=\"0.6s\"/>
</speak>
""".strip(),
    ),
    (
        "Elias",
        """
<speak>
<prosody rate=\"88%\" pitch=\"-1st\">Oli.</prosody>
<break time=\"1.0s\"/>
</speak>
""".strip(),
    ),
]


def sanitize_for_log(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def synthesize_one(
    api_key: str,
    voice_id: str,
    ssml_text: str,
    out_path: Path,
    model_id: str,
    stability: float,
    similarity_boost: float,
    style: float,
    use_speaker_boost: bool,
    previous_request_ids: List[str] | None,
) -> str | None:
    url = f"{API_BASE}/text-to-speech/{voice_id}/stream"
    headers = {
        "xi-api-key": api_key,
        "accept": "audio/mpeg",
        "content-type": "application/json",
    }
    payload = {
        "text": ssml_text,
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
    if previous_request_ids:
        payload["previous_request_ids"] = previous_request_ids[-3:]

    req = request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=120) as r:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("wb") as f:
                while True:
                    chunk = r.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)

            return r.headers.get("request-id")
    except error.HTTPError as e:
        details = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code}: {details}") from e


def merge_mp3s(parts: List[Path], final_path: Path) -> None:
    list_file = final_path.parent / "ffmpeg_concat_list.txt"
    with list_file.open("w", encoding="utf-8") as f:
        for p in parts:
            f.write(f"file '{p.resolve()}'\n")

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_file),
        "-c",
        "copy",
        str(final_path),
    ]
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="ElevenLabs v2 audiobook generator (split + merge)")
    parser.add_argument("--out-dir", default="audio_parts", help="Directory for 0001.mp3, 0002.mp3, ...")
    parser.add_argument("--final", default="final_merged.mp3", help="Final merged mp3")
    parser.add_argument("--model-id", default=os.getenv("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2"))
    parser.add_argument("--stability", type=float, default=0.45)
    parser.add_argument("--similarity-boost", type=float, default=0.75)
    parser.add_argument("--style", type=float, default=0.15)
    parser.add_argument("--no-speaker-boost", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Only print calls, do not hit API")
    args = parser.parse_args()

    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key and not args.dry_run:
        print("Virhe: ELEVENLABS_API_KEY puuttuu ympäristöstä.", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir)
    final_path = Path(args.final)

    generated: List[Path] = []
    request_ids: List[str] = []

    for i, (speaker, ssml_text) in enumerate(SCENE, start=1):
        if speaker not in VOICES:
            raise KeyError(f"Puuttuva voice-id hahmolle: {speaker}")

        part = out_dir / f"{i:04d}.mp3"
        generated.append(part)
        print(f"[{i:02d}/{len(SCENE)}] {speaker} -> {part}")
        print(f"    SSML: {sanitize_for_log(ssml_text)[:120]}...")

        if args.dry_run:
            continue

        req_id = synthesize_one(
            api_key=api_key,
            voice_id=VOICES[speaker],
            ssml_text=ssml_text,
            out_path=part,
            model_id=args.model_id,
            stability=args.stability,
            similarity_boost=args.similarity_boost,
            style=args.style,
            use_speaker_boost=not args.no_speaker_boost,
            previous_request_ids=request_ids,
        )
        if req_id:
            request_ids.append(req_id)

    if args.dry_run:
        print("Dry-run valmis. Ei API-kutsuja eikä yhdistämistä.")
        return 0

    print("Yhdistetään osat ffmpeg:llä...")
    merge_mp3s(generated, final_path)
    print(f"Valmis: {final_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
