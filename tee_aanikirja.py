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
from typing import Any
from xml.sax.saxutils import escape

from openai import OpenAI

API_BASE = "https://api.elevenlabs.io/v1"
OUTPUT_FORMAT = "mp3_44100_128"
DEFAULT_AUDIO_FORMAT = "mp3"
MAX_RENDERER_CHUNK_LIMIT = 4000
DEFAULT_NARRATORS_FILE = "prompt_narrators.txt"
DEFAULT_PRONUNCIATION_FILE = "prompt_pronunciation.txt"
DEFAULT_PRONUNCIATION_DICTIONARY = "pronunciation_dictionary.json"
MAX_TEXT_LEN = 10000
DEFAULT_CHUNK_LIMIT = MAX_RENDERER_CHUNK_LIMIT
NARRATOR_NAME = "Kertoja"
MODEL_MAP = {"v2": "eleven_multilingual_v2", "v3": "eleven_v3"}


def load_narrators(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Kertojatiedoston pitää olla JSON-objekti.")
    if "voices" in data:
        return data
    voices = {name: {"gender": "female", "ids": {"elevenlabs": vid}} for name, vid in data.items()}
    return {"defaults": {}, "voices": voices}


def load_pronunciation_locators(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []

    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        locators = data.get("pronunciation_dictionary_locators", [])
    elif isinstance(data, list):
        locators = data
    else:
        raise ValueError("Ääntämistiedoston pitää olla JSON-lista tai objekti.")

    if not isinstance(locators, list):
        raise ValueError("pronunciation_dictionary_locators pitää olla lista.")

    cleaned: list[dict[str, str]] = []
    for item in locators:
        if not isinstance(item, dict):
            raise ValueError("Jokaisen ääntämisohjeen pitää olla JSON-objekti.")
        dict_id = str(item.get("pronunciation_dictionary_id", "")).strip()
        version_id = str(item.get("version_id", "")).strip()
        if not dict_id or not version_id:
            raise ValueError("Ääntämisohjeesta puuttuu pronunciation_dictionary_id tai version_id.")
        cleaned.append({"pronunciation_dictionary_id": dict_id, "version_id": version_id})
    return cleaned


def load_pronunciation_dictionary(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("pronunciation_dictionary.json pitää olla JSON-objekti.")
    if not isinstance(data.get("entries"), list):
        raise ValueError("pronunciation_dictionary.json: entries pitää olla lista.")
    return data


def build_pronunciation_map(dictionary: dict[str, Any], language: str) -> list[tuple[re.Pattern[str], str, str]]:
    replacements: list[tuple[str, str]] = []
    for entry in dictionary.get("entries", []):
        if not isinstance(entry, dict):
            continue
        graphemes = entry.get("graphemes", [])
        pronunciations = entry.get("pronunciations", {})
        lang_data = pronunciations.get(language, {}) if isinstance(pronunciations, dict) else {}
        if not isinstance(graphemes, list) or not isinstance(lang_data, dict):
            continue
        replacement = lang_data.get("alias") or lang_data.get("spoken")
        if not isinstance(replacement, str) or not replacement.strip():
            continue
        for g in graphemes:
            if isinstance(g, str) and g.strip():
                replacements.append((g.strip(), replacement.strip()))
    replacements.sort(key=lambda x: len(x[0]), reverse=True)
    patterns: list[tuple[re.Pattern[str], str, str]] = []
    for term, repl in replacements:
        if term.startswith("#") or term.startswith("@"):
            regex = re.compile(rf"(?<![\w]){re.escape(term)}(?![\w])")
        else:
            regex = re.compile(rf"(?<![\w@#]){re.escape(term)}(?![\w])")
        patterns.append((regex, repl, term))
    return patterns


def apply_pronunciation_aliases(text: str, pronunciation_map: list[tuple[re.Pattern[str], str, str]]) -> str:
    updated = text
    for pattern, replacement, _ in pronunciation_map:
        updated = pattern.sub(replacement, updated)
    return updated


def export_pls(dictionary: dict[str, Any], language: str, out_path: Path) -> None:
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<lexicon version="1.0" xmlns="http://www.w3.org/2005/01/pronunciation-lexicon" alphabet="ipa" xml:lang="{escape(language)}">',
    ]
    for entry in dictionary.get("entries", []):
        if not isinstance(entry, dict):
            continue
        graphemes = entry.get("graphemes", [])
        pronunciations = entry.get("pronunciations", {})
        lang_data = pronunciations.get(language, {}) if isinstance(pronunciations, dict) else {}
        if not isinstance(graphemes, list) or not isinstance(lang_data, dict):
            continue
        ipa = lang_data.get("ipa")
        alias = lang_data.get("alias")
        if not isinstance(ipa, str) and not isinstance(alias, str):
            continue
        lines.append("  <lexeme>")
        for g in graphemes:
            if isinstance(g, str) and g.strip():
                lines.append(f"    <grapheme>{escape(g.strip())}</grapheme>")
        if isinstance(ipa, str):
            lines.append(f"    <phoneme>{escape(ipa.strip())}</phoneme>")
        else:
            lines.append(f"    <alias>{escape(str(alias).strip())}</alias>")
        lines.append("  </lexeme>")
    lines.append("</lexicon>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_all_pls(dictionary: dict[str, Any], out_dir: Path) -> list[Path]:
    out_paths: list[Path] = []
    for language in dictionary.get("default_languages", ["fi-FI", "en-GB", "en-US", "es-MX"]):
        p = out_dir / f"pronunciation_dictionary.{language}.pls"
        export_pls(dictionary, language, p)
        out_paths.append(p)
    return out_paths


def strip_speak_wrappers(text: str) -> str:
    t = text.strip()
    t = re.sub(r"^\s*<speak[^>]*>", "", t, flags=re.IGNORECASE)
    t = re.sub(r"</speak>\s*$", "", t, flags=re.IGNORECASE)
    return t.strip()


def strip_ssml_tags(text: str) -> str:
    text = re.sub(r"<break\b[^>]*/>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    return re.sub(r"\s+", " ", text).strip()


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

    attrs = m.group(1).strip()
    inner = m.group(2).strip()
    return [f"<voice {attrs}>{part}</voice>" for part in split_large_text(inner, chunk_limit) if part.strip()]


def extract_openai_ssml_options(text: str) -> tuple[str | None, str | None, str | None]:
    m = re.search(r"<voice\s+([^>]*)>", text, flags=re.IGNORECASE)
    if not m:
        return None, None, None
    attrs = m.group(1)
    def read_attr(name: str) -> str | None:
        match = re.search(rf'{name}="([^"]*)"', attrs, flags=re.IGNORECASE)
        return match.group(1).strip() if match else None
    return read_attr("openai_voice"), read_attr("openai_model"), read_attr("openai_instructions")


def merge_mp3_parts(out_dir: Path, merged_path: Path) -> None:
    part_files = sorted(out_dir.glob("*.mp3"))
    if not part_files:
        raise RuntimeError("Ei mp3-osia yhdistettäväksi.")

    list_file = out_dir / "concat_list.txt"
    lines = [f"file '{f.resolve()}'" for f in part_files]
    list_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # IMPORTANT:
    # Piper may write WAV/PCM audio even when the filename ends in .mp3.
    # Silence files are generated as real MP3.
    # Therefore stream-copy concatenation (-c copy) can create corrupt audio
    # and audible spikes at boundaries. Always decode and re-encode.
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_file),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "44100",
        "-c:a",
        "libmp3lame",
        "-b:a",
        "128k",
        str(merged_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        details = (e.stderr or e.stdout or "").strip()
        raise RuntimeError(f"Osien yhdistäminen epäonnistui ffmpeg:llä: {details}") from e


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




def ensure_ffmpeg_installed() -> None:
    if shutil.which("ffmpeg"):
        return
    raise RuntimeError(
        "ffmpeg puuttuu. Asenna ffmpeg (esim. Ubuntu/Debian: sudo apt install ffmpeg, macOS: brew install ffmpeg)."
    )


def ensure_renderer_installed(renderer: str) -> None:
    if renderer in {"elevenlabs", "openai"}:
        return
    binary = "piper" if renderer == "piper" else "kokoro-tts"
    if shutil.which(binary):
        return
    if renderer == "piper":
        raise RuntimeError("Piper puuttuu. Asenna esim: pip install piper-tts tai sudo apt install piper")
    raise RuntimeError("Kokoro puuttuu. Asenna esim: pip install kokoro-onnx")




def piper_download_candidates(voice_id: str) -> list[str]:
    candidates = [voice_id]
    if voice_id.endswith("-medium"):
        candidates.append(voice_id[:-7] + "-low")
    return candidates


def ensure_piper_voice_available(voice_id: str) -> str:
    for candidate in piper_download_candidates(voice_id):
        cmd = [sys.executable, "-m", "piper.download_voices", candidate]
        proc = subprocess.run(cmd, text=True, capture_output=True)
        if proc.returncode == 0:
            if candidate != voice_id:
                print(f"Piper-ääni '{voice_id}' ei ollut saatavilla, käytetään ladattua ääntä '{candidate}'.")
            return candidate
    raise RuntimeError(
        f"Piper-äänen automaattinen lataus epäonnistui äänelle '{voice_id}'. "
        "Tarkista saatavilla olevat Piper-äänet ja anna toimiva --voice-id."
    )


def synthesize_local(renderer: str, voice_id: str, text: str, out_path: Path, audio_format: str, pronunciation_map: list[tuple[re.Pattern[str], str, str]] | None = None) -> str:
    if renderer == "openai":
        client = OpenAI()
        ssml_voice, ssml_model, ssml_instructions = extract_openai_ssml_options(text)
        resolved_voice = ssml_voice or voice_id
        resolved_model = ssml_model or "gpt-4o-mini-tts"
        rewritten = apply_pronunciation_aliases(text, pronunciation_map or [])
        clean_text = strip_ssml_tags(rewritten)
        if not clean_text:
            return resolved_voice
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": resolved_model,
            "voice": resolved_voice,
            "input": clean_text,
            "response_format": audio_format,
        }
        instruction_suffix = "Pronounce Finnish names and words with Finnish pronunciation, English names in English, and Mexican Spanish words in Mexican Spanish. Respect the provided rewritten spoken forms."
        payload["instructions"] = f"{(ssml_instructions or '').strip()} {instruction_suffix}".strip()
        with client.audio.speech.with_streaming_response.create(**payload) as response:
            response.stream_to_file(out_path)
        return resolved_voice

    if renderer == "piper":
        text = apply_pronunciation_aliases(text, pronunciation_map or [])
        cmd = ["piper", "--model", voice_id, "--output_file", str(out_path)]
        proc = subprocess.run(cmd, input=strip_ssml_tags(text), text=True, capture_output=True)
        if proc.returncode == 0:
            return voice_id
        combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
        if "Unable to find voice" in combined:
            resolved_voice = ensure_piper_voice_available(voice_id)
            retry = subprocess.run(["piper", "--model", resolved_voice, "--output_file", str(out_path)], input=strip_ssml_tags(text), text=True, capture_output=True)
            if retry.returncode == 0:
                return resolved_voice
            raise RuntimeError(f"Piper-ajo epäonnistui äänellä '{resolved_voice}': {retry.stderr.strip() or retry.stdout.strip()}")
        raise RuntimeError(f"Piper-ajo epäonnistui äänellä '{voice_id}': {proc.stderr.strip() or proc.stdout.strip()}")

    text = apply_pronunciation_aliases(text, pronunciation_map or [])
    proc = subprocess.run(["kokoro-tts", "--voice", voice_id, "--output", str(out_path), strip_ssml_tags(text)], text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Kokoro-ajo epäonnistui äänellä '{voice_id}': {proc.stderr.strip() or proc.stdout.strip()}")
    return voice_id

def choose_voice_id(speaker: str, narrators: dict[str, Any], renderer: str, default_voice_id: str) -> str:
    voices = narrators.get("voices", {})
    speaker_data = voices.get(speaker, {}) if isinstance(voices, dict) else {}
    if isinstance(speaker_data, dict):
        ids = speaker_data.get("ids", {})
        if isinstance(ids, dict) and ids.get(renderer):
            return str(ids[renderer])

    if speaker.lower().startswith("chat") and isinstance(voices, dict):
        chat_voices = []
        for name, meta in voices.items():
            if str(name).lower().startswith("chat") and isinstance(meta, dict):
                ids = meta.get("ids", {})
                if isinstance(ids, dict) and ids.get(renderer):
                    chat_voices.append(str(ids[renderer]))
        if chat_voices:
            return random.choice(chat_voices)
    return default_voice_id

class QuotaExceededError(RuntimeError):
    """Raised when ElevenLabs quota is exhausted for current request."""


def parse_http_json(details: str) -> Any:
    try:
        return json.loads(details)
    except json.JSONDecodeError:
        return None


def extract_quota_status(details: str) -> str | None:
    payload = parse_http_json(details)
    if not isinstance(payload, dict):
        return None
    detail = payload.get("detail")
    if isinstance(detail, dict):
        status = detail.get("status")
        if isinstance(status, str):
            return status
    return None


def check_credit_balance(api_key: str) -> int | None:
    url = f"{API_BASE}/user/subscription"
    headers = {"xi-api-key": api_key, "accept": "application/json"}
    req = request.Request(url, headers=headers, method="GET")
    try:
        with request.urlopen(req, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8", errors="replace"))
    except Exception:
        return None

    if not isinstance(payload, dict):
        return None
    limit = payload.get("character_limit")
    used = payload.get("character_count")
    if not isinstance(limit, int) or not isinstance(used, int):
        return None
    return limit - used


def synthesize_one(api_key: str, voice_id: str, text: str, out_path: Path, model_id: str, stability: float, similarity_boost: float, style: float, use_speaker_boost: bool, pronunciation_locators: list[dict[str, str]], output_format: str) -> None:
    url = f"{API_BASE}/text-to-speech/{voice_id}/stream"
    headers = {"xi-api-key": api_key, "accept": "audio/mpeg", "content-type": "application/json"}
    payload = {
        "text": text,
        "model_id": model_id,
        "output_format": output_format,
        "enable_ssml_parsing": True,
        "voice_settings": {
            "stability": stability,
            "similarity_boost": similarity_boost,
            "style": style,
            "use_speaker_boost": use_speaker_boost,
        },
    }
    if pronunciation_locators:
        payload["pronunciation_dictionary_locators"] = pronunciation_locators
    req = request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    try:
        with request.urlopen(req, timeout=180) as response:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(response.read())
    except error.HTTPError as e:
        details = e.read().decode("utf-8", errors="replace")
        if e.code in {401, 402} and extract_quota_status(details) == "quota_exceeded":
            raise QuotaExceededError(f"HTTP {e.code}: {details}") from e
        raise RuntimeError(f"HTTP {e.code}: {details}") from e


def render_silence(out_path: Path, seconds: float) -> None:
    cmd = [
        "ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
        "-t", f"{seconds:.2f}", "-q:a", "9", "-acodec", "libmp3lame", str(out_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def pop_leading_break_seconds(text: str) -> tuple[str, float | None]:
    m = re.match(r'^\s*<break\s+[^>]*time="([0-9.]+)s"[^>]*/>\s*', text, flags=re.IGNORECASE)
    if not m:
        return text, None
    remaining = re.sub(r'^\s*<break\s+[^>]*time="[0-9.]+s"[^>]*/>\s*', '', text, count=1, flags=re.IGNORECASE)
    return remaining, float(m.group(1))


def split_text_and_breaks(text: str) -> list[tuple[str, str | float]]:
    parts: list[tuple[str, str | float]] = []
    break_re = re.compile(r'<break\b([^>]*)/?>', flags=re.IGNORECASE)
    pos = 0
    for m in break_re.finditer(text):
        before = text[pos:m.start()]
        if before.strip():
            parts.append(("text", before.strip()))
        attrs = m.group(1) or ""
        tm = re.search(r'time="([0-9.]+)s"', attrs, flags=re.IGNORECASE)
        pause_s = float(tm.group(1)) if tm else 0.2
        parts.append(("break", pause_s))
        pos = m.end()
    tail = text[pos:]
    if tail.strip():
        parts.append(("text", tail.strip()))
    return parts


def main() -> int:
    parser = argparse.ArgumentParser(description="ElevenLabs audiobook generator")
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--out-dir", default="audio_parts", help="Hakemisto, johon luodaan parts/ ja mahdollinen yhdistetty mp3")
    parser.add_argument("--narrators-file", default=DEFAULT_NARRATORS_FILE)
    parser.add_argument("--pronunciation-file", default=DEFAULT_PRONUNCIATION_FILE)
    parser.add_argument("--pronunciation-dictionary", default=None)
    parser.add_argument("--language", default="en-GB")
    parser.add_argument("--generate-pls", action="store_true")
    parser.add_argument("--pls-out-dir", default="pronunciation_exports")
    parser.add_argument("--renderer", choices=["elevenlabs", "piper", "kokoro", "openai"], default="piper")
    parser.add_argument("--voice-name", default="Kertoja")
    parser.add_argument("--voice-id", default=None)
    parser.add_argument("--model", choices=["v2", "v3"], default="v2")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--chunk-limit", type=int, default=DEFAULT_CHUNK_LIMIT, help="Maksimimerkkimäärä per chunk (kaikille renderöijille enintään 4000).")
    parser.add_argument("--format", choices=["mp3"], default=DEFAULT_AUDIO_FORMAT, help="Ulostuloäänen formaatti käyttäjälle. Tällä hetkellä tuettu: mp3. Ohjelma mapittaa tämän renderer-kohtaiseen parametriin.")
    parser.add_argument("--stability", type=float, default=0.45)
    parser.add_argument("--similarity-boost", type=float, default=0.75)
    parser.add_argument("--style", type=float, default=0.15)
    parser.add_argument("--no-speaker-boost", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-merge", action="store_true", help="Älä yhdistä osia lopuksi")
    parser.add_argument("--merged-file", default=None, help="Yhdistetyn mp3:n tiedostonimi")
    args = parser.parse_args()

    content = Path(args.input_file).read_text(encoding="utf-8").strip()
    narrators = load_narrators(Path(args.narrators_file))
    pronunciation_locators = load_pronunciation_locators(Path(args.pronunciation_file))
    pronunciation_dictionary = None
    pronunciation_map: list[tuple[re.Pattern[str], str, str]] = []
    dictionary_path = Path(args.pronunciation_dictionary) if args.pronunciation_dictionary else None
    if dictionary_path and dictionary_path.exists():
        pronunciation_dictionary = load_pronunciation_dictionary(dictionary_path)
        pronunciation_map = build_pronunciation_map(pronunciation_dictionary, args.language)
    elif args.pronunciation_dictionary:
        print(f"Virhe: pronunciation dictionaryä ei löydy: {args.pronunciation_dictionary}", file=sys.stderr)
        return 2
    try:
        ensure_renderer_installed(args.renderer)
        ensure_ffmpeg_installed()
    except RuntimeError as e:
        print(f"Virhe: {e}", file=sys.stderr)
        return 2

    defaults = narrators.get("defaults", {})
    voices = narrators.get("voices", {})
    voice_meta = voices.get(args.voice_name, {}) if isinstance(voices, dict) else {}
    gender = "female"
    if isinstance(voice_meta, dict):
        gender = str(voice_meta.get("gender", "female")).lower()
    gender_defaults = defaults.get(gender, {}) if isinstance(defaults, dict) else {}
    fallback_voice_id = str(gender_defaults.get(args.renderer, "")).strip()

    voice_id = args.voice_id or choose_voice_id(args.voice_name, narrators, args.renderer, fallback_voice_id)
    if not voice_id:
        print(f"Virhe: hahmoa '{args.voice_name}' ei löydy tiedostosta {args.narrators_file}", file=sys.stderr)
        return 2

    model_id = args.model_id or MODEL_MAP[args.model]
    chunk_limit = min(args.chunk_limit, MAX_RENDERER_CHUNK_LIMIT)
    chunks = split_ssml_chunks(content, chunk_limit)
    print(f"Voice: {args.voice_name} -> {voice_id}")
    if args.renderer == "elevenlabs":
        print(f"Model: {args.model} ({model_id})")
    elif args.renderer == "openai":
        print("Model: gpt-4o-mini-tts")
    print(f"Chunkit: {len(chunks)} kpl")
    if pronunciation_locators:
        print(f"Ääntämissanastoja: {len(pronunciation_locators)} kpl")
    print(f"Kieli: {args.language}")
    print(f"Pronunciation-entryjä: {len(pronunciation_map)}")
    if pronunciation_map:
        preview = [f"{term} -> {repl}" for _, repl, term in pronunciation_map[:20]]
        print(f"Aktiiviset replacementit (max20): {preview}")
    if pronunciation_locators:
        print(f"ElevenLabs locatorit: {pronunciation_locators}")

    if args.generate_pls:
        if not pronunciation_dictionary:
            print("Virhe: --generate-pls vaatii --pronunciation-dictionary", file=sys.stderr)
            return 2
        generated = export_all_pls(pronunciation_dictionary, Path(args.pls_out_dir))
        print("Generoitu PLS-tiedostot:")
        for gp in generated:
            print(f"- {gp}")

    if args.dry_run:
        return 0

    api_key = ""
    if args.renderer == "openai" and not os.getenv("OPENAI_API_KEY"):
        print("Virhe: OPENAI_API_KEY puuttuu ympäristöstä.", file=sys.stderr)
        return 2

    if args.renderer == "elevenlabs":
        api_key = os.getenv("ELEVENLABS_API_KEY")
        if not api_key:
            print("Virhe: ELEVENLABS_API_KEY puuttuu ympäristöstä.", file=sys.stderr)
            return 2
        remaining_credits = check_credit_balance(api_key)
        if remaining_credits is not None and remaining_credits < 10000:
            print(f"Varoitus: ElevenLabs-krediittejä jäljellä vain {remaining_credits} (< 10000).", file=sys.stderr)

    out_dir = Path(args.out_dir)
    parts_dir = out_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    script_path = Path(args.input_file).resolve()
    script_dir = script_path.parent
    chapter_prefix = script_path.stem
    part_no = 1
    quota_exhausted = False
    for speaker, chunk in chunks:
        pending = chunk
        while True:
            notif_m = re.search(r'<audio\s+[^>]*src="notification"[^>]*/>', pending, flags=re.IGNORECASE)
            if not notif_m:
                break

            before = pending[:notif_m.start()].strip()
            if before and not is_pause_only_ssml(before):
                i = part_no
                chunk_voice_id = choose_voice_id(speaker, narrators, args.renderer, voice_id)
                out_path = parts_dir / f"{chapter_prefix}_{i:04d}.mp3"
                print(f"[{i}] -> {out_path} ({len(before)} merkkiä) speaker={speaker}")
                try:
                    if args.renderer == "elevenlabs":
                        synthesize_one(api_key, chunk_voice_id, before, out_path, model_id, args.stability, args.similarity_boost, args.style, not args.no_speaker_boost, pronunciation_locators, OUTPUT_FORMAT)
                    else:
                        chunk_voice_id = synthesize_local(args.renderer, chunk_voice_id, before, out_path, args.format, pronunciation_map)
                except QuotaExceededError as e:
                    print(f"Krediitit loppuivat kesken: {e}", file=sys.stderr)
                    quota_exhausted = True
                    break
                except RuntimeError as e:
                    print(f"Virhe: {e}", file=sys.stderr)
                    return 2
                part_no += 1

            notif_src = script_dir / "notification.mp3"
            if notif_src.exists():
                out_path = parts_dir / f"{chapter_prefix}_{part_no:04d}.mp3"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(notif_src, out_path)
                print(f"[{part_no}] -> {out_path} (notification.mp3)")
                part_no += 1

            if quota_exhausted:
                break

            pending = pending[notif_m.end():]
            pending, pause_s = pop_leading_break_seconds(pending)
            if pause_s is None:
                pause_s = 0.2
            out_path = parts_dir / f"{chapter_prefix}_{part_no:04d}.mp3"
            render_silence(out_path, pause_s)
            print(f"[{part_no}] -> {out_path} (silence {pause_s:.2f}s)")
            part_no += 1

        chunk = pending.strip()
        if not chunk or is_pause_only_ssml(chunk):
            continue

        chunk_voice_id = choose_voice_id(speaker, narrators, args.renderer, voice_id)
        if args.renderer == "elevenlabs":
            i = part_no
            out_path = parts_dir / f"{chapter_prefix}_{i:04d}.mp3"
            print(f"[{i}] -> {out_path} ({len(chunk)} merkkiä) speaker={speaker}")
            print("--- 11labs request debug ---")
            print(f"voice_id={chunk_voice_id} model_id={model_id} output_format={OUTPUT_FORMAT} enable_ssml_parsing=True")
            print(f"voice_settings={{stability: {args.stability}, similarity_boost: {args.similarity_boost}, style: {args.style}, use_speaker_boost: {not args.no_speaker_boost}}}")
            if pronunciation_locators:
                print(f"pronunciation_dictionary_locators={pronunciation_locators}")
            print("text:")
            print(chunk)
            print("--- /11labs request debug ---")
        try:
            if args.renderer == "elevenlabs":
                synthesize_one(api_key, chunk_voice_id, chunk, out_path, model_id, args.stability, args.similarity_boost, args.style, not args.no_speaker_boost, pronunciation_locators, OUTPUT_FORMAT)
                part_no += 1
            else:
                for frag_type, value in split_text_and_breaks(chunk):
                    out_path = parts_dir / f"{chapter_prefix}_{part_no:04d}.mp3"
                    if frag_type == "break":
                        pause_s = float(value)
                        render_silence(out_path, pause_s)
                        print(f"[{part_no}] -> {out_path} (silence {pause_s:.2f}s)")
                    else:
                        frag_text = str(value)
                        if is_pause_only_ssml(frag_text):
                            continue
                        clean_text = strip_ssml_tags(frag_text)
                        if not clean_text:
                            continue
                        print(f"[{part_no}] -> {out_path} ({len(frag_text)} merkkiä) speaker={speaker}")
                        chunk_voice_id = synthesize_local(args.renderer, chunk_voice_id, clean_text, out_path, args.format, pronunciation_map)
                    part_no += 1
        except QuotaExceededError as e:
            print(f"Krediitit loppuivat kesken: {e}", file=sys.stderr)
            quota_exhausted = True
            break
        except RuntimeError as e:
            print(f"Virhe: {e}", file=sys.stderr)
            return 2

        if quota_exhausted:
            break

    if not args.no_merge:
        part_files = sorted(parts_dir.glob("*.mp3"))
        if part_files:
            merged_path = Path(args.merged_file) if args.merged_file else out_dir / f"{chapter_prefix}_fullchapter.mp3"
            print(f"Yhdistetään osat tiedostoon: {merged_path}")
            merge_mp3_parts(parts_dir, merged_path)
        else:
            print("Ei yhdistettäviä osia.", file=sys.stderr)

    if quota_exhausted:
        print("Valmis osittain: krediitit loppuivat, mutta tähän asti tuotetut osat yhdistettiin.")
        return 1

    print("Valmis.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
