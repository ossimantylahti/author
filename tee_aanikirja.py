#!/usr/bin/env python3
"""ElevenLabs audiobook generator from SSML/text file."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib import error, request
import shutil
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
    attrs = extract_voice_attrs(text)
    return attrs.get("openai_voice"), attrs.get("openai_model"), attrs.get("openai_instructions")


def extract_voice_attrs(text: str) -> dict[str, str]:
    m = re.search(r"<voice\s+([^>]*)>", text, flags=re.IGNORECASE)
    if not m:
        return {}
    attrs_text = m.group(1)
    supported = {
        "name",
        "elevenlabs_voice",
        "openai_voice",
        "openai_model",
        "openai_instructions",
        "kokoro_voice",
        "piper_voice",
        "language",
        "emotion",
        "pace",
        "tone",
    }
    attrs: dict[str, str] = {}
    for match in re.finditer(r'([a-zA-Z_][\w-]*)="([^"]*)"', attrs_text):
        key = match.group(1).strip().lower()
        if key in supported:
            attrs[key] = match.group(2).strip()
    return attrs


def looks_like_kokoro_voice(value: str) -> bool:
    return bool(re.fullmatch(r"[ab][fm]_[a-z0-9_]+", value.strip()))


def looks_like_piper_voice(value: str) -> bool:
    return bool(re.fullmatch(r"[a-z]{2}_[A-Z]{2}-.+-(low|medium|high)", value.strip()))


def looks_like_renderer_voice_id(renderer: str, value: str) -> bool:
    candidate = value.strip()
    if renderer == "kokoro":
        return looks_like_kokoro_voice(candidate)
    if renderer == "piper":
        return looks_like_piper_voice(candidate)
    if renderer == "openai":
        return candidate in {"alloy", "ash", "ballad", "coral", "echo", "fable", "nova", "onyx", "sage", "shimmer", "verse"}
    return False


def narrator_voice_id(narrators: dict[str, Any], narrator_name: str, renderer: str) -> str | None:
    voices = narrators.get("voices", {})
    speaker_data = voices.get(narrator_name, {}) if isinstance(voices, dict) else {}
    if not isinstance(speaker_data, dict):
        return None
    ids = speaker_data.get("ids", {})
    if isinstance(ids, dict) and ids.get(renderer):
        return str(ids[renderer])
    return None


def fallback_narrator_voice_id(narrators: dict[str, Any], renderer: str) -> str | None:
    return narrator_voice_id(narrators, NARRATOR_NAME, renderer)


def resolve_voice_id_for_fragment(renderer: str, fragment_text: str, speaker: str, narrators: dict[str, Any], cli_voice_name: str | None, cli_voice_id: str | None) -> tuple[str, str]:
    attrs = extract_voice_attrs(fragment_text)
    renderer_attr = attrs.get(f"{renderer}_voice")
    if renderer_attr:
        return renderer_attr, "ssml_renderer_attribute"

    voice_name = attrs.get("name", "").strip()
    if voice_name and looks_like_renderer_voice_id(renderer, voice_name):
        return voice_name, "ssml_direct_voice_id"

    if voice_name:
        mapped = narrator_voice_id(narrators, voice_name, renderer)
        if mapped:
            return mapped, "narrator_mapping"

    speaker_mapped = narrator_voice_id(narrators, speaker, renderer)
    if speaker_mapped:
        return speaker_mapped, "narrator_mapping"

    fallback_voice = fallback_narrator_voice_id(narrators, renderer)
    if fallback_voice:
        return fallback_voice, "narrator_fallback_kertoja"

    if cli_voice_id:
        print("Warning: using deprecated CLI voice fallback. Prefer SSML voice attributes or prompt_narrators.txt.")
        return cli_voice_id, "deprecated_cli_voice_id"

    if cli_voice_name:
        cli_mapped = narrator_voice_id(narrators, cli_voice_name, renderer)
        if cli_mapped:
            print("Warning: using deprecated CLI voice fallback. Prefer SSML voice attributes or prompt_narrators.txt.")
            return cli_mapped, "deprecated_cli_voice_name"

    raise RuntimeError(
        f"No voice id found for renderer '{renderer}'. Add voices.Kertoja.ids.{renderer} to prompt_narrators.txt or specify a renderer-specific SSML voice attribute."
    )


def extract_ssml_languages(text: str) -> set[str]:
    langs = set()
    for attrs in re.findall(r"<voice\s+([^>]*)>", text, flags=re.IGNORECASE):
        m = re.search(r'language="([^"]+)"', attrs, flags=re.IGNORECASE)
        if m and m.group(1).strip():
            langs.add(m.group(1).strip().lower())
    return langs


def has_openai_language_instructions(text: str) -> bool:
    for attrs in re.findall(r"<voice\s+([^>]*)>", text, flags=re.IGNORECASE):
        m = re.search(r'openai_instructions="([^"]+)"', attrs, flags=re.IGNORECASE)
        if not m:
            continue
        instructions = m.group(1).strip().lower()
        if any(k in instructions for k in ("lue suomeksi", "read in english", "lee en español", "lee en espanol")):
            return True
    return False


def resolve_pronunciation_language(cli_language: str | None, ssml_languages: set[str]) -> tuple[str, str | None]:
    if cli_language:
        return cli_language, None
    mapping = {"finnish": "fi-FI", "english": "en-GB", "spanish": "es-MX"}
    if len(ssml_languages) == 1:
        only = next(iter(ssml_languages))
        return mapping.get(only, "en-GB"), None
    if len(ssml_languages) > 1:
        return "en-GB", "Multiple SSML languages detected; using pronunciation fallback language en-GB."
    return "en-GB", None


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


def map_kokoro_language(text: str) -> str | None:
    language_map = {"finnish": "fi", "english": "en-us", "spanish": "es-mx"}
    attrs = extract_voice_attrs(text)
    language = attrs.get("language", "").strip().lower()
    return language_map.get(language)


def synthesize_local(renderer: str, voice_id: str, text: str, out_path: Path, audio_format: str, pronunciation_map: list[tuple[re.Pattern[str], str, str]] | None = None, kokoro_model: str | None = None, kokoro_voices: str | None = None) -> str:
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
    clean_text = strip_ssml_tags(text)
    if not clean_text:
        return voice_id

    out_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = f".{audio_format}"
    temp_input_path: Path | None = None
    temp_output_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as temp_input:
            temp_input.write(clean_text)
            temp_input_path = Path(temp_input.name)

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_output:
            temp_output_path = Path(temp_output.name)

        cmd = [
            "kokoro-tts",
            str(temp_input_path),
            str(temp_output_path),
            "--voice",
            voice_id,
            "--format",
            audio_format,
        ]
        mapped_lang = map_kokoro_language(text)
        if mapped_lang:
            cmd.extend(["--lang", mapped_lang])
        if kokoro_model:
            cmd.extend(["--model", kokoro_model])
        if kokoro_voices:
            cmd.extend(["--voices", kokoro_voices])

        print(f"Kokoro CLI command: {' '.join(cmd)}")
        proc = subprocess.run(cmd, text=True, capture_output=True)
        if proc.returncode != 0:
            details = "\n".join(p for p in [proc.stdout.strip(), proc.stderr.strip()] if p)
            raise RuntimeError(f"Kokoro-ajo epäonnistui äänellä '{voice_id}':\n{details}")

        shutil.copyfile(temp_output_path, out_path)
        return voice_id
    finally:
        for temp_path in (temp_input_path, temp_output_path):
            if temp_path and temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass

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
    parser.add_argument("--language", default=None, help="Pronunciation dictionary language/fallback language, not necessarily the spoken language of every SSML fragment.")
    parser.add_argument("--content", default=None, help="Luvun tunniste muodossa ACT.LUKU, esim. 2.8 (otsikkofragmenttia varten).")
    parser.add_argument("--chapter-title", default=None, help="Luvun otsikko; lisätään ensimmäiseksi fragmentiksi.")
    parser.add_argument("--generate-pls", action="store_true")
    parser.add_argument("--pls-out-dir", default="pronunciation_exports")
    parser.add_argument("--renderer", choices=["elevenlabs", "piper", "kokoro", "openai"], default="piper")
    parser.add_argument("--voice-name", default=None, help="Deprecated fallback narrator name. Prefer SSML voice attributes and prompt_narrators.txt.")
    parser.add_argument("--voice-id", default=None, help="Deprecated fallback renderer voice id. Prefer SSML voice attributes and prompt_narrators.txt.")
    parser.add_argument("--model", choices=["v2", "v3"], default="v2")
    parser.add_argument("--kokoro-model", default=None, help="Optional Kokoro model path/name for kokoro-tts.")
    parser.add_argument("--kokoro-voices", default=None, help="Optional Kokoro voices path for kokoro-tts.")
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
    narrators_path = Path(args.narrators_file)
    if not narrators_path.exists():
        print(f"Error: narrators file is required for fallback narrator resolution: {args.narrators_file}", file=sys.stderr)
        return 2
    narrators = load_narrators(narrators_path)
    ssml_languages = extract_ssml_languages(content)
    pronunciation_language, pronunciation_lang_note = resolve_pronunciation_language(args.language, ssml_languages)
    pronunciation_locators = load_pronunciation_locators(Path(args.pronunciation_file))
    pronunciation_dictionary = None
    pronunciation_map: list[tuple[re.Pattern[str], str, str]] = []
    dictionary_path = Path(args.pronunciation_dictionary) if args.pronunciation_dictionary else None
    if dictionary_path and dictionary_path.exists():
        pronunciation_dictionary = load_pronunciation_dictionary(dictionary_path)
        pronunciation_map = build_pronunciation_map(pronunciation_dictionary, pronunciation_language)
    elif args.pronunciation_dictionary:
        print(f"Virhe: pronunciation dictionaryä ei löydy: {args.pronunciation_dictionary}", file=sys.stderr)
        return 2
    try:
        ensure_renderer_installed(args.renderer)
        ensure_ffmpeg_installed()
    except RuntimeError as e:
        print(f"Virhe: {e}", file=sys.stderr)
        return 2

    fallback_voice_id = fallback_narrator_voice_id(narrators, args.renderer)
    if not fallback_voice_id and not args.voice_id:
        print(
            f"No voice id found for renderer '{args.renderer}'. Add voices.Kertoja.ids.{args.renderer} to prompt_narrators.txt or specify a renderer-specific SSML voice attribute.",
            file=sys.stderr,
        )
        return 2

    model_id = args.model_id or MODEL_MAP[args.model]
    chunk_limit = min(args.chunk_limit, MAX_RENDERER_CHUNK_LIMIT)
    chunks = split_ssml_chunks(content, chunk_limit)
    if args.content and args.chapter_title:
        num_map = {"0": "nolla", "1": "yksi", "2": "kaksi", "3": "kolme", "4": "neljä", "5": "viisi", "6": "kuusi", "7": "seitsemän", "8": "kahdeksan", "9": "yhdeksän"}
        spoken_idx = " ".join("piste" if ch == "." else num_map.get(ch, ch) for ch in args.content.strip())
        heading = f"Luku {args.content.strip()} {args.chapter_title.strip()}"
        heading_spoken = f"Luku {spoken_idx} {args.chapter_title.strip()}"
        heading_chunk = f'<speak>\n<voice name="{NARRATOR_NAME}" language="finnish" openai_instructions="Lue suomeksi selkeästi ja luonnollisesti.">{escape(heading_spoken)}</voice>\n</speak>'
        chunks.insert(0, (NARRATOR_NAME, heading_chunk))
        print(f"Luvun otsikko lisätty: {heading}")
    print(f"Renderer: {args.renderer}")
    print(f"Fallback narrator: {NARRATOR_NAME} -> {fallback_voice_id or 'none'}")
    if args.voice_name or args.voice_id:
        print(f"CLI voice fallback: voice-name={args.voice_name or 'none'} voice-id={args.voice_id or 'none'} (deprecated)")
    else:
        print("CLI voice fallback: none")
    if args.renderer == "elevenlabs":
        print(f"Model: {args.model} ({model_id})")
    elif args.renderer == "openai":
        print("Model: gpt-4o-mini-tts")
    print(f"Chunkit: {len(chunks)} kpl")
    print(f"Pronunciation dictionary language: {pronunciation_language}")
    print(f"Pronunciation entries loaded: {len(pronunciation_map)}")
    print(f"SSML fragment languages detected: {', '.join(sorted(ssml_languages)) if ssml_languages else 'none'}")
    if args.renderer == "openai":
        if has_openai_language_instructions(content):
            print("OpenAI language guidance: from per-fragment openai_instructions")
        elif ssml_languages:
            print("OpenAI language guidance: inferred from <voice language=\"...\"> attributes")
    if pronunciation_lang_note:
        print(pronunciation_lang_note)
    if pronunciation_map:
        preview = [f"{term} -> {repl}" for _, repl, term in pronunciation_map[:20]]
        print(f"Aktiiviset replacementit (max20): {preview}")
    if args.renderer == "elevenlabs":
        if pronunciation_locators:
            print(f"ElevenLabs pronunciation locators: {pronunciation_locators}")
    elif pronunciation_locators and len(pronunciation_map) == 0:
        print(f"Warning: ElevenLabs pronunciation locators are loaded, but renderer is {args.renderer}. These locators are only used by ElevenLabs. OpenAI/Piper/Kokoro need local pronunciation_dictionary.json entries.")

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
                chunk_voice_id, chunk_voice_source = resolve_voice_id_for_fragment(args.renderer, before, speaker, narrators, args.voice_name, args.voice_id)
                out_path = parts_dir / f"{chapter_prefix}_{i:04d}.mp3"
                print(f"[{i}] -> {out_path} ({len(before)} merkkiä) speaker={speaker} voice={chunk_voice_id} voice_source={chunk_voice_source}")
                try:
                    if args.renderer == "elevenlabs":
                        synthesize_one(api_key, chunk_voice_id, before, out_path, model_id, args.stability, args.similarity_boost, args.style, not args.no_speaker_boost, pronunciation_locators, OUTPUT_FORMAT)
                    else:
                        chunk_voice_id = synthesize_local(args.renderer, chunk_voice_id, before, out_path, args.format, pronunciation_map, args.kokoro_model, args.kokoro_voices)
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

        chunk_voice_id, chunk_voice_source = resolve_voice_id_for_fragment(args.renderer, chunk, speaker, narrators, args.voice_name, args.voice_id)
        if args.renderer == "elevenlabs":
            i = part_no
            out_path = parts_dir / f"{chapter_prefix}_{i:04d}.mp3"
            print(f"[{i}] -> {out_path} ({len(chunk)} merkkiä) speaker={speaker} voice={chunk_voice_id} voice_source={chunk_voice_source}")
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
                        resolved_voice_id, resolved_voice_source = resolve_voice_id_for_fragment(args.renderer, frag_text, speaker, narrators, args.voice_name, args.voice_id)
                        clean_text = strip_ssml_tags(frag_text)
                        if not clean_text:
                            continue
                        print(f"[{part_no}] -> {out_path} ({len(frag_text)} merkkiä) speaker={speaker} voice={resolved_voice_id} voice_source={resolved_voice_source}")
                        chunk_voice_id = synthesize_local(args.renderer, resolved_voice_id, clean_text, out_path, args.format, pronunciation_map, args.kokoro_model, args.kokoro_voices)
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
