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
import importlib
from dataclasses import dataclass
from typing import Any
from xml.sax.saxutils import escape

from openai import OpenAI

API_BASE = "https://api.elevenlabs.io/v1"
OUTPUT_FORMAT = "mp3_44100_128"
DEFAULT_AUDIO_FORMAT = "mp3"
MAX_RENDERER_CHUNK_LIMIT = 4000
DEFAULT_NARRATORS_FILE = "prompt_narrators.txt"
DEFAULT_PRONUNCIATION_FILE = "pronunciation/prompt_pronunciation.json"
DEFAULT_PRONUNCIATION_DICTIONARY = "pronunciation/pronunciation_dictionary.json"
MAX_TEXT_LEN = 10000
DEFAULT_CHUNK_LIMIT = MAX_RENDERER_CHUNK_LIMIT
NARRATOR_NAME = "Kertoja"
MODEL_MAP = {"v2": "eleven_multilingual_v2", "v3": "eleven_v3"}


def parse_bool_arg(value: str) -> bool:
    v = (value or "").strip().lower()
    if v in {"true", "yes", "1", "on"}:
        return True
    if v in {"false", "no", "0", "off"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected: true/false/yes/no/1/0/on/off")


def resolve_code_path(path_arg: str, code_directory: Path) -> Path:
    p = Path(path_arg).expanduser()
    return p if p.is_absolute() else (code_directory / p)


def resolve_work_path(path_arg: str, work_directory: Path) -> Path:
    p = Path(path_arg).expanduser()
    return p if p.is_absolute() else (work_directory / p)


def resolve_unique_parts_dir(out_dir: Path) -> Path:
    base_name = "parts"
    candidate = out_dir / base_name
    if not candidate.exists():
        return candidate

    i = 2
    while True:
        candidate = out_dir / f"{base_name}{i:03d}"
        if not candidate.exists():
            return candidate
        i += 1


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


def resolve_pronunciation_locators_path(path_arg: str, code_directory: Path) -> Path:
    preferred = resolve_code_path(path_arg, code_directory)
    if preferred.exists():
        return preferred
    if preferred.suffix == ".json":
        fallback = preferred.with_suffix(".txt")
        if fallback.exists():
            return fallback
    return preferred


def load_pronunciation_dictionary(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("pronunciation_dictionary.json pitää olla JSON-objekti.")
    if not isinstance(data.get("entries"), list):
        raise ValueError("pronunciation_dictionary.json: entries pitää olla lista.")
    return data


@dataclass
class PronunciationRule:
    pattern: re.Pattern[str]
    original: str
    replacement: str | None
    instruction: str | None
    ipa: str | None
    origin_language: str | None


def preview_pronunciation_rules(active_map: list[Any], limit: int = 20) -> list[str]:
    preview: list[str] = []
    for item in active_map[:limit]:
        if isinstance(item, PronunciationRule):
            replacement = item.replacement or "[no rewrite]"
            extra = ""
            if item.instruction:
                extra = f" | instruction: {item.instruction}"
            elif item.ipa:
                extra = f" | IPA: {item.ipa}"
            preview.append(f"{item.original} -> {replacement}{extra}")
        else:
            try:
                _, replacement, original = item
                preview.append(f"{original} -> {replacement}")
            except Exception:
                preview.append(repr(item))
    return preview


def openai_spoken_hint(entry: dict[str, Any], lang_data: dict[str, Any], language: str) -> str | None:
    if not should_use_openai_alias(entry, lang_data):
        return None
    openai_force_alias = lang_data.get("openai_force_alias")
    if isinstance(openai_force_alias, str) and openai_force_alias.strip():
        return openai_force_alias.strip()
    openai_alias = lang_data.get("openai_alias")
    if isinstance(openai_alias, str) and openai_alias.strip():
        return openai_alias.strip()
    alias = lang_data.get("alias")
    if isinstance(alias, str) and alias.strip():
        alias_clean = alias.strip()
        graphemes = entry.get("graphemes", [])
        if isinstance(graphemes, list) and all((not isinstance(g, str) or g.strip() != alias_clean) for g in graphemes):
            return alias_clean
    spoken_hint = lang_data.get("spoken_hint")
    if isinstance(spoken_hint, str) and spoken_hint.strip():
        return spoken_hint.strip()
    return None


def should_use_openai_alias(entry: dict[str, Any], lang_data: dict[str, Any]) -> bool:
    use_alias = lang_data.get("openai_use_alias")
    if use_alias is False:
        return False
    openai_force_alias = lang_data.get("openai_force_alias")
    if isinstance(openai_force_alias, str) and openai_force_alias.strip():
        return True
    openai_alias = lang_data.get("openai_alias")
    alias_clean = openai_alias.strip() if isinstance(openai_alias, str) and openai_alias.strip() else None
    origin_language = entry.get("origin_language")
    if origin_language == "fi-FI" and alias_clean and "-" in alias_clean:
        return False
    alias = lang_data.get("alias")
    alias_clean = alias.strip() if isinstance(alias, str) and alias.strip() else None
    if origin_language == "fi-FI" and alias_clean and "-" in alias_clean:
        return False
    return bool(openai_alias or alias or lang_data.get("spoken_hint"))


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


def apply_pronunciation_aliases_with_hits(text: str, pronunciation_map: list[PronunciationRule]) -> tuple[str, list[dict[str, str]]]:
    updated = text
    hits: list[dict[str, str]] = []
    for rule in pronunciation_map:
        if rule.replacement and rule.replacement != rule.original:
            updated, count = rule.pattern.subn(rule.replacement, updated)
        else:
            count = len(rule.pattern.findall(updated))
        if count > 0 and (rule.replacement or rule.instruction or rule.ipa):
            hit: dict[str, str] = {"original": rule.original}
            if rule.replacement:
                hit["replacement"] = rule.replacement
            if rule.instruction:
                hit["instruction"] = rule.instruction
            if rule.ipa:
                hit["ipa"] = rule.ipa
            if rule.origin_language:
                hit["origin_language"] = rule.origin_language
            hits.append(hit)
    return updated, hits


def build_openai_pronunciation_map(dictionary: dict[str, Any], language: str) -> list[PronunciationRule]:
    rules: list[PronunciationRule] = []
    for entry in dictionary.get("entries", []):
        if not isinstance(entry, dict):
            continue
        graphemes = entry.get("graphemes", [])
        pronunciations = entry.get("pronunciations", {})
        lang_data = pronunciations.get(language, {}) if isinstance(pronunciations, dict) else {}
        if not isinstance(graphemes, list) or not isinstance(lang_data, dict):
            continue
        replacement = openai_spoken_hint(entry, lang_data, language)
        instruction = lang_data.get("instruction")
        instruction = instruction.strip() if isinstance(instruction, str) and instruction.strip() else None
        ipa = lang_data.get("ipa")
        ipa = ipa.strip() if isinstance(ipa, str) and ipa.strip() else None
        origin_language = entry.get("origin_language")
        origin_language = origin_language.strip() if isinstance(origin_language, str) and origin_language.strip() else None
        for g in graphemes:
            if not isinstance(g, str) or not g.strip():
                continue
            term = g.strip()
            if term.startswith("#") or term.startswith("@"):
                regex = re.compile(rf"(?<![\w]){re.escape(term)}(?![\w])")
            else:
                regex = re.compile(rf"(?<![\w@#]){re.escape(term)}(?![\w])")
            rules.append(PronunciationRule(regex, term, replacement, instruction, ipa, origin_language))
    rules.sort(key=lambda x: len(x.original), reverse=True)
    return rules


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


def normalise_ssml_language(language: str | None) -> str | None:
    value = (language or "").strip().lower()
    mapping = {
        "finnish": "fi-FI",
        "fi": "fi-FI",
        "fi-fi": "fi-FI",
        "suomi": "fi-FI",
        "english": "en-GB",
        "en": "en-GB",
        "en-gb": "en-GB",
        "en-us": "en-US",
        "spanish": "es-MX",
        "es": "es-MX",
        "es-mx": "es-MX",
        "espanol": "es-MX",
        "español": "es-MX",
    }
    return mapping.get(value)


def piper_voice_language_prefix(voice_id: str) -> str | None:
    m = re.match(r"^([a-z]{2}_[A-Z]{2})-", voice_id.strip())
    return m.group(1) if m else None


def validate_piper_language_match(text: str, voice_id: str) -> tuple[str | None, str | None]:
    attrs = extract_voice_attrs(text)
    ssml_language = attrs.get("language")
    expected = normalise_ssml_language(ssml_language)
    actual = piper_voice_language_prefix(voice_id)
    print(f"Piper language expected from SSML: {expected or 'unknown'}")
    print(f"Piper voice language: {actual or 'unknown'}")
    if not expected or not actual:
        return expected, actual
    expected_prefix = expected.replace("-", "_") if expected else None
    if expected_prefix and expected_prefix != actual:
        print(
            f"Warning: SSML language is {ssml_language!r} ({expected}), but Piper voice '{voice_id}' appears to be {actual}. "
            "Piper language is controlled by the selected voice model, so pronunciation may be wrong.",
            file=sys.stderr,
        )
    return expected, actual


def _language_variants(language: str | None) -> list[str]:
    if not language:
        return []
    variants: dict[str, list[str]] = {
        "fi-FI": ["fi-FI", "fi", "finnish"],
        "en-GB": ["en-GB", "en", "english"],
        "en-US": ["en-US", "en", "english"],
        "es-MX": ["es-MX", "es", "spanish"],
    }
    return variants.get(language, [language])


def _legacy_voice_matches_language(renderer: str, voice_id: str, language: str | None) -> bool:
    if not language or renderer != "piper":
        return True
    prefixes = {
        "fi-FI": ("fi_FI-",),
        "en-GB": ("en_GB-",),
        "en-US": ("en_US-",),
        "es-MX": ("es_MX-", "es_ES-"),
    }
    return any(voice_id.strip().startswith(prefix) for prefix in prefixes.get(language, ()))


def narrator_voice_id(narrators: dict[str, Any], narrator_name: str, renderer: str, language: str | None = None) -> str | None:
    voices = narrators.get("voices", {})
    speaker_data = voices.get(narrator_name, {}) if isinstance(voices, dict) else {}
    if not isinstance(speaker_data, dict):
        return None
    ids = speaker_data.get("ids", {})
    if not isinstance(ids, dict):
        return None
    renderer_value = ids.get(renderer)
    if isinstance(renderer_value, dict):
        normalised = normalise_ssml_language(language)
        for key in _language_variants(normalised):
            value = renderer_value.get(key)
            if value:
                return str(value)
        return None
    if isinstance(renderer_value, str) and _legacy_voice_matches_language(renderer, renderer_value, normalise_ssml_language(language)):
        return renderer_value
    return None


def fallback_narrator_voice_id(narrators: dict[str, Any], renderer: str, language: str | None = None) -> str | None:
    return narrator_voice_id(narrators, NARRATOR_NAME, renderer, language)


def resolve_voice_id_for_fragment(renderer: str, fragment_text: str, speaker: str, narrators: dict[str, Any], cli_voice_name: str | None, cli_voice_id: str | None, use_multipolyfony: bool = False) -> tuple[str, str]:
    attrs = extract_voice_attrs(fragment_text)
    ssml_language = normalise_ssml_language(attrs.get("language"))
    if not use_multipolyfony:
        fallback_voice = fallback_narrator_voice_id(narrators, renderer, ssml_language) or fallback_narrator_voice_id(narrators, renderer)
        if fallback_voice:
            return fallback_voice, "single_voice_kertoja_mode"

    renderer_attr = attrs.get(f"{renderer}_voice")
    if renderer_attr:
        return renderer_attr, "ssml_renderer_attribute"

    voice_name = attrs.get("name", "").strip()
    if voice_name and looks_like_renderer_voice_id(renderer, voice_name):
        return voice_name, "ssml_direct_voice_id"

    if voice_name:
        mapped = narrator_voice_id(narrators, voice_name, renderer, ssml_language)
        if mapped:
            return mapped, "narrator_mapping_language_specific" if ssml_language else "narrator_mapping"

    speaker_mapped = narrator_voice_id(narrators, speaker, renderer, ssml_language)
    if speaker_mapped:
        return speaker_mapped, "narrator_mapping_language_specific" if ssml_language else "narrator_mapping"

    fallback_voice = fallback_narrator_voice_id(narrators, renderer, ssml_language)
    if fallback_voice:
        return fallback_voice, "narrator_fallback_kertoja_language_specific" if ssml_language else "narrator_fallback_kertoja"

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


def detect_fragment_language(text: str, cli_language: str | None = None) -> str:
    attrs = extract_voice_attrs(text)
    lang = normalise_ssml_language(attrs.get("language"))
    if lang:
        return lang
    if cli_language and normalise_ssml_language(cli_language):
        return normalise_ssml_language(cli_language) or "en-GB"
    stripped = strip_ssml_tags(text)
    if re.search(r"[åäöÅÄÖ]", stripped):
        return "fi-FI"
    if re.search(r"[¿¡ñáéíóúüÑÁÉÍÓÚÜ]", stripped):
        return "es-MX"
    return "en-GB"




def build_elias_override_instruction(language: str | None) -> str:
    if language in {"en-GB", "en-US"}:
        return (
            'Pronounce "Elias" as a Finnish proper name, close to "EL-yahs", with stress at the start. '
            "Do not spell it out."
        )
    return ""


def apply_openai_pronunciation_overrides(text: str, language: str | None, hits: list[dict[str, str]]) -> list[dict[str, str]]:
    updated_hits = list(hits)
    if re.search(r"(?<![\w@#])Elias(?![\w])", text, flags=re.IGNORECASE):
        if language == "fi-FI":
            return updated_hits
        if language not in {"en-GB", "en-US"}:
            return updated_hits
        instruction = build_elias_override_instruction(language)
        if not instruction:
            return updated_hits
        has_elias_hit = any((hit.get("original", "").strip().lower() == "elias" for hit in updated_hits))
        if not has_elias_hit:
            updated_hits.append({
                "original": "Elias",
                "instruction": instruction,
                "origin_language": "fi-FI",
            })
        else:
            for hit in updated_hits:
                if (hit.get("original", "").strip().lower() == "elias"):
                    hit.pop("replacement", None)
                    hit["instruction"] = instruction
                    hit.setdefault("origin_language", "fi-FI")
    return updated_hits
def build_openai_pronunciation_instruction(language: str | None, hits: list[dict[str, str]]) -> str:
    base_by_language = {
        "fi-FI": "Lue teksti suomeksi. Noudata näitä lausumisohjeita täsmällisesti. Jos tekstissä on valmiiksi uudelleenkirjoitettu lausumismuoto, lue se sellaisenaan äläkä palauta alkuperäistä kirjoitusasua.",
        "en-GB": "Read this text in English. Follow the pronunciation instructions exactly. If a term has been rewritten into a spoken form, read the rewritten spoken form as written.",
        "en-US": "Read this text in American English. Follow the pronunciation instructions exactly. If a term has been rewritten into a spoken form, read the rewritten spoken form as written.",
        "es-MX": "Lee este texto en español mexicano. Sigue exactamente las instrucciones de pronunciación. Si un término ya fue reescrito en forma hablada, léelo tal como está escrito.",
    }
    resolved_lang = language if language in base_by_language else "en-GB"
    base = base_by_language[resolved_lang]
    if not hits:
        return base
    has_finnish_origin_hit = any((hit.get("origin_language") or "").strip() == "fi-FI" for hit in hits)
    lines = ["Pronunciation overrides used in this fragment:"]
    if has_finnish_origin_hit:
        if resolved_lang in {"en-GB", "en-US"}:
            lines.append(
                "Finnish proper names in this fragment must keep Finnish pronunciation, not English pronunciation. Use short Finnish syllables and stress the first syllable."
            )
        elif resolved_lang == "fi-FI":
            lines.append(
                "Erisnimet lausutaan suomalaisittain. Pidä tavut lyhyinä ja paino ensimmäisellä tavulla, jos ohjeessa niin sanotaan."
            )
    for hit in hits[:30]:
        original = hit.get("original", "")
        replacement = hit.get("replacement")
        instruction = hit.get("instruction")
        if instruction:
            lines.append(f'- "{original}": {instruction}')
        elif replacement and replacement != original:
            lines.append(f'- "{original}" -> "{replacement}"')
        elif hit.get("ipa"):
            lines.append(f'- "{original}": IPA {hit.get("ipa")}')
    return f"{base}\n" + "\n".join(lines)


def preview_openai_instructions_from_chunk(chunk: str, pronunciation_maps: dict[str, list[tuple[re.Pattern[str], str, str]]] | None, cli_language: str | None = None) -> str:
    fragment_language = detect_fragment_language(chunk, cli_language)
    fragment_map = (pronunciation_maps or {}).get(fragment_language, [])
    _, pronunciation_hits = apply_pronunciation_aliases_with_hits(chunk, fragment_map)
    pronunciation_hits = apply_openai_pronunciation_overrides(chunk, fragment_language, pronunciation_hits)
    return build_openai_pronunciation_instruction(fragment_language, pronunciation_hits)


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


def _check_sounddevice_import() -> tuple[bool, str]:
    proc = subprocess.run(
        [sys.executable, "-c", "import sounddevice"],
        text=True,
        capture_output=True,
    )
    combined = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
    return proc.returncode == 0, combined


def _download_with_help(url: str, destination: Path, what: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {what} to {destination}")
    try:
        request.urlretrieve(url, destination)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download {what} from {url}: {exc}\n"
            f"Manual download:\n"
            f"wget -O {destination} {url}"
        ) from exc


def ensure_kokoro_ready(args: argparse.Namespace) -> tuple[Path, Path]:
    kokoro_cli = shutil.which("kokoro-tts")
    has_kokoro_onnx = False
    try:
        importlib.import_module("kokoro_onnx")
        has_kokoro_onnx = True
    except Exception:
        has_kokoro_onnx = False

    if not kokoro_cli and not has_kokoro_onnx:
        raise RuntimeError(
            "Kokoro is not installed. Activate your virtual environment and run: pip install kokoro-tts kokoro-onnx"
        )

    if kokoro_cli:
        sd_ok, sd_details = _check_sounddevice_import()
        if not sd_ok:
            if "PortAudio library not found" in sd_details:
                raise RuntimeError(
                    "PortAudio is missing for Kokoro CLI.\n"
                    "Install system and Python dependencies manually:\n"
                    "sudo apt update\n"
                    "sudo apt install -y portaudio19-dev libportaudio2 libportaudiocpp0\n"
                    "pip install sounddevice soundfile"
                )
            raise RuntimeError(f"Kokoro sounddevice import failed:\n{sd_details}")

    cache_root = Path(args.data_dir).expanduser()
    cache_dir = cache_root / "kokoro"
    model_candidates = [
        Path(args.kokoro_model).expanduser() if args.kokoro_model else None,
        Path(os.environ["KOKORO_MODEL"]).expanduser() if os.environ.get("KOKORO_MODEL") else None,
        Path.cwd() / "kokoro-v1.0.onnx",
        cache_dir / "kokoro-v1.0.onnx",
    ]
    voices_candidates = [
        Path(args.kokoro_voices).expanduser() if args.kokoro_voices else None,
        Path(os.environ["KOKORO_VOICES"]).expanduser() if os.environ.get("KOKORO_VOICES") else None,
        Path.cwd() / "voices-v1.0.bin",
        cache_dir / "voices-v1.0.bin",
    ]
    model_path = next((p for p in model_candidates if p and p.exists()), None)
    voices_path = next((p for p in voices_candidates if p and p.exists()), None)

    model_url = "https://github.com/nazdridoy/kokoro-tts/releases/download/v1.0.0/kokoro-v1.0.onnx"
    voices_url = "https://github.com/nazdridoy/kokoro-tts/releases/download/v1.0.0/voices-v1.0.bin"
    target_model = cache_dir / "kokoro-v1.0.onnx"
    target_voices = cache_dir / "voices-v1.0.bin"

    if (not model_path or not voices_path) and args.kokoro_auto_download:
        if not model_path:
            _download_with_help(model_url, target_model, "Kokoro model")
            model_path = target_model
        if not voices_path:
            _download_with_help(voices_url, target_voices, "Kokoro voices")
            voices_path = target_voices

    if not model_path or not voices_path:
        raise RuntimeError(
            "Kokoro model files were not found and auto-download is disabled.\n"
            "Please download manually:\n"
            f"wget -O {target_model} {model_url}\n"
            f"wget -O {target_voices} {voices_url}"
        )

    return model_path.resolve(), voices_path.resolve()


def collect_kokoro_voices_from_ssml(text: str) -> set[str]:
    voices: set[str] = set()
    for attrs in re.findall(r"<voice\\s+([^>]*)>", text, flags=re.IGNORECASE):
        explicit = re.search(r'kokoro_voice="([^"]+)"', attrs, flags=re.IGNORECASE)
        if explicit and explicit.group(1).strip():
            voices.add(explicit.group(1).strip())
            continue
        direct = re.search(r'name="([^"]+)"', attrs, flags=re.IGNORECASE)
        if direct and looks_like_kokoro_voice(direct.group(1)):
            voices.add(direct.group(1).strip())
    return voices


def validate_kokoro_voices(ssml_text: str) -> None:
    used = collect_kokoro_voices_from_ssml(ssml_text)
    if not used:
        return
    proc = subprocess.run(["kokoro-tts", "--help-voices"], text=True, capture_output=True)
    if proc.returncode != 0:
        print("Warning: could not list Kokoro voices with 'kokoro-tts --help-voices'. Skipping voice validation.")
        return
    listed = set(re.findall(r"\\b[ab][fm]_[a-z0-9_]+\\b", f"{proc.stdout}\n{proc.stderr}"))
    if not listed:
        print("Warning: could not parse Kokoro voice list from 'kokoro-tts --help-voices'.")
        return
    for voice in sorted(used):
        if voice not in listed:
            print(f"Warning: Kokoro voice '{voice}' was not found in kokoro-tts --help-voices output.")



def piper_download_candidates(voice_id: str) -> list[str]:
    candidates = [voice_id]
    if voice_id.endswith("-medium"):
        candidates.append(voice_id[:-7] + "-low")
    return candidates


def ensure_piper_voice_available(voice_id: str, data_dir: Path) -> str:
    for candidate in piper_download_candidates(voice_id):
        cmd = [sys.executable, "-m", "piper.download_voices", candidate, "--data-dir", str(data_dir)]
        proc = subprocess.run(cmd, text=True, capture_output=True)
        if proc.returncode == 0:
            if candidate != voice_id:
                print(f"Piper-ääni '{voice_id}' ei ollut saatavilla, käytetään ladattua ääntä '{candidate}'.")
            return candidate
    raise RuntimeError(
        f"Piper-äänen automaattinen lataus epäonnistui äänelle '{voice_id}'. "
        "Tarkista saatavilla olevat Piper-äänet ja anna toimiva --voice-id."
    )


def collect_renderer_voices_from_ssml(text: str, renderer: str) -> set[str]:
    voices: set[str] = set()
    for attrs in re.findall(r"<voice\s+([^>]*)>", text, flags=re.IGNORECASE):
        renderer_attr = re.search(rf'{renderer}_voice="([^"]+)"', attrs, flags=re.IGNORECASE)
        if renderer_attr and renderer_attr.group(1).strip():
            voices.add(renderer_attr.group(1).strip())
            continue
        direct = re.search(r'name="([^"]+)"', attrs, flags=re.IGNORECASE)
        if direct and looks_like_renderer_voice_id(renderer, direct.group(1).strip()):
            voices.add(direct.group(1).strip())
    return voices


def ensure_piper_ready(content: str, narrators: dict[str, Any], cli_voice_id: str | None, data_dir: Path) -> None:
    candidate_voices = collect_renderer_voices_from_ssml(content, "piper")
    for language in ("fi-FI", "en-GB", "en-US", "es-MX", None):
        fallback = fallback_narrator_voice_id(narrators, "piper", language)
        if fallback:
            candidate_voices.add(fallback)
    if cli_voice_id:
        candidate_voices.add(cli_voice_id)
    for voice_id in sorted(candidate_voices):
        ensure_piper_voice_available(voice_id, data_dir)


def map_kokoro_language(language: str | None) -> str:
    value = (language or "").strip().lower()
    mapping = {
        "finnish": "fi",
        "fi": "fi",
        "fi-fi": "fi",
        "suomi": "fi",
        "english": "en-us",
        "en": "en-us",
        "en-us": "en-us",
        "en-gb": "en-gb",
        "spanish": "es",
        "es": "es",
        "es-mx": "es",
        "espanol": "es",
        "español": "es",
    }
    return mapping.get(value, "en-us")


def detect_kokoro_fallback_language(text: str, cli_language: str | None) -> str:
    if cli_language and cli_language.strip():
        return map_kokoro_language(cli_language)
    if re.search(r"[åäöÅÄÖ]", text):
        return "fi"
    return "en-us"


def synthesize_local(renderer: str, voice_id: str, text: str, out_path: Path, audio_format: str, pronunciation_maps: dict[str, list[tuple[re.Pattern[str], str, str]]] | None = None, kokoro_model: str | None = None, kokoro_voices: str | None = None, cli_language: str | None = None, piper_data_dir: Path | None = None) -> str:
    if renderer == "openai":
        client = OpenAI()
        ssml_voice, ssml_model, ssml_instructions = extract_openai_ssml_options(text)
        resolved_voice = ssml_voice or voice_id
        resolved_model = ssml_model or "gpt-4o-mini-tts"
        fragment_language = detect_fragment_language(text, cli_language)
        fragment_map = (pronunciation_maps or {}).get(fragment_language, [])
        rewritten, pronunciation_hits = apply_pronunciation_aliases_with_hits(text, fragment_map)
        pronunciation_hits = apply_openai_pronunciation_overrides(text, fragment_language, pronunciation_hits)
        clean_text = strip_ssml_tags(rewritten)
        for rule in fragment_map:
            if (
                isinstance(rule, PronunciationRule)
                and rule.origin_language == "fi-FI"
                and rule.replacement
                and "-" in rule.replacement
                and re.search(rf"(?<![\\w@#]){re.escape(rule.replacement)}(?![\\w])", clean_text)
            ):
                print(
                    f"WARNING: OpenAI input contains hyphenated Finnish pronunciation alias '{rule.replacement}' for '{rule.original}'. "
                    "This may cause TTS to spell or over-segment the name. Use the original name plus natural-language instruction instead."
                )
        if not clean_text:
            return resolved_voice
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": resolved_model,
            "voice": resolved_voice,
            "input": clean_text,
            "response_format": audio_format,
        }
        pronunciation_instruction = build_openai_pronunciation_instruction(fragment_language, pronunciation_hits)
        payload["instructions"] = f"{(ssml_instructions or '').strip()} {pronunciation_instruction}".strip()
        print(
            f"OpenAI fragment debug: language={fragment_language}, voice={resolved_voice}, pronunciation_hits={len(pronunciation_hits)}"
        )
        print(f"OpenAI final text: {clean_text}")
        print(f"OpenAI final instructions: {payload['instructions']}")
        if pronunciation_hits:
            preview_items: list[str] = []
            for hit in pronunciation_hits[:5]:
                original = hit.get("original", "")
                replacement = hit.get("replacement")
                instruction = hit.get("instruction")
                item = f"{original} -> {replacement}" if replacement and replacement != original else f"{original} -> [no rewrite]"
                if instruction:
                    item += f", instruction: {instruction}"
                preview_items.append(item)
            print(f"OpenAI fragment pronunciation preview: {'; '.join(preview_items)}")
        with client.audio.speech.with_streaming_response.create(**payload) as response:
            response.stream_to_file(out_path)
        return resolved_voice

    if renderer == "piper":
        fragment_language = detect_fragment_language(text, cli_language)
        text = apply_pronunciation_aliases(text, (pronunciation_maps or {}).get(fragment_language, []))
        expected = normalise_ssml_language(extract_voice_attrs(text).get("language"))
        if expected == "fi-FI" and not voice_id.strip().startswith("fi_FI-"):
            raise RuntimeError(
                f"Error: Finnish SSML fragment resolved to non-Finnish Piper voice '{voice_id}'. "
                "Configure voices.Kertoja.ids.piper.fi-FI or the character-specific piper.fi-FI voice."
            )
        validate_piper_language_match(text, voice_id)
        cmd = ["piper", "--model", voice_id, "--output_file", str(out_path)]
        proc = subprocess.run(cmd, input=strip_ssml_tags(text), text=True, capture_output=True)
        if proc.returncode == 0:
            return voice_id
        combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
        if "Unable to find voice" in combined:
            if not piper_data_dir:
                raise RuntimeError("Piper data directory is missing for auto-download.")
            resolved_voice = ensure_piper_voice_available(voice_id, piper_data_dir)
            retry = subprocess.run(["piper", "--model", resolved_voice, "--output_file", str(out_path)], input=strip_ssml_tags(text), text=True, capture_output=True)
            if retry.returncode == 0:
                return resolved_voice
            raise RuntimeError(f"Piper-ajo epäonnistui äänellä '{resolved_voice}': {retry.stderr.strip() or retry.stdout.strip()}")
        raise RuntimeError(f"Piper-ajo epäonnistui äänellä '{voice_id}': {proc.stderr.strip() or proc.stdout.strip()}")

    fragment_language = detect_fragment_language(text, cli_language)
    text = apply_pronunciation_aliases(text, (pronunciation_maps or {}).get(fragment_language, []))
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
        attrs = extract_voice_attrs(text)
        mapped_lang = map_kokoro_language(attrs.get("language")) if attrs.get("language") else detect_kokoro_fallback_language(text, cli_language)
        if attrs.get("language") and mapped_lang == "en-us" and normalise_ssml_language(attrs.get("language")) in {"fi-FI", "es-MX"}:
            raise RuntimeError("Kokoro language mapping is missing for this SSML language; configure a language-specific Kokoro voice.")
        cmd.extend(["--lang", mapped_lang])
        print(f"Kokoro language: {mapped_lang}")
        if attrs.get("language", "").strip().lower() == "finnish" and mapped_lang != "fi":
            print("Warning: BUG: Kokoro language mapping mismatch; language='finnish' must map to --lang fi.")
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


def split_text_and_breaks(text: str, default_pause_s: float = 0.2) -> list[tuple[str, str | float]]:
    parts: list[tuple[str, str | float]] = []
    break_re = re.compile(r'<break\b([^>]*)/?>', flags=re.IGNORECASE)
    pos = 0
    for m in break_re.finditer(text):
        before = text[pos:m.start()]
        if before.strip():
            parts.append(("text", before.strip()))
        attrs = m.group(1) or ""
        tm = re.search(r'time="([0-9.]+)s"', attrs, flags=re.IGNORECASE)
        pause_s = float(tm.group(1)) if tm else default_pause_s
        parts.append(("break", pause_s))
        pos = m.end()
    tail = text[pos:]
    if tail.strip():
        parts.append(("text", tail.strip()))
    return parts


class HelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter,
):
    pass


def main() -> int:
    parser = argparse.ArgumentParser(
        description=("Render an SSML audiobook script into audio using ElevenLabs, OpenAI TTS, Piper or Kokoro. By default the renderer uses a single Kertoja voice unless --use-multipolyfony true is given."),
        formatter_class=HelpFormatter,
        epilog="""Examples:
  Render normal single-narrator audiobook audio with OpenAI TTS:
    python3 tee_aanikirja.py \\
      --renderer openai \\
      --input-file 02_05_yokahvilassa_ssml.xml \\
      --out-dir /tmp/abook/audio \\
      --narrators-file prompt_narrators.txt \\
      --pronunciation-dictionary pronunciation/pronunciation_dictionary.json \\
      --merged-file 02_05_yokahvilassa.mp3

  Render with Piper:
    python3 tee_aanikirja.py \\
      --renderer piper \\
      --input-file 02_05_yokahvilassa_ssml.xml \\
      --out-dir /tmp/abook/audio \\
      --narrators-file prompt_narrators.txt \\
      --merged-file 02_05_yokahvilassa.mp3

  Debug chunking without rendering:
    python3 tee_aanikirja.py \\
      --renderer openai \\
      --input-file 02_05_yokahvilassa_ssml.xml \\
      --dry-run

  Experimental polyphonic rendering:
    python3 tee_aanikirja.py \\
      --renderer elevenlabs \\
      --input-file 02_05_yokahvilassa_ssml.xml \\
      --use-multipolyfony true""",
    )
    required = parser.add_argument_group("required input")
    required.add_argument("--input-file", required=True, metavar="SSML", help="Input SSML file. Relative paths are resolved against --work-directory.")
    output = parser.add_argument_group("output")
    output.add_argument("--out-dir", metavar="DIR", default="audio_parts", help="Output directory. The script creates a parts/ directory inside it and writes the merged audio file there unless --no-merge is used.")
    output.add_argument("--merged-file", metavar="MP3", default=None, help="Path or file name for the merged MP3. If omitted, a default name is used.")
    output.add_argument("--format", choices=["mp3"], default=DEFAULT_AUDIO_FORMAT, help="User-facing output audio format. Currently only mp3 is supported.")
    output.add_argument("--no-merge", action="store_true", help="Do not merge rendered parts into one MP3 file.")
    paths = parser.add_argument_group("paths and support files")
    paths.add_argument("--code-directory", metavar="DIR", default=".", help="Directory for support files such as narrator and pronunciation files.")
    paths.add_argument("--work-directory", metavar="DIR", default=".", help="Base directory for input and output paths.")
    paths.add_argument("--narrators-file", metavar="JSON", default=DEFAULT_NARRATORS_FILE, help="Narrator/voice configuration JSON. Relative paths are resolved against --code-directory.")
    paths.add_argument("--pronunciation-file", metavar="JSON", default=DEFAULT_PRONUNCIATION_FILE, help="ElevenLabs pronunciation dictionary locator file. Relative paths are resolved against --code-directory.")
    paths.add_argument("--pronunciation-dictionary", metavar="JSON", default=None, help="Local pronunciation dictionary used especially for OpenAI, Piper and Kokoro. Relative paths are resolved against --code-directory.")
    paths.add_argument("--data-dir", metavar="DIR", default=str(Path.home() / ".cache" / "om-author"), help="Directory for downloaded runtime assets such as Kokoro and Piper models.")
    renderer = parser.add_argument_group("renderer")
    renderer.add_argument("--renderer", choices=["elevenlabs", "piper", "kokoro", "openai"], default="piper", help="TTS backend.")
    renderer.add_argument("--use-multipolyfony", type=parse_bool_arg, default=False, metavar="BOOL", help="Use SSML speaker-specific voices. false forces all fragments to the Kertoja voice and is the normal production mode. Accepted values: true/false, yes/no, 1/0, on/off.")
    renderer.add_argument("--chunk-limit", type=int, default=DEFAULT_CHUNK_LIMIT, metavar="N", help="Maximum characters per renderer chunk. Keep at or below 4000 for safety.")
    renderer.add_argument("--language", metavar="LANG", default=None, help="Pronunciation dictionary fallback language, for example fi-FI, en-GB, en-US or es-MX. Normally inferred from SSML language attributes.")
    eleven = parser.add_argument_group("ElevenLabs options")
    eleven.add_argument("--model", choices=["v2", "v3"], default="v2", help="ElevenLabs model family.")
    eleven.add_argument("--model-id", default=None, metavar="MODEL_ID", help="Explicit ElevenLabs model id. If omitted, --model is mapped to the default model id.")
    eleven.add_argument("--stability", type=float, default=0.45, help="ElevenLabs voice stability.")
    eleven.add_argument("--similarity-boost", type=float, default=0.75, help="ElevenLabs similarity boost.")
    eleven.add_argument("--style", type=float, default=0.15, help="ElevenLabs style value.")
    eleven.add_argument("--no-speaker-boost", action="store_true", help="Disable ElevenLabs speaker boost.")
    kokoro = parser.add_argument_group("Kokoro options")
    kokoro.add_argument("--kokoro-model", metavar="PATH", default=None, help="Kokoro model path. If omitted, checks CLI argument, environment, current directory and cache.")
    kokoro.add_argument("--kokoro-voices", metavar="PATH", default=None, help="Kokoro voices file path. If omitted, checks CLI argument, environment, current directory and cache.")
    kokoro.add_argument("--kokoro-auto-download", dest="kokoro_auto_download", action="store_true", default=True, help="Automatically download missing Kokoro model files into the cache directory.")
    kokoro.add_argument("--no-kokoro-auto-download", dest="kokoro_auto_download", action="store_false", help="Disable automatic Kokoro model download.")
    heading = parser.add_argument_group("chapter heading compatibility")
    heading.add_argument("--content", metavar="ACT.CHAPTER", default=None, help="Chapter identifier used only for renderer-side heading insertion. Normally the heading should already be present in the SSML.")
    heading.add_argument("--chapter-title", metavar="TITLE", default=None, help="Chapter title inserted as the first rendered fragment. Normally generated earlier by tee_aanikasikirjoitus.py; avoid using this unless rendering older SSML.")
    deprecated = parser.add_argument_group("deprecated voice fallback options")
    deprecated.add_argument("--voice-name", default=None, metavar="NAME", help="Deprecated fallback narrator name. Prefer SSML voice attributes and prompt_narrators.txt.")
    deprecated.add_argument("--voice-id", default=None, metavar="VOICE_ID", help="Deprecated fallback renderer voice id. Prefer SSML voice attributes and prompt_narrators.txt.")
    debug = parser.add_argument_group("debug and utilities")
    debug.add_argument("--dry-run", action="store_true", help="Parse, split and print planned chunks without rendering audio.")
    debug.add_argument("--generate-pls", action="store_true", help="Export pronunciation dictionary as PLS files and exit.")
    debug.add_argument("--pls-out-dir", metavar="DIR", default="pronunciation_exports", help="Output directory for generated PLS pronunciation files.")
    debug.add_argument("--dialogue-to-narrator-pause-ms", type=int, default=200, metavar="MS", help="Pause inserted when switching from dialogue to narrator.")
    debug.add_argument("--narrator-to-dialogue-pause-ms", type=int, default=200, metavar="MS", help="Pause inserted when switching from narrator to dialogue.")
    debug.add_argument("--default-inline-pause-ms", type=int, default=200, metavar="MS", help="Default pause for inline SSML break handling when no explicit duration is found.")
    args = parser.parse_args()

    code_directory = Path(args.code_directory).expanduser()
    work_directory = Path(args.work_directory).expanduser()
    input_path = resolve_work_path(args.input_file, work_directory)
    content = input_path.read_text(encoding="utf-8").strip()
    narrators_path = resolve_code_path(args.narrators_file, code_directory)
    if not narrators_path.exists():
        print(f"Error: narrators file is required for fallback narrator resolution: {args.narrators_file}", file=sys.stderr)
        return 2
    narrators = load_narrators(narrators_path)
    ssml_languages = extract_ssml_languages(content)
    pronunciation_language, pronunciation_lang_note = resolve_pronunciation_language(args.language, ssml_languages)
    pronunciation_locators = load_pronunciation_locators(resolve_pronunciation_locators_path(args.pronunciation_file, code_directory))
    pronunciation_dictionary = None
    pronunciation_maps: dict[str, list[Any]] = {}
    dictionary_path = resolve_code_path(args.pronunciation_dictionary, code_directory) if args.pronunciation_dictionary else None
    if dictionary_path and dictionary_path.exists():
        pronunciation_dictionary = load_pronunciation_dictionary(dictionary_path)
        for language in ("fi-FI", "en-GB", "en-US", "es-MX"):
            if args.renderer == "openai":
                pronunciation_maps[language] = build_openai_pronunciation_map(pronunciation_dictionary, language)
            else:
                pronunciation_maps[language] = build_pronunciation_map(pronunciation_dictionary, language)
    elif args.pronunciation_dictionary:
        print(f"Virhe: pronunciation dictionaryä ei löydy: {args.pronunciation_dictionary}", file=sys.stderr)
        return 2
    if args.renderer == "openai" and pronunciation_maps:
        all_rules = [r for rules in pronunciation_maps.values() for r in rules if isinstance(r, PronunciationRule)]
        if all_rules and all((not r.replacement or r.replacement == r.original) and not r.instruction for r in all_rules):
            example = all_rules[0]
            example_replacement = example.replacement or example.original
            print(
                "Warning: pronunciation dictionary contains entries, but matching entries do not provide "
                f"OpenAI-usable aliases or instructions. Example: {example.original} -> {example_replacement}. "
                "Add openai_alias or instruction fields."
            )
    try:
        ensure_renderer_installed(args.renderer)
        ensure_ffmpeg_installed()
        if args.renderer == "kokoro":
            model_path, voices_path = ensure_kokoro_ready(args)
            args.kokoro_model = str(model_path)
            args.kokoro_voices = str(voices_path)
            print("Kokoro backend: kokoro-tts CLI")
            print(f"Kokoro model: {args.kokoro_model}")
            print(f"Kokoro voices: {args.kokoro_voices}")
            print(f"Kokoro auto-download: {'enabled' if args.kokoro_auto_download else 'disabled'}")
            validate_kokoro_voices(content)
        elif args.renderer == "piper":
            ensure_piper_ready(content, narrators, args.voice_id, Path(args.data_dir).expanduser() / "piper")
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
    renderer_ids = narrators.get("voices", {}).get(NARRATOR_NAME, {}).get("ids", {}).get(args.renderer)
    if isinstance(renderer_ids, dict):
        print(f"Fallback narrator voices for renderer {args.renderer}:")
        for language in ("fi-FI", "en-GB", "en-US", "es-MX"):
            value = narrator_voice_id(narrators, NARRATOR_NAME, args.renderer, language)
            print(f"  {language} -> {value or 'not configured'}")
    else:
        print(f"Fallback narrator voice for renderer {args.renderer}:")
        print(f"  legacy -> {fallback_voice_id or 'none'}")
        if isinstance(renderer_ids, str):
            print("Warning: legacy single-language fallback may be unsafe for multilingual SSML.")
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
    active_map = pronunciation_maps.get(pronunciation_language, [])
    print(f"Pronunciation entries loaded: {len(active_map)}")
    print(f"SSML fragment languages detected: {', '.join(sorted(ssml_languages)) if ssml_languages else 'none'}")
    if args.renderer == "openai":
        if has_openai_language_instructions(content):
            print("OpenAI language guidance: from per-fragment openai_instructions")
        elif ssml_languages:
            print("OpenAI language guidance: inferred from <voice language=\"...\"> attributes")
    if pronunciation_lang_note:
        print(pronunciation_lang_note)
    if active_map:
        preview = preview_pronunciation_rules(active_map, limit=20)
        print(f"Aktiiviset replacementit (max20): {preview}")
    if args.renderer == "elevenlabs":
        if pronunciation_locators:
            print(f"ElevenLabs pronunciation locators: {pronunciation_locators}")
    elif pronunciation_locators and len(active_map) == 0:
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
        if args.renderer == "openai":
            preview_text = ""
            for _, chunk in chunks:
                candidate = chunk.strip()
                if candidate and not is_pause_only_ssml(candidate):
                    preview_text = preview_openai_instructions_from_chunk(candidate, pronunciation_maps, args.language)
                    break
            if preview_text:
                print("OpenAI instructions preview:")
                print(preview_text[:1000])
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

    out_dir = resolve_work_path(args.out_dir, work_directory)
    parts_dir = resolve_unique_parts_dir(out_dir)
    parts_dir.mkdir(parents=True, exist_ok=True)
    print(f"Osat tallennetaan hakemistoon: {parts_dir}")
    script_path = input_path.resolve()
    script_dir = script_path.parent
    chapter_prefix = script_path.stem
    part_no = 1
    quota_exhausted = False
    for speaker, chunk in chunks:
        pending = chunk
        while True:
            audio_m = AUDIO_TAG_RE.search(pending)
            if not audio_m:
                break

            before = pending[:audio_m.start()].strip()
            if before and not is_pause_only_ssml(before):
                i = part_no
                chunk_voice_id, chunk_voice_source = resolve_voice_id_for_fragment(args.renderer, before, speaker, narrators, args.voice_name, args.voice_id, args.use_multipolyfony)
                out_path = parts_dir / f"{chapter_prefix}_{i:04d}.mp3"
                before_attrs = extract_voice_attrs(before)
                print(f"[{i}] -> {out_path} ({len(before)} merkkiä) speaker={speaker} voice={chunk_voice_id} voice_source={chunk_voice_source} ssml_language={normalise_ssml_language(before_attrs.get('language')) or 'unknown'}")
                try:
                    if args.renderer == "elevenlabs":
                        synthesize_one(api_key, chunk_voice_id, before, out_path, model_id, args.stability, args.similarity_boost, args.style, not args.no_speaker_boost, pronunciation_locators, OUTPUT_FORMAT)
                    else:
                        chunk_voice_id = synthesize_local(args.renderer, chunk_voice_id, before, out_path, args.format, pronunciation_maps, args.kokoro_model, args.kokoro_voices, args.language, Path(args.data_dir).expanduser() / "piper")
                except QuotaExceededError as e:
                    print(f"Krediitit loppuivat kesken: {e}", file=sys.stderr)
                    quota_exhausted = True
                    break
                except RuntimeError as e:
                    print(f"Virhe: {e}", file=sys.stderr)
                    return 2
                part_no += 1

            cue_name = (audio_m.group(1) or "").strip()
            safe_name = re.sub(r"[^A-Za-z0-9_-]+", "", cue_name)
            cue_src = script_dir / f"{safe_name}.mp3"
            if cue_src.exists():
                out_path = parts_dir / f"{chapter_prefix}_{part_no:04d}.mp3"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(cue_src, out_path)
                print(f"[{part_no}] -> {out_path} ({cue_src.name})")
                part_no += 1
            else:
                print(f"Warning: audio cue file not found: {safe_name}.mp3", file=sys.stderr)

            if quota_exhausted:
                break

            pending = pending[audio_m.end():]
            pending, pause_s = pop_leading_break_seconds(pending)
            if pause_s is None:
                is_dialogue_to_narrator = speaker != NARRATOR_NAME
                pause_s = (args.dialogue_to_narrator_pause_ms if is_dialogue_to_narrator else args.narrator_to_dialogue_pause_ms) / 1000.0
                print(
                    "Pause decision:\n"
                    f"  previous_speaker: {speaker}\n"
                    f"  next_speaker: unknown\n"
                    "  boundary_type: voice_change_same_paragraph\n"
                    f"  pause_ms: {int(pause_s * 1000)}"
                )
            out_path = parts_dir / f"{chapter_prefix}_{part_no:04d}.mp3"
            render_silence(out_path, pause_s)
            print(f"[{part_no}] -> {out_path} (silence {pause_s:.2f}s)")
            part_no += 1

        chunk = pending.strip()
        if not chunk or is_pause_only_ssml(chunk):
            continue

        chunk_voice_id, chunk_voice_source = resolve_voice_id_for_fragment(args.renderer, chunk, speaker, narrators, args.voice_name, args.voice_id, args.use_multipolyfony)
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
                for frag_type, value in split_text_and_breaks(chunk, default_pause_s=args.default_inline_pause_ms / 1000.0):
                    out_path = parts_dir / f"{chapter_prefix}_{part_no:04d}.mp3"
                    if frag_type == "break":
                        pause_s = float(value)
                        render_silence(out_path, pause_s)
                        print(f"[{part_no}] -> {out_path} (silence {pause_s:.2f}s)")
                    else:
                        frag_text = str(value)
                        if is_pause_only_ssml(frag_text):
                            continue
                        resolved_voice_id, resolved_voice_source = resolve_voice_id_for_fragment(args.renderer, frag_text, speaker, narrators, args.voice_name, args.voice_id, args.use_multipolyfony)
                        clean_text = strip_ssml_tags(frag_text)
                        if not clean_text:
                            continue
                        frag_attrs = extract_voice_attrs(frag_text)
                        print(f"[{part_no}] -> {out_path} ({len(frag_text)} merkkiä) speaker={speaker} voice={resolved_voice_id} voice_source={resolved_voice_source} ssml_language={normalise_ssml_language(frag_attrs.get('language')) or 'unknown'}")
                        render_text = frag_text if args.renderer == "piper" else clean_text
                        chunk_voice_id = synthesize_local(args.renderer, resolved_voice_id, render_text, out_path, args.format, pronunciation_maps, args.kokoro_model, args.kokoro_voices, args.language, Path(args.data_dir).expanduser() / "piper")
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
            merged_path = resolve_work_path(args.merged_file, work_directory) if args.merged_file else out_dir / f"{chapter_prefix}_fullchapter.mp3"
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
AUDIO_TAG_RE = re.compile(r'<audio\s+[^>]*src="([^"]+)"[^>]*/>', flags=re.IGNORECASE)
