#!/usr/bin/env python3
"""Extract one Act.Chapter section from a .docx manuscript for ElevenLabs script generation."""

from __future__ import annotations

import argparse
import difflib
import json
import random
import re
import sys
from pathlib import Path

from docx import Document

NARRATOR_NAME = "Kertoja"


def style_level(style_name: str) -> int | None:
    name = style_name.strip().lower()
    m = re.search(r"(\d+)$", name)
    if m:
        return int(m.group(1))
    if "otsikko 1" in name or "heading 1" in name:
        return 1
    if "otsikko 2" in name or "heading 2" in name:
        return 2
    return None


def load_narrators(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Narrators-tiedoston pitää olla JSON-objekti")

    if "voices" in data:
        voices = data.get("voices", {})
        if not isinstance(voices, dict):
            raise ValueError("Narrators-tiedoston 'voices' pitää olla objekti")
        out: dict[str, str] = {}
        for name, meta in voices.items():
            if not isinstance(meta, dict):
                continue
            ids = meta.get("ids", {})
            if not isinstance(ids, dict):
                continue
            eleven = ids.get("elevenlabs")
            if eleven:
                out[str(name)] = str(eleven)
        if NARRATOR_NAME not in out:
            raise ValueError(f"Narrators-tiedostossa pitää olla '{NARRATOR_NAME}' elevenlabs-id")
        return out

    if NARRATOR_NAME not in data:
        raise ValueError(f"Narrators-tiedostossa pitää olla '{NARRATOR_NAME}'")
    return {str(k): str(v) for k, v in data.items()}


def extract_section(docx_path: Path, content_id: str) -> tuple[str, int, int, str]:
    if not re.fullmatch(r"\d+\.\d+", content_id):
        raise ValueError("--content pitää olla muodossa ACT.LUKU, esim. 1.3")

    target_act, target_chapter = map(int, content_id.split("."))
    document = Document(str(docx_path))

    act = 0
    chapter = 0
    collecting = False
    lines: list[str] = []

    chapter_title = "luku"

    for para in document.paragraphs:
        text = para.text.strip()
        level = style_level(getattr(para.style, "name", ""))
        if level == 1:
            act += 1
            chapter = 0
            if collecting:
                break
            continue
        if level == 2:
            chapter += 1
            if collecting:
                break
            collecting = act == target_act and chapter == target_chapter
            if collecting:
                chapter_title = text or f"Luku {chapter}"
            continue
        if collecting and text:
            lines.append(text)

    if not lines:
        raise LookupError(f"Sisältöä ei löytynyt kohdalle {content_id} tiedostosta {docx_path}")

    return "\n\n".join(lines).strip() + "\n", target_act, target_chapter, chapter_title


def normalize_unknown_characters(text: str, narrators: dict[str, str]) -> tuple[str, list[str]]:
    chat_pool = [k for k in narrators if k.startswith("chat") and k not in {"chat-mod", "chat-kaira"}]
    nick_voice: dict[str, str] = {}
    narrators_by_normalized = {
        re.sub(r"[^a-z0-9]+", "", key.lower()): key for key in narrators.keys()
    }

    def resolve_character_name(name: str) -> str | None:
        if name in narrators:
            return name

        normalized_name = re.sub(r"[^a-z0-9]+", "", name.lower())
        if not normalized_name:
            return None

        # 1) exact normalized match
        if normalized_name in narrators_by_normalized:
            return narrators_by_normalized[normalized_name]

        # 2) token/substring matching (e.g. "Sanii Sparkle" -> "Sannii", "Susanna Reinboth" -> "Reinboth")
        parts = [p for p in re.split(r"[\s\-_]+", name) if p]
        normalized_parts = [re.sub(r"[^a-z0-9]+", "", p.lower()) for p in parts]
        normalized_parts = [p for p in normalized_parts if p]
        for part in normalized_parts:
            for normalized_key, original_key in narrators_by_normalized.items():
                if part in normalized_key or normalized_key in part:
                    return original_key

        # 3) best fuzzy ratio against known narrator keys
        best_key = None
        best_score = 0.0
        for normalized_key, original_key in narrators_by_normalized.items():
            score = difflib.SequenceMatcher(None, normalized_name, normalized_key).ratio()
            if score > best_score:
                best_score = score
                best_key = original_key
        if best_key and best_score >= 0.72:
            return best_key
        return None

    def chat_voice_for_nick(nick: str) -> str:
        key = nick.strip().lower()
        if key in nick_voice:
            return nick_voice[key]
        if "kaira" in key and "chat-kaira" in narrators:
            nick_voice[key] = "chat-kaira"
            return nick_voice[key]
        if "mod" in key and "chat-mod" in narrators:
            nick_voice[key] = "chat-mod"
            return nick_voice[key]
        if chat_pool:
            nick_voice[key] = random.choice(chat_pool)
            return nick_voice[key]
        nick_voice[key] = "chat" if "chat" in narrators else NARRATOR_NAME
        return nick_voice[key]

    missing: set[str] = set()
    out_lines: list[str] = []
    for line in text.splitlines():
        nick_match = re.match(r"^\[([^\]]{1,40})\]\s*:\s*(.*)$", line)
        if nick_match:
            nick = nick_match.group(1).strip()
            spoken = nick_match.group(2)
            out_lines.append(f"{chat_voice_for_nick(nick)}: {spoken}")
            continue

        m = re.match(r"^([A-ZÅÄÖa-zåäö][\wÅÄÖåäö\- ]{0,40}):\s*(.*)$", line)
        if not m:
            out_lines.append(line)
            continue
        name, spoken = m.group(1).strip(), m.group(2)
        matched_name = resolve_character_name(name)
        if matched_name:
            out_lines.append(f"{matched_name}: {spoken}")
        else:
            missing.add(name)
            out_lines.append(f"{NARRATOR_NAME}: {spoken}")
    return "\n".join(out_lines), sorted(missing)


def main() -> int:
    parser = argparse.ArgumentParser(description="Poimi yksi luku Word-käsikirjoituksesta")
    parser.add_argument("--input", required=True, help="Syöte .docx")
    parser.add_argument("--content", required=True, help="Muotoa ACT.LUKU, esim. 1.3")
    parser.add_argument("--output", default=None, help="Tulostiedosto (txt)")
    parser.add_argument("--narrators-file", default="prompt_narrators.txt")
    args = parser.parse_args()

    try:
        narrators = load_narrators(Path(args.narrators_file))
        section, act_no, chapter_no, chapter_title = extract_section(Path(args.input), args.content)
    except (ValueError, LookupError, FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Virhe: {e}", file=sys.stderr)
        return 2

    normalized, missing = normalize_unknown_characters(section, narrators)
    if missing:
        print(
            f"Virhe: seuraavat hahmot puuttuvat tiedostosta {args.narrators_file}: {', '.join(missing)}. "
            f"Ne defaultattiin hahmoksi '{NARRATOR_NAME}'.",
            file=sys.stderr,
        )

    safe_title = re.sub(r"[^A-Za-z0-9ÅÄÖåäö_-]+", "_", chapter_title).strip("_") or "luku"
    default_name = f"{act_no:02d}_{chapter_no:02d}_{safe_title}.txt"
    output = Path(args.output) if args.output else Path(default_name)
    output.write_text(normalized + "\n", encoding="utf-8")
    print(f"Kirjoitettu: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
