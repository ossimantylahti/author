#!/usr/bin/env python3
"""Extract one Act.Chapter section from a .docx manuscript for ElevenLabs script generation."""

from __future__ import annotations

import argparse
import json
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
    if NARRATOR_NAME not in data:
        raise ValueError(f"Narrators-tiedostossa pitää olla '{NARRATOR_NAME}'")
    return {str(k): str(v) for k, v in data.items()}


def extract_section(docx_path: Path, content_id: str) -> str:
    if not re.fullmatch(r"\d+\.\d+", content_id):
        raise ValueError("--content pitää olla muodossa ACT.LUKU, esim. 1.3")

    target_act, target_chapter = map(int, content_id.split("."))
    document = Document(str(docx_path))

    act = 0
    chapter = 0
    collecting = False
    lines: list[str] = []

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
            continue
        if collecting and text:
            lines.append(text)

    if not lines:
        raise LookupError(f"Sisältöä ei löytynyt kohdalle {content_id} tiedostosta {docx_path}")

    return "\n\n".join(lines).strip() + "\n"


def normalize_unknown_characters(text: str, narrators: dict[str, str]) -> tuple[str, list[str]]:
    missing: set[str] = set()
    out_lines: list[str] = []
    for line in text.splitlines():
        m = re.match(r"^([A-ZÅÄÖa-zåäö][\wÅÄÖåäö\- ]{0,40}):\s*(.*)$", line)
        if not m:
            out_lines.append(line)
            continue
        name, spoken = m.group(1).strip(), m.group(2)
        if name in narrators:
            out_lines.append(line)
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
        section = extract_section(Path(args.input), args.content)
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

    output = Path(args.output) if args.output else Path(f"aanikasikirjoitus_{args.content}.txt")
    output.write_text(normalized + "\n", encoding="utf-8")
    print(f"Kirjoitettu: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
