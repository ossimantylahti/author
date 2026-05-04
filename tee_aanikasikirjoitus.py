#!/usr/bin/env python3
"""Extract one Act.Chapter section from a .docx manuscript for ElevenLabs script generation."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from docx import Document


def style_level(style_name: str) -> int | None:
    name = style_name.strip().lower()
    m = re.search(r"(\d+)$", name)
    if m:
        return int(m.group(1))
    if "otsikko 1" in name:
        return 1
    if "otsikko 2" in name:
        return 2
    if "heading 1" in name:
        return 1
    if "heading 2" in name:
        return 2
    return None


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
            collecting = (act == target_act and chapter == target_chapter)
            continue

        if collecting and text:
            lines.append(text)

    if not lines:
        raise LookupError(f"Sisältöä ei löytynyt kohdalle {content_id} tiedostosta {docx_path}")

    return "\n\n".join(lines).strip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Poimi yksi luku Word-käsikirjoituksesta")
    parser.add_argument("--input", required=True, help="Syöte .docx")
    parser.add_argument("--content", required=True, help="Muotoa ACT.LUKU, esim. 1.3")
    parser.add_argument("--output", default=None, help="Tulostiedosto (txt)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Virhe: tiedostoa ei löydy: {input_path}", file=sys.stderr)
        return 2

    try:
        section = extract_section(input_path, args.content)
    except (ValueError, LookupError) as e:
        print(f"Virhe: {e}", file=sys.stderr)
        return 2

    output = Path(args.output) if args.output else Path(f"aanikasikirjoitus_{args.content}.txt")
    output.write_text(section, encoding="utf-8")
    print(f"Kirjoitettu: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
