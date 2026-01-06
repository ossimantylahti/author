#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Flatten a DOCX manuscript into line-based records, preserving paragraph styles,
and add deterministic structural tags for downstream rhythm analysis.

Output format (one paragraph per line):
[chapter 0][p1][Heading 1][Otsikko][s=1][stacc=0][delta=0.0]Prologi
[chapter 0][p2][Normal][Teksti][s=1][stacc=1][delta=1.0]Vuosaari. Myöhäisilta. Nyt.
[chapter 0][p3][Leipäteksti][Teksti][s=3][stacc=0][delta=-0.2]Sumu nousee mereltä...
[chapter 0][p4][Leipäteksti][Repliikki][s=1][stacc=0][delta=-0.5]– Mä kerroin meistä.
[chapter 0][p5][Chat][Chat][s=0][stacc=0][delta=-0.3]KappaPride: omg
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Optional

from docx import Document


def normalise_whitespace(text: str, keep_newlines: bool = False) -> str:
    """Normalise whitespace within a paragraph without destroying meaning."""
    if keep_newlines:
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
        return text.strip()
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_chapter_boundary(
    style_name: str,
    chapter_style: Optional[str],
    chapter_style_prefix: Optional[str],
) -> bool:
    if chapter_style is not None:
        return style_name == chapter_style
    if chapter_style_prefix is not None:
        return style_name.startswith(chapter_style_prefix)
    return style_name == "Heading 1"


def count_sentences_heuristic(text: str) -> int:
    """
    Heuristic sentence count based on ., !, ? with ellipses collapsed.

    Extra rule for rhythm analysis:
    - Treat very short 'telegraphic' paragraphs like "Vuosaari. Myöhäisilta. Nyt."
      as 1 sentence (they behave like a single beat).
    """
    if not text:
        return 0

    t = re.sub(r"\.{3,}", ".", text).strip()

    # Telegraphic micro-paragraph rule.
    # If paragraph is short and made of multiple short dot-terminated fragments,
    # count as 1 sentence to better match perceived rhythm.
    if len(t) <= 60:
        fragments = [f.strip() for f in re.split(r"[.!?]+", t) if f.strip()]
        if len(fragments) >= 2 and all(len(f) <= 18 for f in fragments):
            return 1

    return len(re.findall(r"[.!?]", t))


def detect_paragraph_type(style_name: str, text: str) -> str:
    """
    Detect paragraph category.
    Types: Chat, Otsikko, Repliikki, Teksti
    """
    if style_name == "Chat":
        return "Chat"

    s_lower = (style_name or "").strip().lower()
    if ("heading" in s_lower) or ("header" in s_lower) or ("title" in s_lower):
        return "Otsikko"

    if text and text.lstrip().startswith("–"):
        return "Repliikki"

    return "Teksti"


def is_staccato(style_name: str, p_type: str, sentence_count: int) -> int:
    """
    Deterministic staccato classification.
    stacc=1 iff:
      - Type == Teksti
      - Style in {Normal, Leipäteksti}
      - sentence_count is 1 or 2
    """
    if p_type != "Teksti":
        return 0
    if style_name not in ("Normal", "Leipäteksti"):
        return 0
    if sentence_count in (1, 2):
        return 1
    return 0


def delta_for_paragraph(style_name: str, p_type: str, sentence_count: int, stacc: int) -> float:
    """
    Deterministic intensity delta model.

    Rules (as discussed):
    - Staccato-isku: +1.0 (stacc=1)
    - Dialogirepliikki: -0.5
    - Chat: -0.3
    - Long leipäteksti (>3 sentences): -0.2
    - Otsikot: 0.0
    - Otherwise: 0.0
    """
    if p_type == "Otsikko":
        return 0.0
    if p_type == "Chat":
        return -0.3
    if p_type == "Repliikki":
        return -0.5
    if stacc == 1:
        return 1.0
    if p_type == "Teksti" and style_name in ("Normal", "Leipäteksti") and sentence_count > 3:
        return -0.2
    return 0.0


def flatten_docx(
    input_path: Path,
    output_path: Optional[Path],
    include_empty: bool,
    keep_internal_newlines: bool,
    chapter_style: Optional[str],
    chapter_style_prefix: Optional[str],
) -> int:
    doc = Document(str(input_path))

    chapter_idx = 0
    paragraph_idx = 0
    lines_written = 0

    out = sys.stdout if output_path is None else output_path.open(
        "w", encoding="utf-8", newline="\n"
    )

    try:
        for p in doc.paragraphs:
            style_name = getattr(getattr(p, "style", None), "name", None) or "UNKNOWN"
            raw_text = p.text or ""
            text = normalise_whitespace(raw_text, keep_newlines=keep_internal_newlines)

            if not text and not include_empty:
                continue

            if text and is_chapter_boundary(style_name, chapter_style, chapter_style_prefix):
                if paragraph_idx > 0:
                    chapter_idx += 1

            paragraph_idx += 1

            p_type = detect_paragraph_type(style_name, text)
            sent_count = count_sentences_heuristic(text)
            stacc = is_staccato(style_name, p_type, sent_count)
            delta = delta_for_paragraph(style_name, p_type, sent_count, stacc)

            line = (
                f"[chapter {chapter_idx}]"
                f"[p{paragraph_idx}]"
                f"[{style_name}]"
                f"[{p_type}]"
                f"[s={sent_count}]"
                f"[stacc={stacc}]"
                f"[delta={delta:.1f}]"
                f"{text}"
            )
            out.write(line + "\n")
            lines_written += 1

    finally:
        if output_path is not None:
            out.close()

    return lines_written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Flatten a DOCX manuscript into enriched [chapter][p][Style][Type][s=][stacc=][delta=] lines.",
    )
    parser.add_argument("input", type=Path, help="Path to input .docx file")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Write output to file (default: stdout)")
    parser.add_argument("--include-empty", action="store_true", help="Include empty paragraphs")
    parser.add_argument("--keep-internal-newlines", action="store_true", help="Preserve internal newlines")
    parser.add_argument("--chapter-style", type=str, default=None, help='Exact style name for chapter boundary')
    parser.add_argument("--chapter-style-prefix", type=str, default=None, help="Style prefix for chapter boundary")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        print(f"ERROR: Input file does not exist: {args.input}", file=sys.stderr)
        raise SystemExit(2)

    if args.input.suffix.lower() != ".docx":
        print("ERROR: Input must be a .docx file.", file=sys.stderr)
        raise SystemExit(2)

    if args.chapter_style and args.chapter_style_prefix:
        print("ERROR: Use either --chapter-style or --chapter-style-prefix, not both.", file=sys.stderr)
        raise SystemExit(2)

    lines_written = flatten_docx(
        input_path=args.input,
        output_path=args.output,
        include_empty=args.include_empty,
        keep_internal_newlines=args.keep_internal_newlines,
        chapter_style=args.chapter_style,
        chapter_style_prefix=args.chapter_style_prefix,
    )

    print(f"[INFO] Done. Lines written: {lines_written}", file=sys.stderr)


if __name__ == "__main__":
    main()
