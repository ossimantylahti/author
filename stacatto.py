#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute per-logical-chapter staccato metrics from a flattened mts.txt file.

This version adds:
- Per-chapter delta breakdown counts (what *caused* the intensity).
- Act-level z-score anomalies.
- Jump anomalies (sudden changes between consecutive chapters).
- Plateau detection (too-flat runs that a human reader often misses).
- Writes two hard-coded outputs:
    - stacatto-analysis.csv
    - stacatto-anomaly.csv

Key assumptions:
- Heading 1 = act/section context (Näytös I, II, ...), not a chapter.
- Heading 2+ = real chapters.
- Input line format:
  [chapter N][pM][WordStyle][ParaType][s=K][stacc=X][delta=Y]Text...

Delta buckets (from your flatten/staccato logic):
+1.0  -> staccato prose "hits"
 0.0  -> neutral
-0.2  -> long-ish prose line (typical)
-0.3  -> chat-like line
-0.5  -> dialogue-like line
Anything else -> other bucket
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


LINE_RE = re.compile(
    r'^\[chapter (?P<chapter>\d+)\]'
    r'\[p(?P<p>\d+)\]'
    r'\[(?P<style>[^\]]*)\]'
    r'\[(?P<ptype>[^\]]*)\]'
    r'\[s=(?P<s>\d+)\]'
    r'\[stacc=(?P<stacc>\d+)\]'
    r'\[delta=(?P<delta>-?\d+(?:\.\d+)?)\]'
    r'(?P<text>.*)$'
)

HEADING_RE = re.compile(r'^Heading\s+(?P<level>\d+)$', re.IGNORECASE)

# Hard-coded output files (requested).
ANALYSIS_OUT = Path("stacatto-analysis.csv")
ANOMALY_OUT = Path("stacatto-anomaly.csv")

# Anomaly thresholds (tune later if needed).
Z_THRESHOLD = 1.5
JUMP_THRESHOLD = 0.35

# Plateau definition: at least this many consecutive chapters in an act,
# whose intensity range (max-min) stays below this.
PLATEAU_MIN_LEN = 6
PLATEAU_RANGE_MAX = 0.08

# Float bucketing tolerance
EPS = 1e-9


@dataclass(frozen=True)
class ParsedLine:
    chapter: int  # kept for completeness, but not used for grouping
    p: int
    style: str
    ptype: str
    s: int
    stacc: int
    delta: float
    text: str


def parse_lines(lines: Iterable[str]) -> Iterable[ParsedLine]:
    for raw in lines:
        raw = raw.rstrip("\n")
        if not raw.strip():
            continue
        m = LINE_RE.match(raw)
        if not m:
            raise ValueError(f"Unparsable line (unexpected format): {raw[:200]}")
        yield ParsedLine(
            chapter=int(m.group("chapter")),
            p=int(m.group("p")),
            style=m.group("style"),
            ptype=m.group("ptype"),
            s=int(m.group("s")),
            stacc=int(m.group("stacc")),
            delta=float(m.group("delta")),
            text=m.group("text").strip(),
        )


def heading_level(style: str) -> Optional[int]:
    m = HEADING_RE.match(style.strip())
    if not m:
        return None
    return int(m.group("level"))


def is_content_line(ln: ParsedLine, mode: str) -> bool:
    """
    content-mode:
      - strict: exclude headings; require s>0
      - include_s0: exclude headings; allow s>=0
      - all_non_heading: exclude headings; include everything else regardless of s
    """
    lvl = heading_level(ln.style)
    if lvl is not None:
        return False

    if ln.ptype == "Otsikko":
        return False

    if mode == "strict":
        return ln.s > 0
    if mode == "include_s0":
        return ln.s >= 0
    if mode == "all_non_heading":
        return True

    raise ValueError(f"Unknown content mode: {mode}")


def safe_div(n: float, d: float) -> float:
    return 0.0 if d == 0 else (n / d)


def approx_equal(a: float, b: float) -> bool:
    return abs(a - b) <= EPS


def delta_bucket(delta: float) -> str:
    """
    Classify deltas into the buckets we care about.
    """
    if approx_equal(delta, 1.0):
        return "pos"
    if approx_equal(delta, 0.0):
        return "zero"
    if approx_equal(delta, -0.5):
        return "neg_dialogue"
    if approx_equal(delta, -0.3):
        return "neg_chat"
    if approx_equal(delta, -0.2):
        return "neg_long"
    return "other"


def compute_metrics(
    ch_lines: List[ParsedLine],
    content_mode: str,
    delta_mode: str
) -> Dict[str, float | int | str]:
    content = [ln for ln in ch_lines if is_content_line(ln, content_mode)]

    content_lines = len(content)
    stacc_lines = sum(1 for ln in content if ln.stacc == 1)
    stacc_ratio = safe_div(stacc_lines, content_lines)

    deltas = [ln.delta for ln in content]
    sum_delta = float(sum(deltas)) if deltas else 0.0

    if delta_mode == "mean":
        staccato_intensity = safe_div(sum_delta, len(deltas)) if deltas else 0.0
    elif delta_mode == "sum":
        staccato_intensity = sum_delta
    else:
        raise ValueError(f"Unknown delta mode: {delta_mode}")

    s_vals = [ln.s for ln in content]
    mean_sentences_per_line = safe_div(float(sum(s_vals)), len(s_vals)) if s_vals else 0.0

    # Delta breakdown
    bucket_counts = {
        "pos": 0,
        "zero": 0,
        "neg_dialogue": 0,
        "neg_chat": 0,
        "neg_long": 0,
        "other": 0,
    }
    for d in deltas:
        bucket_counts[delta_bucket(d)] += 1

    # Dominant bucket (ties resolved by a simple priority).
    priority = ["pos", "neg_dialogue", "neg_chat", "neg_long", "zero", "other"]
    dominant = max(priority, key=lambda k: (bucket_counts[k], -priority.index(k))) if deltas else "none"

    return {
        "staccato_intensity": round(float(staccato_intensity), 6),
        "content_lines": int(content_lines),
        "stacc_lines": int(stacc_lines),
        "stacc_ratio": round(float(stacc_ratio), 6),
        "sum_delta": round(float(sum_delta), 6),
        "mean_sentences_per_line": round(float(mean_sentences_per_line), 6),
        "delta_pos_lines": int(bucket_counts["pos"]),
        "delta_neg_dialogue_lines": int(bucket_counts["neg_dialogue"]),
        "delta_neg_chat_lines": int(bucket_counts["neg_chat"]),
        "delta_neg_long_lines": int(bucket_counts["neg_long"]),
        "delta_zero_lines": int(bucket_counts["zero"]),
        "delta_other_lines": int(bucket_counts["other"]),
        "dominant_delta": str(dominant),
    }


def split_into_logical_chapters(all_lines: List[ParsedLine]) -> List[Dict]:
    """
    Build logical chapters:
    - Heading 1 updates act_title
    - Heading 2+ starts a new chapter with title = line.text
    """
    act_title: str = ""
    chapters: List[Dict] = []

    current_title: Optional[str] = None
    current_act: str = ""
    current_lines: List[ParsedLine] = []

    def flush() -> None:
        nonlocal current_title, current_act, current_lines
        if current_title is None:
            return
        chapters.append(
            {
                "act_title": current_act,
                "title": current_title,
                "lines": current_lines,
            }
        )
        current_title = None
        current_lines = []

    for ln in sorted(all_lines, key=lambda x: x.p):
        lvl = heading_level(ln.style)

        if lvl == 1:
            if ln.text:
                act_title = ln.text
            continue

        if lvl is not None and lvl >= 2:
            flush()
            current_title = ln.text if ln.text else "(untitled)"
            current_act = act_title
            current_lines = []
            continue

        if current_title is not None:
            current_lines.append(ln)

    flush()
    return chapters


def build_report(parsed: Iterable[ParsedLine], content_mode: str, delta_mode: str) -> List[Dict]:
    all_lines = list(parsed)
    logical = split_into_logical_chapters(all_lines)

    rows: List[Dict] = []
    for i, ch in enumerate(logical, start=1):
        metrics = compute_metrics(ch["lines"], content_mode=content_mode, delta_mode=delta_mode)
        rows.append(
            {
                "chapter_id": i,
                "act_title": ch["act_title"],
                "title": ch["title"],
                **metrics,
            }
        )
    return rows


def mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    m = sum(values) / len(values)
    if len(values) < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in values) / (len(values) - 1)
    return m, math.sqrt(var)


def enrich_with_act_stats(rows: List[Dict]) -> None:
    """
    Add act mean/std and z-score per chapter.
    Mutates rows in place.
    """
    by_act: Dict[str, List[int]] = {}
    for idx, r in enumerate(rows):
        by_act.setdefault(r["act_title"] or "", []).append(idx)

    act_stats: Dict[str, Tuple[float, float]] = {}
    for act, idxs in by_act.items():
        vals = [float(rows[i]["staccato_intensity"]) for i in idxs]
        m, s = mean_std(vals)
        act_stats[act] = (m, s)

    for r in rows:
        act = r["act_title"] or ""
        m, s = act_stats.get(act, (0.0, 0.0))
        x = float(r["staccato_intensity"])
        z = 0.0 if s == 0.0 else (x - m) / s
        r["act_mean_intensity"] = round(m, 6)
        r["act_std_intensity"] = round(s, 6)
        r["z_intensity"] = round(z, 6)


def enrich_with_jumps(rows: List[Dict]) -> None:
    """
    Add previous intensity and jump value.
    Jump is computed within the same act; act boundary resets prev.
    """
    prev_by_act: Dict[str, float] = {}

    for r in rows:
        act = r["act_title"] or ""
        x = float(r["staccato_intensity"])
        if act in prev_by_act:
            prev = prev_by_act[act]
            jump = x - prev
            r["prev_intensity"] = round(prev, 6)
            r["jump_intensity"] = round(jump, 6)
        else:
            r["prev_intensity"] = ""
            r["jump_intensity"] = ""
        prev_by_act[act] = x


def detect_plateaus(rows: List[Dict]) -> Dict[int, int]:
    """
    Detect plateau runs within each act.
    Returns a mapping: chapter_id -> plateau_id (int), where plateau_id starts from 1.
    """
    plateau_map: Dict[int, int] = {}
    plateau_id = 0

    # group by act preserving order
    by_act: Dict[str, List[Dict]] = {}
    for r in rows:
        by_act.setdefault(r["act_title"] or "", []).append(r)

    for act, act_rows in by_act.items():
        intens = [float(r["staccato_intensity"]) for r in act_rows]

        n = len(intens)
        if n < PLATEAU_MIN_LEN:
            continue

        start = 0
        while start < n:
            end = start + PLATEAU_MIN_LEN
            if end > n:
                break

            # Try to extend a minimal window into the longest plateau.
            window = intens[start:end]
            if (max(window) - min(window)) <= PLATEAU_RANGE_MAX:
                # extend
                best_end = end
                best_max = max(window)
                best_min = min(window)

                while best_end < n:
                    new_max = max(best_max, intens[best_end])
                    new_min = min(best_min, intens[best_end])
                    if (new_max - new_min) <= PLATEAU_RANGE_MAX:
                        best_max, best_min = new_max, new_min
                        best_end += 1
                    else:
                        break

                plateau_id += 1
                for i in range(start, best_end):
                    cid = int(act_rows[i]["chapter_id"])
                    plateau_map[cid] = plateau_id

                start = best_end  # skip ahead past this plateau
            else:
                start += 1

    return plateau_map


def build_anomaly_report(rows: List[Dict]) -> List[Dict]:
    """
    Build a compact anomaly report with flags and reasons.
    """
    plateau_map = detect_plateaus(rows)

    anomalies: List[Dict] = []
    for r in rows:
        reasons: List[str] = []

        cid = int(r["chapter_id"])
        intensity = float(r["staccato_intensity"])
        z = float(r.get("z_intensity", 0.0))

        # z-score anomaly
        if abs(z) >= Z_THRESHOLD:
            reasons.append(f"z>={Z_THRESHOLD}")

        # jump anomaly (only if jump exists)
        jump_raw = r.get("jump_intensity", "")
        if jump_raw != "":
            jump = float(jump_raw)
            if abs(jump) >= JUMP_THRESHOLD:
                reasons.append(f"jump>={JUMP_THRESHOLD}")

        # plateau
        if cid in plateau_map:
            reasons.append(f"plateau#{plateau_map[cid]}")

        # data / edge-case flags
        if int(r["content_lines"]) == 0:
            reasons.append("content_lines=0")

        # "mismatch hints" purely from delta composition:
        # e.g. low intensity but many negative dialogue/chat lines, or high stacc_ratio but still low intensity.
        stacc_ratio = float(r["stacc_ratio"])
        if intensity < 0.15 and (int(r["delta_neg_dialogue_lines"]) + int(r["delta_neg_chat_lines"])) >= 10:
            reasons.append("low_intensity_driven_by_dialogue/chat")
        if stacc_ratio >= 0.55 and intensity < 0.20:
            reasons.append("high_stacc_ratio_but_low_intensity")

        if reasons:
            anomalies.append(
                {
                    "chapter_id": cid,
                    "act_title": r["act_title"],
                    "title": r["title"],
                    "staccato_intensity": r["staccato_intensity"],
                    "z_intensity": r.get("z_intensity", ""),
                    "jump_intensity": r.get("jump_intensity", ""),
                    "content_lines": r["content_lines"],
                    "stacc_ratio": r["stacc_ratio"],
                    "dominant_delta": r.get("dominant_delta", ""),
                    "delta_pos_lines": r.get("delta_pos_lines", 0),
                    "delta_neg_dialogue_lines": r.get("delta_neg_dialogue_lines", 0),
                    "delta_neg_chat_lines": r.get("delta_neg_chat_lines", 0),
                    "delta_neg_long_lines": r.get("delta_neg_long_lines", 0),
                    "delta_zero_lines": r.get("delta_zero_lines", 0),
                    "reasons": ";".join(reasons),
                }
            )

    # Sort: most "interesting" first (by abs(z), then abs(jump) if present)
    def sort_key(a: Dict) -> Tuple[float, float]:
        z = abs(float(a.get("z_intensity") or 0.0))
        j = abs(float(a.get("jump_intensity") or 0.0)) if a.get("jump_intensity") not in ("", None) else 0.0
        return (z, j)

    anomalies.sort(key=sort_key, reverse=True)
    return anomalies


def write_csv(rows: List[Dict], path: Path, fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Compute per-logical-chapter staccato metrics from mts.txt")
    ap.add_argument("input", type=Path, help="Path to mts.txt")
    ap.add_argument(
        "--content-mode",
        choices=["strict", "include_s0", "all_non_heading"],
        default="strict",
        help="Which lines are treated as content (default: strict)",
    )
    ap.add_argument(
        "--delta-mode",
        choices=["mean", "sum"],
        default="mean",
        help="How staccato_intensity is computed from delta values (default: mean)",
    )
    args = ap.parse_args(argv)

    text = args.input.read_text(encoding="utf-8")
    parsed = list(parse_lines(text.splitlines()))
    rows = build_report(parsed, content_mode=args.content_mode, delta_mode=args.delta_mode)

    # Enrich analysis rows with act statistics and jumps.
    enrich_with_act_stats(rows)
    enrich_with_jumps(rows)

    # Write analysis CSV (hard-coded).
    analysis_fields = [
        "chapter_id",
        "act_title",
        "title",
        "staccato_intensity",
        "content_lines",
        "stacc_lines",
        "stacc_ratio",
        "sum_delta",
        "mean_sentences_per_line",
        "delta_pos_lines",
        "delta_neg_dialogue_lines",
        "delta_neg_chat_lines",
        "delta_neg_long_lines",
        "delta_zero_lines",
        "delta_other_lines",
        "dominant_delta",
        "act_mean_intensity",
        "act_std_intensity",
        "z_intensity",
        "prev_intensity",
        "jump_intensity",
    ]
    write_csv(rows, ANALYSIS_OUT, analysis_fields)

    # Build and write anomaly CSV (hard-coded).
    anomalies = build_anomaly_report(rows)
    anomaly_fields = [
        "chapter_id",
        "act_title",
        "title",
        "staccato_intensity",
        "z_intensity",
        "jump_intensity",
        "content_lines",
        "stacc_ratio",
        "dominant_delta",
        "delta_pos_lines",
        "delta_neg_dialogue_lines",
        "delta_neg_chat_lines",
        "delta_neg_long_lines",
        "delta_zero_lines",
        "reasons",
    ]
    write_csv(anomalies, ANOMALY_OUT, anomaly_fields)

    # Minimal console output (so piping remains clean if you ever want it).
    print(f"Wrote: {ANALYSIS_OUT} ({len(rows)} chapters)")
    print(f"Wrote: {ANOMALY_OUT} ({len(anomalies)} anomalies)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
