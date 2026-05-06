#!/usr/bin/env python3
"""Extract one Act.Chapter section from a .docx manuscript for ElevenLabs script generation."""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from docx import Document
from openai import OpenAI

NARRATOR_NAME = "Kertoja"
LANGUAGE_ALIASES = {
    "fi": "finnish",
    "finnish": "finnish",
    "suomi": "finnish",
    "en": "english",
    "eng": "english",
    "english": "english",
    "es": "spanish",
    "spa": "spanish",
    "spanish": "spanish",
    "espanol": "spanish",
}
MAX_FRAGMENT_LEN = 4000
INTER_CHUNK_BREAK = "0.4s"
INTER_LINE_BREAK = "0.8s"
EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF\U00002700-\U000027BF\U000024C2-\U0001F251]+", flags=re.UNICODE)

SPEECH_VERB_ROOTS = {
    "finnish": ["sano", "kysy", "vasta", "huuda", "totea", "juttele", "mainitse", "kommentoi", "kuiska", "karju", "mumise", "lausu", "selitä", "ilmoita", "myönnä", "kiistä", "toista", "mutise", "hihku", "naurahda"],
    "english": ["say", "ask", "answer", "reply", "shout", "whisper", "murmur", "remark", "state", "announce", "admit", "deny", "yell", "scream", "observe", "note", "repeat", "explain", "utter", "tell"],
    "spanish": ["dec", "pregunt", "respond", "contest", "grit", "susurr", "murmur", "afirm", "neg", "coment", "explic", "repet", "pronunci", "admit", "anunci", "observ", "añad", "indic", "declar", "dij"],
}
SPEECH_VERB_SUFFIXES = {
    "finnish": ["a", "an", "aa", "aan", "oi", "oin", "oivat", "ot", "omme", "ossa", "osta", "osti", "oivat", "oisi", "oisi", "oimme", "oivat", "n", "t", "mme", "tte", "vat", "in", "it", "imme", "itte", "ivat", "isin", "isit", "isi", "isimme", "isitte", "isivat", "nut", "nyt", "neet", "neet", "massa", "masta", "malla", "malle", "melta"],
    "english": ["", "s", "ed", "ing", "er", "ers", "ly", "able", "ably", "ation", "ations", "ment", "ments", "t", "ts", "d", "en", "es"],
    "spanish": ["ar", "o", "as", "a", "amos", "áis", "an", "é", "aste", "ó", "aron", "aba", "abas", "ábamos", "aban", "aré", "arás", "ará", "arán", "aría", "arías", "arían", "ado", "ada", "ados", "adas", "ando", "en", "emos", "éis"],
}

def _build_speech_verbs() -> set[str]:
    forms: set[str] = set()
    for lang, roots in SPEECH_VERB_ROOTS.items():
        for root in roots:
            for suffix in SPEECH_VERB_SUFFIXES[lang]:
                token = (root + suffix).lower().strip()
                if 3 <= len(token) <= 24:
                    forms.add(token)
    return forms

SPEECH_VERBS = _build_speech_verbs()

@dataclass
class NarratorConfig:
    voices: dict[str, str]
    aliases: dict[str, str]


def normalise_alias_key(value: str) -> str:
    value = value.strip().lower().replace("0", "o")
    return re.sub(r"[^a-zåäöáéíóúüñ0-9]+", "", value, flags=re.IGNORECASE)


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


def load_narrator_config(path: Path) -> NarratorConfig:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Narrators-tiedosto ei ole validia JSONia: {e}") from e
    if not isinstance(data, dict):
        raise ValueError("Narrators-tiedoston pitää olla JSON-objekti")
    voices_raw = data.get("voices")
    if not isinstance(voices_raw, dict):
        raise ValueError("Narrators-tiedoston 'voices' pitää olla objekti")

    voices: dict[str, str] = {}
    for name, meta in voices_raw.items():
        if not isinstance(meta, dict):
            raise ValueError(f"Voice-entry '{name}' on virheellinen: pitää olla objekti")
        ids = meta.get("ids")
        if not isinstance(ids, dict):
            raise ValueError(f"Voice-entry '{name}' on virheellinen: 'ids' puuttuu tai ei ole objekti")
        eleven = ids.get("elevenlabs")
        if not eleven:
            raise ValueError(f"Voice-entry '{name}' on virheellinen: ids.elevenlabs puuttuu")
        voices[str(name)] = str(eleven)

    if NARRATOR_NAME not in voices:
        raise ValueError(f"Narrators-tiedostossa pitää olla '{NARRATOR_NAME}'")

    alias_map = build_alias_map(voices)
    aliases = data.get("aliases", {})
    if aliases is not None:
        if not isinstance(aliases, dict):
            raise ValueError("Narrators-tiedoston 'aliases' pitää olla objekti")
        for canonical, values in aliases.items():
            canonical_name = str(canonical)
            if canonical_name not in voices:
                print(f"Warning: alias canonical voice missing, skipped: {canonical_name}", file=sys.stderr)
                continue
            if not isinstance(values, list):
                print(f"Warning: alias list invalid, skipped: {canonical_name}", file=sys.stderr)
                continue
            for alias in values:
                if isinstance(alias, str) and alias.strip():
                    alias_map[normalise_alias_key(alias)] = canonical_name
    return NarratorConfig(voices=voices, aliases=alias_map)


def build_alias_map(narrators: dict[str, str]) -> dict[str, str]:
    alias_map: dict[str, str] = {}
    ambiguous: set[str] = set()
    for canonical_name in narrators:
        if canonical_name.startswith("chat"):
            continue
        canonical_key = normalise_alias_key(canonical_name)
        alias_map[canonical_key] = canonical_name
        parts = [p for p in re.split(r"[\s_\-]+", canonical_name.strip()) if p]
        for part in parts:
            part_key = normalise_alias_key(part)
            if len(part_key) >= 3:
                if part_key not in alias_map:
                    alias_map[part_key] = canonical_name
                elif alias_map[part_key] != canonical_name:
                    ambiguous.add(part_key)
        if len(parts) >= 2:
            last_key = normalise_alias_key(parts[-1])
            if last_key not in alias_map:
                alias_map[last_key] = canonical_name
            elif alias_map[last_key] != canonical_name:
                ambiguous.add(last_key)
    for key in ambiguous:
        alias_map.pop(key, None)
    alias_map[normalise_alias_key(NARRATOR_NAME)] = NARRATOR_NAME
    return alias_map


def resolve_voice_name(candidate: str, narrators: dict[str, str], alias_map: dict[str, str]) -> str:
    raw = (candidate or "").strip()
    if not raw:
        return NARRATOR_NAME
    if raw in narrators:
        return raw
    if len(raw) > 48 or re.search(r"[,.;!?]", raw):
        return NARRATOR_NAME
    if len(re.findall(r"\w+", raw)) > 5:
        return NARRATOR_NAME
    return alias_map.get(normalise_alias_key(raw), NARRATOR_NAME)

# rest mostly unchanged, shortened for brevity in this patch context
# (kept deterministic parser and chat handling)

def extract_section(docx_path: Path, content_id: str | None) -> tuple[str, int, int, str]:
    document = Document(str(docx_path))
    if not content_id:
        lines = [p.text.strip() for p in document.paragraphs if p.text.strip()]
        return "\n\n".join(lines).strip() + "\n", 1, 1, docx_path.stem
    if not re.fullmatch(r"\d+\.\d+", content_id):
        raise ValueError("--content pitää olla muodossa ACT.LUKU, esim. 1.3")
    target_act, target_chapter = map(int, content_id.split("."))
    act = chapter = 0
    collecting = False
    lines: list[str] = []
    chapter_title = "luku"
    for para in document.paragraphs:
        text = para.text.strip()
        level = style_level(getattr(para.style, "name", ""))
        if level == 1:
            act += 1; chapter = 0
            if collecting: break
            continue
        if level == 2:
            chapter += 1
            if collecting: break
            collecting = act == target_act and chapter == target_chapter
            if collecting: chapter_title = text or f"Luku {chapter}"
            continue
        if collecting and text:
            lines.append(text)
    if not lines:
        raise LookupError(f"Sisältöä ei löytynyt kohdalle {content_id} tiedostosta {docx_path}")
    return "\n\n".join(lines).strip() + "\n", target_act, target_chapter, chapter_title

# Keep existing helper functions from original file.

def detect_language_name(text: str) -> str:
    lowered = text.lower(); tokens = re.findall(r"[a-zåäö]+", lowered)
    if not tokens: return "finnish"
    en = {"the", "and", "with", "viewers", "counter", "strike", "reaction", "chat"}
    es = {"la", "el", "de", "que", "y", "banda"}
    fi = {"on", "ja", "että", "oli", "mutta", "tunnettiin", "nimimerkillä"}
    en_score = sum(1 for t in tokens if t in en); es_score = sum(1 for t in tokens if t in es); fi_score = sum(1 for t in tokens if t in fi)
    fi_score += sum(1 for ch in lowered if ch in "åäö") * 0.6
    if math.isclose(en_score, es_score) and en_score > fi_score and " la " in f" {lowered} ": es_score += 0.2
    if fi_score >= en_score and fi_score >= es_score: return "finnish"
    return "english" if en_score >= es_score else "spanish"

def infer_chat_emotion(line: str, emojis: list[str]) -> str:
    combo = "".join(emojis); lowered = line.lower()
    if any(x in combo for x in ["😂", "😏", "🙃", "😼"]) or "lol" in lowered: return "mocking"
    if any(x in combo for x in ["😨", "😱", "😰", "😬"]) or "apua" in lowered: return "fearful"
    if any(x in combo for x in ["😡", "🤬", "👿"]): return "angry"
    if any(x in combo for x in ["😍", "❤️", "🥰"]): return "excited"
    return "neutral"

def infer_pace_and_tone(emotion: str, is_chat: bool) -> tuple[str, str]:
    emotion_key = emotion.strip().lower(); pace,tone="medium","neutral"
    if emotion_key in {"angry","viha","vihainen"}: pace,tone="fast","intense"
    elif emotion_key in {"fearful","pelokas"}: pace,tone="medium","tense"
    elif emotion_key in {"excited","innostunut","riemukas"}: pace,tone="fast","bright"
    elif emotion_key in {"mocking","ironinen"}: pace,tone="medium","playful"
    elif emotion_key in {"surullinen","sad"}: pace,tone="slow","soft"
    if is_chat and pace=="medium" and tone=="neutral": pace,tone="fast","light"
    return pace,tone

def split_plain_text(text: str, limit: int = MAX_FRAGMENT_LEN) -> list[str]:
    text = text.strip()
    if len(text) <= limit: return [text] if text else []
    parts=[]; rem=text
    while rem:
        if len(rem)<=limit: parts.append(rem.strip()); break
        cut=rem.rfind(" ",0,limit); cut=cut if cut>0 else limit
        parts.append(rem[:cut].strip()); rem=rem[cut:].strip()
    return [p for p in parts if p]

def parse_script_lines(text: str, narrators: dict[str, str], alias_map: dict[str, str]) -> list[dict[str, str]]:
    parsed=[]
    state={"current_scene_subject":NARRATOR_NAME,"last_explicit_speaker":NARRATOR_NAME,"last_dialogue_speaker":NARRATOR_NAME,"pending_speaker_from_lead_in":NARRATOR_NAME}
    speech_pat = "|".join(sorted(SPEECH_VERBS, key=len, reverse=True))
    for raw in text.splitlines():
        line=raw.strip()
        if not line: continue
        speaker=NARRATOR_NAME; spoken=line
        m = re.match(r"^([^:]{1,64}):\s*(.*)$", line)
        if m:
            cand=m.group(1).strip(); resolved=resolve_voice_name(cand,narrators,alias_map)
            if resolved!=NARRATOR_NAME: speaker,spoken=resolved,m.group(2).strip()
        elif line.startswith("–") or line.startswith("“"):
            spoken=line.lstrip("–“").strip(); speaker=state["pending_speaker_from_lead_in"]
            end = re.search(rf",\s*([A-ZÅÄÖa-zåäö][^,.:;!?]{{0,40}})\s+({speech_pat})\b", spoken, re.IGNORECASE)
            if end: speaker = resolve_voice_name(end.group(1), narrators, alias_map)
            elif re.search(rf"\bhän\s+({speech_pat})\b", spoken, re.IGNORECASE) and state["last_explicit_speaker"] != NARRATOR_NAME:
                speaker = state["last_explicit_speaker"]
            if speaker == NARRATOR_NAME:
                speaker = state["last_dialogue_speaker"] if state["last_dialogue_speaker"] != NARRATOR_NAME else NARRATOR_NAME
        else:
            lead = re.search(rf"^([A-ZÅÄÖa-zåäö][^,:]{{0,48}}).{{0,120}}\b({speech_pat})\b.*:\s*$", line, re.IGNORECASE)
            if lead:
                state["pending_speaker_from_lead_in"] = resolve_voice_name(lead.group(1), narrators, alias_map)
            else:
                state["pending_speaker_from_lead_in"] = NARRATOR_NAME
        if speaker != NARRATOR_NAME:
            state["last_explicit_speaker"] = speaker; state["last_dialogue_speaker"] = speaker; state["current_scene_subject"] = speaker
        parsed.append({"speaker": speaker, "text": spoken})
    return parsed

def to_ssml_lines(text: str, narrators: dict[str, str], alias_map: dict[str, str]) -> list[str]:
    parsed = parse_script_lines(text, narrators, alias_map)
    out=[]
    for i,row in enumerate(parsed):
        speaker=resolve_voice_name(row["speaker"], narrators, alias_map)
        emojis=EMOJI_RE.findall(row["text"]); spoken=EMOJI_RE.sub("",row["text"]).strip()
        is_chat=speaker.startswith("chat")
        emotion=infer_chat_emotion(row["text"],emojis) if is_chat else "neutraali"
        pace,tone=infer_pace_and_tone(emotion,is_chat)
        for chunk in split_plain_text(spoken):
            out.append(f'<voice name="{html.escape(speaker)}" language="{detect_language_name(chunk)}" emotion="{emotion}" pace="{pace}" tone="{tone}">{html.escape(chunk)}</voice>')
        if i < len(parsed)-1: out.append(f'<break time="{INTER_LINE_BREAK}"/>')
    return out

def final_voice_gate(ssml: str, narrators: dict[str, str]) -> str:
    def repl(m: re.Match[str]) -> str:
        val = m.group(1)
        if val in narrators: return m.group(0)
        print(f"Warning: invalid voice name replaced with Kertoja: {val}", file=sys.stderr)
        return m.group(0).replace(val, NARRATOR_NAME)
    return re.sub(r'voice name="([^"]*)"', repl, ssml)


def normalise_content_for_filename(content: str | None) -> str:
    if not content:
        return "output"
    value = content.strip()
    if re.fullmatch(r"\d+\.\d+", value):
        major, minor = value.split(".", maxsplit=1)
        return f"{int(major):02d}_{int(minor):02d}"
    cleaned = re.sub(r"[^0-9A-Za-z_-]+", "_", value).strip("_")
    return cleaned or "output"


def resolve_output_path(output_arg: str | None, content: str | None, input_path: Path) -> Path:
    name_root = normalise_content_for_filename(content)
    auto_file_name = f"{name_root}_audiobook_ssml.xml"
    if not output_arg:
        return Path(auto_file_name)

    output_path = Path(output_arg)
    output_looks_like_dir = output_arg.endswith(("/", "\\"))
    if output_path.exists() and output_path.is_dir():
        output_looks_like_dir = True
    if not output_path.exists() and output_path.suffix == "":
        output_looks_like_dir = True

    if output_looks_like_dir:
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path / auto_file_name

    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path

def main() -> int:
    parser = argparse.ArgumentParser(description="Poimi yksi luku Word-käsikirjoituksesta")
    parser.add_argument("--input", required=True); parser.add_argument("--content", required=False)
    parser.add_argument("--output", default=None); parser.add_argument("--narrators-file", default="prompt_narrators.txt")
    args = parser.parse_args()
    try:
        cfg=load_narrator_config(Path(args.narrators_file))
        section,act_no,chapter_no,title=extract_section(Path(args.input), args.content)
    except Exception as e:
        print(f"Virhe: {e}", file=sys.stderr); return 2
    ssml = "<speak>\n" + "\n".join(to_ssml_lines(section, cfg.voices, cfg.aliases)) + "\n</speak>\n"
    ssml = final_voice_gate(ssml, cfg.voices)
    out = resolve_output_path(args.output, args.content, Path(args.input))
    print(f"Writing SSML to: {out}")
    out.write_text(ssml, encoding="utf-8")
    print(f"Kirjoitettu: {out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
