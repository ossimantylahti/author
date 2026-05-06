#!/usr/bin/env python3
"""Extract one Act.Chapter section from a .docx manuscript for ElevenLabs script generation."""

from __future__ import annotations

import argparse
import difflib
import html
import json
import math
import os
import random
import re
import sys
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
MAX_FRAGMENT_LEN = 9500
INTER_CHUNK_BREAK = "0.4s"
INTER_LINE_BREAK = "0.8s"
EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FAFF"
    "\U00002700-\U000027BF"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)


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


def detect_language_name(text: str) -> str:
    lowered = text.lower()
    tokens = re.findall(r"[a-zåäö]+", lowered)
    if not tokens:
        return "finnish"

    english_markers = {"the", "and", "with", "viewers", "counter", "strike", "reaction", "chat"}
    spanish_markers = {"la", "el", "de", "que", "y", "banda"}
    finnish_markers = {"on", "ja", "että", "oli", "mutta", "tunnettiin", "nimimerkillä"}

    en_score = sum(1 for t in tokens if t in english_markers)
    es_score = sum(1 for t in tokens if t in spanish_markers)
    fi_score = sum(1 for t in tokens if t in finnish_markers)

    diacritics = sum(1 for ch in lowered if ch in "åäö")
    fi_score += diacritics * 0.6

    if math.isclose(en_score, es_score) and en_score > fi_score and " la " in f" {lowered} ":
        es_score += 0.2

    if fi_score >= en_score and fi_score >= es_score:
        return "finnish"
    if en_score >= es_score:
        return "english"
    return "spanish"


def add_language_attribute(text: str, forced_language: str | None = None) -> str:
    forced = LANGUAGE_ALIASES.get(forced_language.lower(), forced_language.lower()) if forced_language else None

    def repl(match: re.Match[str]) -> str:
        attrs = match.group("attrs")
        body = match.group("body")
        if re.search(r"\blanguage\s*=", attrs):
            return match.group(0)
        language = forced or detect_language_name(body)
        return f'<voice{attrs} language="{language}">{body}</voice>'

    voice_pattern = re.compile(r"<voice(?P<attrs>[^>]*)>(?P<body>.*?)</voice>", re.DOTALL)
    return voice_pattern.sub(repl, text)


def infer_chat_emotion(line: str, emojis: list[str]) -> str:
    combo = "".join(emojis)
    lowered = line.lower()
    if any(x in combo for x in ["😂", "😏", "🙃", "😼"]) or "lol" in lowered:
        return "mocking"
    if any(x in combo for x in ["😨", "😱", "😰", "😬"]) or "apua" in lowered:
        return "fearful"
    if any(x in combo for x in ["😡", "🤬", "👿"]):
        return "angry"
    if any(x in combo for x in ["😍", "❤️", "🥰"]):
        return "excited"
    return "neutral"

def infer_pace_and_tone(emotion: str, is_chat: bool) -> tuple[str, str]:
    emotion_key = emotion.strip().lower()
    pace = "medium"
    tone = "neutral"

    if emotion_key in {"angry", "viha", "vihainen"}:
        pace, tone = "fast", "intense"
    elif emotion_key in {"fearful", "pelokas"}:
        pace, tone = "medium", "tense"
    elif emotion_key in {"excited", "innostunut", "riemukas"}:
        pace, tone = "fast", "bright"
    elif emotion_key in {"mocking", "ironinen"}:
        pace, tone = "medium", "playful"
    elif emotion_key in {"surullinen", "sad"}:
        pace, tone = "slow", "soft"
    elif emotion_key in {"neutraali", "neutral"}:
        pace, tone = ("medium", "neutral")

    if is_chat and pace == "medium" and tone == "neutral":
        pace, tone = "fast", "light"
    return pace, tone




def escape_xml_text(text: str) -> str:
    return html.escape(text, quote=False)


def split_plain_text(text: str, limit: int = MAX_FRAGMENT_LEN) -> list[str]:
    text = text.strip()
    if len(text) <= limit:
        return [text] if text else []
    parts: list[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= limit:
            parts.append(remaining.strip())
            break
        cut = remaining.rfind(" ", 0, limit)
        if cut <= 0:
            cut = limit
        parts.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    return [p for p in parts if p]


def annotate_with_openai(lines: list[dict[str, str]], model: str) -> list[dict[str, str]]:
    if not os.getenv("OPENAI_API_KEY"):
        return lines
    client = OpenAI()
    payload = [{"speaker": x["speaker"], "text": x["text"], "is_chat": x["is_chat"]} for x in lines]
    payload_json = json.dumps(payload, ensure_ascii=False)
    payload_bytes = len(payload_json.encode("utf-8"))
    print(f"OpenAI API payload-koko: {payload_bytes} tavua ({len(payload_json)} merkkiä), rivejä: {len(payload)}")
    prompt = (
        "Analyze each dialogue line and return a JSON array in the same order. "
        "Fields: language (e.g. finnish/english/spanish), emotion (1-2 words, English), cleaned_text "
        "(same text content without speaker names). No explanations."
    )
    rsp = client.responses.create(
        model=model,
        input=[{"role": "system", "content": prompt}, {"role": "user", "content": payload_json}],
    )
    content = rsp.output_text
    try:
        parsed = json.loads(content)
        if isinstance(parsed, list) and len(parsed) == len(lines):
            for row, meta in zip(lines, parsed):
                if isinstance(meta, dict):
                    row["language"] = str(meta.get("language") or row["language"])
                    row["emotion"] = str(meta.get("emotion") or row["emotion"])
                    row["text"] = str(meta.get("cleaned_text") or row["text"])
    except json.JSONDecodeError:
        return lines
    return lines


def to_ssml_lines(text: str, narrators: dict[str, str], pls_path: str | None, ai_model: str) -> list[str]:
    normalized, _ = normalize_unknown_characters(text, narrators)
    chat_pool = [k for k in narrators if re.fullmatch(r"chat\d*", k)]
    nick_voice: dict[str, str] = {}
    parsed: list[dict[str, str]] = []
    for raw in normalized.splitlines():
        if not raw.strip():
            continue
        m = re.match(r"^([^:]+):\s*(.*)$", raw)
        if m:
            speaker, spoken = m.group(1).strip(), m.group(2).strip()
        else:
            speaker, spoken = NARRATOR_NAME, raw.strip()
        original = spoken
        emojis = EMOJI_RE.findall(spoken)
        spoken = EMOJI_RE.sub("", spoken).strip()
        is_chat = speaker.startswith("chat")
        if is_chat:
            nick = speaker
            if nick in {"chat-kaira", "chat-mod"}:
                voice = nick
            elif nick in nick_voice:
                voice = nick_voice[nick]
            else:
                voice = random.choice(chat_pool) if chat_pool else "chat"
                nick_voice[nick] = voice
            speaker = voice
            emotion = infer_chat_emotion(original, emojis)
        else:
            emotion = "neutraali"
        parsed.append(
            {
                "speaker": speaker,
                "text": spoken,
                "language": detect_language_name(spoken),
                "emotion": emotion,
                "is_chat": "1" if is_chat else "0",
            }
        )
    parsed = annotate_with_openai(parsed, ai_model)
    out: list[str] = []
    if pls_path:
        out.append(f'<lexicon uri="{escape_xml_text(pls_path)}" />')
    for row_idx, row in enumerate(parsed):
        prefix = '<audio src="notification.mp3" /> ' if row["is_chat"] == "1" else ""
        pace, tone = infer_pace_and_tone(row["emotion"], row["is_chat"] == "1")
        chunks = split_plain_text(row["text"], MAX_FRAGMENT_LEN)
        for idx, chunk in enumerate(chunks):
            chunk_prefix = prefix if idx == 0 else ""
            out.append(
                f'<voice name="{escape_xml_text(row["speaker"])}" language="{escape_xml_text(row["language"])}" emotion="{escape_xml_text(row["emotion"])}" pace="{pace}" tone="{tone}">{chunk_prefix}{escape_xml_text(chunk)}</voice>'
            )
            if idx < len(chunks) - 1:
                out.append(f'<break time="{INTER_CHUNK_BREAK}"/>')
        if row_idx < len(parsed) - 1:
            out.append(f'<break time="{INTER_LINE_BREAK}"/>')
    
    return out


def resolve_output_path(output_arg: str | None, act_no: int, chapter_no: int, chapter_title: str) -> Path:
    safe_title = re.sub(r"[^A-Za-z0-9ÅÄÖåäö_-]+", "_", chapter_title).strip("_") or "luku"
    filename = f"{act_no:02d}_{chapter_no:02d}_{safe_title}.ssml"
    if not output_arg:
        return Path(filename)

    out = Path(output_arg).expanduser()
    if output_arg.endswith(("/", "\\")) or (out.exists() and out.is_dir()):
        out.mkdir(parents=True, exist_ok=True)
        return out / filename
    if out.suffix == "":
        out.mkdir(parents=True, exist_ok=True)
        return out / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Poimi yksi luku Word-käsikirjoituksesta")
    parser.add_argument("--input", required=True, help="Syöte .docx")
    parser.add_argument("--content", required=True, help="Muotoa ACT.LUKU, esim. 1.3")
    parser.add_argument("--output", default=None, help="Tulostiedosto (txt)")
    parser.add_argument("--narrators-file", default="prompt_narrators.txt")
    parser.add_argument("--pronunciation-dictionary", default=None, help="PLS pronunciation dictionary URI/polku")
    parser.add_argument("--openai-model", default="gpt-5-mini")
    parser.add_argument(
        "--voice-language",
        default=None,
        help=(
            "Lisää puuttuva language-attribuutti <voice>-tageihin. "
            "Anna kieli (esim. fi/en/es) pakotettuna tai jätä tyhjäksi automaattitulkitsemista varten."
        ),
    )
    args = parser.parse_args()

    print("Aloitetaan ääni-käsikirjoituksen muodostus parametreilla:")
    print(json.dumps(vars(args), ensure_ascii=False, indent=2))
    print("Parametrit OK, jatketaan käsittelyyn...")

    try:
        narrators = load_narrators(Path(args.narrators_file))
        section, act_no, chapter_no, chapter_title = extract_section(Path(args.input), args.content)
    except (ValueError, LookupError, FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Virhe: {e}", file=sys.stderr)
        return 2

    normalized, missing = normalize_unknown_characters(section, narrators)
    ssml_lines = to_ssml_lines(section, narrators, args.pronunciation_dictionary, args.openai_model)
    normalized = "<speak>\n" + "\n".join(ssml_lines) + "\n</speak>\n"
    if args.voice_language is not None:
        normalized = add_language_attribute(normalized, args.voice_language)
    if missing:
        print(
            f"Virhe: seuraavat hahmot puuttuvat tiedostosta {args.narrators_file}: {', '.join(missing)}. "
            f"Ne defaultattiin hahmoksi '{NARRATOR_NAME}'.",
            file=sys.stderr,
        )

    output = resolve_output_path(args.output, act_no, chapter_no, chapter_title)
    output.write_text(normalized, encoding="utf-8")
    print(f"Kirjoitettu: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())