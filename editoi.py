#!/usr/bin/env python3
import os
import sys
from typing import Optional, Tuple, List

from docx import Document
from openai import OpenAI
from openai import NotFoundError

# You can swap the preferred model here.
# GPT-5.6 uses a three-tier naming scheme:
#   gpt-5.6       -> alias for gpt-5.6-sol (highest capability)
#   gpt-5.6-terra -> balanced capability and cost
#   gpt-5.6-luna  -> lowest-cost, high-volume option
# For manuscript analysis, Sol is the default. Change MODEL to
# "gpt-5.6-terra" if you prefer the balanced-price tier.
MODEL = "gpt-5.6"

# Ordered fallback models (MODEL is the primary preferred option).
# If Sol is not enabled for the API project, the script tries Terra and Luna
# before falling back to earlier model generations.
MODEL_FALLBACKS = [
    "gpt-5.6-terra",
    "gpt-5.6-luna",
    "gpt-5.5",
    "gpt-5.4",
    "gpt-5.2",
    "gpt-4",
]

# Max output per single API call
MAX_OUTPUT_TOKENS = 3000
MODEL_PROBE_MAX_OUTPUT_TOKENS = 16

# How many automatic follow-up continuations to request at most
MAX_AUTO_FOLLOWUPS = 10

# If the model produces at least (MAX_OUTPUT_TOKENS - CONTINUE_THRESHOLD_TOKENS),
# we assume it likely hit the ceiling and should continue.
CONTINUE_THRESHOLD_TOKENS = 50

# Hard limit for how many files can be sent at once.
MAX_INPUT_FILES = 10

# A small visible tail is included in follow-up prompts so the model can continue
# from the actual printed output without needing a very long continuation prompt.
FOLLOWUP_TAIL_CHARS = 256


def clean_text(s: str) -> str:
    # Removes illegal surrogates and forces UTF-8 compatibility
    return s.encode("utf-8", errors="ignore").decode("utf-8")

client = OpenAI()  # uses OPENAI_API_KEY from the environment
ACTIVE_MODEL: Optional[str] = None


def _require_active_model() -> str:
    if not ACTIVE_MODEL:
        raise RuntimeError("ACTIVE_MODEL is not set. Call select_active_model() first.")
    return ACTIVE_MODEL


def _style_name_and_id(paragraph) -> Tuple[str, str]:
    """Return a paragraph's style name and style id, defensively."""
    style = getattr(paragraph, "style", None)
    if not style:
        return "", ""
    return (getattr(style, "name", "") or "", getattr(style, "style_id", "") or "")


def _heading_level(paragraph) -> Optional[int]:
    """Detect Word Heading 1 / Heading 2 styles, including localized Finnish names."""
    style_name, style_id = _style_name_and_id(paragraph)
    normalized = f"{style_name} {style_id}".strip().lower().replace("_", " ").replace("-", " ")

    heading_1_markers = ("heading 1", "heading1", "otsikko 1", "otsikko1")
    heading_2_markers = ("heading 2", "heading2", "otsikko 2", "otsikko2")

    if any(marker in normalized for marker in heading_1_markers):
        return 1
    if any(marker in normalized for marker in heading_2_markers):
        return 2
    return None


def load_docx_as_text(path: str) -> str:
    """
    Load a .docx file as plain text while preserving Heading 1 / Heading 2.

    Heading markers are made explicit because the OpenAI API receives only text,
    not Word paragraph styles. The markers are intentionally easy to prompt
    against, e.g. CHAPTER_HEADING_2.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    doc = Document(path)
    paragraphs: List[str] = []
    heading_1_count = 0
    heading_2_count = 0

    for p in doc.paragraphs:
        text = p.text.strip()
        if not text:
            continue

        level = _heading_level(p)

        if level == 1:
            heading_1_count += 1
            paragraphs.append(f"[HEADING_1 #{heading_1_count}: {text}]")
        elif level == 2:
            heading_2_count += 1
            paragraphs.append(f"[CHAPTER_HEADING_2 #{heading_2_count}: {text}]")
        else:
            paragraphs.append(text)

    return "\n\n".join(paragraphs)


def load_text_file(path: str) -> str:
    """Load a non-DOCX file as raw UTF-8 text and return it as-is."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_manuscript(path: str) -> str:
    """
    Load manuscript depending on file type.
    - .docx  -> parsed via python-docx
    - others -> read as raw text (UTF-8)
    """
    _, ext = os.path.splitext(path.lower())
    if ext == ".docx":
        return load_docx_as_text(path)
    return load_text_file(path)


def build_multi_file_payload(paths: List[str]) -> str:
    """
    Combine multiple manuscripts/files into one plain text payload,
    with clear file boundary markers so the model can refer to them.
    """
    chunks: List[str] = []
    for p in paths:
        filename = os.path.basename(p)
        text = load_manuscript(p)

        chunks.append(f"=== FILE: {filename} ===\n{text}\n=== END FILE: {filename} ===")

    return "\n\n".join(chunks)


def _safe_get_output_tokens(response) -> Optional[int]:
    """Safely extract output token count from the response."""
    usage = getattr(response, "usage", None)
    if not usage:
        return None
    return getattr(usage, "output_tokens", None)


def _safe_get_reasoning_tokens(response) -> Optional[int]:
    """Safely extract hidden reasoning-token count, when reported."""
    usage = getattr(response, "usage", None)
    if not usage:
        return None

    details = getattr(usage, "output_tokens_details", None)
    if not details:
        return None

    return getattr(details, "reasoning_tokens", None)


def _incomplete_reason(response) -> Optional[str]:
    """Return the API's reason for an incomplete response, when available."""
    details = getattr(response, "incomplete_details", None)
    if not details:
        return None

    return getattr(details, "reason", None)


def _debug_response(label: str, response) -> None:
    """Print useful response diagnostics without dumping the full response."""
    text = response.output_text or ""

    print(
        f"[DEBUG] {label}: "
        f"status={getattr(response, 'status', None)}, "
        f"incomplete_reason={_incomplete_reason(response)}, "
        f"output_tokens={_safe_get_output_tokens(response)}, "
        f"reasoning_tokens={_safe_get_reasoning_tokens(response)}, "
        f"visible_chars={len(text)}",
        flush=True,
    )


def _call_initial(book_text: str, question: str):
    """Initial API call: send the full manuscript (possibly multiple files) + the question."""
#    instructions = (
#        "Luet käsikirjoitusta kokeneen kaunokirjallisuuden toimittajan näkökulmasta. "
#        "Tarkastelet, millaisen lukukokemuksen teksti rakentaa ja miten sen eri osat palvelevat kokonaisuutta. "
#        "Tunnistat tekstin vahvuudet, jännitteet ja mahdolliset rakenteelliset heikkoudet. "
#        "Palautteesi on analyyttistä, harkittua ja kirjoittajaa kunnioittavaa. "
#        "Arvioi ennen kaikkea sitä, mikä on merkityksellistä teoksen kannalta. "
#        "Aloita palautteesi 10–20 rivin yhteenvedolla ja tärkeimmillä huomioillasi."
#    )
    instructions = (
        "Luet käsikirjoitusta kokeneen kaunokirjallisuuden kustannustoimittajan näkökulmasta. "
        "Analysoit vain ja ainoastaan sinulle annettua materiaalia. "
        "Palautteesi on analyyttistä, harkittua ja kirjoittajaa kunnioittavaa. "
        "Älä tee oletuksia sisällöstä, jota ei ole eksplisiittisesti läsnä tiedostoissa.\n\n"

        "Word-tiedostoista Heading 1 ja Heading 2 -tyylit on muunnettu tekstissä eksplisiittisiksi merkeiksi. "
        "Heading 1 näkyy muodossa [HEADING_1 #N: otsikko]. "
        "Heading 2 näkyy muodossa [CHAPTER_HEADING_2 #N: otsikko]. "
        "Jos käyttäjä pyytää lukuja, lukukohtaista analyysiä tai chapter-listausta, käytä ensisijaisesti "
        "CHAPTER_HEADING_2-merkintöjä varsinaisten lukujen tunnistamiseen, ellei käyttäjä toisin määrää.\n\n"

        "Kun esität tekstistä johdettavia väitteitä (rakenteellisia, temaattisia, psykologisia, dramaturgisia), "
        "ankkuroi jokainen merkittävä väite konkreettiseen kohtaan tekstissä. "
        "Viittaa vähintään yhteen seuraavista: luvun tai jakson otsikko, selkeä tunnistettava kohtaus "
        "tai lyhyt suora/osittainen sitaatti.\n\n"

        "Muokkaus- ja kehitysehdotukset saat esittää ilman sitaattia, mutta jos perustat ehdotuksen "
        "diagnostiikkaan (mikä toimii / mikä ei), se pitää ankkuroida. "
        "Jos et löydä aineistosta suoraa tukea väitteellesi, sano tämä eksplisiittisesti "
        "('Tätä ei voi varmistaa annetusta materiaalista'). "
        "Älä täytä aukkoja yleisillä kustannustoimittajakliseillä.\n\n"

        "Jos käyttäjä pyytää numeerista laskentaa (esim. merkit, liuskat, lukumäärät), "
        "tee laskenta suoraan annetusta tekstistä ja raportoi tulos. "
        "Kerro lyhyesti laskentatapa, jos se sopii pyydettyyn vastausformaattiin.\n\n"

        "Noudata käyttäjän pyytämää vastausformaattia täsmällisesti. "
        "Jos käyttäjä pyytää CSV:tä, taulukkoa, JSONia, listaa tai muuta tiukkaa formaattia, palauta vain se formaatti "
        "ilman johdantoa, yhteenvetoa tai jälkisanoja. "
        "Tee alun yhteenveto vain silloin, kun se selvästi sopii käyttäjän kysymykseen ja vastausmuotoon, "
        "esimerkiksi avoimessa laadullisessa arviopyynnössä. "
        "Älä tee yhteenvetoa, jos käyttäjä kieltää sen tai pyytää pelkkää määrämuotoista tulosta. "
        "Pyydä täsmennystä vain, jos sitä ilman et voi vastata."
    )
   


    print("[DEBUG] Sending INITIAL request (all input files + question)...", flush=True)
    response = client.responses.create(
        model=_require_active_model(),
        instructions=instructions,
        input=[
            {
                "role": "user",
                "content": (
                    "Tässä on käsikirjoitukseni / analyysiaineistoni tiedostoina. "
                    "Pidä mielessä tiedostorajaukset ja viittaa tarvittaessa FILE-nimiin.\n\n"
                    "=== ALKU ===\n"
                    f"{book_text}\n"
                    "=== LOPPU ===\n\n"
                    f"Kysymys: {question}"
                ),
            }
        ],
        max_output_tokens=MAX_OUTPUT_TOKENS,
        store=True,
    )
    print("[DEBUG] INITIAL request finished.", flush=True)

    _debug_response("INITIAL", response)
    return response


def _call_followup(previous_response_id: str, content: str, index: int):
    """Follow-up call: keep the same session, send only a new message."""
    print(f"[DEBUG] Sending FOLLOWUP #{index}...", flush=True)
    response = client.responses.create(
        model=_require_active_model(),
        previous_response_id=previous_response_id,
        input=[
            {
                "role": "user",
                "content": content,
            }
        ],
        max_output_tokens=MAX_OUTPUT_TOKENS,
        store=True,
    )
    print(f"[DEBUG] FOLLOWUP #{index} finished.", flush=True)

    _debug_response(f"FOLLOWUP #{index}", response)
    return response


def _should_continue(response, followups_used_now: int) -> bool:
    """
    Continue when the API explicitly reports an incomplete response.

    A token-count heuristic is retained only as a defensive fallback for cases
    where the API does not provide a useful incomplete status.
    """
    if followups_used_now >= MAX_AUTO_FOLLOWUPS:
        return False

    status = getattr(response, "status", None)
    incomplete_reason = _incomplete_reason(response)

    if status == "incomplete":
        # Automatic continuation is appropriate only when the output budget
        # was exhausted. Do not loop on content filters or other failures.
        return incomplete_reason in (None, "max_output_tokens")

    if incomplete_reason is not None:
        return incomplete_reason == "max_output_tokens"

    output_tokens = _safe_get_output_tokens(response)
    visible_text = response.output_text or ""
    threshold = MAX_OUTPUT_TOKENS - CONTINUE_THRESHOLD_TOKENS

    return (
        bool(visible_text.strip())
        and output_tokens is not None
        and output_tokens >= threshold
    )


def _make_followup_prompt(accumulated: List[str]) -> str:
    """
    Create a continuation prompt appropriate to the visible output.

    If no visible output exists yet, ask the model to proceed to the answer.
    Otherwise provide a short seam marker from the printed response.
    """
    visible_parts = [part for part in accumulated if part and part.strip()]
    current_output = "\n".join(visible_parts)

    if not current_output:
        return (
            "Jatka tehtävän suorittamista käyttäen edellisen vastauksen "
            "säilytettyä päättelykontekstia. Aloita nyt varsinainen näkyvä vastaus. "
            "Älä kommentoi tokenrajaa, päättelyä, jatkokutsua tai sitä, ettei "
            "edellisessä vastauksessa ollut näkyvää tekstiä. "
            "Noudata alkuperäisen käyttäjäpyynnön formaattia täsmällisesti."
        )

    tail = current_output[-FOLLOWUP_TAIL_CHARS:]

    return (
        "Jatka edellistä vastaustasi suoraan siitä kohdasta, johon se jäi.\n"
        "Älä kommentoi jatkamista, tokenrajaa tai teknistä kontekstia.\n"
        "Älä aloita alusta äläkä toista jo annettua tekstiä tai otsikoita.\n"
        "Jos viimeinen lause jäi kesken, jatka sitä suoraan.\n"
        "Säilytä täsmälleen sama vastausformaatti.\n\n"
        "Tulostetun vastauksen loppu saumakohdan tunnistamista varten:\n"
        "---\n"
        f"{tail}\n"
        "---"
    )


def ask_question(
    book_text: str,
    question: str,
    previous_response_id: Optional[str] = None,
) -> Tuple[str, str]:
    """Ask the model and automatically continue incomplete responses."""
    accumulated: List[str] = []
    followups_used = 0

    if previous_response_id is None:
        response = _call_initial(book_text, question)
    else:
        # A new user question inside the existing session.
        response = _call_followup(
            previous_response_id,
            question,
            index=0,
        )

    text = response.output_text or ""
    if text:
        accumulated.append(text)

    last_response_id = response.id

    while _should_continue(response, followups_used):
        followups_used += 1
        followup_prompt = _make_followup_prompt(accumulated)

        response = _call_followup(
            last_response_id,
            followup_prompt,
            index=followups_used,
        )

        text = response.output_text or ""
        if text:
            accumulated.append(text)

        last_response_id = response.id

    full_answer = "\n".join(accumulated)

    if not full_answer.strip():
        status = getattr(response, "status", None)
        reason = _incomplete_reason(response)

        raise RuntimeError(
            "Model returned no visible response after automatic continuations. "
            f"Last status={status!r}, incomplete_reason={reason!r}, "
            f"followups_used={followups_used}."
        )

    print(
        f"[DEBUG] Done. followups_used={followups_used}.",
        flush=True,
    )

    return full_answer, last_response_id


def _usage_exit() -> None:
    print("Usage: python3 editoi.py FILE1 [FILE2 ... FILE10]")
    print("Examples:")
    print("  python3 editoi.py manuscript.docx")
    print("  python3 editoi.py mts.txt staccato.json")
    print("  python3 editoi.py Edit2.docx staccato.csv notes.txt")
    sys.exit(1)


def _candidate_models() -> List[str]:
    """Return primary model + unique fallbacks in order."""
    ordered = [MODEL, *MODEL_FALLBACKS]
    unique: List[str] = []
    for model_name in ordered:
        if model_name and model_name not in unique:
            unique.append(model_name)
    return unique


def _model_accessible(model_name: str) -> bool:
    """Probe whether the model is available to this API key/account."""
    try:
        client.responses.create(
            model=model_name,
            input=[{"role": "user", "content": "ping"}],
            max_output_tokens=MODEL_PROBE_MAX_OUTPUT_TOKENS,
            store=False,
        )
        return True
    except NotFoundError as e:
        print(f"[DEBUG] Model '{model_name}' unavailable: {e}", flush=True)
        return False
    except Exception as e:
        # Keep trying fallbacks even for non-404 failures.
        print(f"[DEBUG] Model '{model_name}' probe failed: {type(e).__name__}: {e}", flush=True)
        return False


def select_active_model() -> str:
    """
    Test candidate models in order and return the first one that is available.
    Raises RuntimeError if no candidate model can be used.
    """
    candidates = _candidate_models()
    print(f"[DEBUG] Model preference order: {candidates}", flush=True)

    for model_name in candidates:
        print(f"[DEBUG] Probing model access: '{model_name}'...", flush=True)
        if _model_accessible(model_name):
            print(f"[DEBUG] Using model: '{model_name}'", flush=True)
            return model_name

    raise RuntimeError(
        "No configured model is accessible. "
        f"Tried: {', '.join(candidates)}"
    )


def main():
    global ACTIVE_MODEL

    if len(sys.argv) < 2:
        _usage_exit()

    paths = sys.argv[1:]

    if len(paths) > MAX_INPUT_FILES:
        print(f"Error: Too many files. Max is {MAX_INPUT_FILES}, got {len(paths)}.")
        sys.exit(1)

    for p in paths:
        if not os.path.exists(p):
            print(f"Error: File not found: {p}")
            sys.exit(1)

    print("Loading input files:")
    for p in paths:
        print(f"  - {p}")

    try:
        book_text = build_multi_file_payload(paths)
    except Exception as e:
        print(f"Error loading files: {e}")
        sys.exit(1)

    print("All files loaded.")
    print(f"Combined payload length: {len(book_text)} characters.\n")

    print("On the first question, ALL input files are sent to the model.")
    print("Subsequent questions continue the same session (previous_response_id).")
    print("Exit with Ctrl-C.\n")

    try:
        ACTIVE_MODEL = select_active_model()
    except RuntimeError as e:
        print(f"Error selecting model: {e}")
        sys.exit(1)

    previous_response_id: Optional[str] = None

    try:
        while True:
            question = clean_text(input("Kysymys> ").strip())

            if not question:
                continue

            if previous_response_id is None:
                print("\n[SENDING ALL INPUT FILES + QUESTION]\n")
                answer, previous_response_id = ask_question(
                    book_text,
                    question,
                    previous_response_id=None,
                )
            else:
                print("\n[SENDING ONLY A NEW QUESTION (session continues, auto-continue enabled)]\n")
                answer, previous_response_id = ask_question(
                    book_text,
                    question,
                    previous_response_id=previous_response_id,
                )

            print("--- RESPONSE ---\n")
            print(answer)
            print("\n" + "-" * 80 + "\n")

    except KeyboardInterrupt:
        print("\n\nCtrl-C received. Exiting. Bye!")


if __name__ == "__main__":
    main()
