#!/usr/bin/env python3
import os
import sys
from typing import Optional, Tuple, List

from docx import Document
from openai import OpenAI

# You can swap model here.
MODEL = "gpt-5.2"

# Max output per single API call
MAX_OUTPUT_TOKENS = 3000

# How many automatic follow-up continuations to request at most
MAX_AUTO_FOLLOWUPS = 10

# If the model produces at least (MAX_OUTPUT_TOKENS - CONTINUE_THRESHOLD_TOKENS),
# we assume it likely hit the ceiling and should continue.
CONTINUE_THRESHOLD_TOKENS = 50

# Hard limit for how many files can be sent at once.
MAX_INPUT_FILES = 10

# A strongly worded continuation prompt to reduce repetition/drift.
FOLLOWUP_PROMPT = (
    "Jatka täsmälleen siitä, mihin jäit.\n"
    "Älä toista aiempaa tekstiä tai otsikoita, jotka ovat jo näkyneet.\n"
    "Jatka samalla otsikkorakenteella ja samalla sävyllä.\n"
    "Aloita keskeneräisestä lauseesta/kappaleesta ja jatka eteenpäin.\n"
)

def clean_text(s: str) -> str:
    # Removes illegal surrogates and forces UTF-8 compatibility
    return s.encode("utf-8", errors="ignore").decode("utf-8")

client = OpenAI()  # uses OPENAI_API_KEY from the environment


def load_docx_as_text(path: str) -> str:
    """Load a .docx file as a single plain-text string (paragraphs separated by blank lines)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    doc = Document(path)
    paragraphs = []
    for p in doc.paragraphs:
        text = p.text.strip()
        if text:
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

    "Kun esität tekstistä johdettavia väitteitä (rakenteellisia, temaattisia, psykologisia, dramaturgisia), "
    "ankkuroi jokainen merkittävä väite konkreettiseen kohtaan tekstissä. "
    "Viittaa aina vähintään yhteen seuraavista:\n"
    "- luvun tai jakson otsikko sellaisena kuin se esiintyy tiedostossa\n"
    "- selkeä tunnistettava kohtaus tai tapahtuma\n"
    "- suora tai osittainen sitaatti (lyhyt)\n\n"
    "Muokkaus- ja kehitysehdotukset saat esittää ilman sitaattia, mutta jos perustat ehdotuksen "
    "diagnostiikkaan (mikä toimii / mikä ei), se pitää ankkuroida.\n\n"

    "Jos et löydä aineistosta suoraa tukea väitteellesi, sano tämä eksplisiittisesti "
    "('Tätä ei voi varmistaa annetusta materiaalista'). "
    "Älä täytä aukkoja yleisillä kustannustoimittajakliseillä.\n\n"

    "Jos käyttäjä pyytää numeerista laskentaa (esim. merkit, liuskat, lukumäärät), "
    "tee laskenta suoraan annetusta tekstistä ja raportoi tulos. "
    "Kerro lyhyesti laskentatapa (rajaukset, lasketaanko otsikot mukaan, lasketaanko rivinvaihdot merkkeinä). "
    "Tätä ei käsitellä kustannustoimittaja-väitteenä eikä se vaadi sitaattiankkurointia.\n\n"

    "Aloita vastauksesi 1–20 rivin yhteenvedolla, jonka jokainen keskeinen väite on jäljitettävissä aineistoon. "
    "Jos vastaus on puhtaasti numeerinen, aloita taulukolla ja lisää 1–3 rivin selite. "
    "Tämän jälkeen voit edetä jäsenneltyyn analyysiin, jos sellaiselle on tarvetta vastauksessa. "
    "Pyydä täsmennystä vain, jos sitä ilman et voi vastata."
    )
   


    print("[DEBUG] Sending INITIAL request (all input files + question)...", flush=True)
    response = client.responses.create(
        model=MODEL,
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

    text = response.output_text
    response_id = response.id
    output_tokens = _safe_get_output_tokens(response)

    print(f"[DEBUG] INITIAL: output_tokens={output_tokens}", flush=True)
    return text, response_id, output_tokens


def _call_followup(previous_response_id: str, content: str, index: int):
    """Follow-up call: keep the same session, send only a new message."""
    print(f"[DEBUG] Sending FOLLOWUP #{index}...", flush=True)
    response = client.responses.create(
        model=MODEL,
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

    text = response.output_text
    response_id = response.id
    output_tokens = _safe_get_output_tokens(response)

    print(f"[DEBUG] FOLLOWUP #{index}: output_tokens={output_tokens}", flush=True)
    return text, response_id, output_tokens


def _should_continue(out_tokens: Optional[int], followups_used_now: int) -> bool:
    """Decide whether to request another continuation."""
    if followups_used_now >= MAX_AUTO_FOLLOWUPS:
        return False

    if out_tokens is None:
        return False

    threshold = MAX_OUTPUT_TOKENS - CONTINUE_THRESHOLD_TOKENS
    return out_tokens >= threshold


def ask_question(
    book_text: str,
    question: str,
    previous_response_id: Optional[str] = None,
) -> Tuple[str, str]:
    """Ask the model and automatically continue if the response is likely truncated."""
    accumulated = []
    followups_used = 0

    if previous_response_id is None:
        text, response_id, output_tokens = _call_initial(book_text, question)
    else:
        # Session continues: send only the new question.
        text, response_id, output_tokens = _call_followup(previous_response_id, question, index=0)

    accumulated.append(text)
    last_response_id = response_id

    while _should_continue(output_tokens, followups_used):
        followups_used += 1
        text, response_id, output_tokens = _call_followup(
            last_response_id,
            FOLLOWUP_PROMPT,
            index=followups_used,
        )
        accumulated.append(text)
        last_response_id = response_id

    full_answer = "\n".join(accumulated)
    print(f"[DEBUG] Done. followups_used={followups_used}.", flush=True)
    return full_answer, last_response_id


def _usage_exit() -> None:
    print("Usage: python3 editoi.py FILE1 [FILE2 ... FILE10]")
    print("Examples:")
    print("  python3 editoi.py manuscript.docx")
    print("  python3 editoi.py mts.txt staccato.json")
    print("  python3 editoi.py Edit2.docx staccato.csv notes.txt")
    sys.exit(1)


def main():
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
