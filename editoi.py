#!/usr/bin/env python3
import hashlib
import importlib
import importlib.metadata
import inspect
import os
import shlex
import sys
from typing import Any, Dict, List, Optional, Tuple


REQUIRED_PACKAGES = (
    ("openai", "openai"),
    ("python-docx", "docx"),
)


def _pip_install_command() -> str:
    """Return an install command that targets this exact Python environment."""
    packages = " ".join(distribution for distribution, _ in REQUIRED_PACKAGES)
    python_executable = shlex.quote(sys.executable)
    return f"{python_executable} -m pip install --upgrade {packages}"


def _distribution_version(distribution_name: str) -> str:
    """Return an installed distribution version without importing the package."""
    try:
        return importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return "ei asennettu"


def _print_dependency_repair(reason: str, details: List[str]) -> None:
    """Print one actionable repair command for dependency problems and exit."""
    print(f"Virhe: {reason}", file=sys.stderr)
    for detail in details:
        print(f"  - {detail}", file=sys.stderr)
    print("\nAsenna tai päivitä kaikki tarvittavat paketit tällä komennolla:", file=sys.stderr)
    print(f"  {_pip_install_command()}", file=sys.stderr)
    print("\nAja sen jälkeen ohjelma uudelleen samassa virtuaaliympäristössä.", file=sys.stderr)
    sys.exit(1)


def _load_third_party_dependencies():
    """Import every non-standard-library dependency and report all failures at once."""
    modules: Dict[str, Any] = {}
    failures: List[str] = []

    for distribution_name, module_name in REQUIRED_PACKAGES:
        try:
            modules[module_name] = importlib.import_module(module_name)
        except Exception as error:
            version = _distribution_version(distribution_name)
            failures.append(
                f"{distribution_name} ({version}): {error.__class__.__name__}: {error}"
            )

    if failures:
        _print_dependency_repair(
            "yksi tai useampi Python-riippuvuus puuttuu tai on rikki.",
            failures,
        )

    openai_module = modules["openai"]
    docx_module = modules["docx"]

    try:
        openai_class = openai_module.OpenAI
        not_found_error = openai_module.NotFoundError
        document_class = docx_module.Document
    except AttributeError as error:
        _print_dependency_repair(
            "asennettu kirjasto ei tarjoa ohjelman tarvitsemaa rajapintaa.",
            [str(error)],
        )

    return document_class, not_found_error, openai_class


def _verify_openai_sdk(openai_class) -> None:
    """Ensure the installed SDK supports GPT-5.6 explicit prompt-cache options."""
    version = _distribution_version("openai")
    probe_client = None

    try:
        # A dummy key is enough for local interface inspection; no API request is sent.
        probe_client = openai_class(api_key="sk-dependency-interface-check")
        create_signature = inspect.signature(probe_client.responses.create)
        parameters = create_signature.parameters
        accepts_arbitrary_keywords = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        supports_cache_options = (
            "prompt_cache_options" in parameters or accepts_arbitrary_keywords
        )
    except Exception as error:
        _print_dependency_repair(
            f"openai SDK:n Responses API -rajapintaa ei voitu tarkistaa (versio {version}).",
            [f"{error.__class__.__name__}: {error}"],
        )
    finally:
        if probe_client is not None:
            try:
                probe_client.close()
            except Exception:
                pass

    if not supports_cache_options:
        _print_dependency_repair(
            f"openai SDK on liian vanha (asennettu versio {version}).",
            [
                "Responses.create() ei tunne prompt_cache_options-parametria, "
                "jota GPT-5.6:n eksplisiittinen prompt caching tarvitsee."
            ],
        )


Document, NotFoundError, OpenAI = _load_third_party_dependencies()
_verify_openai_sdk(OpenAI)
OPENAI_SDK_VERSION = _distribution_version("openai")
PYTHON_DOCX_VERSION = _distribution_version("python-docx")

# Preferred model. GPT-5.6 explicit prompt caching is used when available.
MODEL = "gpt-5.6"

# Ordered fallbacks. Model availability is tested lazily with the real request,
# so the script no longer spends a separate API call on model probing.
MODEL_FALLBACKS = [
    "gpt-5.6-terra",
    "gpt-5.6-luna",
    "gpt-5.5",
    "gpt-5.4",
    "gpt-5.2",
    "gpt-4",
]

# Max output per single API call. 4000 is a compromise between room for a
# manuscript analysis and the timeout risk of very large output budgets.
MAX_OUTPUT_TOKENS = 4000

# How many automatic follow-up continuations to request at most.
MAX_AUTO_FOLLOWUPS = 10

# If the model produces at least (MAX_OUTPUT_TOKENS - threshold), assume it may
# have hit the ceiling when no explicit incomplete reason is available.
CONTINUE_THRESHOLD_TOKENS = 50

# Hard limit for how many files can be sent at once.
MAX_INPUT_FILES = 10

# A short visible tail helps a continuation resume at the correct seam.
FOLLOWUP_TAIL_CHARS = 256

# GPT-5.6 currently supports only a 30-minute explicit-cache minimum TTL.
PROMPT_CACHE_TTL = "30m"
CACHE_KEY_VERSION = "editor-v2"

# Models before GPT-5.6 that support 24-hour extended prompt-cache retention.
LEGACY_EXTENDED_CACHE_PREFIXES = (
    "gpt-5.5",
    "gpt-5.4",
    "gpt-5.2",
)

EDITOR_INSTRUCTIONS = (
    "Toimi kokeneena kaunokirjallisuuden kustannustoimittajana. "
    "Analysoi vain annettua aineistoa äläkä täytä sen aukkoja oletuksilla. "
    "Word-tiedostojen osat on merkitty muodossa [HEADING_1 #N: otsikko] ja "
    "varsinaiset luvut muodossa [CHAPTER_HEADING_2 #N: otsikko]. Käytä "
    "CHAPTER_HEADING_2-merkintöjä lukujen tunnistamiseen, ellei käyttäjä toisin määrää. "
    "Ankkuroi jokainen keskeinen tekstistä johdettu diagnoosi luvun tai jakson otsikkoon, "
    "tunnistettavaan kohtaukseen tai lyhyeen sitaattiin; yksi hyvä ankkuri väitettä kohti riittää. "
    "Erota tekstihavainto ja kehitysehdotus toisistaan. Jos väitettä ei voi varmistaa aineistosta, "
    "sano se eksplisiittisesti. Vältä yleisiä kustannustoimittajakliseitä, tarpeetonta toistoa ja "
    "pitkiä sitaatteja. Tee pyydetyt numeeriset laskelmat annetusta aineistosta ja kerro tarvittaessa "
    "laskentatapa lyhyesti. Noudata käyttäjän pyytämää vastausformaattia täsmällisesti; jos käyttäjä "
    "pyytää tiukkaa formaattia, palauta vain pyydetty sisältö ilman johdantoa tai jälkisanoja. "
    "Pyydä täsmennystä vain, jos tehtävää ei voi muuten suorittaa."
)


def clean_text(s: str) -> str:
    """Remove illegal surrogates and force UTF-8-compatible text."""
    return s.encode("utf-8", errors="ignore").decode("utf-8")


def _create_openai_client():
    """Create the real API client with a clear error if the API key is missing."""
    if not os.environ.get("OPENAI_API_KEY"):
        print("Virhe: OPENAI_API_KEY-ympäristömuuttuja puuttuu.", file=sys.stderr)
        print('Aseta se esimerkiksi: export OPENAI_API_KEY="sk-..."', file=sys.stderr)
        sys.exit(1)
    return OpenAI()


client = _create_openai_client()
ACTIVE_MODEL: Optional[str] = None


def _style_name_and_id(paragraph) -> Tuple[str, str]:
    """Return a paragraph's style name and style id defensively."""
    style = getattr(paragraph, "style", None)
    if not style:
        return "", ""
    return (getattr(style, "name", "") or "", getattr(style, "style_id", "") or "")


def _heading_level(paragraph) -> Optional[int]:
    """Detect Word Heading 1 / Heading 2 styles, including Finnish names."""
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
    """Load DOCX as text while preserving Heading 1 and Heading 2 markers."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    doc = Document(path)
    paragraphs: List[str] = []
    heading_1_count = 0
    heading_2_count = 0

    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue

        level = _heading_level(paragraph)
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
    """Load a non-DOCX file as UTF-8 text."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "r", encoding="utf-8") as file_handle:
        return file_handle.read()


def load_manuscript(path: str) -> str:
    """Load DOCX with python-docx and other files as raw UTF-8 text."""
    _, extension = os.path.splitext(path.lower())
    if extension == ".docx":
        return load_docx_as_text(path)
    return load_text_file(path)


def build_multi_file_payload(paths: List[str]) -> str:
    """Combine files into one stable text payload with explicit boundaries."""
    chunks: List[str] = []
    for path in paths:
        filename = os.path.basename(path)
        text = load_manuscript(path)
        chunks.append(f"=== FILE: {filename} ===\n{text}\n=== END FILE: {filename} ===")
    return "\n\n".join(chunks)


def build_static_material(book_text: str) -> str:
    """Build the exact reusable user-message prefix placed before each task."""
    return (
        "Tässä on käsikirjoitukseni / analyysiaineistoni tiedostoina. "
        "Pidä mielessä tiedostorajaukset ja viittaa tarvittaessa FILE-nimiin.\n\n"
        "=== ALKU ===\n"
        f"{book_text}\n"
        "=== LOPPU ==="
    )


def make_prompt_cache_key(instructions: str, static_material: str) -> str:
    """
    Create a stable local routing key for an exact reusable prompt prefix.

    OpenAI does not return a cache GUID. The service combines this key with its
    own exact-prefix hash. If the instructions or any input file changes, this
    SHA-256 value changes automatically.
    """
    cache_identity = (
        CACHE_KEY_VERSION.encode("utf-8")
        + b"\0"
        + instructions.encode("utf-8")
        + b"\0"
        + static_material.encode("utf-8")
    )
    digest = hashlib.sha256(cache_identity).hexdigest()
    return f"manuscript:{digest[:40]}"


def _candidate_models() -> List[str]:
    """Return primary model plus unique fallbacks in order."""
    ordered = [MODEL, *MODEL_FALLBACKS]
    unique: List[str] = []
    for model_name in ordered:
        if model_name and model_name not in unique:
            unique.append(model_name)
    return unique


def _models_for_initial_request() -> List[str]:
    """Try the active model first, then any remaining candidates."""
    candidates = _candidate_models()
    if ACTIVE_MODEL and ACTIVE_MODEL in candidates:
        return [ACTIVE_MODEL, *[name for name in candidates if name != ACTIVE_MODEL]]
    return candidates


def _supports_explicit_cache(model_name: str) -> bool:
    """Explicit breakpoints and prompt_cache_options are GPT-5.6+ features."""
    return model_name.startswith("gpt-5.6")


def _supports_legacy_extended_cache(model_name: str) -> bool:
    """Return whether a fallback supports prompt_cache_retention='24h'."""
    return model_name.startswith(LEGACY_EXTENDED_CACHE_PREFIXES)


def _cache_request_options(model_name: str, cache_key: str) -> Dict[str, Any]:
    """Return model-compatible request-wide prompt-cache parameters."""
    if _supports_explicit_cache(model_name):
        return {
            "prompt_cache_key": cache_key,
            "prompt_cache_options": {
                "mode": "explicit",
                "ttl": PROMPT_CACHE_TTL,
            },
        }

    if _supports_legacy_extended_cache(model_name):
        return {
            "prompt_cache_key": cache_key,
            "prompt_cache_retention": "24h",
        }

    # Very old fallback models may reject all explicit cache parameters.
    return {}


def _initial_input_payload(
    model_name: str,
    static_material: str,
    question: str,
) -> List[Dict[str, Any]]:
    """Create a stable cached prefix followed by the variable analysis task."""
    task_text = f"\n\n=== ANALYYSITEHTÄVÄ ===\n{question}"

    static_block: Dict[str, Any] = {
        "type": "input_text",
        "text": static_material,
    }

    if _supports_explicit_cache(model_name):
        static_block["prompt_cache_breakpoint"] = {"mode": "explicit"}

    return [
        {
            "role": "user",
            "content": [
                static_block,
                {
                    "type": "input_text",
                    "text": task_text,
                },
            ],
        }
    ]


def _safe_get_usage_value(response, attribute: str) -> Optional[int]:
    usage = getattr(response, "usage", None)
    if not usage:
        return None
    return getattr(usage, attribute, None)


def _safe_get_input_detail(response, attribute: str) -> Optional[int]:
    usage = getattr(response, "usage", None)
    details = getattr(usage, "input_tokens_details", None) if usage else None
    if not details:
        return None
    return getattr(details, attribute, None)


def _safe_get_output_detail(response, attribute: str) -> Optional[int]:
    usage = getattr(response, "usage", None)
    details = getattr(usage, "output_tokens_details", None) if usage else None
    if not details:
        return None
    return getattr(details, attribute, None)


def _incomplete_reason(response) -> Optional[str]:
    details = getattr(response, "incomplete_details", None)
    return getattr(details, "reason", None) if details else None


def _debug_response(label: str, response, cache_key: str) -> None:
    """Print token and cache diagnostics without dumping the response object."""
    text = response.output_text or ""
    input_tokens = _safe_get_usage_value(response, "input_tokens")
    cached_tokens = _safe_get_input_detail(response, "cached_tokens")
    cache_write_tokens = _safe_get_input_detail(response, "cache_write_tokens")
    output_tokens = _safe_get_usage_value(response, "output_tokens")
    reasoning_tokens = _safe_get_output_detail(response, "reasoning_tokens")
    total_tokens = _safe_get_usage_value(response, "total_tokens")

    cache_read_ratio: Optional[float] = None
    if input_tokens and cached_tokens is not None:
        cache_read_ratio = cached_tokens / input_tokens * 100.0

    ratio_text = f"{cache_read_ratio:.1f}%" if cache_read_ratio is not None else "n/a"

    print(
        f"[DEBUG] {label}: "
        f"model={getattr(response, 'model', ACTIVE_MODEL)}, "
        f"status={getattr(response, 'status', None)}, "
        f"incomplete_reason={_incomplete_reason(response)}, "
        f"input_tokens={input_tokens}, "
        f"cached_tokens={cached_tokens}, "
        f"cache_write_tokens={cache_write_tokens}, "
        f"cache_read_ratio={ratio_text}, "
        f"output_tokens={output_tokens}, "
        f"reasoning_tokens={reasoning_tokens}, "
        f"total_tokens={total_tokens}, "
        f"visible_chars={len(text)}, "
        f"cache_key={cache_key}",
        flush=True,
    )


def _create_initial_response(
    static_material: str,
    question: str,
    cache_key: str,
):
    """Create a new independent analysis, lazily falling back by model access."""
    global ACTIVE_MODEL

    last_not_found: Optional[Exception] = None
    for model_name in _models_for_initial_request():
        print(f"[DEBUG] Trying model for real request: '{model_name}'...", flush=True)

        request_args: Dict[str, Any] = {
            "model": model_name,
            "instructions": EDITOR_INSTRUCTIONS,
            "input": _initial_input_payload(model_name, static_material, question),
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "store": True,
        }
        request_args.update(_cache_request_options(model_name, cache_key))

        try:
            response = client.responses.create(**request_args)
            ACTIVE_MODEL = model_name
            print(f"[DEBUG] Active model: '{ACTIVE_MODEL}'", flush=True)
            return response
        except NotFoundError as error:
            last_not_found = error
            print(f"[DEBUG] Model '{model_name}' unavailable: {error}", flush=True)

    raise RuntimeError(
        "No configured model is accessible. "
        f"Tried: {', '.join(_candidate_models())}. "
        f"Last error: {last_not_found}"
    )


def _create_followup_response(
    previous_response_id: str,
    content: str,
    cache_key: str,
):
    """Continue an existing response chain with the already selected model."""
    if not ACTIVE_MODEL:
        raise RuntimeError("No active model exists for a follow-up request.")

    request_args: Dict[str, Any] = {
        "model": ACTIVE_MODEL,
        "previous_response_id": previous_response_id,
        # Instructions are not inherited automatically with previous_response_id.
        "instructions": EDITOR_INSTRUCTIONS,
        "input": [
            {
                "role": "user",
                "content": content,
            }
        ],
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "store": True,
    }
    request_args.update(_cache_request_options(ACTIVE_MODEL, cache_key))
    return client.responses.create(**request_args)


def _call_initial(static_material: str, question: str, cache_key: str):
    print(
        "[DEBUG] Sending NEW ANALYSIS request "
        "(stable manuscript prefix + variable task)...",
        flush=True,
    )
    response = _create_initial_response(static_material, question, cache_key)
    _debug_response("NEW ANALYSIS", response, cache_key)
    return response


def _call_followup(
    previous_response_id: str,
    content: str,
    index: int,
    cache_key: str,
):
    print(f"[DEBUG] Sending FOLLOWUP #{index}...", flush=True)
    response = _create_followup_response(previous_response_id, content, cache_key)
    _debug_response(f"FOLLOWUP #{index}", response, cache_key)
    return response


def _should_continue(response, followups_used_now: int) -> bool:
    """Continue only when output budget exhaustion is explicit or very likely."""
    if followups_used_now >= MAX_AUTO_FOLLOWUPS:
        return False

    status = getattr(response, "status", None)
    incomplete_reason = _incomplete_reason(response)

    if status == "incomplete":
        return incomplete_reason in (None, "max_output_tokens")

    if incomplete_reason is not None:
        return incomplete_reason == "max_output_tokens"

    output_tokens = _safe_get_usage_value(response, "output_tokens")
    visible_text = response.output_text or ""
    threshold = MAX_OUTPUT_TOKENS - CONTINUE_THRESHOLD_TOKENS

    return (
        bool(visible_text.strip())
        and output_tokens is not None
        and output_tokens >= threshold
    )


def _make_followup_prompt(accumulated: List[str]) -> str:
    """Create a short continuation instruction with a visible seam marker."""
    visible_parts = [part for part in accumulated if part and part.strip()]
    current_output = "\n".join(visible_parts)

    if not current_output:
        return (
            "Jatka alkuperäisen tehtävän suorittamista edellisen vastauksen kontekstista. "
            "Aloita nyt varsinainen näkyvä vastaus. Älä kommentoi tokenrajaa, päättelyä, "
            "jatkokutsua tai näkyvän tekstin puuttumista. Noudata alkuperäistä formaattia."
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
    static_material: str,
    question: str,
    cache_key: str,
    previous_response_id: Optional[str] = None,
) -> Tuple[str, str]:
    """Run a new analysis or a deliberate follow-up, then auto-continue if cut."""
    accumulated: List[str] = []
    followups_used = 0

    if previous_response_id is None:
        response = _call_initial(static_material, question, cache_key)
    else:
        response = _call_followup(
            previous_response_id,
            question,
            index=0,
            cache_key=cache_key,
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
            cache_key=cache_key,
        )

        text = response.output_text or ""
        if text:
            accumulated.append(text)
        last_response_id = response.id

    full_answer = "\n".join(accumulated)
    if not full_answer.strip():
        raise RuntimeError(
            "Model returned no visible response after automatic continuations. "
            f"Last status={getattr(response, 'status', None)!r}, "
            f"incomplete_reason={_incomplete_reason(response)!r}, "
            f"followups_used={followups_used}."
        )

    print(f"[DEBUG] Done. followups_used={followups_used}.", flush=True)
    return full_answer, last_response_id


def _usage_exit() -> None:
    print("Usage: python3 editoi.py FILE1 [FILE2 ... FILE10]")
    print("Examples:")
    print("  python3 editoi.py manuscript.docx")
    print("  python3 editoi.py mts.txt staccato.json")
    print("  python3 editoi.py Edit2.docx staccato.csv notes.txt")
    sys.exit(1)


def _print_command_help() -> None:
    print("Commands:")
    print("  <prompt>          Run a new independent analysis using the cached manuscript.")
    print("  /uusi <prompt>    Same as the default: start a new independent analysis.")
    print("  /jatka <prompt>   Continue the most recent response chain.")
    print("  /status           Show model and local prompt-cache key.")
    print("  /help             Show these commands.")
    print("  Ctrl-C            Exit.\n")


def main() -> None:
    if len(sys.argv) < 2:
        _usage_exit()

    paths = sys.argv[1:]
    if len(paths) > MAX_INPUT_FILES:
        print(f"Error: Too many files. Max is {MAX_INPUT_FILES}, got {len(paths)}.")
        sys.exit(1)

    for path in paths:
        if not os.path.exists(path):
            print(f"Error: File not found: {path}")
            sys.exit(1)

    print(
        f"Dependencies OK: openai {OPENAI_SDK_VERSION}, "
        f"python-docx {PYTHON_DOCX_VERSION}."
    )
    print("Loading input files:")
    for path in paths:
        print(f"  - {path}")

    try:
        book_text = build_multi_file_payload(paths)
    except Exception as error:
        print(f"Error loading files: {error}")
        sys.exit(1)

    static_material = build_static_material(book_text)
    cache_key = make_prompt_cache_key(EDITOR_INSTRUCTIONS, static_material)

    print("All files loaded.")
    print(f"Combined manuscript payload length: {len(book_text)} characters.")
    print(f"Stable cache prefix length: {len(static_material)} characters.")
    print(f"Local prompt_cache_key: {cache_key}")
    print(
        "GPT-5.6 uses an explicit cache breakpoint after all input files "
        f"with minimum TTL {PROMPT_CACHE_TTL}."
    )
    print(
        "Each ordinary prompt starts a fresh analysis, so previous analyses do not "
        "inflate or influence the next one. Use /jatka only for a real follow-up.\n"
    )
    _print_command_help()

    last_response_id: Optional[str] = None

    try:
        while True:
            raw_question = clean_text(input("Kysymys> ").strip())
            if not raw_question:
                continue

            if raw_question == "/help":
                _print_command_help()
                continue

            if raw_question == "/status":
                print(f"Active model: {ACTIVE_MODEL or '(selected on first real request)'}")
                print(f"OpenAI SDK: {OPENAI_SDK_VERSION}")
                print(f"python-docx: {PYTHON_DOCX_VERSION}")
                print(f"Prompt cache key: {cache_key}")
                print(f"Explicit cache TTL: {PROMPT_CACHE_TTL}\n")
                continue

            continue_prefix = "/jatka "
            new_prefix = "/uusi "

            if raw_question.startswith(continue_prefix):
                question = raw_question[len(continue_prefix):].strip()
                if not question:
                    print("Anna /jatka-komennon jälkeen varsinainen kysymys.\n")
                    continue
                if not last_response_id:
                    print("Jatkettavaa vastausta ei vielä ole. Aja ensin tavallinen analyysi.\n")
                    continue

                print("\n[SENDING FOLLOW-UP TO THE MOST RECENT RESPONSE]\n")
                answer, last_response_id = ask_question(
                    static_material,
                    question,
                    cache_key,
                    previous_response_id=last_response_id,
                )
            else:
                question = raw_question
                if raw_question.startswith(new_prefix):
                    question = raw_question[len(new_prefix):].strip()
                    if not question:
                        print("Anna /uusi-komennon jälkeen varsinainen analyysiprompti.\n")
                        continue

                print("\n[SENDING NEW INDEPENDENT ANALYSIS WITH CACHED PREFIX]\n")
                answer, last_response_id = ask_question(
                    static_material,
                    question,
                    cache_key,
                    previous_response_id=None,
                )

            print("--- RESPONSE ---\n")
            print(answer)
            print("\n" + "-" * 80 + "\n")

    except KeyboardInterrupt:
        print("\n\nCtrl-C received. Exiting. Bye!")


if __name__ == "__main__":
    main()
