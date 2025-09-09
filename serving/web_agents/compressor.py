# compressor.py
from __future__ import annotations
import functools, json, logging, os, re
from difflib import SequenceMatcher
from io import StringIO
from typing import Dict, List, Tuple, Union

import pandas as pd
import regex  # needed by tiktoken
import tiktoken
from bs4 import BeautifulSoup
from config import CFG
from web_helpers import retry
import requests

# ────────────────────────────────────────────────────────────────────────
# 0. shared helpers
# ------------------------------------------------------------------------
enc = tiktoken.get_encoding("cl100k_base")
_tok = lambda s: len(enc.encode(s))  # fast inline counter


@functools.lru_cache(maxsize=1)
def _nlp():
    import spacy
    return spacy.load("en_core_web_sm")


def _openai_client():
    """Import OpenAI lazily to avoid overhead when not needed."""
    import importlib
    mod = importlib.import_module("openai")
    return getattr(mod, "OpenAI", None)() if hasattr(mod, "OpenAI") else mod


# ────────────────────────────────────────────────────────────────────────
# Together helpers (SDK first, requests fallback)
# ------------------------------------------------------------------------
def _together_api_key() -> str:
    key = os.getenv("TOGETHER_API_KEY")
    if not key:
        raise RuntimeError("TOGETHER_API_KEY environment variable is not set.")
    return key


def _together_client():
    """Return a Together SDK client if available; else None (we’ll fallback to requests)."""
    try:
        import importlib
        mod = importlib.import_module("together")
        # SDK exposes class `Together`
        return getattr(mod, "Together")(_together_api_key())
        print("imported together")
    except Exception:
        return None


# ────────────────────────────────────────────────────────────────────────
# 1. regex patterns (compiled once)
# ------------------------------------------------------------------------
DATE_PATS = [re.compile(p, re.I) for p in [
    r"\d{4}-\d{2}-\d{2}",
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},\s+\d{4}",
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}",
    r"\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}",
    r"\b\d{4}/\d{2}\b",
    r"\b\d{4}\b(?!\s*(?:%|million|billion|thousand))",
]]
EMAIL_PAT = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
URL_PAT = re.compile(r"https?://[^\s\)]+")
PHONE_PAT = re.compile(r"\+?\d[\d\s\-().]{7,}\d")
CURR_PAT = re.compile(r"(\$\s?\d+(?:,\d{3})*(?:\.\d+)?|\d+(?:,\d{3})*(?:\.\d+)?\s*(USD|EUR|GBP|INR|¥|₩|₹|€))", re.I)
DEF_PAT = re.compile(r"([A-Z][A-Za-z0-9\s]+?)\s+(is|are|refers to|means)\s+(.*?)(?:[\.\n])")

MD_TABLE_PAT = re.compile(r"(?:^\|.*?\|\n?)+(?:^\|[-:\s|]+\|\n?)?(?:^\|.*?\|\n?)+", re.M)
CSV_PAT = re.compile(r"((?:^.*?,.*?\n){2,})", re.M)
TSV_PAT = re.compile(r"((?:^.*?\t.*?\n){2,})", re.M)


# ────────────────────────────────────────────────────────────────────────
# helper: model routing detectors
# ------------------------------------------------------------------------
def _is_openai_model(model_name: str) -> bool:
    """Heuristic: treat strings that begin with 'gpt-' as OpenAI model names."""
    return model_name.lower().startswith("openai:")


def _is_together_model(model_name: str) -> bool:
    """
    Treat strings that begin with 'together:' as Together model names.
    Example: 'together:meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo'
    """
    return model_name.lower().startswith("together:")


# ────────────────────────────────────────────────────────────────────────
# 2. core utilities
# ------------------------------------------------------------------------
def deduplicate_items(items: List[str], *, similarity=0.5,
                      other: List[str] | None = None) -> List[str]:
    """Drop near-duplicates; prefer the longest variant."""
    if not items:
        return []
    other = other or []

    def _clean(x: str) -> str:
        x = re.sub(r'\[edit\]|\[\d+\]', '', x)
        return re.sub(r'\s+', ' ', x).strip()

    out, out_clean = [], []
    for orig in items:
        clean = _clean(orig)
        dup = False
        for ref in out_clean + list(map(_clean, other)):
            sim = SequenceMatcher(None, clean, ref).ratio()
            if sim >= similarity or clean in ref or ref in clean:
                dup = True
                # if current is longer than stored, replace
                if clean not in out_clean and len(clean) > len(ref):
                    idx = out_clean.index(ref)
                    out[idx], out_clean[idx] = orig, clean
                break
        if not dup:
            out.append(orig)
            out_clean.append(clean)
    return out


# ────────────────────────────────────────────────────────────────────────
# 3. fact & table extractor
# ------------------------------------------------------------------------
def extract_facts_and_tables(text: str) -> Tuple[str, List[str], List[str]]:
    facts, spans = [], []

    def _add(match):
        facts.append(match.group())
        spans.append(match.span())

    for pat in DATE_PATS:
        [_add(m) for m in pat.finditer(text)]
    for m in EMAIL_PAT.finditer(text):
        _add(m)
    for m in URL_PAT.finditer(text):
        _add(m)
    for m in PHONE_PAT.finditer(text):
        _add(m)
    for m in CURR_PAT.finditer(text):
        _add(m)
    for m in DEF_PAT.finditer(text):
        _add(m)

    # contextual sentences around facts
    doc = _nlp()(text)
    ctx = [s.text.strip() for s in doc.sents
           if any(s.start_char <= s_ <= s.end_char for s_, _ in spans)]
    facts.extend(ctx)
    facts = sorted(set(facts))

    # ── tables
    tables: List[str] = []

    for tbl in MD_TABLE_PAT.findall(text):
        cleaned = "\n".join(l for l in tbl.splitlines()
                            if l.strip() and not re.match(r"^\|[-:\s|]+\|$", l))
        if len(cleaned.splitlines()) < 2:
            continue
        try:
            df = pd.read_csv(StringIO(cleaned), sep="|").dropna(how="all", axis=1)
            tables.append(df.to_markdown(index=False))
        except Exception:
            tables.append(cleaned)

    soup = BeautifulSoup(text, "lxml")
    for html_tbl in soup.find_all("table"):
        try:
            df = pd.read_html(str(html_tbl))[0]
            tables.append(df.to_markdown(index=False))
        except Exception:
            tables.append(str(html_tbl))

    for m in CSV_PAT.finditer(text):
        try:
            df = pd.read_csv(StringIO(m.group(1)))
            if not df.empty:
                tables.append(df.to_markdown(index=False))
        except Exception:
            pass
    for m in TSV_PAT.finditer(text):
        try:
            df = pd.read_csv(StringIO(m.group(1)), sep="\t")
            if not df.empty:
                tables.append(df.to_markdown(index=False))
        except Exception:
            pass

    # ── clean narrative (remove facts & tables)
    narrative = text
    for tbl in tables:
        narrative = narrative.replace(tbl, " ")
    for s, e in sorted(spans, reverse=True):
        narrative = narrative[:s] + narrative[e:]
    narrative = re.sub(r"\s{2,}", " ", narrative).strip()

    return narrative, facts, tables


# ────────────────────────────────────────────────────────────────────────
# 4. OpenAI & Together & vLLM summariser helpers
# ------------------------------------------------------------------------
def _summarise(text: str, pct: float, model: str) -> str:
    target_tokens = int(_tok(text) * pct)
    sys_prompt = (
        "You are an expert abstractor. Summarize the text below to "
        f"approximately {pct*100:.0f}% of its original length (≈{target_tokens} tokens), "
        "while **retaining all key facts, figures, names, dates, places, and events**. "
        "Ensure the summary remains accurate, informative, and faithful to the original content."
    )
    if _is_openai_model(model):
        client = _openai_client()
        rsp = client.chat.completions.create(
            model=model, temperature=0.2,
            messages=[{"role": "system", "content": sys_prompt},
                      {"role": "user", "content": text}],
            max_tokens=CFG.output_limit_per_link
        )
        return rsp.choices[0].message.content
    elif _is_together_model(model):
        return _call_together_chat(
            model, sys_prompt, text,
            temperature=0.2,
            max_tokens=CFG.output_limit_per_link
        )
    else:
        # Treat as sglang/vLLM base URL (non-chat); inline system+user into single prompt
        merged = sys_prompt + "\n\n" + text
        return _call_sglang(
            model, merged,
            temperature=0.2,
            max_tokens=CFG.output_limit_per_link
        )


# ────────────────────────────────────────────────────────────────────────
# 5. compress_text  (public)
# ------------------------------------------------------------------------
def compress_text(text: str, *, pct: float = 0.3,
                  model: str = "gpt-4o-mini") -> str:
    FACTS_TABLES_LIMIT = CFG.output_limit_per_link - CFG.disable_narrative_compress_thresh
    narrative, facts, tables = extract_facts_and_tables(text)

    # narrative compression
    if _tok(narrative) > CFG.disable_narrative_compress_thresh:
        narrative_txt = _summarise(narrative, pct, model)
    else:
        narrative_txt = narrative
    return narrative_txt


# ────────────────────────────────────────────────────────────────────────
# 6. query_text  (goal-oriented extraction)
# ------------------------------------------------------------------------
EXTRACTOR_SYS_PROMPT = (
    "You are a highly skilled information extraction agent. Your job is to analyze long, complex webpages "
    "in the context of a specific user goal. You excel at identifying relevant sections, capturing supporting evidence "
    "in full original context, and providing logically structured summaries. Always ensure precision, completeness, "
    "and alignment with the user’s intent."
)
EXTRACTOR_PROMPT_TEMPLATE = """You are a highly skilled information extraction agent. Your task is to analyze the following webpage content in light of a specific user goal, and extract accurate, well-structured information using plain text format.

## Webpage Content
{webpage_content}

## User Goal
{goal}

## Task Guidelines
1. **Rational**: Briefly explain why this content is relevant to the user’s goal.
2. **Evidence**: Quote the most relevant parts of the webpage that directly support or address the goal. Use bullet points or numbered lines separated by newlines.
3. **Summary**: Provide a clear, logically structured summary of the extracted evidence that addresses the user's goal.

## Output Format
Your response must follow **exactly this format** with the three sections:
Rational: <one paragraph>
Evidence: <first point>\n<second point>...
Summary:<concise paragraph summarizing the evidence>
"""


def extract_regex(text: str) -> Dict[str, str]:
    def extract_section(header: str) -> str:
        # Match the section starting with `Header:` until the next capitalized line followed by `:` or end
        pattern = rf"{header}:\s*(.*?)(?=\n[A-Z][a-z]+:|\Z)"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else "(not found)"

    return {
        "rational": extract_section("Rational"),
        "evidence": extract_section("Evidence"),
        "summary": extract_section("Summary")
    }


def _call_openai(model: str, prompt: str, *, temperature: float,
                 max_tokens: int) -> str:
    """One-shot call to the OpenAI chat endpoint; returns raw text."""
    model_id = model.split(":", 1)[1] if ":" in model else model
    client = _openai_client()
    rsp = client.chat.completions.create(
        model=model_id,
        temperature=temperature,
        messages=[
            {"role": "system", "content": EXTRACTOR_SYS_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
    )
    return rsp.choices[0].message.content


def _call_sglang(base_url: str, prompt: str, *, temperature: float,
               max_tokens: int, stop: List[str] | None = None) -> str:
    """
    Call a vLLM REST endpoint that exposes POST {base_url}/generate.
    Returns the generated text (1st candidate).
    """
    payload = {
        "text": prompt,
        "sampling_params": {
            "temperature": temperature,
            "max_new_tokens": max_tokens,
            "repetition_penalty": 1.05
        },
    }
    if stop:  # only include if provided
        payload["sampling_params"]["stop"] = stop

    resp = requests.post(
        f"{base_url.rstrip('/')}/generate",
        json=payload,
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()

    # vLLM returns {"text": "..."}  or  {"text": ["...", "..."]}
    if isinstance(data, dict) and "text" in data:
        txt = data["text"]
        if isinstance(txt, list):
            return txt[0]
        return txt

    raise ValueError(f"Unexpected vLLM response shape: {data!r}")


def _call_together_chat(model: str, user: str, *,
                        temperature: float, max_tokens: int,
                        stop: List[str] | None = None) -> str:
    """
    Together chat call using the SDK when available, otherwise raw requests.
    `model` may be prefixed with 'together:' and will be stripped for the API.
    """
    model_id = model.split(":", 1)[1] if ":" in model else model
    print(model_id)
    print("+"*100)
    # Prefer SDK (matches your snippet)
    client = _together_client()
    if client is not None:
        print("ITHE"*10)
        resp = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": EXTRACTOR_SYS_PROMPT},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            top_p=1.0,
            max_tokens=max_tokens,
            stop=stop,
        )
        try:
            return resp.choices[0].message.content
        except Exception:
            raise ValueError(f"Unexpected Together SDK response shape: {resp!r}")

    # Fallback: raw HTTP (if SDK not installed)
    headers = {
        "Authorization": f"Bearer {_together_api_key()}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": EXTRACTOR_SYS_PROMPT},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "top_p": 1.0,
        "max_tokens": max_tokens,
        "stop": stop,
    }
    resp = requests.post(
        "https://api.together.xyz/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        raise ValueError(f"Unexpected Together response shape: {data!r}")


def query_text(
    url: str,
    text: str,
    goal: str,
    *,
    model: str = "gpt-4.1-mini",
    max_attempts: int = 3,
    temperature=0,
    max_tokens=1024
) -> Dict[str, str]:
    """Goal-oriented extractor with retries → compress fallback → token trim fallback."""
    prompt = EXTRACTOR_PROMPT_TEMPLATE.format(
        webpage_content=text[:15_000],  # clip for safety
        goal=goal,
    )

    for attempt in range(1, max_attempts + 1):
        try:
            if _is_openai_model(model):
                print("=" * 100)
                print(f"using openai backend for q'uerying {url} with the goal '{goal}'")
                print("=" * 100)
                rsp_text = _call_openai(
                    model, prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            elif _is_together_model(model):
                print("=" * 100)
                print(f"using together backend for querying {url} with the goal {goal}")
                print("=" * 100)
                rsp_text = _call_together_chat(
                    model,
                    prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            else:
                print("=" * 100)
                print(f"using sglang backend for querying {url} with the goal '{goal}'")
                print("=" * 100)
                rsp_text = _call_sglang(
                    model,  # here `model` is the base URL
                    prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

            extracted = extract_regex(rsp_text)

            # Sanity check: evidence + summary must be > 20 characters
            if len(extracted.get("evidence", "")) + len(extracted.get("summary", "")) > 20:
                print("Evidence:", extracted.get("evidence", ""))
                print("Summary:", extracted.get("summary", ""))
                return {
                    "extracted_info": (
                        f"The useful information in {url} for goal “{goal}”:\n\n"
                        f"Rationale:\n{extracted.get('rational')}\n\n"
                        f"Evidence:\n{extracted.get('evidence')}\n\n"
                        f"Summary:\n{extracted.get('summary')}"
                    )
                }

            raise ValueError("LLM returned empty or malformed extraction")

        except Exception as e:
            logging.warning("Attempt %d/%d failed for query-based extraction: %s",
                            attempt, max_attempts, e)

    # ── Retry fallback: compress text ─────────────────────────────────────
    try:
        compressed = compress_text(text, model=model)
        return {
            "extracted_info": (
                f"Goal-based extraction failed after {max_attempts} attempts; "
                f"returning compressed webpage:\n\n{compressed}"
            )
        }
    except Exception as ce:
        logging.error("compress_text also failed: %s", ce)

    # ── Final fallback: hard truncate to token budget ────────────────────
    trunc, _ = trim_to_budget(text, CFG.output_limit_per_link, is_table=False)
    return {
        "extracted_info": (
            "Goal-based extraction and compression both failed; "
            "returning truncated webpage content:\n\n" + trunc
        )
    }


# ────────────────────────────────────────────────────────────────────────
# 7. helper: trim long lists to token budget (string or list)
# ------------------------------------------------------------------------
def trim_to_budget(items: Union[str, List[str]], budget: int, *, is_table: bool = False) -> Tuple[str, int]:
    # If a single long string: clip by token budget
    if isinstance(items, str):
        toks = enc.encode(items)
        if len(toks) <= budget:
            return items, len(toks)
        clipped = enc.decode(toks[:budget])
        return clipped + f"\n[truncated to {budget} tokens]", budget

    # If a list of strings: pack items until budget
    build, used = [], 0
    for it in items:
        toks = _tok(it)
        if used + toks > budget:
            break
        build.append(it)
        used += toks
    if len(build) < len(items):
        build.append(f"[{len(items) - len(build)} {'tables' if is_table else 'facts'} omitted]")
    joined = "\n\n".join(build) if is_table else "\n".join(build)
    return joined, _tok(joined)

# python evals/deep_research_pairwise_evals.py \
#   --input-data /data/home/fractal/shreyas/ydc-deep-research-evals/datasets/DeepConsult/responses_OpenAI-DeepResearch_vs_ours.csv \
#   --output-dir /data/home/fractal/shreyas/ydc-deep-research-evals/results \
#   --model o3-mini-2025-01-31 \
#   --num-workers 64 \
#   --metric-num-workers 64 \
#   --metric-num-trials 
