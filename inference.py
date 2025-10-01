
"""
inference.py — Minimal Fathom-Search-4B single-question runner
===============================================================

- Only supports Fathom-Search (ReCall-based)
- Adds --deepresearch flag to post-process the full trace with a Summary LLM
  (OpenAI model id or a local vLLM /generate endpoint)
- Imports system prompts from prompt.py

Env:
  SUMMARY_LLM     (default: "openai:gpt-4.1-mini")  # e.g., "openai:gpt-4.1-mini" or host an sglang server with the desired model on port XXXX and pass "http
  OPENAI_API_KEY  (if using OpenAI backend)

CLI:
  --question, --model-url, --executors, --tokenizer (optional),
  --temperature, --max-new-tokens, --no-color,
  --deepresearch (bool),
  --summary-llm (optional override),
  --summary-temperature, --summary-max-tokens
"""

#  ./scripts/launch_inference_backend.sh fathom-search-4B /data/home/fractal/shreyas/models/models/stage1-rapo-210 1254 1255

import argparse
import os
import random
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# Prompts (ensure prompt.py is in PYTHONPATH or same dir)
from prompts import DEEPRESEARCH_SYS_PROMPT  # type: ignore

# Optional: HF tokenizer passthrough
try:
    from transformers import AutoTokenizer  # type: ignore
except Exception:
    AutoTokenizer = None  # type: ignore

# Optional: Rich for pretty output
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.rule import Rule
    from rich.syntax import Syntax
    HAVE_RICH = True
except Exception:
    HAVE_RICH = False

# HTTP for vLLM route
try:
    import requests  # type: ignore
except Exception:
    requests = None  # graceful error later if needed


# ──────────────────────────────────────────────────────────────
# Helpers: normalization + boxed answer extraction
# ──────────────────────────────────────────────────────────────

def normalize(s: str) -> str:
    return (s or "").strip().lower()


def _boxed_last_span(s: str) -> Optional[str]:
    if s is None:
        return None
    idx = s.rfind("\\boxed")
    if "\\boxed " in s:
        return "\\boxed " + s.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = s.rfind("\\fbox")
        if idx < 0:
            return None
    i = idx
    depth = 0
    right = None
    while i < len(s):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                right = i
                break
        i += 1
    return s[idx:right + 1] if right is not None else None


def extract_answer_boxed(text: str) -> str:
    try:
        span = _boxed_last_span(text or "")
        if not span:
            return normalize((text or "")[-200:])
        if span.startswith("\\boxed "):
            return normalize(span[len("\\boxed "):])
        left = "\\boxed{"
        if not (span.startswith(left) and span.endswith("}")):
            return normalize((text or "")[-200:])
        return normalize(span[len(left):-1])
    except Exception:
        return normalize((text or "")[-200:])


# ──────────────────────────────────────────────────────────────
# Fathom-Search Agent Adapter (ReCall)
# ──────────────────────────────────────────────────────────────

class FathomSearchAdapter:
    def __init__(self, executor_urls: List[str]):
        from agents import ReCall  # type: ignore
        if not executor_urls:
            raise ValueError("Fathom-Search requires at least one --executors URL")
        self._ReCall = ReCall
        self._executor_urls = list(executor_urls)

    def _pick(self) -> str:
        return random.choice(self._executor_urls)

    def run(
        self,
        env: str,
        func_schemas: List[Dict[str, Any]],
        question: str,
        model_url: Optional[str],
        temperature: float,
        max_new_tokens: int,
        tokenizer: Any,
    ) -> Tuple[str, Any]:
        agent = self._ReCall(executor_url=self._pick())
        return agent.run(
            env=env,
            func_schemas=func_schemas,
            question=question,
            model_url=model_url,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            tokenizer=tokenizer,
        )


# ──────────────────────────────────────────────────────────────
# ReCall tool preset (Fathom only)
# ──────────────────────────────────────────────────────────────

RECALL_ENV = "from search_api import search_urls, query_url"
RECALL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "name": "search_urls",
        "description": "Google search and return links to web-pages with a brief snippet given a text query",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}, "top_k": {"type": "integer", "default": 10}},
            "required": ["query"],
        },
    },
    {
        "name": "query_url",
        "description": "Visit webpage and return evidence based retrieval for the provided goal",
        "parameters": {
            "type": "object",
            "properties": {"url": {"type": "string"}, "goal": {"type": "string"}},
            "required": ["url", "goal"],
        },
    },
]


# ──────────────────────────────────────────────────────────────
# Summary LLM backends (OpenAI or vLLM)
# ──────────────────────────────────────────────────────────────

def _openai_client():
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        raise RuntimeError("openai package not installed. `pip install openai`") from e
    return OpenAI()

def chatml_wrap(system_prompt: str, user_prompt: str) -> str:
    return (
        f"<|im_start|>system\n{system_prompt}\n<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


# --- sglang / vLLM-style endpoint ---
def _call_sglang(base_url: str, system_prompt: str, user_prompt: str, *,
                 temperature: float, max_tokens: int,
                 stop: Optional[List[str]] = None, timeout: int = 400) -> str:
    """
    Call an sglang server that exposes POST {base_url}/generate
    with {"text": <prompt>, "sampling_params": {...}} and return the first text.
    """
    if requests is None:
        raise RuntimeError("requests not installed. `pip install requests`")

    # simple 2-turn format; keep exactly what your backend expects
    # merged = f"[SYSTEM]\n{system_prompt}\n\n[USER]\n{user_prompt}"
    # user_prompt = reformat_trace(user_prompt)
    merged = chatml_wrap(system_prompt, user_prompt)


    payload = {
        "text": merged,
        "sampling_params": {
            "temperature": float(temperature),
            "max_new_tokens": int(max_tokens),
            "repetition_penalty": 1.05,
        },
    }
    if stop:
        payload["sampling_params"]["stop"] = stop

    resp = requests.post(f"{base_url.rstrip('/')}/generate", json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    # print("data", data)

    # sglang/vLLM usually returns {"text": "..."} or {"text": ["...", ...]}
    txt = data.get("text")
    # print("resp", txt)
    if isinstance(txt, list):
        return txt[0]
    if isinstance(txt, str):
        return txt
    raise ValueError(f"Unexpected /generate response: {data!r}")

import re


def reformat_trace(s: str) -> str:
    """Turn ChatML-ish agent transcript into readable plain text."""
    if not s:
        return s
    t = s
    # Remove system prompt block completely (from <|im_start|>system to <|im_end|>)
    t = re.sub(r"<\|im_start\|>system.*?<\|im_end\|>", "", t, flags=re.DOTALL|re.IGNORECASE)
    
    # Replace other speaker tokens with readable labels
    def _speaker(m: re.Match) -> str:
        role = (m.group(1) or "").strip().upper()
        return f"\n{role}:\n"
    t = re.sub(r"<\|im_start\|>(\w+)", _speaker, t, flags=re.IGNORECASE)
    t = re.sub(r"<\|im_end\|>", "\n", t, flags=re.IGNORECASE)
    
    # Remove <think> tags and replace closing with newline
    t = re.sub(r"<think\s*>", "", t, flags=re.IGNORECASE)
    t = re.sub(r"</think\s*>", "\n", t, flags=re.IGNORECASE)
    
    # Replace tool response tags with a clear marker
    t = re.sub(r"<tool_respon[sc]e\s*>", "SEARCH RESULT\n", t, flags=re.IGNORECASE)
    t = re.sub(r"</tool_respon[sc]e\s*>", "\n", t, flags=re.IGNORECASE)
    
    # Remove tool_call tags completely
    t = re.sub(r"</?tool_call\s*>", "", t, flags=re.IGNORECASE)
    
    # Remove any other ChatML tokens (like <|im_start|> and others)
    t = re.sub(r"<\|[^>]+?\|>", "", t)
    
    # Remove any other remaining angle bracket tags (e.g., <something>)
    t = re.sub(r"</?[^>\n]+?>", "", t)
    
    # Clean up multiple blank lines to max two
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    
    return t



def _route_and_summarize(
    summary_llm: str,
    system_prompt: str,
    prompt: str,
    *,
    temperature: float,
    max_tokens: int,
) -> str:
    """
    If `summary_llm` starts with 'http', treat as vLLM base_url; else treat as an OpenAI model id.
    For vLLM, prepend [SYSTEM]/[USER] tags; for OpenAI, pass messages with system+user.
    """
    if summary_llm.strip().lower().startswith("http"):
        return _call_sglang(summary_llm, system_prompt, prompt, temperature=temperature, max_tokens=max_tokens)

    client = _openai_client()
    rsp = client.chat.completions.create(
        model=summary_llm,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=max_tokens,
    )
    return rsp.choices[0].message.content or ""



def build_summary_prompt(question: str, transcript: str, tool_calls: Any) -> str:
    """Assemble the user prompt handed to the summary model."""
    try:
        tool_str = json.dumps(tool_calls, ensure_ascii=False)
    except Exception:
        tool_str = str(tool_calls)
    return (
        "You are given a DeepSearch investigation trace.\n\n"
        f"Question:\n{question}\n\n"
        "Trace (model transcript):\n"
        f"{transcript}\n\n"
        "Tool Calls (as-recorded):\n"
        f"{tool_str}\n\n"
        "— End of trace —"
    )



# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ask a single question with Fathom-Search-4B.")
    parser.add_argument("--question", help="Question to ask (if absent, will prompt interactively).")
    parser.add_argument("--model-url", required=True, help="Model server URL.")
    parser.add_argument("--executors", required=True, help="Comma-separated ReCall executor URLs.")
    parser.add_argument("--tokenizer", default=None, help="Optional HF tokenizer/base ckpt path.")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max-new-tokens", type=int, default=40960)
    parser.add_argument("--no-color", action="store_true", help="Force plain output (no Rich).")
    # Upgrades
    parser.add_argument("--deepresearch", action="store_true",
                        help="If set, produce a DeepResearch-style report with the Summary LLM.")
    parser.add_argument("--summary-llm", default="gpt-4.1-mini",
                        help="Summary LLM backend: OpenAI model (e.g., gpt-4.1-mini) "
                             "or vLLM base URL (e.g., http://0.0.0.0:1255). Defaults to $SUMMARY_LLM or gpt-4.1-mini.")
    parser.add_argument("--summary-temperature", type=float, default=0.4)
    parser.add_argument("--summary-max-tokens", type=int, default=10000)

    args = parser.parse_args()


    question = (args.question or "").strip()
    if not question:
        try:
            question = input("Enter your question: ").strip()
        except EOFError:
            pass
    if not question:
        print("No question provided.", file=sys.stderr)
        sys.exit(2)

    executors = [u.strip() for u in args.executors.split(",") if u.strip()]
    agent = FathomSearchAdapter(executor_urls=executors)

    tok = None
    tok_info = None
    if args.tokenizer:
        if AutoTokenizer is None:
            raise RuntimeError("transformers not installed; `pip install transformers`")
        tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
        tok_info = args.tokenizer

    # Run Fathom-Search
    transcript, tool_calls = agent.run(
        env=RECALL_ENV,
        func_schemas=RECALL_SCHEMAS,
        question=question,
        model_url=args.model_url,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        tokenizer=tok,
    )

    extracted = extract_answer_boxed(transcript or "")

    # Optional: Summary/DeepResearch pass
    summary_text: Optional[str] = None
    # try:

    prompt = build_summary_prompt(question, reformat_trace(transcript) or "", tool_calls)
    if args.deepresearch:
        print("Generating Report ........")
        system_prompt = DEEPRESEARCH_REPORT_SYS_PROMPT 
        try:
            resp = _route_and_summarize(
                summary_llm=args.summary_llm,
                system_prompt=system_prompt,
                prompt=prompt,
                temperature=args.summary_temperature,
                max_tokens=args.summary_max_tokens,
                )
            
            report = re.split(r"</think\s*>", resp, flags=re.IGNORECASE)[-1]
            plan = re.split(r"</think\s*>", resp, flags=re.IGNORECASE)[0]  

        except Exception as e:
            report = f"[Summary LLM error: {e}]"
            plan = f"[Summary LLM error: {e}]"
        print("="*75)
        print("REPORT")
        print(report)
      


if __name__ == "__main__":
    main()




