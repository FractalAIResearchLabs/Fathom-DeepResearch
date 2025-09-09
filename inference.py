#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ask_search.py — Single-question DeepSearch runner with pretty terminal output
============================================================================

- Supports the same agent families as your eval harness:
    * fathom-search, ii-search, jan-nano  (via ReCall)
    * zerosearch
    * r1-searcher
    * search-o1

- Clean CLI:
    --agent, --question (or interactive prompt), --model-url, --executors,
    --tokenizer, --search-preset (legacy|fathom), --temperature, --max-new-tokens

- Output:
    A well-formatted terminal report showing the agent, settings, the question,
    an extracted "Final Answer", optional tool-call summary, and the transcript.

Dependencies
------------
- Python 3.10+
- Your agent wrappers on PYTHONPATH (same as eval harness):
    from agents import ReCall
    from agents import ZeroSearchInference, ZeroSearchConfig
    from agents import R1Searcher, R1SearchConfig as R1Cfg
    from agents import O1Searcher, O1Cfg
- Optional:
    transformers (if you pass --tokenizer)
    rich (for better terminal formatting; falls back to plain if unavailable)
"""


"""
# Fathom-Search with ReCall preset & two executors
python ask_search.py \
  --agent fathom-search \
  --question "Who won the 2024 Nobel Prize in Physics?" \
  --executors http://0.0.0.0:1240 \
  --model-url http://0.0.0.0:1254 \
  --search-preset fathom

# Jan-Nano via ReCall with tokenizer passthrough
python ask_search.py \
  --agent jan-nano \
  --question "Explain CRISPR-Cas9 in two sentences." \
  --executors http://0.0.0.0:1240 \
  --model-url http://0.0.0.0:1254 \
  --tokenizer /path/to/Qwen3-4B \
  --search-preset legacy

# II-search via ReCall with tokenizer passthrough
python ask_search.py \
  --agent ii-search \
  --question "Explain CRISPR-Cas9 in two sentences." \
  --executors http://0.0.0.0:1240 \
  --model-url http://0.0.0.0:1254 \
  --tokenizer /path/to/Qwen3-4B \
  --search-preset legacy

# ZeroSearch
python ask_search.py \
  --agent zerosearch \
  --question "What is the capital of Bhutan?" \
  --model-url http://0.0.0.0:1254

# R1-Searcher
python ask_search.py \
  --agent r1-searcher \
  --question "State Hooke’s law in words." \
  --model-url http://0.0.0.0:1254

# search-o1 (o1-style search wrapper)
python ask_search.py \
  --agent search-o1 \
  --question "Summarize the latest SpaceX Starship launch result." \
  --model-url http://0.0.0.0:1254

"""

from __future__ import annotations

import argparse
import os
import random
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

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


# ──────────────────────────────────────────────────────────────────────────────
# Small helpers: normalization + extraction
# ──────────────────────────────────────────────────────────────────────────────

def normalize(s: str) -> str:
    return (s or "").strip().lower()

_ANS_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.S)

def extract_answer_tagged(text: str) -> str:
    """
    Extract the last <answer>...</answer> block (R1-Searcher, ZeroSearch).
    If none, return last 200 chars normalized.
    """
    matches = _ANS_TAG_RE.findall(text or "")
    if matches:
        return normalize(matches[-1])
    return normalize((text or "")[-200:])

def _boxed_last_span(s: str) -> Optional[str]:
    if s is None:
        return None
    idx = s.rfind("\\boxed")
    if "\\boxed " in s:
        # content after the space until a '$' if present
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
    """
    Extract the content inside the *last* \\boxed{...} (or \\fbox{...}).
    If none, return last 200 chars normalized.
    """
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


# ──────────────────────────────────────────────────────────────────────────────
# Agent interface + adapters
# ──────────────────────────────────────────────────────────────────────────────

class BaseAgent:
    def run(self, *args, **kwargs) -> Tuple[str, Any]:
        raise NotImplementedError

def load_tokenizer(tokenizer_path: Optional[str] = None):
    if not tokenizer_path:
        return None
    if AutoTokenizer is None:
        raise RuntimeError("transformers not installed; cannot load tokenizer. pip install transformers")
    return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

class ReCallAdapter(BaseAgent):
    def __init__(self, executor_urls: List[str]):
        from agents import ReCall  # type: ignore
        if not executor_urls:
            raise ValueError("ReCall requires at least one --executors URL")
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

class ZeroSearchAdapter(BaseAgent):
    def __init__(self, thinker_url: Optional[str]):
        from agents import ZeroSearchInference, ZeroSearchConfig  # type: ignore
        cfg = ZeroSearchConfig(thinker_url=thinker_url)
        self._agent = ZeroSearchInference(cfg)

    def run(self, question: str, tokenizer) -> Tuple[str, Any]:
        return self._agent.run(question, tokenizer=tokenizer)

class R1SearcherAdapter(BaseAgent):
    def __init__(self, model_url: Optional[str]):
        from agents import R1Searcher, R1SearchConfig as R1Cfg  # type: ignore
        cfg = R1Cfg(serper_api_key=os.getenv("SERPER_API_KEY", ""))
        self._agent = R1Searcher(cfg=cfg, model_url=model_url)

    def run(self, question: str, tokenizer) -> Tuple[str, Any]:
        return self._agent.run(question, tokenizer=tokenizer)

class O1SearcherAdapter(BaseAgent):
    def __init__(self, model_url: Optional[str]):
        from agents import O1Searcher, O1Cfg  # type: ignore
        cfg = O1Cfg()
        self._agent = O1Searcher(cfg, thinker_url=model_url)

    def run(self, question: str, tokenizer) -> Tuple[str, Any]:
        return self._agent.run(question, tokenizer=tokenizer)

def build_agent(kind: str, model_url: Optional[str], executors: List[str]) -> BaseAgent:
    k = (kind or "").lower()
    if k in {"fathom-search", "ii-search", "jan-nano"}:
        return ReCallAdapter(executor_urls=executors)
    if k == "zerosearch":
        return ZeroSearchAdapter(thinker_url=model_url)
    if k == "r1-searcher":
        return R1SearcherAdapter(model_url=model_url)
    if k == "search-o1":
        return O1SearcherAdapter(model_url=model_url)
    raise ValueError(f"Unknown agent kind: {kind}")


# ──────────────────────────────────────────────────────────────────────────────
# ReCall tool presets (same as your harness)
# ──────────────────────────────────────────────────────────────────────────────

RECALL_PRESETS: Dict[str, Tuple[str, List[Dict[str, Any]]]] = {
    "legacy": (
        "from search_api import web_search, web_visit",
        [
            {
                "name": "web_search",
                "description": "Google search and return links to web-pages with a brief snippet given a text query",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
            {
                "name": "web_visit",
                "description": "Visit webpage and return its content",
                "parameters": {
                    "type": "object",
                    "properties": {"url": {"type": "string"}},
                    "required": ["url"],
                },
            },
        ],
    ),
    "fathom": (
        "from search_api import search_urls, query_url",
        [
            {
                "name": "search_urls",
                "description": "Google search and return links to web-pages with a brief snippet given a text query",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "top_k": {"type": "integer", "default": 10},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "query_url",
                "description": "Visit webpage and return evidence based retrieval for the provided goal",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string"},
                        "goal": {"type": "string"},
                    },
                    "required": ["url", "goal"],
                },
            },
        ],
    ),
}


# ──────────────────────────────────────────────────────────────────────────────
# Pretty printing
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ViewConfig:
    use_rich: bool = True
    transcript_max_chars: int = 8000  # clamp transcript in view

def print_report(
    question: str,
    agent_kind: str,
    model_url: Optional[str],
    executors: List[str],
    preset: str,
    extracted_answer: str,
    transcript: str,
    tool_calls: Any,
    tokenizer_info: Optional[str],
    view: ViewConfig,
) -> None:
    use_rich = view.use_rich and HAVE_RICH
    if use_rich:
        console = Console()
        console.print(Rule("[bold]DeepSearch • Single-Question Runner[/bold]"))
        meta = Table(show_header=False, box=None, expand=True, padding=(0,1))
        meta.add_row("Agent", f"[bold]{agent_kind}[/bold]")
        meta.add_row("Model URL", model_url or "—")
        if executors:
            meta.add_row("Executors", ", ".join(executors))
        if agent_kind in {"fathom-search", "ii-search", "jan-nano"}:
            meta.add_row("ReCall Preset", preset)
        if tokenizer_info:
            meta.add_row("Tokenizer", tokenizer_info)
        console.print(Panel(meta, title="Configuration", expand=True))

        console.print(Panel(Text(question), title="Question", expand=True))
        console.print(Panel(Text(extracted_answer or "—"), title="Final Answer (extracted)", expand=True))

        # Tool calls summary (best-effort)
        if tool_calls is not None:
            summary = str(tool_calls)
            if len(summary) > 2000:
                summary = summary[:2000] + " ... [truncated]"
            console.print(Panel(Text(summary), title="Tool Calls (summary)", expand=True))

        # Transcript
        view_text = transcript or ""
        if len(view_text) > view.transcript_max_chars:
            view_text = view_text[:view.transcript_max_chars] + "\n...[truncated]"
        try:
            console.print(Panel(Syntax(view_text, "text", word_wrap=True), title="Transcript", expand=True))
        except Exception:
            console.print(Panel(Text(view_text), title="Transcript", expand=True))
        console.print(Rule())
    else:
        # Plain stdout
        print("=" * 78)
        print("DeepSearch • Single-Question Runner")
        print("=" * 78)
        print(f"Agent        : {agent_kind}")
        print(f"Model URL    : {model_url or '-'}")
        if executors:
            print(f"Executors    : {', '.join(executors)}")
        if agent_kind in {"fathom-search", "ii-search", "jan-nano"}:
            print(f"ReCall Preset: {preset}")
        if tokenizer_info:
            print(f"Tokenizer    : {tokenizer_info}")
        print("-" * 78)
        print("Question:")
        print(question)
        print("-" * 78)
        print("Final Answer (extracted):")
        print(extracted_answer or "—")
        print("-" * 78)
        if tool_calls is not None:
            summary = str(tool_calls)
            if len(summary) > 2000:
                summary = summary[:2000] + " ... [truncated]"
            print("Tool Calls (summary):")
            print(summary)
            print("-" * 78)
        view_text = transcript or ""
        if len(view_text) > view.transcript_max_chars:
            view_text = view_text[:view.transcript_max_chars] + "\n...[truncated]"
        print("Transcript:")
        print(view_text)
        print("=" * 78)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ask a single question with a chosen DeepSearch agent.")
    parser.add_argument("--agent", required=True, choices=[
        "fathom-search", "ii-search", "jan-nano", "zerosearch", "r1-searcher", "search-o1"
    ])
    parser.add_argument("--question", help="Question to ask (if absent, will prompt interactively).")
    parser.add_argument("--model-url", help="Model server URL (used by many agents).")
    parser.add_argument("--executors", default="", help="Comma-separated ReCall executor URLs (for ReCall-based agents).")
    parser.add_argument("--tokenizer", default=None, help="Optional HF tokenizer/base ckpt path to pass to the agent.")
    parser.add_argument("--search-preset", choices=list(RECALL_PRESETS.keys()), default="fathom",
                        help="ReCall tool preset for fathom/ii/jan agents.")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max-new-tokens", type=int, default=40960)
    parser.add_argument("--no-color", action="store_true", help="Force plain output (no Rich).")

    args = parser.parse_args()

    question = args.question.strip() if args.question else ""
    if not question:
        try:
            question = input("Enter your question: ").strip()
        except EOFError:
            pass
    if not question:
        print("No question provided.", file=sys.stderr)
        sys.exit(2)

    executors = [u.strip() for u in (args.executors or "").split(",") if u.strip()]
    agent = build_agent(args.agent, args.model_url, executors)

    # Tokenizer (optional)
    tok = None
    tok_info = None
    if args.tokenizer:
        tok = load_tokenizer(args.tokenizer)
        tok_info = args.tokenizer

    # Build ReCall env/schemas if needed
    recall_env, recall_schemas = RECALL_PRESETS.get(args.search_preset, RECALL_PRESETS["fathom"])

    # Dispatch per agent family
    if args.agent in {"fathom-search", "ii-search", "jan-nano"}:
        transcript, tool_calls = agent.run(
            env=recall_env,
            func_schemas=recall_schemas,
            question=question,
            model_url=args.model_url,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            tokenizer=tok,
        )
        # Boxed extraction is the default for these families
        extracted = extract_answer_boxed(transcript or "")
    elif args.agent in {"r1-searcher", "zerosearch"}:
        transcript, tool_calls = agent.run(question=question, tokenizer=tok)
        extracted = extract_answer_tagged(transcript or "")
    else:  # search-o1
        transcript, tool_calls = agent.run(question=question, tokenizer=tok)
        # Many o1-style configs also use <answer> tags; fall back to boxed if none.
        extracted = extract_answer_tagged(transcript or "")
        if not extracted or extracted == normalize((transcript or "")[-200:]):
            extracted = extract_answer_boxed(transcript or "")

    view = ViewConfig(use_rich=not args.no_color)
    print_report(
        question=question,
        agent_kind=args.agent,
        model_url=args.model_url,
        executors=executors,
        preset=args.search_preset,
        extracted_answer=extracted,
        transcript=transcript or "",
        tool_calls=tool_calls,
        tokenizer_info=tok_info,
        view=view,
    )


if __name__ == "__main__":
    main()
