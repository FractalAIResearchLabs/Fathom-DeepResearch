#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_search.py — Unified, well-commented benchmarking harness
==============================================================

Supports multiple agent wrappers (Fathom-Search, II-Search, Jan-Nano, ZeroSearch,
R1-Searcher, search-o1) with single- or multi-threaded execution, robust answer
extraction, OpenAI-based judging, and resumable JSONL outputs.

Key features
------------
- Clean CLI (dataset selection, agent, model URL, workers, limits, etc.)
- Agent factory with optional imports (fails gracefully if a wrapper isn't installed)
- Thread-safe OpenAI client and configurable judge rate-limit via semaphore
- Robust answer extraction: <answer>...</answer> and \boxed{...} helpers
- Deterministic IDs: uses provided `id` or hashes `question`
- Resumable writes: `--resume` to append & skip already evaluated IDs
- Flexible dataset root and output path layout
- Pluggable ReCall tool presets (legacy vs fathom)
- Optional HuggingFace tokenizer pass-through for agents that need it

Requirements
------------
- Python 3.10+
- `openai` for judging (set $OPENAI_API_KEY)
- Agent wrapper modules available on PYTHONPATH:
    * `re_call.ReCall` (used for --agent fathom-search / ii-search / jan-nano)
    * (Optionally) `re_call.ZeroSearchInference`, `re_call.ZeroSearchConfig`
    * (Optionally) `re_call.R1Searcher`, `re_call.R1SearchConfig`
    * (Optionally) `re_call.O1Searcher`, `re_call.O1Cfg`
- `transformers` if you pass --tokenizer to load an HF tokenizer for the agent

Example usage
-------------
Single-threaded:
    python eval_search.py \
        --dataset frames \
        --data-root /path/to/datasets \
        --agent fathom-search \
        --executors http://0.0.0.0:1240,http://0.0.0.0:1241 \
        --model-url http://0.0.0.0:1254 \
        --out-base /tmp/evals \
        --mode single
        --tokenizer 

Multi-threaded (64 workers) with resume:
    python eval_search.py \
        --dataset upsc_2025 \
        --data-root /data/home/fractal/shreyas/eval_datasets \
        --tokenizer /data/home/fractal/shreyas/models/Qwen3-4B \
        --agent jan-nano \
        --executors http://0.0.0.0:1211 \
        --model-url http://0.0.0.0:1254 \
        --out-base /tmp/evals \
        --mode multi \
        --workers 64 \
        --resume 

Output path pattern:
    {out_base}/{agent}/{dataset}-{name}.jsonl

Dataset JSONL format (per line):
    {"id": "...", "question": "...", "answer": "..."}
If `id` is missing, a deterministic SHA256 of the question is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import pathlib
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import unicodedata
from tqdm import tqdm

# --- Optional: transformers tokenizer loading (only if you pass --tokenizer) ---
try:
    from transformers import AutoTokenizer  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    AutoTokenizer = None  # type: ignore

# --- OpenAI judge client ---
try:
    from openai import OpenAI, APIStatusError  # type: ignore
except Exception:
    raise SystemExit("❌ The 'openai' package is required for judging. Install via `pip install openai`.")

# ──────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ──────────────────────────────────────────────────────────────────────────────

def normalize(s: str) -> str:
    return unicodedata.normalize("NFKD", s.strip().lower())


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def parse_limit(limit: Optional[str]) -> Tuple[int, Optional[int]]:
    """
    Parse --limit like "0,2000" or "100" (meaning 0..100).
    Returns (start, end_or_None).
    """
    if not limit:
        return 0, None
    if "," in limit:
        s, e = limit.split(",", 1)
        return int(s.strip()), int(e.strip())
    return 0, int(limit.strip())


# ──────────────────────────────────────────────────────────────────────────────
# Answer extraction utilities
# ──────────────────────────────────────────────────────────────────────────────

_ANS_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.S)

def extract_answer_tagged(text: str) -> str:
    """
    Extract the last <answer>...</answer> block (common for R1-Searcher, ZeroSearch).
    Falls back to the last 200 characters if not found.
    """
    matches = _ANS_TAG_RE.findall(text or "")
    if matches:
        return normalize(matches[-1])
    return normalize((text or "")[-200:])


def _boxed_last_span(s: str) -> Optional[str]:
    """
    Returns the last occurrence of \boxed{...} or \boxed ... (LaTeX style), including braces.
    Also supports \fbox{...} as a fallback.
    """
    if s is None:
        return None
    idx = s.rfind("\\boxed")
    if "\\boxed " in s:
        # E.g., "\boxed 42$ ...", stop at first '$' after it if present
        return "\\boxed " + s.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = s.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    depth = 0
    while i < len(s):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                right_brace_idx = i
                break
        i += 1
    return s[idx:right_brace_idx + 1] if right_brace_idx is not None else None


def extract_answer_boxed(text: str) -> str:
    """
    Extract the content inside the *last* \\boxed{...} (or \\fbox{...}) occurrence.
    If not found, fall back to the last 200 chars.
    """
    try:
        span = _boxed_last_span(text or "")
        if not span:
            return normalize((text or "")[-200:])
        # Normalize two forms: "\boxed " and "\boxed{...}"
        if span.startswith("\\boxed "):
            # content after the space until a terminator ($ or whitespace) was already sliced
            content = span[len("\\boxed "):]
            return normalize(content)
        left = "\\boxed{"
        if not span.startswith(left) or not span.endswith("}"):
            return normalize((text or "")[-200:])
        content = span[len(left):-1]
        return normalize(content)
    except Exception:
        return normalize((text or "")[-200:])


# ──────────────────────────────────────────────────────────────────────────────
# OpenAI judge (thread-safe client + semaphore)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class JudgeConfig:
    model: str = "gpt-4.1-mini"
    temperature: float = 0.0
    max_tokens: int = 64
    concurrency: int = 3  # max parallel judge calls


JUDGE_SYSTEM = """You are an impartial judge evaluating the correctness of a model's answer
against a ground-truth answer for a given question.

Output exactly one word: "correct", "incorrect", or "unknown".

- Treat case and minor formatting differences as irrelevant.
- If it's a multiple-choice question, match by option label (A/B/C/D) where applicable.
- If the model answer is empty or you cannot determine, output "unknown".
"""

def _thread_local_openai() -> OpenAI:
    th = threading.current_thread()
    if not hasattr(th, "_openai_client"):
        th._openai_client = OpenAI()
    return th._openai_client  # type: ignore[attr-defined]


def judge_answer(cfg: JudgeConfig, question: str, ground_truth: str, model_answer: str, sem: threading.Semaphore) -> str:
    if not model_answer:
        return "unknown"
    user_prompt = f"""Question: {question}
Ground Truth: {ground_truth}
Model Answer: {model_answer}

Respond with only one word: correct / incorrect / unknown
"""
    try:
        with sem:
            resp = _thread_local_openai().chat.completions.create(
                model=cfg.model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
            )
        out = (resp.choices[0].message.content or "").strip().lower()
        if out not in {"correct", "incorrect", "unknown"}:
            return "unknown"
        return out
    except APIStatusError:
        return "unknown"
    except Exception:
        return "unknown"


# ──────────────────────────────────────────────────────────────────────────────
# Agent factory + adapters
# ──────────────────────────────────────────────────────────────────────────────

class BaseAgent:
    """
    Minimal interface expected by the harness: `.run(...) -> (transcript, tool_calls)`.
    Concrete adapters wrap your own agent implementations to unify signatures.
    """
    def run(self, *args, **kwargs) -> Tuple[str, Any]:  # transcript, tool_calls
        raise NotImplementedError


def load_tokenizer(tokenizer_path: Optional[str] = None):
    if not tokenizer_path:
        return None
    if AutoTokenizer is None:
        raise RuntimeError("transformers not installed; cannot load tokenizer. pip install transformers")
    return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)


# --- ReCall adapter (used for fathom-search, ii-search, jan-nano) -------------
class ReCallAdapter(BaseAgent):
    def __init__(self, executor_urls: List[str]):
        # Lazy import to keep optional dependency
        from agents import ReCall  # type: ignore
        self._ReCall = ReCall
        self._executor_urls = list(executor_urls) if executor_urls else []
        if not self._executor_urls:
            raise ValueError("ReCall requires at least one --executors URL")

    def _pick_executor(self) -> str:
        # simple random choice; replace with round-robin if you prefer
        return random.choice(self._executor_urls)

    def run(
        self,
        env: str,
        func_schemas: List[Dict[str, Any]],
        question: str,
        model_url: Optional[str] = None,
        temperature: float = 0.6,
        max_new_tokens: int = 40960,
        tokenizer: Any = None,
    ) -> Tuple[str, Any]:
        agent = self._ReCall(executor_url=self._pick_executor())
        return agent.run(
            env=env,
            func_schemas=func_schemas,
            question=question,
            model_url=model_url,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            tokenizer=tokenizer,
        )


# --- ZeroSearch adapter (optional) -------------------------------------------
class ZeroSearchAdapter(BaseAgent):
    def __init__(self, thinker_url: Optional[str] = None):
        from agents import ZeroSearchInference, ZeroSearchConfig  # type: ignore
        cfg = ZeroSearchConfig(thinker_url=thinker_url)
        self._agent = ZeroSearchInference(cfg)

    def run(self, question: str, tokenizer):
        print("Zero")
        return self._agent.run(question, tokenizer = tokenizer)


# --- R1-Searcher adapter (optional) ------------------------------------------
class R1SearcherAdapter(BaseAgent):
    def __init__(self, model_url: Optional[str] = None):
        from agents import R1Searcher, R1SearchConfig as R1Cfg  # type: ignore
        cfg = R1Cfg(serper_api_key=os.getenv("SERPER_API_KEY", ""))
        self._agent = R1Searcher(cfg=cfg, model_url=model_url)

    def run(self, question: str, tokenizer) -> Tuple[str, Any]:
        return  self._agent.run(question, tokenizer = tokenizer)



# --- search-o1 adapter (optional) --------------------------------------------
class O1SearcherAdapter(BaseAgent):
    def __init__(self, model_url: Optional[str] = None):
        from agents import O1Searcher, O1Cfg  # type: ignore
        cfg = O1Cfg()
        self._agent = O1Searcher(cfg, thinker_url=model_url)

    def run(self, question: str, tokenizer) -> Tuple[str, Any]:
        return  self._agent.run(question, tokenizer = tokenizer)



def build_agent(kind: str, model_url: Optional[str], executors: List[str]) -> BaseAgent:
    kind = (kind or "").lower()
    if kind in {"fathom-search", "ii-search", "jan-nano"}:
        return ReCallAdapter(executor_urls=executors)
    if kind in {"zerosearch"}:
        return ZeroSearchAdapter(thinker_url=model_url)
    if kind in {"r1-searcher"}:
        return R1SearcherAdapter(model_url=model_url)
    if kind in {"search-o1"}:
        return O1SearcherAdapter(model_url=model_url)
    raise ValueError(f"Unknown agent kind: {kind}")


# ──────────────────────────────────────────────────────────────────────────────
# Search tool presets for ReCall (choose via --search-preset)
# ──────────────────────────────────────────────────────────────────────────────

RECALL_PRESETS: Dict[str, Tuple[str, List[Dict[str, Any]]]] = {
    # Legacy two-tool preset
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
                    "properties": {
                        "url": {"type": "string", "description": "The URL of the webpage to visit. Must be a single URL"},
                    },
                    "required": ["url"],
                },
            },
        ],
    ),
    # Fathom-style two-tool preset
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
                        "top_k": {"type": "integer", "default": 10}
                    },
                    "required": ["query"]
                },
            },
            {
                "name": "query_url",
                "description": "Visit webpage and return evidence based retrieval for the provided goal",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "The single URL to visit"},
                        "goal": {"type": "string", "description": "The specific information goal for visiting webpage"},
                    },
                    "required": ["url", "goal"],
                },
            },
        ],
    ),
}


# ──────────────────────────────────────────────────────────────────────────────
# Core evaluation per example
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_one(
    example: Dict[str, Any],
    agent_kind: str,
    model_url: Optional[str],
    agent: BaseAgent,
    judge_cfg: JudgeConfig,
    judge_sem: threading.Semaphore,
    # ReCall-specific (ignored by others):
    recall_env: Optional[str],
    recall_schemas: Optional[List[Dict[str, Any]]],
    tokenizer: Any,
) -> Dict[str, Any]:
    """
    Runs a single example through the agent, extracts an answer, gets a judge verdict,
    and returns a JSON-serializable row.
    """
    question = (example.get("question") or "").strip()
    if not question:
        raise ValueError("Example missing 'question'")
    answer_gt = str(example.get("answer") or "").strip()
    ex_id = str(example.get("id") or sha256_text(question))

    # Dispatch:
    # ReCall-backed agents (fathom-search, ii-search, jan-nano) use env/schemas path;
    # other agents (r1-searcher, search-o1, zerosearch) use their own run(question).
    agent_key = agent_kind.lower()
    if agent_key in {"fathom-search", "ii-search", "jan-nano"}:
        # print(agent)
        transcript, tool_calls = agent.run(
            env=recall_env or RECALL_PRESETS["fathom"][0],
            func_schemas=recall_schemas or RECALL_PRESETS["fathom"][1],
            question=question,
            model_url=model_url,
            temperature=0.6,
            max_new_tokens=40960,
            tokenizer=tokenizer,
        )
    else:
        transcript, tool_calls = agent.run(question=question, tokenizer=tokenizer)  # ← pass it through

    # else:
        # transcript, tool_calls = agent.run(question=question)  # type: ignore[arg-type]

    # Heuristic extraction by agent family
    if agent_key in {"r1-searcher", "zerosearch"}:
        pred = extract_answer_tagged(transcript or "")
    else:
        pred = extract_answer_boxed(transcript or "")

    verdict = judge_answer(judge_cfg, normalize(question), normalize(answer_gt), pred, judge_sem)

    return {
        "id": ex_id,
        "question": question,
        "answer_gt": answer_gt,
        "model_answer": pred,
        "judge": verdict,
        "tool_calls": tool_calls,
        "transcript": transcript,
    }


# ──────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_jsonl(path: pathlib.Path) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except Exception:
                continue
    return data


def save_rows_jsonl(path: pathlib.Path, rows: Iterable[Dict[str, Any]], mode: str = "a") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode, encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_out_path(out_base: pathlib.Path, agent: str, dataset: str, name: str) -> pathlib.Path:
    return out_base / f"{agent}" / f"{dataset}{('-' + name) if name else ''}.jsonl"


def collect_existing_ids(path: pathlib.Path) -> set[str]:
    ids: set[str] = set()
    if not path.exists():
        return ids
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
                rid = str(row.get("id") or "")
                if rid:
                    ids.add(rid)
            except Exception:
                continue
    return ids


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Unified benchmarking harness (single or multi-threaded).")
    parser.add_argument("--dataset", required=True, help="Dataset name (file {dataset}.jsonl under --data-root)")
    parser.add_argument("--data-root", default="./datasets", help="Directory containing dataset JSONL files")
    parser.add_argument("--agent", required=True, choices=[
        "fathom-search", "ii-search", "jan-nano", "zerosearch", "r1-searcher", "search-o1"
    ])
    parser.add_argument("--model-url", help="URL of the model server (needed for fathom-search, ii-search, jan-nano)")
    parser.add_argument("--executors", default="", help="Comma-separated ReCall executor URLs")
    parser.add_argument("--out-base", required=True, help="Base directory for outputs")
    parser.add_argument("--name", default="", help="Suffix for output filename, e.g. '128k'")
    parser.add_argument("--mode", choices=["single", "multi"], default="single")
    parser.add_argument("--workers", type=int, default=8, help="Threads for multi-mode")
    parser.add_argument("--limit", type=str, default=None, help='Limit range "start,end" or "N" for first N')
    parser.add_argument("--resume", action="store_true", help="Append to existing file and skip already done IDs")
    parser.add_argument("--tokenizer", default=None, help="Optional HF tokenizer/base ckpt path to pass to agent")
    parser.add_argument("--search-preset", choices=list(RECALL_PRESETS.keys()), default="fathom",
                        help="ReCall tool preset ('legacy' or 'fathom')")
    # Judge config
    parser.add_argument("--judge-model", default="gpt-4.1-mini")
    parser.add_argument("--judge-concurrency", type=int, default=3)
    parser.add_argument("--judge-max-tokens", type=int, default=64)

    args = parser.parse_args()

    # Resolve paths
    data_root = pathlib.Path(args.data_root).expanduser().resolve()
    ds_path = data_root / f"{args.dataset}.jsonl"
    if not ds_path.exists():
        raise FileNotFoundError(f"Dataset not found: {ds_path}")

    out_base = pathlib.Path(args.out_base).expanduser().resolve()
    out_path = build_out_path(out_base, args.agent, args.dataset, args.name)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load dataset
    data = load_jsonl(ds_path)

    # Apply limit
    start, end = parse_limit(args.limit)
    data = data[start: end if end is not None else None]

    # Optional tokenizer
    tok = load_tokenizer(args.tokenizer) if args.tokenizer else None

    # Build agent
    executors = [u.strip() for u in args.executors.split(",") if u.strip()]
    agent = build_agent(args.agent, args.model_url, executors)

    # Judge config and semaphore
    judge_cfg = JudgeConfig(
        model=args.judge_model,
        temperature=0.0,
        max_tokens=args.judge_max_tokens,
        concurrency=max(1, int(args.judge_concurrency)),
    )
    judge_sem = threading.Semaphore(judge_cfg.concurrency)

    # Resume logic: collect existing IDs to skip
    # already_done: set[str] = set()
    # write_mode = "a"
    # if args.resume and out_path.exists():
    #     already_done = collect_existing_ids(out_path)
    #     logging.info("Resuming: %d IDs already present and will be skipped.", len(already_done))
    # else:
    #     write_mode = "w"  # fresh file
    # Resume logic & file handling
    already_done: set[str] = set()
    if args.resume:
        if out_path.exists():
            already_done = collect_existing_ids(out_path)
            logging.info(
               "Resuming: %d IDs already present and will be skipped.",
                len(already_done),
            )
    else:
        # fresh run: delete prior output file if it exists
        if out_path.exists():
            out_path.unlink()
 

    # Select ReCall preset
    recall_env, recall_schemas = RECALL_PRESETS[args.search_preset]

    correct = 0
    total = 0
    start_time = time.perf_counter()

    def handle_result(row: Dict[str, Any]) -> None:
        nonlocal correct, total
        total += 1
        if row.get("judge") == "correct":
            correct += 1
        # add context
        row.update({"agent": args.agent, "dataset": args.dataset})
        save_rows_jsonl(out_path, [row], mode="a")



    # Execute
    if args.mode == "single":
        for ex in tqdm(data, desc="QA loop (single)"):
            ex_id = str(ex.get("id") or sha256_text(ex.get("question", "")))
            if args.resume and ex_id in already_done:
                continue
            try:
                row = evaluate_one(
                    example=ex,
                    agent_kind=args.agent,
                    model_url=args.model_url,
                    agent=agent,
                    judge_cfg=judge_cfg,
                    judge_sem=judge_sem,
                    recall_env=recall_env,
                    recall_schemas=recall_schemas,
                    tokenizer=tok,
                )
                handle_result(row)
            except Exception as e:
                logging.exception("Failed on example id=%s: %s", ex_id, e)
    else:
        workers = max(1, int(args.workers))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = []
            for ex in data:
                ex_id = str(ex.get("id") or sha256_text(ex.get("question", "")))
                if args.resume and ex_id in already_done:
                    continue
                futures.append(pool.submit(
                    evaluate_one,
                    ex,
                    args.agent,
                    args.model_url,
                    agent,
                    judge_cfg,
                    judge_sem,
                    recall_env,
                    recall_schemas,
                    tok,
                ))
            for fut in tqdm(as_completed(futures), total=len(futures), desc="QA loop (multi)"):
                try:
                    row = fut.result()
                    handle_result(row)
                except Exception as e:
                    logging.exception("Worker failed: %s", e)

    elapsed = time.perf_counter() - start_time
    acc = (correct / total) if total else 0.0
    print(f"Accuracy: {correct}/{total} = {acc:.1%}")
    if total:
        print(f"Elapsed time: {elapsed:.2f}s ({elapsed/total:.2f}s/example)")
    print(f"Wrote results to: {out_path}")


if __name__ == "__main__":
    main()
