#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_nosearch.py — Minimal baseline evaluator (single or multi-threaded)

What this script does
---------------------
- Talks to either a local vLLM server (`/generate`) OR any OpenAI-compatible
  server (`/v1/chat/completions`) using the same `--model-url`.
- No tools, no fancy CoT parsing; just a plain prompt and robust "final answer"
  extraction helper.
- Judges with OpenAI (configurable model) behind a small concurrency semaphore.
  If the judge API fails, falls back to a simple substring/normalization match.
- Appends to a JSONL file so runs are resumable and safe to interrupt.

Dataset format (one JSON object per line):
    {"id": "...", "question": "...", "answer": "..."}

Quick start
-----------
export OPENAI_API_KEY=sk-...   # for judge
python eval_nosearch.py \
  --dataset /path/to/frames.jsonl \
  --out /path/to/output/folder/filename.jsonl \
  --model-url http://0.0.0.0:1240 \
  --tokenizer-path /path/to/toeknizer/model \
  --mode multi --workers 64
"""

from __future__ import annotations
import argparse
import hashlib
import json
import pathlib
import re
import threading
import time
import unicodedata
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ----------------------------- Optional HF tokenizer --------------------------
try:
    from transformers import AutoTokenizer  # type: ignore
except Exception:
    AutoTokenizer = None  # type: ignore

# ----------------------------- OpenAI judge client ----------------------------
try:
    from openai import OpenAI, APIStatusError  # type: ignore
except Exception:
    OpenAI = None  # type: ignore
    APIStatusError = Exception  # type: ignore


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                         Small utility helpers                             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def normalize(s: str) -> str:
    """NFKD normalize + lowercase + trim. Good for loose comparisons."""
    return unicodedata.normalize("NFKD", s or "").strip().lower()


def sha_id(text: str) -> str:
    """Deterministic ID from the question string if `id` absent in dataset."""
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def read_jsonl(path: pathlib.Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def collect_existing_ids(path: pathlib.Path) -> set[str]:
    """Scan an existing results file and collect IDs that already have outputs."""
    seen: set[str] = set()
    if not path.exists():
        return seen
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
                rid = str(row.get("id") or "")
                if rid:
                    seen.add(rid)
            except Exception:
                continue
    return seen


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                    Final answer extraction helpers                        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

_BOXED_RE_LAST = re.compile(r"(\\boxed\s*\{.*?\}|\\boxed\s+[^\s$]+|\\fbox\{.*?\})", re.S)

def _find_last_boxed(text: str) -> Optional[str]:
    """Return the last \\boxed{...} / \\boxed ... or \\fbox{...} span, if any."""
    if not text:
        return None
    matches = _BOXED_RE_LAST.findall(text)
    return matches[-1] if matches else None

def _unbox(span: str) -> str:
    """Strip \\boxed{...} / \\boxed ... / \\fbox{...} wrappers."""
    s = span.strip()
    if s.startswith("\\boxed "):
        return s[len("\\boxed "):].strip()
    if s.startswith("\\boxed{") and s.endswith("}"):
        return s[len("\\boxed{"):-1].strip()
    if s.startswith("\\fbox{") and s.endswith("}"):
        return s[len("\\fbox{"):-1].strip()
    return s

def extract_final_answer(text: str) -> str:
    """
    Heuristic final-answer extractor in descending preference:
    1) last \boxed{...} / \boxed ... / \fbox{...}
    2) last <answer>...</answer>
    3) last 'final answer:'/'answer:'/'ans:' pattern on a line
    4) last non-empty line
    """
    if not isinstance(text, str) or not text:
        return ""

    # 1) boxed
    span = _find_last_boxed(text)
    if span:
        return _unbox(span)

    # 2) <answer>...</answer>
    tags = re.findall(r"<answer>(.*?)</answer>", text, flags=re.S | re.I)
    if tags:
        return tags[-1].strip()

    # 3) "final answer:" labels
    lab = re.findall(r"(?i)(?:final answer|answer|ans)\s*[:\-]\s*([^\n\r]+)", text)
    if lab:
        return lab[-1].strip()

    # 4) last non-empty line
    for line in reversed(text.strip().splitlines()):
        if line.strip():
            return line.strip()
    return ""


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                       Tokenizer / prompt templating                       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

_TOKENIZER_CACHE: Dict[str, Any] = {}

def get_tokenizer(path: str):
    if not path:
        return None
    if AutoTokenizer is None:
        raise RuntimeError("transformers not installed; install `pip install transformers`")
    tok = _TOKENIZER_CACHE.get(path)
    if tok is None:
        tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        _TOKENIZER_CACHE[path] = tok
    return tok

def render_prompt(tokenizer, system_prompt: str, user_prompt: str) -> Tuple[str, int]:
    """
    Use the tokenizer chat template to produce a single text prompt (for /generate).
    Returns: (prompt_text, prompt_token_count)
    """
    if tokenizer is None:
        # Fallback: concatenate plainly.
        prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{user_prompt}\n"
        return prompt, 0
    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tokenizer(prompt, return_tensors=None, add_special_tokens=False).get("input_ids", [])
    return prompt, (len(ids) if isinstance(ids, list) else 0)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                        HTTP client (thread-local)                         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

_TL = threading.local()

def _session() -> requests.Session:
    if not hasattr(_TL, "sess"):
        _TL.sess = requests.Session()
    return _TL.sess

def _post_json(url: str, payload: dict, timeout: int = 120) -> dict:
    r = _session().post(url, json=payload, timeout=timeout)
    try:
        r.raise_for_status()
    except requests.HTTPError:
        # Helpful debug on failure
        print("---- REQUEST PAYLOAD ----", file=sys.stderr)
        try:
            print(json.dumps(payload, indent=2)[:4000], file=sys.stderr)
        except Exception:
            print(str(payload)[:4000], file=sys.stderr)
        print("---- RESPONSE TEXT ----", file=sys.stderr)
        print(r.text[:4000], file=sys.stderr)
        raise
    return r.json()


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                      Chat call (vLLM → OpenAI fallback)                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def chat(
    *,
    model_url: str,
    chat_model: str,
    system_prompt: str,
    user_prompt: str,
    tokenizer,
    max_tokens: int,
    temperature: float,
    stop: Optional[List[str]] = None,
) -> str:
    """
    Try vLLM `/generate` first (with a single prompt string).
    If that fails, fall back to OpenAI-compatible `/v1/chat/completions`.
    """
    # Render prompt for /generate and budget new tokens
    prompt, prompt_tokens = render_prompt(tokenizer, system_prompt, user_prompt)
    max_new_tokens = max(1, max_tokens - prompt_tokens - 100)  # safety buffer

    # 1) vLLM /generate
    try:
        gen_url = model_url.rstrip("/") + "/generate"
        payload = {
            "text": prompt,
            "sampling_params": {
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
            },
        }
        if stop:
            payload["sampling_params"]["stop"] = stop
        data = _post_json(gen_url, payload)
        out = data.get("text")
        if isinstance(out, list) and out:
            return (out[0] or "").strip()
        if isinstance(out, str):
            return out.strip()
    except Exception:
        pass  # fall through

    # 2) OpenAI-compatible /v1/chat/completions
    chat_url = model_url.rstrip("/") + "/v1/chat/completions"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]
    payload = {
        "model": chat_model or "unknown-model",
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_new_tokens,
        "stop": stop or None,
    }
    data = _post_json(chat_url, payload)
    try:
        return (data["choices"][0]["message"]["content"] or "").strip()
    except Exception:
        # Last resort: try common vLLM streaming compat payloads
        txt = data.get("text") or ""
        if isinstance(txt, list) and txt:
            return (txt[0] or "").strip()
        return str(txt).strip()


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                                  Judge                                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

JUDGE_SYSTEM = """You are an impartial judge evaluating the correctness of a model's answer
against a ground-truth answer for a given question.

Output exactly one word: "correct", "incorrect", or "unknown".

- Treat case and minor formatting differences as irrelevant.
- If it's multiple-choice, label matches (A/B/C/D) count.
- If the model answer is empty or unclear, output "unknown".
"""

class Judge:
    """OpenAI judge with small concurrency limit and robust fallback."""
    def __init__(self, model: str = "gpt-4.1-mini", concurrency: int = 3, max_tokens: int = 64):
        self.model = model
        self.sem = threading.Semaphore(max(1, int(concurrency)))
        self.max_tokens = max_tokens
        self._client = None
        if OpenAI is not None:
            try:
                self._client = OpenAI()
            except Exception:
                self._client = None

    def _api_judge(self, q: str, gt: str, pred: str) -> Optional[str]:
        if not self._client or not pred.strip():
            return None
        user_prompt = f"Question: {q}\nGround Truth: {gt}\nModel Answer: {pred}\n\nRespond with only one word: correct / incorrect / unknown\n"
        try:
            with self.sem:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": JUDGE_SYSTEM},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.0,
                    max_tokens=self.max_tokens,
                )
            out = (resp.choices[0].message.content or "").strip().lower()
            return out if out in {"correct", "incorrect", "unknown"} else "unknown"
        except Exception:
            return None

    @staticmethod
    def _fallback(q: str, gt: str, pred: str) -> str:
        """Simple normalization + substring test as a last resort."""
        P, G = normalize(pred), normalize(gt)
        if not P:
            return "unknown"
        if P == G or (G and G in P) or (P and P in G):
            return "correct"
        return "incorrect"

    def __call__(self, q: str, gt: str, pred: str) -> str:
        return self._api_judge(q, gt, pred) or self._fallback(q, gt, pred)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                           Core per-example eval                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def eval_one(ex: dict, args, tokenizer, judge: Judge) -> dict:
    """Run a single example through the model, extract final answer, and judge."""
    q = (ex.get("question") or "").strip()
    gt = (ex.get("answer") or "").strip()
    ex_id = str(ex.get("id") or sha_id(q))

    # Build user prompt from template
    user_prompt = (args.prompt_template or "Q: {q}\nA:").format(q=q)

    try:
        transcript = chat(
            model_url=args.model_url,
            chat_model=args.chat_model,
            system_prompt=args.system_prompt,
            user_prompt=user_prompt,
            tokenizer=tokenizer,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            stop=args.stop,
        )
        pred = extract_final_answer(transcript) or transcript.strip()
        verdict = judge(q, gt, pred)
        return {
            "id": ex_id,
            "question": q,
            "answer_gt": gt,
            "model_answer": pred,
            "judge": verdict,
            "tool_calls": [],
            "transcript": transcript,
        }
    except Exception as e:
        return {
            "id": ex_id,
            "question": q,
            "answer_gt": gt,
            "model_answer": "",
            "judge": "unknown",
            "error": str(e),
        }


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                                   Main                                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def main():
    p = argparse.ArgumentParser(description="Minimal baseline evaluator (vLLM or OpenAI-compatible).")
    p.add_argument("--dataset", required=True, help="Path to dataset .jsonl")
    p.add_argument("--out", required=True, help="Output .jsonl (appended).")
    p.add_argument("--model-url", required=True, help="Base URL (e.g., http://0.0.0.0:8000)")
    p.add_argument("--chat-model", default="local-model", help="`model` string for /v1/chat/completions fallback")
    p.add_argument("--tokenizer-path", required=True, help="HF tokenizer path (for chat template rendering)")
    p.add_argument("--limit", type=int, default=0, help="Evaluate first N items (0 = all)")
    p.add_argument("--prompt-template", default="Q: {q}\nA:", help="User prompt template (use {q})")
    p.add_argument("--system-prompt",
                   default="Please answer the following question. Provide your final answer as \\boxed{YOUR_ANSWER}.",
                   help="System prompt string (ignored if --system-prompt-file is set)")
    p.add_argument("--system-prompt-file",
                   help="File containing system prompt; overrides --system-prompt")
    p.add_argument("--mode", choices=["single", "multi"], default="single")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.4)
    p.add_argument("--max-tokens", type=int, default=39000)
    p.add_argument("--resume-from", help="Existing jsonl to skip IDs (optional)")
    p.add_argument("--stop", nargs="*", default=None, help="Optional stop strings")
    # Judge config
    p.add_argument("--judge-model", default="gpt-4.1-mini")
    p.add_argument("--judge-concurrency", type=int, default=3)
    p.add_argument("--judge-max-tokens", type=int, default=64)
    args = p.parse_args()

    ds_path = pathlib.Path(args.dataset).expanduser().resolve()
    out_path = pathlib.Path(args.out).expanduser().resolve()
    assert ds_path.exists(), f"Dataset not found: {ds_path}"

    # Load/override system prompt
    if args.system_prompt_file:
        args.system_prompt = pathlib.Path(args.system_prompt_file).read_text(encoding="utf-8")

    # Tokenizer (for chat template → /generate)
    tokenizer = get_tokenizer(args.tokenizer_path)

    # Data + limit
    data = read_jsonl(ds_path)
    if args.limit and args.limit > 0:
        data = data[: args.limit]

    # Resume (merge from resume-from and current out file)
    seen_ids: set[str] = set()
    if args.resume_from:
        seen_ids |= collect_existing_ids(pathlib.Path(args.resume_from))
    if out_path.exists():
        seen_ids |= collect_existing_ids(out_path)
    if seen_ids:
        data = [ex for ex in data if str(ex.get("id") or sha_id(ex.get("question") or "")) not in seen_ids]

    # Prepare output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Ensure file exists (append-only)
    if not out_path.exists():
        with out_path.open("w", encoding="utf-8"):
            pass

    # Judge
    judge = Judge(
        model=args.judge_model,
        concurrency=args.judge_concurrency,
        max_tokens=args.judge_max_tokens,
    )

    start = time.perf_counter()
    total_attempted = len(data)
    correct = 0
    lock = threading.Lock()

    def write_row(row: dict):
        nonlocal correct
        if row.get("judge") == "correct":
            correct += 1
        with lock:
            with out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Run
    if args.mode == "single":
        for ex in data:
            row = eval_one(ex, args, tokenizer, judge)
            write_row(row)
    else:
        workers = max(1, int(args.workers))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(eval_one, ex, args, tokenizer, judge) for ex in data]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="QA loop (multi)"):
                try:
                    row = fut.result()
                except Exception as e:
                    row = {
                        "id": "unknown", "question": "", "answer_gt": "",
                        "model_answer": "", "judge": "unknown", "error": str(e),
                    }
                write_row(row)

    elapsed = time.perf_counter() - start
    total = max(1, total_attempted)
    acc = correct / total
    print(f"Accuracy: {correct}/{total} = {acc:.2%}")
    print(f"Elapsed: {elapsed:.2f}s ({elapsed/total:.2f}s/example)")
    print(f"Wrote results to: {out_path}")

if __name__ == "__main__":
    main()
