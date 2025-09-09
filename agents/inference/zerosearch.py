# zero_search_inference.py
"""End-to-end inference loop that emulates the ZeroSearch prompting style.

The policy model ("thinker") must:
    • reason inside <think> … </think>
    • place a query inside <search> … </search> whenever it needs external knowledge
    • return the final short answer inside <answer> … </answer>

The wrapper intercepts each <search> request, fulfils it with either:
    (a) a simulated search engine (LLM retriever), or
    (b) a real search backend (Serper.dev) if engine="real".

Tokenizer handling:
    - No hard-coded tokenizer path.
    - Pass an HF tokenizer from the harness (optional). Budgeting falls back if None.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import List, Optional

import requests
from openai import OpenAI

__all__ = ["ZeroSearchInference", "ZeroSearchConfig"]

# ───────────────────────── retry ─────────────────────────
def retry(max_attempts: int = 4, sleep: float = 1, fallback=None):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for i in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception:
                    if i == max_attempts - 1:
                        return fallback
                    time.sleep(sleep)
        return wrapper
    return decorator

# ───────────────────────── OpenAI client (thread-local) ─────────────────────
import threading
def _oa() -> OpenAI:
    th = threading.current_thread()
    if not hasattr(th, "_oa"):
        th._oa = OpenAI()  # uses OPENAI_API_KEY from env
    return th._oa

# ───────────────────────── config ─────────────────────────
@dataclass
class ZeroSearchConfig:
    # thinker LLM endpoint
    thinker_url: str = "http://0.0.0.0:1214"
    thinker_temperature: float = 0.7
    thinker_max_tokens: int = 40960

    # retrieval engine: "sim" (LLM) or "real" (Serper)
    engine: str = "real"

    # simulated search (engine == "sim")
    retriever_model: str = "gpt-4o-mini"
    retriever_top_k: int = 5

    # real search (engine == "real")
    serper_api_key: Optional[str] = os.getenv("SERPER_API_KEY", None)
    serper_url: str = "https://google.serper.dev/search"
    serper_top_k: int = 5

    # loop
    max_rounds: int = 16

# ───────────────────────── main wrapper ─────────────────────────
class ZeroSearchInference:
    SEARCH_OPEN = "<search>"
    SEARCH_CLOSE = "</search>"
    INFO_OPEN = "<information>"
    INFO_CLOSE = "</information>"
    ANSWER_CLOSE = "</answer>"
    THINK_OPEN = "<think>"
    THINK_CLOSE = "</think>"

    # vLLM will stop on these; we may re-append if needed
    STOP_TOKENS = [
        "<|im_end|>", "<|endoftext|>",
        "</search>", " </search>", "</search>\n", " </search>\n", "</search>\n\n", " </search>\n\n",
    ]

    def __init__(self, cfg: ZeroSearchConfig):
        self.cfg = cfg
        self.openai = _oa()
        self.tokenizer = None  # set in run()

    # public API — remains ABI-compatible
    def run(self, user_question: str, tokenizer=None):
        """Run the ZeroSearch loop.
        tokenizer: optional HF tokenizer passed from the harness.
        """
        self.tokenizer = tokenizer
        tool_calls: List[str] = []

        prompt = self._build_initial_prompt(user_question)
        for _ in range(self.cfg.max_rounds):
            generated = self._call_thinker(prompt, tokenizer)
            prompt += generated

            if self.ANSWER_CLOSE in generated:
                break

            query = self._extract_query(generated)
            if not query:
                break

            tool_calls.append(query)
            info_block = self._retrieve_and_format(query)
            prompt += info_block + self.THINK_OPEN  # continue reasoning

        else:  # exceeded rounds
            prompt += "<answer>I don't know.</answer><|im_end|>"

        return prompt, tool_calls

    # ───────── prompt helpers ─────────
    def _build_initial_prompt(self, question: str) -> str:
        user_msg = (
            "Answer the given question. "
            "You must conduct reasoning inside <think> and </think> first every time you get new information. "
            "If you lack knowledge, call the search engine with <search> query </search>; "
            "results will be provided between <information> and </information>. "
            "Search as many times as you need. "
            "If no further knowledge is needed, provide the final short answer inside <answer> and </answer>. "
            f"Question: {question}\n"
        )
        return f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n{self.THINK_OPEN}"

    # ───────── thinker call ─────────
    @retry(fallback="")
    def _call_thinker(self, prompt: str, tokenizer) -> str:
        # budget tokens if tokenizer is available
        if tokenizer is not None:
            try:
                ids = self.tokenizer(prompt, return_tensors=None, add_special_tokens=False)["input_ids"]
                max_left = max(128, self.cfg.thinker_max_tokens - len(ids) - 100)
            except Exception:
                max_left = 8192
        else:
            max_left = 8192

       
        resp = requests.post(
            f"{self.cfg.thinker_url}/generate",
            json={
                "text": prompt,
                "sampling_params": {
                    "temperature": self.cfg.thinker_temperature,
                    "max_new_tokens": max_left,
                    "stop": self.STOP_TOKENS,
                },
                
            },
            timeout=60,
        ).json()
        print(resp)
        generated = resp["text"]                       # what you have now
        matched   = resp["meta_info"]["finish_reason"].get("matched")
        reason = resp["meta_info"]["finish_reason"].get("type")
        # ⇢ append the tag back only if it was removed
        if reason == "stop" and matched in self.STOP_TOKENS:
            if not generated.endswith(matched):
                generated += matched
        if reason == "stop" and matched == 151645:
             if not generated.endswith("<|im_end|>"):
                generated += "<|im_end|>"
        return generated


    # ───────── query extraction ─────────
    def _extract_query(self, gen_text: str) -> Optional[str]:
        if self.SEARCH_OPEN not in gen_text or self.SEARCH_CLOSE not in gen_text:
            return None
        q = gen_text.split(self.SEARCH_OPEN)[-1].split(self.SEARCH_CLOSE)[0].strip()
        return q or None

    # ───────── retrieval ─────────
    def _retrieve_and_format(self, query: str) -> str:
        if self.cfg.engine == "real":
            docs = self._real_search(query)
        else:
            docs = self._simulated_search(query)
        return f"{self.INFO_OPEN}\n{docs}\n{self.INFO_CLOSE}\n\n"

    # simulated search (LLM)
    @retry(fallback="No information available")
    def _simulated_search(self, query: str) -> str:
        messages = [{
            "role": "user",
            "content": (
                "You are a search engine. Return up to "
                f"{self.cfg.retriever_top_k} short documents (titles + snippets) "
                "most relevant to the query, each on a new line.\n\n"
                f"Query: {query}"
            ),
        }]
        resp = self.openai.chat.completions.create(
            model=self.cfg.retriever_model,
            messages=messages,
            max_tokens=256,
            temperature=0.0,
        )
        return resp.choices[0].message.content.strip()

    # real search (Serper)
    @retry(fallback="No information available")
    def _real_search(self, query: str) -> str:
        if not self.cfg.serper_api_key:
            raise ValueError("serper_api_key must be set for real search mode")
        headers = {"X-API-KEY": self.cfg.serper_api_key, "Content-Type": "application/json"}
        payload = {"q": query, "num": self.cfg.serper_top_k}
        resp = requests.post(self.cfg.serper_url, json=payload, headers=headers, timeout=20)
        resp.raise_for_status()
        data = resp.json().get("organic", [])[: self.cfg.serper_top_k]
        lines = []
        for i, item in enumerate(data, 1):
            title = item.get("title", "")
            snippet = item.get("snippet", "")
            lines.append(f"Doc {i}: Title: {title}\nSnippet: {snippet}")
        return "\n".join(lines) or "No information available"
