#!/usr/bin/env python3
# r1_searcher_inference.py
from __future__ import annotations

import os, re, time, json, requests
from dataclasses import dataclass
from typing import List, Optional
from urllib.parse import unquote

from bs4 import BeautifulSoup
import trafilatura
import wikipedia
from openai import OpenAI

# Use environment for OpenAI API key; do NOT hard-code secrets
client = OpenAI()  # reads OPENAI_API_KEY from env

# ─────────────────────────  BASIC UTILS  ──────────────────────────────
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


# ──────────────────────────  CONFIG  ──────────────────────────────────
@dataclass
class R1SearchConfig:
    # Serper.dev parameters
    serper_api_key: str = os.getenv("SERPER_API_KEY", "")
    serper_url: str = "https://google.serper.dev/search"
    gl: str = "us"
    hl: str = "en"

    # Policy model endpoint (vLLM)
    thinker_temperature: float = 0.0
    thinker_max_tokens: int = 40960

    # Loop / misc
    max_rounds: int = 16
    summariser_model: str = "gpt-4o-mini"


# ─────────────────────────  R1-Searcher  ──────────────────────────────
class R1Searcher:
    SYSTEM_PROMPT = """
    You are a helpful assistant.
    Given a question, you should answer it by first thinking about the reasoning
    process in the mind and then providing the final answer.

    The output format of reasoning process and final answer are enclosed within
    <think> </think> and <answer> </answer> tags, respectively, i.e.,
    "<think> reasoning process here </think>

    <answer> final answer here </answer>".

    During the thinking process, you can perform searching with exactly one single-triple query:
    "<|begin_of_query|> keyword_1 keyword_2 ... <|end_of_query|>".

    The system will then provide:
    "<|begin_of_documents|> ...search results... <|end_of_documents|>".
    """.strip()

    SUMMARY_PROMPT = (
        "## Task Description:\n"
        "Given the search query and the content of the searched webpage, "
        "extract information relevant to the query and write one summary paragraph.\n\n"
        "## Guidelines:\n"
        "(1) The extracted content should be relevant to the query.\n"
        "(2) The form of the extracted content must be a summary paragraph rather than a direct answer.\n"
        "(3) If the webpage content is unrelated to the query, output \"None\".\n\n"
        "## Output Format:\n"
        "[Exacted Content]: <summary-paragraph-or-None>\n\n"
        "## Inputs:\n"
        "[Search Query]\n{search_query}\n\n"
        "[Webpage Content]\n{document}\n\n"
        "## Output:\n"
    )

    EOS_TOKEN = "<|im_end|>"
    THINK_OPEN = "<think>"
    ANSWER_CLOSE = "</answer>"
    Q_OPEN, Q_CLOSE = "<|begin_of_query|>", "<|end_of_query|>"
    DOC_OPEN, DOC_CLOSE = "<|begin_of_documents|>", "<|end_of_documents|>"

    # textual stop sequences; server should match these
    STOP_TOKENS = [
        "<|im_end|>",
        "<|endoftext|>",
        "<|end_of_query|>",
        " <|end_of_query|>",
        "<|end_of_query|>\n",
        "<|end_of_query|>\n\n",
        " <|end_of_query|>\n",
        " <|end_of_query|>\n\n",
    ]

    def __init__(self, cfg: R1SearchConfig, model_url: str):
        self.cfg = cfg
        self.openai = client
        self.model_url = model_url.rstrip("/")
        self.tokenizer = None  # will be set from run(...)

        # Wikipedia client setup
        self._wiki = wikipedia
        self._wiki.set_lang("en")
        sess = requests.Session()
        sess.headers.update({"User-Agent": "r1-searcher-bot/1.0"})
        self._wiki._http = sess

    # ── public entry ─────────────────────────────────────────────────
    def run(self, question: str, tokenizer=None) -> tuple[str, List[str]]:
        """tokenizer is optional; if provided, used only to budget tokens."""
        self.tokenizer = tokenizer  # <-- accept tokenizer from harness (optional)

        prompt = (
            f"<|im_start|>system\n{self.SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{question}<|im_end|>\n"
            f"<|im_start|>assistant\n{self.THINK_OPEN}"
        )
        queries: List[str] = []
    

        for i in range(self.cfg.max_rounds):
            model_out = self._call_thinker(prompt, tokenizer)
            prompt += model_out

            if self.ANSWER_CLOSE in model_out:
                break

            query = self._extract_query(model_out)
            print(query)
            if not query:
                break
            queries.append(query)

            doc_block = self._retrieve_block(query)
            print(doc_block)
            prompt += (
                "<|im_start|>user\n" + doc_block + self.EOS_TOKEN +
                "<|im_start|>assistant\n" + self.THINK_OPEN
            )

        else:
            prompt += "<answer>I don't know.</answer><|im_end|>"

        return prompt, queries

    # ── thinker call ────────────────────────────────────────────────
    def _call_thinker(self, prompt: str, tokenizer) -> str:
        # If we have a tokenizer, try to budget tokens. Otherwise use a safe cap.
        if self.tokenizer is not None:
            try:
                ids = self.tokenizer(prompt, return_tensors=None, add_special_tokens=False)["input_ids"]
                max_tokens_left = max(128, self.cfg.thinker_max_tokens - len(ids) - 100)
            except Exception:
                max_tokens_left = 2048
        else:
            max_tokens_left = 2048

        resp = requests.post(
            f"{self.model_url}/generate",
            json={
                "text": prompt,
                "sampling_params": {
                    "temperature": self.cfg.thinker_temperature,
                    "max_new_tokens": max_tokens_left,
                    "stop": self.STOP_TOKENS,
                    "repetition_penalty": 1.05,
                },
            },
            timeout=60,
        ).json()
        generated = resp["text"]                       # what you have now
        matched   = resp["meta_info"]["finish_reason"].get("matched")
        reason = resp["meta_info"]["finish_reason"].get("type")
        #print("-"*100)
        #print(resp)
        #print(matched)
        #print("-"*100)
        # ⇢ append the tag back only if it was removed
        if reason == "stop" and matched in self.STOP_TOKENS:
            if not "<|end_of_query|>" in generated:
                generated += matched + self.EOS_TOKEN
        if reason == "stop" and matched == 151645:
             if not generated.endswith("<|im_end|>"):
                generated += "<|im_end|>"
        if reason == "stop" and matched == 151643:
             if not generated.endswith("<|endoftext|>"):
                generated += "<|endoftext|>"
        return generated
        # Do NOT try to “repair” stop tokens by IDs; keep it simple and robust.
        return generated

    # ── query helpers ───────────────────────────────────────────────
    @staticmethod
    def _extract_query(text: str) -> Optional[str]:
        if R1Searcher.Q_OPEN not in text or R1Searcher.Q_CLOSE not in text:
            return None
        fragment = text.split(R1Searcher.Q_OPEN)[-1].split(R1Searcher.Q_CLOSE)[0]
        fragment = fragment.split("<|")[0]  # handle accidental slip
        return (
            fragment.replace("\t", " ")
            .replace('"', "")
            .replace("'", "")
            .replace("…", "")
            .strip()
        ) or None

    # ── retrieval & summary ─────────────────────────────────────────
    def _retrieve_block(self, query: str) -> str:
        wiki_links = self._serper_wiki_links(query)
        for url in wiki_links[:3]:
            text = self._get_wiki_text(url)
            if not text:
                continue
            summary = self._summarise(query, text[:35000])
            if summary.lower() != "none":
                return f"{self.DOC_OPEN}\n{summary}\n{self.DOC_CLOSE}\n\n"
        return f"{self.DOC_OPEN}\nNone\n{self.DOC_CLOSE}\n\n"

    @retry()
    def _serper_wiki_links(self, q: str) -> List[str]:
        headers = {"X-API-KEY": self.cfg.serper_api_key, "Content-Type": "application/json"}
        payload = {"q": f"{q} site:en.wikipedia.org", "num": 10, "gl": self.cfg.gl, "hl": self.cfg.hl}
        r = requests.post(self.cfg.serper_url, json=payload, headers=headers, timeout=20)
        r.raise_for_status()
        return [
            item.get("link")
            for item in r.json().get("organic", [])
            if item.get("link", "").startswith("https://en.wikipedia.org")
        ]

    def extract_main_text(self, html: str) -> str:
        txt = trafilatura.extract(html, output_format="txt") or ""
        if len(txt) >= 500:
            return txt
        from readability import Document
        soup = BeautifulSoup(Document(html).summary(), "lxml")
        txt = soup.get_text(" ", strip=True)
        if len(txt) >= 400:
            return txt
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        return re.sub(r"\s+", " ", soup.get_text(" ").strip())

    def _get_wiki_text(self, url: str) -> Optional[str]:
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            txt = self.extract_main_text(r.text).strip()
            if not txt:
                return None
            slug = unquote(url.rsplit("/", 1)[-1]).replace("_", " ")
            if slug.lower() not in txt.lower():
                txt = f"{slug}\n\n{txt}"
            return "[Retrieved from Wikipedia] " + txt
        except Exception:
            return None

    @retry(fallback="None")
    def _summarise(self, query: str, doc: str) -> str:
        prompt = self.SUMMARY_PROMPT.format(search_query=query, document=doc)
        resp = self.openai.chat.completions.create(
            model=self.cfg.summariser_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.0,
        )
        text = resp.choices[0].message.content
        return text.split("[Exacted Content]:")[-1].strip()


# ───────────────────────────  CLI (optional)  ─────────────────────────
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("question", type=str)
    ap.add_argument("--model-url", required=True)
    ap.add_argument("--serper-key", type=str, help="override SERPER_API_KEY env")
    args = ap.parse_args()

    cfg = R1SearchConfig(serper_api_key=args.serper_key or os.getenv("SERPER_API_KEY", ""))
    agent = R1Searcher(cfg, model_url=args.model_url)
    final_prompt, issued_queries = agent.run(args.question)  # tokenizer optional
    ans = final_prompt.split("<answer>")[-1].split("</answer>")[0]
    print(ans)
