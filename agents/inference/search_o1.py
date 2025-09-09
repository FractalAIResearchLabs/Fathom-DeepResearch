#!/usr/bin/env python3
"""o1_searcher_inference.py — Serper-based Search-o1 re-implementation
with original summarisation workflow and small robustness fixes.

Changes vs. previous:
- No hard-coded tokenizer at import time.
- Optional tokenizer is accepted via run(..., tokenizer=None).
- Token budgeting uses self.tokenizer if given; otherwise a safe cap.
- No hard-coded summary_url; configurable in O1Cfg and defaults to model_url.
"""
from __future__ import annotations

import os, re, json, time, string
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import requests, trafilatura
import threading
from openai import OpenAI

# -----------------------------------------------------------------------------
# Optional NLTK sentence tokenizer (fallback to regex)
try:
    from nltk.tokenize import sent_tokenize  # type: ignore
except Exception:
    def sent_tokenize(x: str):
        return re.split(r"(?<=[.!?]) +", x)

def _oa() -> OpenAI:
    th = threading.current_thread()
    if not hasattr(th, "_oa"):
        th._oa = OpenAI()  # uses OPENAI_API_KEY from env
    return th._oa

# -----------------------------------------------------------------------------
# Special tags & constants
BEGIN_SEARCH_QUERY  = "<|begin_search_query|>"
END_SEARCH_QUERY    = "<|end_search_query|>"
BEGIN_DOCUMENT_QUERY = "<|begin_of_document|>"
END_DOCUMENT_QUERY   = "<|end_of_document|>"
THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
EOS_TOKEN  = "<|im_end|>"
ANSWER_OPEN, ANSWER_CLOSE = "<answer>", "</answer>"
STOP_STRINGS = [END_SEARCH_QUERY, ANSWER_CLOSE, EOS_TOKEN, "<|endoftext|>"]
ALLOWED_DATASETS = {"musique", "frames", "simpleqa", "browsercomp"}

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

# -----------------------------------------------------------------------------
# Helper functions
def remove_punc(t: str) -> str:
    return t.translate(str.maketrans("", "", string.punctuation))

def f1(a: set, b: set) -> float:
    inter = len(a & b)
    return 0.0 if inter == 0 else 2 * inter / (len(a) + len(b))

def extract_snippet_ctx(text: str, snippet: str, win: int = 2500) -> str:
    """Return window-sized context around the sentence most similar to snippet."""
    text = text[:50_000]
    sn_set = set(remove_punc(snippet.lower()).split())
    best, best_score = None, 0.20
    for sent in sent_tokenize(text):
        score = f1(sn_set, set(remove_punc(sent.lower()).split()))
        if score > best_score:
            best, best_score = sent, score
    if best:
        pos = text.find(best)
        return text[max(0, pos - win): pos + len(best) + win]
    return text[: 2 * win]

# -----------------------------------------------------------------------------
# Config dataclass
@dataclass
class O1Cfg:
    serper_api_key: str = os.getenv("SERPER_API_KEY", "")
    gl: str = "us"; hl: str = "en"
    top_k: int = 10; max_doc_len: int = 3000
    max_search: int = 10; max_turn: int = 15
    use_jina: bool = True
    jina_tpl: str = "https://r.jina.ai/http://{}"
    # generation params
    temperature: float = 0.7; top_p: float = 0.8; top_k_sampling: int = 20
    rep_pen: float = 1.05; thinker_max_tokens: int = 32768
    summariser_model: str = "gpt-4o-mini"
    # optional local summariser model endpoint (defaults to thinker_url if None)
    summary_url: Optional[str] = None

# -----------------------------------------------------------------------------
# Serper search + page fetch
def serper_search(q: str, num: int, key: str, gl="us", hl="en") -> List[Dict]:
    hdr = {"X-API-KEY": key, "Content-Type": "application/json"}
    body = {"q": q, "num": num, "gl": gl, "hl": hl}
    r = requests.post("https://google.serper.dev/search", json=body, headers=hdr, timeout=20)
    r.raise_for_status()
    return r.json().get("organic", [])

def fetch_page(url: str, cfg: O1Cfg, snippet: str = "") -> str:
    try:
        txt = ""
        if cfg.use_jina:
            r = requests.get(cfg.jina_tpl.format(url), timeout=15)
            if r.ok and len(r.text.strip()) > 100:
                txt = r.text.strip()
        if txt == "":
            r = requests.get(url, timeout=15); r.raise_for_status()
            txt = trafilatura.extract(r.text, output_format="txt") or ""
        if snippet:
            txt = extract_snippet_ctx(txt, snippet, cfg.max_doc_len)
        return txt
    except Exception:
        return ""

# -----------------------------------------------------------------------------
# replace_recent_steps (unchanged)
def replace_recent_steps(origin: str, patch: str) -> str:
    step_re = re.compile(r"Step\s+(\d+):\s*")
    def parse(block: str) -> Dict[int, str]:
        cur, buf, out = None, [], {}
        for line in block.splitlines():
            m = step_re.match(line)
            if m:
                if cur is not None: out[cur] = "\n".join(buf).strip()
                cur, buf = int(m.group(1)), [line[m.end():].strip()]
            elif cur is not None:
                buf.append(line)
        if cur is not None: out[cur] = "\n".join(buf).strip()
        return out
    base, mod = parse(origin), parse(patch)
    for k, v in mod.items():
        if "DELETE THIS STEP" in v: base.pop(k, None)
        else: base[k] = v
    return "\n\n".join(base[k] for k in sorted(base))

# -----------------------------------------------------------------------------
class O1Searcher:
    get_webpage_to_reasonchain_instruction = """**Task Instruction:**

    You are tasked with reading and analyzing web pages based on the following inputs: **Previous Reasoning Steps**, **Current Search Query**, and **Searched Web Pages**. Your objective is to extract relevant and helpful information for **Current Search Query** from the **Searched Web Pages** and seamlessly integrate this information into the **Previous Reasoning Steps** to continue reasoning for the original question.

    **Guidelines:**

    1. **Analyze the Searched Web Pages:**
    - Carefully review the content of each searched web page.
    - Identify factual information that is relevant to the **Current Search Query** and can aid in the reasoning process for the original question.

    2. **Extract Relevant Information:**
    - Select the information from the Searched Web Pages that directly contributes to advancing the **Previous Reasoning Steps**.
    - Ensure that the extracted information is accurate and relevant.

    3. **Output Format:**
    - **If the web pages provide helpful information for current search query:** Present the information beginning with **Final Information** as shown below.
    **Final Information**

    [Helpful information]

    - **If the web pages do not provide any helpful information for current search query:** Output the following text.

    **Final Information**

    No helpful information found.

    **Inputs:**
    - **Previous Reasoning Steps:**  
    {prev_reasoning}

    - **Current Search Query:**  
    {search_query}

    - **Searched Web Pages:**  
    {document}

    Now you should analyze each web page and find helpful information based on the current search query {search_query} and previous reasoning steps.
    Return the Helpful information in the <information></information> tags
    """
    SUMMARY_PROMPT = (
        """## Task Description:\n"
        "Given the search query and the content of the searched webpage, "
        "extract information relevant to the query and write one summary paragraph."\n\n"
        "## Guidelines:\n"
        "(1) The extracted content should be relevant to the query.\n"
        "(2) The form of the extracted content **must be a summary paragraph** rather than a direct answer.\n"
        "(3) If the webpage content is unrelated to the query, output \"None\".\n\n"
        "## Output Format:\n"
        "[Exacted Content]: <summary‑paragraph‑or‑None>\n\n"
        "## Inputs:\n"
        "[Search Query]\n{search_query}\n\n"
        "[Webpage Content]\n{document}\n\n"
        "## Output:\n"""
    )

    sys_prompt_multiqa = (
        "You are a reasoning assistant with the ability to perform web searches to help "
        "you answer the user's question accurately. You have special tools:\n\n"
        "- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\n"
        "Then, the system will search and analyze relevant web pages, then provide you with helpful information in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n\n"
        f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to 16.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Example:\n"
        "Question: \"Alice David is the voice of Lara Croft in a video game developed by which company?\"\n"
        "Assistant thinking steps:\n"
        "- I need to find out who voices Lara Croft in the video game.\n"
        "- Then, I need to determine which company developed that video game.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>Alice David Lara Croft voice<|end_search_query|>\n\n"
        "(System returns processed information from relevant web pages)\n\n"
        "Assistant thinks: The search results indicate that Alice David is the voice of Lara Croft in a specific video game. Now, I need to find out which company developed that game.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>video game developed by Alice David Lara Croft<|end_search_query|>\n\n"
        "(System returns processed information from relevant web pages)\n\n"
        "Assistant continues reasoning with the new information...\n\n"
        "Remember:\n"
        "- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.\n"
        "- When done searching, continue your reasoning.\n\n"
        "Always give you final answer between <answer></answer> tags"
    )

   
    def __init__(self, cfg: O1Cfg, thinker_url: str):
        if not cfg.serper_api_key:
            raise ValueError("SERPER_API_KEY required")
        self.cfg = cfg
        self.model_url = thinker_url.rstrip("/")
        self.summary_url = (cfg.summary_url or self.model_url).rstrip("/")
        self.search_cache: Dict[str, List[Dict]] = {}
        self.page_cache: Dict[Tuple[str, str], str] = {}
        self.openai = _oa()
        self.tokenizer = None  # set in run(...)

    # --- low-level generation call ------------------------------------------
    @retry(4, 1)
    def _generate(self, prompt: str, tokenizer) -> str:
        # budget tokens if tokenizer available
        if tokenizer is not None:
            try:
                ids = self.tokenizer(prompt, return_tensors=None, add_special_tokens=False)["input_ids"]
                max_left = max(128, self.cfg.thinker_max_tokens - len(ids) - 100)
            except Exception:
                max_left = 8192
        else:
            max_left = 8192

        resp = requests.post(
            f"{self.model_url}/generate",
            json={
                "text": prompt,
                "sampling_params": {
                    "temperature": self.cfg.temperature,
                    "top_p": self.cfg.top_p,
                    "max_new_tokens": max_left,
                    "repetition_penalty": self.cfg.rep_pen,
                    "stop": STOP_STRINGS,
                },
            },
            timeout=60,
        ).json()
        print(resp)
        # return resp.get("text", "")
        generated = resp["text"]               
        matched   = resp["meta_info"]["finish_reason"].get("matched")
        reason = resp["meta_info"]["finish_reason"].get("type")

        if reason == "stop" and matched in STOP_STRINGS:
            if not "<|end_of_query|>" in generated:
                generated += matched + EOS_TOKEN
        if reason == "stop" and matched == 151645:
             if not generated.endswith("<|im_end|>"):
                generated += "<|im_end|>"
        if reason == "stop" and matched == 151643:
             if not generated.endswith("<|endoftext|>"):
                generated += "<|endoftext|>"
        return generated

    def _generate_summary(self, prompt: str, tokenizer) -> str:
        # summary uses same budgeting rule
        if tokenizer is not None:
            try:
                ids = tokenizer(prompt, return_tensors=None, add_special_tokens=False)["input_ids"]
                max_left = max(256, self.cfg.thinker_max_tokens - len(ids) - 100)
            except Exception:
                max_left = 2048
        else:
            max_left = 2048

        resp = requests.post(
            f"{self.summary_url}/generate",
            json={
                "text": prompt,
                "sampling_params": {
                    "temperature": self.cfg.temperature,
                    "max_new_tokens": max_left,
                    "stop": STOP_STRINGS,
                },
            },
            timeout=60,
        ).json()
        # return resp.get("text", "")
        generated = resp["text"]                       # what you have now
        matched   = resp["meta_info"]["finish_reason"].get("matched")
        reason = resp["meta_info"]["finish_reason"].get("type")
  
        # ⇢ append the tag back only if it was removed
        if reason == "stop" and matched in STOP_STRINGS:
            if not "<|end_of_query|>" in generated:
                generated += matched
        if reason == "stop" and matched == 151645:
             if not generated.endswith("<|im_end|>"):
                generated += "<|im_end|>"
        if reason == "stop" and matched == 151643:
             if not generated.endswith("<|endoftext|>"):
                generated += "<|endoftext|>"
        
        return generated

    def _summarise_openai(self, query: str, doc: str) -> str:
        prompt = self.SUMMARY_PROMPT.format(search_query=query, document=doc)
        resp = self.openai.chat.completions.create(
            model=self.cfg.summariser_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.0,
        )
        text = resp.choices[0].message.content
        return text.split("[Exacted Content]:")[-1].strip()

    # --- public entry -------------------------------------------------------
    def run(self, question: str, tokenizer=None):
        """tokenizer is optional; pass HF tokenizer from the harness if available."""
        self.tokenizer = tokenizer

        prompt = (
            f"<|im_start|>system\n{self.sys_prompt_multiqa}<|im_end|>\n"
            f"<|im_start|>user\n{question}<|im_end|>\n"
            f"<|im_start|>assistant\n{THINK_OPEN}"
        )
        queries: List[str] = []
        seen: set[str] = set()

        for _ in range(self.cfg.max_turn):
            chunk = self._generate(prompt, tokenizer)
            prompt += chunk

            if ANSWER_CLOSE in chunk:
                break

            query = self._extract_query(chunk)
            if not query or len(queries) >= self.cfg.max_search:
                break
            if query in seen:
                continue
            seen.add(query)
            queries.append(query)

            doc = self._retrieve_doc(query)
            prev_reasoning = self._extract_reasoning(prompt)
            summary_block = (
                "\n<|im_start|>user" +
                self._summarise(prev_reasoning, query, doc, tokenizer) +
                EOS_TOKEN + "\n<|im_start|>assistant" + THINK_OPEN
            )
            prompt += summary_block
        else:
            prompt += f"{ANSWER_OPEN}I don't know.{ANSWER_CLOSE}"

        return prompt, queries

    # ---------------------------------------------------------------------
    # helpers
    def _extract_query(self, txt: str) -> Optional[str]:
        if BEGIN_SEARCH_QUERY not in txt or END_SEARCH_QUERY not in txt:
            return None
        frag = txt.split(BEGIN_SEARCH_QUERY)[-1].split(END_SEARCH_QUERY)[0]
        return re.sub(r'[\"\'…\t]', " ", frag.split("<|")[0]).strip()

    def _retrieve_doc(self, query: str) -> str:
        if query not in self.search_cache:
            self.search_cache[query] = serper_search(
                query, self.cfg.top_k, self.cfg.serper_api_key, gl=self.cfg.gl, hl=self.cfg.hl
            )
        for hit in self.search_cache[query]:
            url, sn = hit.get("link", ""), hit.get("snippet", "")
            if not url:
                continue
            key = (url, sn)
            if key not in self.page_cache:
                self.page_cache[key] = fetch_page(url, self.cfg, sn)
            if self.page_cache[key]:
                return self.page_cache[key]
        return ""

    def _summarise(self, prev: str, query: str, doc: str, tokenizer) -> str:
        rid_prompt = self.get_webpage_to_reasonchain_instruction.format(
            prev_reasoning=prev, search_query=query, document=doc
        )
        chat = f"<|im_start|>user\n{rid_prompt}\n<|im_end|>\n<|im_start|>assistant\n"
        resp = self._generate_summary(chat, tokenizer)
        return BEGIN_DOCUMENT_QUERY + self._extract_summary(resp) + END_DOCUMENT_QUERY

    def _extract_summary(self, txt: str) -> str:
        if "<information>" in txt:
            return txt.split("<information>")[-1].split("</information>")[0]
        m = re.search(r"\*\*Final Information\*\*\s*\n(.+?)<\|im_end\|>", txt, re.S)
        if m:
            return m.group(1).strip()
        return txt

    def _extract_reasoning(self, prompt: str) -> str:
        return prompt.split(THINK_OPEN)[-1].split(THINK_CLOSE)[0] if THINK_OPEN in prompt else ""
