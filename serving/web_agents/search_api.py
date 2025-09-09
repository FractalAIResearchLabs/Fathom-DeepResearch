from __future__ import annotations
import asyncio
from typing import Dict, List
from utils import (google_search, url_hits_to_markdown,
                   search_result_to_markdown,
                   async_search_and_extract, _bad)
from fetchers_async import fetch_url
from compressor import compress_text, query_text
from config import CFG
import logging

# print("PYTHONPATH:", sys.path)

def web_search(query):
    return search_urls(query = query, top_k=10)

def web_visit(url):
    return open_url(url = url, compress = False)

def web_fetch(url):
    return open_url(url = url, compress = False)

# ── 1. search_urls ──────────────────────────────────────────────────────
def search_urls(query: str, top_k: int = 10) -> str:
    return url_hits_to_markdown(google_search(query, top_k))

# ── 2. open_url ─────────────────────────────────────────────────────────
def open_url(url: str, *, compress: bool = True, pct: float = CFG.pct,
             model: str = "gpt-4o-mini") -> str:
    if _bad(url): return _bad(url)
    try:
        body = asyncio.run(fetch_url(url))
        body = str(body)
    except Exception as e:
        return f"[error fetching URL: {e}]"
    if compress:
        try:
            body = compress_text(body, pct=pct, model=model)
        except Exception as e:
            body = f"[compression failed: {e}]\n\n{body[:2000]}"
    return body

# ── 3. search_and_parse_query ───────────────────────────────────────────
def search_and_parse_query(query: str, top_k: int = 3, *,
                           compress: bool = True, pct: float = CFG.pct) -> str:
    blocks = asyncio.run(async_search_and_extract(query, top_k))
    if compress:
        for b in blocks:
            try:
                cmp = compress_text(b["body"], pct=pct)
                b["body"] = (f"**Summary:**\n{cmp['narrative']}\n\n"
                             f"**Facts:**\n{cmp['facts']}\n\n"
                             f"**Tables:**\n{cmp['tables']}")
            except Exception as e:
                b["body"] = f"[compression failed: {e}]\n\n{b['body']}"
    return search_result_to_markdown(blocks)

# ── 4. query_url ────────────────────────────────────────────────────────
def query_url(url: str, goal: str) -> str:
    query_llm = CFG.query_llm
    if _bad(url): return _bad(url)
    body = asyncio.run(fetch_url(url))
    if not body or body.startswith("[error"):
        return f"[failed to retrieve content from {url}]\n\n{body}"
    return query_text(url, body, goal, model=query_llm)['extracted_info']
    

