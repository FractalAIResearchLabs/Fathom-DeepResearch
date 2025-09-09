from __future__ import annotations
import asyncio, logging, re, tiktoken
from typing import Dict, List
from config import CFG, _SESS
from fetchers_async import fetch_url
from web_helpers import retry
from urllib.parse import urlparse

enc = tiktoken.get_encoding("cl100k_base")

# ── Google / Serper search ──────────────────────────────────────────────

# def google_search(query: str, top_k: int = 10) -> List[Dict[str,str]]:
#     if not CFG.serper_key:
#         raise EnvironmentError("SERPER_API_KEY not set")
#     r = _SESS.post(
#         CFG.serper_ep,
#         headers={"X-API-KEY": CFG.serper_key, "Content-Type": "application/json"},
#         json={"q": query}, timeout=20)
#     r.raise_for_status()
#     hits = []
#     for it in r.json().get("organic", []):
#         hits.append({"title": it.get("title",""),
#                      "link":  it.get("link",""),
#                      "snippet": it.get("snippet","")})
#         if len(hits) == top_k: break
#     return hits
import hashlib, json, logging, os, time
from typing import List, Dict

def _canon_query(q: str) -> str:
    # Normalize whitespace to avoid duplicate keys for e.g. "foo  bar"
    return " ".join((q or "").strip().split())


def _search_cache_key(query: str, top_k: int) -> str:
    cq = _canon_query(query)
    raw = f"{top_k}|{cq}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest() + ".json"

def _search_cache_paths(query: str, top_k: int) -> str:
    root = CFG.serper_cache_dir
    os.makedirs(root, exist_ok=True)
    return os.path.join(root, _search_cache_key(query, top_k))

def _ttl_seconds() -> int:
    # 0 or missing → no expiry
    try:
        return int(getattr(CFG, "search_cache_ttl", 0) or int(os.environ.get("SEARCH_CACHE_TTL", "0")))
    except Exception:
        return 0

def _load_search_cache(path: str) -> List[Dict[str, str]] | None:
    try:
        if not os.path.exists(path) or os.path.getsize(path) <= 2:
            return None
        ttl = _ttl_seconds()
        if ttl > 0:
            age = time.time() - os.path.getmtime(path)
            if age > ttl:
                return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Basic shape check: list of dicts with expected keys
        if isinstance(data, list):
            return data
    except Exception as e:
        logging.debug("Serper cache read failed (%s): %s", path, e)
    return None

def _save_search_cache(path: str, hits: List[Dict[str, str]]) -> None:
    try:
        tmp = f"{path}.tmp.{os.getpid()}"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(hits, f, ensure_ascii=False)
        os.replace(tmp, path)  # atomic on same FS
    except Exception as e:
        logging.debug("Serper cache write failed (%s): %s", path, e)
        
    
@retry
def google_search(query: str, top_k: int = 10) -> List[Dict[str,str]]:
    if not CFG.serper_key:
        raise EnvironmentError("SERPER_API_KEY not set")

    cpath = _search_cache_paths(query, top_k)
    cached = _load_search_cache(cpath)
    if cached is not None:
        logging.info("Serper search (cache hit) ← %r (top_k=%d)", _canon_query(query), top_k)
        return cached

    r = _SESS.post(
        CFG.serper_ep,
        headers={"X-API-KEY": CFG.serper_key, "Content-Type": "application/json"},
        json={"q": query},
        timeout=20
    )
    r.raise_for_status()
    hits: List[Dict[str, str]] = []
    for it in r.json().get("organic", []):
        hits.append({
            "title": it.get("title", ""),
            "link": it.get("link", ""),
            "snippet": it.get("snippet", ""),
        })
        if len(hits) == top_k:
            break

    _save_search_cache(cpath, hits)
    return hits


# ── async extract per hit ───────────────────────────────────────────────
async def async_search_and_extract(query: str, top_k: int = 5) -> List[Dict]:
    hits = google_search(query, top_k)
    async def enrich(h):
        return {**h, "body": await fetch_url(h["link"])}
    return await asyncio.gather(*(enrich(h) for h in hits))

# ── markdown helpers ────────────────────────────────────────────────────
def url_hits_to_markdown(hits: List[Dict[str,str]]) -> str:
    out = []
    for i, h in enumerate(hits, 1):
        out.append(f"### {i}. {h['title']}\n**URL**: {h['link']}\n\n**Snippet**: {h['snippet']}\n")
    return "\n---\n\n".join(out)

def search_result_to_markdown(blocks: List[Dict]) -> str:
    out = []
    for i, b in enumerate(blocks, 1):
        out.append(f"### {i}. **Title**: {b['title']}\n**URL**: {b['link']}\n\n"
                   f"**Snippet**: {b['snippet']}\n\n**Content**:\n{b['body']}\n")
    return "\n---\n\n".join(out)

def trim_to_tokens(text: str, limit: int, model: str = "gpt-3.5-turbo") -> str:
    ids = enc.encode(text)
    if len(ids) <= limit: return text
    keep = limit // 2
    return enc.decode(ids[:keep] + ids[-keep:])

def _bad(url: str) -> str|None:
    p = urlparse(url)
    if p.scheme not in ("http","https") or not p.netloc:
        return "[error: invalid URL – must start with http:// or https://]"
    return None

