# """
# fetchers_async.py – Orchestrates multiple specialised fetchers **without changing
# its public surface** (`async def fetch_url(url: str) -> str`).

# Order of strategies (after specialised handlers):
#     1. **Jina AI**         – fast & cheap full‑text extraction
#     2. **Crawl4AI**        – browser‑based heavy‑weight fallback
#     3. **Legacy HTML**     – trafilatura / readability last‑chance scrape

# Specialised fetchers (PDF, YouTube, Reddit) remain unchanged.
# """
# from __future__ import annotations

# import asyncio, logging
# from typing import Callable

# from web_helpers import retry
# from fetchers.pdf_fetcher import fetch_pdf
# from fetchers.youtube_fetcher import fetch_youtube
# from fetchers.reddit_fetcher import fetch_reddit
# from fetchers.github_fetcher import fetch_github

# from fetchers.jina_fetcher import fetch_jina
# from fetchers.crawl4ai_fetcher import fetch_crawl4ai
# from fetchers.basic_fetcher import fetch_html


# _ERR_PREFIXES = ("[error", "[failed", "[unable")


# def _looks_error(txt: str | None) -> bool:
#     return not txt or txt.strip().lower().startswith(_ERR_PREFIXES)


# async def _thread_wrapper(fn: Callable[[str], str], url: str) -> str | None:
#     try:
#         return await asyncio.to_thread(fn, url)
#     except Exception as exc:
#         logging.debug("%s threw in thread: %s", fn.__name__, exc)

# @retry
# async def fetch_url(url: str) -> str:
#     url_l = url.lower()
    

#     # 1 – Jina AI ------------------------------------------------------------
#     if (out := await _thread_wrapper(fetch_jina, url)) and not _looks_error(out):
#         return out
    
#     # if (out := await _thread_wrapper(fetch_html, url)) and not _looks_error(out):
#     #     return out

#     # 2 – Crawl4AI -----------------------------------------------------------
#     try:
#         md = await fetch_crawl4ai(url)
#         if not _looks_error(md):
#             return md
#     except Exception as e:
#         logging.debug("Crawl4AI failed: %s", e)
        
#     if "pdf" in url_l:
#         if (out := await _thread_wrapper(fetch_pdf, url)) and not _looks_error(out):
#             return out
        
#     if "reddit" in url_l:
#         if (out := await _thread_wrapper(fetch_reddit, url)) and not _looks_error(out):
#             return out
#     if "youtube" in url_l:
#         if (out := await _thread_wrapper(fetch_youtube, url)) and not _looks_error(out):
#             return out
#     if "github" in url_l:
#         if (out := await _thread_wrapper(fetch_github, url)) and not _looks_error(out):
#             return out

#     # 3 – Basic HTML --------------------------------------------------------
#     if (out := await _thread_wrapper(fetch_html, url)) and not _looks_error(out):
#         return out

#     return "[error fetch_url exhausted all methods]"



import asyncio, logging, time

from fetchers.pdf_fetcher     import fetch_pdf
from fetchers.youtube_fetcher import fetch_youtube
from fetchers.reddit_fetcher  import fetch_reddit
from fetchers.github_fetcher  import fetch_github
from fetchers.jina_fetcher    import fetch_jina
from fetchers.crawl4ai_fetcher import fetch_crawl4ai
from fetchers.basic_fetcher   import fetch_html

_ERR_PREFIXES = ("[error", "[failed", "[unable]")

def _looks_error(txt: str | None) -> bool:
    return not txt or txt.strip().lower().startswith(_ERR_PREFIXES)

# per-fetcher hard caps (seconds)
_FETCHER_TIMEOUTS = {
    "fetch_jina":      20.0,
    "fetch_github":    10.0,
    "fetch_crawl4ai":  40.0,
    "fetch_html":      20.0,
    "fetch_pdf":       30.0,
    "fetch_youtube":   30.0,
    "fetch_reddit":    10.0,
}


async def fetch_url(url: str) -> str:
    url_l = url.lower()

    async def timed_fetch(fn) -> str | None:
        name     = fn.__name__
        timeout  = _FETCHER_TIMEOUTS.get(name, 60.0)
        start_ts = time.perf_counter()
        try:
            # choose sync or async execution path
            coro = fn(url) if asyncio.iscoroutinefunction(fn) else asyncio.to_thread(fn, url)
            result = await asyncio.wait_for(coro, timeout=timeout)
            elapsed = (time.perf_counter() - start_ts) * 1000
            if result and not _looks_error(result):
                logging.info(f"[{name}] ✅ success in {elapsed:.1f} ms")
                return result
            logging.warning(f"[{name}] ❌ error response in {elapsed:.1f} ms")
        except asyncio.TimeoutError:
            logging.warning(f"[{name}] ⏱️ timed-out after {timeout}s")
        except Exception as e:
            elapsed = (time.perf_counter() - start_ts) * 1000
            logging.warning(f"[{name}] 💥 exception in {elapsed:.1f} ms → {e}")
        return None

    async def try_chain(*fetchers) -> str | None:
        for fn in fetchers:
            if result := await timed_fetch(fn):
                return result
        return None

    # # -------------- domain-specific chains ---------------
    if "github.com"   in url_l:
        return await try_chain(fetch_jina, fetch_github,  fetch_crawl4ai)
    if "wikipedia.org" in url_l:
        return await try_chain(fetch_html, fetch_jina,     fetch_crawl4ai)
    if "reddit.com"   in url_l:
        return await try_chain(fetch_jina, fetch_reddit,   fetch_html)
    if "quora.com"    in url_l:
        return await try_chain(fetch_crawl4ai, fetch_jina, fetch_html)
    if "youtube.com"  in url_l or "youtu.be" in url_l:
        return await try_chain(fetch_jina, fetch_youtube)
    if url_l.endswith(".pdf") or "pdf" in url_l:
        return await try_chain(fetch_jina, fetch_pdf, fetch_html, fetch_crawl4ai)
  

   
    # return await try_chain(fetch_jina) or "[error could not load page]"
   

    # -------------- generic fallback ---------------------
    return (await try_chain(fetch_jina, fetch_crawl4ai, fetch_html)
            or "[error fetch_url exhausted all methods]")
