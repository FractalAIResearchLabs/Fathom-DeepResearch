"""
Asynchronous wrapper around **Crawl4AI** so that other coroutines can await a
single call – identical to the previous implementation but isolated in its own
module to satisfy clean‑architecture / layering.

Public API
==========
async def fetch_crawl4ai(url: str) -> str
    Returns markdown extracted by Crawl4AI or raises `RuntimeError` on failure.
"""
from __future__ import annotations

import asyncio, logging
from dataclasses import dataclass, field
from typing import Any

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from config import CFG

# ----------------------------------------------------------------------------
_MAX_CONCURRENT_PAGES = 6
_MAX_ATTEMPTS = 5
_RETRYABLE = (
    "handler is closed",
    "browser has disconnected",
    "transport closed",
    "target crashed",
)

# Globals bound to the *event‑loop* currently active
_CRAWLER: AsyncWebCrawler | None = None
_CRAWLER_LOOP: asyncio.AbstractEventLoop | None = None
_SEMAPHORES: dict[asyncio.AbstractEventLoop, asyncio.Semaphore] = {}
_CFG = CrawlerRunConfig(markdown_generator=DefaultMarkdownGenerator())


def _get_semaphore() -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    if loop not in _SEMAPHORES:
        _SEMAPHORES[loop] = asyncio.Semaphore(_MAX_CONCURRENT_PAGES)
    return _SEMAPHORES[loop]


async def _ensure_crawler() -> None:
    global _CRAWLER, _CRAWLER_LOOP
    loop = asyncio.get_running_loop()
    if _CRAWLER is None or loop is not _CRAWLER_LOOP:
        if _CRAWLER is not None:
            try:
                await _CRAWLER.aclose()
            except Exception:
                pass
        _CRAWLER = AsyncWebCrawler()
        await _CRAWLER.__aenter__()
        _CRAWLER_LOOP = loop


@dataclass
class _Page:
    success: bool
    markdown: str | None = None
    error: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


async def _crawl_once(url: str) -> _Page:
    global _CRAWLER
    await _ensure_crawler()

    try:
        result = await _CRAWLER.arun(url, config=_CFG)
        if result.success and result.markdown:
            return _Page(True, result.markdown, meta=result.__dict__)
        return _Page(False, error=result.error_message or "no markdown")
    except Exception as exc:
        return _Page(False, error=str(exc))


async def fetch_crawl4ai(url: str) -> str:
    """Return markdown extracted by Crawl4AI or raise on failure."""
    sem = _get_semaphore()
    async with sem:
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            page = await _crawl_once(url)
            if page.success and page.markdown:
                print(len(page.markdown))
                return "[Retrieved from Craw4AI]" + page.markdown[:CFG.text_cap]

            err = page.error or "unknown"
            logging.warning("Crawl4AI attempt %d/%d failed: %s", attempt, _MAX_ATTEMPTS, err)

            if attempt < _MAX_ATTEMPTS and any(p in err.lower() for p in _RETRYABLE):
                # reset shared browser & retry after back‑off
                global _CRAWLER
                try:
                    await _CRAWLER.aclose()
                except Exception:
                    pass
                _CRAWLER = None
                await asyncio.sleep(1.5 * attempt)
                continue

            raise RuntimeError(err)