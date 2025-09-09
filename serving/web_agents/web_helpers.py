from __future__ import annotations
import functools, logging, random, re, time, requests, trafilatura
from typing import Callable
from bs4 import BeautifulSoup
from config import CFG, _RND

# ── retry ────────────────────────────────────────────────────────────────
def retry(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def _wrap(*a, **kw):
        for i in range(CFG.retries):
            try:
                return fn(*a, **kw)
            except Exception as e:
                if i == CFG.retries - 1:
                    raise
                delay = CFG.backoff * (2 ** i) * (1 + 0.3 * _RND.random())
                logging.warning("Retry %s/%s %s: %s (%.2fs)",
                                i+1, CFG.retries, fn.__name__, e, delay)
                time.sleep(delay)
    return _wrap

# ── text extraction ──────────────────────────────────────────────────────
def extract_main_text(html: str) -> str:
    txt = trafilatura.extract(html, output_format="txt") or ""
    if len(txt) >= 500:
        return txt
    from readability import Document
    soup = BeautifulSoup(Document(html).summary(), "lxml")
    txt  = soup.get_text(" ", strip=True)
    if len(txt) >= 400:
        return txt
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return re.sub(r"\s+", " ", soup.get_text(" ").strip())

# ── last‑chance fetch when everything fails ──────────────────────────────
@retry
def fetch_blocked_site(url: str) -> str:
    hdrs = {"User-Agent": CFG.ua, "Referer": "https://www.google.com/"}
    sess = requests.Session(); sess.headers.update(hdrs)

    # 1) direct
    try:
        r = sess.get(url, timeout=(CFG.connect_to, CFG.read_to))
        r.raise_for_status()
        txt = extract_main_text(r.text)
        if len(txt) > 500:
            return "[Retrieved from redirected attempt]\n\n" + txt[:CFG.text_cap]
    except Exception as e:
        logging.debug("Direct scrape failed %s: %s", url, e)

    # 2) wayback
    try:
        wb = f"https://web.archive.org/web/2023/{url}"
        r  = sess.get(wb, timeout=(CFG.connect_to, CFG.read_to))
        r.raise_for_status()
        txt = extract_main_text(r.text)
        if len(txt) > 500:
            return "[Retrieved from archive.org]\n\n" + txt[:CFG.text_cap]
    except Exception as e:
        logging.debug("Wayback scrape failed %s: %s", url, e)

    return f"[Error accessing {url}. Try VPN or manual archive.org check.]"
