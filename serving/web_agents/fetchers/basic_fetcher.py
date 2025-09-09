from __future__ import annotations
import logging
from urllib.parse import unquote
from config import CFG, _SESS
from web_helpers import extract_main_text, fetch_blocked_site

_BINARY = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".zip", ".tar",
           ".gz", ".mp3", ".mp4", ".mkv", ".exe")

_ERROR = ["wrong", "error", "try again"]

def _looks_like_error(txt):
    if len(txt) < 300:
        for err in _ERROR:
            if err in txt:
                return True
    return False 


def fetch_html(url: str) -> str:
    if url.lower().endswith(_BINARY):
        return "[binary omitted]"
    try:
        r = _SESS.get(url, stream=True, timeout=(CFG.connect_to, CFG.read_to))
        r.raise_for_status()
        ctype = (r.headers.get("content-type") or "").lower()
        if "pdf" in ctype or not ("text" in ctype or "html" in ctype):
            return "[binary omitted]"
        raw = r.raw.read(CFG.stream_html_cap, decode_content=True)
        html = raw.decode(r.encoding or "utf-8", errors="ignore")
        txt  = extract_main_text(html).strip()
        if "wikipedia.org" in url:
            slug = unquote(url.rsplit("/", 1)[-1]).replace("_", " ")
            if slug.lower() not in txt.lower():
                txt = f"{slug}\n\n{txt}"
        if _looks_like_error(txt):
            return f"[Error fetching url: {url}]"
        else:
            return "[Retrived using HTML] " + txt
    except Exception as e:
        logging.error("Generic fetch failed %s: %s", url, e)
        return fetch_blocked_site(url)
