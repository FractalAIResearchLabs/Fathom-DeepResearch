from __future__ import annotations
from config import CFG, _SESS, _RND
import logging
import re
from bs4 import BeautifulSoup
import functools
import random
import requests
import trafilatura
import time 
from web_helpers import retry, fetch_blocked_site     # ⬅️ shared


def fetch_github(url: str) -> str:
    def _markdown_cleanup(md: str) -> str:
        md = re.sub(r"```.*?```", "", md, flags=re.S)
        md = re.sub(r"^#+\s*", "", md, flags=re.M)
        return re.sub(r"[ \t]{2,}", " ", md).strip()

    hdr = {"User-Agent": "ii-research-bot/0.6"}
    try:
        m = re.match(r"https?://github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$", url)
        if m:
            owner, repo = m.groups()
            api = f"https://api.github.com/repos/{owner}/{repo}/readme"
            hdr_api = hdr | {"Accept": "application/vnd.github.v3.raw"}
            if (tok := os.getenv("GITHUB_TOKEN")):
                hdr_api["Authorization"] = f"Bearer {tok}"
            r = _SESS.get(api, headers=hdr_api, timeout=(CFG.connect_to, CFG.read_to))
            if r.ok and len(r.text) > 30:
                return _markdown_cleanup(r.text)[:CFG.text_cap]
        
        if "/blob/" in url or "/tree/" in url:
            raw = re.sub(
                r"https://github\.com/([^/]+)/([^/]+)/(?:blob|tree)/",
                r"https://raw.githubusercontent.com/\\1/\\2/",
                url,
                count=1,
            ).split("?", 1)[0]
            r = _SESS.get(raw, headers=hdr, timeout=(CFG.connect_to, CFG.read_to))
            if r.ok and "text" in (r.headers.get("content-type") or "") and len(r.text) > 0:
                return r.text[:CFG.text_cap]
            
            raw1 = url + ("?raw=1" if "?" not in url else "&raw=1")
            r = _SESS.get(raw1, headers=hdr, timeout=(CFG.connect_to, CFG.read_to))
            if r.ok and "text" in (r.headers.get("content-type") or "") and len(r.text) > 0:
                return r.text[:CFG.text_cap]
            
            plain = url + ("?plain=1" if "?" not in url else "&plain=1")
            html = _SESS.get(plain, headers=hdr, timeout=(CFG.connect_to, CFG.read_to)).text
            soup = BeautifulSoup(html, "lxml")
            pre = soup.find("pre")
            if pre and len(pre.text) > 10:
                return pre.text[:CFG.text_cap]
        
        if "raw.githubusercontent.com" in url:
            r = _SESS.get(url.split("?", 1)[0], headers=hdr, timeout=(CFG.connect_to, CFG.read_to))
            if r.ok and "text" in (r.headers.get("content-type") or ""):
                return "[Retrieved from raw.githubusercontent.com]" + r.text[:CFG.text_cap]
        
        raise RuntimeError("github: unable to retrieve plain text")
    except Exception as e:
        logging.error(f"GitHub fetch failed for {url}: {e}")
        return _fetch_blocked_site(url)