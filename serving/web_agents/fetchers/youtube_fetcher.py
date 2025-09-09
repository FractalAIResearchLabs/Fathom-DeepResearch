from __future__ import annotations
import logging, re
from config import CFG, _SESS
from web_helpers import retry, fetch_blocked_site

try:
    import yt_dlp
    _HAS = True
except ImportError:
    _HAS = False

_LANGS = ["en", "en-US"]

@retry
def fetch_youtube(url: str) -> str:
    if not _HAS:
        return fetch_blocked_site(url)[:CFG.text_cap]

    try:
        ydl_opts = {"quiet": True, "no_warnings": True,
                    "writesubtitles": True, "writeautomaticsub": True,
                    "skip_download": True}
        with yt_dlp.YoutubeDL(ydl_opts) as y:
            info = y.extract_info(url, download=False)

        subs = info.get("subtitles", {}) or {}
        auto = info.get("automatic_captions", {}) or {}
        tracks = next((subs.get(l) or auto.get(l) for l in _LANGS
                       if subs.get(l) or auto.get(l)), None)
        if not tracks:
            tracks = next(iter(subs.values()), []) or next(iter(auto.values()), [])

        if tracks:
            cap_url = tracks[0]["url"]
            if "fmt=" not in cap_url: cap_url += "&fmt=json3"
            r = _SESS.get(cap_url, timeout=(CFG.connect_to, CFG.read_to))
            r.raise_for_status()
            if cap_url.endswith(".vtt"):
                text = " ".join(l for l in r.text.splitlines()
                                if l and "-->" not in l and not re.match(r"\d{2}:\d{2}", l))
            else:
                text = " ".join(seg["utf8"] for ev in r.json()["events"]
                                for seg in ev.get("segs", []))
            if text: return text[:CFG.text_cap]

        meta = (info.get("title","") + "\n\n" + info.get("description","")).strip()
        return "[Retrieved from yt-dlp] " + meta[:CFG.text_cap]
    except Exception as e:
        logging.error("YouTube fetch failed %s: %s", url, e)
        return fetch_blocked_site(url)[:CFG.text_cap]
