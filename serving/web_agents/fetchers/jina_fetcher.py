# """
# Jina AI powered web-page fetcher.

# Provides `fetch_jina(url: str) -> str` which returns a **plain-text or markdown** body
# prefixed with `[Retrieved from Jina AI]` so callers can recognise the source.
# If the Jina endpoint cannot return usable text (HTTP error, short / empty body, etc.)
# this function raises an Exception – letting the orchestrator fall back to other
# fetchers.

# The implementation is **stateless** and thread-safe – no global mutable state is
# kept apart from the shared requests session from `config` (mirroring the rest of
# the code-base).
# """

# from __future__ import annotations

# import logging
# import os
# import urllib.parse as _u

# from config import CFG, _SESS  # shared requests session and config
# from web_helpers import retry

# _JINA_ENDPOINT = "https://r.jina.ai/{url}"  # Note: will prepend http:// when formatting


# @retry
# def fetch_jina(url: str) -> str:
#     """Return article text extracted by **Jina AI Read API**.

#     Raises:
#         RuntimeError – if the endpoint does not yield usable text
#     """
#     api_url = _JINA_ENDPOINT.format(url=url)
#     headers = {
#         "Authorization": f"Bearer {CFG.jina_key}"
#     }
#     logging.debug("Jina fetch → %s", api_url)
 
#     # Make request
#     r = _SESS.get(api_url, headers=headers, timeout=(CFG.connect_to, CFG.read_to))
#     r.raise_for_status()

#     txt = r.text.strip()

#     # Treat short or errorful body as failure
#     if len(txt) < 200 and any(err in txt.lower() for err in ["403", "forbidden", "error"]):
#         raise RuntimeError("Jina AI returned no content")

#     return "[Retrieved from Jina AI] " + txt[: CFG.text_cap]

"""
Jina AI powered web-page fetcher with URL-based disk cache.

- Cache key: canonicalized URL (sha256)
- Cache location: <CFG.cache_dir or $CACHE_DIR or ".cache">/jina_read/
- Always stores the *raw* Jina body (without the "[Retrieved...]" prefix).
- Atomic writes via os.replace for basic thread/process safety.
"""

from __future__ import annotations

import hashlib
import logging
import os
import urllib.parse as _u
from typing import Tuple

from config import CFG, _SESS  # shared requests session and config
from web_helpers import retry

_JINA_ENDPOINT = "https://r.jina.ai/{url}"  # expects a fully-qualified, url-encoded target


def _canonicalize_url(url: str) -> str:
    """Ensure URL has a scheme and is normalized for caching/API calls."""
    p = _u.urlparse(url.strip())
    if not p.scheme:
        # Default to http if missing; Jina reader prefers explicit scheme.
        p = _u.urlparse("http://" + url.strip())

    # Normalize: lowercase scheme/netloc, drop fragment, keep query & path
    p = p._replace(scheme=p.scheme.lower(), netloc=p.netloc.lower(), fragment="")
    # Ensure path is at least "/"
    path = p.path if p.path else "/"
    return _u.urlunparse((p.scheme, p.netloc, path, "", p.query, ""))


def _cache_paths(nurl: str) -> Tuple[str, str]:
    """Return (cache_dir, cache_file_path) for a normalized URL."""
    cache_root = CFG.jina_cache_dir
    cache_dir = os.path.join(cache_root, "jina_read")
    os.makedirs(cache_dir, exist_ok=True)
    h = hashlib.sha256(nurl.encode("utf-8")).hexdigest()
    return cache_dir, os.path.join(cache_dir, f"{h}.txt")


def _load_from_cache(cpath: str) -> str | None:
    try:
        if os.path.exists(cpath) and os.path.getsize(cpath) > 0:
            with open(cpath, "r", encoding="utf-8") as f:
                return f.read()
    except Exception as e:
        logging.debug("Jina cache read failed (%s): %s", cpath, e)
    return None


def _save_to_cache(cpath: str, body: str) -> None:
    try:
        tmp = f"{cpath}.tmp.{os.getpid()}"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(body)
        os.replace(tmp, cpath)  # atomic on the same filesystem
    except Exception as e:
        logging.debug("Jina cache write failed (%s): %s", cpath, e)


@retry
def fetch_jina(url: str) -> str:
    """Return article text extracted by **Jina AI Read API** with disk cache.

    Raises:
        RuntimeError – if the endpoint does not yield usable text
    """
    nurl = _canonicalize_url(url)
    cache_dir, cpath = _cache_paths(nurl)

    # 1) Try cache
    cached = _load_from_cache(cpath)
    if cached:
        logging.info("Jina fetch (cache hit) ← %s", nurl)
        return "[Retrieved from Jina AI] " + cached[: CFG.text_cap]

    # 2) Fetch from Jina
    api_url = _JINA_ENDPOINT.format(url=_u.quote(nurl, safe=""))
    headers = {"Authorization": f"Bearer {CFG.jina_key}"}
    logging.debug("Jina fetch (cache miss) → %s", api_url)

    r = _SESS.get(api_url, headers=headers, timeout=(CFG.connect_to, CFG.read_to))
    r.raise_for_status()
    body = r.text.strip()

    # 3) Validate
    if len(body) < 200 and any(err in body.lower() for err in ("403", "forbidden", "error")):
        raise RuntimeError("Jina AI returned no content")

    # 4) Save to cache (store the raw body; callers always get the standard prefix)
    _save_to_cache(cpath, body)

    return "[Retrieved from Jina AI] " + body[: CFG.text_cap]

