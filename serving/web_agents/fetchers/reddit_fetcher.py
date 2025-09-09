

from __future__ import annotations
from config import CFG, _SESS, _RND 
import logging
import re
from bs4 import BeautifulSoup
import functools
import random
import requests
import time 
import trafilatura
from web_helpers import retry, fetch_blocked_site


_REDDIT_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"


@retry
def _reddit_json_api(url: str) -> str | None:
    api_url = re.sub(r"/comments/([a-z0-9]{6,8}).*", r"/comments/\1.json", url)
    try:
        headers = {"User-Agent": _REDDIT_UA, "Accept": "application/json"}
        r = _SESS.get(
            api_url,
            params={"limit": 5, "depth": 2, "raw_json": 1},
            headers=headers,
            timeout=(CFG.connect_to, CFG.read_to),
        )
        if "blocked" in r.text.lower() or r.status_code != 200:
            return None
        
        data = r.json()
        post_data = data[0]["data"]["children"][0]["data"]
        title = post_data.get("title", "")
        selftext = post_data.get("selftext", "")
        author = post_data.get("author", "")
        
        comments = []
        if len(data) > 1:
            for comment in data[1]["data"]["children"][:50]:
                if comment["kind"] == "t1":
                    c_author = comment["data"].get("author", "")
                    c_body = comment["data"].get("body", "")
                    if c_body:
                        comments.append(f"u/{c_author}: {c_body}")
        
        result = f"Title: {title}\nPosted by: u/{author}\n\n"
        if selftext:
            result += f"{selftext}\n\n"
        if comments:
            result += "Top comments:\n\n" + "\n\n".join(comments)
        
        return result.strip()
    except Exception:
        return None
import urllib.parse as _u

_ID_RE = re.compile(r"([a-z0-9]{6,8})", re.I)

def _extract_post_id(url: str) -> str | None:
    """
    Heuristics to find the 6–8‑char base‑36 Reddit ID in *any* post URL:
      • short‑link  redd.it/<id>
      • /r/sub/abc123/…               (old style)
      • /comments/<id>/…              (API‑friendly)
    """
    # 1) short‑link host
    u = _u.urlparse(url)
    if u.netloc in {"redd.it", "www.redd.it"}:
        return u.path.lstrip("/").split("/")[0] or None

    # 2) /comments/<id>/ pattern (works already)
    m = re.search(r"/comments/([a-z0-9]{6,8})", url, re.I)
    if m:
        return m.group(1)

    # 3) generic “/r/<sub>/<id>/” or trailing “…/<id>”
    parts = [p for p in u.path.split("/") if p]
    for p in parts[::-1]:                       # search from right‑most
        if _ID_RE.fullmatch(p):
            return p
    return None

# ----------------------------------------------------------------------
# Reddit OAuth helper – app‑only token (read‑only)
# ----------------------------------------------------------------------
import base64
import threading

_TOKEN_LOCK = threading.Lock()
_REDDIT_TOKEN_CACHE: dict[str, float | str] = {"token": None, "expires": 0.0}

def get_reddit_token(client_id: str, client_secret: str) -> str | None:
    """
    Return a cached bearer token obtained via Reddit's client‑credentials flow.
    Returns None on error so callers can fall back to other scraping paths.
    """
    now = time.time()

    # Fast path – cached and still valid
    if (_tok := _REDDIT_TOKEN_CACHE["token"]) and now < _REDDIT_TOKEN_CACHE["expires"] - 30:
        return _tok                       # 30‑sec grace

    with _TOKEN_LOCK:                     # only one thread refreshes
        # Re‑check after acquiring the lock
        if (_tok := _REDDIT_TOKEN_CACHE["token"]) and now < _REDDIT_TOKEN_CACHE["expires"] - 30:
            return _tok

        try:
            auth = requests.auth.HTTPBasicAuth(client_id, client_secret)
            headers = {"User-Agent": _REDDIT_UA}
            data = {"grant_type": "client_credentials"}  # app‑only, read‑only
            r = requests.post(
                "https://www.reddit.com/api/v1/access_token",
                auth=auth,
                data=data,
                headers=headers,
                timeout=10,
            )
            r.raise_for_status()
            payload = r.json()
            token  = payload["access_token"]
            ttl    = int(payload.get("expires_in", 3600))
            _REDDIT_TOKEN_CACHE.update({"token": token, "expires": now + ttl})
            return token
        except Exception as e:
            logging.warning("Reddit token fetch failed: %s", e)
            return None



@retry
def reddit_official_api(url: str, client_id: str, client_secret: str) -> str | None:
    """
    • Works for *any* Reddit permalink or short‑link.
    • If the URL is a subreddit root (/r/<sub>) it still fetches 3 hot posts + top comment (unchanged).
    """
    token = get_reddit_token(client_id, client_secret)
    if not token:
        return None

    headers = {
        "Authorization": f"bearer {token}",
        "User-Agent": _REDDIT_UA,
    }

    # ────────────────────────────────────────────────────────────────────
    # 1.  Try to treat it as a *post* link by extracting an ID
    # ────────────────────────────────────────────────────────────────────
    post_id = _extract_post_id(url)
    if post_id:
        try:
            r = requests.get(
                f"https://oauth.reddit.com/comments/{post_id}",
                headers=headers,
                params={"limit": 5, "depth": 2, "raw_json": 1},
                timeout=10,
            )
            r.raise_for_status()
            data = r.json()

            post   = data[0]["data"]["children"][0]["data"]
            title  = post.get("title", "")
            body   = post.get("selftext", "")
            author = post.get("author", "")

            comments = []
            if len(data) > 1:
                for c in data[1]["data"]["children"][:50]:
                    if c["kind"] == "t1":
                        c_auth = c["data"].get("author", "")
                        c_body = c["data"].get("body", "")
                        if c_body:
                            comments.append(f"u/{c_auth}: {c_body}")

            out = f"Title: {title}\nPosted by: u/{author}\n\n"
            if body:
                out += f"{body}\n\n"
            if comments:
                out += "Top comments:\n\n" + "\n\n".join(comments)
            return out.strip()

        except Exception as e:
            logging.debug("Official API post fetch failed (%s); will try other strategies", e)

    # ────────────────────────────────────────────────────────────────────
    # 2.  If not a post (or the fetch above failed) treat as *subreddit*
    #     root and list 3 hot posts, each with top comment (unchanged).
    # ────────────────────────────────────────────────────────────────────
    m_sub = re.search(r"reddit\.com/r/([^/?#]+)", url)
    if not m_sub:
        return None  # allow caller to fall back

    subreddit = m_sub.group(1)
    try:
        r = requests.get(
            f"https://oauth.reddit.com/r/{subreddit}/hot",
            headers=headers,
            params={"limit": 3, "raw_json": 1},
            timeout=10,
        )
        r.raise_for_status()
        posts = r.json()["data"]["children"]

        out_blocks = []
        for p in posts:
            pd = p["data"]
            pid   = pd["id"]
            title = pd.get("title", "")
            auth  = pd.get("author", "")
            link  = pd.get("permalink", "")

            # fetch one top comment
            top_comment = ""
            try:
                c = requests.get(
                    f"https://oauth.reddit.com/comments/{pid}",
                    headers=headers,
                    params={"limit": 1, "depth": 1, "raw_json": 1},
                    timeout=10,
                ).json()
                if len(c) > 1:
                    for cmt in c[1]["data"]["children"]:
                        if cmt["kind"] == "t1":
                            cauth = cmt["data"].get("author", "")
                            cbody = cmt["data"].get("body", "")
                            top_comment = f"u/{cauth}: {cbody}"
                            break
            except Exception:
                pass

            block = f"Title: {title}\nPosted by: u/{auth}\nLink: https://www.reddit.com{link}\n"
            if top_comment:
                block += f"Top comment:\n{top_comment}"
            out_blocks.append(block)

        return "\n\n---\n\n".join(out_blocks)

    except Exception as e:
        logging.debug("Official API subreddit fetch failed: %s", e)
        return None


@retry
def _reddit_old_version(url: str) -> str | None:
    old_url = url.replace("www.reddit.com", "old.reddit.com")
    try:
        r = _SESS.get(old_url, headers={"User-Agent": _REDDIT_UA}, timeout=(CFG.connect_to, CFG.read_to))
        if r.status_code != 200:
            return None
        
        soup = BeautifulSoup(r.text, "lxml")
        title = soup.select_one(".title").text.strip() if soup.select_one(".title") else ""
        author = soup.select_one(".author").text.strip() if soup.select_one(".author") else ""
        post_body = soup.select_one(".usertext-body") 
        post_text = post_body.get_text(strip=True) if post_body else ""
        
        comments = []
        for comment in soup.select(".comment")[:50]:
            c_author = comment.select_one(".author")
            c_body = comment.select_one(".usertext-body")
            if c_author and c_body:
                comments.append(f"u/{c_author.text}: {c_body.get_text(strip=True)}")
        
        result = f"Title: {title}\nPosted by: u/{author}\n\n"
        if post_text:
            result += f"{post_text}\n\n"
        if comments:
            result += "Top comments:\n\n" + "\n\n".join(comments)
        
        return result.strip()
    except Exception:
        print("old reddit failed")
        return None

@retry
def _pushshift_fallback(url: str) -> str | None:
    m = re.search(r"/comments/([a-z0-9]{6,8})", url)
    if not m:
        return None
    link_id = m.group(1)
    try:
        pst = _SESS.get(
            "https://api.pushshift.io/reddit/submission/search/",
            params={"ids": link_id, "size": 1},
            timeout=10,
        ).json()["data"]
        post_txt = pst[0]["selftext"] if pst else ""
        
        com = _SESS.get(
            "https://api.pushshift.io/reddit/comment/search/",
            params={"link_id": link_id, "sort": "desc", "size": 3},
            timeout=10,
        ).json()["data"]
        top_txt = "\n\n".join(c["body"] for c in com)
        
        txt = (post_txt + "\n\n" + top_txt).strip()
        return txt or None
    except Exception:
        return None

def fetch_reddit(url: str) -> str:
    txt = _reddit_old_version(url)
    if txt:
        return "[Retrieved from Reddit]" + txt[:CFG.text_cap]

    if CFG.reddit_client_id and CFG.reddit_client_secret:
        # print("AAAA")
        txt = reddit_official_api(url, CFG.reddit_client_id, CFG.reddit_client_secret)
        if txt:
            return "[Retrieved from Reddit]" +  txt[:CFG.text_cap]

    txt = _reddit_json_api(url)
    if txt:
        return "[Retrieved from Reddit]" + txt[:CFG.text_cap]
    
    txt = _pushshift_fallback(url)
    if txt:
        return "[Retrieved from Reddit]" + txt[:CFG.text_cap]

    
    return fetch_blocked_site(url)[:CFG.text_cap]
    
