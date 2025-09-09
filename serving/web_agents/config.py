from __future__ import annotations
import logging, os, random, requests

class _Cfg:
    ua: str = (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
    )
    serper_key            = os.getenv("SERPER_API_KEY", "")
    jina_cache_dir             =  os.getenv("JINA_CACHE_DIR", "")
    serper_cache_dir             =  os.getenv("SERPER_CACHE_DIR", "")
    jina_key              = os.getenv("JINA_API_KEY", "")
    query_llm             = os.getenv("QUERY_LLM", "gpt-4.1-mini")
    serper_ep             = "https://google.serper.dev/search"
    retries               = 3
    backoff               = 0.8
    connect_to            = 5
    read_to               = 10
    stream_html_cap       = 200_000
    pdf_size_cap          = 32_000_000
    pdf_pages_cap         = 40
    pdf_chars_cap         = 40_000
    text_cap              = 400_000
    output_limit_per_link = 6_000
    disable_narrative_compress_thresh = 2_000
    pct                  = 0.25          # narrative compression pct
    reddit_client_id = "Q2tovcGfYmo3hPNvzTpkXA"
    reddit_client_secret = "geu4gH3pEOrNnsMpQvdTTVhQvDABgg"


CFG   = _Cfg()
_RND  = random.Random()
_SESS = requests.Session()
_SESS.headers.update({"User-Agent": CFG.ua})

# logging.basicConfig(level=logging.INFO,
#                     format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger().setLevel(logging.INFO)   # bump root to DEBUG
