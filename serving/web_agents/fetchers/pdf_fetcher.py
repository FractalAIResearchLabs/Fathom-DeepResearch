from __future__ import annotations
from config import CFG, _SESS
import io, logging, re, pymupdf as fitz

from web_helpers import retry, fetch_blocked_site     # ⬅️ shared
# ----------------------------------------------------------------------

class PDFExtractError(RuntimeError): ...

@retry
def _download_pdf(url: str) -> bytes:
    with _SESS.get(url, stream=True, timeout=(CFG.connect_to, CFG.read_to)) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0) or 0)
        if 0 < total > CFG.pdf_size_cap:
            raise RuntimeError("pdf too large")
        buf = io.BytesIO()
        for chunk in r.iter_content(16_384):
            buf.write(chunk)
            if buf.tell() > CFG.pdf_size_cap:
                raise RuntimeError("pdf exceeds cap")
        return buf.getvalue()

def _extract_pdf(buf: bytes) -> str:
    try:
        doc = fitz.open(stream=buf, filetype="pdf")
    except Exception as e:
        raise PDFExtractError(e)
    parts, chars = [], 0
    for page in doc:
        if len(parts) >= CFG.pdf_pages_cap:
            break
        text = (
            page.get_text("text")
            .replace("\u00A0", " ")
            .replace("-\n", "")
        )
        parts.append(text)
        chars += len(text)
        if chars > CFG.pdf_chars_cap:
            break
    out = " ".join(parts).strip()[:CFG.pdf_chars_cap]
    if not out:
        raise PDFExtractError("empty / scanned pdf")
    return "[Retrieved from PyMUPDF]" + out

def fetch_pdf(url: str) -> str:
    try:
        buf = _download_pdf(url)
        return _extract_pdf(buf)
    except Exception as e:
        logging.error("PDF fetch failed for %s: %s", url, e)
        return fetch_blocked_site(url)[:CFG.text_cap]
