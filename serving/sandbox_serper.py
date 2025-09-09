#!/usr/bin/env python3
"""sandbox_serper.py – resilient Serper sandbox v2.1

Fixes
-----
* Moved `global _MAX_OUTBOUND, _SEM` declaration to the **top of `main()`**
  before any reference, eliminating the `SyntaxError: name used prior to
  global declaration`.
* No functional changes otherwise.
"""

from __future__ import annotations
import argparse, asyncio, logging, os, time, traceback
from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
import uvicorn
import time 
# ───────────────────────── logging setup ──────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("sandbox_serper")

app = FastAPI()

class Req(BaseModel):
    env: str
    call: str
    timeout: int = 60

# ───────────────────── global throughput gate ─────────────────────
_MAX_OUTBOUND = int(os.getenv("MAX_OUTBOUND", "10"))
_SEM = asyncio.Semaphore(_MAX_OUTBOUND)

# ───────────────────────── endpoint ───────────────────────────────
@app.post("/execute")
async def execute(req: Req):
    # async with _SEM:
    async with _SEM:          #  ❰❰  throttle
        result = await run_in_threadpool(_safe_eval, req.env,
                                          req.call, req.timeout)

    return {
        "output": "",
        "result": result,
        "error": None if not str(result).startswith("[tool-error]") else result,
    }

# ───────────────────── sandbox evaluator ──────────────────────────

def _safe_eval(env: str, call: str, timeout: int):
    start = time.time(); loc: dict = {}
    try:
        exec(env, {}, loc)
        exec(f"response = {call}", {}, loc)
        if time.time() - start > timeout:
            raise TimeoutError(f"wall-clock timeout for call {call}")
        return loc.get("response", "[tool-error] no response var")
    except Exception as e:
        log.error("tool error: %s\n%s", e, traceback.format_exc())
        return f"[tool-error] {e}"

# ─────────────────────────── main ────────────────────────────────

def main():
    global _MAX_OUTBOUND, _SEM  # ← moved to top

    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=1211)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--reload", action="store_true")
    # ap.add_argument("--max_outbound", type=int, default=_MAX_OUTBOUND,
                    # help="simultaneous outbound calls across all workers")
    args = ap.parse_args()

    _SEM = asyncio.Semaphore(_MAX_OUTBOUND)

    if args.reload and args.workers > 1:
        raise SystemExit("--reload and --workers>1 are mutually exclusive")

    # log.info("Starting sandbox :%d | workers=%d | max_outbound=%d",
            #  args.port, args.workers, _MAX_OUTBOUND)

    if args.workers > 1:
        uvicorn.run("sandbox_serper:app", host="0.0.0.0", port=args.port, workers=args.workers)
    else:
        uvicorn.run(app, host="0.0.0.0", port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()