"""Scratch helper: strip ANSI/control characters from a log and keep its tail."""

import glob
import os
import re
import sys
import traceback

src = sys.argv[1]
dst = sys.argv[2]
tail = int(sys.argv[3]) if len(sys.argv) > 3 else 40

try:
    with open(src, encoding="utf-8", errors="replace") as fh:
        raw = fh.read()
    text = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", raw)
    text = "".join(ch if ch == "\n" or ch >= " " else ("\n" if ch == "\r" else "") for ch in text)
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    payload = "\n".join(lines[-tail:])
except Exception:  # noqa: BLE001
    payload = "ERROR\n" + traceback.format_exc() + "\nCWD=" + os.getcwd() + "\nGLOB=" + repr(glob.glob(os.path.join(os.environ.get("TEMP", ""), "*.log")))

with open(dst, "w", encoding="utf-8") as fh:
    fh.write(payload)
