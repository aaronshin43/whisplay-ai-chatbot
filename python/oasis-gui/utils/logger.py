import json
import os
import re
from datetime import datetime

# Resolve relative to project root (two levels up from this file)
_PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "../../../")
LOG_DIR = os.path.normpath(os.path.join(_PROJECT_ROOT, "data/oasis_logs"))


def log_response(query: str, response: str):
    """Write a JSONL entry to data/oasis_logs/.

    Format matches OasisAdapter.ts logOasisResponse() for unified log analysis.
    Logging failures are silently swallowed — must never affect the response path.
    """
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        entry = json.dumps({
            "ts": datetime.now().isoformat(),
            "query": query,
            "response": response,
            "steps": len(re.findall(r'^\d+\.', response, re.MULTILINE)),
            "has_markdown": bool(re.search(r'[*_#`]', response)),
        })
        log_file = os.path.join(LOG_DIR, f"oasis_{datetime.now():%Y-%m-%d}.jsonl")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(entry + "\n")
    except Exception:
        pass
