"""Bounded local event metadata; never record prompts, URLs, headers or secrets."""
import json
import os
from pathlib import Path
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
_LOCK = threading.Lock()


def record(surface, reason):
    # Only developer-owned identifiers are accepted, no attacker-controlled text.
    if not all(isinstance(v, str) and v.replace('_', '').isalnum() and len(v) <= 48 for v in (surface, reason)):
        return
    try:
        with _LOCK:
            directory = ROOT/'.sea/security'
            if directory.is_symlink() or (ROOT/'.sea').is_symlink(): return
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            path = directory/'events.jsonl'
            if path.is_symlink(): return
            if path.exists() and path.stat().st_size > 1024*1024:
                path.replace(directory/'events.previous.jsonl')
            fd = os.open(path, os.O_WRONLY|os.O_CREAT|os.O_APPEND|getattr(os, 'O_NOFOLLOW', 0), 0o600)
            with os.fdopen(fd, 'w') as f:
                f.write(json.dumps({'time': time.time(), 'surface': surface, 'reason': reason})+'\n')
    except OSError:
        pass # Logging failure never changes the deny decision.
