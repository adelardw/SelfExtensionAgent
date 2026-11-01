"""Known-credential egress checks; not a universal personal-data classifier."""
import base64
import os
import re
from urllib.parse import unquote

from .events import record


class SecurityViolation(ValueError):
    pass


def guard_text(text):
    candidates = [str(text)]
    for _ in range(2): candidates.append(unquote(candidates[-1]))
    variants = set()
    for key, value in os.environ.items():
        if len(value) >= 8 and re.search(r'(key|token|secret|password|credential)', key, re.I):
            variants.update((value, base64.b64encode(value.encode()).decode(),
                             base64.urlsafe_b64encode(value.encode()).decode().rstrip('=')))
    if any(v in text for text in candidates for v in variants) or any(
        re.search(r'\bsk-[A-Za-z0-9_-]{16,}|\bBearer\s+[A-Za-z0-9._~-]{12,}|-----BEGIN .*PRIVATE KEY-----', t) for t in candidates):
        record('egress', 'known_secret')
        raise SecurityViolation('Outbound request blocked: known credential content')
