"""Project-local evidence board and bounded, verified experience retrieval.

Shared facts are invalidated by file content changes. Workflows remain historical
examples, never trusted instructions. Nothing here modifies prompts or tool permissions.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def fingerprint(root):
    names = ('pyproject.toml', 'package.json', 'Cargo.toml', 'go.mod', 'uv.lock')
    return digest(''.join(n + (root/n).read_text(errors='replace') for n in names if (root/n).is_file()))


def tokens(text):
    return set(re.findall(r'\w{3,}', text.casefold()))


def select(items, query, limit=2):
    scored = [(len(tokens(query) & tokens(x.get('trigger', '') + ' ' + x.get('text', ''))), x) for x in items]
    return [x for score, x in sorted(scored, key=lambda p: p[0], reverse=True)[:limit] if score >= 1]


class ContextBoard:
    def __init__(self, workspace):
        self.ws = workspace
        self.path = workspace.directory / 'shared.json'
        try:
            self.items = json.loads(self.path.read_text()) if self.path.exists() else []
        except (ValueError, OSError):
            self.items = []

    def publish(self, text, evidence, files=(), trigger='', kind='finding'):
        from .harness import _atomic
        if not text.strip() or not evidence:
            raise ValueError('shared context requires text and real evidence ids')
        for eid in evidence:
            if not str(eid).isdigit() or not (self.ws.directory/f'{eid}.json').is_file():
                raise ValueError('unknown evidence id')
        hashes = {}
        for name in files:
            hashes[name] = self.ws.file_digest(name)
        item = {'id': 'ctx-' + digest(text + json.dumps(evidence)), 'text': text[:1400],
                'trigger': trigger[:300], 'evidence': evidence[:8], 'files': hashes,
                'kind': kind, 'verified': False, 'ts': time.time()}
        self.items = [x for x in self.items if x['id'] != item['id']] + [item]
        _atomic(self.path, json.dumps(self.items[-30:], ensure_ascii=False, indent=2))
        return item

    def valid(self, item):
        try:
            return all(self.ws.file_digest(p) == h for p,h in item['files'].items())
        except (ValueError, OSError):
            return False

    def retrieve(self, query, limit=2):
        return select([x for x in self.items if self.valid(x)], query, limit)

    def inventory(self):
        return [{'id': x['id'], 'valid': self.valid(x), 'kind': x['kind'], 'evidence': x['evidence']} for x in self.items]


class ExperienceStore:
    def __init__(self, root):
        self.root = root
        self.path = root / '.sea' / 'coding' / 'experience.json'
        self.stack = fingerprint(root)
        try:
            self.items = json.loads(self.path.read_text()) if self.path.exists() else []
        except (ValueError, OSError):
            self.items = []

    def retrieve(self, query):
        self.stack = fingerprint(self.root)
        return select([x for x in self.items if x.get('stack') == self.stack and x.get('verified')], query)

    def promote(self, trigger, lesson, actions, checks, evidence):
        from .harness import _atomic
        if not checks or not evidence:
            return None
        self.stack = fingerprint(self.root)
        entry = {'id': 'exp-' + digest(trigger.casefold().strip() + self.stack), 'trigger': trigger[:400],
                 'text': lesson[:900], 'example': actions[-6:], 'checks': checks,
                 'evidence': evidence[-8:], 'verified': True, 'stack': self.stack, 'ts': time.time()}
        self.items = [x for x in self.items if x['id'] != entry['id']] + [entry]
        _atomic(self.path, json.dumps(self.items[-40:], ensure_ascii=False, indent=2))
        return entry
