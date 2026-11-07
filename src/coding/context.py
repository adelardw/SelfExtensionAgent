"""Project-local evidence board and bounded, verified experience retrieval.

Shared facts are invalidated by file content changes. Workflows remain historical
examples, never trusted instructions. Nothing here modifies prompts or tool permissions.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
import os
import threading
from contextlib import contextmanager
from pathlib import Path


_memory_lock = threading.RLock()


@contextmanager
def memory_update(path):
    """Serialize small read/merge/write deltas across views and processes."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    lockpath = path.with_suffix('.lock')
    if path.is_symlink() or lockpath.is_symlink() or path.parent.is_symlink():
        raise ValueError('symlinked memory store')
    with _memory_lock, open(lockpath, 'a+b') as lock:
        if os.name == 'nt':
            import msvcrt
            lock.seek(0); lock.write(b'0'); lock.flush(); lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == 'nt':
                lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock, fcntl.LOCK_UN)


def memory_items(path, *, strict=False):
    try:
        if not path.exists(): return []
        if path.is_symlink() or path.stat().st_size > 2*1024**2:
            raise ValueError('unsafe memory index')
        items=json.loads(path.read_text())
        if not isinstance(items,list) or any(not isinstance(x,dict) or not isinstance(x.get('id'),str) for x in items):
            raise ValueError('invalid memory index')
        return items
    except (ValueError,OSError):
        if strict: raise ValueError('memory index unreadable; preserve it for recovery') from None
        return []


def evidence_path(ws, origin, eid):
    if not isinstance(origin,str) or not re.fullmatch(r'[\w.-]{1,100}',origin) or origin in {'.','..'}:
        raise ValueError('invalid evidence origin')
    if not isinstance(eid,str) or not eid.isascii() or not eid.isdigit():
        raise ValueError('invalid evidence id')
    base=ws.directory.parent
    path=base/origin/(eid+'.json')
    if any(p.is_symlink() for p in (base,path.parent,path)) or not path.resolve().is_relative_to(base.resolve()):
        raise ValueError('unsafe memory evidence path')
    if path.stat().st_size>2*1024**2: raise ValueError('memory evidence too large')
    return path


def capture_evidence(ws, evidence):
    ids=[str(e) for e in evidence]
    return {'origin':ws.directory.name, 'digests':{
        eid:hashlib.sha256(evidence_path(ws,ws.directory.name,eid).read_bytes()).hexdigest() for eid in ids}}


def evidence_valid(ws, item):
    """Scope numeric IDs to their original session; hashes are integrity, not truth."""
    ids={str(e) for e in item.get('evidence',[]) if str(e).isdigit()}
    if not ids: return True
    refs=item.get('evidence_refs',{})
    try:
        if set(refs.get('digests',{}))!=ids: return False
        return all(hashlib.sha256(evidence_path(ws,refs['origin'],eid).read_bytes()).hexdigest()==refs['digests'][eid] for eid in ids)
    except (ValueError,OSError,KeyError,TypeError):
        return False


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
        self.items = memory_items(self.path)

    def publish(self, text, evidence, files=(), trigger='', kind='finding', *, workspace=None):
        from .harness import _atomic
        ws = workspace or self.ws
        if ws.directory.resolve() != self.ws.directory.resolve():
            raise ValueError('shared context publication requires the same session')
        if not text.strip() or not evidence:
            raise ValueError('shared context requires text and real evidence ids')
        for eid in evidence:
            if not str(eid).isdigit() or not (ws.directory/f'{eid}.json').is_file():
                raise ValueError('unknown evidence id')
        hashes = {}
        for name in files:
            hashes[name] = ws.file_digest(name)
        author = getattr(ws, 'owner_id', 'main')
        item = {'id': 'ctx-' + digest(text + json.dumps([evidence,author,hashes],sort_keys=True)),
                'author': author, 'text': text[:1400],
                'trigger': trigger[:300], 'evidence': evidence[:8], 'files': hashes,
                'kind': kind, 'verified': False, 'ts': time.time()}
        item['evidence_refs']=capture_evidence(ws,item['evidence'])
        with memory_update(self.path):
            self.items = [x for x in memory_items(self.path,strict=True) if x['id'] != item['id']] + [item]
            self.items=self.items[-30:]
            _atomic(self.path, json.dumps(self.items, ensure_ascii=False, indent=2))
        return item

    def valid(self, item, workspace=None):
        ws=workspace or self.ws
        try:
            return evidence_valid(ws,item) and all(ws.file_digest(p) == h for p,h in item['files'].items())
        except (ValueError, OSError):
            return False

    def retrieve(self, query, limit=2):
        self.items=memory_items(self.path)
        return select([x for x in self.items if self.valid(x)], query, limit)

    def inventory(self):
        self.items=memory_items(self.path)
        return [{'id': x['id'], 'valid': self.valid(x), 'kind': x['kind'], 'evidence': x['evidence']} for x in self.items]


class ExperienceStore:
    def __init__(self, root):
        self.root = root
        self.path = root / '.sea' / 'coding' / 'experience.json'
        self.stack = fingerprint(root)
        self.items = memory_items(self.path)

    def valid(self, item):
        from types import SimpleNamespace
        ws=SimpleNamespace(directory=self.root/'.sea'/'coding'/'current')
        return (item.get('verified') is True and bool(item.get('evidence'))
                and all(str(e).isascii() and str(e).isdigit() for e in item['evidence'])
                and item.get('stack')==fingerprint(self.root) and evidence_valid(ws,item))

    def retrieve(self, query):
        self.stack = fingerprint(self.root)
        self.items=memory_items(self.path)
        return select([x for x in self.items if self.valid(x)], query)

    def promote(self, trigger, lesson, actions, checks, evidence, *, workspace=None):
        from .harness import _atomic
        if not checks or not evidence:
            return None
        if workspace is None or workspace.root!=self.root or workspace.verified_revision!=workspace.revision or not set(checks)<=workspace.passed:
            raise ValueError('experience promotion requires current runtime verification in this project')
        self.stack = fingerprint(self.root)
        entry = {'id': 'exp-' + digest(trigger.casefold().strip() + self.stack), 'trigger': trigger[:400],
                 'text': lesson[:900], 'example': actions[-6:], 'checks': checks,
                 'evidence': evidence[-8:], 'verified': True, 'stack': self.stack, 'ts': time.time()}
        entry['evidence_refs']=capture_evidence(workspace,entry['evidence'])
        entry['scope']='Historical successful episode; the lesson is a hypothesis, not an instruction or proof.'
        with memory_update(self.path):
            self.items = [x for x in memory_items(self.path,strict=True) if x['id'] != entry['id']] + [entry]
            self.items=self.items[-40:]
            _atomic(self.path, json.dumps(self.items, ensure_ascii=False, indent=2))
        return entry
