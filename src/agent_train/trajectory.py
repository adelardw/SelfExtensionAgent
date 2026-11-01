"""Local, bounded training telemetry. An episode outcome is never a reward by itself."""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def redact(value):
    """Best-effort credential filtering; redacted samples are excluded from training."""
    count = 0
    secrets = [v for k, v in os.environ.items() if len(v) >= 8 and
               re.search(r'(API_KEY|ACCESS_TOKEN|SECRET|PASSWORD)$', k, re.I)]
    def visit(item, key=''):
        nonlocal count
        if re.fullmatch(r'(api[_-]?key|authorization|password|secret|access_token)', key, re.I):
            count += 1
            return '[REDACTED]'
        if isinstance(item, dict): return {str(k): visit(v, str(k)) for k, v in item.items()}
        if isinstance(item, (list, tuple)): return [visit(v) for v in item]
        if isinstance(item, str):
            for secret in secrets:
                if secret in item:
                    count += 1
                    item = item.replace(secret, '[REDACTED]')
            item, n = re.subn(r'\bsk-[A-Za-z0-9_-]{16,}|\bBearer\s+[A-Za-z0-9._~-]{12,}', '[REDACTED]', item)
            count += n
        return item
    return visit(value), count


class TrajectoryLog:
    """One invocation per file, including resume; request IDs isolate parallel actors.

    Large payloads are content addressed. Logging failure never authorizes an action
    or changes a model result. Incomplete/redacted logs cannot become training data.
    """
    def __init__(self, directory, metadata, max_bytes=32 * 1024 * 1024):
        self.run_id = uuid.uuid4().hex
        self.directory = Path(directory) / 'trajectories' / self.run_id
        self.path = self.directory / 'events.jsonl'
        self.max_bytes = max_bytes
        self.bytes = 0
        self.sequence = 0
        self.complete = True
        self.redactions = 0
        self.error = ''
        self.calls = {}
        self._blobs = set()
        try:
            self.directory.mkdir(parents=True, mode=0o700)
            (self.directory / 'blobs').mkdir(mode=0o700)
        except OSError as exc:
            self.complete = False
            self.error = type(exc).__name__
        self.record('run_start', metadata)

    def _write(self, path, value, append=False):
        flags = os.O_WRONLY | os.O_CREAT | (os.O_APPEND if append else os.O_EXCL)
        flags |= getattr(os, 'O_NOFOLLOW', 0)
        with os.fdopen(os.open(path, flags, 0o600), 'wb') as f: f.write(value)

    def record(self, kind, data, agent='main', call_id=None):
        if not self.complete: return
        try:
            clean, count = redact(data)
            raw = json.dumps(clean, ensure_ascii=False, sort_keys=True).encode()
            sha = hashlib.sha256(raw).hexdigest()
            event = {'v': 1, 'run_id': self.run_id, 'seq': self.sequence, 'time': time.time(),
                     'kind': kind, 'agent': agent, 'call_id': call_id,
                     'blob': sha, 'redactions': count}
            line = (json.dumps(event) + '\n').encode()
            added = len(line) + (0 if sha in self._blobs else len(raw))
            if self.bytes + added > self.max_bytes:
                self.complete = False
                self.error = 'trajectory byte limit exceeded'
                return
            if sha not in self._blobs:
                self._write(self.directory / 'blobs' / (sha + '.json'), raw)
                self._blobs.add(sha)
            self._write(self.path, line, append=True)
            self.sequence += 1
            self.bytes += added
            self.redactions += count
        except (OSError, TypeError, ValueError) as exc:
            self.complete = False
            self.error = type(exc).__name__

    def model_event(self, agent, kind, data):
        if kind == 'request':
            self.calls[agent] = uuid.uuid4().hex
        if kind in {'request', 'response', 'validated', 'end', 'provider-retry', 'output-budget'}:
            self.record(kind, data, agent, self.calls.get(agent))

    def action(self, agent, name, ok, evidence, directory):
        try:
            observation = json.loads((Path(directory) / (str(evidence) + '.json')).read_text())
        except (OSError, ValueError):
            observation = {'unavailable': True}
        self.record('action', {'tool': name, 'ok': ok, 'evidence_id': str(evidence),
                              'observation': observation}, agent, self.calls.get(agent))

    def outcome(self, result):
        self.record('outcome', result)

    def evaluation(self, *, task_id, family, split, verifier_id, answer_ok, process_ok,
                   status, observation, eligible=False, source_unchanged=True, infrastructure_error=False):
        """Called by the external evaluator, never exposed as an agent tool.

        A positive finite-check reward is not a universal correctness certificate.
        No reward for infrastructure errors, changed harness code or missing checks.
        """
        known = type(answer_ok) is bool and type(process_ok) is bool
        scorable = known and status in {'done', 'paused'} and source_unchanged and not infrastructure_error
        reward = float(answer_ok and process_ok and status == 'done') if scorable else None
        self.record('evaluation', {'task_id': task_id, 'family': family, 'split': split,
            'verifier_id': verifier_id, 'answer_ok': answer_ok, 'process_ok': process_ok,
            'status': status, 'reward': reward, 'observation': observation,
            'training_eligible': bool(eligible and scorable and self.complete and not self.redactions),
            'source_unchanged': source_unchanged,
            'infrastructure_error': infrastructure_error,
            'scope': 'External executable check; coverage limited to this verifier.'})

    def receipt(self):
        return {'run_id': self.run_id, 'path': str(self.path), 'bytes': self.bytes,
                'complete': self.complete, 'redactions': self.redactions, 'error': self.error}


def read_events(path):
    """Validate stored blobs; never follow a path supplied by a trace event."""
    path = Path(path)
    events = []
    for sequence, line in enumerate(path.read_text().splitlines()):
        event = json.loads(line)
        if event.get('v') != 1 or event.get('seq') != sequence:
            raise ValueError('Invalid trajectory sequence/version')
        sha = event['blob']
        if not re.fullmatch('[a-f0-9]{64}', sha): raise ValueError('Invalid blob id')
        blob = path.parent / 'blobs' / (sha + '.json')
        if blob.is_symlink(): raise ValueError('Symlinked trajectory blob')
        raw = blob.read_bytes()
        if hashlib.sha256(raw).hexdigest() != sha: raise ValueError('Corrupt trajectory blob')
        events.append({**event, 'data': json.loads(raw)})
    if len({e['run_id'] for e in events}) != 1: raise ValueError('Mixed trajectory runs')
    return events


def harness_manifest():
    root = Path(__file__).resolve().parents[2]
    paths = sorted((root / 'src' / 'coding').glob('*.py'))
    paths += sorted((root / 'src' / 'agent_train').glob('*.py'))
    paths += [root / 'pyproject.toml', root / 'uv.lock']
    manifest = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in paths if p.is_file()}
    return {'source_id': digest(manifest), 'files': manifest}
