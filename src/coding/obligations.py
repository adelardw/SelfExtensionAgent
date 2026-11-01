"""Persistent requirement/claim/evidence DAG. Support is scoped evidence, never truth.

Only the critic can record a review. Agent claims remain proposals, and changed
source, steering, evidence or dependencies invalidate their previous support.
"""
from __future__ import annotations

import hashlib
import json
import re
from pydantic import BaseModel, ConfigDict, Field


class ClaimSpec(BaseModel):
    model_config = ConfigDict(extra='forbid')
    id: str = Field(pattern=r'^c[\w-]{1,63}$')
    text: str = Field(min_length=1, max_length=1200)
    requirements: list[str] = Field(default_factory=list, max_length=24)
    evidence: list[str] = Field(default_factory=list, max_length=8)
    files: list[str] = Field(default_factory=list, max_length=16)
    depends_on: list[str] = Field(default_factory=list, max_length=8)
    assumptions: list[str] = Field(default_factory=list, max_length=8)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class ObligationGraph:
    def __init__(self, ws, owner='main'):
        if not re.fullmatch(r'[\w-]{1,80}', owner):
            raise ValueError('Invalid graph owner')
        self.ws = ws
        self.path = ws.directory / ('obligations-' + owner + '.json')
        try:
            self.data = json.loads(self.path.read_text())
        except (OSError, ValueError):
            self.data = {'version': '', 'requirements': [], 'claims': {}}

    def save(self):
        from .harness import _atomic
        _atomic(self.path, json.dumps(self.data, ensure_ascii=False, indent=2))

    def sync(self, contract, steering):
        version = digest([contract.model_dump(), steering])
        if version != self.data['version']:
            self.data['version'] = version
            self.data['steering'] = digest(steering)
            self.data['requirements'] = [r.model_dump() for r in contract.requirements]
            self.data.pop('next_steps', None)
            self.data.pop('followthrough_scope', None)
            self.save()

    def evidence(self, ids):
        result = {}
        for eid in ids:
            eid = str(eid)
            if not eid.isdigit():
                raise ValueError('Evidence must reference a real numeric record')
            value = (self.ws.directory / (eid + '.json')).read_bytes()
            result[eid] = hashlib.sha256(value).hexdigest()
        return result

    def propose(self, *, id, text, requirements=(), evidence=(), files=(), depends_on=(), assumptions=()):
        if not re.fullmatch(r'c[\w-]{1,63}', id):
            raise ValueError('Use a claim id starting with c, e.g. c-boundary')
        claims = self.data['claims']
        if not id.startswith('c-review-') and id not in claims and sum(not k.startswith('c-review-') for k in claims) >= 40:
            raise ValueError('Agent claim budget is 40; remaining slots are reserved for reviews')
        if len(claims) >= 64 and id not in claims:
            raise ValueError('Claim budget is 64; update an existing claim')
        if not text.strip() or len(text) > 1200:
            raise ValueError('Claim text must contain 1..1200 characters')
        if len(depends_on) > 8 or len(evidence) > 8 or len(files) > 16 or len(assumptions) > 8:
            raise ValueError('Claim exceeds dependency/evidence/file/assumption budget')
        known = {r['id'] for r in self.data['requirements']}
        if set(requirements) - known:
            raise ValueError('Unknown requirement id; inspect context first')
        if set(depends_on) - claims.keys() or id in depends_on:
            raise ValueError('Unknown or cyclic claim dependency')
        def reaches(current, seen):
            if current == id: return True
            if current in seen: return False
            return any(reaches(d, seen | {current}) for d in claims[current]['depends_on'])
        if any(reaches(d, set()) for d in depends_on):
            raise ValueError('Cyclic claim dependency')
        claim = {'id': id, 'text': text, 'requirements': list(requirements),
                 'assumptions': [str(a)[:300] for a in assumptions], 'depends_on': list(depends_on),
                 'dependency_hashes': {d: digest(claims[d]) for d in depends_on},
                 'evidence': self.evidence(evidence),
                 'files': {p: self.ws.file_digest(p) for p in files},
                 'contract_version': self.data['version'], 'status': 'proposed',
                 'support_kind': 'agent_claim'}
        claims[id] = claim
        self.save()
        return claim

    def status(self, claim, visiting=None, cache=None):
        cache = {} if cache is None else cache
        visiting = (visiting or set()) | {claim['id']}
        try:
            control = getattr(getattr(self.ws, 'orchestrator', None), 'control', None)
            if control and self.data.get('steering') != digest(control.instructions(getattr(self.ws, 'owner_id', 'main'))): return 'stale'
            if claim['contract_version'] != self.data['version']: return 'stale'
            if self.evidence(claim['evidence']) != claim['evidence']: return 'stale'
            if claim.get('source_manifest'):
                if 'manifest' not in cache: cache['manifest']=digest(sorted(self.ws.files()))
                if cache['manifest'] != claim['source_manifest']: return 'stale'
            for p,h in claim['files'].items():
                if ('file',p) not in cache: cache['file',p]=self.ws.file_digest(p)
                if cache['file',p] != h: return 'stale'
            for dep in claim['depends_on']:
                other = self.data['claims'].get(dep)
                if not other or dep in visiting or digest(other) != claim['dependency_hashes'][dep]: return 'stale'
                state = self.status(other, visiting, cache)
                if state in {'stale', 'missing', 'uncertain', 'proposed'}: return state
            return claim['status']
        except (OSError, ValueError, KeyError):
            return 'stale'

    def review(self, assessment, sources):
        """Runtime-only: link each criterion to a review node with current provenance."""
        self.data['ready'] = assessment.ready
        self.data['issues'] = assessment.issues
        self.data['next_steps'] = [s.model_dump() for s in assessment.next_steps]
        control = getattr(getattr(self.ws, 'orchestrator', None), 'control', None)
        self.data['followthrough_scope'] = {'contract_version':self.data['version'],
            'revision':self.ws.revision,
            'steering_version':control.version(getattr(self.ws,'owner_id','main')) if control else 0}
        source_manifest=digest(sorted(self.ws.files())) if 'probe' in self.ws.verifier_tools else None
        source_hashes={p:self.ws.file_digest(p) for p in sources}
        for criterion in assessment.criteria:
            rid = criterion.requirement.split(':', 1)[0].strip()
            if rid not in {r['id'] for r in self.data['requirements']}: continue
            claim = self.propose(id='c-review-' + rid, text=criterion.requirement,
                                 requirements=[rid], evidence=criterion.evidence)
            claim['files'] = source_hashes
            claim['source_manifest'] = source_manifest
            claim['status'] = criterion.status
            claim['support_kind'] = 'model_review'  # Does not assert a formal proof.
        self.save()

    def packet(self):
        claims = self.data['claims']
        cache = {}
        status = {cid: self.status(c, cache=cache) for cid, c in claims.items()}
        requirements = []
        for r in self.data['requirements']:
            cid = 'c-review-' + r['id']
            requirements.append({**r, 'status': status.get(cid, 'unreviewed'), 'review': cid if cid in claims else None})
        return {'version': self.data['version'], 'requirements': requirements,
                'claims': [{**{k: c[k] for k in ('id', 'requirements', 'assumptions', 'depends_on', 'support_kind')},
                            'text': c['text'][:300],
                            'evidence': list(c['evidence']), 'status': status[cid]} for cid, c in claims.items()],
                'ready': bool(self.data.get('ready')) and all(r['status']=='supported' for r in requirements),
                'issues': self.data.get('issues', []),
                'verification': {'observations': [{**p, 'output': p['output'][-600:]} for p in
                    self.data.get('verification', {}).get('observations', [])],
                    **{key:self.data.get('verification',{}).get(key,[] if key!='dispute' else {})
                       for key in ('repairs','resolutions','dispute')},
                    'limitation': self.data.get('verification', {}).get('limitation', '')}, 'formal_proof': False}
