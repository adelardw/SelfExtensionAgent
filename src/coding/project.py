"""Small typed project notes and SWE guidance, shaped by explicit interaction evidence."""
import hashlib
import json
from pathlib import Path

from .context import select, digest

KINDS = {'fact','hypothesis','decision','procedure','pitfall','question','preference','reference'}


def stack(root):
    found = []
    for marker, name in [('pyproject.toml','Python'),('CMakeLists.txt','C++/CMake'),('Makefile','Make'),
                         ('package.json','Node'),('Dockerfile','Docker'),('compose.yml','Docker Compose')]:
        if (root/marker).exists(): found.append(name)
    guidance = {
        'Python':'Inspect pyproject/lock and active interpreter. Reproduce with a targeted test. Preserve public APIs; avoid global pip installs.',
        'C++/CMake':'Inspect compiler and CMake targets; build out of tree in scratch, then compile and run ctest. Check UB, ownership and ABI assumptions.',
        'Make':'Read targets before invoking them; never assume a clean target is safe.',
        'Node':'Inspect package scripts and lockfile. Use existing local dependencies; typecheck/build and focused tests.',
        'Docker':'Inspect Dockerfile/Compose before execution. Docker socket is host-level authority; no privileged containers or host-root mounts without explicit approval.',
        'Docker Compose':'Inspect rendered service configuration and volumes before starting it; preserve existing volumes.',
    }
    return {'detected':found, 'guidance':[guidance[x] for x in found],
            'terminal':'Use diff/grep/read/status to inspect; commands run with a restricted filesystem and no network by default. Never use sudo or broad cleanup. Check working directory and tool versions before acting.'}


class ProjectContext:
    def __init__(self, workspace, control):
        self.ws, self.control = workspace, control
        self.directory = workspace.root/'.sea'/'context'
        self.path = self.directory/'index.json'
        try: self.notes = json.loads(self.path.read_text()) if self.path.exists() else []
        except (ValueError,OSError): self.notes = []

    def add(self, kind, title, text, evidence=(), files=()):
        from .harness import _atomic
        if kind not in KINDS: raise ValueError('unknown project note kind')
        if not title.strip() or not text.strip(): raise ValueError('note requires title and text')
        evidence = [str(e) for e in evidence]
        human = {e['id'] for e in self.control.history if e.get('origin') == 'user'} | {'task'}
        if kind == 'preference' and not any(e in human for e in evidence):
            raise ValueError('preferences require explicit human interaction evidence, not a personality inference')
        for eid in evidence:
            if eid not in human and (not eid.isdigit() or not (self.ws.directory/f'{eid}.json').is_file()):
                raise ValueError('unknown note evidence')
        hashes = {name:hashlib.sha256(self.ws.path(name).read_bytes()).hexdigest() for name in files}
        status = 'hypothesis' if kind in {'hypothesis','preference'} or not evidence else 'observed'
        if kind == 'question': status = 'open'
        note = {'id':'note-'+digest(kind+title), 'kind':kind, 'trigger':title[:160], 'text':text[:1800],
                'status':status, 'evidence':evidence[:8], 'files':hashes}
        self.notes = [n for n in self.notes if n['id'] != note['id']] + [note]
        self.notes = self.notes[-60:]
        _atomic(self.path,json.dumps(self.notes,ensure_ascii=False,indent=2))
        _atomic(self.directory/(note['id']+'.md'), '# '+title+'\n\nType: '+kind+' · Status: '+status+'\n\n'+text+'\n\nEvidence: '+', '.join(evidence)+'\n')
        _atomic(self.directory/'INDEX.md', '# Project context\n\n'+'\n'.join(f"- [{n['trigger']}]({n['id']}.md) — {n['kind']} / {n['status']}" for n in self.notes))
        return note

    def valid(self,note):
        try: return all(hashlib.sha256(self.ws.path(p).read_bytes()).hexdigest()==h for p,h in note['files'].items())
        except (ValueError,OSError): return False

    def retrieve(self, query):
        notes=select([n for n in self.notes if self.valid(n)],query,3)
        return {'stack':stack(self.ws.root),'notes':notes,
                'index':[{'id':n['id'],'kind':n['kind'],'title':n['trigger'],'status':n['status'],'current':self.valid(n)} for n in self.notes][-15:]}
