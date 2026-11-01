"""Isolated branch snapshots and checked, conflict-aware patch adoption.

Works with dirty repositories and ordinary directories; never commits, resets or
mutates the user's Git index. A branch can write only inside its own snapshot.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path


def sha(data):
    return hashlib.sha256(data).hexdigest()


@dataclass
class Snapshot:
    workspace: object
    parent: object
    baseline: dict
    owner: str
    parent_id: str


class BranchStore:
    def __init__(self, workspace, control, config):
        self.root, self.control, self.config = workspace, control, config
        self.snapshots, self.proposals = {}, {}

    def files(self, ws):
        result, size = {}, 0
        for name in ws.files():
            path = ws.path(name)
            if path.is_symlink():
                continue
            length = path.stat().st_size
            if length > int(self.config.get('snapshot_file_bytes', 2_000_000)):
                raise ValueError(f'Snapshot file too large: {name}; narrow the workspace')
            size += length
            if size > int(self.config.get('snapshot_bytes', 64_000_000)) or len(result) >= 14999:
                raise ValueError('Snapshot budget exceeded; narrow the workspace')
            result[name] = path.read_bytes()
        return result

    def fork(self, parent, owner, parent_id):
        from .harness import Workspace
        directory = self.root.directory/'branches'/owner
        root = directory/'workspace'
        if root.exists():
            raise ValueError('Branch workspace already exists')
        baseline = self.files(parent)
        root.mkdir(parents=True, mode=0o700)
        for name, data in baseline.items():
            target = root/name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            target.chmod(parent.path(name).stat().st_mode & 0o777)
        checks = {}
        for name, argv in parent.checks.items():
            translated = []
            for i, arg in enumerate(argv):
                candidate = Path(arg)
                if i == 0 and (parent.root/candidate).is_file():
                    # Reuse installed interpreters, without exposing their source project for writing.
                    # Keep the lexical executable path: resolving a venv's Python
                    # symlink loses pyvenv.cfg and its installed dependencies.
                    translated.append(os.path.abspath(parent.root/candidate))
                elif candidate.is_absolute() and candidate.is_relative_to(parent.root):
                    translated.append(str(root/candidate.relative_to(parent.root)))
                else:
                    translated.append(arg)
            checks[name] = translated
        child = Workspace(root, self.root.directory, checks)
        child.isolated = True
        child.allow_web = parent.allow_web
        child.record = self.root.record
        child.board = parent.board  # Shared findings carry source/evidence provenance.
        child.proposal_store, child.owner_id = self, owner
        if owner.startswith('mind-') and getattr(parent,'test_provenance',None):
            child.test_provenance=parent.test_provenance.fork(child,owner)
        snapshot = Snapshot(child, parent, {n:sha(d) for n,d in baseline.items()}, owner, parent_id)
        self.snapshots[owner] = snapshot
        from .harness import _atomic
        _atomic(directory/'snapshot.json',json.dumps({'owner':owner,'parent':parent_id,
            'baseline':snapshot.baseline,'checks':checks},ensure_ascii=False))
        return child

    def restore(self):
        """Restore reviewable work/provenance; old check results never become current verification."""
        from .harness import Workspace
        pending=[]
        base=self.root.directory/'branches'
        if not base.exists(): return {}
        for directory in base.iterdir():
            if not directory.is_dir() or not directory.name.startswith('mind-'): continue
            proposal_file=directory/'proposal.json'
            manifest_file=directory/'snapshot.json'
            try:
                proposal=json.loads(proposal_file.read_text()) if proposal_file.exists() else None
                manifest=json.loads(manifest_file.read_text()) if manifest_file.exists() else {}
                parent=manifest.get('parent') or (proposal or {}).get('parent')
                if parent: pending.append((directory,manifest,proposal,parent))
            except (ValueError,OSError): continue
        restored={}
        for _ in range(len(pending)+1):
            progress=False
            for item in list(pending):
                directory,manifest,proposal,parent_id=item
                if parent_id!='main' and parent_id not in restored: continue
                parent=self.root if parent_id=='main' else restored[parent_id]
                root=directory/'workspace'
                if not root.is_dir() or not root.resolve().is_relative_to(self.root.directory):
                    pending.remove(item);continue
                checks=manifest.get('checks') or parent.checks
                ws=Workspace(root,self.root.directory,checks)
                ws.isolated,ws.allow_web=True,parent.allow_web
                ws.record=self.root.record
                ws.proposal_store,ws.owner_id=self,directory.name
                if getattr(parent,'test_provenance',None):
                    from .test_provenance import TestProvenance
                    ws.test_provenance=TestProvenance.restore(ws,self.root.directory/('test-provenance-'+directory.name+'.json'))
                current=self.files(ws)
                baseline=manifest.get('baseline') or {n:sha(d) for n,d in current.items()}
                if not manifest and proposal:
                    for n,before in proposal.get('before',{}).items():
                        if before is None: baseline.pop(n,None)
                        else: baseline[n]=before
                snapshot=Snapshot(ws,parent,baseline,directory.name,parent_id)
                self.snapshots[directory.name]=snapshot
                if proposal:
                    changes={n:current.get(n) for n in proposal['files']}
                    valid=all((sha(d) if d is not None else None)==proposal['after'][n] for n,d in changes.items())
                    proposal['changes']=changes
                    proposal['viewed_by']=[]
                    if not valid and proposal['status']=='proposed':
                        proposal.update(status='stale',reason='Retained branch source changed after checkpoint')
                    self.proposals[directory.name]=proposal
                    ws.changed.update(proposal['files'])
                else:
                    self.propose(directory.name)
                restored[directory.name]=ws
                pending.remove(item);progress=True
            if not progress: break
        return restored

    def propose(self, owner):
        snapshot = self.snapshots[owner]
        current = self.files(snapshot.workspace)
        names = sorted(set(current) | set(snapshot.baseline))
        changes = {n:current.get(n) for n in names if
                   (sha(current[n]) if n in current else None) != snapshot.baseline.get(n)}
        snapshot.workspace.changed.update(changes)
        if len(changes) > int(self.config.get('max_patch_files', 64)):
            raise ValueError('Patch file budget exceeded')
        for data in changes.values():
            if data is not None:
                data.decode('utf-8')  # Binary patches need a separate explicit workflow.
        proposal = {'id':owner, 'parent':snapshot.parent_id, 'status':'proposed',
                    'files':list(changes), 'before':{n:snapshot.baseline.get(n) for n in changes},
                    'after':{n:sha(d) if d is not None else None for n,d in changes.items()},
                    'checks':sorted(snapshot.workspace.passed), 'viewed_by':[],
                    'changes':changes, 'verification':[]}
        self.proposals[owner] = proposal
        self.persist(proposal)
        return self.view(proposal)

    def persist(self, proposal):
        from .harness import _atomic
        value = {k:v for k,v in proposal.items() if k != 'changes'}
        _atomic(self.root.directory/'branches'/proposal['id']/'proposal.json', json.dumps(value,ensure_ascii=False,indent=2))

    def checkpoint(self,owner):
        if owner not in self.snapshots: return
        from .harness import _atomic
        s=self.snapshots[owner]
        _atomic(self.root.directory/'branches'/owner/'snapshot.json',json.dumps({
            'owner':owner,'parent':s.parent_id,'baseline':s.baseline,
            'checks':s.workspace.checks,'revision':s.workspace.revision},ensure_ascii=False))

    @staticmethod
    def view(proposal):
        return {k:v for k,v in proposal.items() if k not in {'changes','viewed_by'}}

    def pending(self, parent='main'):
        return [{k:p[k] for k in ('id','parent','status','files')} for p in self.proposals.values()
                if p['parent']==parent and p['status']=='proposed' and p['files']]

    def patch(self, owner, proposal_id):
        import difflib
        proposal = self.proposals[proposal_id]
        if proposal['parent'] != owner:
            raise ValueError('Only the direct parent may inspect/adopt this proposal')
        parent = self.snapshots[proposal_id].parent
        patches = []
        for name, data in proposal['changes'].items():
            path = parent.path(name)
            before = path.read_text().splitlines(True) if path.exists() else []
            after = data.decode().splitlines(True) if data is not None else []
            patches.extend(difflib.unified_diff(before,after,fromfile='a/'+name,tofile='b/'+name))
        proposal['viewed_by'].append(owner)
        self.persist(proposal)
        return ''.join(patches)[:16000] or 'No source changes'

    def reject(self, owner, proposal_id, reason):
        proposal = self.proposals[proposal_id]
        if proposal['parent'] != owner or not reason.strip():
            raise ValueError('Direct parent and a reason required')
        if proposal['status'] == 'accepted':
            raise ValueError('An accepted proposal cannot be rejected retrospectively')
        proposal.update(status='rejected',reason=reason[:1000])
        self.persist(proposal)
        return self.view(proposal)

    async def accept(self, parent, owner, proposal_id):
        from .harness import _atomic
        from src.runtime import hitl
        proposal = self.proposals[proposal_id]
        if proposal['parent'] != owner or parent is not self.snapshots[proposal_id].parent:
            raise ValueError('Proposal belongs to a different parent workspace')
        if proposal['status'] == 'accepted':
            return {'ok':True,'output':'Already accepted; no duplicate writes'}
        if proposal['status'] != 'proposed' or owner not in proposal['viewed_by']:
            raise ValueError('Inspect patch {id} before accepting a proposed change')
        if hitl.is_plan() or self.control.execution == 'plan':
            raise ValueError('Plan mode forbids patch adoption')
        if not parent.checks:
            return {'ok':False,'output':'No registered checks; proposal retained, not adopted'}
        revision = self.control.version(owner)
        before = self.files(parent)
        conflicts = [n for n in proposal['files'] if
                     (sha(before[n]) if n in before else None) != proposal['before'][n]]
        if conflicts:
            return {'ok':False,'output':'Parent changed after fork; conflicts: '+', '.join(conflicts),'conflicts':conflicts}
        # Validate composition against the CURRENT parent, not just the child's old baseline.
        staging_id = 'validate-'+proposal_id+'-'+str(parent.revision)
        staging = self.fork(parent,staging_id,owner)
        try:
            for name,data in proposal['changes'].items():
                path = staging.path(name)
                if data is None:
                    path.unlink()
                else:
                    _atomic(path,data.decode())
            verification = []
            for name in parent.checks:
                result = await staging.execute('check',{'name':name})
                eid = self.root.record(result)
                verification.append({'name':name,'ok':result['ok'],'evidence_id':eid})
                if not result['ok']:
                    proposal['verification'] = verification
                    self.persist(proposal)
                    return {'ok':False,'output':'Combined branch check failed: '+name+'\n'+result['output'],
                            'verification':verification}
            for name,expected in proposal['after'].items():
                path = staging.path(name)
                actual = sha(path.read_bytes()) if path.exists() else None
                if actual != expected:
                    return {'ok':False,'output':'Check modified proposed source: '+name}
            if revision != self.control.version(owner) or self.control.pause or self.control.is_cancelled(owner):
                return {'ok':False,'output':'Operator changed instructions during validation; adoption cancelled'}
            current = self.files(parent)
            if {n:sha(d) for n,d in current.items()} != {n:sha(d) for n,d in before.items()}:
                return {'ok':False,'output':'Parent files changed during validation; adoption cancelled'}
            written = []
            try:
                for name,data in proposal['changes'].items():
                    target = parent.path(name)
                    if data is None:
                        target.unlink()
                    else:
                        _atomic(target,data.decode())
                    written.append(name)
                    if data is not None and getattr(parent,'test_provenance',None):
                        parent.test_provenance.note_edit(parent,name,before.get(name,b'').decode(),data.decode())
            except BaseException:
                for name in reversed(written):
                    target = parent.path(name)
                    if name in before:
                        _atomic(target,before[name])
                    elif target.exists():
                        target.unlink()
                raise
            parent.changed.update(proposal['files'])
            parent.revision += 1
            parent.passed.clear()
            parent.verified_revision = -1  # Parent must also run its own post-adoption checks.
            proposal.update(status='accepted',verification=verification)
            self.persist(proposal)
            return {'ok':True,'output':json.dumps(self.view(proposal),ensure_ascii=False),
                    'adopted':proposal['files'],'verification':verification}
        finally:
            staging._scratch.cleanup()
            # Validation snapshots are disposable; worker snapshots and diffs remain reviewable.
            shutil.rmtree(staging.root.parent,ignore_errors=True)
            self.snapshots.pop(staging_id,None)

    async def accept_many(self,parent,owner,ids):
        ids=list(dict.fromkeys(str(i) for i in ids))
        if not ids or len(ids)>64: raise ValueError('Use 1..64 proposal ids')
        if len(ids)==1: return await self.accept(parent,owner,ids[0])
        changes,before,after={}, {}, {}
        for proposal_id in ids:
            p=self.proposals[proposal_id]
            if p['parent']!=owner or p['status']!='proposed' or owner not in p['viewed_by']:
                raise ValueError('All proposals must belong to this parent and be inspected')
            for name,data in p['changes'].items():
                if name in after and (after[name]!=p['after'][name] or before[name]!=p['before'][name]):
                    return {'ok':False,'output':'Overlapping branch patches conflict: '+name,'conflicts':[name]}
                changes[name],before[name],after[name]=data,p['before'][name],p['after'][name]
        batch='batch-'+sha(json.dumps(ids).encode()+str(parent.revision).encode())[:16]
        combined={'id':batch,'parent':owner,'status':'proposed','files':sorted(changes),
                  'before':before,'after':after,'changes':changes,'checks':[],'viewed_by':[owner],'verification':[]}
        self.proposals[batch]=combined
        self.snapshots[batch]=Snapshot(None,parent,{},batch,owner)
        try:
            result=await self.accept(parent,owner,batch)
            for proposal_id in ids:
                p=self.proposals[proposal_id]
                p['verification']=combined['verification']
                if result['ok']: p.update(status='accepted',batch=batch)
                self.persist(p)
            return result
        finally:
            self.proposals.pop(batch,None)
            self.snapshots.pop(batch,None)

    def close(self):
        for snapshot in self.snapshots.values():
            if snapshot.workspace is not None: snapshot.workspace._scratch.cleanup()
