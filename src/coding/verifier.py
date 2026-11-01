"""Active review tools. Experiments run in a disposable snapshot, never the artifact."""
from __future__ import annotations

import shutil
import sys
import uuid
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Probe(BaseModel):
    model_config = ConfigDict(extra='forbid')
    tool: Literal['read', 'search', 'tree', 'evidence', 'math', 'probe','answer']
    args: dict
    purpose: str = Field(max_length=400)


class ProbePlan(BaseModel):
    model_config = ConfigDict(extra='forbid')
    probes: list[Probe] = Field(max_length=3)
    limitation: str = Field(max_length=400)


async def isolated_probe(ws, code, *, preserve_source=False):
    from .branches import BranchStore, sha
    from .control import SessionControl
    from src.runtime import hitl
    if hitl.is_plan() or getattr(getattr(ws, 'orchestrator', None), 'control', None) and ws.orchestrator.control.execution == 'plan':
        return {'ok': False, 'output': 'Plan mode forbids active execution'}
    if not isinstance(code, str) or not 0 < len(code) <= 12000:
        raise ValueError('Probe code must contain 1..12000 characters')
    store = BranchStore(ws, SessionControl(), {})
    owner = 'probe-' + uuid.uuid4().hex[:12]
    staging = store.fork(ws, owner, getattr(ws, 'owner_id', 'main'))
    baseline = store.snapshots[owner].baseline
    try:
        # -c imports the snapshot's current project. Temporary tests may mutate only this copy.
        result = await staging.command([sys.executable, '-c', code], timeout=15)
        result['verification_scope'] = 'finite experiment in isolated current-source snapshot; not a proof'
        result['source_hashes'] = baseline
        if preserve_source:
            current={n:sha(d) for n,d in store.files(staging).items()}
            changed=[name for name,expected in baseline.items() if current.get(name)!=expected]
            if changed:
                result.update(ok=False,error_kind='tool_error',
                    output='Diagnostic probe changed original snapshot files; observations do not describe the supplied program: '+
                           ', '.join(changed[:8])+'\n'+str(result.get('output',''))[-2000:])
        if {n: sha(d) for n, d in store.files(ws).items()} != baseline:
            return {'ok': False, 'stale': True, 'output': 'Source changed during probe; discard its conclusion'}
        return result
    finally:
        store.close()
        shutil.rmtree(staging.root.parent, ignore_errors=True)


async def run_probes(ws, plan, allowed, supervisor, observer, control, candidate=None,answer_keys=()):
    observations = []
    version = control.version(getattr(ws, 'owner_id', 'main'))
    for probe in plan.probes:
        from .control import Steered
        if control.pause or control.version(getattr(ws, 'owner_id', 'main')) != version:
            raise Steered()
        if probe.tool not in allowed:
            result = {'ok': False, 'error_kind':'tool_error','output': 'Verifier tool unavailable in this workspace: ' + probe.tool}
        else:
            try:
                if probe.tool in {'probe','answer'}:
                    if probe.tool=='answer':
                        from .answer_verification import run_answer_program
                        result=await run_answer_program(ws,probe.args.get('code',''),
                            # New programs export only computed fields; prose keys are excluded.
                            tuple(probe.args.get('outputs',{})) if 'outputs' in probe.args else answer_keys,
                            candidate=candidate,outputs=probe.args.get('outputs'))
                    else:result = await isolated_probe(ws, probe.args.get('code'),
                        preserve_source=bool(probe.args.get('preserve_source',False)))
                    if probe.tool=='answer' and result['ok']:
                        from .answer_verification import bind_answer
                        result={**result,**bind_answer(candidate,result['output'],answer_keys)}
                    elif probe.tool=='answer':result['error_kind']='tool_error'
                else:
                    result = await ws.execute(probe.tool, probe.args, readonly=True)
            except (ValueError, OSError, KeyError, TypeError) as exc:
                result = {'ok': False, 'error_kind':'tool_error','output': str(exc)[:1000]}
        result.update(verifier=True, tool=probe.tool, purpose=probe.purpose, args=probe.args)
        eid = ws.record(result)
        supervisor.action(observer, probe.tool, result['ok'], eid)
        observations.append({'id': eid, 'tool': probe.tool, 'purpose': probe.purpose,
                             'args': probe.args,
                             'ok': result['ok'], 'exit_code': result.get('exit_code'),
                             'output': str(result.get('output', ''))[-3500:],
                             'verification_scope': result.get('verification_scope', 'source observation'),
                             'error_kind':result.get('error_kind'),
                             **{k:result[k] for k in ('answer_checks','answer_binding','checked_fields') if k in result},
                             'stale': result.get('stale', False)})
    return observations
