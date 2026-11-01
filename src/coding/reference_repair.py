"""Bounded, opt-in repair of agent-authored scalar test expectations.

The same diagnostic call budget buys a blind executable reference, not a verdict.
Only supported, proven-new tests may be changed, after isolated checks succeed.
"""
from __future__ import annotations

import difflib
import json
import shutil
import sys
import uuid
from pydantic import BaseModel,ConfigDict,Field


class ReferenceProgram(BaseModel):
    model_config=ConfigDict(extra='forbid')
    code:str=Field(min_length=1,max_length=12000)


async def prepare_repair(critic,node,ws,check,failure):
    from src.runtime import hitl
    from .assertion_repair import prepare_assertion_plan
    from .test_trace import target_test
    history=getattr(ws,'test_provenance',None)
    contract=critic.contracts.get((node.id,critic.control.version(node.id)))
    if (not critic.config.get('reference_repair',False) or not history or not hitl.full_auto()
            or hitl.is_plan() or critic.control.execution=='plan' or not critic.config.get('active',False)
            or 'probe' not in ws.verifier_tools or not {'edit','check'}<=node.ceiling
            or node.selection is not None and not {'edit','check'}<=node.selection
            or node.override and not {'edit','check'}<=node.tools
            or not getattr(ws,'orchestrator',None) or not node.memory.plan
            or not contract or not any(r.validation=='new_or_updated_tests' for r in contract.requirements)):
        return None
    selected=target_test(ws.checks[check],str(failure.get('output','')))
    if not selected:return None
    target,path=selected;cls,method=target.split('.')[-2:]
    original=history.eligible_original(ws,path,cls,method,critic.control.version(node.id))
    if original is None:return None
    try:
        source=await ws.source_text(path,131073)
        if len(source)>131072:return None
        plan=prepare_assertion_plan(source,original,cls,method)
        if not await ws.file_exists(plan.module.replace('.','/')+'.py'):return None
        return path,plan
    except (SyntaxError,ValueError,OSError,TypeError):return None


def read_values(result,count):
    if not result.get('ok') or result.get('stale'):raise ValueError('Unusable execution observation')
    packets=[json.loads(line.removeprefix('SEA_ASSERTION_VALUES '))
        for line in str(result.get('output','')).splitlines() if line.startswith('SEA_ASSERTION_VALUES ')]
    if len(packets)!=1 or not isinstance(packets[0],list) or len(packets[0])!=count:
        raise ValueError('Incomplete or ambiguous execution values')
    return packets[0]


async def repair_expectations(critic,node,ws,prepared,observer):
    from .branches import BranchStore,sha
    from .harness import Workspace,_atomic
    from .verifier import isolated_probe
    from .control import Steered
    from .model import BudgetExceeded
    path,plan=prepared;control=critic.control
    version=control.version(node.id);generation=node.generation;revision=ws.revision
    store=BranchStore(ws,control,{})
    baseline={n:sha(data) for n,data in store.files(ws).items()}
    evidence=[];post_checks=[]
    def current():
        return (version==control.version(node.id) and generation==node.generation and revision==ws.revision
            and not control.pause and not control.is_cancelled(node.id)
            and baseline=={n:sha(data) for n,data in store.files(ws).items()})
    def record(result,tool):
        result.update(kind='reference_repair_observation',critic_source=True)
        eid=ws.record(result);evidence.append(eid)
        critic.supervisor.action(observer,tool,bool(result.get('ok')),eid)
        return eid
    artifact=ws.directory/'reference-repairs'/uuid.uuid4().hex[:12]
    artifact.mkdir(parents=True)
    oracle_root=artifact/'oracle';oracle_root.mkdir()
    oracle=Workspace(oracle_root,ws.directory,{})
    staging=None
    applied=False;problem=None
    try:
        messages=[{'role':'system','content':'Write a simple independent Python reference for the task. Export '+
            plan.constructor+'. No tests. Preserve the required behavior, including state changes. '
            'You have not seen the candidate implementation, test assertions or observed results.'},
            {'role':'user','content':json.dumps({'task':node.task,'human_steering':control.instructions(node.id)},ensure_ascii=False)}]
        engine=critic.remote or critic.model
        cap=max(256,min(8192,int(critic.config.get('reference_output_tokens',1024))))
        with critic.supervisor.scope(observer):
            reference=await control.call(lambda: engine.bounded_call(messages,ReferenceProgram,'low',cap)
                if hasattr(engine,'bounded_call') else engine(messages,ReferenceProgram,'low'),observer)
        reference=ReferenceProgram.model_validate(reference.model_dump() if isinstance(reference,BaseModel) else reference)
        if not current():raise Steered()
        code_id=ws.record({'ok':True,'kind':'blind_reference_program','critic_source':True,'output':reference.code,
            'scope':'Generated from task/operator instructions, without candidate source or observed values'})
        evidence.append(code_id)
        # The reference runs in an empty workspace; candidate files are never copied into it.
        expected=await oracle.command([sys.executable,'-c',plan.observation_program(reference.code)],timeout=15)
        record(expected,'probe')
        values=read_values(expected,len(plan.operations))
        if not current():raise Steered()
        actual=await isolated_probe(ws,plan.observation_program(),preserve_source=True)
        record(actual,'probe')
        observations=read_values(actual,len(plan.operations))
        proposal=plan.proposal([values[a['step']] for a in plan.assertions],observations)
        if proposal==plan.source:raise ValueError('No supported expectation change was established')
        if not current():raise Steered()
        candidate='reference-candidate-'+uuid.uuid4().hex[:12]
        staging=store.fork(ws,candidate,node.id)
        _atomic(staging.path(path),proposal)
        expected_files={**baseline,path:sha(proposal.encode())}
        for name in ws.checks:
            if not current():raise Steered()
            checked=await staging.execute('check',{'name':name});record(checked,'check')
            if not checked['ok']:raise ValueError('Candidate failed registered check: '+name)
        if expected_files!={n:sha(data) for n,data in store.files(staging).items()}:
            raise ValueError('Candidate check changed source files; discard repair')
        if not current():raise Steered()
        # Revalidate the source history after all awaits, including external edits.
        if ws.test_provenance.eligible_original(ws,path,plan.class_name,plan.method_name,version) is None:
            raise ValueError('Authorship evidence changed during repair')
        patch=''.join(difflib.unified_diff(plan.source.splitlines(True),proposal.splitlines(True),fromfile='a/'+path,tofile='b/'+path))
        record({'ok':True,'output':patch,'kind':'reference_repair_candidate'},'patch')
        scheduler=ws.orchestrator.scheduler
        if not node.override:scheduler.switch(node.id,'repair',override=False)
        if not {'edit','check'}<=node.tools:raise ValueError('Current profile does not authorize repair')
        result=await ws.execute('edit',{'path':path,'old':plan.source,'new':proposal})
        record(result,'edit')
        if not result['ok']:raise ValueError('Applying the checked repair was denied or failed')
        applied=True
        for name in ws.checks:
            if version!=control.version(node.id) or control.pause or control.is_cancelled(node.id):raise Steered()
            result=await ws.execute('check',{'name':name});eid=record(result,'check')
            post_checks.append({'name':name,'evidence_id':eid,'result':result})
            if not result['ok']:break
        if not node.override:scheduler.switch(node.id,'verify',override=False)
    except (Steered,BudgetExceeded):raise
    except (OSError,ValueError,KeyError,TypeError) as exc:
        problem=type(exc).__name__+': '+str(exc)[:600]
    finally:
        store.close();oracle._scratch.cleanup()
        if staging is not None:shutil.rmtree(staging.root.parent,ignore_errors=True)
        shutil.rmtree(artifact,ignore_errors=True)
    return {'status':'checked_reference_repair' if applied else 'reference_repair_not_applied',
        'category':'test_expectation' if applied else 'uncertain',
        'reason':('A blind executable reference supplied matching scalar expectations; finite evidence, not proof.'
                  if applied else 'No checked expectation change was applied; inspect the reported problem.'),
        'repair_applied':applied,'changed_paths':[path] if applied else [],'post_checks':post_checks,
        'evidence':evidence,'problem':problem,
        'scope':{'revision':ws.revision,'steering_version':control.version(node.id),
            'source_hashes':{p:ws.file_digest(p) for p in (path,plan.module.replace('.','/')+'.py')}},
        'next_steps':['Review against the full task and current evidence; no completion authority is granted.']}
