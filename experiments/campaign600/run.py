"""600-case opt-in campaign: 599 batched cases plus separately built product.

Synthetic variants are reported by family and evaluation depth. No benchmark claim.
"""
from __future__ import annotations
import argparse
import asyncio
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from pydantic import BaseModel,ConfigDict,Field
from codex_behavior_eval import CodexInference
from src.coding.budget import BudgetExceeded,token_scope
from src.eval.refinement import Proposal,ShadowRouter,ROUTING_PROMPT_V2,fingerprint
from src.eval.refinement_cases import score
from experiments.campaign600.cases import build
from experiments.campaign600.sandbox import IMAGE,canonical,grade,run as sandbox_run,write_files


class Strict(BaseModel):model_config=ConfigDict(extra='forbid')
class Decision(Strict):
    case_id:str
    proposal:Proposal
class DialogBatch(Strict):
    replies:list[Decision]=Field(min_length=1,max_length=8)
class Answer(Strict):
    case_id:str
    answer_json:str
    explanation:str=Field(max_length=400)
class AnswerBatch(Strict):
    replies:list[Answer]=Field(min_length=1,max_length=10)
class File(Strict):
    path:str
    content:str
class Artifact(Strict):
    case_id:str
    files:list[File]=Field(min_length=1,max_length=24)
    summary:str=Field(max_length=800)
class ArtifactBatch(Strict):
    replies:list[Artifact]=Field(min_length=1,max_length=6)


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2));temp.replace(path)


def usage(directory):
    calls=tin=tout=unknown=0
    for path in directory.glob('pool-*/call-*/request.json'):
        calls+=1;events=path.parent/'events.jsonl';counts=[]
        if events.exists():
            for line in events.read_text().splitlines():
                try:event=json.loads(line)
                except ValueError:continue
                if 'usage' in event:counts.append(event['usage'])
        if not counts:unknown+=1
        for item in counts:tin+=item['input_tokens'];tout+=item['output_tokens']
    return {'calls':calls,'known_input_tokens':tin,'known_output_tokens':tout,
            'known_total_tokens':tin+tout,'calls_without_receipt':unknown,'cost_usd':None}


class Campaign:
    def __init__(self,output):
        self.output=output;self.rows=build();self.shared={'calls':0,'tokens':0}
        self.sandboxes=asyncio.Semaphore(2);self.started=time.monotonic()
        self.manifest=fingerprint(self.rows)
        files=[Path(__file__),ROOT/'scripts/codex_behavior_eval.py',
               ROOT/'experiments/campaign600/cases.py',ROOT/'experiments/campaign600/sandbox.py',
               ROOT/'src/eval/refinement.py',ROOT/'src/eval/refinement_cases.py']
        self.source_paths=files
        self.hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        protocol={'cases':600,'manifest_sha256':self.manifest,'source_hashes':self.hashes,
                  'limits':{'calls':800,'tokens':8000000,'cloud_concurrency':2,'repair_rounds':1},
                  'docker_image':IMAGE,'backend':'authenticated Codex CLI default; model ID not pinned',
                  'dialogue_scope':'shadow state transitions, three actual successive decisions',
                  'code_scope':'generated Python solve() executed in Docker against external cases',
                  'project_scope':'generated multi-file CLI projects, external tests plus own unit tests',
                  'product_scope':'separate real SEA CodingHarness actor/tool/critic run plus external acceptance',
                  'variants':'24 dialogue families x10 contexts; 20 algorithm families x8 per code/reasoning tier; 13 project families x3',
                  'gold_sent_to_actor':False,'cost_usd':None}
        if (output/'protocol.json').exists():
            if json.loads((output/'protocol.json').read_text())!=protocol:raise ValueError('campaign source/config changed; use a new output directory')
        else:
            output.mkdir(parents=True,exist_ok=True)
            save(output/'cases.json',self.rows);save(output/'protocol.json',protocol)
        old=usage(output/'calls');self.shared.update(calls=old['calls'],tokens=old['known_total_tokens'])
        self.prior_bound=old['known_total_tokens']+old['calls_without_receipt']*100000

    def verify_source(self):
        current={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in self.source_paths}
        if current!=self.hashes:raise RuntimeError('Evaluator source changed during campaign')

    def result_path(self,c):return self.output/'results'/(c['id']+'.json')

    def complete(self,c):
        path=self.result_path(c)
        return path.exists() and json.loads(path.read_text()).get('status')!='infrastructure_error'

    def record(self,c,result):
        save(self.result_path(c),{'id':c['id'],'kind':c['kind'],'family':c['family'],
                                'variant':c['variant'],**result})
        rows=[json.loads(p.read_text()) for p in (self.output/'results').glob('*.json')]
        counts={}
        for row in rows:
            item=counts.setdefault(row['kind'],{'completed':0,'passed':0,'infrastructure_errors':0})
            if row['status']=='infrastructure_error':item['infrastructure_errors']+=1
            else:item['completed']+=1;item['passed']+=bool(row.get('passed'))
        status={'target':600,'categories':counts,'runtime_calls':self.shared['calls'],
                'runtime_known_tokens':self.shared['tokens'],'elapsed_seconds':round(time.monotonic()-self.started),
                'product':'evaluated separately; never counted as passed here'}
        save(self.output/'progress.json',status)
        if len(rows)%10==0:print(json.dumps(status),flush=True)

    async def ask(self,model,system,cards,schema):
        self.verify_source()
        for attempt in range(2):
            try:
                response=await model([{'role':'system','content':system},
                    {'role':'user','content':json.dumps({'independent_cases':cards},ensure_ascii=False)}],schema,'low')
                replies={r.case_id:r for r in response.replies}
                if len(replies)!=len(response.replies) or set(replies)!={c['case_id'] for c in cards}:
                    raise ValueError('response did not cover each case exactly once')
                self.verify_source();return replies
            except BudgetExceeded:raise
            except (RuntimeError,ValueError,asyncio.TimeoutError):
                if attempt:raise
                await asyncio.sleep(2)

    async def dialogue(self,model,batch):
        states={}
        for c in batch:
            path=self.output/'dialogues'/(c['id']+'.json')
            states[c['id']]=json.loads(path.read_text()) if path.exists() else {
                'tasks':copy.deepcopy(c['tasks']),'events':{},'pending':[],'revision':0,'turns':[]}
        for index in range(3):
            active=[c for c in batch if len(states[c['id']]['turns'])==index]
            if not active:continue
            cards=[{'case_id':c['id'],'tasks':states[c['id']]['tasks'],'event':c['turns'][index]['event']} for c in active]
            replies=await self.ask(model,ROUTING_PROMPT_V2,cards,DialogBatch)
            for c in active:
                state=states[c['id']];gold=c['turns'][index]['expected'];event=c['turns'][index]['event']
                proposal=replies[c['id']].proposal;row=score(proposal,gold)
                # Prerequisite order has no scheduling meaning; retain exact separately.
                normalized=proposal.model_copy(deep=True)
                for change in normalized.changes:change.depends_on.sort()
                ng=copy.deepcopy(gold)
                for g in ng:g['depends_on'].sort()
                row['operation_match']=score(normalized,ng)['operation_match']
                router=ShadowRouter(state['tasks']);router.events=state['events'];router.pending=state['pending'];router.revision=state['revision']
                before=router.snapshot();old=copy.deepcopy(router.tasks)
                row['proposal']=proposal.model_dump()
                try:
                    effects=router.apply(event,proposal,before);after=router.snapshot()
                    duplicate=router.apply(event,proposal,before)
                    retained=all(router.tasks[tid]['requirements'].get(rid)==req
                        for tid,task in old.items() for rid,req in task['requirements'].items()
                        if not any(g['task_id']==tid and g['requirement_id']==rid and g['operation'] in {'revise','retract'} for g in gold))
                    row.update(accepted=True,effects=effects,retained=retained,
                               deduplicated=duplicate['duplicate'] and router.snapshot()==after)
                except ValueError as exc:row.update(accepted=False,error=str(exc),atomic_rejection=router.snapshot()==before)
                row['passed']=bool(row['operation_match'] and row['accepted'] and row.get('retained') and row.get('deduplicated'))
                state.update(tasks=router.tasks,events=router.events,pending=router.pending,revision=router.revision)
                state['turns'].append(row);save(self.output/'dialogues'/(c['id']+'.json'),state)
        for c in batch:
            turns=states[c['id']]['turns']
            self.record(c,{'status':'evaluated','passed':len(turns)==3 and all(t['passed'] for t in turns),
                           'turns_passed':sum(t['passed'] for t in turns),'turns':len(turns),
                           'exact_turns':sum(t['exact'] for t in turns)})

    async def reasoning(self,model,batch):
        cards=[{'case_id':c['id'],'task':c['spec'],'input':c['input']} for c in batch]
        replies=await self.ask(model,'Solve each independent calculation/data task. Return answer_json encoding exactly its requested JSON value, and a brief explanation. Do not execute tools or invent results.',cards,AnswerBatch)
        for c in batch:
            reply=replies[c['id']]
            try:answer=json.loads(reply.answer_json);passed=canonical(answer)==canonical(c['expected'])
            except (ValueError,TypeError):answer=reply.answer_json;passed=False
            self.record(c,{'status':'evaluated','passed':passed,'answer':answer,'explanation':reply.explanation,
                           'scope':'Exact structured output; explanation not automatically judged'})

    async def artifact(self,model,batch):
        cards=[]
        for c in batch:
            layout=('Create solution.py implementing solve(data). Python 3.12 stdlib only. No input/output at import time.' if c['kind']=='code' else
                    'Create a complete small Python 3.12 stdlib project: core.py with solve(data), cli.py reading JSON lines from stdin and writing one result JSON per line, README.md with usage, and runnable unittest tests. Use python cli.py and python -m unittest discover. No dependencies or installation.')
            cards.append({'case_id':c['id'],'task':c['spec'],'layout':layout,'public_examples':c['public']})
        system='Build each independent requested artifact. Return actual files, not a plan. Keep code concise, general and deterministic; implement edge cases from the contract. Never use external tools, networking or packages. Do not embed example answers as lookup tables.'
        replies=await self.ask(model,system,cards,ArtifactBatch)
        entries={}
        async def inspect(c,reply,attempt):
            directory=self.output/'artifacts'/c['id']/f'attempt-{attempt}-{uuid.uuid4().hex[:8]}'
            files=[f.model_dump() for f in reply.files]
            paths=write_files(directory,files)
            async with self.sandboxes:
                dev=await grade(directory,c['checks'][:8],c['kind'])
                unit=await sandbox_run(directory,['python','-m','unittest','discover','-v']) if c['kind']=='project' else None
            if dev['process']['exit_code']==125:raise RuntimeError('Docker infrastructure failure: '+dev['process']['stderr'][:500])
            layout=(('solution.py' in paths) if c['kind']=='code' else
                    all(p in paths for p in ('core.py','cli.py','README.md')) and any(Path(p).name.startswith('test') for p in paths))
            if unit is not None:layout=layout and unit['exit_code']==0 and 'Ran 0 tests' not in unit['stderr']
            result={'directory':str(directory),'files':files,'development':dev,'unit':unit,'layout':bool(layout),'attempt':attempt}
            save(directory/'evaluation.json',{k:v for k,v in result.items() if k!='files'})
            return result
        for c in batch:
            try:entries[c['id']]=await inspect(c,replies[c['id']],1)
            except ValueError as exc:entries[c['id']]={'files':[f.model_dump() for f in replies[c['id']].files],
                'development':{'passed':False,'error':str(exc)},'layout':False,'unit':None,'attempt':1}
        failed=[c for c in batch if not entries[c['id']]['development']['passed'] or not entries[c['id']]['layout']]
        if failed:
            repairs=[]
            for c in failed:
                entry=entries[c['id']]
                repairs.append({**next(card for card in cards if card['case_id']==c['id']),
                    'previous_files':entry['files'],'development_feedback':entry['development'],
                    'unit_feedback':entry['unit'],'layout_valid':entry['layout']})
            fixed=await self.ask(model,system+' Repair the actual failures; return the complete corrected file set. This is the only repair round.',repairs,ArtifactBatch)
            for c in failed:
                try:entries[c['id']]=await inspect(c,fixed[c['id']],2)
                except ValueError as exc:entries[c['id']]={'development':{'passed':False,'error':str(exc)},'layout':False,'attempt':2}
        for c in batch:
            entry=entries[c['id']];hidden={'passed':False,'matched':0,'checks':len(c['checks'][8:])}
            if entry.get('directory'):
                async with self.sandboxes:hidden=await grade(Path(entry['directory']),c['checks'][8:],c['kind'])
                save(Path(entry['directory'])/'heldout.json',hidden)
            self.record(c,{'status':'evaluated','passed':bool(entry['layout'] and entry['development']['passed'] and hidden['passed']),
                'attempts':entry['attempt'],'development_passed':entry['development']['passed'],
                'heldout_passed':hidden['passed'],'heldout_matched':hidden['matched'],'heldout_checks':hidden['checks'],
                'layout_passed':entry['layout'],'artifact':entry.get('directory'),
                'scope':'Generated artifact adapter, not the full interactive SEA tool loop'})

    async def execute(self):
        jobs=[]
        for kind,size in [('dialogue',8),('reasoning',10),('code',6),('project',3)]:
            rows=[c for c in self.rows if c['kind']==kind and not self.complete(c)]
            jobs.extend((kind,rows[i:i+size]) for i in range(0,len(rows),size))
        random.Random(1147).shuffle(jobs)
        queue=asyncio.Queue()
        for job in jobs:queue.put_nowait(job)
        async def worker(index):
            directory=self.output/'calls'/f'pool-{index}'
            model=CodexInference(directory,self.shared,max_calls=800,shared_call_limit=800,
                shared_token_limit=8000000,timeout_seconds=240,output_bound=24000,payload_object=True)
            model.calls=max((int(p.name.split('-')[-1]) for p in directory.glob('call-*')),default=0)
            model.config['max_total_tokens']=8000000
            consecutive_errors=0
            while not queue.empty():
                kind,batch=queue.get_nowait()
                try:
                    method=self.dialogue if kind=='dialogue' else self.reasoning if kind=='reasoning' else self.artifact
                    await method(model,batch);consecutive_errors=0
                except BudgetExceeded:raise
                except (RuntimeError,ValueError,asyncio.TimeoutError) as exc:
                    consecutive_errors+=1
                    for c in batch:self.record(c,{'status':'infrastructure_error','passed':False,'error':type(exc).__name__+': '+str(exc)[:600]})
                    if consecutive_errors>=3:raise RuntimeError('Three consecutive infrastructure/protocol failures; preserve pending cases') from exc
                finally:queue.task_done()
        with token_scope(8000000) as budget:
            budget.spent=self.prior_bound
            workers=[asyncio.create_task(worker(0)),asyncio.create_task(worker(1))]
            try:await asyncio.gather(*workers)
            finally:
                for worker_task in workers:
                    if not worker_task.done():worker_task.cancel()
                await asyncio.gather(*workers,return_exceptions=True)
                save(self.output/'budget.json',budget.packet());save(self.output/'usage.json',usage(self.output/'calls'))
        rows=[json.loads(p.read_text()) for p in (self.output/'results').glob('*.json')]
        save(self.output/'result-599.json',{'expected_cases':599,'records':len(rows),
            'evaluated':sum(r['status']=='evaluated' for r in rows),'passed':sum(r.get('passed',False) for r in rows),
            'usage':usage(self.output/'calls'),'results':rows,'product_pending_separate_acceptance':True})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live',action='store_true')
    parser.add_argument('--output',type=Path,default=ROOT/'data/eval/campaign600-20261107/batch')
    args=parser.parse_args()
    campaign=Campaign(args.output.resolve())
    if args.live:asyncio.run(campaign.execute())
    else:print(json.dumps({'prepared':600,'model_calls':0,'manifest':campaign.manifest}))
