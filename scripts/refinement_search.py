"""One bounded reflective prompt-search pilot. Explicit --live, no runtime promotion."""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from src.eval.prompt_search import (Candidate, PromptDelta, REFLECTION_PROMPT,
                                    curate, freeze, mutate, split_manifest)
from src.eval.refinement import Proposal, ROUTING_PROMPT, ROUTING_PROMPT_V2, ShadowRouter, fingerprint
from src.eval.refinement_cases import cases, score
from src.eval.refinement_holdout import validation_cases, test_cases
from refinement_eval import Batch
from codex_behavior_eval import CodexInference


def save(directory, name, value):
    # Every trial uses an exclusive directory; individual artifacts are write-once.
    with (directory/name).open('x') as f:
        json.dump(value,f,ensure_ascii=False,indent=2)


def assess(case, proposal):
    row = {'id':case['id'],**score(proposal,case['expected']),'proposal':proposal.model_dump()}
    router = ShadowRouter(case['tasks']); before = router.snapshot()
    try:
        result = router.apply(case['event'],proposal,before)
        after = router.snapshot()
        retry = router.apply(case['event'],proposal,before)
        retained = all(router.tasks[tid]['requirements'].get(rid)==req
            for tid,task in case['tasks'].items() for rid,req in task['requirements'].items()
            if not any(c['task_id']==tid and c['requirement_id']==rid and c['operation'] in {'revise','retract'}
                       for c in case['expected']))
        row.update(accepted=True,result=result,state=router.tasks,pending=router.pending,
                   untouched_constraints_retained=retained,
                   replay_deduplicated=retry['duplicate'] and router.snapshot()==after)
    except ValueError as exc:
        row.update(accepted=False,error=str(exc),atomic_rejection=router.snapshot()==before)
    row['verified_score'] = int(row['operation_match'] and row['accepted']
        and row.get('untouched_constraints_retained') and row.get('replay_deduplicated'))
    return row


def training_records(archive, train):
    """Re-score archived outputs with today's frozen deterministic evaluator.

    This asserts reproducibility of the replay, not historical source integrity
    (the first experiment did not capture a complete source manifest).
    """
    archived_cases = json.loads((archive/'cases.json').read_text())
    protocol = json.loads((archive/'protocol.json').read_text())
    if fingerprint(archived_cases)!=protocol['cases_sha256'] or archived_cases!=train:
        raise ValueError('training input differs from archived protocol')
    report = json.loads((archive/'result.json').read_text())
    rows = {r['id']:r for r in report['results']}
    records = []
    for case in train:
        old = rows[case['id']]
        if 'error' in old: raise ValueError('archive contains an incomplete run')
        proposal = Proposal.model_validate(old['proposal'])
        row = assess(case,proposal)
        records.append({'case_id':case['id'],'split':'train',
            'input':{'tasks':case['tasks'],'event':case['event']},'output':proposal.model_dump(),
            'feedback':json.dumps({'expected':case['expected'],'assessment':{
                k:v for k,v in row.items() if k not in {'state','pending','proposal','result'}}},ensure_ascii=False),
            'score':row['verified_score'],'complete':True,'infrastructure_error':False,
            'verifier_ok':True,'source_unchanged':True,
            'scope':'Archived proposal re-scored by frozen deterministic evaluator; limited field/text checks.'})
    return records


async def search(directory, archive):
    directory.mkdir(parents=True,exist_ok=False)
    splits = {'train':cases(),'validation':validation_cases(),'test':test_cases()}
    manifest = split_manifest(splits)
    paths = [Path(__file__).resolve(),ROOT/'scripts/refinement_eval.py',ROOT/'scripts/codex_behavior_eval.py']
    paths += [ROOT/'src/eval'/name for name in ('prompt_search.py','refinement.py','refinement_cases.py','refinement_holdout.py')]
    def source_hashes(): return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    before = source_hashes()
    protocol = {'splits':manifest,'source_hashes':before,'calls_limit':6,'tokens_limit':150000,
        'candidate_count':3,'reflection_rounds':1,'batch_size':6,
        'score':'matched expected operations, no extra mutation, valid transition, retained constraints, deduplication',
        'archive_sha256':hashlib.sha256((archive/'result.json').read_bytes()).hexdigest(),
        'backend':'ChatGPT-authenticated Codex CLI default, model ID not pinned',
        'gold_sent_to_actor':False,'test_sent_to_reflector':False,'production_promotion':False}
    save(directory,'protocol.json',protocol)
    save(directory,'cases.json',splits)
    records, excluded = curate(training_records(archive,splits['train']),set(manifest['train']['ids']),limit=6)
    if source_hashes()!=before: raise ValueError('source changed during training replay')
    save(directory,'curation.json',{'selected':records,'excluded':excluded})
    shared = {'calls':0,'tokens':0}
    model = CodexInference(directory/'calls',shared,max_calls=6)
    model.config['max_total_tokens'] = 150000
    trials = {}
    try:
        base = Candidate(ROUTING_PROMPT)
        delta = await model([{'role':'system','content':REFLECTION_PROMPT},
            {'role':'user','content':json.dumps({'instruction':base.prompt,'training_records':records},ensure_ascii=False)}],
            PromptDelta,'low')
        reflected = mutate(base,delta,records)
        manual = Candidate(ROUTING_PROMPT_V2,base.id,'manual-diagnosis-20261106')
        candidates = {c.id:c for c in (base,manual,reflected)}
        labels = {base.id:'v1',manual.id:'v2-manual',reflected.id:'reflected'}
        save(directory,'candidates.json',{'candidates':{k:asdict(v) for k,v in candidates.items()},
                                         'labels':labels,'delta':delta.model_dump()})

        async def evaluate(candidate, batch):
            if source_hashes()!=before: raise ValueError('source changed during campaign')
            cards = [{'case_id':c['id'],'tasks':c['tasks'],'event':c['event']} for c in batch]
            answer = await model([{'role':'system','content':candidate.prompt},
                {'role':'user','content':json.dumps({'independent_cases':cards},ensure_ascii=False)}],Batch,'low')
            proposals = {d.case_id:d.proposal for d in answer.decisions}
            if len(proposals)!=len(answer.decisions) or set(proposals)!={c['id'] for c in batch}:
                raise ValueError('case coverage mismatch')
            if source_hashes()!=before: raise ValueError('source changed during campaign')
            return [assess(c,proposals[c['id']]) for c in batch]

        for cid,candidate in candidates.items():
            trials[cid] = await evaluate(candidate,splits['validation'])
            save(directory,'validation-'+cid+'.json',trials[cid])
            print(json.dumps({'candidate':labels[cid],'verified':sum(r['verified_score'] for r in trials[cid]),
                              'cases':len(trials[cid])}),flush=True)
        vectors = {k:[r['verified_score'] for r in rows] for k,rows in trials.items()}
        frozen = freeze(candidates,vectors,manifest['validation']['ids'],manifest)
        # No test call occurs before this immutable record is written.
        save(directory,'selection.json',frozen)
        selected = candidates[frozen['selected']]
        tested = []
        for offset in range(0,len(splits['test']),6):
            batch = await evaluate(selected,splits['test'][offset:offset+6])
            tested += batch
            save(directory,f'test-{offset//6+1}.json',batch)
        report = {'selected':labels[selected.id],'selection':frozen,'test_cases':len(tested),
            'test_verified':sum(r['verified_score'] for r in tested),'test_exact':sum(r['exact'] for r in tested),
            'calls':shared['calls'],'usage':{k:v for k,v in model.usage().items() if not k.startswith('cost')},
            'cost_usd':None,'source_unchanged':source_hashes()==before,'production_promoted':False,
            'limits':['one seed, small synthetic hand-labelled split','same unpinned strong CLI model for reflection and action',
                      'finite field/text oracle; no proof of general semantic correctness',
                      'validation selection is not statistical evidence of superiority','shadow only, no worker execution'],
            'results':tested}
        save(directory,'result.json',report)
        print(json.dumps({k:v for k,v in report.items() if k not in {'selection','results','limits'}}),flush=True)
    except BaseException as exc:
        save(directory,'incomplete.json',{'status':'incomplete','error':type(exc).__name__+': '+str(exc)[:500],
                                         'calls':shared['calls'],'tokens':shared['tokens'],
                                         'completed_validation':list(trials),'production_promoted':False})
        raise


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live',action='store_true')
    parser.add_argument('--output',type=Path,default=Path('data/eval/refinement-fix-20261107/search'))
    parser.add_argument('--archive',type=Path,default=Path('data/eval/refinement-20261106/live'))
    args = parser.parse_args()
    if args.live: asyncio.run(search(args.output.resolve(),args.archive.resolve()))
    else: print(json.dumps({'live':False,'reflection_rounds':1,'candidates':3,'calls_limit':6,
                           'tokens_limit':150000,'validation_cases':6,'test_cases':12,'production_promotion':False}))
