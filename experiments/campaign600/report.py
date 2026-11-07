"""Summarize recorded results without rerunning models or changing raw verdicts."""
import argparse
import json
from collections import Counter,defaultdict
from pathlib import Path


def receipts(directory):
    result=Counter(calls=0,unknown_receipts=0,input_tokens=0,output_tokens=0,cached_input_tokens=0)
    for request in directory.rglob('request.json'):
        result['calls']+=1
        events=request.with_name('events.jsonl');found=False
        if events.exists():
            for line in events.read_text().splitlines():
                try:row=json.loads(line)
                except ValueError:continue
                if 'usage' not in row:continue
                found=True
                for key in ('input_tokens','output_tokens','cached_input_tokens'):
                    result[key]+=int(row['usage'].get(key,0))
        if not found:result['unknown_receipts']+=1
    return dict(result,total_tokens=result['input_tokens']+result['output_tokens'],cost_usd=None)


def build(root):
    batch=root/'batch-v1';rows=[json.loads(p.read_text()) for p in sorted((batch/'results').glob('*.json'))]
    categories=defaultdict(Counter);families=defaultdict(Counter)
    for row in rows:
        for target in (categories[row['kind']],families[row['kind']+'/'+row['family']]):
            target['records']+=1;target['passed']+=bool(row.get('passed'))
            target['evaluated']+=row['status']=='evaluated'
            if 'attempts' in row:
                target['repair_used']+=row['attempts']>1
            target['heldout_matched']+=row.get('heldout_matched',0)
            target['heldout_checks']+=row.get('heldout_checks',0)
    attempts=[]
    attempt_dirs=[p for p in root.glob('product-attempt-*') if p.is_dir() and p.name.rsplit('-',1)[-1].isdigit()]
    for path in [root/'product',*sorted(attempt_dirs,key=lambda p:int(p.name.rsplit('-',1)[-1]))]:
        if not path.is_dir():continue
        outcome=path/'result.json' if (path/'result.json').exists() else path/'incomplete.json'
        data=json.loads(outcome.read_text()) if outcome.exists() else {}
        attempts.append({'attempt':path.name,'status':data.get('status','interrupted' if outcome.exists() else 'running'),
                         'completion':data.get('completion',{}).get('state'),
                         'usage':receipts(path/'calls')})
    result={'campaign':'2026-11-07','requested_cases':600,'batch_records':len(rows),
        'batch_evaluated':sum(r['status']=='evaluated' for r in rows),
        'batch_raw_passed':sum(bool(r.get('passed')) for r in rows),
        'categories':dict(categories),'families':dict(families),
        'raw_failures':[r for r in rows if not r.get('passed')],
        'batch_usage':receipts(batch/'calls'),'product_attempts':attempts,
        'product_current_status':'Requires separate full-harness result and independent acceptance; never inferred from batch scores',
        'limits':['Synthetic parameterized families, not 600 distinct algorithms',
                  'Only product runs use the full SEA actor/tool loop',
                  'Dialogue scores include literal text matching; inspect false negatives',
                  'CLI model ID not pinned; no small-model or public-benchmark claim',
                  'Operator product implementation was removed at user request and is not counted']}
    result['product_usage']={key:sum(a['usage'][key] for a in attempts)
                            for key in ('calls','unknown_receipts','input_tokens','output_tokens','cached_input_tokens','total_tokens')}
    result['product_usage']['cost_usd']=None
    validation=root/'product-validation.json'
    if validation.exists():
        result['product_validation']=json.loads(validation.read_text())
        result['product_current_status']=result['product_validation'].get('outcome',result['product_current_status'])
    diagnostics={}
    for name in ('transport-canary','transport-canary-v2','critic-transport-replay'):
        if (root/name).exists():diagnostics[name]=receipts(root/name)
    result['separate_transport_diagnostics']=diagnostics
    adjudications=root/'adjudications.json'
    if adjudications.exists():result['separate_manual_adjudications']=json.loads(adjudications.read_text())
    (root/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('batch_records','batch_evaluated','batch_raw_passed','categories')},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path('data/eval/campaign600-20261107'))
    build(parser.parse_args().root)
