"""Opt-in integration: real cloud workers edit a multi-file CLI and parent adopts patches.

The independent acceptance program is outside the model's workspace and is never
passed to it. This is an integration scenario, not a benchmark score.
"""
from __future__ import annotations
import argparse
import asyncio
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

FILES={
 'reader.py': '''def records(path):
    with open(path, encoding="utf-8") as source:
        for line in source:
            if line.strip():
                yield line.rstrip("\\n")
''',
 'writer.py': '''def write_rows(rows, destination):
    with open(destination, "w", encoding="utf-8") as out:
        out.write("name,count\\n")
        for name, count in rows:
            out.write(f"{name},{count}\\n")
''',
 'app.py': '''import argparse
from collections import Counter
from reader import records
from writer import write_rows

def convert(source, destination):
    counts=Counter(records(source))
    write_rows(sorted(counts.items()), destination)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("destination")
    args=parser.parse_args()
    convert(args.source,args.destination)

if __name__ == "__main__": main()
''',
 'test_existing.py': '''import tempfile, unittest
from pathlib import Path
from app import convert

class Existing(unittest.TestCase):
    def test_plain_compatibility(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); src=root/"in.txt"; dst=root/"out.csv"
            src.write_text("alice\\nbob\\nalice\\n")
            convert(src,dst)
            self.assertEqual(dst.read_text(),"name,count\\nalice,2\\nbob,1\\n")
''',
 'README.md': '''# Event-name counter
Run: python app.py source.txt result.csv
API: reader.records(path) yields nonempty lines without their newline;
writer.write_rows(rows, destination) writes header name,count followed by pairs;
app.convert(source,destination) counts names and sorts them.
The unit check is registered as unit. Preserve these APIs and plain-text behavior.
'''
}

ACCEPTANCE=r'''
import csv,gzip,tempfile,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from app import convert
from reader import records
from writer import write_rows
with tempfile.TemporaryDirectory() as t:
    d=Path(t); src=d/'events.txt.gz'; dst=d/'result.csv'
    with gzip.open(src,'wt',encoding='utf-8') as f:
        f.write('alpha,beta\n"quoted"\nalpha,beta\nкириллица\n\n')
    assert list(records(src))==['alpha,beta','"quoted"','alpha,beta','кириллица']
    convert(src,dst)
    with dst.open(newline='',encoding='utf-8') as f: rows=list(csv.reader(f))
    assert rows==[['name','count'],['"quoted"','1'],['alpha,beta','2'],['кириллица','1']],rows
    dst.write_text('previous valid output',encoding='utf-8')
    def fail_mid_stream():
        yield ('first',1)
        raise RuntimeError('source broke')
    try: write_rows(fail_mid_stream(),dst)
    except RuntimeError: pass
    else: raise AssertionError('must propagate producer failure')
    assert dst.read_text()=='previous valid output','partial file replaced valid output'
    write_rows([('line\nbreak',2),('quote"comma,',3)],dst)
    with dst.open(newline='',encoding='utf-8') as f: rows=list(csv.reader(f))
    assert rows==[['name','count'],['line\nbreak','2'],['quote"comma,','3']],rows
print('PASS: gzip input, Unicode, quoted CSV, generator failure, atomic replacement, API compatibility')
'''

TASK='''Extend this event-name counter for production data files, preserving the documented APIs and existing behavior.
Requirements:
1. reader.records supports both plain UTF-8 text and .gz files, remains streaming and skips blank lines.
2. writer.write_rows emits valid CSV for commas, quotes, Unicode and embedded newlines.
3. Destination replacement is atomic: if row generation fails, retain the previous destination content,
   propagate the error and remove temporary output. Preserve the header and plain-data output format.
4. Keep the CLI working end to end. Add targeted regression tests and run the registered unit check.
Use two independent implement branches: reader.py input handling, and writer.py CSV/atomic output.
Let those agents do the edits and relevant tests in isolated copies. Inspect/adopt their patches in the
parent, integrate if needed and rerun parent checks. Do not implement both modules directly in the parent.
'''

async def evaluate(args):
    from src.coding.harness import CodingHarness
    from src.coding.control import SessionControl
    from src.config.config_paths import load_base_config
    from src.runtime import hitl,run_context
    root=Path(args.work_dir)/('event-counter-'+uuid.uuid4().hex[:8]);root.mkdir(parents=True)
    for name,text in FILES.items(): (root/name).write_text(text)
    config=dict(load_base_config().get('coding',{}))
    config.update(backend='openrouter',model='google/gemini-3.1-flash-lite',max_turns=40,
                  checks={'unit':[sys.executable,'-B','-m','unittest','-q']},allow_web=False,learn=False,
                  max_cost_usd=.10,max_total_tokens=180000)
    config['orchestration']=dict(config['orchestration'])
    config['orchestration'].update(local_slots=2,mind_turns=18,isolated_writes=True)
    control=SessionControl();agent=CodingHarness(root,config=config)
    started=time.monotonic()
    steering=[]
    with run_context.request_scope('development-scenario','development-scenario'):
        hitl.set_work_mode('auto')
        task=asyncio.create_task(agent.run(TASK,reasoning='low',control=control))
        # Exercise live delivery after real worker work begins; do not inject a solution.
        delivered=False
        while not task.done():
            supervisor=getattr(control,'supervisor',None)
            if supervisor and not delivered:
                child=next((a for a in supervisor.agents.values() if a['id'].startswith('mind-') and a['actions']),None)
                if child:
                    steering.append(control.post('Also preserve Unicode names and the original public function signatures.',interrupt=False))
                    delivered=True
            await asyncio.sleep(.1)
        try: result=await task
        except Exception as exc: result={'status':'error','error':type(exc).__name__+': '+str(exc),**agent.model.usage()}
    # The acceptance file is deliberately outside the agent's project and tools.
    acceptance=root.parent/(root.name+'-acceptance.py');acceptance.write_text(ACCEPTANCE)
    verified=subprocess.run([sys.executable,str(acceptance),str(root.resolve())],capture_output=True,text=True,timeout=20)
    agent_code=Path(__file__).resolve().parents[1]/'src'/'coding'
    report={'scenario':'multi_file_development_not_benchmark','workspace':str(root),
            'harness_id':hashlib.sha256(b''.join(p.read_bytes() for p in sorted(agent_code.glob('*.py')))).hexdigest()[:16],
            'seconds':round(time.monotonic()-started,2),'independent_passed':verified.returncode==0,
            'acceptance_output':(verified.stdout+verified.stderr)[-3000:],'steering':steering,'result':result,
            'proposals': [agent.control.orchestrator.branches.view(p) for p in agent.control.orchestrator.branches.proposals.values()]}
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in {'result','proposals'}},ensure_ascii=False),flush=True)
    print(json.dumps({'status':result['status'],'cost':result.get('cost'),'worker_tokens':result.get('tokens'),
                      'proposals':[(p['id'],p['status'],p['files']) for p in report['proposals']],
                      'error':result.get('error'),'answer':result.get('final_answer')},ensure_ascii=False),flush=True)
    return 0 if verified.returncode==0 and result['status']=='done' else 1

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live',action='store_true')
    parser.add_argument('--work-dir',default='data/eval/development')
    parser.add_argument('--output',default='data/eval/development-scenario.json')
    args=parser.parse_args()
    if not args.live: print('No API calls. Use --live for the bounded cheap-cloud integration.');return 0
    return asyncio.run(evaluate(args))

if __name__=='__main__': raise SystemExit(main())
