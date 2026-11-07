"""Real SEA actor/tool/verification loop for case 600; no direct Codex file tools."""
import asyncio
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from codex_behavior_eval import CodexInference
from src.coding.harness import CodingHarness
from src.runtime import hitl,run_context
from src.interface import clarify


async def main(args):
    output=ROOT/'data/eval/campaign600-20261107'/('product' if args.attempt==1 else f'product-attempt-{args.attempt}')
    output.mkdir(parents=True,exist_ok=False)
    project=ROOT/'artifacts/flow-video'
    project.mkdir(parents=True,exist_ok=True)
    brief=(ROOT/'experiments/campaign600/product_brief.md').read_text()
    feedback=args.feedback.read_text() if args.feedback else ''
    (project/'PRODUCT_BRIEF.md').write_text(brief)
    shared={'calls':0,'tokens':0}
    limit=(args.token_limit if args.token_limit is not None else
           400000 if args.attempt>=4 else 1600000 if args.attempt>=3 else 2700000 if args.resume else 4000000)
    model=CodexInference(output/'calls',shared,max_calls=100,shared_call_limit=100,
                         shared_token_limit=limit,timeout_seconds=180,output_bound=20000,payload_object=True)
    model.config['max_total_tokens']=limit
    config={'planning':'adaptive','max_turns':90,'max_total_tokens':limit,
            'learn':False,'allow_web':False,'adaptive_tools':True,'context_policy':args.context_policy,
            'completion_review':args.completion_review,
            'source_cache_chars':args.source_cache_chars,'source_window_bytes':args.source_window_bytes,
            'checks':{'unit':[sys.executable,'-m','pytest','-q','-p','no:cacheprovider']},
            'orchestration':{'enabled':True,'auto_domains':True,'same_model':True,'max_nodes':1,
                'decision':{'max_calls':0},'critic':{'enabled':True,'active':False,'max_calls':8}}}
    config['orchestration']['critic']['source_context_chars']=args.review_source_chars
    config['browser_runtime']=args.browser_runtime
    config['result_memory']=True
    (output/'protocol.json').write_text(json.dumps({'case_id':'project-600-video-service','brief':brief,
        'config':config,'scope':'SEA CodingHarness + Codex CLI inference only',
        'limits':{'calls':100,'tokens':limit},'resume':args.resume,'external_acceptance_pending':True,
        'external_feedback':feedback,
        'source_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
            ['src/coding/harness.py','src/coding/critic.py','src/coding/progress.py','src/coding/review_context.py','src/coding/completion.py','src/coding/followthrough.py','src/coding/containment.py']},
        'adapter_sha256':hashlib.sha256((ROOT/'scripts/codex_behavior_eval.py').read_bytes()).hexdigest()},ensure_ascii=False,indent=2))
    started=time.monotonic()
    try:
        with run_context.request_scope('campaign600-product','evaluation'):
            hitl.set_work_mode('auto');clarify.set_clarifier(None);clarify.reset_ledger()
            hitl.set_confirmer(lambda description:description.startswith(
                'Additional command authority: project_write=True, network=False:'))
            harness=CodingHarness(project,model=model,config=config)
            from src.coding.progress import ProgressWatchdog
            watchdog=ProgressWatchdog(16)
            def progress(event):
                if watchdog.observe(event['progress_state']):
                    harness.control.pause=True
                    print(json.dumps({'watchdog':'stagnation','calls_without_new_evidence_or_change':watchdog.stalled}),flush=True)
            result=await harness.run(feedback if args.resume and feedback else brief,
                reasoning='medium',session_id='video-product',resume=args.resume,progress=progress)
        (output/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
        print(json.dumps({'status':result['status'],'usage':shared,'seconds':round(time.monotonic()-started),
                          'completion':result.get('completion'),'project':str(project)},ensure_ascii=False))
    except BaseException as exc:
        (output/'incomplete.json').write_text(json.dumps({'error':type(exc).__name__+': '+str(exc),
            'usage':shared,'seconds':round(time.monotonic()-started)},ensure_ascii=False,indent=2))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--attempt',type=int,default=1)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--context-policy',choices=['single_action','bounded'],default='bounded')
    parser.add_argument('--token-limit',type=int)
    parser.add_argument('--completion-review',action='store_true')
    parser.add_argument('--feedback',type=Path)
    parser.add_argument('--source-cache-chars',type=int,default=16000)
    parser.add_argument('--source-window-bytes',type=int,default=24000)
    parser.add_argument('--review-source-chars',type=int,default=24000)
    parser.add_argument('--browser-runtime',action='store_true',help='Allow read-only installed Playwright Chromium assets; no profiles, network or extra writes')
    asyncio.run(main(parser.parse_args()))
