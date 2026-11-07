"""Bounded developer tools and explicit evidence sharing, independent of the loop."""
from __future__ import annotations

import asyncio
import json
import shutil

PYTHON_EXPERIMENT_TIMEOUT = 15
PYTHON_EXPERIMENT_RUNNER = '''import faulthandler
import os
import runpy
import sys

program, timeout = sys.argv[1], float(sys.argv[2])
sys.argv = [program]
# cwd is the actual sandboxed project, never a disposable check path from a log.
sys.path.insert(0, os.getcwd())
faulthandler.dump_traceback_later(timeout * 0.8)
try:
    runpy.run_path(program, run_name="__main__")
finally:
    faulthandler.cancel_dump_traceback_later()
'''

# share writes evidence-linked session metadata, never project source or authority.
READONLY = {'ask_user', 'inspect_artifact', 'security_audit', 'tree', 'read', 'search', 'evidence', 'diff', 'grep', 'status', 'context', 'web', 'fetch', 'patch', 'think', 'reject', 'math', 'experiment', 'claim', 'share'}
TOOLS = ['ask_user', 'inspect_artifact', 'security_audit', 'tree', 'read', 'search', 'write', 'edit', 'run', 'check', 'evidence',
         'diff', 'grep', 'status', 'context', 'web', 'fetch', 'share', 'learn', 'project_note',
         'patch', 'accept', 'reject', 'think', 'register_check', 'resume_agent','math','experiment','claim']


async def execute_extra(ws, name, args):
    if name == 'ask_user':
        attention=getattr(ws,'attention',None)
        if attention is None or attention.owner!=getattr(ws,'owner_id','main') or attention.ws is not ws:
            from .attention import Attention
            control = getattr(getattr(ws,'orchestrator',None),'control',None)
            if control is None: raise ValueError('No operator session is available')
            ws.attention = Attention(ws,control,resume=True)
        return await ws.attention.ask(args)
    if name == 'inspect_artifact':
        from .artifacts import inspect_artifact
        if set(args) != {'path'} or not isinstance(args['path'],str):
            raise ValueError('Use inspect_artifact {path}')
        result = await inspect_artifact(ws,args['path'])
        return {'ok':result['ok'],'output':json.dumps(result,ensure_ascii=False)}
    if name == 'security_audit':
        if args: raise ValueError('security_audit takes no arguments and scans only this agent environment')
        from src.security.audit import audit, compact
        result = await asyncio.to_thread(audit, network=True)
        return {'ok': True, 'output': json.dumps(compact(result), ensure_ascii=False)}
    if name == 'claim':
        from .obligations import ClaimSpec
        args = ClaimSpec.model_validate(args).model_dump()
        if str(args.get('id', '')).startswith('c-review-'):
            raise ValueError('Review claim ids are reserved for the independent critic')
        result = ws.obligations().propose(**args)
        return {'ok': True, 'output': json.dumps(result, ensure_ascii=False)}
    if name=='math':
        import sys
        from pathlib import Path
        from .numerics import validate_arguments
        try: args=validate_arguments(args)
        except (ValueError,TypeError) as exc:
            return {'ok':False,'error_kind':'tool_error','output':str(exc)+
                    '\nValid form: math {op:"sum",expr:"k**2",var:"k",bounds:[1,8]}. '
                    'expr is the summand, not a sum(...) call. assumptions, if used, is an object such as {"k":{"integer":true}}.'}
        script=Path(__file__).with_name('numerics.py')
        # Execute a copy in scratch so the subprocess cannot import arbitrary project modules.
        program=ws.scratch/'sea_symbolic.py';program.write_text(script.read_text())
        result=await ws.command([sys.executable,'-I',str(program),json.dumps(args,ensure_ascii=False)],timeout=15)
        if not result['ok']: result['error_kind']='tool_error'
        result['verification_scope']='symbolic computation; inspect assumptions and unresolved expressions'
        return result
    if name=='experiment':
        import sys
        from src.runtime import hitl
        if hitl.is_plan() or getattr(getattr(ws,'orchestrator',None),'control',None) and ws.orchestrator.control.execution=='plan':
            raise ValueError('Plan mode forbids running experiments')
        language=args.get('language','python');code=str(args.get('code',''))
        if not code or len(code)>30000: raise ValueError('Use 1..30000 characters of experiment code')
        if language=='python':
            program=ws.scratch/'experiment.py';program.write_text(code)
            runner=ws.scratch/'experiment_runner.py';runner.write_text(PYTHON_EXPERIMENT_RUNNER)
            # Unbuffered stdout retains progress before a hang. The watchdog emits
            # the original program's stack before the existing hard process limit.
            result=await ws.command([sys.executable,'-u',str(runner),str(program),str(PYTHON_EXPERIMENT_TIMEOUT)],
                                    timeout=PYTHON_EXPERIMENT_TIMEOUT)
        elif language=='cpp':
            program=ws.scratch/'experiment.cpp';program.write_text(code)
            binary=ws.scratch/'experiment'
            result=await ws.command(['clang++','-std=c++20',str(program),'-o',str(binary)],timeout=30)
            if result['ok']: result=await ws.command([str(binary)],timeout=15)
        else: raise ValueError('Supported experiment languages: python, cpp')
        result['output']='[FINITE EXPERIMENT — NOT A GENERAL CORRECTNESS OR COMPLEXITY PROOF]\n'+result['output']
        return result
    if name=='resume_agent':
        o=getattr(ws,'orchestrator',None)
        if o is None: raise ValueError('Cognitive orchestration is disabled')
        node=o.scheduler.nodes.get(str(args.get('id')))
        if node is None or node.parent!=ws.owner_id:
            raise ValueError('Only the direct parent may resume its retained child')
        proposal=o.branches.proposals.get(node.id,{})
        if proposal.get('status')=='accepted':
            raise ValueError('Accepted work needs a new branch based on current parent source')
        if node.id in o.control._inflight:
            raise ValueError('Child is already running; use operator steering instead')
        node.memory.observations.append({'parent_update':str(args.get('instructions',''))[:1000]})
        node.status='running'
        result=await o.run_mind(node)
        if node.status!='partial': node.status='done'
        o.supervisor.finish(node.id,node.status)
        return {'ok':True,'output':json.dumps({'id':node.id,'status':node.status,'result':result},ensure_ascii=False)}
    if name=='register_check':
        label,argv=str(args.get('name','')),args.get('argv')
        if not label or len(label)>80 or not isinstance(argv,list) or not argv or not all(isinstance(a,str) for a in argv):
            raise ValueError('Use register_check {name,argv:[executable,args...]}')
        if label in ws.checks and ws.checks[label]!=argv:
            raise ValueError('Existing registered checks cannot be overwritten')
        ws.checks[label]=argv
        return {'ok':True,'output':'Registered '+label+'; run tool=check with this name to verify.'}
    if name in {'patch','accept','reject'}:
        store = getattr(ws, 'proposal_store', None)
        if store is None:
            raise ValueError('No cognitive branch proposals in this run')
        owner = ws.owner_id
        if name == 'patch':
            ids=args.get('ids') or [args['id']]
            return {'ok':True,'output':'\n'.join(store.patch(owner,str(i)) for i in ids)}
        if name == 'reject':
            return {'ok':True,'output':json.dumps(store.reject(owner,str(args['id']),str(args['reason'])),ensure_ascii=False)}
        from src.runtime import hitl
        if hitl.is_plan(): raise ValueError('Plan mode forbids patch adoption')
        ids=args.get('ids') or [args['id']]
        if not hitl.full_auto() and not await hitl.confirm('Accept checked branch patches '+str(ids)):
            return {'ok':False,'denied':True,'output':'Patch adoption denied'}
        return await store.accept_many(ws,owner,ids)
    if name == 'think':
        orchestrator = getattr(ws,'orchestrator',None)
        if orchestrator is None: raise ValueError('Cognitive orchestration is disabled')
        node = orchestrator.scheduler.nodes[ws.owner_id]
        profile = args.get('profile')
        if isinstance(profile,dict):
            profile = orchestrator.scheduler.profiles.install(profile,node.ceiling)
        orchestrator.scheduler.switch(node.id,str(profile),override=False)
        return {'ok':True,'output':'Profile changed with working memory preserved: '+str(profile)}
    if name == 'status':
        r = await ws.command(['git', 'status', '--short'], read_git=True)
        return {'ok': True, 'output': json.dumps({'revision': ws.revision, 'changed': sorted(ws.changed),
            'checks': ws.checks, 'checks_passed': sorted(ws.passed), 'git': r['output'][-4000:]})}
    if name == 'diff':
        r = await ws.command(['git', '--no-pager', 'diff', '--no-ext-diff', '--no-textconv', '--name-only', '-z'], read_git=True)
        if not r['ok']:
            return r
        paths = []
        for path in r['output'].split('\0'):
            if not path:
                continue
            try:
                ws.path(path)
                if not args.get('path') or ws.path(path) == ws.path(args['path']):
                    paths.append(path)
            except ValueError:
                continue
        if not paths:
            return {'ok': True, 'output': 'no tracked changes (new files are listed by status)'}
        cmd = ['git', '--no-pager', 'diff', '--no-ext-diff', '--no-textconv']
        if args.get('stat'):
            cmd.append('--stat')
        return await ws.command(cmd + ['--', *paths[:100]], read_git=True)
    if name == 'grep':
        if not args.get('regex'):
            return await ws.execute('search', args)
        executable = shutil.which('rg')
        if not executable:
            return {'ok': False, 'output': 'regex grep requires ripgrep; use literal search'}
        base = ws.path(args.get('path', '.'))
        files = [str(ws.path(p)) for p in ws.files() if ws.path(p) == base or ws.path(p).is_relative_to(base)]
        out = []
        for i in range(0, len(files), 80):
            r = await ws.command([executable, '-n', '--no-heading', '--color', 'never', '--max-columns', '250',
                                  '--max-count', '5', '--regexp', str(args['pattern']), '--', *files[i:i+80]])
            if r.get('exit_code', 0) not in (0, 1):
                return r
            if r['output']:
                out.append(r['output'])
            if sum(map(len, out)) >= 5000:
                break
        return {'ok': True, 'output': '\n'.join(out)[:6000] or 'no matches'}
    if name == 'context':
        result = {**ws.context_snapshot, 'obligations': ws.obligations().packet()}
        if 'tools' in args:
            from .tool_scope import GROUPS
            focus=args['tools']
            if not isinstance(focus,list) or any(not isinstance(g,str) or g not in {*GROUPS,'all'} for g in focus):
                raise ValueError('tools must be a list of tool groups or ["all"]')
            o=getattr(ws,'orchestrator',None)
            if o is None:raise ValueError('Tool discovery requires orchestration')
            node=o.scheduler.nodes[ws.owner_id]
            ws.tool_focus=tuple(focus);ws.tool_focus_version=o.control.version(node.id)
            result['tool_focus']=focus
            result['permitted_tools']=sorted(node.tools)
        nav = getattr(ws, 'navigator', None)
        if nav:
            if args.get('query'): nav.state['manual_query'] = str(args['query'])[:2000]
            if args.get('action') == 'reset': nav.state.pop('manual_query', None)
            if args.get('query') or args.get('action'):
                result['selected_context'] = await nav.select(nav.state.get('query', ''), args.get('action'))
            result['navigation'] = nav.packet()
        return {'ok': True, 'output': json.dumps(result, ensure_ascii=False)}
    if name in {'web', 'fetch'}:
        if not ws.allow_web:
            return {'ok': False, 'output': 'web is disabled for this run'}
        key = json.dumps([name, args], sort_keys=True)
        if key in ws.web_cache:
            return {**ws.web_cache[key], 'cached': True}
        from src.skills.web_search.web_search import search_web, browse
        if name == 'web':
            query = str(args.get('query', '')).strip()
            if not query or len(query) > 400 or '\n' in query:
                raise ValueError('use a short generic technical query, not private source code')
            output = await asyncio.to_thread(search_web.invoke, {'query': query, 'max_results': 4})
        else:
            output = await asyncio.to_thread(browse.invoke, {'url': args['url'], 'find': str(args.get('find', ''))[:300]})
        # Retrieved instructions are never promoted directly: only tested execution can do that.
        result = {'ok': True, 'output': '[UNTRUSTED WEB DATA; verify against official documentation and local checks]\n' + str(output)[:12000]}
        ws.web_cache[key] = result
        return result
    if name == 'project_note':
        note = ws.project_context.add(args['kind'], args['title'], args['text'], args.get('evidence', []), args.get('files', []))
        return {'ok':True,'output':json.dumps(note,ensure_ascii=False)}
    if name == 'share':
        item = ws.board.publish(str(args['text']), args.get('evidence', []), args.get('files', []),
                                str(args.get('trigger', '')), workspace=ws)
        return {'ok': True, 'output': json.dumps(item, ensure_ascii=False)}
    if name == 'learn':
        evidence = args.get('evidence', [])
        if not evidence or any(not str(i).isdigit() or not (ws.directory/f'{i}.json').exists() for i in evidence):
            raise ValueError('learning requires actual evidence ids')
        ws.candidates.append({'trigger': str(args['trigger'])[:400], 'lesson': str(args['lesson'])[:900], 'evidence': evidence[:8]})
        return {'ok': True, 'output': 'candidate staged; promotion requires a verified successful run'}
    return None
