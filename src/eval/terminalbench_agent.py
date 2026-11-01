"""Harbor external-agent adapter: same SEA loop, shell tools in the task container.

Host filesystem tools, web retrieval, cross-task learning and delegates are disabled.
Harbor's verifier, rather than model self-assessment, determines the task reward.
Optional dependency: run through `uv tool run --with omegaconf --from harbor harbor`.
"""
from __future__ import annotations

import json
import base64
import shlex
from pathlib import Path

from harbor.agents.base import BaseAgent

from src.coding.harness import CodingHarness, Turn, Workspace, SYSTEM
from src.config.config_paths import load_base_config
from src.runtime import hitl, run_context


class TerminalTurn(Turn):
    @classmethod
    def model_json_schema(cls, **kwargs):
        schema = super().model_json_schema(**kwargs)
        for branch in schema['oneOf']:
            if branch['properties']['action']['const'] == 'tool':
                branch['properties']['tool']['enum'] = ['run','check','register_check','read','write','edit','tree','search','grep','evidence','think']
        return schema


class ContainerWorkspace(Workspace):
    # Never execute host experiments against this remote workspace's local metadata directory.
    verifier_tools = frozenset({'read', 'search', 'tree', 'evidence'})
    def __init__(self, root, directory, checks, environment):
        super().__init__(root, directory, {})
        self.environment = environment
        self.hashes = {}
        self.remote_root='/app'

    def file_digest(self,name):
        return self.hashes.get(str(name))

    async def file_tool(self,name,args):
        data=base64.b64encode(json.dumps([name,args,self.remote_root]).encode()).decode()
        result=await self.environment.exec(command=shlex.join(['python3','-I','-c',REMOTE_FILES,data]),timeout_sec=30)
        if result.return_code!=0:
            return {'ok':False,'output':(result.stderr or result.stdout or 'File tool failed')[-2500:]}
        return json.loads(result.stdout)

    async def source_text(self,name,limit=4000):
        result=await self.file_tool('read',{'path':name,'start':1,'end':200})
        if not result['ok']: raise ValueError(result['output'])
        if result.get('source_hash'): self.hashes[str(name)] = result['source_hash']
        return result['output'][:limit]

    async def file_exists(self,name):
        result=await self.file_tool('exists',{'path':name})
        return result.get('ok') and result.get('exists',False)

    async def execute(self, name, args, readonly=False):
        if name=='edit' and 'changes' in args:
            from src.coding.edits import execute_batch
            return await execute_batch(self,args,readonly=readonly)
        if name in {'evidence','register_check','think'}:
            return await super().execute(name,args,readonly=readonly)
        if name in {'tree','read','search','grep','write','edit'}:
            if readonly and name in {'write','edit'}:
                return {'ok':False,'output':'Read-only branch'}
            result=await self.file_tool(name,args)
            if result['ok']:
                if name in {'write','edit'}:
                    self.revision+=1;self.passed.clear();self.changed.add(str(args['path']))
                if result.get('source_hash'):
                    self.hashes[str(args['path'])]=result['source_hash']
            return result
        if name not in {'run','check'} or readonly:
            return {'ok':False,'output':'Tool unavailable in this container profile'}
        argv=self.checks.get(str(args.get('name'))) if name=='check' else args.get('argv')
        if not isinstance(argv,list) or not argv or not all(isinstance(s,str) for s in argv):
            return {'ok':False,'output':'Use a nonempty argv list or a registered check name: '+', '.join(self.checks)}
        if name=='run':
            self.revision+=1;self.passed.clear()
        self.hashes.clear()  # Commands may modify source; refresh prior inspected content.
        try:
            result=await self.environment.exec(command=shlex.join(argv),timeout_sec=120)
        except (RuntimeError,TimeoutError) as exc:
            return {'ok':False,'output':str(exc)[:1500],'command':shlex.join(argv)}
        ok=result.return_code==0
        if name=='check':
            self.last_check_revision=self.revision
            if ok: self.verified_revision=self.revision;self.passed.add(str(args['name']))
            else: self.verified_revision=-1;self.passed.discard(str(args['name']))
        output=((result.stdout or '')+('\n[stderr]\n'+result.stderr if result.stderr else ''))[-20000:]
        if not output:
            output='Command completed; exit code '+str(result.return_code)+'. Do not repeat unchanged commands.\n'+shlex.join(argv)[:2500]
        return {'ok':ok,'exit_code':result.return_code,'command':shlex.join(argv),'output':output}


REMOTE_FILES = Path(__file__).with_name('remote_files.py').read_text()

class SeaAgent(BaseAgent):
    def __init__(self, *args, local_base_url='http://127.0.0.1:8081/v1', backend='openrouter', max_turns=48,
                 reasoning='auto', **kwargs):
        super().__init__(*args, **kwargs)
        self.local_base_url, self.max_turns, self.reasoning = local_base_url, int(max_turns), reasoning
        if backend not in {'openrouter','llama_cpp'}:
            raise ValueError('Use explicit openrouter or llama_cpp backend')
        self.backend = backend

    @staticmethod
    def name(): return 'sea-local'

    def version(self): return '0.1.0-local-harness'

    async def setup(self, environment):
        pass  # The model stays on the host; all tool execution stays in Harbor's container.

    async def run(self, instruction, environment, context):
        config = dict(load_base_config().get('coding', {}))
        config.update(backend=self.backend, base_url=self.local_base_url,
                      model=(self.model_name or ('sea-local' if self.backend == 'llama_cpp' else 'google/gemini-3.1-flash-lite')).removeprefix('openai/'),
                      max_turns=self.max_turns, max_delegates=2, checks={},
                      allow_web=False, learn=False, supervisor_review=False,
                      orchestration={'enabled':True,'isolated_writes':False,'observer':True,
                                     'decision':dict(config.get('orchestration',{}).get('decision',{})),
                                     'critic':dict(config.get('orchestration',{}).get('critic',{}))}, terminal_sandbox=True)
        root = Path(self.logs_dir)/'sea'
        location=await environment.exec(command='pwd',timeout_sec=10)
        remote_root=(location.stdout or '').strip()
        if not remote_root.startswith('/') or remote_root=='/':
            location=await environment.exec(command='if [ -d /testbed ]; then echo /testbed; else echo /app; fi',timeout_sec=10)
            remote_root=(location.stdout or '/app').strip()
        def factory(root,directory,checks):
            ws=ContainerWorkspace(root,directory,checks,environment)
            ws.remote_root=remote_root
            return ws
        system = """All task source and command tools execute in the task's isolated Linux container.
Use read/write/edit/tree/search for files under the current task working directory; run takes argv and executes a container command.
No host filesystem access. Implement the user's task, create your OWN relevant behavioral checks, register
those commands with register_check, then run check. Current checks come from the task project or you,
not the external grader. Inspect error paths and cleanup requirements as well as the happy path.
Do not search for benchmark solutions, oracle scripts, hidden verifier tests or grading rules. External
Harbor verification happens only after you finish and is the independent task score. Do not invent success.
"""
        system+='\nTask workspace: '+remote_root+'\n'
        agent = CodingHarness(root, config=config, workspace_factory=factory,
                              turn_schema=TerminalTurn, system_prompt=system)
        with run_context.request_scope('harbor-'+str(self.context_id or 'trial'), 'harbor'):
            hitl.set_work_mode('auto')
            result = await agent.run(instruction, reasoning=self.reasoning, session_id='trial')
        usage = result
        context.n_input_tokens = usage['tokens_in']
        context.n_output_tokens = usage['tokens_out']
        context.n_cache_tokens = usage.get('cached_tokens', 0)
        context.cost_usd = result.get('cost',0)
        context.metadata = {'harness_status':result['status'], 'model_file':usage.get('model_file'),
                            'verification':'Harbor external verifier', 'learning':False, 'web':False,
                            'calls':usage['calls'], 'decision_usage':result.get('decision_usage',{}),
                            'review_usage':result.get('review_usage',{})}
        (root/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
