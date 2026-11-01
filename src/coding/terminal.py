"""Interactive developer console: commands remain available while the agent works."""
from __future__ import annotations

import asyncio
import json
import shlex
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout

from .control import SessionControl
from .harness import CodingHarness, Workspace

HELP = '''Text starts a task; text while busy steers the current task.
/diff [--stat] [path]  /grep [--regex] pattern [path]  /read path [start end]
/status  /context  /agents  /cache  /steer AGENT message
/reasoning auto|low|medium|high  /plan  /pause  /resume  /help  /exit
Commands do not run the LLM. /plan never grants write permission.'''


class DeveloperConsole:
    def __init__(self, root, config, *, auto=False, plan=False, reasoning='auto', session='cli'):
        self.root, self.config = root.resolve(), config
        self.auto, self.plan, self.reasoning, self.session = auto, plan, reasoning, session
        self.control = SessionControl()
        self.active = None
        self.harness = None
        self.confirmation = None
        self.clarification = None
        self.reader = PromptSession()
        self.directory = self.root/'.sea'/'console'
        self.directory.mkdir(parents=True, exist_ok=True)
        self.workspace = Workspace(self.root, self.directory, config.get('checks'))

    async def confirm(self, text):
        self.confirmation = asyncio.get_running_loop().create_future()
        print('\nТребуется подтверждение:', text, '\nВведите да/нет (или команду /pause).')
        try:
            return await self.confirmation
        finally:
            self.confirmation = None

    async def clarify(self, items):
        self.clarification=asyncio.get_running_loop().create_future()
        item=items[0]
        print('\n'+item['question']+'\n'+item.get('why',''))
        if item.get('options'): print(' / '.join(item['options']))
        try: return await self.clarification
        finally: self.clarification=None

    def progress(self, event):
        phase = event['phase']
        if phase != self.control.snapshot.get('_printed_phase'):
            print('[' + event['message'] + ']')
            self.control.snapshot['_printed_phase'] = phase

    async def start(self, task='', resume=False):
        from src.runtime import hitl, run_context
        self.control = SessionControl()
        self.harness = CodingHarness(self.root, config=self.config)
        async def work():
            try:
                with run_context.request_scope('coding-terminal', 'coding-terminal'):
                    hitl.set_work_mode('plan' if self.plan else 'auto' if self.auto else 'manual')
                    hitl.set_confirmer(self.confirm)
                    from src.interface import clarify
                    clarify.set_clarifier(self.clarify)
                    result = await self.harness.run(task, reasoning=self.reasoning,
                        execution='plan' if self.plan else 'auto', session_id=self.session,
                        resume=resume, control=self.control, progress=self.progress)
                print('\n' + result['final_answer'])
            except asyncio.CancelledError:
                print('\nОстановлено; контрольная точка сохранена.')
            except Exception as exc:
                print('\nОшибка:', exc)
        self.active = asyncio.create_task(work())

    async def line(self, text):
        text = text.strip()
        if not text:
            return True
        running = self.active is not None and not self.active.done()
        if not text.startswith('/'):
            if self.clarification and not self.clarification.done():
                self.clarification.set_result([text])
            elif self.confirmation and not self.confirmation.done():
                self.confirmation.set_result(text.casefold() in {'да','yes','y'})
            elif running:
                print(self.control.post(text))
            else:
                await self.start(text,resume=bool(self.control.snapshot.get('human_action')))
            return True
        args = shlex.split(text)
        command, args = args[0], args[1:]
        if command == '/exit':
            if running:
                self.active.cancel()
                await asyncio.gather(self.active, return_exceptions=True)
            return False
        if command == '/help': print(HELP)
        elif command == '/pause':
            self.control.pause = True
            self.control.post('Pause at the next safe boundary.')
            if self.confirmation and not self.confirmation.done(): self.confirmation.set_result(False)
            if self.clarification and not self.clarification.done(): self.clarification.set_result([])
        elif command == '/resume':
            if running: print('Текущий прогон ещё активен.')
            else: await self.start(resume=True)
        elif command == '/plan':
            self.plan = True
            self.control.execution = 'plan'
            if running: self.control.post('Continue in read-only planning mode.')
        elif command == '/reasoning':
            if not args or args[0] not in {'auto','low','medium','high'}:
                print('Usage: /reasoning auto|low|medium|high')
            else:
                self.reasoning = self.control.reasoning = args[0]
                if running: self.control.post('Reasoning level changed by the operator.')
        elif command == '/steer':
            if len(args) < 2: print('Usage: /steer main|worker-1 message')
            else: print(self.control.post(' '.join(args[1:]), target=args[0]))
        elif command in {'/status','/context','/agents','/cache'}:
            supervisor = getattr(self.control, 'supervisor', None)
            if command == '/agents': value = supervisor.view() if supervisor else {}
            elif command == '/cache': value = self.harness.model.usage() if self.harness and hasattr(self.harness.model,'usage') else {}
            elif command == '/context': value = self.control.snapshot.get('context', {})
            else: value = {'running':running,'thinking':self.control.thinking,'phase':self.control.snapshot.get('phase'),'plan':self.control.snapshot.get('plan')}
            print(json.dumps(value, ensure_ascii=False, indent=2))
        elif command in {'/diff','/grep','/read'}:
            if command == '/diff':
                params = {'stat':'--stat' in args}
                paths = [a for a in args if a != '--stat']
                if paths: params['path'] = paths[0]
                tool = 'diff'
            elif command == '/grep':
                regex = '--regex' in args; values = [a for a in args if a != '--regex']
                if not values: print('Usage: /grep [--regex] pattern [path]'); return True
                tool, params = 'grep', {'pattern':values[0], 'path':values[1] if len(values)>1 else '.', 'regex':regex}
            else:
                if not args: print('Usage: /read path [start end]'); return True
                tool, params = 'read', {'path':args[0], 'start':int(args[1]) if len(args)>1 else 1, 'end':int(args[2]) if len(args)>2 else 120}
            result = await self.workspace.execute(tool, params, readonly=True)
            print(result['output'])
        else: print('Unknown command. /help')
        return True

    async def run(self, task=''):
        print(HELP)
        if task: await self.start(task)
        with patch_stdout():
            while True:
                try:
                    line = await self.reader.prompt_async('sea code> ')
                    if not await self.line(line): break
                except (EOFError, KeyboardInterrupt):
                    await self.line('/exit'); break
                except (ValueError, OSError) as exc:
                    print('Ошибка команды:', exc)
