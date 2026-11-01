"""Human steering with cancellation only at model-generation boundaries."""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections import deque
from pathlib import Path


class Steered(Exception):
    pass


class SessionControl:
    def __init__(self):
        self.pending = deque()
        self.history = []
        self.revision = 0
        self.interrupt = asyncio.Event()
        self.thinking = False
        self.pause = False
        self.cancelled_agents = set()
        self.reasoning = None
        self.execution = None
        self.active_agent = 'main'
        self._inflight = {}
        self.snapshot = {}
        self.path: Path | None = None

    def bind(self, path: Path, resume=False):
        self.path = path
        previous = [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if resume and path.exists() else []
        if not resume:
            path.write_text('')
        queued = list(self.history)
        self.history = previous + queued
        self.revision += len(previous)
        if queued:
            with path.open('a', encoding='utf-8') as f:
                for entry in queued: f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    def post(self, text: str, interrupt: bool = True, target='main', origin='user') -> dict:
        text = text.strip()
        if not text or len(text) > 8000:
            raise ValueError('steering must contain 1..8000 characters')
        self.revision += 1
        entry = {'id': uuid.uuid4().hex, 'text': text, 'ts': time.time(), 'target': target, 'origin': origin}
        self.pending.append(entry)
        self.history.append(entry)
        if self.path:
            with self.path.open('a', encoding='utf-8') as f:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        interrupted = self.interrupt_agent(target) if interrupt else False
        return {'accepted': True, 'id': entry['id'], 'interrupted_generation': interrupted}

    def drain(self, target='main'):
        entries = [e for e in self.pending if e['target'] in {target, '*'}]
        self.pending = deque(e for e in self.pending if e not in entries)
        self.interrupt.clear()
        return entries

    def interrupt_agent(self, target):
        interrupted = False
        supervisor = getattr(self, 'supervisor', None)
        def within(agent):
            if target in {'main', '*'}: return True
            seen = set()
            while agent and agent not in seen:
                if agent == target: return True
                seen.add(agent)
                agent = supervisor.agents.get(agent, {}).get('parent', '') if supervisor else ''
            return False
        for agent, (_, event) in self._inflight.items():
            if within(agent): event.set(); interrupted = True
        return interrupted

    def is_cancelled(self, agent):
        return any(a in self.cancelled_agents for a in self.ancestors(agent))

    def ancestors(self, agent):
        supervisor = getattr(self, 'supervisor', None)
        seen = set()
        while agent and agent not in seen:
            seen.add(agent)
            agent = supervisor.agents.get(agent, {}).get('parent', '') if supervisor else ''
        return seen | {'main', '*'}

    def instructions(self, agent='main'):
        targets = self.ancestors(agent)
        return [entry for entry in self.history if entry['target'] in targets]

    def version(self, agent='main'):
        return len(self.instructions(agent))

    async def call(self, factory, agent_id='main'):
        self.thinking = True
        self.active_agent = agent_id
        event = asyncio.Event()
        task = asyncio.create_task(factory())
        self._inflight[agent_id] = (task,event)
        steering = asyncio.create_task(event.wait())
        try:
            done, _ = await asyncio.wait({task, steering}, return_when=asyncio.FIRST_COMPLETED)
            if steering in done:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                raise Steered()
            return await task
        finally:
            self._inflight.pop(agent_id,None)
            self.thinking = bool(self._inflight)
            for pending in (task, steering):
                if not pending.done(): pending.cancel()
            await asyncio.gather(task, steering, return_exceptions=True)
