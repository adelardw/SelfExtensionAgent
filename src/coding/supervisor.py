"""Event-driven operator view. Observation is free; model review runs only on anomalies."""
from __future__ import annotations

import contextvars
import asyncio
import json
import time
from collections import deque
from contextlib import contextmanager
from typing import Literal

from pydantic import BaseModel
from .control import Steered

_active = contextvars.ContextVar('coding_agent_scope', default='main')


class Advice(BaseModel):
    action: Literal['continue', 'redirect', 'replan', 'pause']
    message: str


class Supervisor:
    def __init__(self, directory, control, model, enabled=True):
        self.directory, self.control, self.model = directory, control, model
        self.enabled = enabled
        self.agents = {}
        self.tool_counts = {}
        self.events = deque(maxlen=80)
        self.reviews = 0
        self.failures = 0
        self.last_review = -10
        self.tick = 0
        self.changed = asyncio.Event()
        self.trajectory = None
        self.register('main', 'developer', '')
        model.on_event = self.model_event

    def register(self, agent_id, role, task):
        self.agents[agent_id] = {'id': agent_id, 'role': role, 'task': task[:1200], 'status': 'queued',
                                 'context': [], 'reasoning': '', 'output': '', 'actions': [], 'receipt': {}}
        self.event(agent_id, 'created', role)

    @contextmanager
    def scope(self, agent_id):
        token = _active.set(agent_id)
        try:
            yield
        finally:
            _active.reset(token)

    def event(self, agent_id, kind, text='', **fields):
        event = {'agent': agent_id, 'kind': kind, 'text': str(text)[:1600], 'ts': time.time(), **fields}
        if self.trajectory and kind not in {'request', 'response', 'validated', 'end', 'action', 'reasoning-tail'}:
            self.trajectory.record('runtime_event', {'kind': kind, 'text': str(text), **fields},
                                   agent_id, self.trajectory.calls.get(agent_id))
        self.events.append(event)
        with (self.directory/'supervisor.jsonl').open('a', encoding='utf-8') as log:
            log.write(json.dumps(event, ensure_ascii=False) + '\n')

    def model_event(self, kind, data):
        agent_id = _active.get()
        if self.trajectory:
            self.trajectory.model_event(agent_id, kind, data)
        if agent_id not in self.agents:
            self.register(agent_id, agent_id, '')
        agent = self.agents[agent_id]
        if kind == 'request':
            agent.update(status='thinking', context=data.get('messages', []), reasoning='', output='',
                         receipt=data.get('receipt', {}))
        elif kind == 'reasoning':
            agent['reasoning'] = (agent['reasoning'] + str(data))[-6000:]
            return  # Live view updates immediately; avoid one disk write per streamed token.
        elif kind == 'content':
            agent['output'] = (agent['output'] + str(data))[-3000:]
            return
        elif kind == 'end':
            agent['status'] = 'deciding'
            if agent['reasoning']:
                self.event(agent_id, 'reasoning-tail', agent['reasoning'][-1600:])
        self.event(agent_id, kind, data if isinstance(data, str) else '',
                   receipt=data.get('receipt') if isinstance(data, dict) else None)

    def action(self, agent_id, name, ok, evidence, signature=''):
        if self.trajectory:
            self.trajectory.action(agent_id, name, ok, evidence, self.directory)
        counts=self.tool_counts.setdefault(agent_id,{})
        counts[name]=counts.get(name,0)+1
        agent = self.agents[agent_id]
        agent['status'] = 'working'
        agent['actions'] = (agent['actions'] + [{'tool': name, 'ok': ok, 'evidence': evidence,'signature':signature}])[-10:]
        self.event(agent_id, 'action', name, ok=ok, evidence=evidence)
        self.tick += 1
        self.changed.set()
        self.failures = self.failures + 1 if not ok else 0

    def finish(self, agent_id, status='done'):
        if self.trajectory:
            self.trajectory.record('finish', {'status': status}, agent_id, self.trajectory.calls.get(agent_id))
        self.agents[agent_id]['status'] = status
        self.event(agent_id, status)

    def view(self, include_context=False):
        return {'agents': [{k:v for k,v in a.items() if include_context or k != 'context'} for a in self.agents.values()],
                'tool_counts':{k:dict(v) for k,v in self.tool_counts.items()},
                'events': list(self.events)[-25:], 'reviews': self.reviews,
                'policy': 'observe continuously; bounded model review on repeated failures; permissions unchanged'}

    async def review(self, task, plan, anomaly=None):
        if self.control.pending or not self.enabled or self.reviews >= 2 or (not anomaly and self.failures < 2):
            return None
        if self.tick - self.last_review < 2:
            return None
        self.last_review = self.tick
        self.reviews += 1
        self.register('supervisor', 'operator', task)
        with self.scope('supervisor'):
            try:
                advice = await self.control.call(lambda: self.model([
                    {'role':'system','content':'You supervise a coding agent. Use concrete events only. Do not expand permissions or weaken tests. Return JSON action continue/redirect/replan/pause and a short message. Redirect only when the evidence supports a better next step. Pause hopeless repetition.'},
                    {'role':'user','content':json.dumps({'task':task[:1500],'plan':plan,'anomaly':anomaly,'events':[e for e in self.events if e['kind'] in {'action','created','decision','invalid'}][-8:]}, ensure_ascii=False)}], Advice, 'low'), 'supervisor')
                if not isinstance(advice, Advice):
                    advice = Advice.model_validate(advice)
            except (ValueError, KeyError):
                return None
            except Steered:
                return None
        self.event('supervisor', 'decision', advice.message, action=advice.action)
        self.finish('supervisor')
        if advice.action == 'pause':
            self.control.pause = True
        return advice
