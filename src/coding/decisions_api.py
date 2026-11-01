"""Bounded OpenRouter Decisions transport shared by retrieval and orchestration."""
from __future__ import annotations

import asyncio
import copy
import json
import math
import re

import httpx

from .obligations import digest


def probability(value):
    return not isinstance(value,bool) and isinstance(value,(int,float)) and math.isfinite(value) and 0 <= value <= 1


class DecisionsClient:
    def __init__(self, config):
        self.config = dict(config)
        self.name = str(self.config.get('model', 'typesafe/jev-1.13'))
        if not re.fullmatch(r'[A-Za-z0-9_-]+/[A-Za-z0-9_.:-]+', self.name):
            raise ValueError('Use an explicit author/model id for the Decisions API')
        self.calls = self.tokens = self.input_tokens = self.output_tokens = 0
        self.cost = 0.0
        self.cache = {}
        self.estimated_calls = 0
        self.lock = asyncio.Lock()
        self.metadata = None
        self.on_event = None

    def usage(self):
        return {'calls':self.calls,'cost':self.cost,'tokens':self.tokens,
                'tokens_in':self.input_tokens,'tokens_out':self.output_tokens,
                'cost_estimated':self.estimated_calls > 0,'model':self.name}

    async def prepare(self, client):
        response = await client.get('https://openrouter.ai/api/v1/models/' + self.name + '/endpoints')
        response.raise_for_status()
        meta = response.json()['data']
        if meta.get('id') != self.name or 'decisions' not in meta.get('architecture',{}).get('output_modalities',[]):
            raise ValueError('Configured model does not expose the Decisions API')
        if not meta.get('endpoints'): raise ValueError('Decision model has no available provider endpoints')
        ceiling = float(self.config.get('max_input_usd_per_million',.05))
        if not math.isfinite(ceiling) or ceiling < 0: raise ValueError('Invalid input price ceiling')
        for endpoint in meta['endpoints']:
            prices = endpoint.get('pricing',{})
            rate = float(prices.get('prompt','nan'))
            if (not math.isfinite(rate) or rate < 0 or rate*1e6 > ceiling or
                    float(prices.get('completion','nan')) != 0 or float(prices.get('request',0) or 0) != 0):
                raise ValueError('Decision provider exceeds the cheap scoring price ceiling')
        self.metadata = meta

    async def request(self, state, questions):
        """Serialize spend reservations; canceled calls remain conservatively charged."""
        async with self.lock:
            return await self._request(state,questions)

    async def _request(self, state, questions):
        from src.llm.llm import api_key
        if not api_key(): raise ValueError('OpenRouter key missing')
        if not 1 <= len(questions) <= 24: raise ValueError('Use 1..24 decision questions')
        for question in questions.values():
            if question.get('type') not in {'choice','noul'} or not question.get('instructions'):
                raise ValueError('Use typed choice/noul questions with instructions')
            if question['type']=='choice' and not 1 <= len(question.get('criteria',{})) <= 64:
                raise ValueError('Use 1..64 choice alternatives')
        payload = {'model':self.name,'state':state,'questions':questions}
        encoded = json.dumps(payload,ensure_ascii=False)
        if len(encoded.encode()) > int(self.config.get('max_input_bytes',32000)):
            raise ValueError('Decision input exceeds compact-context budget; no silent constraint truncation')
        key = digest(payload)
        if key in self.cache: return copy.deepcopy(self.cache[key])
        if self.calls >= int(self.config.get('max_calls',2)):
            raise ValueError('Decision call budget exhausted')
        reserve = (len(encoded.encode())+1024)*float(self.config.get('max_input_usd_per_million',.05))/1e6
        limit = float(self.config.get('max_cost_usd',.002))
        if not math.isfinite(limit) or limit < 0 or not math.isfinite(reserve) or reserve < 0 or self.cost+reserve > limit:
            raise ValueError('Decision spend envelope reached')
        async with httpx.AsyncClient(timeout=float(self.config.get('timeout_seconds',8))) as client:
            if self.metadata is None: await self.prepare(client)
            from .budget import budget_for
            with budget_for(self).reserve(len(encoded.encode())+1024) as reservation:
                if self.on_event:
                    self.on_event('request',{'messages':[{'role':'user','content':encoded}],
                        'receipt':{'decision_model':self.name,'questions':list(questions),'input_utf8_bytes':len(encoded.encode())}})
                self.calls += 1
                self.cost += reserve
                self.estimated_calls += 1
                reservation.sent = True
                response = await client.post('https://openrouter.ai/api/alpha/decisions',json=payload,
                                             headers={'Authorization':'Bearer '+api_key()})
                if response.status_code in {400,401,402,403,404,422,429}:
                    reservation.rejected()
                response.raise_for_status()
                data = response.json()
                usage = data.get('usage',{})
                reservation.usage(usage, 'input_tokens', 'output_tokens')
            try: cost = float(usage.get('cost','nan'))
            except (ValueError,TypeError): cost = math.nan
            if math.isfinite(cost) and cost >= 0:
                self.cost += cost-reserve
                self.estimated_calls -= 1
            self.input_tokens += max(0,int(usage.get('input_tokens',0)))
            self.output_tokens += max(0,int(usage.get('output_tokens',0)))
            self.tokens = self.input_tokens+self.output_tokens
            answers = data.get('answers',{})
            if not isinstance(answers,dict) or set(answers) != set(questions):
                raise ValueError('Incomplete or foreign decision answer ids')
            for qid,question in questions.items():
                answer = answers[qid]
                if not isinstance(answer,dict): raise ValueError('Invalid decision answer object')
                if question['type']=='noul':
                    if not probability(answer.get('noul')): raise ValueError('Invalid noul probability')
                else:
                    options = set(question['criteria'])
                    scores = answer.get('probabilities',{})
                    choice = answer.get('choice')
                    if not isinstance(choice,str) or choice not in options or not isinstance(scores,dict) or set(scores) != options or not all(probability(v) for v in scores.values()):
                        raise ValueError('Invalid choice or probability ids')
                    if abs(sum(scores.values())-1) > .05 or scores[choice] < max(scores.values()):
                        raise ValueError('Inconsistent choice distribution')
                    if 'confidence' in answer and not probability(answer['confidence']):
                        raise ValueError('Invalid choice confidence')
            if self.on_event:
                self.on_event('content',json.dumps(answers,ensure_ascii=False))
                self.on_event('end',{})
            self.cache[key] = copy.deepcopy(answers)
            if len(self.cache) > 48: self.cache.pop(next(iter(self.cache)))
            return answers
