"""Explicit local/cheap-cloud structured turns, without implicit provider fallback."""
from __future__ import annotations

import json
import re
import asyncio
import time

import httpx
from pydantic import BaseModel

from src.config.config_paths import load_base_config
from .budget import BudgetExceeded, budget_for


class LocalModel:
    def __init__(self, config: dict | None = None):
        self.config = config if config is not None else dict(load_base_config().get("coding", {}))
        from src.llm.llm import ollama_base_url
        self.base = ollama_base_url().removesuffix("/v1")
        self.name = str(self.config.get("model", "qwen3:8b"))
        self.input_tokens = self.output_tokens = self.calls = 0
        self.ready = False
        self.thinking = False

    async def prepare(self, client):
        r = await client.post(self.base + "/api/show", json={"model": self.name})
        r.raise_for_status()
        data = r.json()
        count = data.get("model_info", {}).get("general.parameter_count")
        if not count:
            size = str(data.get("details", {}).get("parameter_size", ""))
            m = re.fullmatch(r"([\d.]+)\s*([BMK])", size, re.I)
            if m:
                count = float(m[1]) * {"B": 1e9, "M": 1e6, "K": 1e3}[m[2].upper()]
        if not count:
            raise ValueError("Ollama не сообщил размер модели; профиль требует проверяемую модель ≤10B.")
        if float(count) > min(10, float(self.config.get("max_parameters_b", 10))) * 1e9:
            raise ValueError(f"{self.name}: модель превышает лимит локального профиля 10B.")
        self.thinking = "thinking" in data.get("capabilities", [])
        self.ready = True

    async def __call__(self, messages: list[dict], schema: type[BaseModel], effort: str):
        timeout = float(self.config.get("timeout_seconds", 180))
        async with httpx.AsyncClient(timeout=timeout) as client:
            if not self.ready:
                await self.prepare(client)
            output = {"low": 1536, "medium": 3072, "high": 4096}[effort]
            from .request import fit_messages
            context_tokens = int(self.config.get("context_tokens", 16384))
            messages, self.context_receipt = fit_messages(messages, schema, context_tokens, output)
            if getattr(self, 'on_event', None): self.on_event('request', {'messages': messages,
                'receipt': self.context_receipt, 'schema': schema.model_json_schema(),
                'generation': {'model': self.name, 'temperature': 0, 'effort': effort}})
            payload = {"model": self.name, "messages": messages, "stream": False,
                       "format": schema.model_json_schema(), "keep_alive": "10m",
                       "options": {"temperature": 0, "num_ctx": context_tokens,
                                   "num_predict": output}}
            if self.thinking:
                payload["think"] = effort != "low"
            bound = self.context_receipt['input_utf8_bytes'] + len(json.dumps(payload['format']).encode()) + output + 384
            with budget_for(self).reserve(bound) as reservation:
                reservation.sent = True
                r = await client.post(self.base + "/api/chat", json=payload)
                r.raise_for_status()
                data = r.json()
                reservation.usage(data, 'prompt_eval_count', 'eval_count')
            self.calls += 1
            self.input_tokens += int(data.get("prompt_eval_count", 0))
            self.output_tokens += int(data.get("eval_count", 0))
            if getattr(self, 'on_event', None): self.on_event('response', {
                'content': data['message']['content'], 'tool_calls': [],
                'usage': {k: data.get(k) for k in ('prompt_eval_count', 'eval_count')},
                'finish_reason': data.get('done_reason')})
            result = schema.model_validate_json(data["message"]["content"])
            if getattr(self, 'on_event', None):
                self.on_event('validated', result.model_dump())
                self.on_event('end', {})
            return result

    def usage(self) -> dict:
        return {"calls": self.calls, "tokens_in": self.input_tokens,
                "tokens_out": self.output_tokens, "tokens": self.input_tokens + self.output_tokens,
                "model": self.name, "provider": "ollama"}


class LlamaCppModel(LocalModel):
    """Unsloth GGUFs through a local llama-server with Metal/CPU inference."""
    def __init__(self, config: dict | None = None):
        self.config = config if config is not None else dict(load_base_config().get("coding", {}))
        self.base = str(self.config.get("base_url", "http://127.0.0.1:8081/v1")).rstrip("/")
        self.name = str(self.config.get("model", "sea-local"))
        self.input_tokens = self.output_tokens = self.calls = 0
        self.ready = False
        self.thinking = True
        self.cached_tokens = 0
        self.generation_seconds = 0.0
        self.generated_tokens = 0
        self.server_context = None
        self.parameter_count = self.model_size_bytes = 0
        self.model_file = ""
        self.reasoning_calls = 0

    async def prepare(self, client):
        r = await client.get(self.base + "/models")
        r.raise_for_status()
        models = r.json().get("data", [])
        model = next((m for m in models if m.get("id") == self.name), None)
        if model is None:
            raise ValueError(f"llama.cpp model {self.name!r} not loaded; available: {[m.get('id') for m in models]}")
        count = model.get("meta", {}).get("n_params")
        if not count or count > min(10, float(self.config.get("max_parameters_b", 10))) * 1e9:
            raise ValueError("llama.cpp must report a loaded model with ≤10B parameters")
        self.parameter_count = int(count)
        self.model_size_bytes = int(model.get("meta", {}).get("size", 0))
        # A server may reserve a smaller per-slot context than the harness default.
        props = await client.get(self.base.removesuffix("/v1") + "/props")
        if props.is_success:
            data = props.json()
            from pathlib import Path
            self.model_file = Path(data.get("model_path", "")).name
            settings = data.get("default_generation_settings", {})
            self.server_context = settings.get("n_ctx")
        self.ready = True

    async def __call__(self, messages: list[dict], schema: type[BaseModel], effort: str):
        async with httpx.AsyncClient(timeout=float(self.config.get("timeout_seconds", 180))) as client:
            if not self.ready:
                await self.prepare(client)
            output = {"low": 1536, "medium": 3072, "high": 4096}[effort]
            context = min(int(self.config.get("context_tokens", 16384)), self.server_context or 16384)
            schema_dict = schema.model_json_schema()
            from .request import fit_messages
            messages, self.context_receipt = fit_messages(messages, schema, context, output)
            if getattr(self, 'on_event', None): self.on_event('request', {'messages': messages,
                'receipt': self.context_receipt, 'schema': schema_dict,
                'generation': {'model': self.name, 'effort': effort, 'seed': self.config.get('seed', 3407)}})
            payload = {"model": self.name, "messages": messages, "stream": False,
                       "response_format": {"type": "json_schema", "json_schema": {"name": schema.__name__, "schema": schema_dict}},
                       "max_tokens": output, "temperature": 0.6 if effort != "low" else 0.7,
                       "top_p": 0.95 if effort != "low" else 0.8, "top_k": 20, "min_p": 0,
                       "chat_template_kwargs": {"enable_thinking": effort != "low"},
                       "reasoning_effort": "none" if effort == "low" else effort,
                       "reasoning_budget_tokens": {"low": 0, "medium": 512, "high": 1536}[effort],
                       "seed": int(self.config.get('seed', 3407)),
                       "cache_prompt": True, "stream": True, "stream_options": {"include_usage": True}}
            content_parts, reasoning_parts = [], []
            data = {}
            bound = self.context_receipt['input_utf8_bytes'] + len(json.dumps(schema_dict).encode()) + output + 384
            with budget_for(self).reserve(bound) as reservation:
                reservation.sent = True
                async with client.stream('POST', self.base + '/chat/completions', json=payload) as response:
                    response.raise_for_status()
                    if 'text/event-stream' not in response.headers.get('content-type', ''):
                        data = json.loads(await response.aread())
                        reservation.usage(data.get('usage', {}))
                    else:
                        async for line in response.aiter_lines():
                            if not line.startswith('data: '): continue
                            raw = line[6:]
                            if raw == '[DONE]': break
                            event = json.loads(raw)
                            for key in ('usage', 'timings'):
                                if event.get(key): data[key] = event[key]
                            if event.get('usage'): reservation.usage(event['usage'])
                            choices = event.get('choices', [])
                            delta = choices[0].get('delta', {}) if choices else {}
                            for field, parts, kind in (('content', content_parts, 'content'), ('reasoning_content', reasoning_parts, 'reasoning')):
                                if delta.get(field):
                                    parts.append(delta[field])
                                    if getattr(self, 'on_event', None): self.on_event(kind, delta[field])
                        data['choices'] = [{'message': {'content': ''.join(content_parts), 'reasoning_content': ''.join(reasoning_parts)}}]
            self.calls += 1
            self.reasoning_calls += int(bool(data.get("choices", [{}])[0].get("message", {}).get("reasoning_content")))
            usage = data.get("usage", {})
            self.input_tokens += int(usage.get("prompt_tokens", 0))
            self.output_tokens += int(usage.get("completion_tokens", 0))
            timings = data.get("timings", {})
            self.cached_tokens += int(timings.get("cache_n", 0))
            self.generation_seconds += float(timings.get("predicted_ms", 0)) / 1000
            self.generated_tokens += int(timings.get("predicted_n", 0))
            content = data["choices"][0]["message"].get("content") or ""
            if getattr(self, 'on_event', None): self.on_event('response', {
                'content': content, 'tool_calls': [], 'usage': usage,
                'finish_reason': data['choices'][0].get('finish_reason')})
            if not content.strip():
                raise ValueError("model exhausted output budget without a JSON action; try low reasoning")
            result = schema.model_validate_json(content)
            if getattr(self, 'on_event', None):
                self.on_event('validated', result.model_dump())
                self.on_event('end', {})
            return result

    def usage(self) -> dict:
        return {**super().usage(), "provider": "llama_cpp", "cached_tokens": self.cached_tokens,
                "model_file": self.model_file, "parameter_count": self.parameter_count,
                "model_size_bytes": self.model_size_bytes,
                "reasoning_calls": self.reasoning_calls,
                "generation_tokens_per_second": round(self.generated_tokens / self.generation_seconds, 2) if self.generation_seconds else 0}


def build_local_model(config: dict):
    backend = config.get("backend", "ollama")
    if backend == "llama_cpp":
        return LlamaCppModel(config)
    if backend == "ollama":
        return LocalModel(config)
    if backend == 'openrouter':
        return OpenRouterModel(config)
    raise ValueError("coding.backend must be llama_cpp, ollama or explicitly openrouter; no implicit fallback")


class ProviderUnavailable(BudgetExceeded):
    """Checkpoint a rejected request instead of treating it as bad model reasoning."""
    def __init__(self,message,details=None):
        self.details=dict(details or {})
        super().__init__(message)


class OutputBudgetExceeded(BudgetExceeded):
    """An exhausted completion budget is not malformed JSON to retry blindly."""
    def __init__(self, details, provider_draft=''):
        self.details = dict(details)
        self.provider_draft=provider_draft
        super().__init__('Model output budget exhausted: '+json.dumps(details,ensure_ascii=False)+
                         '; checkpoint retained. Adjust reasoning/output allocation within the total spend limit before resuming.')


class OpenRouterModel(LocalModel):
    """Explicit cheap cloud profile. No local inference or model-name fallback."""
    def __init__(self, config=None):
        self.config = config if config is not None else dict(load_base_config().get('coding', {}))
        from src.llm.llm import openrouter_base_url
        self.base = openrouter_base_url()
        self.name = str(self.config.get('model', 'google/gemini-3.1-flash-lite'))
        # Sampling is model-dependent. Omitting a control preserves the provider's
        # default; forcing greedy decoding can degrade some reasoning models.
        import math
        self.sampling={}
        for name,upper in (('temperature',2),('top_p',1)):
            value=self.config.get(name)
            if value is None:continue
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not 0<=value<=upper:
                raise ValueError(f'Invalid sampling {name}: expected a finite number in [0,{upper}] or null')
            self.sampling[name]=value
        self.input_tokens = self.output_tokens = self.calls = self.cached_tokens = 0
        self.cost = self.reserved = 0.0
        self.estimated_cost = False
        self.ready = False
        self.rates = (0.0, 0.0)
        self.reasoning_meta = {}
        self.reasoning_supported = True
        self.supported_parameters = None
        self.endpoints_used = {}
        self.failed_requests=0
        self.provider_error=None
        self.provider_retries=0
        self.provider_retry_wait_seconds=0.0
        self.reasoning_tokens = self.reasoning_usage_calls = self.truncated_outputs = 0
        self.last_output = {}
        self.output_recovery_attempts = self.recovered_structured_outputs = 0
        self.json_envelope_repairs = 0
        self._request_gate=asyncio.Lock()
        self._last_request_started=None
        if isinstance(self.config.get('provider_only'),str):
            raise ValueError('provider_only must be a list of provider slugs, not a string')

    async def prepare(self, client):
        response = await client.get(self.base+'/models')
        response.raise_for_status()
        meta = next((m for m in response.json()['data'] if m['id'] == self.name), None)
        if not meta:
            raise ValueError('Configured OpenRouter model is unavailable; no fallback permitted')
        pricing = meta.get('pricing', {})
        self.reasoning_meta=meta.get('reasoning') or {}
        self.supported_parameters=meta.get('supported_parameters')
        self.reasoning_supported='reasoning' in meta.get('supported_parameters',['reasoning'])
        self.rates = (float(pricing['prompt']), float(pricing['completion']))
        ceilings = (float(self.config.get('max_input_usd_per_million', .30)),
                    float(self.config.get('max_output_usd_per_million', 2.0)))
        if any(rate < 0 or rate*1e6 > cap for rate,cap in zip(self.rates,ceilings)):
            raise ValueError('OpenRouter model exceeds the cheap-profile price ceiling')
        if float(pricing.get('request',0) or 0) > 0:
            raise ValueError('Per-request charges are unsupported by this bounded cheap profile')
        self.ready = True

    def reject_response(self,response,key):
        from .provider_errors import rejection_details
        self.provider_error=rejection_details(response,key)
        self.failed_requests+=1
        hint=' Check provider credits or limits before resuming.' if response.status_code==402 else ' Inspect provider availability/request compatibility before resuming.'
        raise ProviderUnavailable('OpenRouter HTTP '+str(response.status_code)+': '+self.provider_error['message']+hint,
                                  self.provider_error)

    async def _provider_request(self,messages,schema,effort,*,output_cap=None):
        """Retry only a rejected 429/503 with an explicit bounded wait hint.

        The allowance is shared by all roles using this engine. Every attempt
        re-enters normal pacing, context validation and spend reservation.
        """
        attempt=0
        while True:
            try:return await self._request(messages,schema,effort,output_cap=output_cap)
            except ProviderUnavailable as exc:
                detail=exc.details
                delay=detail.get('retry_after_seconds')
                if detail.get('status') not in {429,503} or delay is None:raise
                maximum=max(0,min(8,int(self.config.get('provider_retries',4))))
                wait_budget=max(0,min(60,float(self.config.get('provider_retry_wait_seconds',15))))
                delay=max(1,delay)*(2**attempt)
                if self.provider_retries>=maximum or self.provider_retry_wait_seconds+delay>wait_budget:raise
                # Claim before awaiting: concurrent workers cannot multiply limits.
                self.provider_retries+=1
                self.provider_retry_wait_seconds+=delay
                attempt+=1
                if getattr(self,'on_event',None):
                    self.on_event('provider-retry',json.dumps({'attempt':self.provider_retries,
                        'wait_seconds':delay,'status':detail['status'],
                        'provider':detail.get('provider_name'),'limit_source':detail.get('limit_source')}))
                await asyncio.sleep(delay)

    async def __call__(self, messages, schema, effort):
        from .output_recovery import actor_schema,recovery_messages
        try:
            return await self._provider_request(messages,schema,effort)
        except OutputBudgetExceeded as exc:
            enabled=bool(self.config.get('output_recovery',False))
            maximum=max(0,min(2,int(self.config.get('max_output_recoveries',1))))
            if (not enabled or self.output_recovery_attempts>=maximum or not actor_schema(schema)
                    or self.reasoning_meta.get('mandatory',False) or not exc.provider_draft
                    or (exc.details.get('reasoning_config') or {}).get('enabled') is False):
                raise
            repaired=recovery_messages(messages,exc.provider_draft,exc.details)
            if repaired is None:raise
            # Claim a global per-engine allowance before any await. The first
            # reservation has already been released by _request's finally block.
            self.output_recovery_attempts+=1
            cap=max(1,min(exc.details['max_output_tokens'],int(self.config.get('output_recovery_tokens',1536))))
            if getattr(self,'on_event',None):
                self.on_event('output-recovery',json.dumps({'attempt':self.output_recovery_attempts,
                    'draft_chars':len(exc.provider_draft),'max_output_tokens':cap,'reasoning_enabled':False}))
            # This completion phase cannot recursively repair output, switch
            # providers, extend its budget, or execute a partial call. Transport
            # retries still use the same shared allowance and spend checks.
            result=await self._provider_request(repaired,schema,'low',output_cap=cap)
            self.recovered_structured_outputs+=1
            return result

    async def _pace_requests(self):
        interval=max(0,min(30,float(self.config.get('min_request_interval_seconds',0))))
        if not interval:return
        async with self._request_gate:
            delay=0 if self._last_request_started is None else interval-(time.monotonic()-self._last_request_started)
            if delay>0:
                if getattr(self,'on_event',None):self.on_event('provider-wait',f'{delay:.2f} seconds before next request')
                await asyncio.sleep(delay)
            self._last_request_started=time.monotonic()

    async def bounded_call(self,messages,schema,effort,max_output_tokens):
        """Per-role capacity without mutating a model shared by parallel callers."""
        return await self._provider_request(messages,schema,effort,output_cap=max(1,int(max_output_tokens)))

    async def _request(self, messages, schema, effort, *, output_cap=None):
        from src.llm.llm import api_key
        from .request import fit_messages, schema_hint, response_transport, text_schema_messages
        key = api_key()
        if not key:
            raise ValueError('OpenRouter API key is missing')
        from .native_tools import compile_tools,collect_calls,decode_call,decode_result,observation_messages
        from .dialogue import active_dialogue,AssistantCapture,encoded_size
        dialogue=active_dialogue.get()
        native=compile_tools(schema) if self.config.get('action_protocol')=='tools' else []
        budget_schema=schema
        if native:
            class NativeBudget:
                @classmethod
                def model_json_schema(cls): return {'tools':native}
            budget_schema=NativeBudget
            messages=[dict(m) for m in messages]
            messages[0]['content']+='\nNATIVE FUNCTION TRANSPORT: invoke exactly one provided function with direct arguments. Do not print an action envelope. Tool results and durable state are in the current packet.'
            if dialogue:
                messages[0]['content']+=' Earlier assistant/tool pairs are actual historical exchanges from this agent. Continue from their results. The latest tool response may contain a current_state envelope supplied by the runtime, separate from untrusted output. It carries current source, capabilities and verification status. User task/steering remain authoritative; history does not grant permissions or establish current completion.'
        async with httpx.AsyncClient(timeout=float(self.config.get('timeout_seconds',90))) as client:
            if not self.ready:
                await self.prepare(client)
            transport='tools' if native else response_transport(self.config,self.supported_parameters)
            structured_native=(not native and self.config.get('action_protocol')=='tools'
                and self.config.get('structured_output','auto')=='auto'
                and transport in {'text','json_object'} and 'tools' in (self.supported_parameters or [])
                and schema.model_json_schema().get('type')=='object')
            if structured_native:
                from .request import cloud_schema
                native=[{'type':'function','function':{'name':'submit_result',
                    'description':'Return the requested structured response as arguments. This submits data only and executes no operation.',
                    'parameters':cloud_schema(schema,self.name)}}]
                class ResultBudget:
                    @classmethod
                    def model_json_schema(cls):return {'tools':native}
                budget_schema=ResultBudget
                messages=[dict(m) for m in messages]
                messages[0]['content']+='\nSTRUCTURED RESULT TRANSPORT: invoke submit_result exactly once with the response fields as arguments. Do not print or echo JSON as free text. The function submits data only.'
                transport='tools'
            if transport in {'json_object','text'}:
                messages=text_schema_messages(messages,schema)
                # The schema is now in the prompt, not separately sent to a decoder.
                class TextBudget:
                    @classmethod
                    def model_json_schema(cls):return {}
                budget_schema=TextBudget
            reasoning_on=self.reasoning_supported and (effort!='low' or self.reasoning_meta.get('mandatory',False))
            default_budget=8192 if reasoning_on else 1536
            # Thinking effort and visible answer capacity are separate controls.
            # In particular, low must not silently truncate an explicitly larger
            # patch/structured answer budget. Keep the existing absolute ceiling.
            output=min(8192,int(self.config.get('max_output_tokens',default_budget)))
            if output_cap is not None:output=min(output,int(output_cap))
            if output<1:raise ValueError('max_output_tokens must be positive')
            context = int(self.config.get('cloud_context_tokens',self.config.get('context_tokens',16384)))
            if transport=='json_schema' and self.config.get('schema_hint',True):
                messages=[dict(m) for m in messages]
                hint=schema_hint(schema)
                if hint:
                    if messages[0]['role']=='system': messages[0]['content']+=hint
                    else: messages.insert(0,{'role':'system','content':hint})
            messages, receipt = fit_messages(messages,budget_schema,context,output)
            receipt['action_protocol']='tools' if native else 'json'
            receipt['response_transport']=transport
            if structured_native:receipt['structured_tool_result']=True
            receipt['sampling']=dict(self.sampling)
            if native and not structured_native:
                if dialogue:
                    transcript,ids=dialogue.attach(messages,context-output-512-encoded_size(budget_schema.model_json_schema()))
                    receipt['native_dialogue_ids']=ids
                else:
                    transcript,ids=observation_messages(messages,native)
                size=len(json.dumps(transcript,ensure_ascii=False).encode())
                schema_size=len(json.dumps(budget_schema.model_json_schema()).encode())
                if size+schema_size+output+512<=context:
                    messages=transcript
                    receipt.update(input_utf8_bytes=size)
                    if not dialogue:receipt['native_observation_ids']=ids
            # UTF-8 bytes conservatively bound ordinary text tokens; reserve output too.
            prompt_bound = receipt['input_utf8_bytes'] + len(json.dumps(budget_schema.model_json_schema()).encode()) + 256
            # Waiting is cancellable and does not create a billable reservation.
            # Another shared caller may spend while this call waits, so recheck.
            await self._pace_requests()
            token_budget=budget_for(self)
            requested_output=output
            output=token_budget.output_capacity(prompt_bound+128,output)
            receipt['output_capacity']={'requested':requested_output,'admitted':output}
            # Provider routing may choose a faster endpoint above the catalogue minimum.
            # Reserve at the configured provider ceilings, including a small token margin.
            reserve = (prompt_bound*float(self.config.get('max_input_usd_per_million',.30))+
                       (output+128)*float(self.config.get('max_output_usd_per_million',2.0)))/1e6
            limit = float(self.config.get('max_cost_usd',.10))
            if self.cost+self.reserved+reserve > limit:
                raise BudgetExceeded(f'Cloud spend envelope reached (${limit:g}); checkpoint retained')
            token_reservation = token_budget.reserve(prompt_bound + output + 128)
            self.reserved += reserve
            accounted = False
            try:
                self.context_receipt=receipt
                from .request import cloud_schema
                payload = {'model':self.name,'messages':messages,**self.sampling,
                    'response_format':{'type':'json_schema','json_schema':{
                        'name':schema.__name__,'strict':bool(self.config.get('strict_schema',True)),
                        'schema':cloud_schema(schema,self.name)}},
                    'max_tokens':output,'stream':True,'stream_options':{'include_usage':True},
                    'provider':{'require_parameters':True,'sort':self.config.get('provider_sort','throughput'),'max_price':{
                        'prompt':float(self.config.get('max_input_usd_per_million',.30)),
                        'completion':float(self.config.get('max_output_usd_per_million',2.0))}}}
                if self.config.get('provider_only'):
                    providers=list(self.config['provider_only'])
                    if not providers or not all(isinstance(p,str) and p.strip() for p in providers):
                        raise ValueError('provider_only must contain explicit provider slugs')
                    payload['provider'].update(only=providers,allow_fallbacks=False)
                if transport=='json_object':payload['response_format']={'type':'json_object'}
                elif transport=='text':payload.pop('response_format')
                if native:
                    payload.pop('response_format')
                    # Some tool-capable endpoints do not advertise parallel_tool_calls;
                    # require_parameters would exclude them. The decoder enforces one call.
                    # Several endpoints advertise tool_choice but only implement auto.
                    # The client still requires exactly one valid call before execution.
                    payload['tools']=native
                    if self.supported_parameters is None or 'tool_choice' in self.supported_parameters:
                        payload['tool_choice']='auto'
                if self.reasoning_supported:
                    if effort=='low' and not self.reasoning_meta.get('mandatory',False):
                        payload['reasoning']={'enabled':False}
                    elif 'supported_efforts' not in self.reasoning_meta:
                        # Omitted means no effort selector; do not pretend that
                        # low/medium/high has a calibrated effect on this model.
                        payload['reasoning']={'enabled':True}
                    else:
                        supported=self.reasoning_meta.get('supported_efforts')
                        chosen='minimal' if effort=='low' else effort
                        if supported and chosen not in supported:
                            chosen=next((level for level in ['low','medium','high','max'] if level in supported),supported[-1])
                        payload['reasoning']={'effort':chosen}
                if getattr(self,'on_event',None):
                    self.on_event('request',{'messages':messages,'receipt':receipt,
                        'schema':schema.model_json_schema(),
                        'generation':{k:v for k,v in payload.items() if k not in {'messages','stream','stream_options'}}})
                usage, parts, endpoint, native_calls, finish_reason = {}, [], '', {}, None
                from .output_recovery import ProviderDraft
                draft=ProviderDraft(self.config.get('output_recovery_draft_chars',12000))
                capture=AssistantCapture() if native and dialogue else None
                token_reservation.sent = True
                async with client.stream('POST',self.base+'/chat/completions',json=payload,
                                         headers={'Authorization':'Bearer '+key}) as response:
                    if response.is_error:
                        if response.status_code in {400,401,402,403,404,422,429}:
                            token_reservation.rejected()
                        # Preserve the provider's diagnostic before closing a streamed response.
                        await response.aread()
                        self.reject_response(response,key)
                    response.raise_for_status()
                    if 'text/event-stream' not in response.headers.get('content-type',''):
                        data = json.loads(await response.aread())
                        endpoint=data.get('provider','')
                        usage = data.get('usage',{})
                        token_reservation.usage(usage)
                        choice=data['choices'][0]
                        finish_reason=choice.get('finish_reason')
                        message=choice['message']
                        if capture:capture.add(message,complete=True)
                        parts.append(message.get('content') or '')
                        returned_reasoning=message.get('reasoning') or message.get('reasoning_content')
                        draft.add(returned_reasoning)
                        if returned_reasoning and getattr(self,'on_event',None):self.on_event('reasoning',returned_reasoning)
                        if native: collect_calls(native_calls,[{**c,'index':i} for i,c in enumerate(message.get('tool_calls',[]))])
                    else:
                        async for line in response.aiter_lines():
                            if not line.startswith('data: '): continue
                            raw = line[6:]
                            if raw == '[DONE]': break
                            data = json.loads(raw)
                            endpoint=data.get('provider') or endpoint
                            if data.get('error'):
                                raise ValueError('OpenRouter stream error: '+str(data['error'].get('message','failed'))[:200])
                            if data.get('usage'):
                                usage = data['usage']
                                token_reservation.usage(usage)
                            choices = data.get('choices',[])
                            if choices and choices[0].get('finish_reason') is not None:
                                finish_reason=choices[0]['finish_reason']
                            delta = choices[0].get('delta',{}) if choices else {}
                            if capture:capture.add(delta)
                            if native: collect_calls(native_calls,delta.get('tool_calls',[]))
                            if delta.get('content'):
                                parts.append(delta['content'])
                                if getattr(self,'on_event',None): self.on_event('content',delta['content'])
                            # Only provider-returned text is observable; never synthesize hidden reasoning.
                            reasoning = delta.get('reasoning') or delta.get('reasoning_content')
                            draft.add(reasoning)
                            if reasoning and getattr(self,'on_event',None): self.on_event('reasoning',reasoning)
                self.calls += 1
                self.provider_error=None
                if endpoint: self.endpoints_used[endpoint]=self.endpoints_used.get(endpoint,0)+1
                self.input_tokens += int(usage.get('prompt_tokens',0))
                self.output_tokens += int(usage.get('completion_tokens',0))
                reasoning_tokens=(usage.get('completion_tokens_details') or {}).get('reasoning_tokens')
                if reasoning_tokens is not None:
                    reasoning_tokens=max(0,int(reasoning_tokens))
                    self.reasoning_tokens+=reasoning_tokens
                    self.reasoning_usage_calls+=1
                self.cached_tokens += int(usage.get('prompt_tokens_details',{}).get('cached_tokens',0))
                if usage.get('cost') is not None:
                    self.cost += max(0,float(usage['cost']))
                else:
                    self.cost += reserve
                    self.estimated_cost = True
                accounted = True
                if getattr(self,'on_event',None):
                    self.on_event('response',{'content':''.join(parts),
                        'tool_calls':list(native_calls.values()), 'usage':usage,
                        'finish_reason':finish_reason,'endpoint':endpoint})
                details={'finish_reason':finish_reason,'max_output_tokens':output,
                    'completion_tokens':int(usage.get('completion_tokens',0)),
                    'reasoning_tokens':reasoning_tokens,'visible_chars':sum(len(p) for p in parts),
                    'reasoning_config':payload.get('reasoning')}
                self.last_output=details
                def exhausted():
                    self.truncated_outputs+=1
                    if getattr(self,'on_event',None):
                        self.on_event('end',details)
                        self.on_event('output-budget',json.dumps(details,ensure_ascii=False))
                    return OutputBudgetExceeded(details,draft.packet())
                if finish_reason in {'length','max_tokens'}:
                    # Includes apparently valid native arguments: a truncated tool
                    # proposal must never execute just because its JSON prefix parses.
                    raise exhausted()
                try:
                    if native:
                        parts=[json.dumps(decode_result(native_calls) if structured_native else
                                          decode_call(native_calls,native),ensure_ascii=False)]
                        if getattr(self,'on_event',None): self.on_event('content',parts[0])
                    response_text=''.join(parts)
                    if transport in {'json_object','text'}:
                        from .request import normalize_json_envelope
                        response_text,repair=normalize_json_envelope(response_text)
                        if repair:
                            self.json_envelope_repairs+=1
                            if getattr(self,'on_event',None):
                                self.on_event('json-envelope-repair',repair)
                    result=schema.model_validate_json(response_text)
                except ValueError as exc:
                    if int(usage.get('completion_tokens',0))>=output:
                        raise exhausted()
                    if capture:
                        # A syntactically valid proposal can be rejected by the
                        # current scope/repetition/delivery rules. Return that
                        # actual rejection through the tool protocol as well as
                        # raising it; never pretend the proposal was executed.
                        from pydantic import ValidationError
                        errors=([e['msg'] for e in exc.errors(include_input=False,include_context=False,include_url=False)]
                                if isinstance(exc,ValidationError) else [str(exc)])
                        dialogue.capture(capture.finish(native_calls))
                        dialogue.record({'ok':False,'output':'Runtime rejected this proposal; nothing was executed. '+
                                         '; '.join(errors)[:600]})
                    if getattr(self,'on_event',None): self.on_event('end',details)
                    raise
                if getattr(self,'on_event',None):
                    self.on_event('validated',result.model_dump())
                    self.on_event('end',details)
                if capture:
                    dialogue.capture(capture.finish(native_calls))
                return result
            finally:
                token_reservation.__exit__()
                self.reserved -= reserve
                if not accounted:
                    # Cancelled/failed streams may already have been billed; reserve conservatively.
                    self.cost += reserve
                    self.estimated_cost = True

    def usage(self):
        return {**super().usage(),'provider':'openrouter','cost':round(self.cost,8),
                'cost_estimated':self.estimated_cost,'cached_tokens':self.cached_tokens,
                'endpoints_used':dict(self.endpoints_used),'failed_requests':self.failed_requests,
                'provider_retries':self.provider_retries,
                'provider_retry_wait_seconds':self.provider_retry_wait_seconds,
                'reasoning_tokens':self.reasoning_tokens,'reasoning_usage_calls':self.reasoning_usage_calls,
                'truncated_outputs':self.truncated_outputs,'last_output':dict(self.last_output),
                'output_recovery_attempts':self.output_recovery_attempts,
                'recovered_structured_outputs':self.recovered_structured_outputs,
                'json_envelope_repairs':self.json_envelope_repairs,
                **({'provider_error':dict(self.provider_error)} if self.provider_error else {})}
