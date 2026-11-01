"""Bounded native dialogue, local to one actor and one steering version.

Only completed provider messages paired with actual runtime outcomes are replayed.
This is conversational context, never a verification cache or an execution path.
"""
from __future__ import annotations

import copy
import contextvars
import json


active_dialogue = contextvars.ContextVar('native_actor_dialogue', default=None)


def encoded_size(value):
    return len(json.dumps(value, ensure_ascii=False).encode())


class AssistantCapture:
    """Collect a complete assistant message without clipping signed reasoning.

    Ambiguous detail fragments or oversized messages disable replay, not action
    decoding. Their full tool result remains available through ordinary evidence.
    """
    def __init__(self, limit=12000):
        self.limit = limit
        self.size = 0
        self.valid = True
        self.message = {'role': 'assistant', 'content': None}
        self.blocks = {}

    def add(self, delta, *, complete=False):
        if not self.valid:
            return
        for field in ('content', 'reasoning'):
            text = delta.get(field)
            if field == 'reasoning' and not text:
                text = delta.get('reasoning_content')
            if text is not None:
                if not isinstance(text, str):
                    self.valid = False
                    return
                self.message[field] = (self.message.get(field) or '') + text
        details = delta.get('reasoning_details')
        if details is None:
            self._bound()
            return
        if not isinstance(details, list):
            self.valid = False
            return
        if complete:
            self.message['reasoning_details'] = copy.deepcopy(details)
            self._bound()
            return
        for item in details:
            if not isinstance(item, dict) or not isinstance(item.get('index'), int):
                self.valid = False
                return
            index = item['index']
            # Stream order is preserved; never sort or guess unidentified blocks.
            if self.blocks and index < max(self.blocks):
                self.valid = False
                return
            block = self.blocks.setdefault(index, {})
            for key, value in item.items():
                if value is None:
                    block.setdefault(key, value)
                elif key in {'text', 'summary', 'data', 'signature'} and isinstance(value, str):
                    block[key] = (block.get(key) or '') + value
                elif block.get(key) is None or block[key] == value:
                    block[key] = copy.deepcopy(value)
                else:
                    self.valid = False
                    return
        self._bound()

    def _bound(self):
        self.size = encoded_size(self.message) + encoded_size(self.blocks)
        if self.size > self.limit:
            self.valid = False
            self.message.clear()
            self.blocks.clear()

    def finish(self, calls):
        if not self.valid or len(calls) != 1:
            return None
        call = next(iter(calls.values()))
        if not call.get('id'):
            return None  # Real ids only; don't synthesize a provider conversation.
        try:
            if not isinstance(json.loads(call['arguments']), dict):
                return None
        except (ValueError, TypeError, KeyError):
            return None
        message = copy.deepcopy(self.message)
        if self.blocks:
            message['reasoning_details'] = list(self.blocks.values())
        if message.get('reasoning_details'):
            message.pop('reasoning', None)  # Same content may arrive in both forms.
        message['tool_calls'] = [{'id': call['id'], 'type': 'function',
            'function': {'name': call['name'], 'arguments': call['arguments']}}]
        return message if encoded_size(message) <= self.limit else None


class ActorDialogue:
    def __init__(self, enabled=False, limit=12000, exchanges=3):
        self.enabled = enabled
        self.limit = max(0, min(16000, limit))
        self.exchanges = max(0, min(4, exchanges))
        self.history = []
        self.pending = None
        self.epoch = None

    async def generate(self, model, messages, schema, effort, epoch):
        if epoch != self.epoch:
            self.history.clear()
            self.epoch = epoch
        self.pending = None
        token = active_dialogue.set(self if self.enabled else None)
        try:
            return await model(messages, schema, effort)
        except BaseException:
            self.pending = None
            raise
        finally:
            active_dialogue.reset(token)

    def capture(self, message):
        self.pending = copy.deepcopy(message)

    def record(self, result, evidence_id=None):
        """Called by runtime AFTER execution or another accepted state transition."""
        message, self.pending = self.pending, None
        if message is None or not isinstance(result.get('ok'), bool):
            return
        output = str(result.get('output', ''))
        response = {'ok': result['ok'], 'output': output[:2800],
                    'historical': True}
        if len(output) > 2800:
            response['output_truncated'] = True
        if evidence_id is not None:
            response['evidence_id'] = str(evidence_id)
        pair = [message, {'role': 'tool',
                'tool_call_id': message['tool_calls'][0]['id'],
                'content': json.dumps(response, ensure_ascii=False)}]
        self.history.append(pair)
        while self.history and (len(self.history) > self.exchanges or encoded_size(self.history) > self.limit):
            self.history.pop(0)

    def attach(self, messages, budget):
        """Keep a complete recent suffix; current state and permissions win."""
        try:
            packet = json.loads(messages[-1]['content'])
        except (ValueError, TypeError, KeyError):
            packet = None

        def render(pairs):
            history = copy.deepcopy([m for p in pairs for m in p])
            if not isinstance(packet, dict) or not isinstance(packet.get('task'), str):
                return messages[:-1] + history + messages[-1:]
            state = copy.deepcopy(packet)
            task = {'task': state.pop('task')}
            if 'human_steering' in state:
                task['human_steering'] = state.pop('human_steering')
            # Continue the assistant -> tool exchange instead of starting a fresh
            # user turn after every tool. The runtime state is separate from the
            # recorded tool output, which remains data, not instructions.
            ids = set()
            reads = {}
            for action, result in pairs:
                observation = json.loads(result['content'])
                if observation.get('evidence_id'):
                    ids.add(observation['evidence_id'])
                function = action['tool_calls'][0]['function']
                if function['name'] == 'read' and observation.get('ok'):
                    args = json.loads(function['arguments'])
                    reads[args.get('path')] = observation['output']
            memory = state.get('working_memory', {})
            for owner, key in ((state, 'recent'), (memory, 'observations')):
                if key in owner:
                    owner[key] = [r for r in owner[key] if r.get('evidence_id') not in ids]
            for owner in (state, memory):
                content = owner.get('inspected_content', {})
                owner['inspected_content'] = {p: text for p, text in content.items() if reads.get(p) != text}
            latest = json.loads(history[-1]['content'])
            latest['current_state'] = state
            history[-1]['content'] = json.dumps(latest, ensure_ascii=False)
            return messages[:-1] + [{'role': 'user', 'content': json.dumps(task, ensure_ascii=False)}] + history

        selected = []
        result = messages
        for pair in reversed(self.history):
            candidate = [pair] + selected
            transcript = render(candidate)
            if encoded_size(transcript) > budget:
                break
            selected = candidate
            result = transcript
        ids = [p[0]['tool_calls'][0]['id'] for p in selected]
        return copy.deepcopy(result), ids
