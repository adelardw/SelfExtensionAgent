"""Durable human questions, inspired by DSH superagent's Attention lifecycle.

Only frontend replies or cited operator messages resolve a question. A model's
inferred answer, ordinary progress, and a timeout cannot resolve it.
"""
from __future__ import annotations

import json
from pydantic import BaseModel, ConfigDict, Field
from .obligations import digest


class Question(BaseModel):
    model_config = ConfigDict(extra='forbid')
    question: str = Field(min_length=1, max_length=600)
    why: str = Field(min_length=1, max_length=500)
    options: list[str] = Field(default_factory=list, max_length=3)
    answer_id: str = Field(default='', max_length=64)


class Attention:
    def __init__(self, ws, control, *, resume=False):
        self.ws, self.control = ws, control
        self.owner = getattr(ws, 'owner_id', 'main')
        self.path = ws.directory / ('attention-'+self.owner+'.json')
        self.data = json.loads(self.path.read_text()) if resume and self.path.exists() else {'items':[]}
        self.save()

    def save(self):
        from .harness import _atomic
        _atomic(self.path,json.dumps(self.data,ensure_ascii=False))

    def pending(self):
        return [x for x in self.data['items'] if x['status']=='pending']

    def reply_candidates(self):
        """New user messages are candidates, never inferred answers or consent."""
        result=[]
        for entry in self.pending():
            newer=self.control.instructions(self.owner)[entry['steering_version']:]
            ids=[x['id'] for x in newer if x.get('origin')=='user'][-4:]
            if ids: result.append({'question_id':entry['id'],'answer_ids':ids})
        return result

    async def ask(self, args):
        spec = Question.model_validate(args)
        if not spec.question.strip() or not spec.why.strip() or any(not s.strip() or len(s)>200 for s in spec.options):
            raise ValueError('Use a concrete question, its material impact and up to three short options')
        key = digest(spec.question.strip().casefold())[:24]
        entry = next((x for x in self.data['items'] if x['id']==key),None)
        if entry is None:
            if len(self.data['items']) >= 8:
                raise ValueError('Clarification budget exhausted; inspect existing questions and operator answers')
            entry = {'id':key,'question':spec.question,'why':spec.why,'options':spec.options,
                     'status':'pending','steering_version':self.control.version(self.owner)}
            self.data['items'].append(entry)
        if entry['status']=='answered':
            return {'ok':True,'output':json.dumps(entry,ensure_ascii=False)}
        self.save()
        if self.owner=='main': self.control.snapshot['human_action'] = self.pending()
        if spec.answer_id:
            newer = self.control.instructions(self.owner)[entry['steering_version']:]
            reply = next((x for x in newer if x['id']==spec.answer_id and x.get('origin')=='user'),None)
            if reply is None: raise ValueError('answer_id must cite a newer real operator message')
            answer = reply['text']
        elif self.owner != 'main':
            return {'ok':False,'needs_input':True,'human_action':entry,
                    'output':'Parent must resolve this question: '+entry['question']}
        else:
            from src.interface import clarify
            resolved = await clarify.ask([spec.model_dump(exclude={'answer_id'})],require_answer=True)
            answer = resolved[0]['answer'] if resolved and resolved[0]['status'] in {'answered','reused'} else ''
            if answer:
                receipt = self.control.post('Clarification: '+entry['question']+'\nOperator answer: '+answer,
                                            interrupt=False,target=self.owner)
                entry['answer_id'] = receipt['id']
        if not answer:
            return {'ok':False,'needs_input':True,'human_action':entry,
                    'output':'Требуется ответ: '+entry['question']+'\nПочему: '+entry['why']}
        entry.update(status='answered',answer=answer)
        if spec.answer_id: entry['answer_id']=spec.answer_id
        self.save()
        if self.owner=='main': self.control.snapshot['human_action']=self.pending()
        return {'ok':True,'output':json.dumps(entry,ensure_ascii=False)}
