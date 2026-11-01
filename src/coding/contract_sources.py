"""Bind requirement citations to exact input spans without model transcription."""
from __future__ import annotations

import re
from pydantic import model_validator


def source_packet(task,instructions):
    index={}
    def blocks(text,prefix):
        # Preserve every character in order, including code indentation. Bound
        # each stored quote; segmentation is a pointer scheme, not NLP inference.
        ends=[m.end() for m in re.finditer(r'(?<=[.!?;])\s+|\n+',text)]
        if not ends or ends[-1]!=len(text):ends.append(len(text))
        pieces=[];start=0
        for stop in ends:
            while start<stop:
                end=min(start+400,stop)
                if end<stop:
                    spaces=list(re.finditer(r'\s+',text[start:end]))
                    if spaces and spaces[-1].end()>200:end=start+spaces[-1].end()
                source_id=prefix+str(len(pieces))
                index[source_id]=text[start:end]
                pieces.append({'id':source_id,'text':text[start:end]})
                start=end
        return pieces
    packet={'task':blocks(task,'t'),
        'human_steering':[{**entry,'text':blocks(entry['text'],f'u{i}:')} for i,entry in enumerate(instructions)]}
    return packet,index


def source_contract(base,index):
    class SourceContract(base):
        @model_validator(mode='before')
        @classmethod
        def bind_sources(cls,value):
            if not isinstance(value,dict) or not isinstance(value.get('requirements'),list):return value
            rules=[]
            for rule in value['requirements']:
                if isinstance(rule,dict) and 'source_id' in rule:
                    rule=dict(rule);source_id=rule.pop('source_id')
                    if not isinstance(source_id,str) or source_id not in index:
                        raise ValueError('source_id must identify a supplied task/operator block')
                    quote=index[source_id]
                    if rule.get('source_quote') not in (None,'',quote):
                        raise ValueError('source_quote conflicts with the referenced source_id')
                    rule['source_quote']=quote
                rules.append(rule)
            return {**value,'requirements':rules}

        @classmethod
        def model_json_schema(cls,**kwargs):
            schema=super().model_json_schema(**kwargs)
            requirement=schema['$defs']['Requirement']
            requirement['properties'].pop('source_quote',None)
            requirement['properties']['source_id']={'type':'string','enum':list(index),
                'description':'ID of the task/operator block containing the requirement. Reuse a block for separate requirements.'}
            requirement['required']=[k for k in requirement['required'] if k!='source_quote']+['source_id']
            return schema
    SourceContract.__name__=base.__name__
    return SourceContract
