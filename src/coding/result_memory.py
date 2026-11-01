"""Bounded, evidence-linked observations that survive the rolling action window.

This is a record of tool returns, never a result cache or a completion authority.
It does not execute, interpret mathematical expressions, or infer missing answers.
"""
from __future__ import annotations

import hashlib
import json
import re

from .progress import call_signature

COMPUTATIONS=frozenset({'math','experiment','run','check'})


def excerpt(value,limit):
    text=value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,separators=(',',':'))
    return text if len(text)<=limit else text[:limit]+' [excerpt; use evidence for full result]'


class ResultMemory:
    def __init__(self,ws,owner='main'):
        if not re.fullmatch(r'[\w-]{1,80}',owner):raise ValueError('Invalid result-memory owner')
        self.ws=ws
        self.path=ws.directory/('results-'+owner+'.json')
        try:self.entries=json.loads(self.path.read_text())
        except (OSError,ValueError):self.entries=[]
        if not isinstance(self.entries,list):self.entries=[]

    def observe(self,tool,args,result,evidence_id,steering_version):
        if tool not in COMPUTATIONS or result.get('ok') is not True:return
        eid=str(evidence_id)
        if not eid.isdigit():raise ValueError('Result memory requires recorded evidence')
        path=self.ws.directory/(eid+'.json')
        raw=path.read_bytes()
        if json.loads(raw)!=result:raise ValueError('Result does not match its evidence record')
        output=result.get('output','')
        if tool=='math':
            try:
                symbolic=json.loads(output)
                if isinstance(symbolic,dict) and symbolic.get('ok') is True:
                    output=json.dumps({k:symbolic[k] for k in ('result','resolved','assumptions','limitation') if k in symbolic},
                                      ensure_ascii=False,separators=(',',':'))
            except (ValueError,TypeError):pass
        signature=call_signature(tool,args)
        entry={'signature':signature,'tool':tool,'args':json.loads(signature)[1],'output':excerpt(output,850),
               'evidence_id':eid,'evidence_hash':hashlib.sha256(raw).hexdigest(),
               'steering_version':steering_version,'revision':self.ws.revision}
        self.entries=[e for e in self.entries if e.get('signature')!=signature]+[entry]
        self.entries=self.entries[-12:]
        from .harness import _atomic
        _atomic(self.path,json.dumps(self.entries,ensure_ascii=False))

    def packet(self,steering_version,limit=5000):
        rows=[];used=0
        for entry in reversed(self.entries):
            if entry.get('steering_version')!=steering_version:continue
            eid=str(entry.get('evidence_id',''))
            if not eid.isdigit():continue
            try:valid=hashlib.sha256((self.ws.directory/(eid+'.json')).read_bytes()).hexdigest()==entry['evidence_hash']
            except (OSError,KeyError):continue
            if not valid:continue
            args=entry['args']
            row={'tool':entry['tool'],'args':args if len(json.dumps(args,ensure_ascii=False))<=650 else excerpt(args,650),
                 'output':entry['output'],'evidence_id':eid}
            if entry['tool']!='math':
                # External edits, clocks and environment changes may invalidate a
                # computation even without a workspace revision. Never call it current.
                row['scope']='Historical execution, not current verification; revalidate source/environment-dependent conclusions.'
                row['source_revision_changed']=entry['revision']!=self.ws.revision
            size=len(json.dumps(row,ensure_ascii=False))
            if used+size>limit:continue
            used+=size;rows.append(row)
        return list(reversed(rows))
