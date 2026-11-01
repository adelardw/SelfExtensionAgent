"""Observe the actual failing unittest, without asking a model to recreate its steps.

This is a bounded diagnostic experiment, never an acceptance oracle. Only a
locally present test named by a registered unittest failure is selected.
"""
from __future__ import annotations

import json
import re


def target_test(argv, output):
    if not isinstance(argv,list) or not any(argv[i:i+2]==['-m','unittest'] for i in range(len(argv)-1)):
        return None
    match=re.search(r'^(?:FAIL|ERROR): (\w+) \(((?:\w+\.){2,}\w+)\)\s*$',output,re.M)
    if not match:return None
    name=match[2]
    if name.split('.')[-1]!=match[1] or len(name)>240:return None
    return name,'.'.join(name.split('.')[:-2]).replace('.','/')+'.py'


# Runs only in an existing disposable, network-restricted probe snapshot.
# No custom repr, iteration protocols or properties are evaluated for snapshots.
TRACE_PROGRAM=r'''
import collections, inspect, itertools, json, sys, types, unittest
from pathlib import Path
root=Path.cwd().resolve()
events=collections.deque(maxlen=32)
pending={}
count=0
observations_lost=0
def sensitive(name):
    return type(name) is str and name.lower() in {'api_key','apikey','access_token','refresh_token',
        'password','passwd','authorization','private_key','client_secret','secret'}
def sample(value, depth=3, budget=None):
    if budget is None:budget=[64]
    budget[0]-=1
    if budget[0]<0 or depth<0:return '<omitted>'
    kind=type(value)
    if value is None or kind in (bool,int,float):
        if kind is int and value.bit_length()>512:return '<large integer>'
        if kind is float and not (-float('inf')<value<float('inf')):return '<nonfinite float>'
        return value
    if kind is str:return value[:120]
    if kind in (list,tuple):
        return [sample(item,depth-1,budget) for item in value[:10]]+(['<more>'] if len(value)>10 else [])
    if kind is dict:
        items=list(itertools.islice(value.items(),10))
        result={str(key)[:60]:('<redacted>' if sensitive(key) else sample(item,depth-1,budget))
            for key,item in items if type(key) in (str,int,bool)}
        if len(value)>10:result['<omitted>']=len(value)-10
        return result
    descriptor=inspect.getattr_static(value,'__dict__',None)
    if type(descriptor) is types.GetSetDescriptorType:
        attributes=descriptor.__get__(value,kind)
        if type(attributes) is dict:
            return sample(attributes,depth-1,budget)
    return '<'+type.__getattribute__(kind,'__name__')+'>'
def location(frame):
    if frame is None:return None
    try:
        path=Path(frame.f_code.co_filename).resolve().relative_to(root)
        if any(p.startswith('.') for p in path.parts):return None
        return str(path)+':'+str(frame.f_lineno)
    except (ValueError,OSError):return None
def trace(frame,event,arg):
    global count,observations_lost
    try:
        if event=='line':return trace
        at=location(frame)
        if at is None:return None
        if event=='call':
            if isinstance(frame.f_locals.get('self'),unittest.TestCase):return trace
            if len(pending)>=128:
                observations_lost+=1;return None
            pending[id(frame)]={'at':location(frame.f_back),'call':frame.f_code.co_name,
                'in':{k:('<redacted>' if sensitive(k) else sample(v))
                    for k,v in itertools.islice(frame.f_locals.items(),8) if k not in ('self','cls')}}
        elif event=='return' and id(frame) in pending:
            record=pending.pop(id(frame))
            record['out']=sample(arg)
            if 'self' in frame.f_locals:record['state_after']=sample(frame.f_locals['self'])
            events.append(record);count+=1
        elif event=='exception' and isinstance(frame.f_locals.get('self'),unittest.TestCase):
            events.append({'at':at,'exception':type.__getattribute__(arg[0],'__name__'),
                'locals':{k:('<redacted>' if sensitive(k) else sample(v))
                    for k,v in itertools.islice(frame.f_locals.items(),8) if k!='self'}})
            count+=1
    except Exception:
        observations_lost+=1
    return trace
suite=unittest.defaultTestLoader.loadTestsFromName(target)
result=unittest.TestResult()
previous=sys.gettrace()
try:
    sys.settrace(trace)
    suite.run(result)
finally:
    sys.settrace(previous)
failures=[{'test':test.id(),'kind':kind,'message':text.strip().splitlines()[-1][:500]}
    for kind,items in [('failure',result.failures),('error',result.errors)] for test,text in items]
rows=list(events)
packet={'test':target,'tests_run':result.testsRun,'failures':failures[:2],
    'observed_test_failed':any(f['test']==target for f in failures),
    'skipped':len(result.skipped),'events':rows,'omitted_events':max(0,count-len(rows))+observations_lost,
    'scope':'Actual test operations; expected assertions remain hypotheses, not a specification.'}
while len(json.dumps(packet,ensure_ascii=False))>5500 and rows:
    rows.pop(0);packet['omitted_events']+=1
print('SEA_TEST_TRACE '+json.dumps(packet,ensure_ascii=False,separators=(',',':')))
'''


async def trace_failed_test(ws,check,failure):
    """Return a recorded, compact trace or None when this runner is unsupported."""
    target=target_test(ws.checks.get(check),str(failure.get('output','')))
    if target is None or not await ws.file_exists(target[1]):return None
    from .verifier import isolated_probe
    program='target='+json.dumps(target[0])+'\n'+TRACE_PROGRAM
    result=await isolated_probe(ws,program,preserve_source=True)
    # Keep the complete execution in evidence; do not forward the tracing program
    # or unrelated stdout into the model context.
    result.update(kind='failed_test_trace',critic_source=True,test=target[0])
    eid=ws.record(result)
    packet=None
    if result.get('ok') and not result.get('stale'):
        for line in str(result.get('output','')).splitlines():
            if not line.startswith('SEA_TEST_TRACE '):continue
            try:value=json.loads(line.removeprefix('SEA_TEST_TRACE '))
            except ValueError:continue
            if isinstance(value,dict) and value.get('test')==target[0] and len(line)<7000:packet=value
    return {'evidence_id':eid,'ok':packet is not None,'observation':packet,
        'stale':bool(result.get('stale')),
        'limitation':'Selected unittest only, in a disposable copy. Not a passing registered check or proof.',
        **({'error':str(result.get('output',''))[-600:]} if packet is None else {})}
