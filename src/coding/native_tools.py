"""Compile the same bounded Turn protocol to native function calls, without executing them."""
from __future__ import annotations

import json
from .prompts import HINTS

STRING={'type':'string'}
INTEGER={'type':'integer'}
BOOLEAN={'type':'boolean'}
STRINGS={'type':'array','items':STRING}


def parameters(name):
    if name=='edit':
        replacement={'type':'object','properties':{'path':STRING,'old':STRING,'new':STRING},
                     'required':['path','old','new'],'additionalProperties':False}
        return {'type':'object','properties':{**replacement['properties'],
                    'changes':{'type':'array','items':replacement,'minItems':1,'maxItems':8}},
                'anyOf':[{'required':['path','old','new']},{'required':['changes']}],
                'additionalProperties':False}
    fields={
        'ask_user':({'question':STRING,'why':STRING,'options':STRINGS,'answer_id':STRING},['question','why']),
        'inspect_artifact':({'path':STRING},['path']),
        'security_audit':({},[]),
        'tree':({},[]),
        'read':({'path':STRING,'start':INTEGER,'end':INTEGER},['path']),
        'write':({'path':STRING,'content':STRING},['path','content']),
        'search':({'pattern':STRING,'path':STRING},['pattern']),
        'grep':({'pattern':STRING,'path':STRING,'regex':BOOLEAN},['pattern']),
        'check':({'name':STRING},['name']),
        'run':({'argv':STRINGS,'network':BOOLEAN,'project_write':BOOLEAN},['argv']),
        'register_check':({'name':STRING,'argv':STRINGS},['name','argv']),
        'experiment':({'language':{'type':'string','enum':['python','cpp']},'code':STRING},['language','code']),
        'math':({'op':{'type':'string','enum':['simplify','factor','expand','solve','diff','integrate','limit','sum','identity']},
                 'expr':{**STRING,'description':'Arithmetic expression only, without =. For solve this is the left-hand side.'},
                 'var':{**STRING,'description':'One variable name, e.g. x.'},
                 'rhs':{**STRING,'description':'Right-hand expression for solve/identity, defaults to 0. For systems use experiment.'},'point':STRING,
                 'bounds':{'type':'array','items':{'anyOf':[STRING,{'type':'number'}]},'minItems':2,'maxItems':2},
                 'assumptions':{'type':'object','additionalProperties':{'type':'object','additionalProperties':BOOLEAN}}},['op','expr']),
        'evidence':({'id':STRING,'start':INTEGER,'end':INTEGER},['id']),
        'web':({'query':STRING},['query']),
        'fetch':({'url':STRING,'find':STRING},['url']),
        'resume_agent':({'id':STRING,'instructions':STRING},['id']),
        'context':({'action':{'type':'string','enum':['next','back','reset','query']},'query':STRING,
                    'tools':{'type':'array','items':{'type':'string','enum':['compute','files','develop','coordinate','memory','research','all']}}},[]),
        'status':({},[]),
        'diff':({'path':STRING,'stat':BOOLEAN},[]),
        'patch':({'id':STRING,'ids':STRINGS},[]),
        'accept':({'id':STRING,'ids':STRINGS},[]),
        'reject':({'id':STRING,'reason':STRING},['id','reason']),
    }
    if name not in fields:
        return {'type':'object','properties':{},'additionalProperties':True}
    props,required=fields[name]
    return {'type':'object','properties':props,'required':required,'additionalProperties':False}


def compile_tools(schema):
    wire=schema.model_json_schema()
    if 'oneOf' not in wire: return []
    definitions=[]
    for branch in wire['oneOf']:
        props=branch['properties'];action=props['action']['const']
        if action=='tool':
            for name in props['tool']['enum']:
                args=parameters(name)
                if name=='check' and props['args'].get('properties',{}).get('name',{}).get('enum'):
                    args=props['args']
                definitions.append({'type':'function','function':{'name':name,
                    'description':HINTS.get(name,name),'parameters':args}})
        else:
            descriptions={'plan':'Record a short executable plan when needed to coordinate work. A plan does not grant execution permissions.',
                          'delegate':'Assign a narrower independent subtask; not a copy of the parent task.',
                          'finish':'Propose the final answer. Runtime verification still decides completion.'}
            definitions.append({'type':'function','function':{'name':action,'description':descriptions[action],
                'parameters':{'type':'object','properties':{k:v for k,v in props.items() if k!='action'},
                              'required':[k for k in branch['required'] if k!='action'],'additionalProperties':False}}})
    return definitions


def decode_call(calls,definitions):
    if len(calls)!=1: raise ValueError('Return exactly ONE native function call per step; none executed')
    call=next(iter(calls.values()));name=call['name']
    allowed={d['function']['name'] for d in definitions}
    if name not in allowed: raise ValueError('Function outside this step capability set: '+name)
    args=json.loads(call['arguments'])
    if not isinstance(args,dict): raise ValueError('Function arguments must be an object')
    if name in {'plan','delegate','finish'}: return {**args,'action':name}
    return {'action':'tool','tool':name,'args':args}


def decode_result(calls):
    """Decode a typed result, never an executable actor action."""
    if len(calls)!=1:raise ValueError('Return exactly ONE submit_result call')
    call=next(iter(calls.values()))
    if call['name']!='submit_result':raise ValueError('Only submit_result is allowed for this structured response')
    value=json.loads(call['arguments'])
    if not isinstance(value,dict):raise ValueError('submit_result arguments must be an object')
    return value


def collect_calls(target,chunks):
    for chunk in chunks:
        index=chunk.get('index',0)
        if not isinstance(index,int) or not 0<=index<8: raise ValueError('Invalid native call index')
        value=target.setdefault(index,{'name':'','arguments':''})
        if chunk.get('id'):
            value['id']=value.get('id','')+chunk['id']
        function=chunk.get('function',{})
        value['name']+=function.get('name') or ''
        value['arguments']+=function.get('arguments') or ''
        if len(value['arguments'])>100000: raise ValueError('Native arguments exceed bounded action size')


def observation_messages(messages,definitions):
    """Expose two actual recorded observations through the native tool protocol.

    No generated examples, replayed executions or inferred successes. The final
    state packet remains authoritative for revisions, constraints and ownership.
    """
    import copy
    try: packet=json.loads(messages[-1]['content'])
    except (ValueError,TypeError): return messages,[]
    if not isinstance(packet,dict): return messages,[]
    allowed={d['function']['name'] for d in definitions}&{'read','tree','search','grep','math','check'}
    memory=packet.get('working_memory',{})
    records=packet.get('recent',memory.get('observations',[]))
    reads={**packet.get('inspected_content',{}),**memory.get('inspected_content',{})}
    completed={r['evidence_id']:r for r in packet.get('completed_results',[])}
    selected=[]
    for index,item in enumerate(records):
        if item.get('tool') not in allowed or not isinstance(item.get('args'),dict): continue
        args=json.dumps(item['args'],ensure_ascii=False)
        if item.get('args_truncated') or len(args)>1600 or '[omitted;' in args: continue
        if not item.get('evidence_id') or not isinstance(item.get('ok'),bool): continue
        output=item.get('output','')
        if isinstance(output,str) and output.startswith('Result is included in completed_results;'):
            retained=completed.get(item['evidence_id'])
            if retained is None:continue
            output=retained['output']
        if isinstance(output,str) and output.startswith('Source already included in inspected_content['):
            path=item['args'].get('path')
            if path not in reads: continue
            output=reads[path]
        selected.append((index,item,args,output))
    selected=selected[-2:]
    if not selected:return messages,[]
    history=[]
    for index,item,args,output in selected:
        call_id='observation_'+str(index)
        history.extend([
            {'role':'assistant','content':None,'tool_calls':[{'id':call_id,'type':'function',
                'function':{'name':item['tool'],'arguments':args}}]},
            {'role':'tool','tool_call_id':call_id,'content':json.dumps({
                'ok':item['ok'],'output':output,'evidence_id':item['evidence_id']},ensure_ascii=False)}])
    keep=[item for i,item in enumerate(records) if i not in {s[0] for s in selected}]
    if 'recent' in packet:packet['recent']=keep
    else:memory['observations']=keep
    result=copy.deepcopy(messages[:-1])+history+[{'role':'user','content':json.dumps(packet,ensure_ascii=False)}]
    return result,[str(item['evidence_id']) for _,item,_,_ in selected]
