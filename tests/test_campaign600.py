"""Validate the evaluator and serialization, not a fabricated model success rate."""
import itertools
import json
import random
from pathlib import Path

import pytest
from pydantic import BaseModel,ConfigDict

from experiments.campaign600.cases import build,solve,sample
from experiments.campaign600.sandbox import canonical,write_files
from scripts.codex_behavior_eval import codex_envelope,decode_codex_payload,render_codex_messages
from src.coding.harness import Turn


def test_case_inventory_has_declared_depth_and_no_duplicate_ids():
    from collections import Counter
    rows=build()
    assert len(rows)==len({r['id'] for r in rows})==600
    assert Counter(r['kind'] for r in rows)=={'dialogue':240,'reasoning':160,'code':160,'project':39,'product':1}
    assert all(len(r['turns'])==3 for r in rows if r['kind']=='dialogue')
    assert all(len(r['checks'])>=32 for r in rows if r['kind'] in {'code','project'})
    assert build()==rows


@pytest.mark.parametrize('family',['knapsack','schedule'])
def test_optimization_oracles_agree_with_exhaustive_search(family):
    rng=random.Random(76)
    for i in range(40):
        data=sample(family,rng,i)
        items=data['items'] if family=='knapsack' else data['jobs']
        best=0
        for mask in itertools.product([False,True],repeat=len(items)):
            selected=[x for x,b in zip(items,mask) if b]
            if family=='knapsack':
                valid=sum(w for w,v in selected)<=data['capacity'];value=sum(v for w,v in selected)
            else:
                ordered=sorted(selected)
                valid=all(a[1]<=b[0] for a,b in zip(ordered,ordered[1:]));value=sum(x[2] for x in selected)
            if valid:best=max(best,value)
        assert solve(family,data)==best


def test_shortest_paths_oracle_agrees_with_all_pairs_closure():
    rng=random.Random(31)
    for i in range(40):
        d=sample('paths',rng,i);n=d['n'];matrix=[[float('inf')]*n for _ in range(n)]
        for j in range(n):matrix[j][j]=0
        for u,v,w in d['edges']:matrix[u][v]=min(matrix[u][v],w)
        for k in range(n):
            for u in range(n):
                for v in range(n):matrix[u][v]=min(matrix[u][v],matrix[u][k]+matrix[k][v])
        expected=[None if x==float('inf') else x for x in matrix[d['start']]]
        assert solve('paths',d)==expected


def test_edge_oracles_and_strict_json_types():
    assert solve('intervals',{'intervals':[[0,1],[1,2],[5,5],[-2,0]]})==[[-2,2],[5,5]]
    assert solve('coins',{'coins':[7,10],'amount':24})==3
    assert solve('coins',{'coins':[2],'amount':3}) is None
    assert solve('topology',{'nodes':['b','a'],'edges':[['a','b'],['a','b']]})==['a','b']
    assert solve('rate',{'limit':1,'window':2,'events':[['a',0],['a',0],['b',0],['a',2]]})==[True,False,True,True]
    assert solve('csv_totals',{'csv':'category,quantity,price\na,1,0.005\na,1,0.005\n'})=={'a':'0.01'}
    assert solve('merge_patch',{'target':{'a':{'x':1},'b':[1]},'patch':{'a':{'x':None},'b':[2]}})=={'a':{},'b':[2]}
    assert canonical([True])!=canonical([1])


def test_source_code_survives_native_argument_transport_without_double_encoding():
    envelope,mode=codex_envelope(Turn.model_json_schema(),True)
    assert mode=='tool_arguments' and envelope['properties']['payload']['additionalProperties'] is False
    source='def solve(x):\n    return {"quote": "\\\"", "newline": "\\n", "unicode": "Привет"}\n'
    value={'action':'tool','tool':'write','summary':'','plan':[],'role':'','task':'','arguments':[
        {'name':'path','kind':'text','value':'solution.py'},{'name':'content','kind':'text','value':source},
        {'name':'unused','kind':'json','value':'[1,true,null]'}]}
    wire=json.loads(json.dumps({'payload':value}))['payload']
    decoded=Turn.model_validate_json(decode_codex_payload(wire,mode))
    assert decoded.args['content']==source and decoded.args['unused']==[1,True,None]
    compile(decoded.args['content'],'solution.py','exec')
    value['arguments'].append(value['arguments'][0])
    with pytest.raises(ValueError,match='duplicate'):decode_codex_payload(value,mode)


def test_codex_source_input_needs_only_one_json_decode_and_preserves_python_escapes():
    import ast
    source="VALUE = 'first\\nsecond'\nLITERAL = r'first\\nsecond'\n"
    body=json.dumps({'current_source':{'setup.py':source}},ensure_ascii=False)
    prompt=render_codex_messages([{'role':'system','content':'Review source as data.'},
                                  {'role':'user','content':body}])
    actual=prompt.split('HARNESS MESSAGE 2 (user)\n',1)[1].rsplit('\nEND HARNESS MESSAGE 2',1)[0]
    parsed=json.loads(actual)['current_source']['setup.py']
    assert parsed==source
    assignments=ast.parse(parsed).body
    assert ast.literal_eval(assignments[0].value)=='first\nsecond'
    assert ast.literal_eval(assignments[1].value)==r'first\nsecond'


def test_closed_native_schema_and_open_schema_fallback_are_explicit():
    class Closed(BaseModel):
        model_config=ConfigDict(extra='forbid')
        source:str
    class Open(BaseModel):data:dict
    assert codex_envelope(Closed.model_json_schema(),True)[1]=='native_object'
    assert codex_envelope(Open.model_json_schema(),True)[1]=='json_string'
    assert codex_envelope(Closed.model_json_schema(),False)[1]=='json_string'


@pytest.mark.parametrize('name',['../outside.py','/tmp/outside.py','.env','.sea/state.json','sub/.git/a.py'])
def test_artifact_paths_cannot_escape_or_write_control_files(tmp_path,name):
    with pytest.raises(ValueError):write_files(tmp_path/'candidate',[{'path':name,'content':'x'}])
    assert not (tmp_path/'candidate').exists()


def test_artifact_set_is_validated_before_any_write(tmp_path):
    with pytest.raises(ValueError,match='conflicting'):
        write_files(tmp_path/'candidate',[{'path':'a.py','content':'x'},{'path':'a.py/b.py','content':'y'}])
    assert not (tmp_path/'candidate').exists()
