import asyncio
from fractions import Fraction
from itertools import combinations
import json

import pytest

from src.eval.quality_cases import cases,check_answer,exact_solve


def test_exact_linear_oracle_and_parameterized_cases():
    assert exact_solve([[2,1],[1,-1]],[5,1])==[Fraction(2),Fraction(1)]
    assert [vars(c) for c in cases(101)]==[vars(c) for c in cases(101)]
    assert cases(101)[1].expected!=cases(102)[1].expected
    c=cases(101)[0]
    good={'value':str(c.expected.numerator)+'/'+str(c.expected.denominator),'derivation':'finite expectation'}
    assert check_answer(c,json.dumps(good))
    good['value']=str(c.expected.numerator*2)+'/'+str(c.expected.denominator*2)
    assert not check_answer(c,json.dumps(good))


def test_task_declares_prose_types_required_by_host_oracle():
    from src.coding.delivery import explicit_json_fields,validate_delivery_fields
    for case in cases(23303):
        if case.domain=='software':continue
        fields=explicit_json_fields(case.task)
        prose=['derivation'] if case.domain=='mathematics' else ['algorithm','correctness','complexity']
        for key in prose:
            assert fields[key]=={'type':'string'}
            # This representation was previously rejected only by the hidden
            # host grader; it must now be visibly invalid at delivery as well.
            with pytest.raises(ValueError,match='requires string'):
                validate_delivery_fields({key:{'steps':['explanation']}},fields)


def test_finite_graph_oracles_accept_certificates_and_reject_bad_witnesses():
    c=next(c for c in cases(101) if c.name=='bipartite_cover')
    vertices=next(v for v in combinations(c.data['nodes'],c.expected)
                  if all(a in v or b in v for a,b in c.data['edges']))
    answer={'vertices':vertices,'size':c.expected,'algorithm':'matching','correctness':'cover','complexity':'polynomial'}
    assert check_answer(c,json.dumps(answer))
    answer['vertices']=[]
    assert not check_answer(c,json.dumps(answer))


REFERENCES={
'lru_cache':('cache.py','''class LRUCache:
    def __init__(self,capacity):
        if type(capacity) is not int or capacity<=0:raise ValueError()
        self.capacity=capacity;self.data={};self.order=[]
    def get(self,key,default=None):
        if key not in self.data:return default
        self.order.remove(key);self.order.append(key)
        return self.data[key]
    def put(self,key,value):
        if key in self.data:self.order.remove(key)
        self.data[key]=value;self.order.append(key)
        if len(self.order)>self.capacity:del self.data[self.order.pop(0)]
'''),
'topological_order':('ordering.py','''def toposort(graph):
    edges={a:set(xs) for a,xs in graph.items()}
    nodes=set(edges)|{b for xs in edges.values() for b in xs}
    result=[]
    while nodes:
        possible=sorted(v for v in nodes if not any(v in edges.get(a,()) for a in nodes))
        if not possible:raise ValueError('cycle')
        result.append(possible[0]);nodes.remove(possible[0])
    return result
'''),
'cpp_ranges':('ranges.cpp','''#include "ranges.hpp"
#include <algorithm>
#include <stdexcept>
std::vector<Range> merge_ranges(const std::vector<Range>& input){
    auto v=input; for(auto r:v)if(r.first>=r.second)throw std::invalid_argument("range");
    std::sort(v.begin(),v.end());std::vector<Range> out;
    for(auto r:v){
        if(out.empty()||out.back().second<r.first)out.push_back(r);
        else out.back().second=std::max(out.back().second,r.second);
    }return out;
}
''')}


@pytest.mark.parametrize('name',list(REFERENCES))
def test_withheld_software_oracle_rejects_bug_and_accepts_reference(tmp_path,name):
    from src.coding.harness import Workspace
    from src.coding.verifier import isolated_probe
    from src.runtime import hitl
    hitl.set_work_mode('auto')
    case=next(c for c in cases(101) if c.name==name)
    for path,source in case.files.items():(tmp_path/path).write_text(source)
    ws=Workspace(tmp_path,tmp_path/'.sea',{})
    bad=asyncio.run(isolated_probe(ws,case.acceptance))
    assert not bad['ok'],bad
    path,source=REFERENCES[name];(tmp_path/path).write_text(source)
    good=asyncio.run(isolated_probe(ws,case.acceptance))
    assert good['ok'],good
