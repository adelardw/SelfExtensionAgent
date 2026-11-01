"""Parameterized development tasks and host-only acceptance oracles.

These are not public benchmark scores or formal proof checks. Case source and
oracles are never copied to the agent's workspace; only task and initial files are.
"""
from __future__ import annotations

from dataclasses import dataclass,field
from fractions import Fraction
from itertools import combinations
import json
from math import comb,gcd
import random
import re


@dataclass
class Case:
    name:str
    domain:str
    task:str
    files:dict[str,str]=field(default_factory=dict)
    acceptance:str=''
    expected:object=None
    data:dict=field(default_factory=dict)


def exact_solve(matrix,values):
    a=[[Fraction(x) for x in row]+[Fraction(y)] for row,y in zip(matrix,values)]
    for col in range(len(a)):
        pivot=next(i for i in range(col,len(a)) if a[i][col])
        a[col],a[pivot]=a[pivot],a[col]
        scale=a[col][col];a[col]=[v/scale for v in a[col]]
        for i in range(len(a)):
            if i!=col:
                scale=a[i][col];a[i]=[x-scale*y for x,y in zip(a[i],a[col])]
    return [row[-1] for row in a]


def cases(seed=101):
    rng=random.Random(seed)
    result=[]
    red,blue,draw=rng.randint(4,8),rng.randint(3,6),rng.randint(3,5)
    weights={r:comb(red,r)*comb(blue,draw-r) for r in range(draw+1)
             if r<=red and 0<=draw-r<=blue and draw-r>=1}
    expected=Fraction(sum(r*w for r,w in weights.items()),sum(weights.values()))
    result.append(Case('conditional_urn','mathematics',
        f'An urn contains {red} red and {blue} blue balls. Draw {draw} balls uniformly without replacement. '
        'Find the exact conditional expectation of the number of red balls, GIVEN at least one blue '
        'ball was drawn. Explain the conditioning and check with exact executable arithmetic. '
        'Do not create project files. Return JSON with value (reduced fraction p/q) and derivation (string).',expected=expected))
    p=[[Fraction() for _ in range(5)] for _ in range(5)]
    for i in range(4):
        forward,back=rng.randint(2,5),rng.randint(1,3)
        denominator=forward+back+1
        p[i][i+1]+=Fraction(forward,denominator)
        p[i][max(0,i-1)]+=Fraction(back,denominator)
        p[i][i]+=Fraction(1,denominator)
    p[4][4]=Fraction(1)
    expected=exact_solve([[int(i==j)-p[i][j] for j in range(4)] for i in range(4)],[1]*4)[0]
    result.append(Case('markov_hitting','mathematics',
        'A discrete-time Markov chain has states 0..4 and transition matrix (rows are FROM states): '
        +json.dumps([[str(v) for v in row] for row in p])+'. State 4 is absorbing. Starting from state 0, '
        'find the exact expected number of transitions until FIRST reaching 4. Derive the equations, '
        'justify that the expectation is finite and check with exact executable arithmetic. '
        'Do not create project files. Return JSON with value (reduced fraction p/q) and derivation (string).',expected=expected))
    left=[f'L{i}' for i in range(5)];right=[f'R{i}' for i in range(4)]
    edges=[(a,b) for a in left for b in right if rng.random()<.48]
    nodes=left+right
    cover=next(xs for n in range(len(nodes)+1) for xs in combinations(nodes,n)
               if all(a in xs or b in xs for a,b in edges))
    result.append(Case('bipartite_cover','algorithms',
        'Find a minimum vertex cover of this bipartite graph. Left vertices: '+json.dumps(left)+
        '; right vertices: '+json.dumps(right)+'; edges: '+json.dumps(edges)+
        '. Give a polynomial-time general algorithm and correctness argument. Independently check '
        'this finite instance by exhaustive enumeration. Do not create project files. Return JSON '
        'with vertices (list of labels), size (integer), algorithm (string), correctness (string), complexity (string).',
        expected=len(cover),data={'nodes':nodes,'edges':edges}))
    edges=[(i,j,rng.randint(-5,9)) for i in range(6) for j in range(i+1,6)
           if j==i+1 or rng.random()<.45]
    paths=[]
    def walk(path,cost):
        if path[-1]==5: paths.append((cost,path));return
        for a,b,w in edges:
            if a==path[-1]:walk(path+[b],cost+w)
    walk([0],0)
    best=min(c for c,_ in paths);count=sum(c==best for c,_ in paths)
    result.append(Case('negative_dag','algorithms',
        'For a directed acyclic graph on vertices 0..5, weighted edges are '+json.dumps(edges)+
        '. Find the shortest distance from 0 to 5, the number of DISTINCT shortest vertex paths, '
        'and one shortest path. Explain a general algorithm that handles negative edge weights and '
        'give its invariant and complexity. Independently verify the finite result by enumerating paths. '
        'Do not create project files. Return JSON with distance (integer), count (integer), path '
        '(list of integers), algorithm (string), correctness (string), complexity (string).',expected=(best,count),data={'edges':edges}))
    software_suffix=' Preserve public signatures. Add focused regression tests and run the registered unit check. Do not weaken existing tests.'
    result.append(Case('lru_cache','software',
        'Fix cache.py LRUCache. Capacity must be a positive integer (reject bool/non-integers with ValueError). '
        'put(key,value) inserts or updates a hashable key; successful get(key,default=None) and updates '
        'make that key most recently used. Missing get returns default WITHOUT changing cache order. '
        'Evict the least recently used key only when inserting beyond capacity. Values may be None.'+software_suffix,
        files={'cache.py':'''class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.data = {}
    def get(self, key, default=None):
        return self.data.get(key, default)
    def put(self, key, value):
        if len(self.data) >= self.capacity:
            del self.data[next(iter(self.data))]
        self.data[key] = value
''','test_cache.py':'''import unittest
from cache import LRUCache
class TestCache(unittest.TestCase):
    def test_insert(self):
        c=LRUCache(2); c.put('a',1)
        self.assertEqual(c.get('a'),1)
'''},acceptance=f'''import random
from collections import OrderedDict
from cache import LRUCache
for bad in (0,-1,True,False,1.5,'2',None):
    try: LRUCache(bad)
    except ValueError: pass
    else: raise AssertionError(('invalid capacity',repr(bad)))
rng=random.Random({seed+997})
for capacity in range(1,5):
    c=LRUCache(capacity);reference=OrderedDict()
    for _ in range(120):
        key=rng.randrange(7)
        if rng.random()<.6:
            value=rng.choice([None,0,1,'x']);c.put(key,value)
            reference[key]=value;reference.move_to_end(key)
            if len(reference)>capacity:reference.popitem(last=False)
        else:
            expected=reference.get(key,'missing');actual=c.get(key,'missing')
            assert actual==expected,(capacity,key,actual,expected)
            if key in reference:reference.move_to_end(key)
print('PASS: independent stateful differential LRU checks')
'''))
    result.append(Case('topological_order','software',
        'Fix ordering.toposort(graph). Input maps string node labels to lists of directed outgoing '
        'neighbors. Include labels appearing only as neighbors; duplicate edges count once. Return '
        'the lexicographically smallest valid topological ordering. Raise ValueError on every cycle, '
        'including self-loops. Empty graph returns []. Do not mutate the graph or its lists.'+software_suffix,
        files={'ordering.py':'def toposort(graph):\n    return sorted(graph)\n',
        'test_ordering.py':'''import unittest
from ordering import toposort
class TestOrdering(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(toposort({}),[])
'''},acceptance=f'''from itertools import permutations
import copy,random
from ordering import toposort
rng=random.Random({seed+1013})
graphs=[{{'a':['a']}},{{'a':['b','b'],'b':[]}},{{'z':['a']}},{{'a':['b'],'b':['a']}},{{}}]
for _ in range(18):
    labels=['a','b','c','d','e'];rng.shuffle(labels)
    graphs.append({{a:[b for b in labels if rng.random()<.22] for a in labels[:-1]}})
for graph in graphs:
    before=copy.deepcopy(graph);nodes=set(graph)|{{v for xs in graph.values() for v in xs}}
    valid=[p for p in permutations(sorted(nodes)) if all(p.index(a)<p.index(b) for a,xs in graph.items() for b in xs)]
    try:result=toposort(graph)
    except ValueError:assert not valid,('rejected DAG',graph)
    else:assert valid and result==list(min(valid)),(graph,result,valid[:1])
    assert graph==before,'input mutated'
print('PASS: exhaustive reference ordering, cycles, duplicates and mutation')
'''))
    cpp_test='''#include "ranges.hpp"
#include <cassert>
int main(){ auto r=merge_ranges({{1,3},{2,4}}); assert(r.size()==1 && r[0]==Range(1,4)); }
'''
    runner='''import os, pathlib, subprocess, unittest
class TestCpp(unittest.TestCase):
    def test_cpp(self):
        binary=str(pathlib.Path(os.environ['SEA_SCRATCH'])/'ranges-test')
        subprocess.run(['clang++','-std=c++20','ranges.cpp','test_ranges.cpp','-o',binary],check=True)
        subprocess.run([binary],check=True)
'''
    result.append(Case('cpp_ranges','software',
        'Fix the C++20 merge_ranges API in ranges.cpp. Input is a vector of half-open integer ranges '
        '[start,end). Reject ANY start>=end with std::invalid_argument. Return the sorted disjoint union, '
        'merging touching ranges too. Handle the entire long long range without overflow. Preserve the '
        'input and public signature. Add regression cases to test_ranges.cpp and run the registered unit check.',
        files={'ranges.hpp':'#pragma once\n#include <vector>\n#include <utility>\nusing Range=std::pair<long long,long long>;\nstd::vector<Range> merge_ranges(const std::vector<Range>& input);\n',
        'ranges.cpp':'''#include "ranges.hpp"
#include <algorithm>
std::vector<Range> merge_ranges(const std::vector<Range>& input){
    auto ranges=input; std::sort(ranges.begin(),ranges.end());
    std::vector<Range> out;
    for(auto r:ranges){
        if(!out.empty() && r.first<out.back().second) out.back().second=r.second;
        else out.push_back(r);
    }
    return out;
}
''','test_ranges.cpp':cpp_test,'test_runner.py':runner},
        acceptance='''import os,pathlib,subprocess
scratch=pathlib.Path(os.environ['SEA_SCRATCH'])
source=scratch/'accept.cpp'
source.write_text(r"""#include "ranges.hpp"
#include <cassert>
#include <climits>
#include <stdexcept>
int main(){
    auto input=std::vector<Range>{{5,9},{1,10},{3,4},{10,12},{-5,-1},{-1,1}};
    auto original=input;
    assert(merge_ranges(input)==std::vector<Range>({{-5,12}})); assert(input==original);
    assert(merge_ranges({}).empty());
    assert(merge_ranges({{LLONG_MIN,0},{0,LLONG_MAX}})==std::vector<Range>({{LLONG_MIN,LLONG_MAX}}));
    for(auto r:std::vector<Range>{{0,0},{3,1},{LLONG_MAX,LLONG_MIN}}){
        bool threw=false; try{merge_ranges({{1,5},r});}catch(const std::invalid_argument&){threw=true;}
        assert(threw);
    }
}
""")
binary=str(scratch/'accept')
subprocess.run(['clang++','-std=c++20','-I.','ranges.cpp',str(source),'-o',binary],check=True)
subprocess.run([binary],check=True)
print('PASS: independent C++ boundaries, nested ranges, invalid input and preservation')
'''))
    return result


def check_answer(case,answer):
    value=json.loads(answer)
    if case.domain=='mathematics':
        if not isinstance(value['value'],str) or not re.fullmatch(r'-?\d+/[1-9]\d*',value['value']):return False
        n,d=map(int,value['value'].split('/'))
        if gcd(n,d)!=1:return False
        return Fraction(value['value'])==case.expected and isinstance(value.get('derivation'),str) and bool(value['derivation'].strip())
    if not all(isinstance(value.get(k),str) and value[k].strip() for k in ('algorithm','correctness','complexity')):return False
    if case.name=='bipartite_cover':
        vertices=value['vertices']
        return (type(value['size']) is int and len(set(vertices))==len(vertices)==value['size']==case.expected and
                set(vertices)<=set(case.data['nodes']) and
                all(a in vertices or b in vertices for a,b in case.data['edges']))
    if case.name=='negative_dag':
        path=value['path'];weights={(a,b):w for a,b,w in case.data['edges']}
        return (type(value['distance']) is int and type(value['count']) is int and
                all(type(v) is int for v in path) and path[0]==0 and path[-1]==5 and len(set(path))==len(path) and
                (value['distance'],value['count'])==case.expected and
                sum(weights[a,b] for a,b in zip(path,path[1:]))==case.expected[0])
    raise ValueError('No host answer oracle for '+case.name)
