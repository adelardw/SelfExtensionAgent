"""Additional families, written before evaluating the frozen answer-binding harness.

These finite tasks are a transfer diagnostic, not a public benchmark. Host acceptance
programs never enter the solving workspace. No formal proof of prose is attempted.
"""
from fractions import Fraction
from itertools import permutations
import json
import random

from .quality_cases import Case


def connectivity_probability(edges,source,target):
    total=Fraction()
    for mask in range(1<<len(edges)):
        probability=Fraction(1);adj={}
        for index,(a,b,p) in enumerate(edges):
            if mask>>index&1:
                probability*=p;adj.setdefault(a,set()).add(b);adj.setdefault(b,set()).add(a)
            else:probability*=1-p
        seen={source};frontier=[source]
        while frontier:
            for node in adj.get(frontier.pop(),set())-seen:
                seen.add(node);frontier.append(node)
        if target in seen:total+=probability
    return total


def transfer_cases(seed=407):
    rng=random.Random(seed)
    edges=[(a,b,Fraction(rng.randint(1,4),5)) for a,b in
           [(0,1),(0,2),(1,2),(1,3),(2,3),(2,4),(3,4)]]
    network=Case('network_reliability','mathematics',
        'An undirected network has vertices 0 through 4. Each edge independently works with the '
        'probability given in (u,v,p): '+repr([(a,b,str(p)) for a,b,p in edges])+'. '
        'Find the EXACT probability that a working path connects 0 to 4. Account for paths sharing '
        'edges. Explain your probability calculation and verify it with executable exact arithmetic. '
        'Do not create project files. Return JSON with probability (reduced p/q string) and explanation.',
        expected=connectivity_probability(edges,0,4),data={'edges':edges})
    matrix=[[rng.randint(-6,12) if rng.random()>.22 or row==col else None for col in range(6)] for row in range(4)]
    options=[(sum(matrix[r][c] for r,c in enumerate(cols)),cols) for cols in permutations(range(6),4)
             if all(matrix[r][c] is not None for r,c in enumerate(cols))]
    best=min(cost for cost,_ in options)
    assignment=Case('constrained_assignment','algorithms',
        'Assign each of 4 workers (rows 0..3) to a distinct one of 6 jobs (columns 0..5). '
        'Costs are '+json.dumps(matrix)+'. null means forbidden; costs may be negative. '
        'Compute the minimum total cost, the NUMBER of distinct assignments attaining it, and one '
        'optimal assignment as a list of four job indices in worker order. Describe an algorithm '
        'that also counts optima, explain its invariant and complexity in n workers and m jobs. '
        'Independently verify this finite instance by execution. Do not create project files. '
        'Return JSON with cost (integer), count (integer), assignment (list), algorithm, correctness and complexity.',
        expected=(best,sum(cost==best for cost,_ in options)),data={'matrix':matrix})
    database=Case('sqlite_atomic_batch','software',
        'Repair inventory.adjust_stock(conn, changes). The SQLite table stock has sku TEXT PRIMARY KEY '
        'and quantity INTEGER NOT NULL. changes is a list of (sku, delta) pairs; sku is a nonempty string '
        'and delta is an int (bool is invalid). Apply entries IN ORDER, rejecting an unknown sku, invalid '
        'entry, or a quantity becoming negative at any intermediate step with ValueError. On success '
        'return a dict of the final quantities for touched skus. Empty changes returns {}. '
        'The entire call is atomic on failure. Preserve an existing caller transaction: never commit '
        'or roll back unrelated caller changes. Do not mutate changes. Support arbitrary sku text '
        'including SQL punctuation. Preserve the public signature. Add regression tests and run '
        'the registered unit check.',files={
        'inventory.py':'''def adjust_stock(conn, changes):
    result={}
    for sku,delta in changes:
        conn.execute("UPDATE stock SET quantity=quantity+? WHERE sku=?",(delta,sku))
        result[sku]=conn.execute("SELECT quantity FROM stock WHERE sku=?",(sku,)).fetchone()[0]
    conn.commit()
    return result
''',
        'test_inventory.py':'''import sqlite3,unittest
from inventory import adjust_stock
class TestInventory(unittest.TestCase):
    def test_empty(self):
        with sqlite3.connect(':memory:') as conn:
            conn.execute('CREATE TABLE stock (sku TEXT PRIMARY KEY, quantity INTEGER NOT NULL)')
            self.assertEqual(adjust_stock(conn,[]),{})
'''},acceptance='''import sqlite3,copy,random
from inventory import adjust_stock
def setup():
    conn=sqlite3.connect(':memory:')
    conn.execute('CREATE TABLE stock (sku TEXT PRIMARY KEY, quantity INTEGER NOT NULL)')
    conn.executemany('INSERT INTO stock VALUES (?,?)',[('a',10),('b',2),("x'; DROP TABLE stock;--",3)])
    conn.commit()
    return conn
def state(c):return dict(c.execute('SELECT sku, quantity FROM stock'))
for changes in [[('a',-2),('missing',1)],[('a',-11),('a',11)],[('a',-2),('b',True)],
                [('a',-2),('',1)],[('a',-2),('b',1.5)],[('a',-2),('b',)], [('a',-2),(None,1)]]:
    c=setup();c.execute("UPDATE stock SET quantity=12 WHERE sku='a'")
    # For the intermediate-negative case choose a delta below the current quantity.
    if changes[0]==('a',-11):changes=[('a',-13),('a',13)]
    before=state(c);original=copy.deepcopy(changes)
    try:adjust_stock(c,changes)
    except ValueError:pass
    else:raise AssertionError(('invalid batch accepted',changes))
    assert state(c)==before,('partial change or caller rollback',changes,state(c))
    assert c.in_transaction and changes==original
    c.rollback();assert state(c)['a']==10;c.close()
c=setup();c.execute("UPDATE stock SET quantity=12 WHERE sku='a'")
changes=[('a',-2),('b',4),('a',1),("x'; DROP TABLE stock;--",-1)]
original=copy.deepcopy(changes)
assert adjust_stock(c,changes)=={'a':11,'b':6,"x'; DROP TABLE stock;--":2}
assert c.in_transaction and changes==original
c.rollback();assert state(c)=={'a':10,'b':2,"x'; DROP TABLE stock;--":3}
assert adjust_stock(c,[])=={};c.close()
c=setup();assert adjust_stock(c,[('a',-10)])=={'a':0};c.close()
print('PASS: atomicity, caller transaction preservation, ordered validation, SQL text and input preservation')
''')
    return [network,assignment,database]


def check_transfer_answer(case,answer):
    value=json.loads(answer)
    if case.name=='network_reliability':
        from math import gcd
        import re
        fraction=value.get('probability')
        if not isinstance(fraction,str) or not re.fullmatch(r'-?\d+/[1-9]\d*',fraction):return False
        n,d=map(int,fraction.split('/'))
        return (gcd(n,d)==1 and Fraction(n,d)==case.expected
                and isinstance(value.get('explanation'),str) and bool(value['explanation'].strip()))
    if case.name=='constrained_assignment':
        cols=value['assignment'];matrix=case.data['matrix']
        return (type(value['cost']) is int and type(value['count']) is int
            and (value['cost'],value['count'])==case.expected and isinstance(cols,list)
            and len(cols)==4 and all(type(c) is int and 0<=c<6 for c in cols) and len(set(cols))==4
            and all(matrix[r][c] is not None for r,c in enumerate(cols))
            and sum(matrix[r][c] for r,c in enumerate(cols))==case.expected[0]
            and all(isinstance(value.get(k),str) and value[k].strip() for k in ['algorithm','correctness','complexity']))
    raise ValueError('No host answer oracle for '+case.name)
