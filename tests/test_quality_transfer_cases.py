import asyncio
from fractions import Fraction
from itertools import permutations
import json

from src.eval.quality_transfer_cases import transfer_cases,check_transfer_answer,connectivity_probability


def test_network_oracle_against_series_parallel_identities():
    p,q=Fraction(1,3),Fraction(2,5)
    assert connectivity_probability([(0,1,p),(1,2,q)],0,2)==p*q
    assert connectivity_probability([(0,1,p),(0,1,q)],0,1)==1-(1-p)*(1-q)
    c=transfer_cases()[0]
    good={'probability':f'{c.expected.numerator}/{c.expected.denominator}','explanation':'exact enumeration'}
    assert check_transfer_answer(c,json.dumps(good))
    good['probability']='0/1';assert not check_transfer_answer(c,json.dumps(good))


def test_assignment_oracle_validates_witness_and_integer_types():
    c=transfer_cases()[1];m=c.data['matrix']
    witness=next(cols for cols in permutations(range(6),4) if all(m[r][v] is not None for r,v in enumerate(cols))
                 and sum(m[r][v] for r,v in enumerate(cols))==c.expected[0])
    good=dict(cost=c.expected[0],count=c.expected[1],assignment=witness,algorithm='subset DP',correctness='induction',complexity='exponential')
    assert check_transfer_answer(c,json.dumps(good))
    good['assignment']=[0]*4;assert not check_transfer_answer(c,json.dumps(good))
    assert transfer_cases(407)[0].expected!=transfer_cases(408)[0].expected


def test_database_acceptance_detects_partial_commits_and_accepts_savepoint_reference(tmp_path):
    from src.coding.harness import Workspace
    from src.coding.verifier import isolated_probe
    from src.runtime import hitl
    hitl.set_work_mode('auto')
    c=transfer_cases()[2]
    for name,source in c.files.items():(tmp_path/name).write_text(source)
    ws=Workspace(tmp_path,tmp_path/'.sea',{})
    assert not asyncio.run(isolated_probe(ws,c.acceptance))['ok']
    (tmp_path/'inventory.py').write_text('''def adjust_stock(conn,changes):
    import uuid
    name='stock_'+uuid.uuid4().hex
    conn.execute('SAVEPOINT '+name)
    result={}
    try:
        for item in changes:
            if not isinstance(item,(tuple,list)) or len(item)!=2:raise ValueError('pair')
            sku,delta=item
            if not isinstance(sku,str) or not sku or type(delta) is not int:raise ValueError('type')
            row=conn.execute('SELECT quantity FROM stock WHERE sku=?',(sku,)).fetchone()
            if row is None or row[0]+delta<0:raise ValueError('stock')
            result[sku]=row[0]+delta
            conn.execute('UPDATE stock SET quantity=? WHERE sku=?',(result[sku],sku))
    except Exception:
        conn.execute('ROLLBACK TO '+name);conn.execute('RELEASE '+name);raise
    conn.execute('RELEASE '+name)
    return result
''')
    result=asyncio.run(isolated_probe(ws,c.acceptance))
    assert result['ok'],result
