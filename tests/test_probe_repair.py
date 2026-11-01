import asyncio
import json

import pytest

from src.coding.critic import ContextCritic
from src.coding.probe_repair import ProbeEdit,patch_probe,repairable_failure
from test_active_verification import setup


def test_probe_repair_preserves_assertions_and_imports_and_requires_unique_edit():
    source='import sqlite3\nconn=sqlite3.connect(":memory:")\nconn.execute("BEGIN")\nassert 2+3==5\n'
    fixed=patch_probe(source,[ProbeEdit(old='conn.execute("BEGIN")',new='conn.commit()\nconn.execute("BEGIN")')])
    assert 'conn.commit()' in fixed and 'assert 2+3==5' in fixed
    with pytest.raises(ValueError,match='assertions'):
        patch_probe(source,[ProbeEdit(old='assert 2+3==5',new='assert True')])
    with pytest.raises(ValueError,match='assertions'):
        patch_probe(source,[ProbeEdit(old='import sqlite3',new='sqlite3=None')])
    with pytest.raises(ValueError,match='exact'):
        patch_probe(source,[ProbeEdit(old='missing',new='pass')])
    with pytest.raises(ValueError,match='executable'):
        patch_probe(source,[ProbeEdit(old='import sqlite3',new='import sqlite3 # comment')])
    common={'tool':'probe','ok':False,'stale':False}
    assert repairable_failure({**common,'output':'Traceback (most recent call last):\nsqlite3.OperationalError: transaction already open'})
    assert not repairable_failure({**common,'output':'Traceback (most recent call last):\nAssertionError: candidate wrong'})


@pytest.mark.parametrize('valid,resolve,expected',[(True,True,True),(True,False,False),(False,True,False)])
def test_sqlite_probe_repairs_setup_retains_counterexample_and_requires_resolution(tmp_path,valid,resolve,expected):
    (tmp_path/'app.py').write_text('def update(conn):\n    conn.execute("UPDATE stock SET quantity=quantity+1")\n' if valid else
                                  'def update(conn):\n    conn.execute("UPDATE stock SET quantity=quantity+2")\n')
    code='''import sqlite3
from app import update
conn=sqlite3.connect(':memory:')
conn.execute('CREATE TABLE stock(quantity INTEGER)')
conn.execute('INSERT INTO stock VALUES (10)')
conn.execute('BEGIN')
update(conn)
assert conn.execute('SELECT quantity FROM stock').fetchone()[0]==11
conn.rollback()
assert conn.execute('SELECT quantity FROM stock').fetchone()[0]==10
print('PASS: update and outer rollback')
'''
    class Model:
        async def __call__(self,messages,schema,effort):
            data=json.loads(messages[-1]['content'])
            if schema.__name__=='TaskContract':return schema.model_validate({
                'requirements':[{'id':'r1','text':'Increment once without committing caller transaction','validation':'model_evidence'}],
                'domains':['software'],'response_format':'text','response_keys':[]})
            if schema.__name__=='ProbePlan':return schema.model_validate({
                'probes':[{'tool':'probe','args':{'code':code},'purpose':'Caller rollback must restore initial quantity'}],'limitation':'One boundary'})
            if schema.__name__=='RepairSchema':
                bad=data['failed_probes'][0]
                assert 'cannot start a transaction within a transaction' in bad['output']
                assert bad['args']['code']==code
                return schema.model_validate({'repairs':[{'id':bad['id'],'diagnosis':'probe_error',
                    'reason':'The fixture INSERT implicitly opened a transaction; commit setup before starting the caller transaction.',
                    'edits':[{'old':"conn.execute('BEGIN')",'new':"conn.commit()\nconn.execute('BEGIN')"}]}]})
            bad,new=data['active_verification']
            assert not bad['ok'] and new['repair_of']==bad['id']
            assert new['ok'] is valid
            return schema.model_validate({'aligned':True,'ready':True,'criteria':[{
                'requirement':'r1','status':'supported','evidence':[new['id']]}],
                'issues':[],'next_mode':'verify','probe_resolutions':([{'id':bad['id'],'kind':'oracle_error',
                'reason':'Original checker failed in fixture setup before invoking the candidate; replacement preserves both assertions.',
                'replacement_evidence':[new['id']]}] if resolve else [])})
    model=Model();ws,control,supervisor,node=setup(tmp_path,model)
    node.task='Increment once and preserve caller transaction';node.memory.files={'app.py':ws.file_digest('app.py')}
    critic=ContextCritic(model,supervisor,control,{'enabled':True,'active':True,'max_calls':4})
    before=(tmp_path/'app.py').read_text()
    result=asyncio.run(critic.assess(node,ws,'completion'))
    assert result.ready is expected and critic.calls==4
    assert (tmp_path/'app.py').read_text()==before
    observed=ws.obligations().data['verification']['observations']
    assert len(observed)==2 and not observed[0]['ok']


def test_plan_wire_uses_structured_steps_without_redundant_summary():
    from src.coding.harness import Turn
    from src.coding.orchestration import scoped_schema
    from src.coding.native_tools import compile_tools,decode_call
    schema=scoped_schema(Turn,{'read'})
    plan=next(x for x in schema.model_json_schema()['oneOf'] if x['properties']['action']['const']=='plan')
    assert set(plan['properties'])=={'action','plan'}
    tools=compile_tools(schema)
    assert schema.model_validate(decode_call({0:{'name':'plan','arguments':'{"plan":["Inspect","Implement","Verify"]}'}},tools)).plan==['Inspect','Implement','Verify']
    # Old checkpoints/scripted clients still accept their existing notes.
    assert Turn(action='plan',plan=['Inspect'],summary='Retained legacy note').summary=='Retained legacy note'
