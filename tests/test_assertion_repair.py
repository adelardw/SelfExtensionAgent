"""A blind prediction can propose only scalar changes in a newly added test."""
import ast

import pytest

from src.coding.assertion_repair import prepare_assertion_plan


@pytest.fixture(autouse=True)
def permissions(monkeypatch):
    from src.runtime import hitl
    monkeypatch.setattr(hitl,'_work_mode',{'':'manual'})


ORIGINAL='''import unittest
from meter import Meter
class Tests(unittest.TestCase):
    def test_original(self):
        self.assertEqual(1,1)
'''
NEW=ORIGINAL+'''    def test_added(self):
        meter=Meter(7)
        _=meter.consume(3)
        self.assertEqual(meter.read(),7)
        self.assertIsNone(meter.read())
'''


def test_blind_packet_contains_inputs_but_no_wrong_expectations():
    plan=prepare_assertion_plan(NEW,ORIGINAL,'Tests','test_added')
    assert plan.blind_input()=={'operations':[['Meter',[7],{}],['consume',[3],{}],['read',[],{}],['read',[],{}]],'predict_steps':[2,3]}
    changed=plan.proposal([4,4],[None,4,4,4])
    assert changed.startswith(ORIGINAL)
    assert changed.count('self.assertEqual(meter.read(),4)')==1
    assert 'self.assertEqual(meter.read(), 4)' in changed
    tree=ast.parse(changed)
    assert len([n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr.startswith('assert')])==3
    assert plan.masked_test()=='''def test_added(self):
    meter=Meter(7)
    _=meter.consume(3)
    self.assertEqual(meter.read(),EXPECTED_1)
    self.assertEqual(meter.read(), EXPECTED_2)'''


@pytest.mark.parametrize('values,actual',[
    ([4,4],[None,4,5,5]),([4],[None,4,4,4]),([4,4],[None,4,4]),([True,True],[None,1,1,1]),
    ([float('nan'),4],[None,4,float('nan'),4]),
])
def test_disagreement_incomplete_values_and_type_confusion_block_patch(values,actual):
    plan=prepare_assertion_plan(NEW,ORIGINAL,'Tests','test_added')
    with pytest.raises(ValueError):plan.proposal(values,actual)


def test_preexisting_test_cannot_be_rewritten():
    with pytest.raises(ValueError,match='original test'):
        prepare_assertion_plan(NEW,NEW,'Tests','test_added')


@pytest.mark.parametrize('source',[
    NEW.replace('_=meter.consume(3)','for i in range(3): meter.consume(1)'),
    NEW.replace('_=meter.consume(3)','value=meter.consume(3)'),
    NEW.replace('meter.read(),7','7,meter.read()'),
    NEW.replace('meter.read(),7','meter.read(),compute_expected()'),
    NEW.replace('class Tests(unittest.TestCase):','class Tests(CustomTest):'),
    NEW.replace('    def test_added(self):','    @unittest.skip("skip")\n    def test_added(self):'),
    NEW+'    def setUp(self): pass\n',
    NEW+'\ndef setUpModule(): pass\n',
])
def test_unsupported_control_flow_fixtures_and_expectations_are_not_guessed(source):
    with pytest.raises(ValueError):prepare_assertion_plan(source,ORIGINAL,'Tests','test_added')


def test_unicode_columns_and_existing_none_assertion_are_preserved():
    source=ORIGINAL+'''    def test_added(self):
        meter=Meter(7)
        self.assertEqual(meter.read('ключ'),7)
        self.assertIsNone(meter.read('нет'))
'''
    plan=prepare_assertion_plan(source,ORIGINAL,'Tests','test_added')
    changed=plan.proposal([4,None],[None,4,None])
    assert "self.assertEqual(meter.read('ключ'),4)" in changed
    assert "self.assertIsNone(meter.read('нет'))" in changed


def test_reference_runs_without_candidate_imports_and_binds_only_matching_values(tmp_path):
    import asyncio,json,sys
    from src.coding.harness import Workspace
    from src.coding.verifier import isolated_probe
    from src.runtime import hitl
    reference='''import importlib.util
assert importlib.util.find_spec('meter') is None
class Meter:
    def __init__(self,value): self.value=value
    def consume(self,n): self.value-=n; return self.value
    def read(self): return self.value
'''
    async def run():
        hitl.set_work_mode('auto')
        candidate=tmp_path/'candidate';candidate.mkdir();directory=candidate/'.sea';directory.mkdir()
        (candidate/'meter.py').write_text(reference.replace("assert importlib.util.find_spec('meter') is None\n",''))
        (candidate/'test_meter.py').write_text(NEW)
        ws=Workspace(candidate,directory,{'unit':[sys.executable,'-B','-m','unittest','-q']})
        plan=prepare_assertion_plan(NEW,ORIGINAL,'Tests','test_added')
        actual=await isolated_probe(ws,plan.observation_program(),preserve_source=True)
        empty=tmp_path/'empty';empty.mkdir();logs=empty/'.sea';logs.mkdir()
        oracle=Workspace(empty,logs,{})
        expected=await oracle.command([sys.executable,'-c',plan.observation_program(reference)],timeout=15)
        def values(result):
            assert result['ok']
            return json.loads(next(l.removeprefix('SEA_ASSERTION_VALUES ') for l in result['output'].splitlines() if l.startswith('SEA_ASSERTION_VALUES ')))
        observations=values(actual);predictions=values(expected)
        changed=plan.proposal([predictions[a['step']] for a in plan.assertions],observations)
        assert changed.startswith(ORIGINAL)
        assert (await ws.execute('edit',{'path':'test_meter.py','old':NEW,'new':changed}))['ok']
        assert (await ws.execute('check',{'name':'unit'}))['ok']
    asyncio.run(run())
