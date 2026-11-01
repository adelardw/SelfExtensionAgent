"""Same engine, domain permissions, exact arithmetic and sandboxed algorithm experiments."""
import asyncio
import json

import pytest

from src.coding.harness import Workspace,CodingHarness
from src.coding.numerics import calculate,expression


def test_equation_syntax_is_rejected_with_actionable_explicit_side_mapping():
    with pytest.raises(ValueError,match='expr and rhs must be expressions'):
        calculate({'op':'solve','expr':'E0 - (3/5)*E1 = 1','rhs':'1','var':'E0'})
    corrected=calculate({'op':'solve','expr':'E0-(3/5)*E1','rhs':'1','var':'E0'})
    assert corrected['result']=='[3*E1/5 + 1]' and corrected['resolved']
from src.coding.control import SessionControl
from src.coding.supervisor import Supervisor
from src.coding.orchestration import Orchestrator


def test_math_uses_exact_rationals_and_explicit_assumptions():
    assert calculate({'expr':'1/3+1/6'})['result']=='1/2'
    assert calculate({'op':'diff','expr':'x**3','var':'x'})['result']=='3*x**2'
    result=calculate({'op':'identity','expr':'(x+1)**2','rhs':'x**2+2*x+1'})
    assert 'symbolic_identity' in result['result']
    assert 'not a general proof' in result['limitation']


@pytest.mark.parametrize('text',["__import__('os').system('echo bad')",'x.__class__','[x for x in range(10)]','(lambda: 1)()'])
def test_math_parser_never_evaluates_python(text):
    with pytest.raises(ValueError): expression(text,{})


def test_symbolic_budgets_block_explosive_expressions():
    for expr in ['2**100000000','factorial(100000000)','(x+y+z)**10000']:
        with pytest.raises(ValueError): calculate({'op':'expand','expr':expr})


def test_symbolic_bounds_normalize_json_array_without_indexing_a_string():
    assert calculate({'op':'sum','expr':'k','var':'k','bounds':'[1, 100]'})['result']=='5050'
    with pytest.raises(ValueError): calculate({'op':'sum','expr':'k','bounds':'1,100'})


def test_unknown_mathematical_function_explains_the_supported_vocabulary():
    with pytest.raises(ValueError,match=r'Unknown mathematical function C.*binomial'):
        calculate({'expr':'C(7,3)'})
    assert calculate({'expr':'binomial(7,3)'})['result']=='35'
    assert calculate({'op':'sum','expr':'k*binomial(4,k)*binomial(5,5-k)/binomial(9,5)',
                      'var':'k','bounds':[0,4]})['result']=='20/9'


def test_mathematics_changes_profile_without_granting_source_edits(tmp_path):
    class Model:
        async def __call__(self,messages,schema,effort):
            return schema.model_validate({'domains':['mathematics'],'requirements':[
                {'id':'r1','text':'Find derivative','validation':'model_evidence'}]})
    async def run():
        directory=tmp_path/'.sea';directory.mkdir()
        ws=Workspace(tmp_path,directory,{})
        harness=CodingHarness(tmp_path,model=Model(),config={});harness.control=SessionControl()
        supervisor=Supervisor(directory,harness.control,harness.model,False)
        o=Orchestrator(harness,ws,supervisor,{'auto_domains':True,'critic':{'enabled':True}})
        await o.prepare('Find derivative of x cubed')
        node=o.update_main('Find derivative','plan','',[],[],{})
        assert node.mode=='mathematical-analysis' and 'math' in node.tools
        o.scheduler.switch('main','implement')
        assert 'edit' not in node.tools and 'write' not in node.tools
        assert o.domains==['mathematics']
    asyncio.run(run())


def test_math_and_algorithm_tools_run_without_mutating_source(tmp_path):
    async def run():
        directory=tmp_path/'.sea';directory.mkdir()
        ws=Workspace(tmp_path,directory,{})
        (tmp_path/'source.py').write_text('VALUE=1')
        result=await ws.execute('math',{'expr':'1/3+1/6'})
        assert result['ok'],result
        assert json.loads(result['output'])['result']=='1/2'
        result=await ws.execute('experiment',{'code':'print(sum(range(101)))'})
        assert result['ok'] and '5050' in result['output']
        assert 'NOT A GENERAL' in result['output']
        result=await ws.execute('experiment',{'code':'from pathlib import Path; Path("source.py").write_text("bad")'})
        assert not result['ok'] and (tmp_path/'source.py').read_text()=='VALUE=1'
        assert ws.revision==0
    asyncio.run(run())


def test_strict_backbone_has_no_hidden_remote_critic_or_controller(tmp_path):
    directory=tmp_path/'.sea';directory.mkdir()
    class Model: pass
    model=Model()
    harness=CodingHarness(tmp_path,model=model,config={});harness.control=SessionControl()
    ws=Workspace(tmp_path,directory,{})
    supervisor=Supervisor(directory,harness.control,model,False)
    o=Orchestrator(harness,ws,supervisor,{'same_model':True,
        'decision':{'backend':'decisions','planner':{'max_calls':2}},'critic':{'backend':'openrouter','enabled':True}})
    assert o.router.remote is None and o.router.planner is None and o.critic.remote is None
    assert o.router.model is model and o.critic.model is model
