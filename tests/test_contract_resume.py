"""Resume reuses extraction only under its original task and producer scope."""
import asyncio
import json

import pytest

from src.coding.critic import ContextCritic
from src.coding.control import SessionControl
from src.coding.harness import Workspace
from src.coding.minds import MindScheduler
from src.coding.model import BudgetExceeded
from src.coding.supervisor import Supervisor


class ContractModel:
    def __init__(self):
        self.name='small-model'
        self.config={'provider_only':['fixed-provider']}
        self.calls=0

    async def __call__(self,messages,schema,effort):
        self.calls+=1
        return schema.model_validate({'requirements':[{'id':'r1','text':'Add regression tests',
            'validation':'new_or_updated_tests','source_quote':'Add regression tests'}],
            'domains':['software'],'response_format':'text','response_keys':[]})


def setup(tmp_path):
    directory=tmp_path/'.sea';directory.mkdir()
    control,model=SessionControl(),ContractModel()
    supervisor=Supervisor(directory,control,model,False)
    node=MindScheduler(control,supervisor,{},Workspace(tmp_path,directory,{})).nodes['main']
    node.task='Add regression tests for the function.'
    config={'max_calls':2,'answer_verification':True}
    critic=ContextCritic(model,supervisor,control,config)
    return critic,node,model,supervisor,control,config


def test_resume_reuses_contract_without_another_model_call_even_at_call_limit(tmp_path):
    critic,node,model,supervisor,control,config=setup(tmp_path)
    control.post('Keep the public API.',interrupt=False)
    result=asyncio.run(critic.contract(node))
    resumed=ContextCritic(ContractModel(),supervisor,control,{**config,'max_calls':0})
    assert asyncio.run(resumed.contract(node))==result
    assert resumed.calls==resumed.model.calls==0 and model.calls==1
    assert any(e['kind']=='contract-reused' for e in supervisor.events)


@pytest.mark.parametrize('change',['task','steering','model','provider','answer_verification','code'])
@pytest.mark.parametrize('in_memory',[False,True])
def test_contract_cache_cannot_cross_task_or_producer_scope(tmp_path,monkeypatch,change,in_memory):
    critic,node,model,supervisor,control,config=setup(tmp_path)
    asyncio.run(critic.contract(node))
    if change=='task':node.task+=' Include the empty input.'
    elif change=='steering':control.post('Keep the public API.',interrupt=False)
    elif change=='model':model.name='different-model'
    elif change=='provider':model.config['provider_only']=['different-provider']
    elif change=='answer_verification':config['answer_verification']=False
    else:
        import src.coding.critic as module
        changed=tmp_path/'new_critic.py';changed.write_text('# Changed extraction implementation')
        monkeypatch.setattr(module,'__file__',str(changed))
    active=critic if in_memory else ContextCritic(model,supervisor,control,config)
    active.config['max_calls']=active.calls
    with pytest.raises(BudgetExceeded):asyncio.run(active.contract(node))
    assert model.calls==1


@pytest.mark.parametrize('corruption',['syntax','type','missing','ungrounded','invalid_key'])
def test_invalid_persisted_contract_is_reextracted_and_grounded(tmp_path,corruption):
    critic,node,model,supervisor,control,config=setup(tmp_path)
    result=asyncio.run(critic.contract(node))
    path=next(supervisor.directory.glob('contract-*.json'))
    stored=json.loads(path.read_text())
    if corruption=='syntax':path.write_text('{')
    elif corruption=='type':path.write_text('[]')
    else:
        if corruption=='missing':stored.pop('contract')
        elif corruption=='ungrounded':stored['contract']['requirements'][0]['source_quote']='Invented obligation'
        else:stored['contract'].update(response_format='json',response_keys=['invented_key'])
        path.write_text(json.dumps(stored))
    resumed=ContextCritic(model,supervisor,control,config)
    assert asyncio.run(resumed.contract(node))==result
    assert resumed.calls==1 and model.calls==2
