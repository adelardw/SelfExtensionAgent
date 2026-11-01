"""A real source quotation must not turn answer syntax into project side effects."""
import asyncio

import pytest
from pydantic import ValidationError

from src.coding.critic import TaskContract,ContextCritic
from src.coding.delivery import explicit_json_keys,json_delivery_only
from src.coding.control import SessionControl
from src.coding.supervisor import Supervisor
from types import SimpleNamespace


DELIVERY='Return JSON with value (reduced fraction p/q) and derivation.'


@pytest.mark.parametrize('quote',[
    DELIVERY,'Верни JSON с ключами "ответ" и `тесты`.',
    '  Return a JSON final answer with keys status and tests; ',
])
@pytest.mark.parametrize('kind',['new_or_updated_tests','registered_checks','adopted_worker_changes','artifact_exists'])
def test_delivery_quote_cannot_authorize_mechanical_work(quote,kind):
    data={'requirements':[{'id':'r1','text':quote,'validation':kind,'source_quote':quote}]}
    assert json_delivery_only(quote)
    with pytest.raises(ValidationError,match='JSON answer format'):
        TaskContract.model_validate(data,context={'sources':['Calculate something. '+quote]})
    data['requirements'][0]['validation']='model_evidence'
    assert TaskContract.model_validate(data,context={'sources':[quote]}).requirements


@pytest.mark.parametrize('quote',[
    'Add regression tests. '+DELIVERY,
    DELIVERY+' Add regression tests.',
    'Add tests for JSON serialization.',
    'Return JSON with fields selected as you think appropriate.',
    'Write result.json containing the calculated value.',
    'Return JSON with value and derivation (unclosed',
])
def test_compound_artifact_or_unrecognized_instructions_are_not_reclassified(quote):
    assert not json_delivery_only(quote)


def test_answer_format_and_real_tests_remain_separate_requirements():
    task='Add regression tests and run the unit check. '+DELIVERY
    contract={'requirements':[
        {'id':'r1','text':'Add regression tests','validation':'new_or_updated_tests','source_quote':'Add regression tests'},
        {'id':'r2','text':'Run unit check','validation':'registered_checks','source_quote':'run the unit check'},
        {'id':'r3','text':DELIVERY,'validation':'model_evidence','source_quote':DELIVERY}],
        'domains':['software'],'response_format':'json','response_keys':['value','derivation']}
    checked=TaskContract.model_validate(contract,context={'sources':[task],'ground_answer_keys':True})
    assert [r.validation for r in checked.requirements]==['new_or_updated_tests','registered_checks','model_evidence']
    assert explicit_json_keys(task)==['value','derivation']


@pytest.mark.parametrize('call_limit',[1,2])
def test_recorded_misclassification_is_repaired_before_becoming_an_obligation(tmp_path,call_limit):
    class Model:
        calls=0
        async def __call__(self,messages,schema,effort):
            self.calls+=1
            if self.calls==2:assert 'JSON answer format' in messages[0]['content']
            return schema.model_validate({'requirements':[
                {'id':'r1','text':'Compute exact value','validation':'model_evidence'},
                {'id':'r2','text':DELIVERY,'validation':'new_or_updated_tests' if self.calls==1 else 'model_evidence',
                 'source_quote':DELIVERY}],
                'domains':['mathematics'],'response_format':'json','response_keys':['value','derivation']})
    control,model=SessionControl(),Model()
    supervisor=Supervisor(tmp_path,control,model,False)
    critic=ContextCritic(model,supervisor,control,{'enabled':True,'answer_verification':True,'max_calls':call_limit})
    node=SimpleNamespace(id='main',task='Compute an exact value. Do not create files. '+DELIVERY)
    if call_limit==1:
        with pytest.raises(ValidationError):asyncio.run(critic.contract(node))
        assert not critic.contracts and not list(tmp_path.glob('contract-*.json'))
    else:
        result=asyncio.run(critic.contract(node))
        assert {r.validation for r in result.requirements}=={'model_evidence'}
        assert result.response_keys==['value','derivation'] and len(result.requirements)==2
    assert model.calls==critic.calls==call_limit
