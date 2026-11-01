import asyncio
import json

import httpx
import pytest

from src.coding.harness import Workspace
from src.coding.navigation import ContextNavigator, DecisionsScorer, JevScorer, validate_scores


def workspace(tmp_path):
    directory=tmp_path/'.sea';directory.mkdir()
    ws=Workspace(tmp_path,directory,{})
    eid=ws.record({'ok':True,'output':'observed'})
    for i in range(7): ws.board.publish(f'needle context {i}',[eid],kind='fact' if i<4 else 'procedure')
    return ws


def test_navigation_preserves_all_alternatives_backtracks_and_resumes(tmp_path):
    ws=workspace(tmp_path)
    nav=ContextNavigator(ws)
    first=asyncio.run(nav.select('needle'))
    second=asyncio.run(nav.select('needle','next'))
    assert {x['id'] for x in first}.isdisjoint(x['id'] for x in second)
    resumed=ContextNavigator(ws)
    assert asyncio.run(resumed.select('needle'))==second
    assert asyncio.run(resumed.select('needle','back'))==first
    frontier=nav.packet()['frontier']
    assert {eid for page in frontier for eid in page['ids']}=={x['id'] for x in ws.board.items}


def test_stale_candidates_removed_before_scoring_and_query_override_persists(tmp_path):
    ws=workspace(tmp_path)
    (tmp_path/'app.py').write_text('a')
    stale=ws.board.publish('stale needle',["1"],files=['app.py'])
    (tmp_path/'app.py').write_text('b')
    nav=ContextNavigator(ws);ws.navigator=nav
    asyncio.run(nav.select('needle'))
    assert stale['id'] not in {i for p in nav.packet()['frontier'] for i in p['ids']}
    asyncio.run(ws.execute('context',{'query':'procedure','action':'next'}))
    asyncio.run(nav.select('needle'))
    assert nav.state['query']=='procedure'


def test_scoring_cached_and_failures_do_not_prune_alternatives(tmp_path):
    ws=workspace(tmp_path)
    class Scorer:
        calls=0
        async def score(self,query,candidates):
            self.calls+=1
            return {x['id']:0.5 for x in candidates}
    scorer=Scorer();nav=ContextNavigator(ws,scorer=scorer)
    asyncio.run(nav.select('needle'));asyncio.run(nav.select('needle','next'))
    assert scorer.calls==1 and nav.packet()['scorer']=='decisions'
    class Broken:
        async def score(self,*a): return {'invented': float('nan')}
    nav=ContextNavigator(ws,scorer=Broken())
    asyncio.run(nav.select('needle'))
    assert nav.packet()['scorer']=='lexical' and nav.packet()['limitation']
    assert sum(len(p['ids']) for p in nav.packet()['frontier'])==7


@pytest.mark.parametrize('scores',[{'a':float('nan')},{'a':2},{'a':True},{'b':.4},{'a':None}])
def test_invalid_scorer_values_are_rejected(scores):
    with pytest.raises(ValueError): validate_scores(scores,{'a'})


def test_same_model_disables_hidden_decision_backbone(tmp_path):
    ws=workspace(tmp_path)
    assert ContextNavigator(ws,config={'backend':'jev'},same_model=True,scorer=object()).scorer is None


def test_child_rejects_parent_fact_after_editing_its_isolated_copy(tmp_path):
    parent=workspace(tmp_path)
    (tmp_path/'app.py').write_text('VALUE=1')
    fact=parent.board.publish('Value is one',["1"],files=['app.py'])
    child_root=tmp_path/'child';child_root.mkdir()
    (child_root/'app.py').write_text('VALUE=2')
    child=Workspace(child_root,parent.directory,{})
    child.board=parent.board
    nav=ContextNavigator(child)
    assert fact['id'] not in {x['id'] for x in nav.inventory()}


@pytest.mark.parametrize('model',['typesafe/jev-1.13','liquid/d1-ultra'])
def test_decisions_protocol_uses_explicit_endpoint_metadata_and_accounts_cost(monkeypatch,model):
    import src.llm.llm as llm
    monkeypatch.setattr(llm,'api_key',lambda:'test-key')
    requests=[]
    def handle(request):
        if request.method=='GET':
            assert request.url.path=='/api/v1/models/'+model+'/endpoints'
            return httpx.Response(200,json=metadata(model))
        payload=json.loads(request.content);requests.append(payload)
        assert request.url.host=='openrouter.ai' and request.url.path=='/api/alpha/decisions'
        assert payload['state']['query']=='q' and set(payload['questions'])=={'q0','q1'}
        assert payload['model']==model
        return httpx.Response(200,json={'answers':{'q0':{'noul':.1},'q1':{'noul':.9}},
                                      'usage':{'cost':.00001,'input_tokens':100,'output_tokens':0}})
    factory=httpx.AsyncClient
    monkeypatch.setattr(httpx,'AsyncClient',lambda **kw:factory(transport=httpx.MockTransport(handle),**kw))
    scorer=DecisionsScorer({'max_calls':1,'model':model})
    candidates=[{'id':'a','text':'A'},{'id':'b','text':'B'}]
    assert asyncio.run(scorer.score('q',candidates))=={'a':.1,'b':.9}
    asyncio.run(scorer.score('q',candidates))
    assert len(requests)==1 and scorer.cost==pytest.approx(.00001)
    assert not scorer.usage()['cost_estimated'] and scorer.usage()['model']==model
    with pytest.raises(ValueError,match='budget'): asyncio.run(scorer.score('other',candidates))


def metadata(model='typesafe/jev-1.13'):
    return {'data':{'id':model,'architecture':{'output_modalities':['decisions']},
                    'endpoints':[{'pricing':{'prompt':'0.000000042','completion':'0'}}]}}


@pytest.mark.parametrize('problem',['modality','empty','expensive_provider','unknown_price','output_price','request_price'])
def test_unusable_or_expensive_endpoints_never_send_a_paid_request(monkeypatch,problem):
    import src.llm.llm as llm
    monkeypatch.setattr(llm,'api_key',lambda:'test-key')
    data=metadata()['data']
    if problem=='modality': data['architecture']['output_modalities']=['text']
    elif problem=='empty': data['endpoints']=[]
    elif problem=='expensive_provider': data['endpoints'].append({'pricing':{'prompt':'0.0001','completion':'0'}})
    else:
        prices=data['endpoints'][0]['pricing']
        prices[{'unknown_price':'prompt','output_price':'completion','request_price':'request'}[problem]]='nan' if problem=='unknown_price' else '0.001'
    def handle(request):
        assert request.method=='GET'
        return httpx.Response(200,json={'data':data})
    factory=httpx.AsyncClient
    monkeypatch.setattr(httpx,'AsyncClient',lambda **kw:factory(transport=httpx.MockTransport(handle),**kw))
    scorer=DecisionsScorer({})
    with pytest.raises(ValueError): asyncio.run(scorer.score('q',[{'id':'a','text':'A'}]))
    assert scorer.calls==0 and scorer.cost==0


def test_failed_or_cancelled_request_keeps_reserved_cost_and_call_budget(monkeypatch):
    import src.llm.llm as llm
    monkeypatch.setattr(llm,'api_key',lambda:'test-key')
    async def handle(request):
        if request.method=='GET': return httpx.Response(200,json=metadata())
        raise asyncio.CancelledError()
    factory=httpx.AsyncClient
    monkeypatch.setattr(httpx,'AsyncClient',lambda **kw:factory(transport=httpx.MockTransport(handle),**kw))
    scorer=DecisionsScorer({'max_calls':1})
    with pytest.raises(asyncio.CancelledError): asyncio.run(scorer.score('q',[{'id':'a','text':'A'}]))
    assert scorer.calls==1 and scorer.cost>0 and scorer.usage()['cost_estimated']
    with pytest.raises(ValueError,match='budget'): asyncio.run(scorer.score('q',[{'id':'a','text':'A'}]))


def test_concurrent_scoring_shares_a_single_call_budget(monkeypatch):
    import src.llm.llm as llm
    monkeypatch.setattr(llm,'api_key',lambda:'test-key')
    posts=[]
    async def handle(request):
        await asyncio.sleep(0)
        if request.method=='GET': return httpx.Response(200,json=metadata())
        posts.append(request)
        return httpx.Response(200,json={'answers':{'q0':{'noul':.9}},'usage':{'cost':.00001}})
    factory=httpx.AsyncClient
    monkeypatch.setattr(httpx,'AsyncClient',lambda **kw:factory(transport=httpx.MockTransport(handle),**kw))
    scorer=DecisionsScorer({'max_calls':1})
    async def run():
        return await asyncio.gather(scorer.score('one',[{'id':'a','text':'A'}]),
                                    scorer.score('two',[{'id':'b','text':'B'}]),return_exceptions=True)
    results=asyncio.run(run())
    assert len(posts)==1 and sum(isinstance(r,ValueError) for r in results)==1


def test_legacy_jev_alias_and_both_backend_names_preserve_strict_backbone(tmp_path):
    assert JevScorer is DecisionsScorer
    ws=workspace(tmp_path)
    for backend in ('jev','decisions'):
        assert ContextNavigator(ws,config={'backend':backend},same_model=True).scorer is None
        assert isinstance(ContextNavigator(ws,config={'backend':backend}).scorer,DecisionsScorer)
