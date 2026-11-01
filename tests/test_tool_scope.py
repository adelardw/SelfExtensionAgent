from src.coding.tool_scope import offered_tools,GROUPS
from src.coding.worktools import TOOLS


def scope(**kwargs):
    return offered_tools(TOOLS,domains=['mathematics'],has_files=False,checks={},pending=[],children=[],web=False,learn=False,**kwargs)


def test_math_menu_excludes_project_work_and_metadata_but_can_discover_allowed_tools():
    tools,hidden=scope()
    assert tools=={'context','evidence','think','math','experiment','ask_user'}
    assert 'files' in hidden and 'memory' in hidden
    expanded,_=scope(focus=['files','memory'])
    assert {'read','tree','claim','project_note'}<=expanded
    assert not expanded&{'web','learn','accept','resume_agent'}


def test_discovery_never_expands_readonly_authority_and_restores_pending_actions():
    tools,_=offered_tools({'read','evidence','context','think'},domains=['software'],has_files=True,
        checks={'unit':[]},pending=[{}],children=[{}],web=True,learn=True,focus=['all'])
    assert tools=={'read','evidence','context','think'}
    tools,_=offered_tools(TOOLS,domains=['software'],has_files=True,checks={'unit':[]},
        pending=[{}],children=[{}],web=True,learn=True)
    assert {'accept','patch','reject','resume_agent','check'}<=tools
    assert set(GROUPS['develop'])<=tools


def test_tool_discovery_preserves_requirement_details_outside_clipped_task():
    from types import SimpleNamespace
    from src.coding.tool_scope import scope_packet
    from src.coding.request import compact_packet
    packet={'task':'Keep this API.', 'obligations':{'requirements':[
        {'id':'r1','source_quote':'Keep this API.'},
        {'id':'r2','source_quote':'Preserve Unicode normalization semantics.'}]}}
    ws=SimpleNamespace(checks={},allow_web=False)
    node=SimpleNamespace(id='main',tools={'context','math'})
    scope_packet(ws,node,['mathematics'],{'adaptive_tools':True},packet)
    compact=compact_packet(packet)
    assert 'source_quote' not in compact['obligations']['requirements'][0]
    assert compact['obligations']['requirements'][1]['source_quote']=='Preserve Unicode normalization semantics.'


def test_implementation_keeps_inspection_available_with_missing_tests():
    from types import SimpleNamespace
    from src.coding.tool_scope import scope_packet
    ws=SimpleNamespace(checks={'unit':[]},allow_web=False)
    node=SimpleNamespace(id='main',tools={'read','tree','edit','write','check','context','evidence'})
    packet={'task':'Fix and add tests','phase':'implement','workspace':{'paths':['app.py','test_app.py','config.py']},
            'inspected_content':{'app.py':'source','test_app.py':'tests'},
            'completion_blockers':[{'id':'r1','validation':'new_or_updated_tests','requirement':'Add tests'}]}
    tools=scope_packet(ws,node,['software'],{'adaptive_tools':True},packet)
    assert tools==node.tools
    assert packet['task']=='Fix and add tests' and packet['completion_blockers']
    assert node.tools=={'read','tree','edit','write','check','context','evidence'}
    node.tools={'read','context','evidence'}
    assert scope_packet(ws,node,['software'],{'adaptive_tools':True},packet)<=node.tools
