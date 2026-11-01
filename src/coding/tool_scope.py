"""Progressive tool disclosure: state-dependent menus within existing authority."""
from __future__ import annotations

GROUPS={
    'compute':{'math','experiment'},
    'files':{'tree','read','search','grep','diff','status','security_audit','inspect_artifact'},
    'develop':{'write','edit','run','check','register_check'},
    'coordinate':{'patch','accept','reject','resume_agent','share'},
    'memory':{'claim','learn','project_note','share'},
    'research':{'web','fetch'},
}
CORE={'context','evidence','think','ask_user'}


def offered_tools(permitted,*,domains,has_files,checks,pending,children,web,learn,focus=()):
    """Return a menu and discoverable groups; never grant a runtime capability."""
    available=set(permitted)
    if not web:available-={'web','fetch'}
    if not learn:available.discard('learn')
    if not checks:available.discard('check')
    if not pending:available-={'patch','accept','reject'}
    if not children:available.discard('resume_agent')
    if 'all' in focus:chosen=set(available)
    else:
        groups={'compute'}
        if 'software' in domains:groups|={'files','develop','coordinate'}
        elif has_files:groups.add('files')
        if 'research' in domains:groups.add('research')
        if pending or children:groups.add('coordinate')
        groups|=set(focus)&set(GROUPS)
        chosen=CORE|set().union(*(GROUPS[g] for g in groups))
        chosen&=available
    hidden={group:sorted(names&available-chosen) for group,names in GROUPS.items() if names&available-chosen}
    return chosen,hidden


def scope_packet(ws,node,domains,config,packet):
    if not config.get('adaptive_tools',False):return set(node.tools)
    version=getattr(getattr(ws,'orchestrator',None),'control',None)
    current=version.version(node.id) if version else 0
    focus=getattr(ws,'tool_focus',()) if getattr(ws,'tool_focus_version',current)==current else ()
    tools,hidden=offered_tools(node.tools,domains=domains,
        has_files=bool(packet.get('workspace',{}).get('paths')),
        checks=ws.checks,pending=packet.get('pending_proposals'),
        children=packet.get('subtasks',{}).get('items'),web=ws.allow_web,
        learn=config.get('learn',True),focus=focus)
    packet['tool_discovery']={'hidden_groups':hidden,
        'usage':'context {tools:[group]} reveals permitted tools. Use ["all"] for all, [] to reset.'} if hidden else {}
    if 'software' not in domains and not packet.get('workspace',{}).get('paths'):
        # No task/constraint/evidence is removed. Omit only irrelevant empty-project scaffolding.
        for key in ('workspace','project_guidance_files','verified_workflows'):
            packet.pop(key,None)
        if not packet.get('project_context',{}).get('notes'):packet.pop('project_context',None)
    # compact_packet removes only quotes still present verbatim in transmitted
    # task/steering. Tool discovery must not discard otherwise missing details.
    return tools
