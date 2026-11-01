"""Structural context compaction and receipts for what actually reaches the model."""
import copy
import json


def compact_packet(packet):
    """Remove duplicate observations, not operator constraints or evidence identities."""
    data=copy.deepcopy(packet)
    reads=data.get('inspected_content',{})
    memory=data.get('working_memory',{})
    reads={**reads,**memory.get('inspected_content',{})}
    failure_id=data.get('active_failure',{}).get('evidence_id')
    retained_results={r['evidence_id'] for r in data.get('completed_results',[])}
    for records in (data.get('recent',[]),memory.get('observations',[])):
        for item in records:
            if item.get('evidence_id') in retained_results and item.get('output'):
                item['output']='Result is included in completed_results; evidence id is retained.'
            if failure_id and item.get('evidence_id')==failure_id and item.get('output'):
                item['output']='Failure output is included in active_failure; evidence id is retained.'
            if item.get('tool')=='read' and item.get('output'):
                matching=next((p for p,t in reads.items() if t==item['output']),None)
                if matching:
                    item['output']='Source already included in inspected_content['+matching+']; evidence id is retained.'
    handoff=memory.get('handoff',{})
    if handoff.get('parent_task')==handoff.get('goal'): handoff.pop('parent_task',None)
    # Requirements remain explicit. Omit only storage defaults that carry no extra information.
    for rule in data.get('obligations',{}).get('requirements',[]):
        for key,default in (('path',''),('paths',[]),('inputs',[]),('count',1),('review',None)):
            if rule.get(key)==default: rule.pop(key,None)
        quote=rule.get('source_quote')
        sources=[data.get('task',''),*[item.get('text','') for item in data.get('human_steering',[])]]
        if quote and any(quote in source for source in sources if isinstance(source,str)):
            # The original instruction is still transmitted verbatim. Provenance
            # stays on disk; don't pay for a duplicate quote in every actor turn.
            rule.pop('source_quote')
    graph=data.get('obligations',{})
    if graph:
        # Review claims repeat the requirement text and status already present above.
        # Keep the originals in the graph/context tool, not three copies in every actor call.
        graph['claims']=[c for c in graph.get('claims',[]) if c.get('support_kind')!='model_review']
        verification=graph.get('verification',{})
        if verification.get('observations'):
            verification['observations']=[{k:p[k] for k in ('id','tool','ok','purpose','stale') if k in p}
                                         for p in verification['observations']]
    assessment=data.get('context_assessment')
    if assessment:
        assessment['criteria']=[{**c,'requirement':c['requirement'].split(':',1)[0]}
                                for c in assessment.get('criteria',[]) if c.get('status')!='supported']
    feedback=data.get('cognitive_feedback')
    if feedback and assessment:
        # Current verdict is transmitted once; older verdicts are not current evidence.
        data['cognitive_feedback']=[x for x in feedback if 'assessment' not in x]
    return {k:v for k,v in data.items() if v is not None and v!=[] and v!={}}


def schema_hint(schema):
    """Small semantic wire contract for providers whose decoder sees the schema alone."""
    raw=schema.model_json_schema()
    if 'oneOf' in raw: return ''  # The action protocol is already described by the scoped prompt.
    def compact(node,depth=0):
        if not isinstance(node,dict): return node
        if depth>8: return {}
        if '$ref' in node:
            name=node['$ref'].rsplit('/',1)[-1]
            return compact(raw.get('$defs',{}).get(name,{}),depth+1)
        result={k:node[k] for k in ('type','enum','const','required') if k in node}
        for key in ('items','properties','anyOf'):
            if key not in node: continue
            value=node[key]
            result[key]=([compact(x,depth+1) for x in value] if isinstance(value,list) else
                         {k:compact(v,depth+1) for k,v in value.items()} if key=='properties' else compact(value,depth+1))
        return result
    text=json.dumps(compact(raw),ensure_ascii=False,separators=(',',':'))
    if len(text)>6000: return ''
    return '\nOUTPUT CONTRACT (all required keys; use exact enum values, no HTML escaping):\n'+text


def cloud_schema(schema,model):
    """Keep runtime validation strict while avoiding Gemini's array grammar state explosion.

    Pydantic still enforces lengths after decoding. Tool enums, types and action scopes
    stay in the wire schema. No provider/model fallback is involved.
    """
    value=schema.model_json_schema()
    if not model.startswith('google/'): return value
    def compact(node):
        if isinstance(node,dict):
            node.pop('minItems',None);node.pop('maxItems',None)
            for child in node.values():compact(child)
        elif isinstance(node,list):
            for child in node:compact(child)
    compact(value)
    return value


def response_transport(config,supported_parameters):
    """Select a declared wire capability, never a different model or validator."""
    requested=config.get('structured_output','auto')
    if requested not in {'auto','json_schema','json_object','text'}:
        raise ValueError('structured_output must be auto, json_schema, json_object or text')
    if requested!='auto':return requested
    # Older compatible gateways omit capability metadata; retain their schema path.
    if supported_parameters is None or 'structured_outputs' in supported_parameters:return 'json_schema'
    return 'json_object' if 'response_format' in supported_parameters else 'text'


def text_schema_messages(messages,schema):
    """Expose the output grammar when the provider does not constrain decoding."""
    messages=copy.deepcopy(messages)
    instruction=('\nReturn exactly one JSON object matching this schema. No Markdown fences or prose outside JSON. '
                 'The application validates every field before any action can execute.\nOUTPUT JSON SCHEMA:\n'+
                 json.dumps(schema.model_json_schema(),ensure_ascii=False,separators=(',',':')))
    if messages[0]['role']=='system':messages[0]['content']+=instruction
    else:messages.insert(0,{'role':'system','content':instruction})
    return messages


def normalize_json_envelope(text):
    """Remove only redundant closers after ONE complete JSON object.

    Never fill missing syntax, change values, extract JSON from prose or choose
    between multiple actions. Callers must reject truncated output beforehand
    and validate the normalized object against the unchanged action schema.
    """
    stripped=text.strip()
    try:value,end=json.JSONDecoder().raw_decode(stripped)
    except (ValueError,TypeError):return text,None
    remainder=stripped[end:].strip()
    if isinstance(value,dict) and 0<len(remainder)<=8 and set(remainder)<={'}',']'}:
        return stripped[:end],'surplus_closing_delimiters'
    return text,None


def fit_messages(messages, schema, context_tokens, output_tokens):
    messages = copy.deepcopy(messages)
    schema_bytes = len(json.dumps(schema.model_json_schema()).encode())
    budget = context_tokens - output_tokens - schema_bytes - 512
    remaining = budget - sum(len(m['content'].encode()) for m in messages[:-1])
    if remaining < 1200:
        raise ValueError('context too small for system, schema and output; use at least 16384 tokens')
    text = messages[-1]['content']
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        data = None
    if isinstance(data, dict):
        protected = {'task', 'human_steering', 'checks', 'phase'}
        def encode():
            return json.dumps(data, ensure_ascii=False)
        for key in ('recent', 'evidence', 'verified_workflows', 'inspected_content', 'experience', 'shared_context'):
            while len(encode().encode()) > remaining and data.get(key):
                value = data[key]
                if isinstance(value, list) and len(value) > 1:
                    value.pop(0)
                elif isinstance(value, dict) and len(value) > 1:
                    value.pop(next(iter(value)))
                else:
                    break
        # Shrink long optional strings while preserving JSON and evidence identities.
        for _ in range(80):
            if len(encode().encode()) <= remaining:
                break
            candidates = []
            def visit(obj, prefix):
                if isinstance(obj, dict):
                    for k,v in obj.items():
                        if prefix or k not in protected:
                            if isinstance(v, str) and len(v) > 200 and k not in {'id','evidence_id','path'}:
                                candidates.append((len(v), obj, k))
                            else:
                                visit(v, True)
                elif isinstance(obj, list):
                    for value in obj: visit(value, True)
            visit(data, False)
            if not candidates:
                raise ValueError('task and human steering exceed context; compact the request before continuing')
            _, parent, key = max(candidates, key=lambda v:v[0])
            value = parent[key]
            parent[key] = value[:len(value)//4] + '\n[omitted; use evidence]\n' + value[-len(value)//4:]
        text = encode()
    elif len(text.encode()) > remaining:
        raise ValueError('unstructured request exceeds the context budget')
    messages[-1]['content'] = text
    effective = data if isinstance(data, dict) else {}
    recovery=effective.get('output_recovery')
    if not isinstance(recovery,dict) or not isinstance(recovery.get('provider_returned_draft'),str):
        recovery=None
    receipt = {'experience': [x['id'] for x in effective.get('experience', [])],
               'shared': [x['id'] for x in effective.get('shared_context', [])],
               'steering': [x['id'] for x in effective.get('human_steering', [])],
               'mode': effective.get('cognitive_mode', effective.get('phase')),
               'memory_evidence': effective.get('working_memory', {}).get('evidence_ids', []),
               'subtasks': [x['id'] for x in effective.get('subtasks',{}).get('items',[])],
               'project_notes': [x['id'] for x in effective.get('project_context', {}).get('notes', [])],
               'requirements': [x['id'] for x in effective.get('obligations', {}).get('requirements', [])],
               'claims': [x['id'] for x in effective.get('obligations', {}).get('claims', [])],
               'navigation_page': effective.get('context_navigation', {}).get('page'),
               'active_failure': effective.get('active_failure', {}).get('evidence_id'),
               'focus_requirement':(effective.get('followthrough',{}).get('focus') or {}).get('requirement'),
               'human_action':[q['id'] for q in effective.get('human_action',[])],
               'completed_results': [x['evidence_id'] for x in effective.get('completed_results', [])],
               'output_recovery': ({'draft_chars':len(recovery['provider_returned_draft']),
                                    'status':recovery.get('status')} if recovery else None),
               'input_utf8_bytes': sum(len(m['content'].encode()) for m in messages)}
    return messages, receipt
