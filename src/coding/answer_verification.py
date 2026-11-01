"""Bind independently computed oracle values to the exact delivered JSON answer.

The oracle remains fallible model-written code. Equality is deterministic; the
semantic correctness/completeness of that oracle is not a formal proof.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import ast
from typing import Any,Literal

from pydantic import BaseModel,ConfigDict,Field,model_validator


class AnswerCheck(BaseModel):
    model_config=ConfigDict(extra='forbid')
    field:str=Field(min_length=1,max_length=100)
    op:Literal['equal','fraction','set_equal','one_of','set_one_of','predicate']
    expected:Any


class AnswerChecks(BaseModel):
    model_config=ConfigDict(extra='forbid')
    checks:list[AnswerCheck]=Field(min_length=1,max_length=24)


def answer_program_schema(fields):
    """Make exporting every required field a decoding constraint, not prose advice."""
    fields=tuple(fields)
    class AnswerProgram(BaseModel):
        model_config=ConfigDict(extra='forbid')
        code:str=Field(min_length=1,max_length=6000)
        outputs:dict[str,str]
        limitation:str=Field(max_length=400)

        @model_validator(mode='after')
        def complete_outputs(self):
            if set(self.outputs)!=set(fields):
                raise ValueError('outputs must contain exactly these fields: '+', '.join(fields))
            if any(not 0<len(expr)<=1000 for expr in self.outputs.values()):
                raise ValueError('Each output is a Python variable name or expression of 1..1000 characters')
            return self

        @classmethod
        def model_json_schema(cls,**kwargs):
            schema=super().model_json_schema(**kwargs)
            schema['properties']['outputs']={'type':'object','properties':{
                name:{'type':'string','minLength':1,'maxLength':1000} for name in fields},
                'required':list(fields),'additionalProperties':False}
            return schema
    return AnswerProgram


def compile_answer_program(code,outputs,fields):
    """Bind named computed expressions without guessing which variable is the answer."""
    program=answer_program_schema(fields).model_validate({'code':code,'outputs':outputs,'limitation':''})
    exports=[]
    for name,expr in program.outputs.items():
        try:ast.parse(expr,mode='eval')
        except SyntaxError as exc:raise ValueError('Output '+name+' must be one Python expression') from exc
        exports.append('export('+repr(name)+',('+expr+'))')
    return program.code+'\n'+'\n'.join(exports)+'\n'


def validate_oracle_program(code):
    """Reject a common failure: compute values, then print unrelated literal guesses.

    This is a liveness heuristic, not a security boundary or a proof of the oracle.
    Real task inputs may be literals; reported expected outputs must use computation.
    """
    try:tree=ast.parse(code)
    except (SyntaxError,TypeError) as exc:raise ValueError('Oracle must be valid Python: '+str(exc)[:250]) from exc
    definitions={};expressions=[];predicates=[];automatic=[]
    def exported(target,value):
        name=(target.id if isinstance(target,ast.Name) else target.value.id
              if isinstance(target,ast.Subscript) and isinstance(target.value,ast.Name) else None)
        if name not in {'EXPECTED','VALIDATORS'}:return
        values=list(value.values) if isinstance(target,ast.Name) and isinstance(value,ast.Dict) else [value]
        (expressions if name=='EXPECTED' else predicates).extend(values)
    for node in ast.walk(tree):
        if isinstance(node,ast.FunctionDef):definitions[node.name]=node
        elif isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name):definitions[target.id]=node.value
                exported(target,node.value)
        elif isinstance(node,ast.AnnAssign) and isinstance(node.target,ast.Name):
            definitions[node.target.id]=node.value
            exported(node.target,node.value)
        elif isinstance(node,ast.AugAssign) and isinstance(node.target,ast.Name):
            definitions[node.target.id]=node
        if isinstance(node,ast.Dict):
            expressions.extend(v for k,v in zip(node.keys,node.values) if isinstance(k,ast.Constant) and k.value=='expected')
        elif isinstance(node,ast.Call):
            expressions.extend(k.value for k in node.keywords if k.arg=='expected')
            if isinstance(node.func,ast.Name) and node.func.id=='check' and len(node.args)>=2:
                expressions.append(node.args[1])
            if isinstance(node.func,ast.Name) and node.func.id=='verify' and len(node.args)>=2:
                predicates.append(node.args[1])
            if isinstance(node.func,ast.Name) and node.func.id=='export' and len(node.args)>=2:
                automatic.append(node.args[1])
    for expr in automatic:
        resolved=expr;seen=set()
        while isinstance(resolved,ast.Name) and resolved.id in definitions and resolved.id not in seen:
            seen.add(resolved.id);resolved=definitions[resolved.id]
        (predicates if isinstance(resolved,(ast.Lambda,ast.FunctionDef)) else expressions).append(
            resolved if isinstance(resolved,(ast.Lambda,ast.FunctionDef)) else expr)
    def computed(expr,seen=()):
        if isinstance(expr,ast.Name) and expr.id in definitions:
            return expr.id not in seen and computed(definitions[expr.id],(*seen,expr.id))
        if isinstance(expr,ast.Constant):return False
        if isinstance(expr,ast.UnaryOp):return computed(expr.operand,seen)
        if isinstance(expr,(ast.List,ast.Tuple,ast.Set)):return any(computed(e,seen) for e in expr.elts)
        if isinstance(expr,ast.Dict):return any(computed(e,seen) for e in expr.values)
        return expr is not None
    if (not expressions and not predicates) or any(not computed(expr) for expr in expressions):
        raise ValueError('Oracle expected outputs must reference computed variables/expressions, not literal guesses. '
                         'Use the values the algorithm actually computed in EVERY expected field.')
    for predicate in predicates:
        if isinstance(predicate,ast.Name):predicate=definitions.get(predicate.id)
        if not isinstance(predicate,(ast.Lambda,ast.FunctionDef)):
            raise ValueError('verify requires a defined function or lambda using its candidate argument')
        parameters={a.arg for a in [*predicate.args.posonlyargs,*predicate.args.args]}
        if not any(isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load) and n.id in parameters for n in ast.walk(predicate)):
            raise ValueError('Witness predicate must use its candidate argument, not return a constant')


async def run_answer_program(ws,code,fields=(),candidate=None,outputs=None):
    """The runtime serializes observed variables, avoiding model-written JSON boilerplate."""
    from .verifier import isolated_probe
    if outputs is not None:code=compile_answer_program(code,outputs,fields)
    validate_oracle_program(code)
    tree=ast.parse(code)
    helper=any((isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in {'check','verify','export'})
               or isinstance(n,ast.Name) and n.id in {'EXPECTED','VALIDATORS'} for n in ast.walk(tree))
    if not helper:return await isolated_probe(ws,code)  # old checkpoints/oracles stay executable
    if candidate is not None:
        value=json.loads(candidate)
        candidate=json.dumps({k:value[k] for k in machine_fields(value,fields)},ensure_ascii=False)
    wrapper=('CODE='+repr(code)+'\nFIELDS_LITERAL='+repr(tuple(fields))+'\nCANDIDATE='+repr(candidate)+'\n'+'''import json, sys, contextlib, hashlib, copy
from fractions import Fraction
from numbers import Integral, Rational
_checks=[]
_validators=[]
_fields=FIELDS_LITERAL
def field_name(field):
    if len(_checks)+len(_validators)>=24:raise ValueError("At most 24 answer checks")
    if type(field) is int:field=_fields[field]
    if _fields and field not in _fields:raise ValueError("Use exact answer fields: "+str(_fields)+"; received "+str(field))
    return field
def check(field,expected,op="equal"):
    if op=="predicate":raise ValueError("Use verify(field,predicate) for executable witness checks")
    field=field_name(field)
    _checks.append(dict(field=field,expected=expected,op=op))
def verify(field,predicate):
    field=field_name(field)
    if not callable(predicate):raise TypeError("Witness predicate must be callable")
    _validators.append((field,predicate))
def export(field,value):
    if callable(value):verify(field,value)
    else:
        op="fraction" if isinstance(value,Rational) and not isinstance(value,Integral) else "set_equal" if isinstance(value,set) else "equal"
        check(field,value,op=op)
def serialize(value):
    if isinstance(value,Integral):return int(value)
    if isinstance(value,Rational):return str(Fraction(int(value.numerator),int(value.denominator)))
    if isinstance(value,set):return sorted(value,key=repr)
    raise TypeError("Expected values must be JSON-compatible or exact Fractions")
namespace={"check":check,"verify":verify,"export":export,"EXPECTED":{},"VALIDATORS":{},"Fraction":Fraction,"FIELDS":_fields,"__name__":"__main__"}
compiled=compile(CODE,"<answer-oracle>","exec")
with contextlib.redirect_stdout(sys.stderr):
    exec(compiled,namespace,namespace)
    expected,validators=namespace['EXPECTED'],namespace['VALIDATORS']
    if not isinstance(expected,dict) or not isinstance(validators,dict):
        raise TypeError("EXPECTED and VALIDATORS must be dictionaries with exact field-name keys")
    for field,value in expected.items():
        export(field,value)
    for field,predicate in validators.items():verify(field,predicate)
    for field,predicate in _validators:
        actual=json.loads(CANDIDATE)[field]
        binding=hashlib.sha256(json.dumps(actual,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        matched=predicate(copy.deepcopy(actual))
        if type(matched) is not bool:raise TypeError("Witness predicate must return bool")
        _checks.append(dict(field=field,op="predicate",expected={"matched":matched,"value_sha256":binding}))
print("\\nSEA_ANSWER_CHECKS:"+json.dumps({"checks":_checks},default=serialize))
''')
    result=await isolated_probe(ws,wrapper)
    if result['ok']:
        exported=next((l[len('SEA_ANSWER_CHECKS:'):] for l in reversed(result['output'].splitlines())
                       if l.startswith('SEA_ANSWER_CHECKS:')),None)
        if exported is None:raise ValueError('Oracle did not export any answer checks')
        result={**result,'output':exported}
    return result


def canonical(value):
    return json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False)


def machine_fields(answer,keys):
    """Conservative required coverage for structured values and exact scalar literals."""
    result=set()
    for key,value in answer.items():
        if keys and key not in keys:continue
        if not isinstance(value,str):result.add(key)
        elif len(value)<=100:
            try:Fraction(value);result.add(key)
            except (ValueError,ZeroDivisionError):pass
    return result


def bind_answer(answer,checks,keys=()):
    if isinstance(checks,str):checks=AnswerChecks.model_validate_json(checks)
    elif not isinstance(checks,AnswerChecks):checks=AnswerChecks.model_validate(checks)
    value=json.loads(answer)
    if not isinstance(value,dict):raise ValueError('Answer binding requires a JSON object')
    comparisons=[]
    for check in checks.checks:
        if check.field not in value or (keys and check.field not in keys):
            raise ValueError('Oracle field is not an explicit answer field: '+check.field)
        actual=value[check.field];expected=check.expected
        if check.op=='predicate':
            if (not isinstance(expected,dict) or type(expected.get('matched')) is not bool
                    or not isinstance(expected.get('value_sha256'),str)):
                raise ValueError('Malformed witness predicate observation')
            matched=expected['matched'] and expected['value_sha256']==hashlib.sha256(canonical(actual).encode()).hexdigest()
        elif check.op=='fraction':
            try:expected_fraction=Fraction(str(expected))
            except (ValueError,ZeroDivisionError) as exc:raise ValueError('Invalid expected fraction') from exc
            try:matched=not isinstance(actual,bool) and Fraction(str(actual))==expected_fraction
            except (ValueError,ZeroDivisionError):matched=False
        elif check.op=='set_equal':
            if not isinstance(expected,list):raise ValueError('set_equal requires an expected list')
            matched=(isinstance(actual,list) and len({canonical(x) for x in actual})==len(actual)
                and {canonical(x) for x in actual}=={canonical(x) for x in expected})
        elif check.op=='one_of':
            if not isinstance(expected,list):raise ValueError('one_of requires a list of acceptable values')
            matched=any(canonical(actual)==canonical(x) for x in expected)
        elif check.op=='set_one_of':
            if not isinstance(expected,list) or any(not isinstance(x,list) for x in expected):
                raise ValueError('set_one_of requires expected=[[...],[...]]: a list of ALL acceptable lists. '
                                 'You passed one solution or a scalar; pass the computed alternatives collection without indexing it. '
                                 'Alternatively use verify(field, predicate) for a witness validity check.')
            matched=(isinstance(actual,list) and len({canonical(x) for x in actual})==len(actual)
                and any({canonical(x) for x in actual}=={canonical(x) for x in option} for option in expected))
        else:matched=canonical(actual)==canonical(expected)
        comparisons.append({'field':check.field,'op':check.op,'actual':actual,'expected':expected,'matched':matched})
    uncovered=sorted(machine_fields(value,keys)-{c.field for c in checks.checks})
    if uncovered:raise ValueError('Oracle omitted machine-readable answer fields: '+', '.join(uncovered))
    ok=all(c['matched'] for c in comparisons)
    return {'ok':ok,'answer_checks':checks.model_dump(),
        'answer_binding':hashlib.sha256(answer.encode()).hexdigest(),
        'checked_fields':sorted({c.field for c in checks.checks}),
        'output':json.dumps({'comparisons':comparisons,'uncovered_fields':uncovered},ensure_ascii=False),
        'error_kind':None if ok else 'counterexample',
        'verification_scope':'Exact candidate-field comparison with independently executed model-written oracle; not formal proof'}


async def rebind_observation(ws,observation,answer,keys,supervisor,observer,control):
    if any(c['op']=='predicate' for c in observation['answer_checks']['checks']):
        # A new valid witness need not equal the old one. Re-execute the SAME blind
        # validator; never reuse a boolean computed for a different candidate.
        from .verifier import ProbePlan,run_probes
        plan=ProbePlan.model_validate({'probes':[{k:observation[k] for k in ('tool','args','purpose')}],
                                      'limitation':'Re-executed cached witness validator; not formal proof'})
        return (await run_probes(ws,plan,['answer'],supervisor,observer,control,candidate=answer,answer_keys=keys))[0]
    result=bind_answer(answer,observation['answer_checks'],keys)
    result.update(verifier=True,tool='answer',derived_from=observation['id'],purpose=observation['purpose'])
    eid=ws.record(result)
    supervisor.action(observer,'answer',result['ok'],eid)
    return {**observation,**result,'id':eid,'stale':False}
