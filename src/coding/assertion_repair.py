"""Experimental mechanical binding of blind predictions to NEW scalar assertions.

Not an oracle: predictions must also agree with an independent execution. This
module never writes files and never accepts edits to preexisting test methods.
Unsupported control flow, fixtures or expressions fail closed.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
import json
import math


def scalar(value):
    return (value is None or type(value) in (bool,int,str)
            or type(value) is float and math.isfinite(value))


@dataclass
class AssertionPlan:
    source:str
    class_name:str
    method_name:str
    module:str
    constructor:str
    variable:str
    operations:list
    assertions:list

    def blind_input(self):
        return {'operations':[[o['name'],o['args'],o['kwargs']] for o in self.operations],
                'predict_steps':[a['step'] for a in self.assertions]}

    def masked_test(self):
        """Keep familiar source syntax while withholding every assertion value."""
        changed=self.source
        for i,site in reversed(list(enumerate(self.assertions,1))):
            value='EXPECTED_'+str(i)
            if site['kind']=='assertIsNone':value='self.assertEqual('+site['expression']+', '+value+')'
            changed=changed[:site['start']]+value+changed[site['end']:]
        cls=next(n for n in ast.parse(changed).body if isinstance(n,ast.ClassDef) and n.name==self.class_name)
        node=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==self.method_name)
        from textwrap import dedent
        return dedent('\n'.join(changed.splitlines()[node.lineno-1:node.end_lineno]))

    def observation_program(self,reference_code=None):
        # Parameters are JSON data, never interpolated into executable expressions.
        prefix=('import importlib,json\nmodule=importlib.import_module('+repr(self.module)+')\n'
                if reference_code is None else reference_code+'\nimport json\n')
        constructor='getattr(module,op["name"])' if reference_code is None else 'globals()[op["name"]]'
        return (prefix
            +'operations=json.loads('+repr(json.dumps(self.operations))+')\n'
            +'obj=None;values=[]\n'
            +'for i,op in enumerate(operations):\n'
            +'    if i==0: obj='+constructor+'(*op["args"],**op["kwargs"]);value=None\n'
            +'    else: value=getattr(obj,op["name"])(*op["args"],**op["kwargs"])\n'
            +'    values.append(value)\n'
            +'print("SEA_ASSERTION_VALUES "+json.dumps(values,allow_nan=False))\n')

    def proposal(self, predictions, observed):
        if len(predictions)!=len(self.assertions) or len(observed)!=len(self.operations):
            raise ValueError('Incomplete predictions or executed observations')
        changes=[]
        for site,value in zip(self.assertions,predictions):
            actual=observed[site['step']]
            if not scalar(value) or type(actual) is not type(value) or actual!=value:
                raise ValueError('Blind prediction disagrees with actual execution; do not rewrite tests')
            if site['kind']=='assertEqual':
                original=self.source[site['start']:site['end']]
                replacement=repr(value)
            else:
                original=self.source[site['start']:site['end']]
                replacement=original if value is None else 'self.assertEqual('+site['expression']+', '+repr(value)+')'
            if original!=replacement:changes.append((site['start'],site['end'],replacement))
        changed=self.source
        for start,end,value in sorted(changes,reverse=True):changed=changed[:start]+value+changed[end:]
        ast.parse(changed)
        return changed


def prepare_assertion_plan(source, original, class_name, method_name):
    tree=ast.parse(source);baseline=ast.parse(original)
    classes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==class_name]
    if len(classes)!=1:raise ValueError('Ambiguous test class')
    cls=classes[0]
    if cls.decorator_list or len(cls.bases)!=1 or ast.unparse(cls.bases[0])!='unittest.TestCase':
        raise ValueError('Only an undecorated standard unittest.TestCase is supported')
    if any(isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in
            {'setUp','tearDown','setUpClass','tearDownClass','__init__','run'} for n in cls.body):
        raise ValueError('Fixtures or custom test execution are not supported')
    if any(isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in {'setUpModule','tearDownModule'} for n in tree.body):
        raise ValueError('Module fixtures are not supported')
    for previous in baseline.body:
        if isinstance(previous,ast.ClassDef) and previous.name==class_name:
            if any(isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==method_name for n in previous.body):
                raise ValueError('Cannot rewrite an original test method')
    methods=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==method_name]
    if len(methods)!=1 or methods[0].decorator_list:raise ValueError('Ambiguous or decorated test method')
    method=methods[0];operations=[];assertions=[];variable=None;module=None;constructor=None
    lines=source.splitlines(keepends=True)
    def span(node):
        # Python AST columns count UTF-8 bytes, while string slices count characters.
        def offset(line,column):return sum(map(len,lines[:line-1]))+len(lines[line-1].encode()[:column].decode())
        return offset(node.lineno,node.col_offset),offset(node.end_lineno,node.end_col_offset)
    def call_data(call):
        if not isinstance(call,ast.Call) or any(k.arg is None for k in call.keywords):
            raise ValueError('Only direct calls with constant arguments are supported')
        args=[ast.literal_eval(a) for a in call.args];kwargs={k.arg:ast.literal_eval(k.value) for k in call.keywords}
        if not all(scalar(v) for v in [*args,*kwargs.values()]):raise ValueError('Only scalar inputs are supported')
        if variable is None:
            if not isinstance(call.func,ast.Name):raise ValueError('Constructor must be a directly imported class')
            name=call.func.id
        else:
            if not isinstance(call.func,ast.Attribute) or not isinstance(call.func.value,ast.Name) or call.func.value.id!=variable:
                raise ValueError('Calls must use the same constructed object')
            name=call.func.attr
        operations.append({'name':name,'args':args,'kwargs':kwargs})
    for statement in method.body:
        if variable is None:
            if not isinstance(statement,ast.Assign) or len(statement.targets)!=1 or not isinstance(statement.targets[0],ast.Name):
                raise ValueError('Test must start by constructing one object')
            call_data(statement.value);constructor=operations[0]['name'];variable=statement.targets[0].id
            imports=[n for n in tree.body if isinstance(n,ast.ImportFrom) and n.level==0 and n.module and
                any((a.asname or a.name)==constructor and a.name==constructor for a in n.names)]
            if len(imports)!=1:raise ValueError('Constructor import must be unambiguous')
            module=imports[0].module
            continue
        value=statement.value if isinstance(statement,(ast.Expr,ast.Assign)) else None
        if isinstance(statement,ast.Assign) and (len(statement.targets)!=1 or not isinstance(statement.targets[0],ast.Name) or statement.targets[0].id!='_'):
            raise ValueError('Only ignored intermediate assignments are supported')
        is_assert=(isinstance(value,ast.Call) and isinstance(value.func,ast.Attribute) and
            isinstance(value.func.value,ast.Name) and value.func.value.id=='self')
        if not is_assert:
            call_data(value);continue
        kind=value.func.attr
        if kind not in {'assertEqual','assertIsNone'} or value.keywords or len(value.args)!=(2 if kind=='assertEqual' else 1):
            raise ValueError('Only assertEqual(actual, literal) and assertIsNone(actual) are supported')
        if kind=='assertEqual' and not scalar(ast.literal_eval(value.args[1])):
            raise ValueError('Only scalar expectations are supported')
        call_data(value.args[0])
        start,end=span(value.args[1] if kind=='assertEqual' else value)
        a,b=span(value.args[0])
        assertions.append({'step':len(operations)-1,'kind':kind,'start':start,'end':end,'expression':source[a:b]})
    if not assertions or len(operations)>32:raise ValueError('Requires 1..32 operations and at least one assertion')
    return AssertionPlan(source,class_name,method_name,module,constructor,variable,operations,assertions)
