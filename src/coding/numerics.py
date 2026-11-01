"""Exact symbolic calculations with an AST interpreter; expressions are never eval'd."""
from __future__ import annotations
import ast
import json
import operator
import math
import sys
import sympy as s

FUNCTIONS={name:getattr(s,name) for name in ('sin','cos','tan','exp','log','sqrt','Abs','factorial','binomial','Rational')}
CONSTANTS={'pi':s.pi,'E':s.E,'I':s.I,'oo':s.oo}
OPS={ast.Add:operator.add,ast.Sub:operator.sub,ast.Mult:operator.mul,ast.Div:operator.truediv,ast.Pow:operator.pow,ast.Mod:operator.mod}


def expression(text,symbols):
    if not isinstance(text,str) or len(text)>6000: raise ValueError('Expression must be at most 6000 characters')
    tree=ast.parse(text.replace('^','**'),mode='eval')
    if sum(1 for _ in ast.walk(tree))>500: raise ValueError('Expression AST too large')
    def build(node,depth=0):
        if depth>60: raise ValueError('Expression nesting too deep')
        if isinstance(node,ast.Expression): return build(node.body,depth+1)
        if isinstance(node,ast.Constant) and type(node.value) in (int,float):
            return s.Rational(str(node.value))
        if isinstance(node,ast.Name):
            if node.id in CONSTANTS: return CONSTANTS[node.id]
            if node.id.startswith('_') or not node.id.isidentifier(): raise ValueError('Invalid symbol')
            return symbols.setdefault(node.id,s.Symbol(node.id))
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.UAdd,ast.USub)):
            return build(node.operand,depth+1)*(1 if isinstance(node.op,ast.UAdd) else -1)
        if isinstance(node,ast.BinOp) and type(node.op) in OPS:
            left,right=build(node.left,depth+1),build(node.right,depth+1)
            if isinstance(node.op,ast.Pow) and right.is_number and abs(right)>10000:
                raise ValueError('Exponent budget exceeded')
            if isinstance(node.op,ast.Pow) and right.is_Integer and right>0:
                if left.is_Integer and abs(int(left)).bit_length()*int(right)>1000000:
                    raise ValueError('Integer size budget exceeded')
                if left.is_Add and math.comb(len(left.args)+int(right)-1,int(right))>20000:
                    raise ValueError('Polynomial expansion budget exceeded')
            return OPS[type(node.op)](left,right)
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and not node.keywords:
            if node.func.id not in FUNCTIONS:
                raise ValueError('Unknown mathematical function '+node.func.id+'. Available: '+', '.join(sorted(FUNCTIONS))+
                                 '. Use binomial(n,k) for combinations, factorial(n) for factorials; use experiment for custom code.')
            args=[build(arg,depth+1) for arg in node.args]
            if node.func.id in {'factorial','binomial'} and any(v.is_number and abs(v)>10000 for v in args):
                raise ValueError('Combinatorics budget exceeded')
            return FUNCTIONS[node.func.id](*args)
        raise ValueError('Only arithmetic, symbols and whitelisted mathematical functions are allowed')
    return build(tree)


def validate_arguments(args):
    args=dict(args)
    if args.get('op','simplify') not in {'simplify','factor','expand','solve','diff','integrate','limit','sum','identity'}:
        raise ValueError('Unknown math operation; use simplify/factor/expand/solve/diff/integrate/limit/sum/identity')
    if any('=' in str(args.get(field,'')) for field in ('expr','rhs')):
        raise ValueError('expr and rhs must be expressions, not equations containing =. '
                         'For x+1=3 use {"op":"solve","expr":"x+1","rhs":"3","var":"x"}. '
                         'Use experiment with exact arithmetic for systems of equations or matrices.')
    if 'bounds' in args:
        bounds=args['bounds']
        if isinstance(bounds,str): bounds=json.loads(bounds)
        if not isinstance(bounds,list) or len(bounds)!=2:
            raise ValueError('bounds must be [lower, upper], for example [1, 100]')
        args['bounds']=bounds
    if args.get('op')=='sum' and ('bounds' not in args or 'var' not in args):
        raise ValueError('sum requires expr (summand), var (symbol), bounds [lower,upper]')
    assumptions=args.get('assumptions',{})
    if not isinstance(assumptions,dict): raise ValueError('assumptions must be an object of symbol-property objects')
    return args


def calculate(args):
    args=validate_arguments(args)
    assumptions=args.get('assumptions',{})
    symbols={}
    for name,properties in assumptions.items():
        if not name.isidentifier() or name.startswith('_') or not isinstance(properties,dict): raise ValueError('Invalid symbol assumptions')
        if set(properties)-{'real','positive','integer','nonnegative','nonzero'} or not all(type(v) is bool for v in properties.values()):
            raise ValueError('Unsupported assumptions')
        symbols[name]=s.Symbol(name,**properties)
    parse=lambda value:expression(str(value),symbols)
    expr=parse(args.get('expr','0'))
    variable=parse(args.get('var','x'))
    if not isinstance(variable,s.Symbol): raise ValueError('var must name a symbol')
    op=args.get('op','simplify')
    scope='exact symbolic computation under stated assumptions'
    if op in {'simplify','factor','expand'}: result=getattr(s,op)(expr)
    elif op=='solve': result=s.solve(expr-parse(args.get('rhs','0')),variable)
    elif op=='diff': result=s.diff(expr,variable)
    elif op=='integrate':
        bounds=args.get('bounds')
        result=s.integrate(expr,(variable,parse(bounds[0]),parse(bounds[1])) if bounds else variable)
    elif op=='limit': result=s.limit(expr,variable,parse(args['point']),dir=args.get('dir','+'))
    elif op=='sum': result=s.summation(expr,(variable,parse(args['bounds'][0]),parse(args['bounds'][1])))
    elif op=='identity':
        difference=s.simplify(expr-parse(args.get('rhs','0')))
        result={'difference':str(difference),'verified':difference==0,
                'status':'symbolic_identity' if difference==0 else 'not_established'}
    else: raise ValueError('Unknown operation')
    unresolved=isinstance(result,s.Basic) and result.has(s.Integral,s.Sum,s.Limit)
    return {'result':str(result),'resolved':not unresolved,'operation':op,'assumptions':assumptions,'scope':scope,
            'limitation':'A symbolic result is not a general proof of an arbitrary theorem. Finite numeric tests are not proofs.'}

if __name__=='__main__':
    try: print(json.dumps({'ok':True,**calculate(json.loads(sys.argv[1]))},ensure_ascii=False))
    except Exception as exc:
        print(json.dumps({'ok':False,'error':type(exc).__name__+': '+str(exc)}));sys.exit(1)
