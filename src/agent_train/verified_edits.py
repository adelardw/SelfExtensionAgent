"""A bounded single-edit RLVR environment. It is not a SWE benchmark.

Generated Python expressions are interpreted with a small arithmetic AST, never
eval/exec. Reward checks behavior exhaustively on the stated finite input domain.
The policy receives the task and source; the evaluator stays outside its prompt.
"""
import ast
import json
import operator
import random
from dataclasses import dataclass

OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
       ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod}
CMP = {ast.Lt: operator.lt, ast.LtE: operator.le, ast.Gt: operator.gt,
       ast.GtE: operator.ge, ast.Eq: operator.eq, ast.NotEq: operator.ne}


def evaluate(node, x):
    if isinstance(node, ast.Expression): value = evaluate(node.body, x)
    elif isinstance(node, ast.Name) and node.id == 'x': value = x
    elif isinstance(node, ast.Constant) and type(node.value) is int: value = node.value
    elif isinstance(node, ast.BinOp) and type(node.op) in OPS:
        value = OPS[type(node.op)](evaluate(node.left, x), evaluate(node.right, x))
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = evaluate(node.operand, x) * (-1 if isinstance(node.op, ast.USub) else 1)
    elif isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in CMP:
        value = CMP[type(node.ops[0])](evaluate(node.left, x), evaluate(node.comparators[0], x))
    elif isinstance(node, ast.IfExp):
        value = evaluate(node.body if evaluate(node.test, x) else node.orelse, x)
    elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and
          node.func.id in {'min', 'max', 'abs'} and not node.keywords and 1 <= len(node.args) <= 4):
        value = {'min': min, 'max': max, 'abs': abs}[node.func.id](*[evaluate(a, x) for a in node.args])
    else: raise ValueError('Expression outside the arithmetic language')
    if not isinstance(value, (int, bool)) or abs(value) > 10**9: raise ValueError('Value outside bounds')
    return value


@dataclass(frozen=True)
class EditTask:
    family: str
    instruction: str
    parameters: tuple[int, ...]
    split: str

    @property
    def prompt(self):
        return ('You are editing value.py. Its complete current content is:\n'
                'def transform(x):\n    return x\n\n'
                + self.instruction + ' The input domain is integers from -20 through 20.\n'
                'The source is already available; choose one useful tool action now. '
                'Return only JSON: {"action":"tool","tool":"edit","args":'
                '{"path":"value.py","old":"return x","new":"return YOUR_EXPRESSION"}}. '
                'Use one arithmetic expression with x, integer constants, + - * // %, '
                'comparisons, conditional expressions, min/max/abs. Preserve the function signature.')

    def expected(self, x):
        if self.family == 'clamp': return min(max(x, self.parameters[0]), self.parameters[1])
        if self.family == 'affine': return self.parameters[0] * x + self.parameters[1]
        if self.family == 'distance': return abs(x - self.parameters[0])
        raise ValueError('Unknown task family')

    def verify(self, completion):
        try:
            action = json.loads(completion)
            if set(action) != {'action', 'tool', 'args'} or action['action'] != 'tool' or action['tool'] != 'edit':
                return 0.0
            args = action['args']
            if set(args) != {'path', 'old', 'new'} or args['path'] != 'value.py' or args['old'] != 'return x': return 0.0
            new = args['new']
            if not isinstance(new, str) or not new.startswith('return ') or len(new) > 600: return 0.0
            tree = ast.parse(new[7:], mode='eval')
            if len(list(ast.walk(tree))) > 80: return 0.0
            return float(all(type(value := evaluate(tree, x)) is int and value == self.expected(x)
                             for x in range(-20, 21)))
        except (ValueError, TypeError, KeyError, SyntaxError, ArithmeticError, RecursionError):
            return 0.0


def task_set(seed=811, count=24, split='train'):
    if split not in {'train', 'test'}: raise ValueError('Use train or test')
    rng = random.Random(seed)
    if split == 'test':
        # Entire task family withheld, not random individual steps of one trace.
        return [EditTask('distance', f'Return the absolute distance between x and {n}.', (n,), split)
                for n in rng.sample(range(-15, 16), min(count, 31))]
    tasks = []
    for i in range(count):
        if i % 2:
            a, b = rng.choice([-4, -3, -2, 2, 3, 4]), rng.randrange(-12, 13)
            tasks.append(EditTask('affine', f'Return {a} times x plus ({b}).', (a, b), split))
        else:
            lo, hi = rng.randrange(-15, 0), rng.randrange(1, 16)
            tasks.append(EditTask('clamp', f'Clamp x into the inclusive range [{lo}, {hi}].', (lo, hi), split))
    return tasks
