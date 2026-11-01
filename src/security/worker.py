"""Isolated stdlib-only AST analyst. Reads bounded JSON stdin; executes no project code."""
import ast
from collections import Counter
import json
import sys


def analyze(packet):
    calls, imports, errors = [], [], []
    for name, text in packet['sources'].items():
        try: tree = ast.parse(text)
        except (SyntaxError, ValueError, RecursionError): errors.append(name); continue
        aliases = {}
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                for a in n.names: aliases[a.asname or a.name] = a.name
            if isinstance(n, ast.ImportFrom):
                for a in n.names: aliases[a.asname or a.name] = (n.module or '')+'.'+a.name
        def dotted(n):
            if isinstance(n, ast.Name): return aliases.get(n.id, n.id)
            if isinstance(n, ast.Attribute): return dotted(n.value)+'.'+n.attr
            return ''
        for n in ast.walk(tree):
            if isinstance(n, (ast.Import, ast.ImportFrom)):
                modules = [a.name for a in n.names] if isinstance(n, ast.Import) else [n.module or '']
                imports.extend({'file': name, 'module': m} for m in modules if m.startswith('src.'))
            if isinstance(n, ast.Call):
                call = dotted(n.func)
                kind = ('network' if call in {'urllib.request.urlopen', 'urllib.request.build_opener',
                    'httpx.AsyncClient', 'httpx.Client', 'http.client.HTTPConnection', 'http.client.HTTPSConnection',
                    'requests.get', 'requests.post', 'socket.create_connection'} else
                    'execution' if call in {'subprocess.run', 'subprocess.Popen', 'os.system', 'exec', 'eval'} else '')
                if kind: calls.append({'file': name, 'line': n.lineno, 'kind': kind, 'primitive': call})
    return {'files_analyzed': len(packet['sources']), 'call_counts': dict(Counter(c['kind'] for c in calls)),
            'calls_to_review': calls[:120], 'internal_imports': imports[:150], 'parse_failures': errors,
            'truncated': len(calls)>120 or len(imports)>150,
            'method': 'static AST inventory, not proof of safety or malware detection'}


if __name__ == '__main__':
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (8, 8))
    resource.setrlimit(resource.RLIMIT_FSIZE, (2*1024**2, 2*1024**2))
    raw = sys.stdin.buffer.read(16*1024**2+1)
    if len(raw)>16*1024**2: raise ValueError('Analysis input limit exceeded')
    print(json.dumps(analyze(json.loads(raw))))
