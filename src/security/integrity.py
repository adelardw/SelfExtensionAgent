"""Operator-enrolled source baseline; model tools cannot enroll or update it."""
import hashlib
import json
from pathlib import Path

from .events import record
from .sensitive import SecurityViolation

ROOT = Path(__file__).resolve().parents[2]


def source_hashes(root=ROOT):
    root = Path(root).resolve()
    extensions = {'.py', '.md', '.json', '.yaml', '.yml', '.js', '.tsx', '.ts', '.html', '.css'}
    files = []
    for folder in ('src', 'frontend/src', 'frontend/dist'):
        directory = root/folder
        if directory.is_symlink(): raise ValueError('Symlinked source tree')
        for p in directory.rglob('*'):
            if p.is_symlink(): raise ValueError('Symlinked source entry')
            if p.is_file() and p.suffix in extensions: files.append(p)
    files.sort()
    files += [root/n for n in ('config.yml', 'pyproject.toml', 'uv.lock', 'main.py', 'desktop.py') if (root/n).is_file()]
    if len(files) > 2000: raise ValueError('Source inventory limit exceeded')
    result = {}
    for p in files:
        if p.is_symlink() or not p.resolve().is_relative_to(root) or p.stat().st_size > 2*1024**2:
            raise ValueError('Unsafe or oversized source entry')
        data = p.read_bytes()
        if p == root/'src/skills/registry.json':
            # Counters are runtime data. Trust flags, definitions and paths remain covered.
            registry = json.loads(data)
            if not isinstance(registry, dict) or not all(isinstance(v, dict) for v in registry.values()):
                raise ValueError('Invalid skill registry')
            for meta in registry.values():
                for key in ('uses', 'wins', 'last_used_at'): meta.pop(key, None)
            data = json.dumps(registry, sort_keys=True, ensure_ascii=False).encode()
        result[str(p.relative_to(root))] = hashlib.sha256(data).hexdigest()
    return result


def inspect(root=ROOT):
    root = Path(root).resolve()
    baseline = root/'.sea/security/integrity.json'
    if not baseline.exists(): return {'status': 'unenrolled', 'changed': [], 'detail': 'Operator enrollment required'}
    try:
        if baseline.is_symlink() or baseline.parent.is_symlink() or (root/'.sea').is_symlink(): raise ValueError()
        if baseline.stat().st_size > 2*1024**2: raise ValueError()
        expected = json.loads(baseline.read_text())
        if not isinstance(expected, dict) or expected.get('version') != 1 or not isinstance(expected.get('files'), dict): raise ValueError()
        actual = source_hashes(root)
        changed = [p for p in sorted(set(expected['files'])|set(actual)) if expected['files'].get(p) != actual.get(p)]
        return {'status': 'changed' if changed else 'verified', 'changed': changed[:100], 'files': len(actual)}
    except (OSError, ValueError, TypeError):
        return {'status': 'invalid', 'changed': [], 'detail': 'Baseline cannot be verified'}


def enroll(root=ROOT):
    root = Path(root).resolve()
    path = root/'.sea/security/integrity.json'
    if (root/'.sea').is_symlink() or path.parent.is_symlink(): raise ValueError('Unsafe baseline directory')
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    # No automatic re-enrollment: a changed baseline requires operator review/removal.
    with path.open('x') as f: json.dump({'version': 1, 'files': source_hashes(root)}, f, indent=2)
    path.chmod(0o600)
    return inspect(root)


def require_integrity():
    status = inspect()
    if status['status'] in {'changed', 'invalid'}:
        record('integrity', 'source_changed')
        raise SecurityViolation('Agent source integrity failed; stop and request operator review')


def protected_paths():
    return [ROOT/n for n in ('src', 'frontend', '.sea', 'config.yml', 'pyproject.toml', 'uv.lock', 'main.py', 'desktop.py')]


def protect_agent_write(path):
    """Agent self-modification cannot rewrite its own runtime/security boundary."""
    path = Path(path).resolve()
    if any(path == p or path.is_relative_to(p) for p in protected_paths()):
        record('filesystem', 'protected_runtime')
        raise SecurityViolation('Agent runtime is protected; prepare a patch for operator review outside the live source tree')
