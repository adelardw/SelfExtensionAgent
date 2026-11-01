"""Host command confinement. Source edits use scoped edit/write tools; commands get scratch space.

No silent unsandboxed fallback. Docker/host networking and project-write commands
require a separate operator approval, even when the ordinary coding loop is automatic.
"""
import json
import os
import platform
import re
import shutil
import sys
import tempfile
from pathlib import Path


def secret_pattern(root, names=None, parts=None):
    from src.skills.code.code import _SECRET_NAMES, _SECRET_SUBSTR
    def literal(value):
        return ''.join('['+c.lower()+c.upper()+']' if c.isalpha() else '['+c+']' if c in '.-' else re.escape(c)
                       for c in value)
    names = '|'.join(literal(n) for n in sorted(_SECRET_NAMES if names is None else names))
    parts = '|'.join(literal(n) for n in (_SECRET_SUBSTR if parts is None else parts))
    # Seatbelt regex literals preserve backslashes; bracket literals avoid double escaping.
    root_pattern = ''.join('[[]' if c == '[' else '[]]' if c == ']' else
                           '['+c+']' if c in '.+?*$^(){}|' else c for c in str(root))
    if '\\' in root_pattern:
        raise RuntimeError('Unsupported sandbox root path')
    alternatives = ([names] if names else []) + (['[^/]*('+parts+')[^/]*'] if parts else [])
    return '^'+root_pattern+'/([^/]+/)*('+'|'.join(alternatives)+')(/|$)'


def secret_paths(root):
    """Mount masks on Linux; fail closed if scanning the project cannot be bounded."""
    from src.skills.code.code import _is_secret_file
    count = 0
    for base, directories, files in os.walk(root, followlinks=False):
        for name in list(directories):
            path = Path(base)/name
            if name in {'.sea','.git'} or _is_secret_file(path):
                directories.remove(name)
                yield path
        for name in files:
            count += 1
            if count > 100000:
                raise RuntimeError('Project too large for bounded secret masks; use a container environment.')
            path = Path(base)/name
            if _is_secret_file(path):
                yield path


def clean_environment(scratch):
    env = {k:v for k,v in os.environ.items() if not re.search(r'key|token|secret|password|credential|authorization|cookie|ssh_auth_sock', k, re.I)}
    env.update(TMPDIR=str(scratch)+'/', TMP=str(scratch), TEMP=str(scratch),
               PYTHONDONTWRITEBYTECODE='1', GIT_OPTIONAL_LOCKS='0', SEA_SCRATCH=str(scratch))
    env.update(XDG_CACHE_HOME=str(scratch/'cache'), UV_CACHE_DIR=str(scratch/'uv'),
               CLANG_MODULE_CACHE_PATH=str(scratch/'clang'), npm_config_cache=str(scratch/'npm'),
               GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_SYSTEM=os.devnull)
    return env


def prefix(root, scratch, output, *, project_write=False, network=False, read_git=False):
    from src.security.integrity import protected_paths
    root, scratch = root.resolve(), scratch.resolve()
    system = platform.system()
    if system == 'Darwin' and shutil.which('sandbox-exec'):
        q = lambda p: json.dumps(str(p))
        home = Path.home().resolve()
        runtime = Path(sys.executable).resolve().parent.parent
        profile = '(version 1)(allow default)(deny file-write*)'
        profile += '(deny file-read* (subpath "/Users") (subpath "/private/var/folders"))'
        profile += '(allow file-read-metadata)'
        profile += f'(allow file-read* (subpath {q(root)}) (subpath {q(runtime)}) (subpath {q(sys.prefix)}) (subpath {q(scratch)}))'
        profile += f'(allow file-write* (subpath {q(scratch)}) (literal {q(output)}))'
        profile += '(allow file-write* (literal "/dev/null"))'
        temp = Path(tempfile.gettempdir()).resolve()
        cache_pattern = '^' + re.escape(str(temp)) + '/xcrun_db(-[^/]+)?$'
        profile += f'(allow file-read* file-write* (regex #{json.dumps(cache_pattern)}))'
        if project_write:
            profile += f'(allow file-write* (subpath {q(root)}))'
        if not network:
            profile += '(deny network*)'
        # These denials are deliberately more specific than the project allowance.
        profile += f'(deny file-read* (subpath {q(root/".sea")}))'
        if not read_git: profile += f'(deny file-read* (subpath {q(root/".git")}))'
        # Seatbelt has a small regex literal limit: long disposable workspace paths
        # must not make a growing secret list invalidate the entire sandbox profile.
        from src.skills.code.code import _SECRET_NAMES, _SECRET_SUBSTR
        names = sorted(_SECRET_NAMES)
        patterns = [secret_pattern(root, names[i:i+4], ()) for i in range(0, len(names), 4)]
        patterns += [secret_pattern(root, (), _SECRET_SUBSTR[i:i+4]) for i in range(0, len(_SECRET_SUBSTR), 4)]
        for pattern in patterns:
            if len(pattern.encode()) > 950: raise RuntimeError('Sandbox project path is too long')
            profile += '(deny file-read* (regex #'+json.dumps(pattern)+'))'
        profile += f'(deny file-write* (subpath {q(root/".git")}) (subpath {q(root/".sea")}))'
        for path in protected_paths():
            profile += f'(deny file-write* (subpath {q(path)}))'
        # stdout is a parent-opened log, not authority to edit checkpoint metadata.
        profile += f'(allow file-write* (literal {q(output)}))'
        return ['sandbox-exec', '-p', profile]
    if system == 'Linux' and shutil.which('bwrap'):
        runtime = str(Path(sys.executable).resolve().parent.parent)
        cmd = ['bwrap']
        # Construct a minimal root rather than exposing other users' files read-only.
        for path in ('/usr','/bin','/sbin','/lib','/lib64','/opt'):
            if Path(path).exists(): cmd += ['--ro-bind',path,path]
        for path in ('/etc/ld.so.cache','/etc/localtime','/etc/nsswitch.conf','/etc/hosts',
                     '/etc/resolv.conf','/etc/ssl/certs','/etc/os-release'):
            if Path(path).exists(): cmd += ['--ro-bind',path,path]
        cmd += ['--proc','/proc','--dev','/dev','--tmpfs','/tmp',
               '--ro-bind',runtime,runtime,'--ro-bind',str(Path(sys.prefix).resolve()),str(Path(sys.prefix).resolve()),
               '--bind' if project_write else '--ro-bind',str(root),str(root),
               '--bind',str(scratch),str(scratch),'--bind',str(output),str(output),
               '--unshare-pid','--unshare-ipc','--die-with-parent','--new-session']
        if not network: cmd += ['--unshare-net']
        for path in protected_paths():
            if path.exists() and path.is_relative_to(root):
                cmd += ['--ro-bind', str(path), str(path)]
        for p in secret_paths(root):
            if read_git and p == root/'.git': continue
            if p.is_file(): cmd += ['--ro-bind','/dev/null',str(p)]
            elif p.is_dir(): cmd += ['--tmpfs',str(p)]
        return cmd
    raise RuntimeError('No supported host sandbox. Use a Docker task environment or install bubblewrap on Linux.')
