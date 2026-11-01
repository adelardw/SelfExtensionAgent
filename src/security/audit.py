"""Read-only host inventory plus sandboxed architecture analysis; no LLM/network calls."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from .integrity import ROOT, inspect, source_hashes


def network_snapshot():
    if not shutil.which('lsof'):
        return {'status': 'unavailable', 'detail': 'lsof not installed; no network probing fallback'}
    # Passive socket metadata for this OS user, not packet capture or scanning.
    try:
        result = subprocess.run(['lsof', '-nP', '-a', '-u', str(os.getuid()), '-i', '-FpcnT'],
            capture_output=True, text=True, timeout=5)
        if result.returncode not in (0, 1): raise ValueError('lsof failed')
        current_pid = None
        connections, listeners, wildcard, own = 0, 0, 0, 0
        for line in result.stdout[:262144].splitlines():
            if line.startswith('p'): current_pid = line[1:]
            if line.startswith('n'):
                connections += 1
                if current_pid == str(os.getpid()): own += 1
                if line[1:].startswith(('*:', '[::]:', '0.0.0.0:')): wildcard += 1
            if line == 'TST=LISTEN': listeners += 1
        return {'status': 'observed', 'scope': 'current OS user socket metadata',
            'socket_records': connections, 'listeners': listeners, 'wildcard_bindings': wildcard,
            'current_process_sockets': own, 'truncated': len(result.stdout)>262144,
            'packet_payloads': False, 'tls_decryption': False, 'remote_probes': False}
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return {'status': 'unavailable', 'detail': 'Socket inventory unavailable; no raw diagnostics exported'}


def isolated_analysis(root):
    from src.coding.containment import prefix
    source = {}
    total = 0
    for name in source_hashes(root):
        if not name.endswith('.py'): continue
        text = (root/name).read_text()
        total += len(text.encode())
        if total > 8*1024**2: raise ValueError('Source analysis byte limit exceeded')
        source[name] = text
    with tempfile.TemporaryDirectory(prefix='sea-security-audit-') as folder:
        scratch = Path(folder).resolve()
        worker = scratch/'worker.py'
        worker.write_bytes(Path(__file__).with_name('worker.py').read_bytes())
        output = scratch/'result.json'
        output.touch(mode=0o600)
        command = prefix(scratch, scratch, output) + [sys.executable, '-I', str(worker)]
        env = {'PATH': os.defpath, 'LANG': 'C.UTF-8', 'PYTHONNOUSERSITE': '1', 'PYTHONDONTWRITEBYTECODE': '1'}
        with output.open('w') as out:
            result = subprocess.run(command, input=json.dumps({'sources': source}), text=True,
                stdout=out, stderr=subprocess.DEVNULL, cwd=scratch, env=env, timeout=15)
        if result.returncode: raise ValueError('Isolated analyst failed; no unsandboxed fallback')
        return json.loads(output.read_text())


def audit(root=ROOT, *, network=False):
    root = Path(root).resolve()
    findings = []
    def finding(code, severity, detail, action):
        findings.append({'code': code, 'severity': severity, 'detail': detail, 'action': action})
    integrity = inspect(root)
    if integrity['status'] != 'verified':
        finding('source_integrity', 'critical' if integrity['status'] in {'changed','invalid'} else 'warning',
            'Source baseline is '+integrity['status'], 'Operator must review source changes and enroll a trusted baseline')
    token = os.getenv('AGENT_API_TOKEN', '')
    from .http import loopback
    host = os.getenv('AGENT_BIND_HOST', '127.0.0.1')
    local = host == 'localhost' or loopback(host)
    if len(token)<32:
        finding('api_authentication', 'warning' if local else 'critical',
            'No strong API token configured; local OS processes remain trusted' if local else 'Remote bind without strong authentication',
            'Set AGENT_API_TOKEN (32+ random characters); remote access also requires TLS and explicit allowed hosts')
    if not os.getenv('AGENT_WEB_ALLOWED_HOSTS', '').strip():
        finding('public_egress', 'warning', 'Public HTTP hosts are not restricted to an operator allowlist',
            'Set AGENT_WEB_ALLOWED_HOSTS for sensitive projects; credential matching cannot detect all data exfiltration')
    if os.getenv('AGENT_ALLOW_RISKY_SKILLS') == '1':
        finding('risky_skills', 'critical', 'Risky generated skills override is enabled', 'Remove AGENT_ALLOW_RISKY_SKILLS')
    metadata = []
    for name in ('.env', 'config.local.yml', 'data/browser_bridge.token', 'data/memory.db', 'data/checkpoints.db'):
        path = root/name
        if not path.exists(): continue
        mode = path.stat().st_mode & 0o777
        metadata.append({'path': name, 'mode': oct(mode), 'symlink': path.is_symlink()})
        if mode & 0o077 or path.is_symlink():
            finding('private_file_permissions', 'warning', name+' has broad permissions or is a symlink', 'Owner should review and restrict this file to 0600')
    try:
        architecture = isolated_analysis(root)
        sandbox = {'status': 'verified', 'network': 'denied', 'host_secrets': 'not provided', 'project_code': 'parsed, never executed'}
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        architecture = {}
        sandbox = {'status': 'unavailable', 'error_type': type(exc).__name__}
        finding('analyst_isolation', 'critical', 'Sandboxed analysis could not complete', 'Fix sandbox support; analysis never falls back to unrestricted execution')
    sockets = network_snapshot() if network else {'status': 'not_requested'}
    if sockets.get('wildcard_bindings'):
        finding('network_exposure', 'warning', 'Current-user processes have wildcard socket bindings', 'Review listening services and firewall rules; this does not by itself establish an intrusion')
    if sockets.get('status') == 'unavailable':
        finding('network_visibility', 'warning', sockets['detail'], 'Inspect OS networking separately')
    return {'version': 1, 'scope': 'local agent architecture, configuration and optional passive user-level socket inventory',
        'status': 'attention' if findings else 'no_findings_in_checked_scope', 'findings': findings,
        'integrity': integrity, 'analyst': sandbox, 'architecture': architecture, 'network': sockets,
        'private_file_metadata': metadata,
        'boundaries': ['API -> runtime', 'model -> tools', 'tools -> workspace/process sandbox',
                       'web/data tools -> pinned public HTTP', 'runtime -> local data/cloud provider'],
        'limitations': ['Not an antivirus, vulnerability-database scan, packet IDS, or proof of security.',
            'No packet contents, TLS decryption, privileged process inventory, or foreign-network probes.',
            'Core browser/device/MCP/provider transports require separate review; this HTTP guard is not a host-wide firewall.',
            'Same-user native malware, in-memory compromise, kernel/admin compromise or replacement of the guard can bypass application controls.',
            'The local source baseline detects changes since operator enrollment; it cannot prove that enrolled code was benign.']}


def compact(report):
    """Keep architecture sources/import lists out of the model's working context."""
    return {k: report[k] for k in ('status', 'findings', 'integrity', 'analyst', 'network', 'limitations')} | {
        'architecture': {k: report.get('architecture', {}).get(k) for k in ('files_analyzed', 'call_counts', 'parse_failures')}}


def make_audit_tool():
    from langchain_core.tools import StructuredTool
    from pydantic import BaseModel, ConfigDict
    class NoArguments(BaseModel):
        model_config = ConfigDict(extra='forbid')
    async def run():
        import asyncio
        return json.dumps(compact(await asyncio.to_thread(audit, network=True)), ensure_ascii=False)
    return StructuredTool.from_function(name='security_audit', coroutine=run, args_schema=NoArguments,
        description='Read-only self-security audit: isolated static code inventory, integrity, configuration and passive local sockets. No targets, packet capture, LLM calls, repairs or enrollment.')
