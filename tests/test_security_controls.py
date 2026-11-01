import asyncio
import base64
import json
from pathlib import Path
import socket

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.security import egress, events, integrity
from src.security.http import AgentHTTPGuard, validate_bind
from src.security.sensitive import SecurityViolation, guard_text


@pytest.fixture(autouse=True)
def local_security_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(events, 'ROOT', tmp_path)
    for key in ('AGENT_API_TOKEN', 'AGENT_ALLOWED_HOSTS', 'AGENT_WEB_ALLOWED_HOSTS'):
        monkeypatch.delenv(key, raising=False)


def client(peer='127.0.0.1', base='http://localhost'):
    app = FastAPI()
    app.add_middleware(AgentHTTPGuard)
    @app.get('/private')
    def private(): return {'value': 'private user data'}
    return TestClient(app, base_url=base, client=(peer, 10000))


def test_local_ui_works_and_has_browser_exfiltration_headers():
    response = client().get('/private')
    assert response.status_code == 200
    assert response.headers['cache-control'] == 'no-store'
    assert "img-src 'self' data: blob:" in response.headers['content-security-policy']
    assert response.headers['referrer-policy'] == 'no-referrer'


def test_request_limit_applies_before_route_parses_body():
    response = client().post('/upload', headers={'Content-Length': str(50*1024**2)})
    assert response.status_code == 413


@pytest.mark.parametrize('headers', [
    {'Host': 'attacker.example'}, {'Host': 'localhost.attacker.example'},
    {'Origin': 'https://attacker.example'}, {'Origin': 'null'},
    {'Origin': 'http://localhost:9999'}, {'Sec-Fetch-Site': 'cross-site'},
    {'Host': 'user@localhost'}, {'Host': 'localhost/path'},
])
def test_rebinding_and_cross_site_requests_are_rejected(headers):
    response = client().get('/private', headers=headers)
    assert response.status_code == 403 and 'private user data' not in response.text


def test_real_remote_peer_cannot_spoof_loopback_with_forwarded_headers():
    response = client(peer='192.168.1.20').get('/private', headers={'X-Forwarded-For':'127.0.0.1'})
    assert response.status_code == 403


def test_optional_token_applies_to_local_and_remote_requests(monkeypatch):
    token = 'test-secret-owner-token-0123456789abcdef'
    monkeypatch.setenv('AGENT_API_TOKEN', token)
    assert client().get('/private').status_code == 401
    assert client().get('/private', headers={'Authorization':'Bearer wrong'}).status_code == 401
    assert client().get('/private', headers={'Authorization':'Bearer '+token}).status_code == 200
    basic = base64.b64encode(('sea:'+token).encode()).decode()
    assert client().get('/private', headers={'Authorization':'Basic '+basic}).status_code == 200
    assert client(peer='192.168.1.20').get('/private', headers={'Authorization':'Bearer '+token}).status_code == 403
    assert client(peer='192.168.1.20', base='https://localhost').get('/private', headers={'Authorization':'Bearer '+token}).status_code == 200


def test_unsafe_bind_fails_closed(monkeypatch):
    validate_bind('127.0.0.1')
    with pytest.raises(RuntimeError): validate_bind('0.0.0.0')
    monkeypatch.setenv('AGENT_API_TOKEN', 'short')
    with pytest.raises(RuntimeError): validate_bind('::')
    monkeypatch.setenv('AGENT_API_TOKEN', 'a'*32)
    validate_bind('0.0.0.0')


def test_integrity_failure_blocks_even_authenticated_api(monkeypatch):
    monkeypatch.setattr(integrity, 'require_integrity', lambda: (_ for _ in ()).throw(SecurityViolation('changed')))
    assert client().get('/private').status_code == 503


def resolver(address):
    return [(socket.AF_INET6 if ':' in address else socket.AF_INET, socket.SOCK_STREAM, 6, '', (address, 443))]


@pytest.mark.parametrize('address', ['127.0.0.1', '10.1.2.3', '169.254.169.254', '192.168.1.1',
    '::1', '::ffff:127.0.0.1', '64:ff9b::a00:1', '224.0.0.1', '100.64.0.1', 'fc00::1'])
def test_dns_private_multicast_and_metadata_answers_are_blocked(monkeypatch, address):
    monkeypatch.setattr(socket, 'getaddrinfo', lambda *a, **k: resolver(address))
    with pytest.raises(SecurityViolation): egress.destination('https://example.com')


@pytest.mark.parametrize('url', ['file:///etc/passwd', 'https://u:p@example.com', 'https://example.com:8000',
    'https://example.com\\@localhost', 'https://example.com/\r\nX: secret'])
def test_url_syntax_is_restricted_before_network(url):
    with pytest.raises(SecurityViolation): egress.destination(url)


def test_secret_exfiltration_is_blocked_before_even_dns(monkeypatch):
    secret = 'canary-private-credential-123456789'
    monkeypatch.setenv('TEST_API_KEY', secret)
    monkeypatch.setattr(socket, 'getaddrinfo', lambda *a, **k: pytest.fail('Secret reached DNS'))
    for encoded in (secret, ''.join('%%%02x'%ord(c) for c in secret), base64.b64encode(secret.encode()).decode()):
        with pytest.raises(SecurityViolation): egress.destination('https://example.com/?data='+encoded)


class Response:
    def __init__(self, status=200, location=None, body=b'ok'): self.status, self.location, self.body = status, location, body
    def getheader(self, name): return self.location if name == 'Location' else None
    def read(self, size): return self.body[:size]


class Connection:
    def __init__(self, response): self.response, self.closed, self.sent = response, False, []
    def request(self, *args, **kwargs): self.sent.append((args, kwargs))
    def getresponse(self): return self.response
    def close(self): self.closed = True


def test_public_redirect_to_private_is_rejected_without_connecting(monkeypatch):
    monkeypatch.setattr(socket, 'getaddrinfo', lambda host, *a, **k: resolver('93.184.216.34' if host=='example.com' else '127.0.0.1'))
    first = Connection(Response(302, 'https://internal.example/secrets'))
    opened = []
    def connect(host, port, ip, scheme, timeout): opened.append(ip); return first
    monkeypatch.setattr(egress, '_connection', connect)
    with pytest.raises(SecurityViolation): egress.get('https://example.com')
    assert opened == ['93.184.216.34'] and first.closed


def test_pinned_socket_keeps_original_tls_host_and_never_reresolves_hostname(monkeypatch):
    called = []
    monkeypatch.setattr(socket, 'create_connection', lambda address, **kw: called.append(address))
    connection = egress._connection('example.com', 443, '93.184.216.34', 'https', 2)
    connection._create_connection(('example.com', 443), 2)
    assert called == [('93.184.216.34', 443)] and connection.host == 'example.com'


def test_egress_allowlist_is_exact_and_response_is_bounded(monkeypatch):
    monkeypatch.setenv('AGENT_WEB_ALLOWED_HOSTS', 'example.com')
    monkeypatch.setattr(socket, 'getaddrinfo', lambda *a, **k: resolver('93.184.216.34'))
    with pytest.raises(SecurityViolation): egress.destination('https://example.com.attacker.test')
    conn = Connection(Response(body=b'12345'))
    monkeypatch.setattr(egress, '_connection', lambda *a: conn)
    with pytest.raises(ValueError, match='size'): egress.get('https://example.com', max_bytes=4)
    assert conn.closed
    headers = conn.sent[0][1]['headers']
    assert 'Authorization' not in headers and 'Cookie' not in headers


def test_source_baseline_detects_add_modify_delete_without_importing_code(tmp_path):
    (tmp_path/'src').mkdir()
    source = tmp_path/'src/module.py'
    source.write_text('raise RuntimeError("must never execute")')
    assert integrity.enroll(tmp_path)['status'] == 'verified'
    source.write_text('changed')
    assert integrity.inspect(tmp_path)['status'] == 'changed'
    source.unlink()
    assert integrity.inspect(tmp_path)['changed'] == ['src/module.py']
    with pytest.raises(FileExistsError): integrity.enroll(tmp_path)


def test_model_cannot_write_live_runtime_or_baseline(tmp_path, monkeypatch):
    monkeypatch.setattr(integrity, 'ROOT', tmp_path)
    for name in ('src/security/http.py', 'src/coding/harness.py', '.sea/security/integrity.json', 'config.yml'):
        with pytest.raises(SecurityViolation): integrity.protect_agent_write(tmp_path/name)
    integrity.protect_agent_write(tmp_path/'some_other_project/main.py')


def test_security_events_never_log_payload_or_credentials(tmp_path):
    events.record('http', 'origin_denied')
    events.record('http', 'https://secret.example/token=private')
    path = tmp_path/'.sea/security/events.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 1 and set(rows[0]) == {'surface', 'reason', 'time'}
    assert path.stat().st_mode & 0o777 == 0o600


def test_ast_analyst_reports_primitives_without_secrets_or_execution(tmp_path):
    from src.security.worker import analyze
    marker = tmp_path/'not-created'
    source = f'import urllib.request as r\nSECRET="private-canary"\nopen({str(marker)!r}, "w")\nr.urlopen(SECRET)'
    report = analyze({'sources': {'src/module.py': source}})
    assert report['call_counts']['network'] == 1
    assert 'private-canary' not in json.dumps(report) and not marker.exists()


def test_audit_failure_never_falls_back_to_inprocess_analysis(tmp_path, monkeypatch):
    import src.security.audit as audit
    monkeypatch.setattr(audit, 'isolated_analysis', lambda root: (_ for _ in ()).throw(RuntimeError('no sandbox')))
    monkeypatch.setattr(audit, 'network_snapshot', lambda: pytest.fail('Network not requested'))
    report = audit.audit(tmp_path)
    assert report['analyst']['status'] == 'unavailable'
    assert any(f['code']=='analyst_isolation' and f['severity']=='critical' for f in report['findings'])


def test_secret_search_cannot_reinclude_protected_files(tmp_path, monkeypatch):
    from src.skills.code.code import grep_repo
    monkeypatch.setenv('AGENT_PROJECT_ROOT', str(tmp_path))
    (tmp_path/'config.local.yml').write_text('private-canary-value')
    (tmp_path/'BROWSER_BRIDGE.TOKEN').write_text('private-canary-value')
    for pattern in ('*', 'config.local.yml', 'BROWSER_BRIDGE.TOKEN'):
        result = grep_repo.invoke({'pattern': 'private-canary-value', 'file_glob': pattern})
        assert 'private-canary-value' not in result
    assert 'private-canary-value' not in grep_repo.invoke({'pattern': 'private', 'path': 'config.local.yml'})


def test_attachment_admission_denies_disguised_executable_macro_and_large_file(tmp_path, monkeypatch):
    import zipfile
    from src.security import attachments
    executable = tmp_path/'report.txt'
    executable.write_bytes(b'MZ'+b'x'*100)
    with pytest.raises(SecurityViolation): attachments.validate(executable)
    office = tmp_path/'report.docx'
    with zipfile.ZipFile(office, 'w') as z: z.writestr('word/vbaProject.bin', b'code')
    with pytest.raises(SecurityViolation): attachments.validate(office)
    plain = tmp_path/'allowed.py'
    plain.write_text('print("treated only as text")')
    assert attachments.validate(plain) == plain
    monkeypatch.setattr(attachments, 'MAX_BYTES', 4)
    with pytest.raises(SecurityViolation): attachments.validate(plain)


def test_session_attachment_rejects_symlink_and_cannot_escape_with_session_id(tmp_path, monkeypatch):
    from src.data import knowledge_base as kb
    import src.media
    monkeypatch.setattr(kb, '_SESSION_ROOT', tmp_path/'sessions')
    monkeypatch.setattr(src.media, 'read_file', lambda *a: 'A harmless document with enough text to index.')
    source = tmp_path/'notes.txt'
    source.write_text('private document')
    for sid in ('..', '.', '../../outside', 'ordinary-session'):
        target = Path(kb.add_session_file(sid, source))
        assert target.resolve().is_relative_to(tmp_path/'sessions')
        assert target.stat().st_mode & 0o777 == 0o600
    link = tmp_path/'symlink.txt'
    link.symlink_to(source)
    with pytest.raises(SecurityViolation): kb.add_session_file('other', link)


def test_gzip_expansion_is_bounded(monkeypatch):
    import gzip
    from src.runtime import data_fetch
    monkeypatch.setattr(data_fetch, '_MAX_BYTES', 100)
    with pytest.raises(ValueError, match='size limit'):
        data_fetch._maybe_gunzip('file.gz', gzip.compress(b'a'*1000))


def test_generated_skill_cannot_patch_live_runtime(monkeypatch):
    from src.tools import skill_creation as sc
    from src.security.integrity import ROOT
    monkeypatch.setattr(sc, 'SKILLS_DIR', ROOT/'src/skills')
    monkeypatch.setattr(sc, 'REGISTRY_FILE', ROOT/'src/skills/registry.json')
    monkeypatch.setattr(sc, '_load_reg_at', lambda p: {})
    result = sc.create_skill.invoke({'name':'security_test_not_created', 'description':'test', 'scope':'global'})
    assert 'protected' in result
    assert not (ROOT/'src/skills/security_test_not_created').exists()


def test_mac_command_policy_protects_runtime_even_with_project_write(tmp_path, monkeypatch):
    from src.coding import containment
    monkeypatch.setattr(integrity, 'ROOT', tmp_path)
    monkeypatch.setattr(containment.platform, 'system', lambda: 'Darwin')
    monkeypatch.setattr(containment.shutil, 'which', lambda name: '/usr/bin/'+name)
    command = containment.prefix(tmp_path, tmp_path/'scratch', tmp_path/'out', project_write=True)
    assert '(deny file-write* (subpath '+json.dumps(str(tmp_path/'src'))+'))' in command[-1]


def test_registry_usage_is_volatile_but_trust_flags_are_covered(tmp_path):
    p = tmp_path/'src/skills/registry.json'
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({'web': {'uses': 1, 'protected': True}}))
    integrity.enroll(tmp_path)
    p.write_text(json.dumps({'web': {'uses': 2, 'wins': 1, 'last_used_at':'today', 'protected': True}}))
    assert integrity.inspect(tmp_path)['status'] == 'verified'
    p.write_text(json.dumps({'web': {'uses': 2, 'protected': False}}))
    assert integrity.inspect(tmp_path)['status'] == 'changed'


def test_uploaded_executable_is_rejected_before_parser(tmp_path, monkeypatch):
    from src.interface.server import app
    from src.data import knowledge_base as kb
    import src.media
    monkeypatch.setattr(kb, '_SESSION_ROOT', tmp_path/'sessions')
    monkeypatch.setattr(src.media, 'read_file', lambda *a: pytest.fail('Executable reached parser'))
    response = TestClient(app, base_url='http://localhost', client=('127.0.0.1', 10000)).post(
        '/upload', params={'thread_id':'../../outside'}, files={'file':('report.txt', b'MZ'+b'bad'*10)})
    assert response.status_code == 400


def test_corrupt_baseline_and_symlinked_new_module_fail_closed(tmp_path):
    (tmp_path/'src').mkdir()
    integrity.enroll(tmp_path)
    (tmp_path/'src/injected').symlink_to(tmp_path, target_is_directory=True)
    assert integrity.inspect(tmp_path)['status'] == 'invalid'
    (tmp_path/'src/injected').unlink()
    (tmp_path/'.sea/security/integrity.json').write_text('[]')
    assert integrity.inspect(tmp_path)['status'] == 'invalid'
