"""Single-owner API boundary: Host/Origin/peer checks and optional token auth."""
import base64
import hmac
import ipaddress
import os
from urllib.parse import urlsplit

from starlette.responses import JSONResponse
from starlette.exceptions import HTTPException

from .events import record

MAX_REQUEST_BYTES = 40 * 1024**2


def loopback(value):
    try: return ipaddress.ip_address(value).is_loopback
    except ValueError: return False


def validate_bind(host):
    if host != 'localhost' and not loopback(host) and len(os.getenv('AGENT_API_TOKEN', '')) < 32:
        raise RuntimeError('Non-loopback bind requires AGENT_API_TOKEN with at least 32 characters')


def authority(value):
    p = urlsplit('//'+value)
    if not p.hostname or p.username or p.password or p.path or p.query or p.fragment or any(c.isspace() for c in value):
        raise ValueError('Invalid authority')
    return p.hostname.lower().rstrip('.'), p.port


class AgentHTTPGuard:
    def __init__(self, app): self.app = app

    async def __call__(self, scope, receive, send):
        if scope['type'] not in {'http', 'websocket'}: return await self.app(scope, receive, send)
        async def deny(reason, status=403):
            record('http', reason)
            if scope['type'] == 'websocket': return await send({'type': 'websocket.close', 'code': 1008})
            response = JSONResponse({'detail': 'Security policy rejected this request', 'reason': reason}, status_code=status,
                headers={'WWW-Authenticate': 'Basic realm="SEA owner", charset="UTF-8"'} if status == 401 else {})
            await response(scope, receive, send)
        try:
            headers = {}
            for k, v in scope.get('headers', []):
                key = k.decode('latin1').lower()
                if key in headers and key in {'host', 'origin', 'authorization', 'content-length'}: return await deny('duplicate_header')
                headers[key] = v.decode('latin1')
            length = int(headers.get('content-length', '0'))
            if length < 0 or length > MAX_REQUEST_BYTES: return await deny('body_size', 413)
            host, port = authority(headers.get('host', ''))
            allowed = {'localhost', '127.0.0.1', '::1'} | {h.strip().lower() for h in os.getenv('AGENT_ALLOWED_HOSTS', '').split(',') if h.strip()}
            if host not in allowed: return await deny('host_denied')
            if headers.get('sec-fetch-site') == 'cross-site': return await deny('cross_site')
            if 'origin' in headers:
                origin = urlsplit(headers['origin'])
                expected_scheme = 'https' if scope.get('scheme') in {'https', 'wss'} else 'http'
                if origin.scheme != expected_scheme or origin.path or origin.query or origin.fragment:
                    return await deny('origin_denied')
                oh, op = authority(origin.netloc)
                default = 443 if expected_scheme == 'https' else 80
                if (oh, op or default) != (host, port or default): return await deny('origin_denied')
            peer = (scope.get('client') or ('', 0))[0]
            token = os.getenv('AGENT_API_TOKEN', '')
            if not loopback(peer):
                if len(token) < 32: return await deny('remote_without_auth')
                if scope.get('scheme') not in {'https', 'wss'}: return await deny('remote_requires_tls')
            if token:
                if len(token) < 32: return await deny('weak_token')
                supplied = headers.get('authorization', '')
                if supplied.startswith('Bearer '): supplied = supplied[7:]
                elif supplied.startswith('Basic '):
                    decoded = base64.b64decode(supplied[6:], validate=True).decode('utf8')
                    user, _, supplied = decoded.partition(':')
                    if user != 'sea': return await deny('invalid_auth', 401)
                else: return await deny('auth_required', 401)
                if not hmac.compare_digest(supplied.encode(), token.encode()): return await deny('invalid_auth', 401)
        except (ValueError, UnicodeError):
            return await deny('invalid_request')
        from .integrity import require_integrity
        from .sensitive import SecurityViolation
        try: require_integrity()
        except SecurityViolation: return await deny('integrity_failed', 503)
        async def secured_send(message):
            if message['type'] == 'http.response.start':
                extra = [(b'x-content-type-options', b'nosniff'), (b'referrer-policy', b'no-referrer'),
                    (b'cache-control', b'no-store'), (b'x-frame-options', b'DENY'),
                    (b'content-security-policy', b"default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; connect-src 'self'; frame-ancestors 'none'; object-src 'none'; base-uri 'none'; form-action 'self'")]
                message = {**message, 'headers': list(message.get('headers', []))+extra}
            await send(message)
        size = 0
        async def bounded_receive():
            nonlocal size
            message = await receive()
            if message['type'] == 'http.request':
                size += len(message.get('body', b''))
                if size > MAX_REQUEST_BYTES:
                    record('http', 'body_size')
                    raise HTTPException(413, 'Request size limit exceeded')
            return message
        await self.app(scope, bounded_receive, secured_send)
