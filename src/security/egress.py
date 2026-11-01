"""Public HTTP GET with pinned DNS, per-redirect validation, no ambient credentials.

No browser fallback: Chromium subrequests would bypass this transport policy.
"""
import http.client
import ipaddress
import os
import socket
import ssl
from urllib.parse import urlsplit, urlunsplit, urljoin

from .events import record
from .sensitive import SecurityViolation, guard_text


def public_address(value):
    ip = ipaddress.ip_address(value)
    if not ip.is_global or ip.is_multicast or ip.is_reserved or ip.is_unspecified: return False
    if isinstance(ip, ipaddress.IPv6Address):
        embedded = ip.ipv4_mapped or ip.sixtofour
        if embedded is not None and not public_address(str(embedded)): return False
        if ip in ipaddress.ip_network('64:ff9b::/96') and not public_address(str(ipaddress.IPv4Address(int(ip) & 0xffffffff))):
            return False
    return True


def destination(url):
    guard_text(url) # Before DNS: hostnames can themselves leak credentials.
    try:
        p = urlsplit(url)
        if p.scheme not in {'http', 'https'} or not p.hostname or p.username is not None or p.password is not None:
            raise ValueError()
        if any(ord(c) <= 32 for c in url) or '\\' in url or len(url) > 8192: raise ValueError()
        host = p.hostname.rstrip('.').encode('idna').decode('ascii').lower()
        port = p.port or (443 if p.scheme == 'https' else 80)
        if port not in {80, 443}: raise ValueError()
        allowed = {h.strip().lower().rstrip('.') for h in os.getenv('AGENT_WEB_ALLOWED_HOSTS', '').split(',') if h.strip()}
        if allowed and host not in allowed: raise ValueError()
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        addresses = list(dict.fromkeys(i[4][0] for i in infos))
        if not addresses or any(not public_address(a) for a in addresses): raise ValueError()
    except (ValueError, OSError, UnicodeError):
        record('egress', 'destination_denied')
        raise SecurityViolation('Destination blocked: use an allowed public HTTP(S) host on port 80/443') from None
    authority = f'[{host}]' if ':' in host else host
    if p.port: authority += ':'+str(port)
    normalized = urlunsplit((p.scheme, authority, p.path or '/', p.query, ''))
    return normalized, host, port, addresses[0]


def _connection(host, port, ip, scheme, timeout):
    connection = (http.client.HTTPSConnection(host, port, timeout=timeout, context=ssl.create_default_context())
                  if scheme == 'https' else http.client.HTTPConnection(host, port, timeout=timeout))
    # TLS still verifies host/SNI; the socket connects only to the already checked IP.
    connection._create_connection = lambda address, timeout=None, source_address=None: socket.create_connection(
        (ip, port), timeout=timeout, source_address=source_address)
    return connection


def get(url, *, max_bytes=2*1024**2, timeout=20, redirects=4):
    from .integrity import require_integrity
    require_integrity()
    for hop in range(redirects+1):
        url, host, port, ip = destination(url)
        parsed = urlsplit(url)
        connection = _connection(host, port, ip, parsed.scheme, timeout)
        try:
            connection.request('GET', urlunsplit(('', '', parsed.path, parsed.query, '')),
                headers={'User-Agent': 'SEA-safe-fetch/1.0', 'Accept-Encoding': 'identity'})
            response = connection.getresponse()
            if response.status in {301, 302, 303, 307, 308}:
                location = response.getheader('Location')
                if not location or hop == redirects: raise SecurityViolation('Redirect limit or missing location')
                next_url = urljoin(url, location)
                if parsed.scheme == 'https' and urlsplit(next_url).scheme != 'https':
                    raise SecurityViolation('HTTPS downgrade blocked')
                url = next_url
                continue
            if response.status >= 400: raise ValueError('Remote HTTP status '+str(response.status))
            body = response.read(max_bytes+1)
            if len(body) > max_bytes: raise ValueError('Response exceeds size limit')
            return body
        finally:
            connection.close()
    raise SecurityViolation('Redirect limit exceeded')
