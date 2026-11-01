"""Bounded attachment admission, not malware detection or a parser sandbox."""
import os
from pathlib import Path
import stat
import zipfile

from .events import record
from .sensitive import SecurityViolation

MAX_BYTES = 32 * 1024**2
_ALLOWED = {'.txt', '.md', '.csv', '.json', '.yaml', '.yml', '.toml', '.log', '.py', '.cpp', '.c',
    '.h', '.hpp', '.rs', '.go', '.js', '.ts', '.tsx', '.java', '.sql', '.sh', '.xml',
    '.pdf', '.docx', '.pptx', '.xlsx', '.png', '.jpg', '.jpeg', '.webp', '.bmp', '.heic',
    '.mp3', '.wav', '.m4a', '.ogg', '.flac', '.webm', '.mp4', '.mov', '.gif'}
_EXEC_MAGIC = (b'MZ', b'\x7fELF', b'\xcf\xfa\xed\xfe', b'\xce\xfa\xed\xfe',
               b'\xfe\xed\xfa\xcf', b'\xfe\xed\xfa\xce', b'\xca\xfe\xba\xbe')


def validate(path):
    from src.skills.code.code import _is_secret_file
    p = Path(path)
    try:
        if p.is_symlink() or not stat.S_ISREG(p.stat().st_mode): raise ValueError()
        if p.stat().st_size > MAX_BYTES or p.suffix.lower() not in _ALLOWED: raise ValueError()
        if any(_is_secret_file(Path(part)) for part in p.parts): raise ValueError()
        with p.open('rb') as f:
            if f.read(4).startswith(_EXEC_MAGIC): raise ValueError()
        if p.suffix.lower() in {'.docx', '.pptx', '.xlsx'}:
            with zipfile.ZipFile(p) as z:
                entries = z.infolist()
                if len(entries) > 2000 or sum(e.file_size for e in entries) > 64*1024**2: raise ValueError()
                for e in entries:
                    name = e.filename.lower()
                    if e.flag_bits & 1 or '..' in Path(name).parts or name.startswith('/'):
                        raise ValueError()
                    if 'vbaproject' in name or '/embeddings/' in name or name.endswith(('.exe', '.dll')):
                        raise ValueError()
    except (OSError, ValueError, zipfile.BadZipFile):
        record('attachment', 'admission_denied')
        raise SecurityViolation('Attachment rejected: unsupported, sensitive, executable, active Office content or oversized file') from None
    return p


def copy_private(source, target):
    source, target = validate(source), Path(target)
    if source.resolve() == target.resolve():
        if target.is_symlink(): raise SecurityViolation('Unsafe attachment destination')
        target.chmod(0o600)
        return
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as out, source.open('rb') as inp:
        os.fchmod(out.fileno(), 0o600)
        # Recheck after bounded copy: a source may change between admission and open.
        data = inp.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES: raise SecurityViolation('Attachment size limit exceeded')
        out.write(data)
    validate(target)
