"""One writer per workspace, including across CLI and desktop processes."""
from contextlib import contextmanager
from pathlib import Path
import os


@contextmanager
def workspace_lease(root: Path):
    directory = root / ".sea"
    if not directory.resolve().is_relative_to(root):
        raise ValueError("checkpoint directory escapes workspace")
    directory.mkdir(mode=0o700, exist_ok=True)
    path = directory / "coding.lock"
    if path.is_symlink():
        raise ValueError("symlink lock path forbidden")
    fd = os.open(path, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(fd, "r+b") as lock:
        if os.name == "nt":
            import msvcrt
            lock.write(b"0")
            lock.flush()
            lock.seek(0)
            try:
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError("another coding run owns this workspace") from exc
        else:
            import fcntl
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError("another coding run owns this workspace") from exc
        try:
            yield
        finally:
            if os.name == "nt":
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock, fcntl.LOCK_UN)
