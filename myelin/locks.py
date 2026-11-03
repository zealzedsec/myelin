"""Advisory cross-process file locks, on POSIX (`flock`) and Windows (`msvcrt`).

Three things need them: one scheduler per vault (`daemon.lock`), one sleep cycle
at a time across the CLI, the daemon and the web server (`sleep.lock`), and the
ledger head checkpoint, which every writing process updates after it commits.
"""
from __future__ import annotations

import contextlib
import time
from pathlib import Path
from typing import IO, Iterator

try:
    import fcntl
except ImportError:  # Windows
    fcntl = None  # type: ignore[assignment]
    import msvcrt


def try_lock(path: Path) -> IO[str] | None:
    """Take the lock without waiting. Returns the open handle that holds it, or
    None if another process has it. Closing the handle releases the lock."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    try:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        else:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        handle.close()
        return None
    return handle


def release(handle: IO[str] | None) -> None:
    if handle is None:
        return
    with contextlib.suppress(OSError):
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        else:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    handle.close()


@contextlib.contextmanager
def held(path: Path, timeout: float = 10.0) -> Iterator[bool]:
    """Wait up to `timeout` seconds for the lock. Yields whether it was taken,
    so a caller whose work is optional can carry on without it."""
    deadline = time.monotonic() + timeout
    handle = try_lock(path)
    while handle is None and time.monotonic() < deadline:
        time.sleep(0.02)
        handle = try_lock(path)
    try:
        yield handle is not None
    finally:
        release(handle)
