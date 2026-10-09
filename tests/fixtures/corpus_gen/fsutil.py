"""Filesystem helpers for the corpus generator.

Workaround for a verified environment defect
--------------------------------------------
Measured on this machine (Windows, CPython 3.13.12)::

    os.mkdir(p)          -> write into p : OK
    os.mkdir(p, 0o755)   -> write into p : OK
    os.mkdir(p, 0o777)   -> write into p : OK
    os.mkdir(p, 0o700)   -> write into p : PermissionError [Errno 13]
    tempfile.mkdtemp()   -> write into it: PermissionError [Errno 13]
    os.chmod(mkdtemp_dir, 0o777)         -> PermissionError [WinError 5]

``tempfile.mkdtemp`` creates its directory with mode ``0o700``, so *every*
``tempfile`` directory is unusable here — this is also what makes ``pip``
fail under the confined file sandbox.

Consequences for this package:

* never call ``tempfile.mkdtemp`` / ``tempfile.TemporaryDirectory``;
* never create a directory with an explicit ``0o700`` mode;
* create every directory through :func:`ensure_dir` and every scratch
  directory through :func:`scratch_dir`.

Both helpers use the process-default mode (``0o777``), which behaves correctly.
"""

from __future__ import annotations

import hashlib
import os
import uuid
from pathlib import Path
from typing import Union

PathLike = Union[str, Path]

#: Mode used for every directory this package creates.  ``0o700`` is broken on
#: this platform (see module docstring), so we deliberately do not use it.
DIR_MODE = 0o777


def ensure_dir(path: PathLike) -> Path:
    """Create *path* and all parents if needed, then return it as a ``Path``.

    Uses the process-default mode.  Idempotent.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True, mode=DIR_MODE)
    return p


def scratch_dir(root: PathLike, prefix: str = "scratch-") -> Path:
    """Create a fresh uniquely-named directory under *root*.

    Drop-in replacement for ``tempfile.mkdtemp(dir=root)`` that does not
    create an unwritable ``0o700`` directory.
    """
    base = ensure_dir(root)
    while True:
        candidate = base / f"{prefix}{uuid.uuid4().hex[:12]}"
        try:
            candidate.mkdir(mode=DIR_MODE)
        except FileExistsError:
            continue
        return candidate


def atomic_write_bytes(path: PathLike, data: bytes) -> None:
    """Write *data* to *path* atomically (sibling ``.tmp`` then ``os.replace``)."""
    target = Path(path)
    ensure_dir(target.parent)
    tmp = target.with_name(target.name + ".tmp")
    with open(tmp, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, target)


def atomic_write_text(path: PathLike, text: str, encoding: str = "utf-8") -> None:
    """Write *text* to *path* atomically."""
    atomic_write_bytes(path, text.encode(encoding))


def sha256_file(path: PathLike, chunk_size: int = 1 << 16) -> str:
    """Return the hex SHA256 of a file's contents."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    """Return the hex SHA256 of a string's UTF-8 encoding."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def dir_size(path: PathLike) -> int:
    """Total size in bytes of every file under *path* (0 if it does not exist)."""
    total = 0
    root = Path(path)
    if not root.exists():
        return 0
    for entry in root.rglob("*"):
        try:
            if entry.is_file():
                total += entry.stat().st_size
        except OSError:
            continue
    return total
