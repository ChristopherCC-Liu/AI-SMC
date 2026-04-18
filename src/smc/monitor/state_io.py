"""Atomic JSON state persistence utility."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def atomic_write_json(path: Path, data: Any) -> None:
    """Write JSON atomically via tmp file + os.replace().

    Uses a tmp-file + rename approach so the destination file is never left
    in a partial state.  POSIX rename() and Windows MoveFileEx() are both
    atomic at the filesystem level.

    - Creates parent directory tree if it does not exist.
    - Best-effort cleanup of the tmp file on failure; the original exception
      is always re-raised so callers know the write failed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(json.dumps(data, default=str, indent=2), encoding="utf-8")
        os.replace(tmp, path)
    except Exception:
        if tmp.exists():
            try:
                tmp.unlink()
            except Exception:
                pass
        raise


def load_json(path: Path, default: Any = None) -> Any:
    """Read JSON from *path*, returning *default* on any error (fail-open).

    Missing files, corrupt JSON, and permission errors all return *default*
    rather than raising so callers get graceful degradation.
    """
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return default if default is not None else {}
