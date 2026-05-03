"""Atomic JSON cache writer for the HedgeRock EA.

The MQL5 EA reads `RegimeCache.json` from the MT5 sandbox directory on
each OnTimer tick. If the file were updated non-atomically, the EA could
read a half-written file and either get a JSON parse error (best case)
or — worse — silently drop fields and apply stale defaults.

This module solves that with the standard pattern:

    1. Write to a sibling temp file in the same directory.
    2. Flush + fsync so the data hits disk.
    3. os.replace(temp_path, final_path) — atomic on POSIX, atomic on
       Windows since Python 3.3 for files that fit in one MFT entry.

The default cache path is the MT5 Windows sandbox layout. It can be
overridden via the env var `HEDGEROCK_CACHE_PATH` or by passing `path`
explicitly. Tests always pass `path` so they never touch the real
sandbox.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from smc.hedgerock.schemas import SignalEnvelope

__all__ = [
    "DEFAULT_CACHE_FILENAME",
    "ENV_CACHE_PATH",
    "resolve_cache_path",
    "write_envelope",
]

DEFAULT_CACHE_FILENAME: str = "RegimeCache.json"
ENV_CACHE_PATH: str = "HEDGEROCK_CACHE_PATH"


def resolve_cache_path(explicit: Path | str | None = None) -> Path:
    """Decide where to write the cache file.

    Order of resolution:
      1. `explicit` argument (always wins; tests use this).
      2. `HEDGEROCK_CACHE_PATH` env var.
      3. RuntimeError — the caller must opt in.

    The MT5 sandbox path (e.g. `%APPDATA%/MetaQuotes/Terminal/<HASH>/MQL5/Files/`)
    is host-specific; we never guess it here. The deployment script writes
    the env var when it discovers the path on the target VPS.
    """
    if explicit is not None:
        return Path(explicit)
    env = os.environ.get(ENV_CACHE_PATH)
    if env:
        return Path(env)
    raise RuntimeError(
        f"Cache path not configured. Set {ENV_CACHE_PATH} env var or pass `path` explicitly."
    )


def write_envelope(envelope: SignalEnvelope, path: Path | str | None = None) -> Path:
    """Serialise `envelope` and atomically write it to `path`.

    Returns the resolved final path on success.

    Raises:
        RuntimeError: when no path is configured and no explicit path given.
        OSError: on filesystem failures (parent missing + cannot create, etc.).
    """
    final_path = resolve_cache_path(path)
    final_path.parent.mkdir(parents=True, exist_ok=True)

    payload = envelope.model_dump(mode="json")
    blob = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)

    # Write to sibling temp, fsync, then atomic replace.
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{final_path.name}.",
        suffix=".tmp",
        dir=str(final_path.parent),
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(blob)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, final_path)
    except Exception:
        # Best-effort cleanup; do not mask the real error.
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise
    return final_path
