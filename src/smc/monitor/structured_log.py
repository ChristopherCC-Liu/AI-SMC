"""Structured JSON logging with severity tags for grep-ability."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from typing import Any, Literal, TextIO

Severity = Literal["CRIT", "WARN", "INFO", "DEBUG"]


def log_event(
    severity: Severity,
    event: str,
    stream: TextIO | None = None,
    **fields: Any,
) -> None:
    """Emit a structured log line to stderr (or given stream).

    Format: ``[<SEV>] {"ts": "...", "event": "...", ...}``

    - Tag on its own is grep-friendly (``tail -f | grep CRIT``)
    - JSON body makes parsing downstream easy
    - ``default=str`` handles datetime, Path, Decimal etc.
    """
    out = stream if stream is not None else sys.stderr
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **fields,
    }
    line = f"[{severity}] {json.dumps(payload, default=str)}\n"
    try:
        out.write(line)
        out.flush()
    except Exception:
        pass  # never raise from logging


def crit(event: str, **fields: Any) -> None:
    """Emit a CRIT-severity structured log line."""
    log_event("CRIT", event, **fields)


def warn(event: str, **fields: Any) -> None:
    """Emit a WARN-severity structured log line."""
    log_event("WARN", event, **fields)


def info(event: str, **fields: Any) -> None:
    """Emit an INFO-severity structured log line."""
    log_event("INFO", event, **fields)


__all__ = ["Severity", "log_event", "crit", "warn", "info"]
