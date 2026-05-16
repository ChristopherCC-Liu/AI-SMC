"""Stage 6-followup task 3 — paper-test ledger schema (append-only).

A `PaperTestEntry` records one paper-traded position whose
parameters were drawn from a recommended candidate. The ledger
sits alongside the shadow-test queue: queued candidates that pass
the shadow runner can run a paper test, and the per-trade results
are appended here.

Invariants:
  * Frozen dataclass; ``report_only=True`` is non-negotiable.
  * Append-only JSONL ledger: no public dequeue / pop / clear API.
  * Refuses paths under ``policy_registry/approved/`` or
    ``policy_registry/pointer.json``.
  * XAUUSD only — paper trades for any other symbol raise
    ``ValueError`` at construction time.
  * No imports from the live trading runtime.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


__all__ = [
    "PaperTestEntry",
    "PaperTestLedger",
    "build_paper_test_entry",
]


_FORBIDDEN_PATH_FRAGMENTS = (
    "policy_registry/approved",
    "policy_registry/pointer.json",
)

_ALLOWED_SYMBOLS = ("XAUUSD",)
_ALLOWED_SIDES = ("long", "short")


def _assert_path_safe(path: Path) -> None:
    text = str(path)
    for fragment in _FORBIDDEN_PATH_FRAGMENTS:
        if fragment in text:
            raise ValueError(
                f"paper-test ledger path lands under a forbidden "
                f"location: {text!r} (matched {fragment!r})"
            )


@dataclass(frozen=True)
class PaperTestEntry:
    """One paper-traded position. PnL and drawdown are stored as
    floats in the unit the operator agreed on (e.g. account
    currency); the ledger does not enforce a unit — only that the
    numbers are floats and that the symbol is XAUUSD."""

    candidate_id: str
    symbol: str
    entry_at: str  # ISO 8601
    exit_at: str   # ISO 8601
    entry_price: float
    exit_price: float
    side: str  # "long" | "short"
    size_lots: float
    pnl: float
    drawdown: float  # negative number; max adverse excursion during the trade
    duration_seconds: int
    gates_at_entry: tuple[str, ...]
    audit_log_path: str
    report_only: bool = True
    recorded_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


def build_paper_test_entry(
    *,
    candidate_id: str,
    symbol: str,
    entry_at: datetime,
    exit_at: datetime,
    entry_price: float,
    exit_price: float,
    side: str,
    size_lots: float,
    pnl: float,
    drawdown: float,
    gates_at_entry: tuple[str, ...],
    audit_log_path: str,
) -> PaperTestEntry:
    if symbol not in _ALLOWED_SYMBOLS:
        raise ValueError(
            f"paper-test ledger is XAUUSD-only; got symbol={symbol!r}"
        )
    if side not in _ALLOWED_SIDES:
        raise ValueError(
            f"paper-test entry side must be in {_ALLOWED_SIDES}; "
            f"got {side!r}"
        )
    if exit_at < entry_at:
        raise ValueError(
            f"paper-test exit_at ({exit_at!r}) precedes entry_at "
            f"({entry_at!r})"
        )
    duration = int((exit_at - entry_at).total_seconds())
    return PaperTestEntry(
        candidate_id=candidate_id,
        symbol=symbol,
        entry_at=entry_at.astimezone(timezone.utc).isoformat(),
        exit_at=exit_at.astimezone(timezone.utc).isoformat(),
        entry_price=float(entry_price),
        exit_price=float(exit_price),
        side=side,
        size_lots=float(size_lots),
        pnl=float(pnl),
        drawdown=float(drawdown),
        duration_seconds=duration,
        gates_at_entry=tuple(gates_at_entry),
        audit_log_path=str(audit_log_path),
    )


class PaperTestLedger:
    """Append-only JSONL ledger.

    Public surface: :meth:`append`, :meth:`summarise`, :attr:`path`.
    No removal / pop / mutation methods exist.
    """

    __slots__ = ("path", "_audit_log_path")

    def __init__(self, *, path: Path, audit_log_path: Path) -> None:
        _assert_path_safe(Path(path))
        self.path: Path = Path(path)
        self._audit_log_path: Path = Path(audit_log_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, entry: PaperTestEntry) -> PaperTestEntry:
        """Append a paper-test entry. Raises ``ValueError`` if the
        entry is malformed (e.g. wrong symbol)."""
        if entry.symbol not in _ALLOWED_SYMBOLS:
            raise ValueError(
                f"paper-test ledger only accepts {_ALLOWED_SYMBOLS}; "
                f"got {entry.symbol!r}"
            )
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(asdict(entry), ensure_ascii=False, sort_keys=True)
                + "\n"
            )
        return entry

    def append_many(self, entries: Iterable[PaperTestEntry]) -> int:
        n = 0
        for e in entries:
            self.append(e)
            n += 1
        return n

    def summarise(self) -> dict[str, dict[str, Any]]:
        """Read-only aggregate over the ledger.

        Returns a mapping ``candidate_id -> {trades, pnl_sum,
        max_drawdown}``. Reading the file does not mutate it.
        """
        if not self.path.exists():
            return {}
        agg: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"trades": 0, "pnl_sum": 0.0, "max_drawdown": 0.0}
        )
        for ln in self.path.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                d = json.loads(ln)
            except json.JSONDecodeError:
                continue
            cid = d.get("candidate_id", "?")
            slot = agg[cid]
            slot["trades"] += 1
            slot["pnl_sum"] += float(d.get("pnl", 0.0))
            dd = float(d.get("drawdown", 0.0))
            if dd < slot["max_drawdown"]:
                slot["max_drawdown"] = dd
        # default 0.0 max_drawdown is misleading when no negative dd
        # was recorded; surface the actual minimum for downstream
        # reports.
        return {k: dict(v) for k, v in agg.items()}
