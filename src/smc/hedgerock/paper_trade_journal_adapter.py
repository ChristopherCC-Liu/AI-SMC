"""Stage F enabler — load real paper-trade journals into ``alpha_validation``.

Phase 4.1 ``alpha_validation`` was designed Protocol-first so we can plug
real trade records in place of the synthetic generator without touching
the validation harness. This module ships the two production loaders we
need for Stage F paper trading:

- :func:`load_csv` — broker-style CSV export (one row per closed trade).
  Compatible with MT5's "Trade History → Export" CSV (semicolon-separated
  in some locales, comma in others — we sniff the delimiter).
- :func:`load_json` — HedgeRock's own journal format (one JSON object
  per line, written by the EA via WebRequest or local file logging).

Both loaders return ``tuple[TradeRecord, ...]`` ready for
:func:`smc.hedgerock.alpha_validation.run_alpha_validation`.

Regime context handling — production sources (broker CSV especially)
typically lack ``regime_at_entry`` / ``regime_mismatch``. The loader
accepts an optional ``regime_lookup`` callable so the caller can
backfill regime data from ``decision_replay`` history. When omitted
both fields default to ``"TREND_UP"`` / ``False`` — the validation
harness still runs, but AC-5 mismatch grouping degrades to a single
bucket and downstream AC-6 sentinel becomes degenerate (full=mismatch).
The ``[GO]`` spec calls this out explicitly: "real journals → real
regime backfill in Stage F prep".
"""

from __future__ import annotations

import csv
import json
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from smc.ai.models import MarketRegimeAI
from smc.hedgerock.alpha_validation import TradeRecord


__all__ = [
    "RegimeLookup",
    "load_csv",
    "load_json",
    "load_paper_trade_journal",
]


RegimeLookup = Callable[[datetime], "tuple[MarketRegimeAI, bool]"]
"""``ts → (regime, is_regime_mismatch)`` resolver.

Production: closure over the ``decision_replay`` regime store from the
same paper trading window — gives every closed trade its regime
context at entry.
Tests: stub returning fixed values.
"""


def _default_regime_lookup(_ts: datetime) -> tuple[MarketRegimeAI, bool]:
    """Stable fallback when the caller does not provide a lookup.

    Returns ``("TREND_UP", False)`` so AC-5 sees a single uniform group.
    Stage F prep should always inject a real regime_lookup; this default
    is for smoke tests + degraded-mode runs only.
    """
    return "TREND_UP", False


# ---------------------------------------------------------------------------
# CSV loader — broker-style export
# ---------------------------------------------------------------------------


def load_csv(
    path: str | Path,
    *,
    regime_lookup: RegimeLookup | None = None,
    ts_column: str = "close_ts",
    pnl_column: str = "profit_usd",
) -> tuple[TradeRecord, ...]:
    """Parse a broker CSV trade history into :class:`TradeRecord` records.

    Expected columns (case-insensitive, defaults match MT5 export):
    - ``close_ts`` — ISO timestamp when the position closed.
    - ``profit_usd`` — net P&L in USD.

    Other columns are ignored. Rows missing either required column or
    failing to parse are skipped (silent — the caller can audit by
    comparing input row count to returned tuple length).

    Args:
        path: CSV file path.
        regime_lookup: optional callable filling regime context per trade.
        ts_column / pnl_column: override the default header names.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
    """
    lookup = regime_lookup or _default_regime_lookup
    text = Path(path).read_text()
    # Sniff dialect — handles MT5 locales using ';' or ','.
    try:
        dialect = csv.Sniffer().sniff(text[:1024], delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    reader = csv.DictReader(text.splitlines(), dialect=dialect)

    records: list[TradeRecord] = []
    for row in reader:
        normalised = {(k or "").strip().lower(): (v or "").strip() for k, v in row.items()}
        ts_raw = normalised.get(ts_column.lower())
        pnl_raw = normalised.get(pnl_column.lower())
        if not ts_raw or not pnl_raw:
            continue
        try:
            ts = _parse_iso_timestamp(ts_raw)
            pnl = float(pnl_raw)
        except (ValueError, TypeError):
            continue
        regime, mismatch = lookup(ts)
        records.append(
            TradeRecord(
                ts=ts,
                pnl_usd=pnl,
                regime_at_entry=regime,
                regime_mismatch=mismatch,
            )
        )
    return tuple(records)


# ---------------------------------------------------------------------------
# JSON loader — HedgeRock journal format
# ---------------------------------------------------------------------------


def load_json(
    path: str | Path,
    *,
    regime_lookup: RegimeLookup | None = None,
) -> tuple[TradeRecord, ...]:
    """Parse a HedgeRock JSONL journal (one trade object per line).

    Required keys per object:
    - ``close_ts``: ISO timestamp string.
    - ``pnl_usd``: signed float.

    Optional keys: ``regime_at_entry`` (overrides regime_lookup if set),
    ``regime_mismatch`` (likewise). This lets richer journals embed
    regime context directly.

    Lines that fail to JSON-parse or miss required keys are skipped.
    """
    lookup = regime_lookup or _default_regime_lookup
    records: list[TradeRecord] = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        ts_raw = obj.get("close_ts")
        pnl_raw = obj.get("pnl_usd")
        if not isinstance(ts_raw, str) or not isinstance(pnl_raw, (int, float)):
            continue
        try:
            ts = _parse_iso_timestamp(ts_raw)
        except ValueError:
            continue
        regime_in_payload = obj.get("regime_at_entry")
        mismatch_in_payload = obj.get("regime_mismatch")
        if isinstance(regime_in_payload, str) and isinstance(mismatch_in_payload, bool):
            regime: MarketRegimeAI = regime_in_payload  # type: ignore[assignment]
            mismatch = mismatch_in_payload
        else:
            regime, mismatch = lookup(ts)
        records.append(
            TradeRecord(
                ts=ts,
                pnl_usd=float(pnl_raw),
                regime_at_entry=regime,
                regime_mismatch=mismatch,
            )
        )
    return tuple(records)


# ---------------------------------------------------------------------------
# Convenience dispatcher
# ---------------------------------------------------------------------------


def load_paper_trade_journal(
    path: str | Path,
    *,
    regime_lookup: RegimeLookup | None = None,
) -> tuple[TradeRecord, ...]:
    """Auto-dispatch by extension.

    ``.csv`` → :func:`load_csv`, ``.json`` / ``.jsonl`` → :func:`load_json`.
    """
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return load_csv(path, regime_lookup=regime_lookup)
    if suffix in (".json", ".jsonl"):
        return load_json(path, regime_lookup=regime_lookup)
    raise ValueError(
        f"Unsupported journal format {suffix!r} (path={path!s}); "
        "use .csv or .json/.jsonl, or call load_csv/load_json directly."
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_iso_timestamp(value: str) -> datetime:
    """Permissive ISO-8601 parser — accepts naive timestamps as UTC."""
    cleaned = value.strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(cleaned)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt
