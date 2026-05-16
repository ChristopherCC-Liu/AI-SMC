"""Ticket 2 Step 2 — Data-slice identity helpers.

**Pure functions; no imports from production runtime.** Reads the
forex data lake (already a stable read-only API) and computes:

  - :class:`DataSliceIdentity` — hash-pinned identity of the data
    slice the shadow runner used. Comparison key for G8 to detect
    "artefact data slice ≠ current report's data slice".
  - :class:`DecisionWindow` — per-bar trailing H1/H4/D1 frames that
    obey the strict-prior closed-bar rule, plus self-reported
    invariants (no lookahead, no partial bar, decision_ts < trade_ts).

The strict-prior rule mirrors what
``smc.hedgerock.phase_d_walk_forward._load_data`` enforces in
production:

  - H1 frame for decision at ``ts_i`` is rows ``[i - lookback, i)``
    — bar ``i`` itself is the trade-fill bar, not a decision input.
  - H4 frame includes only H4 bars whose 4-hour period CLOSED at or
    before ``ts_i`` (``h4_open + 4h ≤ ts_i``). No partial bar.
  - D1 frame's day floor is strictly less than day floor of ``ts_i``.

This module **does not** call any production decision function; it
only assembles the trailing data context. Step 5's shadow runner
will hand these frames to the rule engine mirror, never to the live
``rule_engine``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl


__all__ = [
    "CLOSED_BAR_RULE_VERSION",
    "DataSliceIdentity",
    "DecisionWindow",
    "DecisionWindowInvariants",
    "build_decision_window",
    "compute_data_slice_identity",
]


# Pinned closed-bar rule version. Any semantic change to the
# strict-prior windowing logic in this module MUST bump this string
# AND a major bump of ShadowArtefact's schema_version.
CLOSED_BAR_RULE_VERSION: str = "phase_d_strict_prior_v1"


# ---------------------------------------------------------------------------
# Identity dataclass (pinned in artefact via DataSliceIdentity above)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DataSliceIdentity:
    """Hash-pinned identity of a (lake, symbol, time-range) slice.

    Mirrors the shadow_artefact.DataSliceIdentity wire schema. Stored
    here to keep the helper module self-contained; the runner copies
    its fields when constructing the artefact."""

    symbols: tuple[str, ...]
    time_range_start: str  # ISO date
    time_range_end: str    # ISO date
    timeframes: tuple[str, ...]
    closed_bar_rule_version: str
    lake_snapshot_hash: str
    lake_snapshot_row_counts: dict[str, int]


@dataclass(frozen=True)
class DecisionWindowInvariants:
    same_bar_set_used: bool
    decision_only_uses_strictly_prior_data: bool
    h4_partial_bar_in_window: bool
    d1_partial_bar_in_window: bool
    decision_uses_data_with_ts_lt_trade_bar_ts: bool


@dataclass(frozen=True)
class DecisionWindow:
    """Per-bar trailing context for one decision_ts.

    Frames may be ``None`` when the lookback isn't satisfied at the
    start of the time range. Callers that need fully-formed frames
    must check before using them; the runner downstream of Step 5
    treats a ``None`` frame as "skip this bar"."""

    decision_ts: datetime
    h1_frame: pl.DataFrame
    h4_frame: pl.DataFrame | None
    d1_frame: pl.DataFrame | None
    invariants: DecisionWindowInvariants


# ---------------------------------------------------------------------------
# compute_data_slice_identity
# ---------------------------------------------------------------------------


_TF_NAMES: tuple[str, ...] = ("H1", "H4", "D1")


def _tf_enum(timeframe_name: str):
    """Resolve the lake's Timeframe enum lazily so tests using a stub
    lake don't have to import the production enum."""
    from smc.data.schemas import Timeframe
    return getattr(Timeframe, timeframe_name)


def _hash_dataframe(df: pl.DataFrame) -> str:
    """SHA-256 of a DataFrame's serialised content. Sort by ts for a
    deterministic byte stream; include every column."""
    if df.is_empty():
        return hashlib.sha256(b"empty").hexdigest()
    sorted_df = df.sort("ts")
    payload = sorted_df.write_csv().encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def compute_data_slice_identity(
    lake: Any, symbol: str, start: datetime, end: datetime,
) -> DataSliceIdentity:
    """Build the hash-pinned identity of (lake, symbol, [start, end)).

    The hash covers each timeframe's content (CSV-serialised, sorted
    by ts) concatenated under timeframe-name prefixes. Row counts
    are emitted separately so a "hash matches but row count differs"
    truncation cannot hide.
    """
    h = hashlib.sha256()
    # Bind the slice to its (symbol, time-range) tuple so two slices
    # over the same underlying bars but different requested ranges
    # produce distinct hashes (defends against
    # "lake holds 30 days; both queries return same rows" cases).
    h.update(symbol.encode("utf-8"))
    h.update(b"\0")
    h.update(start.isoformat().encode("utf-8"))
    h.update(b"\0")
    h.update(end.isoformat().encode("utf-8"))
    h.update(b"\0")
    h.update(CLOSED_BAR_RULE_VERSION.encode("utf-8"))
    h.update(b"\0")

    row_counts: dict[str, int] = {}
    for tf_name in _TF_NAMES:
        tf = _tf_enum(tf_name)
        df = lake.query(symbol, tf, start, end)
        row_counts[tf_name] = df.height
        h.update(tf_name.encode("utf-8"))
        h.update(b"\0")
        h.update(_hash_dataframe(df).encode("utf-8"))
        h.update(b"\0")

    return DataSliceIdentity(
        symbols=(symbol,),
        time_range_start=start.date().isoformat(),
        time_range_end=end.date().isoformat(),
        timeframes=_TF_NAMES,
        closed_bar_rule_version=CLOSED_BAR_RULE_VERSION,
        lake_snapshot_hash=h.hexdigest(),
        lake_snapshot_row_counts=row_counts,
    )


# ---------------------------------------------------------------------------
# build_decision_window — strict-prior closed-bar rule
# ---------------------------------------------------------------------------


_H4_PERIOD = timedelta(hours=4)


def _floor_to_day(ts: datetime) -> datetime:
    return ts.replace(hour=0, minute=0, second=0, microsecond=0)


def build_decision_window(
    *,
    lake: Any,
    symbol: str,
    start: datetime,
    end: datetime,
    decision_ts: datetime,
    h1_lookback: int,
    h4_lookback: int,
) -> DecisionWindow:
    """Assemble per-bar trailing H1/H4/D1 frames at ``decision_ts``,
    obeying the strict-prior closed-bar rule.

    The frame contents are filtered (and re-checked) inside this
    function. Any partial-bar inclusion sets the corresponding
    invariant flag to True (a violation), so callers can honestly
    report it via the artefact.
    """
    h1_full = lake.query(
        symbol, _tf_enum("H1"),
        # Use a generous warmup region so the trailing window
        # populates even for early decision_ts inside [start, end).
        start - timedelta(days=30), end,
    )
    # H1 strict prior: ts < decision_ts.
    h1_prior = h1_full.filter(pl.col("ts") < decision_ts)
    h1_frame = h1_prior.tail(h1_lookback) if h1_prior.height >= h1_lookback else h1_prior

    # H4 strict prior: h4_open + 4h ≤ decision_ts.
    h4_full = lake.query(
        symbol, _tf_enum("H4"),
        start - timedelta(days=60), end,
    )
    h4_close_ok = h4_full.filter(
        (pl.col("ts") + _H4_PERIOD) <= decision_ts
    )
    h4_frame: pl.DataFrame | None = (
        h4_close_ok.tail(h4_lookback)
        if h4_close_ok.height >= h4_lookback
        else (h4_close_ok if h4_close_ok.height > 0 else None)
    )

    # D1 strict prior: day floor < day floor of decision_ts.
    d1_full = lake.query(
        symbol, _tf_enum("D1"),
        start - timedelta(days=120), end,
    )
    decision_day = _floor_to_day(decision_ts)
    d1_prior = d1_full.filter(pl.col("ts") < decision_day)
    d1_frame: pl.DataFrame | None = d1_prior if d1_prior.height > 0 else None

    # Self-report invariants from the assembled frames.
    h1_max_ts = (
        h1_frame["ts"].max() if h1_frame.height > 0 else None
    )
    h4_max_ts = (
        h4_frame["ts"].max() if h4_frame is not None and h4_frame.height > 0 else None
    )
    d1_max_ts = (
        d1_frame["ts"].max() if d1_frame is not None and d1_frame.height > 0 else None
    )

    decision_only_strict_prior = (h1_max_ts is None) or (h1_max_ts < decision_ts)
    h4_partial = (
        h4_max_ts is not None
        and (h4_max_ts + _H4_PERIOD) > decision_ts
    )
    d1_partial = d1_max_ts is not None and d1_max_ts >= decision_day

    invariants = DecisionWindowInvariants(
        same_bar_set_used=True,
        decision_only_uses_strictly_prior_data=decision_only_strict_prior,
        h4_partial_bar_in_window=h4_partial,
        d1_partial_bar_in_window=d1_partial,
        decision_uses_data_with_ts_lt_trade_bar_ts=decision_only_strict_prior,
    )

    return DecisionWindow(
        decision_ts=decision_ts,
        h1_frame=h1_frame,
        h4_frame=h4_frame,
        d1_frame=d1_frame,
        invariants=invariants,
    )
