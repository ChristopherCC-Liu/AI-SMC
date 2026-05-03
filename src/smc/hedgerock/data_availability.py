"""Phase D-cont3-preflight — Data availability scanner (DIAGNOSTIC ONLY).

Read-only scan over the ForexDataLake to answer:

    - Which instruments are present?
    - For each (instrument, timeframe), what date range is covered?
    - How many bars? Are there gaps?
    - Is the coverage sufficient for ≥3y multi-year replication?
    - Do we have a second symbol for cross-symbol robustness?

The scanner does NOT load full bar data into memory long-term — it
reads each month's parquet only to compute first/last ts and bar
count. Gap detection inspects consecutive timestamp deltas.

This module does NOT touch rule_engine, decision_server, or any
production trading code. It is a pure introspection layer for
deciding what Phase D-cont3 follow-on work the data actually
supports.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable

import polars as pl

from smc.data.schemas import Timeframe


__all__ = [
    "EXPECTED_BARS_PER_WEEK",
    "REPLICATION_MIN_DAYS",
    "REPLICATION_MIN_YEARS",
    "SymbolCoverage",
    "AvailabilityReport",
    "expected_interval_hours",
    "compute_gap_summary",
    "scan_symbol_timeframe",
    "scan_lake",
]


# Forex 24/5 market — H1 has at most 120 bars/week, H4 at most 30,
# D1 at most 5. These are upper bounds for completeness ratio
# computation; the actual count varies by holiday.
EXPECTED_BARS_PER_WEEK: dict[Timeframe, int] = {
    Timeframe.H1: 120,
    Timeframe.H4: 30,
    Timeframe.D1: 5,
}

# Bar-spacing in hours, used for intra-week gap detection.
def expected_interval_hours(tf: Timeframe) -> float:
    return {Timeframe.H1: 1.0, Timeframe.H4: 4.0, Timeframe.D1: 24.0}[tf]


# A symbol is sufficient for a "multi-year" replication when it spans
# at least this many years of contiguous-ish coverage.
REPLICATION_MIN_YEARS: int = 3
REPLICATION_MIN_DAYS: int = REPLICATION_MIN_YEARS * 365


@dataclass(frozen=True)
class SymbolCoverage:
    """Per-(instrument, timeframe) coverage descriptor.

    Frozen so repeated scans yield comparable, hashable results.
    """

    instrument: str
    timeframe: Timeframe
    start_ts: datetime | None
    end_ts: datetime | None
    bar_count: int
    span_days: int
    # Largest intra-week gap (hours). Weekend gaps (Sat 00 → Mon 00,
    # ~48-72h) are NOT counted as anomalies; only deltas larger than
    # `expected_interval` AND smaller than 48h count as intra-week.
    largest_intraweek_gap_hours: float
    intraweek_gap_count: int
    # Coverage completeness vs theoretical 24/5 schedule. Crude — for
    # the diagnostic, anything ≥ 0.80 is "good"; < 0.50 is "thin".
    completeness_ratio: float

    @property
    def sufficient_for_3y_replication(self) -> bool:
        """True when this (instrument, timeframe) spans ≥ 3 years of
        bar data with reasonable completeness. Used by the replication
        runner to decide which symbol/year combos are worth running."""
        if self.span_days < REPLICATION_MIN_DAYS:
            return False
        if self.completeness_ratio < 0.50:
            return False
        return True


@dataclass(frozen=True)
class AvailabilityReport:
    lake_root: str
    instruments: tuple[str, ...]
    coverages: tuple[SymbolCoverage, ...]

    def by_instrument(self, instrument: str) -> tuple[SymbolCoverage, ...]:
        return tuple(c for c in self.coverages if c.instrument == instrument)

    def replication_candidates(self) -> tuple[str, ...]:
        """Instruments where H1 + H4 + D1 each pass the 3y bar."""
        ok: list[str] = []
        for inst in self.instruments:
            tf_to_cov = {
                c.timeframe: c for c in self.by_instrument(inst)
            }
            needed = (Timeframe.H1, Timeframe.H4, Timeframe.D1)
            if not all(t in tf_to_cov for t in needed):
                continue
            if not all(tf_to_cov[t].sufficient_for_3y_replication for t in needed):
                continue
            ok.append(inst)
        return tuple(ok)


# ---------------------------------------------------------------------------
# Gap detection
# ---------------------------------------------------------------------------


def compute_gap_summary(
    df: pl.DataFrame, *, timeframe: Timeframe,
) -> tuple[int, float]:
    """Find intra-week gaps in the bar sequence.

    Returns ``(intraweek_gap_count, largest_intraweek_gap_hours)``.

    A "gap" is a delta between consecutive timestamps that exceeds
    the expected interval. Weekend gaps (40h–80h, roughly Sat 00 →
    Mon 00) are excluded — those are normal forex closures. Any
    delta strictly between expected_interval and 40h counts.
    """
    if df.is_empty() or df.height < 2:
        return 0, 0.0
    interval_h = expected_interval_hours(timeframe)
    ts = df["ts"].to_list()
    weekend_lo = 40.0
    weekend_hi = 80.0
    count = 0
    largest = 0.0
    for i in range(1, len(ts)):
        delta_h = (ts[i] - ts[i - 1]).total_seconds() / 3600.0
        if delta_h <= interval_h + 1e-6:
            continue  # contiguous
        if weekend_lo <= delta_h <= weekend_hi:
            continue  # normal weekend
        # Intra-week gap.
        count += 1
        largest = max(largest, delta_h)
    return count, largest


# ---------------------------------------------------------------------------
# Scanners
# ---------------------------------------------------------------------------


def _completeness(span_days: int, bar_count: int, tf: Timeframe) -> float:
    """Crude actual / expected ratio assuming 24/5 forex schedule."""
    if span_days <= 0:
        return 0.0
    expected_bars_per_week = EXPECTED_BARS_PER_WEEK[tf]
    expected = expected_bars_per_week * (span_days / 7.0)
    if expected <= 0:
        return 0.0
    return min(1.0, bar_count / expected)


def scan_symbol_timeframe(
    lake, instrument: str, timeframe: Timeframe,
) -> SymbolCoverage:
    """Compute coverage for a single (instrument, timeframe) pair.

    Reads the full timeframe in one query — fine for diagnostic use,
    a 5-year H1 load is a few MB.
    """
    far_past = datetime(2000, 1, 1, tzinfo=timezone.utc)
    far_future = datetime(2100, 1, 1, tzinfo=timezone.utc)
    df = lake.query(instrument, timeframe, far_past, far_future)
    if df.is_empty():
        return SymbolCoverage(
            instrument=instrument, timeframe=timeframe,
            start_ts=None, end_ts=None,
            bar_count=0, span_days=0,
            largest_intraweek_gap_hours=0.0,
            intraweek_gap_count=0,
            completeness_ratio=0.0,
        )
    df = df.sort("ts")
    start = df["ts"][0]
    end = df["ts"][-1]
    span_days = (end - start).days
    gap_count, largest_gap = compute_gap_summary(df, timeframe=timeframe)
    completeness = _completeness(span_days, df.height, timeframe)
    return SymbolCoverage(
        instrument=instrument, timeframe=timeframe,
        start_ts=start, end_ts=end,
        bar_count=df.height, span_days=span_days,
        largest_intraweek_gap_hours=largest_gap,
        intraweek_gap_count=gap_count,
        completeness_ratio=completeness,
    )


def scan_lake(
    lake, *,
    timeframes: Iterable[Timeframe] = (Timeframe.H1, Timeframe.H4, Timeframe.D1),
) -> AvailabilityReport:
    """Scan every instrument in the lake across the requested
    timeframes. Returns an :class:`AvailabilityReport` with one
    :class:`SymbolCoverage` per (instrument, timeframe).
    """
    instruments = tuple(lake.list_instruments()) if hasattr(
        lake, "list_instruments"
    ) else ()
    coverages: list[SymbolCoverage] = []
    for inst in instruments:
        for tf in timeframes:
            coverages.append(scan_symbol_timeframe(lake, inst, tf))
    root_repr = str(getattr(lake, "_root", ""))
    return AvailabilityReport(
        lake_root=root_repr,
        instruments=instruments,
        coverages=tuple(coverages),
    )
