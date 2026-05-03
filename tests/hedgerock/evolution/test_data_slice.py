"""Ticket 2 Step 2 — Data-slice invariant tests.

Pinned guarantees (per R6 step 2):
  - DataSliceIdentity is deterministic for the same (lake, symbol,
    range): identical hashes + row counts on repeat calls.
  - Different time ranges → different lake_snapshot_hash.
  - Per-bar trailing windows respect strict-prior closed-bar rule:
      * H1 frame for decision at ts_i contains only bars with ts < ts_i.
      * H4 frame contains only H4 bars whose 4-hour period CLOSED
        before ts_i (no partial / in-progress H4 bar).
      * D1 frame's day floor is strictly < day floor of ts_i (no
        in-progress D1 bar).
  - decision-vs-trade ts separation: bar i's decision sees data with
    ts < ts_i; bar i's OHLC is the trade fill but NOT the decision input.
  - closed_bar_rule_version is fixed and reported.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl
import pytest

from smc.hedgerock.evolution.data_slice import (
    CLOSED_BAR_RULE_VERSION,
    DataSliceIdentity,
    build_decision_window,
    compute_data_slice_identity,
)


# ---------------------------------------------------------------------------
# Synthetic lake stub (mirrors Phase D test pattern)
# ---------------------------------------------------------------------------


def _bars(start: datetime, n: int, hours_step: float = 1.0) -> pl.DataFrame:
    rows = []
    for i in range(n):
        ts = start + timedelta(hours=hours_step * i)
        rows.append({
            "ts": ts, "open": 100.0, "high": 100.5, "low": 99.5,
            "close": 100.0, "volume": 100.0,
        })
    if not rows:
        return pl.DataFrame()
    from smc.data.schemas import Timeframe  # noqa: F401  (import for parity)
    return pl.DataFrame(rows).with_columns(
        pl.col("ts").dt.replace_time_zone("UTC")
    )


class _StubLake:
    def __init__(self, data: dict[tuple[str, str], pl.DataFrame]):
        # key: (instrument, timeframe-as-str)
        self._data = data
        self._root = Path("/tmp/stub-lake")

    def list_instruments(self) -> list[str]:
        return sorted({k[0] for k in self._data})

    def query(self, instrument: str, timeframe, start, end) -> pl.DataFrame:
        df = self._data.get((instrument, str(timeframe)))
        if df is None or df.is_empty():
            return pl.DataFrame()
        return df.filter((pl.col("ts") >= start) & (pl.col("ts") < end))


@pytest.fixture
def stub_lake():
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return _StubLake({
        ("XAUUSD", "H1"): _bars(base, n=24 * 30),
        ("XAUUSD", "H4"): _bars(base, n=6 * 30, hours_step=4.0),
        ("XAUUSD", "D1"): _bars(base, n=30, hours_step=24.0),
    })


# ---------------------------------------------------------------------------
# 1. compute_data_slice_identity is deterministic
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_data_slice_identity_returns_typed_object(stub_lake) -> None:
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 31, tzinfo=timezone.utc)
    identity = compute_data_slice_identity(
        lake=stub_lake, symbol="XAUUSD", start=start, end=end,
    )
    assert isinstance(identity, DataSliceIdentity)
    assert identity.symbols == ("XAUUSD",)
    assert identity.timeframes == ("H1", "H4", "D1")
    assert identity.closed_bar_rule_version == CLOSED_BAR_RULE_VERSION
    assert "H1" in identity.lake_snapshot_row_counts
    assert identity.lake_snapshot_row_counts["H1"] > 0


@pytest.mark.unit
def test_data_slice_identity_deterministic(stub_lake) -> None:
    """Two calls with the same lake + window → identical hash and row counts."""
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 31, tzinfo=timezone.utc)
    a = compute_data_slice_identity(stub_lake, "XAUUSD", start, end)
    b = compute_data_slice_identity(stub_lake, "XAUUSD", start, end)
    assert a == b
    assert a.lake_snapshot_hash == b.lake_snapshot_hash
    assert a.lake_snapshot_row_counts == b.lake_snapshot_row_counts


@pytest.mark.unit
def test_data_slice_identity_changes_with_time_range(stub_lake) -> None:
    """Different time range → different hash (data slice identity)."""
    a = compute_data_slice_identity(
        stub_lake, "XAUUSD",
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 1, 31, tzinfo=timezone.utc),
    )
    b = compute_data_slice_identity(
        stub_lake, "XAUUSD",
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 2, 28, tzinfo=timezone.utc),
    )
    assert a.lake_snapshot_hash != b.lake_snapshot_hash


@pytest.mark.unit
def test_data_slice_identity_iso_dates_in_window(stub_lake) -> None:
    a = compute_data_slice_identity(
        stub_lake, "XAUUSD",
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 1, 31, tzinfo=timezone.utc),
    )
    assert a.time_range_start == "2024-01-01"
    assert a.time_range_end == "2024-01-31"


# ---------------------------------------------------------------------------
# 2. build_decision_window — strict-prior closed-bar rule
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_build_decision_window_h1_excludes_current_bar(stub_lake) -> None:
    """Decision at ts_i must see H1 bars with ts < ts_i ONLY. The
    bar AT ts_i is the trade-fill bar, not a decision input."""
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 31, tzinfo=timezone.utc)
    decision_ts = datetime(2024, 1, 15, 12, tzinfo=timezone.utc)

    window = build_decision_window(
        lake=stub_lake, symbol="XAUUSD",
        start=start, end=end,
        decision_ts=decision_ts,
        h1_lookback=240, h4_lookback=60,
    )
    h1_ts_max = window.h1_frame["ts"].max()
    assert h1_ts_max < decision_ts, (
        f"H1 frame contains bar at {h1_ts_max} but decision at {decision_ts} "
        "must only see strictly prior bars (lookahead violation)"
    )


@pytest.mark.unit
def test_build_decision_window_h4_no_partial_bar(stub_lake) -> None:
    """H4 frame must not contain an in-progress 4-hour bar; only
    bars whose 4-hour period closed before ts_i."""
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 31, tzinfo=timezone.utc)
    # decision_ts = 14:00 → the H4 bar [12:00, 16:00) is still in progress
    decision_ts = datetime(2024, 1, 15, 14, tzinfo=timezone.utc)

    window = build_decision_window(
        lake=stub_lake, symbol="XAUUSD",
        start=start, end=end,
        decision_ts=decision_ts,
        h1_lookback=240, h4_lookback=60,
    )
    if window.h4_frame is not None and window.h4_frame.height > 0:
        h4_ts_max = window.h4_frame["ts"].max()
        # 4-hour period: bar opened at h4_ts_max closes at h4_ts_max + 4h.
        # Must close BEFORE decision_ts (≤).
        assert h4_ts_max + timedelta(hours=4) <= decision_ts, (
            f"H4 partial bar present: bar at {h4_ts_max} closes at "
            f"{h4_ts_max + timedelta(hours=4)}, decision_ts={decision_ts}"
        )


@pytest.mark.unit
def test_build_decision_window_d1_uses_prior_closed_day(stub_lake) -> None:
    """D1 frame day floor must be strictly less than day floor of ts_i."""
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 31, tzinfo=timezone.utc)
    decision_ts = datetime(2024, 1, 15, 12, tzinfo=timezone.utc)
    decision_day = decision_ts.replace(hour=0, minute=0, second=0, microsecond=0)

    window = build_decision_window(
        lake=stub_lake, symbol="XAUUSD",
        start=start, end=end,
        decision_ts=decision_ts,
        h1_lookback=240, h4_lookback=60,
    )
    if window.d1_frame is not None and window.d1_frame.height > 0:
        d1_ts_max = window.d1_frame["ts"].max()
        assert d1_ts_max < decision_day, (
            f"D1 frame includes day {d1_ts_max} ≥ decision day "
            f"{decision_day} (lookahead violation)"
        )


@pytest.mark.unit
def test_build_decision_window_invariants_self_report_clean(stub_lake) -> None:
    """The window object MUST self-report each invariant; all True
    on a clean call."""
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 31, tzinfo=timezone.utc)
    decision_ts = datetime(2024, 1, 15, 12, tzinfo=timezone.utc)
    window = build_decision_window(
        lake=stub_lake, symbol="XAUUSD",
        start=start, end=end,
        decision_ts=decision_ts,
        h1_lookback=240, h4_lookback=60,
    )
    assert window.invariants.same_bar_set_used is True
    assert window.invariants.decision_only_uses_strictly_prior_data is True
    assert window.invariants.h4_partial_bar_in_window is False
    assert window.invariants.d1_partial_bar_in_window is False
    assert window.invariants.decision_uses_data_with_ts_lt_trade_bar_ts is True


@pytest.mark.unit
def test_build_decision_window_two_calls_deterministic(stub_lake) -> None:
    """Same inputs → same H1/H4/D1 frame bytes (no clock-/state-dependent
    randomness)."""
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 31, tzinfo=timezone.utc)
    decision_ts = datetime(2024, 1, 15, 12, tzinfo=timezone.utc)
    a = build_decision_window(
        lake=stub_lake, symbol="XAUUSD", start=start, end=end,
        decision_ts=decision_ts, h1_lookback=240, h4_lookback=60,
    )
    b = build_decision_window(
        lake=stub_lake, symbol="XAUUSD", start=start, end=end,
        decision_ts=decision_ts, h1_lookback=240, h4_lookback=60,
    )
    assert a.h1_frame.equals(b.h1_frame)
    if a.h4_frame is not None:
        assert a.h4_frame.equals(b.h4_frame)
    if a.d1_frame is not None:
        assert a.d1_frame.equals(b.d1_frame)


# ---------------------------------------------------------------------------
# 3. closed_bar_rule_version is stable + tagged
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_closed_bar_rule_version_is_pinned() -> None:
    """The rule version is a public string constant. Bumping it is
    a major-version event and must be paired with an artefact
    schema bump."""
    assert isinstance(CLOSED_BAR_RULE_VERSION, str)
    assert CLOSED_BAR_RULE_VERSION.startswith("phase_d_strict_prior")


# ---------------------------------------------------------------------------
# 4. Decision-vs-trade ts separation (encoded in invariant flag)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_decision_vs_trade_ts_separation_encoded(stub_lake) -> None:
    """The window's invariant.decision_uses_data_with_ts_lt_trade_bar_ts
    flag must be True after build_decision_window. This is the
    contract the runner relies on; if False, runner must ABSTAIN."""
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 31, tzinfo=timezone.utc)
    decision_ts = datetime(2024, 1, 15, 12, tzinfo=timezone.utc)
    window = build_decision_window(
        lake=stub_lake, symbol="XAUUSD", start=start, end=end,
        decision_ts=decision_ts, h1_lookback=240, h4_lookback=60,
    )
    assert window.invariants.decision_uses_data_with_ts_lt_trade_bar_ts is True
