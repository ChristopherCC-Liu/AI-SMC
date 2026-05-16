"""Phase D-cont3-preflight — data availability + year-replication tests.

Pinned guarantees:
    1. compute_gap_summary correctly classifies weekend vs intra-week gaps.
    2. scan_symbol_timeframe returns deterministic SymbolCoverage on a
       synthetic lake.
    3. SymbolCoverage.sufficient_for_3y_replication enforces both
       span ≥ REPLICATION_MIN_DAYS AND completeness ≥ 0.50.
    4. AvailabilityReport.replication_candidates() returns only
       instruments where ALL of H1 / H4 / D1 pass the bar.
    5. build_replication_cell extracts the right buckets from an
       AtlasReport and computes a 95% CI.
    6. ReplicationCell verdict prose: INCONCLUSIVE under sample floor,
       (NEG) when CI excludes zero negative, (CI∋0) when CI brackets zero.
    7. replication_e1_verdict: single-symbol gating, year-pass counting.
    8. Report writer emits the required sections.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl
import pytest

from smc.data.schemas import Timeframe
from smc.hedgerock.data_availability import (
    EXPECTED_BARS_PER_WEEK,
    REPLICATION_MIN_DAYS,
    REPLICATION_MIN_YEARS,
    AvailabilityReport,
    SymbolCoverage,
    compute_gap_summary,
    expected_interval_hours,
    scan_lake,
    scan_symbol_timeframe,
)


# ---------------------------------------------------------------------------
# Lake stub
# ---------------------------------------------------------------------------


def _bars(start: datetime, n: int, hours_step: float = 1.0,
          *, drop_indices: set[int] | None = None) -> pl.DataFrame:
    rows = []
    for i in range(n):
        if drop_indices and i in drop_indices:
            continue
        ts = start + timedelta(hours=hours_step * i)
        rows.append({
            "ts": ts, "open": 100.0, "high": 100.5, "low": 99.5,
            "close": 100.0, "volume": 100.0,
        })
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows).with_columns(
        pl.col("ts").dt.replace_time_zone("UTC")
    )


class _StubLake:
    """Minimal in-memory lake for scanner tests."""

    def __init__(self, data: dict[tuple[str, Timeframe], pl.DataFrame]):
        self._data = data
        self._root = Path("/tmp/synthetic-lake")

    def list_instruments(self) -> list[str]:
        return sorted({k[0] for k in self._data})

    def query(self, instrument: str, timeframe: Timeframe,
              start, end) -> pl.DataFrame:
        df = self._data.get((instrument, timeframe))
        if df is None or df.is_empty():
            return pl.DataFrame()
        return df.filter((pl.col("ts") >= start) & (pl.col("ts") < end))


# ---------------------------------------------------------------------------
# 1. compute_gap_summary — weekend exclusion
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_gap_summary_contiguous_h1_no_gaps() -> None:
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    df = _bars(base, n=24)
    count, largest = compute_gap_summary(df, timeframe=Timeframe.H1)
    assert count == 0
    assert largest == 0.0


@pytest.mark.unit
def test_gap_summary_excludes_weekend_gap() -> None:
    """A 65-hour gap between Friday close and Monday open is normal
    forex closure — must NOT count as an intra-week gap."""
    rows = [
        # Friday 23:00 UTC
        {"ts": datetime(2024, 1, 5, 23, tzinfo=timezone.utc),
         "open": 100, "high": 100, "low": 100, "close": 100, "volume": 1.0},
        # Monday 00:00 UTC — 49h later
        {"ts": datetime(2024, 1, 8, 0, tzinfo=timezone.utc),
         "open": 100, "high": 100, "low": 100, "close": 100, "volume": 1.0},
    ]
    df = pl.DataFrame(rows).with_columns(
        pl.col("ts").dt.replace_time_zone("UTC")
    )
    count, largest = compute_gap_summary(df, timeframe=Timeframe.H1)
    assert count == 0
    assert largest == 0.0


@pytest.mark.unit
def test_gap_summary_counts_intraweek_gap() -> None:
    """A 5-hour gap mid-day Tuesday IS an anomaly."""
    rows = [
        {"ts": datetime(2024, 1, 9, 10, tzinfo=timezone.utc),  # Tue 10:00
         "open": 100, "high": 100, "low": 100, "close": 100, "volume": 1.0},
        {"ts": datetime(2024, 1, 9, 15, tzinfo=timezone.utc),  # Tue 15:00 (5h gap)
         "open": 100, "high": 100, "low": 100, "close": 100, "volume": 1.0},
    ]
    df = pl.DataFrame(rows).with_columns(
        pl.col("ts").dt.replace_time_zone("UTC")
    )
    count, largest = compute_gap_summary(df, timeframe=Timeframe.H1)
    assert count == 1
    assert largest == pytest.approx(5.0)


@pytest.mark.unit
def test_gap_summary_empty_dataframe() -> None:
    count, largest = compute_gap_summary(pl.DataFrame(), timeframe=Timeframe.H1)
    assert count == 0
    assert largest == 0.0


# ---------------------------------------------------------------------------
# 2. scan_symbol_timeframe + SymbolCoverage
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_scan_symbol_timeframe_records_first_last_bar_count() -> None:
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    df = _bars(base, n=240)  # 10 days of H1
    lake = _StubLake({("XAUUSD", Timeframe.H1): df})
    cov = scan_symbol_timeframe(lake, "XAUUSD", Timeframe.H1)
    assert cov.instrument == "XAUUSD"
    assert cov.timeframe == Timeframe.H1
    assert cov.start_ts == base
    assert cov.bar_count == 240
    assert cov.span_days >= 9  # 240 hours = 10 days
    # Since span is far short of 3y, sufficiency must be False.
    assert cov.sufficient_for_3y_replication is False


@pytest.mark.unit
def test_scan_returns_zeros_for_missing_instrument() -> None:
    lake = _StubLake({})
    cov = scan_symbol_timeframe(lake, "EURUSD", Timeframe.H1)
    assert cov.bar_count == 0
    assert cov.start_ts is None
    assert cov.end_ts is None
    assert cov.completeness_ratio == 0.0
    assert cov.sufficient_for_3y_replication is False


@pytest.mark.unit
def test_sufficiency_requires_both_span_and_completeness() -> None:
    """Exactly 3y span, but only 1% completeness → not sufficient.
    Exactly 3y span with 0.55 completeness → sufficient."""
    base = datetime(2020, 1, 1, tzinfo=timezone.utc)
    end = datetime(2023, 1, 1, tzinfo=timezone.utc)
    span_days = (end - base).days
    expected = EXPECTED_BARS_PER_WEEK[Timeframe.H1] * (span_days / 7.0)

    # Thin: only 100 bars over 3y → completeness ≈ 0
    thin = SymbolCoverage(
        instrument="X", timeframe=Timeframe.H1,
        start_ts=base, end_ts=end,
        bar_count=100, span_days=span_days,
        largest_intraweek_gap_hours=0.0, intraweek_gap_count=0,
        completeness_ratio=100 / expected,
    )
    assert thin.sufficient_for_3y_replication is False

    # Healthy: completeness ≥ 0.50
    ok = SymbolCoverage(
        instrument="X", timeframe=Timeframe.H1,
        start_ts=base, end_ts=end,
        bar_count=int(expected * 0.6), span_days=span_days,
        largest_intraweek_gap_hours=0.0, intraweek_gap_count=0,
        completeness_ratio=0.60,
    )
    assert ok.sufficient_for_3y_replication is True

    # Short window even with full completeness: not sufficient.
    short = SymbolCoverage(
        instrument="X", timeframe=Timeframe.H1,
        start_ts=base, end_ts=base + timedelta(days=30),
        bar_count=720, span_days=30,
        largest_intraweek_gap_hours=0.0, intraweek_gap_count=0,
        completeness_ratio=1.0,
    )
    assert short.sufficient_for_3y_replication is False


# ---------------------------------------------------------------------------
# 3. scan_lake + replication_candidates
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_scan_lake_iterates_all_instruments_x_timeframes() -> None:
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    h1 = _bars(base, n=24)
    h4 = _bars(base, n=6, hours_step=4.0)
    d1 = _bars(base, n=1, hours_step=24.0)
    lake = _StubLake({
        ("XAUUSD", Timeframe.H1): h1,
        ("XAUUSD", Timeframe.H4): h4,
        ("XAUUSD", Timeframe.D1): d1,
        ("EURUSD", Timeframe.H1): h1,  # only one timeframe → not candidate
    })
    rep = scan_lake(lake)
    assert set(rep.instruments) == {"EURUSD", "XAUUSD"}
    # 2 instruments × 3 timeframes = 6 coverages.
    assert len(rep.coverages) == 6


@pytest.mark.unit
def test_replication_candidates_requires_all_three_timeframes() -> None:
    """Only an instrument where H1 + H4 + D1 EACH pass the 3y/0.50
    bar appears in replication_candidates."""
    base = datetime(2020, 1, 1, tzinfo=timezone.utc)
    end = datetime(2023, 6, 1, tzinfo=timezone.utc)  # ~3.5 years
    # Generous bar counts to clear completeness 0.50.
    span_h1 = int((end - base).total_seconds() / 3600)  # one bar/hr
    span_h4 = span_h1 // 4
    span_d1 = (end - base).days
    h1 = _bars(base, n=span_h1)
    h4 = _bars(base, n=span_h4, hours_step=4.0)
    d1 = _bars(base, n=span_d1, hours_step=24.0)
    # XAUUSD has all three; EURUSD has only H1.
    lake = _StubLake({
        ("XAUUSD", Timeframe.H1): h1,
        ("XAUUSD", Timeframe.H4): h4,
        ("XAUUSD", Timeframe.D1): d1,
        ("EURUSD", Timeframe.H1): h1,
    })
    rep = scan_lake(lake)
    candidates = rep.replication_candidates()
    assert candidates == ("XAUUSD",)


# ---------------------------------------------------------------------------
# 4. build_replication_cell + verdict logic
# ---------------------------------------------------------------------------


def _import_repl_helpers():
    import sys
    scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
    sys.path.insert(0, str(scripts_dir))
    try:
        from hedgerock_data_availability import (
            ReplicationCell,
            build_replication_cell,
            replication_e1_verdict,
            _write_report,
            _ci_95,
        )
    finally:
        sys.path.pop(0)
    return ReplicationCell, build_replication_cell, replication_e1_verdict, _write_report, _ci_95


@pytest.mark.unit
def test_ci_95_zero_when_n_below_two() -> None:
    _RC, _build, _verdict, _wr, ci_95 = _import_repl_helpers()
    assert ci_95(stdev=1.0, n=1) == 0.0
    assert ci_95(stdev=0.0, n=100) == 0.0


@pytest.mark.unit
def test_replication_cell_inconclusive_below_min_sample() -> None:
    RC, _b, _v, _wr, _c = _import_repl_helpers()
    cell = RC(
        symbol="XAUUSD", year=2020, decision_bars=100,
        trend_up_n=10, trend_up_mean=1.0, trend_up_ci_95=0.1,
        range_aggressive_n=200, range_aggressive_mean=0.05, range_aggressive_ci_95=0.02,
        breakout_signed_n=5, breakout_signed_mean=0.3, breakout_signed_ci_95=0.1,
        halt_event_count=3, min_sample_for_signal=30,
    )
    # n=10 below floor 30 → INCONCLUSIVE
    assert "INCONCLUSIVE" in cell.trend_up_verdict
    # n=200 above floor with mean > CI → positive
    assert cell.range_aggressive_verdict.startswith("+")
    # n=5 below floor 30 → INCONCLUSIVE (regardless of mean magnitude)
    assert "INCONCLUSIVE" in cell.breakout_signed_verdict


@pytest.mark.unit
def test_replication_cell_neg_marker_when_ci_excludes_zero_negative() -> None:
    RC, _b, _v, _wr, _c = _import_repl_helpers()
    cell = RC(
        symbol="XAUUSD", year=2020, decision_bars=100,
        trend_up_n=400, trend_up_mean=-0.5, trend_up_ci_95=0.1,
        range_aggressive_n=0, range_aggressive_mean=0.0, range_aggressive_ci_95=0.0,
        breakout_signed_n=0, breakout_signed_mean=0.0, breakout_signed_ci_95=0.0,
        halt_event_count=0, min_sample_for_signal=30,
    )
    assert "(NEG)" in cell.trend_up_verdict


@pytest.mark.unit
def test_replication_cell_ci_brackets_zero_marker() -> None:
    RC, _b, _v, _wr, _c = _import_repl_helpers()
    cell = RC(
        symbol="XAUUSD", year=2020, decision_bars=100,
        trend_up_n=400, trend_up_mean=0.05, trend_up_ci_95=0.10,
        range_aggressive_n=0, range_aggressive_mean=0.0, range_aggressive_ci_95=0.0,
        breakout_signed_n=0, breakout_signed_mean=0.0, breakout_signed_ci_95=0.0,
        halt_event_count=0, min_sample_for_signal=30,
    )
    # mean=0.05, CI=±0.10 → CI brackets zero
    assert "(CI∋0)" in cell.trend_up_verdict


@pytest.mark.unit
def test_build_replication_cell_extracts_correct_buckets() -> None:
    """build_replication_cell pulls trend_up by regime-name match,
    range@>=0.80 by bucket-name match, breakout_signed_by_h4 by
    regime-name match. None of them confused."""
    from smc.hedgerock.regime_opportunity_atlas import (
        AtlasReport, AtlasConfig, GraceWaterfall, RegimeBucketStat,
    )
    _RC, build_cell, _v, _wr, _c = _import_repl_helpers()
    cfg = AtlasConfig(min_sample_for_signal=30)
    range_stats = [
        RegimeBucketStat(regime="range", bucket="<0.45", count=0,
                         mean_return_24h=0, median_return_24h=0,
                         stdev_return_24h=0, mean_mae_24h=0,
                         mean_mfe_24h=0, mean_range_width_24h=0),
        RegimeBucketStat(regime="range", bucket=">=0.80", count=200,
                         mean_return_24h=0.20, median_return_24h=0.18,
                         stdev_return_24h=0.85, mean_mae_24h=-0.6,
                         mean_mfe_24h=0.7, mean_range_width_24h=1.3),
    ]
    trend_stats = [
        RegimeBucketStat(regime="trend_up", bucket="all", count=900,
                         mean_return_24h=0.14, median_return_24h=0.20,
                         stdev_return_24h=1.05, mean_mae_24h=-0.7,
                         mean_mfe_24h=0.86, mean_range_width_24h=1.5),
        RegimeBucketStat(regime="breakout_signed_by_h4", bucket="h4_trend proxy",
                         count=350, mean_return_24h=-0.02, median_return_24h=0.05,
                         stdev_return_24h=0.95, mean_mae_24h=-0.91,
                         mean_mfe_24h=0.78, mean_range_width_24h=1.6),
    ]
    report = AtlasReport(
        config=cfg, records=[], waterfall=GraceWaterfall(stages=()),
        regime_x_bucket_count={},
        range_opportunity=range_stats,
        trend_opportunity=trend_stats,
        halt_aftermath=[],
    )
    cell = build_cell(symbol="XAUUSD", year=2020, atlas=report, min_sample=30)
    # trend_up extracted correctly
    assert cell.trend_up_n == 900
    assert cell.trend_up_mean == pytest.approx(0.14)
    # range@>=0.80 extracted correctly
    assert cell.range_aggressive_n == 200
    assert cell.range_aggressive_mean == pytest.approx(0.20)
    # breakout_signed_by_h4 extracted correctly — NOT confused with magnitude
    assert cell.breakout_signed_n == 350
    assert cell.breakout_signed_mean == pytest.approx(-0.02)


# ---------------------------------------------------------------------------
# 5. replication_e1_verdict
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_e1_verdict_single_symbol_blocks_promotion() -> None:
    _RC, _b, e1_verdict, _wr, _c = _import_repl_helpers()
    RC, *_ = _import_repl_helpers()
    # 5 years of XAUUSD all passing → still NOT promotable because
    # second symbol is missing.
    cells = [
        RC(symbol="XAUUSD", year=y, decision_bars=5500,
           trend_up_n=900, trend_up_mean=0.15, trend_up_ci_95=0.07,
           range_aggressive_n=2000, range_aggressive_mean=0.18,
           range_aggressive_ci_95=0.04,
           breakout_signed_n=350, breakout_signed_mean=-0.02,
           breakout_signed_ci_95=0.10,
           halt_event_count=1, min_sample_for_signal=30)
        for y in range(2020, 2025)
    ]
    headline, bullets = e1_verdict(cells)
    assert "NO second-symbol evidence" in headline
    assert any("Only 1 symbol" in b for b in bullets)


@pytest.mark.unit
def test_e1_verdict_year_replication_partial_does_not_promote() -> None:
    _RC, _b, e1_verdict, _wr, _c = _import_repl_helpers()
    RC, *_ = _import_repl_helpers()
    # 5 years, only 2 pass → PARTIAL, no promotion.
    cells = []
    for y, passing in zip(range(2020, 2025),
                          [True, True, False, False, False]):
        if passing:
            mean, ci = 0.15, 0.07
        else:
            mean, ci = 0.05, 0.10  # CI brackets zero
        cells.append(RC(
            symbol="XAUUSD", year=y, decision_bars=5500,
            trend_up_n=900, trend_up_mean=mean, trend_up_ci_95=ci,
            range_aggressive_n=2000, range_aggressive_mean=0.10,
            range_aggressive_ci_95=0.05,
            breakout_signed_n=350, breakout_signed_mean=-0.02,
            breakout_signed_ci_95=0.10,
            halt_event_count=1, min_sample_for_signal=30,
        ))
    headline, _ = e1_verdict(cells)
    assert "PARTIAL" in headline
    assert "promote" in headline.lower()


@pytest.mark.unit
def test_e1_verdict_year_replication_zero_pass_rejects() -> None:
    _RC, _b, e1_verdict, _wr, _c = _import_repl_helpers()
    RC, *_ = _import_repl_helpers()
    # 5 years, all CI brackets zero → 2024 single-window edge was noise.
    cells = [
        RC(symbol="XAUUSD", year=y, decision_bars=5500,
           trend_up_n=900, trend_up_mean=0.05, trend_up_ci_95=0.10,
           range_aggressive_n=2000, range_aggressive_mean=0.05,
           range_aggressive_ci_95=0.05,
           breakout_signed_n=350, breakout_signed_mean=-0.02,
           breakout_signed_ci_95=0.10,
           halt_event_count=1, min_sample_for_signal=30)
        for y in range(2020, 2025)
    ]
    headline, _ = e1_verdict(cells)
    assert "does NOT replicate" in headline


@pytest.mark.unit
def test_e1_verdict_empty_input_safe() -> None:
    _RC, _b, e1_verdict, _wr, _c = _import_repl_helpers()
    headline, bullets = e1_verdict([])
    assert "INSUFFICIENT" in headline
    assert bullets == []


# ---------------------------------------------------------------------------
# 6. Report writer renders required sections
# ---------------------------------------------------------------------------


def _import_action_gate():
    import sys
    scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
    sys.path.insert(0, str(scripts_dir))
    try:
        from hedgerock_data_availability import _fmt_action_gate
    finally:
        sys.path.pop(0)
    return _fmt_action_gate


@pytest.mark.unit
def test_action_gate_blocks_strategy_when_single_symbol() -> None:
    """A single-symbol lake should always trip
    NO_STRATEGY_CHANGE: true regardless of replication results."""
    fmt_gate = _import_action_gate()
    RC, *_ = _import_repl_helpers()
    base = datetime(2020, 1, 1, tzinfo=timezone.utc)
    cov = SymbolCoverage(
        instrument="XAUUSD", timeframe=Timeframe.H1,
        start_ts=base, end_ts=datetime(2024, 12, 31, tzinfo=timezone.utc),
        bar_count=30000, span_days=1825,
        largest_intraweek_gap_hours=2.0, intraweek_gap_count=4,
        completeness_ratio=0.85,
    )
    availability = AvailabilityReport(
        lake_root="/tmp/lake",
        instruments=("XAUUSD",),
        coverages=(cov,),
    )
    # Even with 5/5 years passing, single-symbol must still gate.
    cells = [
        RC(symbol="XAUUSD", year=y, decision_bars=5500,
           trend_up_n=900, trend_up_mean=0.20, trend_up_ci_95=0.05,
           range_aggressive_n=2000, range_aggressive_mean=0.18,
           range_aggressive_ci_95=0.04,
           breakout_signed_n=350, breakout_signed_mean=0.0,
           breakout_signed_ci_95=0.10,
           halt_event_count=10, min_sample_for_signal=30)
        for y in range(2020, 2025)
    ]
    text = "\n".join(fmt_gate(availability=availability, cells=cells))
    assert "NO_STRATEGY_CHANGE: true" in text
    assert "single-symbol lake" in text
    # Whitelist + blocklist both present.
    assert "data_acquisition_second_symbol" in text
    assert "momentum_module_E1" in text
    assert "Disallowed_next_work" in text


@pytest.mark.unit
def test_action_gate_lists_reverse_signed_year_explicitly() -> None:
    """When a year has trend_up CI excluding zero on the negative
    side, the gate must name that year — this is the strongest
    failure mode (classifier WRONG, not just inconclusive)."""
    fmt_gate = _import_action_gate()
    RC, *_ = _import_repl_helpers()
    base = datetime(2021, 1, 1, tzinfo=timezone.utc)
    cov = SymbolCoverage(
        instrument="XAUUSD", timeframe=Timeframe.H1,
        start_ts=base, end_ts=datetime(2024, 12, 31, tzinfo=timezone.utc),
        bar_count=30000, span_days=1500,
        largest_intraweek_gap_hours=2.0, intraweek_gap_count=4,
        completeness_ratio=0.85,
    )
    availability = AvailabilityReport(
        lake_root="/tmp/lake",
        instruments=("XAUUSD",),
        coverages=(cov,),
    )
    cells = [
        # 2021 is reverse-signed.
        RC(symbol="XAUUSD", year=2021, decision_bars=5500,
           trend_up_n=900, trend_up_mean=-0.07, trend_up_ci_95=0.05,
           range_aggressive_n=2000, range_aggressive_mean=0.0,
           range_aggressive_ci_95=0.05,
           breakout_signed_n=350, breakout_signed_mean=0.0,
           breakout_signed_ci_95=0.10,
           halt_event_count=1, min_sample_for_signal=30),
        # 2024 passes.
        RC(symbol="XAUUSD", year=2024, decision_bars=5500,
           trend_up_n=900, trend_up_mean=0.14, trend_up_ci_95=0.07,
           range_aggressive_n=2000, range_aggressive_mean=0.17,
           range_aggressive_ci_95=0.04,
           breakout_signed_n=350, breakout_signed_mean=-0.02,
           breakout_signed_ci_95=0.10,
           halt_event_count=1, min_sample_for_signal=30),
    ]
    text = "\n".join(fmt_gate(availability=availability, cells=cells))
    assert "REVERSE-signed" in text
    assert "2021" in text
    assert "NO_STRATEGY_CHANGE: true" in text


@pytest.mark.unit
def test_action_gate_flags_low_halt_event_corpus() -> None:
    fmt_gate = _import_action_gate()
    RC, *_ = _import_repl_helpers()
    base = datetime(2020, 1, 1, tzinfo=timezone.utc)
    cov = SymbolCoverage(
        instrument="XAUUSD", timeframe=Timeframe.H1,
        start_ts=base, end_ts=datetime(2024, 12, 31, tzinfo=timezone.utc),
        bar_count=30000, span_days=1825,
        largest_intraweek_gap_hours=2.0, intraweek_gap_count=4,
        completeness_ratio=0.85,
    )
    availability = AvailabilityReport(
        lake_root="/tmp/lake",
        instruments=("XAUUSD",),
        coverages=(cov,),
    )
    cells = [
        RC(symbol="XAUUSD", year=y, decision_bars=5500,
           trend_up_n=900, trend_up_mean=0.20, trend_up_ci_95=0.05,
           range_aggressive_n=2000, range_aggressive_mean=0.18,
           range_aggressive_ci_95=0.04,
           breakout_signed_n=350, breakout_signed_mean=0.0,
           breakout_signed_ci_95=0.10,
           halt_event_count=1, min_sample_for_signal=30)
        for y in range(2020, 2025)
    ]
    text = "\n".join(fmt_gate(availability=availability, cells=cells))
    # 5 halt events total, floor 30 → flagged.
    assert "halt-event sample n=5 < 30" in text
    assert "halt_event_corpus_expansion" in text


@pytest.mark.unit
def test_action_gate_clears_only_when_all_reasons_resolved() -> None:
    """Hypothetical: 2 symbols × 5 years × all-pass × halt corpus ≥30.
    All gates clear → NO_STRATEGY_CHANGE: false."""
    fmt_gate = _import_action_gate()
    RC, *_ = _import_repl_helpers()
    base = datetime(2020, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 12, 31, tzinfo=timezone.utc)
    cov_x = SymbolCoverage(
        instrument="XAUUSD", timeframe=Timeframe.H1,
        start_ts=base, end_ts=end, bar_count=30000, span_days=1825,
        largest_intraweek_gap_hours=0.0, intraweek_gap_count=0,
        completeness_ratio=0.9,
    )
    cov_e = SymbolCoverage(
        instrument="EURUSD", timeframe=Timeframe.H1,
        start_ts=base, end_ts=end, bar_count=30000, span_days=1825,
        largest_intraweek_gap_hours=0.0, intraweek_gap_count=0,
        completeness_ratio=0.9,
    )
    availability = AvailabilityReport(
        lake_root="/tmp/lake",
        instruments=("EURUSD", "XAUUSD"),
        coverages=(cov_x, cov_e),
    )
    cells = []
    for sym in ("XAUUSD", "EURUSD"):
        for y in range(2020, 2025):
            cells.append(RC(
                symbol=sym, year=y, decision_bars=5500,
                trend_up_n=900, trend_up_mean=0.20, trend_up_ci_95=0.05,
                range_aggressive_n=2000, range_aggressive_mean=0.18,
                range_aggressive_ci_95=0.04,
                breakout_signed_n=350, breakout_signed_mean=0.0,
                breakout_signed_ci_95=0.10,
                halt_event_count=8, min_sample_for_signal=30,
            ))
    text = "\n".join(fmt_gate(availability=availability, cells=cells))
    # 2 symbols × 5 years × 8 halts = 80 halt events; 5/5 trend_up
    # passing per symbol; both symbols present. Gate clears.
    assert "NO_STRATEGY_CHANGE: false" in text
    assert "Reason: []" in text
    assert "Disallowed_next_work" not in text


@pytest.mark.unit
def test_report_writer_includes_action_gate_section(tmp_path) -> None:
    """The Action gate MUST be part of the rendered report — without
    it, the next session might rely on the softer 'PARTIAL' wording
    in the verdict and start strategy work."""
    _RC, _b, _v, write_report, _c = _import_repl_helpers()
    RC, *_ = _import_repl_helpers()
    base = datetime(2020, 1, 1, tzinfo=timezone.utc)
    cov = SymbolCoverage(
        instrument="XAUUSD", timeframe=Timeframe.H1,
        start_ts=base, end_ts=datetime(2024, 12, 31, tzinfo=timezone.utc),
        bar_count=30000, span_days=1825,
        largest_intraweek_gap_hours=2.0, intraweek_gap_count=4,
        completeness_ratio=0.85,
    )
    availability = AvailabilityReport(
        lake_root="/tmp/lake",
        instruments=("XAUUSD",),
        coverages=(cov,),
    )
    cell = RC(
        symbol="XAUUSD", year=2024, decision_bars=5500,
        trend_up_n=900, trend_up_mean=0.14, trend_up_ci_95=0.07,
        range_aggressive_n=1700, range_aggressive_mean=0.17,
        range_aggressive_ci_95=0.04,
        breakout_signed_n=370, breakout_signed_mean=-0.02,
        breakout_signed_ci_95=0.10,
        halt_event_count=1, min_sample_for_signal=30,
    )
    out = tmp_path / "report.md"
    write_report(availability, [cell], min_sample=30, report_path=out)
    body = out.read_text()
    # Section heading + machine-greppable keys all present.
    assert "## Action gate" in body
    assert "NO_STRATEGY_CHANGE: true" in body
    assert "Allowed_next_work:" in body
    assert "Disallowed_next_work:" in body


@pytest.mark.unit
def test_report_writer_renders_required_sections(tmp_path) -> None:
    _RC, _b, _v, write_report, _c = _import_repl_helpers()
    RC, *_ = _import_repl_helpers()
    base = datetime(2020, 1, 1, tzinfo=timezone.utc)
    cov = SymbolCoverage(
        instrument="XAUUSD", timeframe=Timeframe.H1,
        start_ts=base, end_ts=datetime(2024, 12, 31, tzinfo=timezone.utc),
        bar_count=30000, span_days=1825,
        largest_intraweek_gap_hours=2.0, intraweek_gap_count=4,
        completeness_ratio=0.85,
    )
    availability = AvailabilityReport(
        lake_root="/tmp/lake",
        instruments=("XAUUSD",),
        coverages=(cov,),
    )
    cell = RC(
        symbol="XAUUSD", year=2024, decision_bars=5500,
        trend_up_n=900, trend_up_mean=0.14, trend_up_ci_95=0.07,
        range_aggressive_n=1700, range_aggressive_mean=0.17,
        range_aggressive_ci_95=0.04,
        breakout_signed_n=370, breakout_signed_mean=-0.02,
        breakout_signed_ci_95=0.10,
        halt_event_count=1, min_sample_for_signal=30,
    )
    out = tmp_path / "report.md"
    write_report(availability, [cell], min_sample=30, report_path=out)
    body = out.read_text()
    for required in (
        "## Lake scan",
        "## Year-replication summary",
        "## E1 trend_up promotion verdict",
        "## Action gate",
        "## Caveats",
    ):
        assert required in body, f"missing section {required!r}"
    # Cell verdict text propagates through the table.
    assert "XAUUSD" in body
    assert "2024" in body