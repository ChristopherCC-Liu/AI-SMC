"""Phase D-cont2 — Regime / Opportunity Atlas tests.

Pinned guarantees:
    1. confidence_bucket() maps the 5 named bands correctly and
       boundary values land in the higher-numbered bucket.
    2. first_grace_failure_gate() returns the FIRST failing gate in
       the production order; "" only when every gate passes.
    3. Outcome labels use ONLY future bars [i+1, i+24], NEVER bar i.
       Synthetic bar sequence with a known step verifies the read.
    4. The grace-eligibility waterfall is monotone non-increasing —
       no gate can ADD survivors.
    5. Atlas aggregations are deterministic — same input → same output.
    6. The report writer emits all required section headers when run
       end-to-end on a tiny synthetic lake.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl
import pytest

from smc.hedgerock.regime_opportunity_atlas import (
    AtlasConfig,
    AtlasReport,
    BUCKET_AGGRESSIVE,
    BUCKET_BAND_HI,
    BUCKET_BAND_LO,
    BUCKET_GRACE,
    BUCKET_LABELS,
    BUCKET_RANGE_2,
    DecisionRecord,
    GraceWaterfall,
    HaltAftermathRecord,
    OutcomeLabel,
    RegimeBucketStat,
    aggregate_grace_waterfall,
    aggregate_range_opportunity,
    aggregate_regime_x_confidence,
    aggregate_trend_opportunity,
    build_neutral_cold_ea_state,
    compute_outcome_labels,
    confidence_bucket,
    find_halt_aftermath,
    first_grace_failure_gate,
    merge_live_envelope_entry,
    run_atlas,
)


# ---------------------------------------------------------------------------
# 1. confidence_bucket — boundary semantics
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_confidence_bucket_lower_band() -> None:
    assert confidence_bucket(0.0) == BUCKET_BAND_LO
    assert confidence_bucket(0.44) == BUCKET_BAND_LO
    # 0.45 is the grace floor → goes UP to grace bucket.
    assert confidence_bucket(0.45) == BUCKET_GRACE


@pytest.mark.unit
def test_confidence_bucket_grace_band() -> None:
    assert confidence_bucket(0.45) == BUCKET_GRACE
    assert confidence_bucket(0.50) == BUCKET_GRACE
    assert confidence_bucket(0.5499) == BUCKET_GRACE
    # 0.55 is OBSERVE floor → goes UP.
    assert confidence_bucket(0.55) == BUCKET_BAND_HI


@pytest.mark.unit
def test_confidence_bucket_band_hi() -> None:
    assert confidence_bucket(0.55) == BUCKET_BAND_HI
    assert confidence_bucket(0.64) == BUCKET_BAND_HI
    assert confidence_bucket(0.65) == BUCKET_RANGE_2


@pytest.mark.unit
def test_confidence_bucket_range_2() -> None:
    assert confidence_bucket(0.65) == BUCKET_RANGE_2
    assert confidence_bucket(0.79) == BUCKET_RANGE_2
    assert confidence_bucket(0.80) == BUCKET_AGGRESSIVE


@pytest.mark.unit
def test_confidence_bucket_aggressive() -> None:
    assert confidence_bucket(0.80) == BUCKET_AGGRESSIVE
    assert confidence_bucket(0.95) == BUCKET_AGGRESSIVE
    assert confidence_bucket(1.00) == BUCKET_AGGRESSIVE


@pytest.mark.unit
def test_bucket_labels_ordered_low_to_high() -> None:
    """The BUCKET_LABELS tuple drives report column order. It must
    walk strictly low → high so the report reads naturally."""
    assert BUCKET_LABELS == (
        BUCKET_BAND_LO, BUCKET_GRACE, BUCKET_BAND_HI,
        BUCKET_RANGE_2, BUCKET_AGGRESSIVE,
    )


# ---------------------------------------------------------------------------
# 2. first_grace_failure_gate — order matches production
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_grace_failure_first_gate_rule_mode() -> None:
    """Even with everything else aligned, mode != observe → first fail."""
    assert first_grace_failure_gate(
        mode="hedgerock", cooldown_active=False, regime="range",
        confidence=0.50, dd_pct_present=True, spread_pts_present=True,
        recent_sample_count=None,
    ) == "rule_mode_not_observe"


@pytest.mark.unit
def test_grace_failure_cooldown_after_observe() -> None:
    assert first_grace_failure_gate(
        mode="observe", cooldown_active=True, regime="range",
        confidence=0.50, dd_pct_present=True, spread_pts_present=True,
        recent_sample_count=None,
    ) == "cooldown_active"


@pytest.mark.unit
def test_grace_failure_regime_check() -> None:
    assert first_grace_failure_gate(
        mode="observe", cooldown_active=False, regime="trend_up",
        confidence=0.50, dd_pct_present=True, spread_pts_present=True,
        recent_sample_count=None,
    ) == "regime_not_range"


@pytest.mark.unit
def test_grace_failure_below_grace_floor() -> None:
    assert first_grace_failure_gate(
        mode="observe", cooldown_active=False, regime="range",
        confidence=0.30, dd_pct_present=True, spread_pts_present=True,
        recent_sample_count=None,
    ) == "confidence_below_grace_floor"


@pytest.mark.unit
def test_grace_failure_above_observe_floor() -> None:
    assert first_grace_failure_gate(
        mode="observe", cooldown_active=False, regime="range",
        confidence=0.65, dd_pct_present=True, spread_pts_present=True,
        recent_sample_count=None,
    ) == "confidence_at_or_above_observe_floor"


@pytest.mark.unit
def test_grace_failure_history_already_warm() -> None:
    assert first_grace_failure_gate(
        mode="observe", cooldown_active=False, regime="range",
        confidence=0.50, dd_pct_present=True, spread_pts_present=True,
        recent_sample_count=20,
    ) == "history_already_warm"


@pytest.mark.unit
def test_grace_failure_all_gates_pass() -> None:
    assert first_grace_failure_gate(
        mode="observe", cooldown_active=False, regime="range",
        confidence=0.50, dd_pct_present=True, spread_pts_present=True,
        recent_sample_count=None,
    ) == ""


# ---------------------------------------------------------------------------
# 3. compute_outcome_labels — strictly future bars only
# ---------------------------------------------------------------------------


def _bars_with_step(start_close: float, step: float, n: int) -> pl.DataFrame:
    """Synthetic H1 sequence where close[i] = start_close + step*i.
    Predictable returns let us assert exact values."""
    rows = []
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    for i in range(n):
        c = start_close + step * i
        rows.append({
            "ts": base + timedelta(hours=i),
            "open": c, "high": c + 0.5, "low": c - 0.5,
            "close": c, "volume": 100.0,
        })
    return pl.DataFrame(rows).with_columns(
        pl.col("ts").dt.replace_time_zone("UTC")
    )


@pytest.mark.unit
def test_outcome_labels_use_only_future_bars() -> None:
    """Bar 10 has close=110. Bar 10+1=11 has close=111. The 1h return
    must be (111-110)/110 = +0.909%. If lookahead leaked, the function
    would use bar 10's own close in some way and the value would
    differ."""
    bars = _bars_with_step(start_close=100.0, step=1.0, n=50)
    out = compute_outcome_labels(decision_idx=10, h1_bars=bars,
                                 horizons=(1, 4, 12, 24))
    assert out is not None
    # close[10] = 110, close[11] = 111 → +0.909...%
    assert out.returns[1] == pytest.approx((111 - 110) / 110 * 100, rel=1e-9)
    # close[14] = 114 → +3.636%
    assert out.returns[4] == pytest.approx((114 - 110) / 110 * 100, rel=1e-9)
    # close[34] = 134 → +21.818%
    assert out.returns[24] == pytest.approx((134 - 110) / 110 * 100, rel=1e-9)
    # decision_close anchored on bar 10
    assert out.decision_close == pytest.approx(110.0)


@pytest.mark.unit
def test_outcome_labels_returns_none_when_window_truncated() -> None:
    """Decision near the end of the lake — not enough future bars
    to fill the 24-bar window. Must return None rather than emit a
    short label."""
    bars = _bars_with_step(start_close=100.0, step=1.0, n=20)
    out = compute_outcome_labels(decision_idx=10, h1_bars=bars,
                                 horizons=(1, 4, 12, 24))
    assert out is None  # only 9 future bars, need 24


@pytest.mark.unit
def test_outcome_labels_mae_mfe_within_window() -> None:
    """Construct a window where future bars walk down then up. MAE
    must be the lowest low AND occur strictly AFTER decision_idx."""
    rows = []
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    # Decision bar at index 0, close=100. Bars 1..12 dip to 95, bars
    # 13..24 climb to 108. Bar 0's low is 90 (well below 95) — if the
    # function used bar 0, MAE would be -10%; with no lookahead, MAE
    # = (95 - 100)/100 = -5%.
    rows.append({
        "ts": base, "open": 100.0, "high": 100.5, "low": 90.0,
        "close": 100.0, "volume": 100.0,
    })
    for i in range(1, 13):
        c = 100.0 - i * 0.4  # dips to 95.2 by bar 12
        rows.append({
            "ts": base + timedelta(hours=i),
            "open": c, "high": c + 0.5, "low": c - 0.5,
            "close": c, "volume": 100.0,
        })
    for i in range(13, 30):
        c = 95.0 + (i - 12) * 1.0
        rows.append({
            "ts": base + timedelta(hours=i),
            "open": c, "high": c + 0.5, "low": c - 0.5,
            "close": c, "volume": 100.0,
        })
    bars = pl.DataFrame(rows).with_columns(
        pl.col("ts").dt.replace_time_zone("UTC"),
    )
    out = compute_outcome_labels(decision_idx=0, h1_bars=bars,
                                 horizons=(1, 4, 12, 24))
    assert out is not None
    # Bar-0's low (90) MUST NOT be the MAE — that would be lookahead.
    # Lowest low in bars 1..24 is around 95.2 - 0.5 = 94.7.
    assert out.mae_24h > -8.0  # << -10 (which is what bar 0's 90 would give)
    assert out.mfe_24h > 0.0  # bars 13..24 climb above 100


# ---------------------------------------------------------------------------
# 4. Waterfall monotonicity
# ---------------------------------------------------------------------------


def _record(grace_failed_at: str, *, ts_offset_h: int = 0) -> DecisionRecord:
    """Minimal DecisionRecord fixture for waterfall tests."""
    return DecisionRecord(
        ts=datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(hours=ts_offset_h),
        regime="range", confidence=0.50,
        confidence_bucket=BUCKET_GRACE,
        classifier_reason="fixture",
        rule_votes=(),
        volatility_rank=0.5, h4_trend_bars=1, hh_count=3, ll_count=3,
        rule_mode="observe", rule_risk_tier="observe",
        rule_reason="fixture", rule_lot_factor=0.0,
        rule_cooldown_active=False,
        grace_failed_at=grace_failed_at,  # type: ignore[arg-type]
        grace_eligible=(grace_failed_at == ""),
    )


@pytest.mark.unit
def test_waterfall_monotone_non_increasing() -> None:
    """Each subsequent waterfall stage must have ≤ survivors of prev.
    No gate can mint new bars."""
    recs = [
        _record("rule_mode_not_observe", ts_offset_h=0),
        _record("cooldown_active", ts_offset_h=1),
        _record("regime_not_range", ts_offset_h=2),
        _record("confidence_below_grace_floor", ts_offset_h=3),
        _record("", ts_offset_h=4),  # all-pass survivor
    ]
    wf = aggregate_grace_waterfall(recs)
    survivors = [s.survivors for s in wf.stages]
    for a, b in zip(survivors, survivors[1:]):
        assert b <= a, f"non-monotone: {survivors}"
    # Total = 5; only 1 bar passes everything.
    assert survivors[0] == 5
    assert survivors[-1] == 1


@pytest.mark.unit
def test_waterfall_all_pass_holds_count_through_all_gates() -> None:
    """If every record passes, survivor count is constant across all
    gates."""
    recs = [_record("", ts_offset_h=i) for i in range(7)]
    wf = aggregate_grace_waterfall(recs)
    assert all(s.survivors == 7 for s in wf.stages)


@pytest.mark.unit
def test_waterfall_with_failure_at_each_gate() -> None:
    """One record failing at each gate — waterfall drops by exactly
    1 at each stage."""
    recs = [
        _record("rule_mode_not_observe"),
        _record("cooldown_active"),
        _record("regime_not_range"),
        _record("confidence_below_grace_floor"),
        _record("confidence_at_or_above_observe_floor"),
        _record("risk_snapshot_incomplete"),
        _record("history_already_warm"),
        _record(""),
    ]
    wf = aggregate_grace_waterfall(recs)
    survivors = [s.survivors for s in wf.stages]
    # total = 8, drops by 1 each gate, ending at 1 (all-pass).
    assert survivors == [8, 7, 6, 5, 4, 3, 2, 1]


# ---------------------------------------------------------------------------
# 5. Aggregation determinism
# ---------------------------------------------------------------------------


def _label(*, ret_24: float, mae: float = -1.0, mfe: float = 1.5,
           range_w: float = 2.0) -> OutcomeLabel:
    return OutcomeLabel(
        decision_close=100.0,
        returns={1: 0.1, 4: 0.2, 12: 0.5, 24: ret_24},
        mae_24h=mae, mfe_24h=mfe, range_width_24h=range_w,
    )


def _dec(*, regime: str, bucket: str, idx: int) -> DecisionRecord:
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return DecisionRecord(
        ts=base + timedelta(hours=idx),
        regime=regime, confidence=0.50, confidence_bucket=bucket,
        classifier_reason="", rule_votes=(),
        volatility_rank=0.5, h4_trend_bars=1, hh_count=3, ll_count=3,
        rule_mode="observe", rule_risk_tier="observe",
        rule_reason="", rule_lot_factor=0.0, rule_cooldown_active=False,
        grace_failed_at="", grace_eligible=True,
    )


@pytest.mark.unit
def test_regime_x_bucket_count_deterministic() -> None:
    recs = [
        _dec(regime="range", bucket="0.65-0.80", idx=0),
        _dec(regime="range", bucket="0.65-0.80", idx=1),
        _dec(regime="trend_up", bucket=">=0.80", idx=2),
    ]
    a = aggregate_regime_x_confidence(recs)
    b = aggregate_regime_x_confidence(recs)
    assert a == b
    assert a[("range", "0.65-0.80")] == 2
    assert a[("trend_up", ">=0.80")] == 1


@pytest.mark.unit
def test_range_opportunity_aggregation_deterministic() -> None:
    pairs = [
        (_dec(regime="range", bucket="0.65-0.80", idx=0), _label(ret_24=0.5)),
        (_dec(regime="range", bucket="0.65-0.80", idx=1), _label(ret_24=-0.3)),
        (_dec(regime="range", bucket="0.65-0.80", idx=2), _label(ret_24=0.1)),
        # Different regime — should not contribute.
        (_dec(regime="trend_up", bucket="0.65-0.80", idx=3), _label(ret_24=99)),
    ]
    a = aggregate_range_opportunity(pairs)
    b = aggregate_range_opportunity(pairs)
    assert a == b
    bucket = next(s for s in a if s.bucket == "0.65-0.80")
    assert bucket.count == 3
    assert bucket.mean_return_24h == pytest.approx((0.5 - 0.3 + 0.1) / 3)


@pytest.mark.unit
def test_trend_opportunity_resigns_trend_down() -> None:
    """trend_down with future +0.5% return should score as -0.5%
    directional (because price went UP when down was expected)."""
    pairs = [
        (_dec(regime="trend_down", bucket="all", idx=0), _label(ret_24=0.5)),
        (_dec(regime="trend_down", bucket="all", idx=1), _label(ret_24=-0.4)),
    ]
    out = aggregate_trend_opportunity(pairs)
    td = next(s for s in out if s.regime == "trend_down")
    assert td.count == 2
    # Re-signed: -0.5, +0.4 → mean = -0.05
    assert td.mean_return_24h == pytest.approx(-0.05)


@pytest.mark.unit
def test_trend_opportunity_uses_abs_for_breakout_magnitude() -> None:
    """Phase D-cont2-hotfix-2: the legacy single `breakout` row was
    replaced by `breakout_magnitude` (|return|) plus
    `breakout_signed_by_h4`. The magnitude row keeps the |·| logic
    but is labelled so the verdict logic refuses to E1-promote it."""
    pairs = [
        (_dec(regime="breakout", bucket="all", idx=0), _label(ret_24=-1.0)),
        (_dec(regime="breakout", bucket="all", idx=1), _label(ret_24=+0.5)),
    ]
    out = aggregate_trend_opportunity(pairs)
    bk = next(s for s in out if s.regime == "breakout_magnitude")
    assert bk.count == 2
    assert bk.mean_return_24h == pytest.approx((1.0 + 0.5) / 2)
    # The unsplit "breakout" regime no longer exists.
    assert not any(s.regime == "breakout" for s in out)


# ---------------------------------------------------------------------------
# 6. End-to-end: report contains required sections on synthetic lake
# ---------------------------------------------------------------------------


def _ohlcv(*, start, n_bars, bar_minutes):
    rows = []
    price = 2000.0
    for i in range(n_bars):
        ts = start + timedelta(minutes=bar_minutes * i)
        delta = ((i % 50) - 25) * 0.2
        price = price + delta
        rows.append({
            "ts": ts, "open": price - delta / 2,
            "high": price + 5, "low": price - 5,
            "close": price, "volume": 100.0,
        })
    return pl.DataFrame(rows).with_columns(pl.col("ts").dt.replace_time_zone("UTC"))


class _FakeLake:
    def __init__(self, h1, h4, d1):
        self._h1, self._h4, self._d1 = h1, h4, d1

    def query(self, instrument, timeframe, start, end):
        df = {"H1": self._h1, "H4": self._h4, "D1": self._d1}.get(str(timeframe))
        if df is None or df.is_empty():
            return pl.DataFrame()
        return df.filter((pl.col("ts") >= start) & (pl.col("ts") < end))


@pytest.fixture
def small_lake():
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return _FakeLake(
        h1=_ohlcv(start=start, n_bars=24 * 30, bar_minutes=60),
        h4=_ohlcv(start=start, n_bars=6 * 30, bar_minutes=240),
        d1=_ohlcv(start=start, n_bars=30, bar_minutes=1440),
    )


@pytest.mark.unit
def test_run_atlas_end_to_end_produces_records(small_lake) -> None:
    cfg = AtlasConfig(
        instrument="XAUUSD",
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 1, 31, tzinfo=timezone.utc),
        h1_lookback=120, h4_lookback=20,
        # Very low sample floor so the synthetic data still emits
        # tradeable / inconclusive verdicts.
        min_sample_for_signal=5,
    )
    report = run_atlas(cfg, small_lake)
    assert isinstance(report, AtlasReport)
    assert len(report.records) > 0
    # Waterfall total must equal record count.
    assert report.waterfall.stages[0].survivors == len(report.records)
    # Aggregates have all 5 bucket entries (even if zero-count).
    assert {s.bucket for s in report.range_opportunity} == set(BUCKET_LABELS)


@pytest.mark.unit
def test_report_writer_emits_required_sections(tmp_path, small_lake) -> None:
    """End-to-end: invoke the script's main() and assert that all
    required section headers appear in the rendered report."""
    import sys
    scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
    sys.path.insert(0, str(scripts_dir))
    try:
        from hedgerock_regime_opportunity_atlas import main
    finally:
        sys.path.pop(0)

    # The script's main() loads from a real lake root via ForexDataLake,
    # so we can't pass our fake lake through. Instead, exercise the
    # writer functions directly with an in-memory atlas report.
    cfg = AtlasConfig(
        instrument="XAUUSD",
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 1, 31, tzinfo=timezone.utc),
        h1_lookback=120, h4_lookback=20,
        min_sample_for_signal=5,
    )
    report = run_atlas(cfg, small_lake)

    sys.path.insert(0, str(scripts_dir))
    try:
        from hedgerock_regime_opportunity_atlas import _write_report
    finally:
        sys.path.pop(0)

    report_path = tmp_path / "atlas.md"
    records_path = tmp_path / "atlas.records.jsonl"
    _write_report(report, report_path, {"decisions": records_path})

    body = report_path.read_text()
    required_sections = [
        "# Phase D-cont2 — Regime / Opportunity Atlas",
        "## Run config",
        "## Regime × confidence-bucket distribution",
        "## Cold-start-grace eligibility waterfall",
        "## Range opportunity atlas",
        "## Trend / breakout missed-opportunity atlas",
        "## Halt aftermath atlas",
        "## D-cont2 verdict",
        "## Caveats — what this atlas does NOT prove",
    ]
    for section in required_sections:
        assert section in body, (
            f"missing required section: {section!r}\n--- got ---\n"
            + body[:1000]
        )


@pytest.mark.unit
def test_jsonl_records_label_keys_are_explicitly_prefixed(
    tmp_path, small_lake,
) -> None:
    """The user-spec calls out future bars as LABELS only. The JSONL
    writer prefixes every label key with `label_` so a downstream
    consumer can never confuse a decision input with an outcome."""
    import sys
    scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
    sys.path.insert(0, str(scripts_dir))
    try:
        from hedgerock_regime_opportunity_atlas import _record_to_dict, _write_jsonl
    finally:
        sys.path.pop(0)

    cfg = AtlasConfig(
        instrument="XAUUSD",
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 1, 31, tzinfo=timezone.utc),
        h1_lookback=120, h4_lookback=20, min_sample_for_signal=5,
    )
    report = run_atlas(cfg, small_lake)
    rows = [_record_to_dict(d, o) for d, o in report.records]
    # Must have at least one row with full outcome labels.
    labelled = [r for r in rows if r.get("outcome_available")]
    assert len(labelled) > 0
    # All future-data fields use a `label_` prefix.
    label_keys = {
        "label_decision_close", "label_returns_pct",
        "label_mae_24h_pct", "label_mfe_24h_pct",
        "label_range_width_24h_pct",
    }
    for r in labelled:
        present = label_keys & set(r.keys())
        assert present == label_keys, f"missing label keys in row: {r.keys()}"
        # Decision-side keys must NOT be prefixed with `label_`.
        decision_keys = {
            "ts", "regime", "confidence", "rule_mode", "rule_reason",
            "grace_failed_at",
        }
        assert decision_keys.issubset(r.keys())


# ---------------------------------------------------------------------------
# 7. find_halt_aftermath — leading-edge detection
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_find_halt_aftermath_only_leading_edge() -> None:
    """A continuous halt streak counts as ONE event, not many. The
    function must detect transitions from non-halt → halt only."""
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    # 60 bars: first 5 hedgerock, next 50 halt, last 5 observe.
    rows = []
    for i in range(60):
        rows.append({
            "ts": base + timedelta(hours=i),
            "open": 2000.0, "high": 2005.0, "low": 1995.0,
            "close": 2000.0, "volume": 100.0,
        })
    h1 = pl.DataFrame(rows).with_columns(
        pl.col("ts").dt.replace_time_zone("UTC"),
    )
    log: list[dict] = []
    for i in range(60):
        ts = (base + timedelta(hours=i)).isoformat()
        if i < 5:
            mode = "hedgerock"
        elif i < 55:
            mode = "halt"
        else:
            mode = "observe"
        log.append({"ts": ts, "mode": mode, "effective_mode": mode})

    out = find_halt_aftermath(h1_bars=h1, envelope_log=log,
                              horizons=(4, 12, 24))
    # Exactly one leading-edge transition (bar 5).
    assert len(out) == 1
    assert out[0].halt_ts == base + timedelta(hours=5)


@pytest.mark.unit
def test_find_halt_aftermath_skips_event_with_truncated_window() -> None:
    """Halt event near the end of the lake → not enough future bars
    to fill all horizons → event dropped silently."""
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    rows = [
        {
            "ts": base + timedelta(hours=i),
            "open": 2000.0, "high": 2005.0, "low": 1995.0,
            "close": 2000.0, "volume": 100.0,
        }
        for i in range(30)
    ]
    h1 = pl.DataFrame(rows).with_columns(
        pl.col("ts").dt.replace_time_zone("UTC"),
    )
    log: list[dict] = []
    for i in range(30):
        log.append({
            "ts": (base + timedelta(hours=i)).isoformat(),
            "mode": "halt" if i == 25 else "hedgerock",
            "effective_mode": "halt" if i == 25 else "hedgerock",
        })
    out = find_halt_aftermath(h1_bars=h1, envelope_log=log, horizons=(72,))
    # halt at bar 25, only 4 future bars → can't fill 72-bar horizon.
    assert out == []


# ---------------------------------------------------------------------------
# 8. NEUTRAL-COLD ea state has the right shape
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_neutral_cold_ea_state_signals_cold_start() -> None:
    ea = build_neutral_cold_ea_state(init_equity=10_000.0, spread_pts=20)
    # All risk-snapshot fields populated (gate 5 passes).
    assert ea.dd_pct == 0.0
    assert ea.spread_pts == 20
    # All history fields None (gate 6 — history cold — passes).
    assert ea.consec_losses is None
    assert ea.recent_closed_pnl is None
    assert ea.recent_sample_count is None


# ===========================================================================
# Phase D-cont2-hotfix-1 — live-equivalent envelope merge
# ===========================================================================


def _bare_record(*, ts_offset_h: int = 0) -> DecisionRecord:
    """Minimal NEUTRAL-COLD DecisionRecord; live_* defaults to None/False."""
    return DecisionRecord(
        ts=datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(hours=ts_offset_h),
        regime="range", confidence=0.50, confidence_bucket=BUCKET_GRACE,
        classifier_reason="fixture",
        rule_votes=(),
        volatility_rank=0.5, h4_trend_bars=1, hh_count=3, ll_count=3,
        rule_mode="observe", rule_risk_tier="observe",
        rule_reason="fixture", rule_lot_factor=0.0,
        rule_cooldown_active=False,
        grace_failed_at="", grace_eligible=True,
    )


@pytest.mark.unit
def test_decision_record_live_fields_default_none_false() -> None:
    """A fresh NEUTRAL-COLD record has live_* defaults: None for value
    fields, False for booleans. Reports must not interpret defaults
    as live data."""
    r = _bare_record()
    assert r.live_raw_mode is None
    assert r.live_raw_reason is None
    assert r.live_effective_mode is None
    assert r.live_risk_tier is None
    assert r.live_lot_factor is None
    assert r.live_cooldown_active is False
    assert r.live_cooldown_until is None
    assert r.live_transition_lock_active is False
    assert r.live_transition_lock_until_ts is None


@pytest.mark.unit
def test_merge_live_envelope_entry_none_returns_record_unchanged() -> None:
    """When the dynamic loop has no envelope entry for this ts (e.g.
    lookback warmup), the merge is a no-op."""
    r = _bare_record()
    out = merge_live_envelope_entry(r, None)
    assert out == r


@pytest.mark.unit
def test_merge_live_envelope_entry_transition_lock_demotes_hedgerock() -> None:
    """The CORE invariant: a bar where the dynamic simulator's
    rule_engine returned hedgerock but transition_lock was active
    must be recorded as live_raw_mode=hedgerock AND
    live_effective_mode=observe — distinguishable from each other and
    not collapsed."""
    r = _bare_record()
    lock_until = datetime(2024, 1, 1, 13, tzinfo=timezone.utc)
    cd_until = datetime(2024, 1, 1, 12, 30, tzinfo=timezone.utc)
    entry = {
        "ts": r.ts.isoformat(),
        "mode": "hedgerock",
        "reason": "range + normal",
        "effective_mode": "observe",
        "risk_tier": "normal",
        "lot_factor": 1.0,
        "cooldown_until": cd_until.isoformat(),
        "transition_lock_active": True,
        "transition_lock_until_ts": lock_until.isoformat(),
    }
    out = merge_live_envelope_entry(r, entry)
    # Two distinct fields — they MUST disagree under transition lock.
    assert out.live_raw_mode == "hedgerock"
    assert out.live_effective_mode == "observe"
    assert out.live_raw_mode != out.live_effective_mode

    # Cooldown / transition-lock parsed back to datetimes.
    assert out.live_cooldown_active is True
    assert out.live_cooldown_until == cd_until
    assert out.live_transition_lock_active is True
    assert out.live_transition_lock_until_ts == lock_until

    # NEUTRAL-COLD raw-intent fields preserved unchanged.
    assert out.rule_mode == "observe"
    assert out.rule_risk_tier == "observe"


@pytest.mark.unit
def test_merge_live_envelope_handles_missing_optional_fields() -> None:
    """Older envelope-log entries (or ones with cleared cooldown) may
    have None for cooldown/transition_lock — merge tolerates."""
    r = _bare_record()
    entry = {
        "ts": r.ts.isoformat(),
        "mode": "observe",
        "reason": "history_incomplete",
        "effective_mode": "observe",
        "risk_tier": "observe",
        "lot_factor": 0.0,
        "cooldown_until": None,
        "transition_lock_active": False,
        "transition_lock_until_ts": None,
    }
    out = merge_live_envelope_entry(r, entry)
    assert out.live_cooldown_active is False
    assert out.live_cooldown_until is None
    assert out.live_transition_lock_active is False
    assert out.live_transition_lock_until_ts is None
    assert out.live_effective_mode == "observe"


@pytest.mark.unit
def test_jsonl_record_includes_live_equivalent_fields(tmp_path) -> None:
    """The JSONL writer must surface every live_* field so a downstream
    consumer can compute their own raw-vs-effective deltas."""
    import sys
    scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
    sys.path.insert(0, str(scripts_dir))
    try:
        from hedgerock_regime_opportunity_atlas import _record_to_dict
    finally:
        sys.path.pop(0)

    r = _bare_record()
    lock_until = datetime(2024, 1, 1, 13, tzinfo=timezone.utc)
    entry = {
        "ts": r.ts.isoformat(),
        "mode": "hedgerock", "reason": "range + normal",
        "effective_mode": "observe", "risk_tier": "normal",
        "lot_factor": 1.0,
        "cooldown_until": None,
        "transition_lock_active": True,
        "transition_lock_until_ts": lock_until.isoformat(),
    }
    merged = merge_live_envelope_entry(r, entry)
    row = _record_to_dict(merged, None)
    required = {
        "live_raw_mode", "live_raw_reason", "live_effective_mode",
        "live_risk_tier", "live_lot_factor",
        "live_cooldown_active", "live_cooldown_until",
        "live_transition_lock_active", "live_transition_lock_until_ts",
    }
    assert required.issubset(row.keys())
    assert row["live_raw_mode"] == "hedgerock"
    assert row["live_effective_mode"] == "observe"
    assert row["live_transition_lock_active"] is True
    assert row["live_transition_lock_until_ts"] == lock_until.isoformat()


@pytest.mark.unit
def test_report_distinguishes_raw_intent_from_live_effective(
    tmp_path, small_lake,
) -> None:
    """The report MUST emit a side-by-side raw-intent vs live-effective
    section AND label the waterfall as NEUTRAL-COLD raw intent so a
    reader cannot misread raw intent as live execution."""
    cfg = AtlasConfig(
        instrument="XAUUSD",
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 1, 31, tzinfo=timezone.utc),
        h1_lookback=120, h4_lookback=20, min_sample_for_signal=5,
    )
    report = run_atlas(cfg, small_lake)
    import sys
    scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
    sys.path.insert(0, str(scripts_dir))
    try:
        from hedgerock_regime_opportunity_atlas import _write_report
    finally:
        sys.path.pop(0)

    report_path = tmp_path / "atlas.md"
    _write_report(report, report_path, {"decisions": tmp_path / "rec.jsonl"})
    body = report_path.read_text()
    # New raw-vs-live section must be present and distinguish the views.
    assert "Raw classifier intent vs live-effective execution" in body
    assert "NEUTRAL-COLD rule intent" in body
    assert "Phase D live raw" in body
    assert "Phase D live effective" in body
    # Waterfall heading must be tagged so it's not confused with live.
    assert "NEUTRAL-COLD raw intent" in body  # appears in waterfall heading too


# ===========================================================================
# Phase D-cont2-hotfix-2 — breakout split: magnitude (never E1) +
# signed-by-h4 (E1 only with signed CI clearing zero)
# ===========================================================================


def _dec_breakout(*, h4_trend_bars: int, idx: int) -> DecisionRecord:
    """Breakout fixture with controllable h4_trend_bars sign."""
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return DecisionRecord(
        ts=base + timedelta(hours=idx),
        regime="breakout", confidence=0.65, confidence_bucket=BUCKET_RANGE_2,
        classifier_reason="", rule_votes=(),
        volatility_rank=0.5, h4_trend_bars=h4_trend_bars,
        hh_count=3, ll_count=3,
        rule_mode="observe", rule_risk_tier="observe",
        rule_reason="", rule_lot_factor=0.0, rule_cooldown_active=False,
        grace_failed_at="", grace_eligible=True,
    )


@pytest.mark.unit
def test_breakout_split_emits_two_distinct_rows() -> None:
    """aggregate_trend_opportunity now emits both
    breakout_magnitude (|return|) AND breakout_signed_by_h4 (signed)
    instead of a single ambiguous breakout row."""
    pairs = [
        (_dec_breakout(h4_trend_bars=+1, idx=0), _label(ret_24=+1.0)),
        (_dec_breakout(h4_trend_bars=-1, idx=1), _label(ret_24=-0.5)),
    ]
    out = aggregate_trend_opportunity(pairs)
    regimes = {s.regime for s in out}
    assert "breakout_magnitude" in regimes
    assert "breakout_signed_by_h4" in regimes
    # The unsplit ambiguous "breakout" label is gone.
    assert "breakout" not in regimes


@pytest.mark.unit
def test_breakout_magnitude_uses_abs_return() -> None:
    """breakout_magnitude is |return| — both +0.5 and -0.5 contribute
    +0.5 to the mean."""
    pairs = [
        (_dec_breakout(h4_trend_bars=0, idx=0), _label(ret_24=+0.5)),
        (_dec_breakout(h4_trend_bars=0, idx=1), _label(ret_24=-0.5)),
    ]
    out = aggregate_trend_opportunity(pairs)
    mag = next(s for s in out if s.regime == "breakout_magnitude")
    assert mag.count == 2
    assert mag.mean_return_24h == pytest.approx(0.5)


@pytest.mark.unit
def test_breakout_signed_by_h4_resigns_negative_trend() -> None:
    """breakout_signed_by_h4: H4-up + price-up = +ret; H4-down +
    price-down = +ret too (matched expectation). H4-zero bars are
    DROPPED from this row."""
    pairs = [
        # H4 up, price +0.5 → matches → +0.5
        (_dec_breakout(h4_trend_bars=+2, idx=0), _label(ret_24=+0.5)),
        # H4 down, price -0.4 → matches → +0.4 after re-sign
        (_dec_breakout(h4_trend_bars=-3, idx=1), _label(ret_24=-0.4)),
        # H4 zero → DROPPED (proxy ambiguous)
        (_dec_breakout(h4_trend_bars=0, idx=2), _label(ret_24=+99.0)),
    ]
    out = aggregate_trend_opportunity(pairs)
    sig = next(s for s in out if s.regime == "breakout_signed_by_h4")
    assert sig.count == 2  # H4=0 dropped
    assert sig.mean_return_24h == pytest.approx((0.5 + 0.4) / 2)


@pytest.mark.unit
def test_breakout_magnitude_never_qualifies_as_e1_in_verdict() -> None:
    """The verdict logic in the script MUST NOT promote
    breakout_magnitude to E1, even when |return| has a positive mean
    that would clear the 95% CI of zero."""
    import sys
    scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
    sys.path.insert(0, str(scripts_dir))
    try:
        from hedgerock_regime_opportunity_atlas import (
            _fmt_dcont2_verdict, _fmt_trend_opportunity,
        )
    finally:
        sys.path.pop(0)

    # Fabricate a stat row where breakout_magnitude has 100 samples
    # with a clearly significant +1.0% mean and tiny stdev — by raw
    # CI math this would otherwise pass the E1 bar.
    big_n = [_label(ret_24=+1.0) for _ in range(100)]
    pairs_mag = [
        (_dec_breakout(h4_trend_bars=0, idx=i), big_n[i]) for i in range(100)
    ]
    # Force only breakout_magnitude to be high-conviction (signed gets
    # zero samples because h4_trend_bars=0 drops them).
    out = aggregate_trend_opportunity(pairs_mag)
    mag = next(s for s in out if s.regime == "breakout_magnitude")
    sig = next(s for s in out if s.regime == "breakout_signed_by_h4")
    assert mag.count == 100
    assert mag.mean_return_24h == pytest.approx(1.0)
    assert sig.count == 0

    # Build a minimal AtlasReport and run the verdict block. We need
    # a real config; reuse defaults.
    cfg = AtlasConfig(min_sample_for_signal=5)
    report = AtlasReport(
        config=cfg,
        records=[],
        waterfall=GraceWaterfall(stages=()),
        regime_x_bucket_count={},
        range_opportunity=[],
        trend_opportunity=out,
        halt_aftermath=[],
    )
    verdict_text = "\n".join(_fmt_dcont2_verdict(report))
    trend_text = "\n".join(_fmt_trend_opportunity(report))

    # Magnitude is mentioned, but explicitly DISQUALIFIED.
    assert "breakout_magnitude" in trend_text
    assert "Magnitude only" in trend_text or "magnitude-only" in verdict_text.lower()
    assert "E1 momentum candidate" not in trend_text or (
        # If "candidate" appears in the trend text it must NOT be tied
        # to breakout_magnitude — search for the rejection sentence.
        ("Magnitude only" in trend_text)
        or ("cannot be an E1" in trend_text)
    )
    # The net verdict MUST NOT list breakout_magnitude under E1
    # candidates.
    assert "WINDOW-EVIDENCED for: `breakout_magnitude`" not in verdict_text
    assert "breakout_magnitude," not in verdict_text.split("WINDOW-EVIDENCED for: ")[-1] if "WINDOW-EVIDENCED for: " in verdict_text else True


@pytest.mark.unit
def test_caveats_describe_neutral_cold_and_live_split(
    tmp_path, small_lake,
) -> None:
    """Phase D-cont2-closeout: the post-hotfix caveat MUST acknowledge
    that live-equivalent fields enter via the raw-vs-live-effective
    section AND halt aftermath, not "only" via halt aftermath. This
    test fails if anyone reverts the wording."""
    cfg = AtlasConfig(
        instrument="XAUUSD",
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 1, 31, tzinfo=timezone.utc),
        h1_lookback=120, h4_lookback=20, min_sample_for_signal=5,
    )
    report = run_atlas(cfg, small_lake)
    import sys
    scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
    sys.path.insert(0, str(scripts_dir))
    try:
        from hedgerock_regime_opportunity_atlas import _write_report
    finally:
        sys.path.pop(0)
    report_path = tmp_path / "atlas.md"
    _write_report(report, report_path, {"decisions": tmp_path / "x.jsonl"})
    body = report_path.read_text()
    # The "only via halt aftermath" wording is the bug the closeout fixes.
    assert "only enters this report via the halt aftermath" not in body
    # The new wording must mention BOTH locations explicitly.
    assert "Raw classifier intent vs live-effective execution" in body
    assert "Halt aftermath atlas" in body
    # NEUTRAL-COLD scope is confined to raw intent + grace waterfall.
    assert (
        "NEUTRAL-COLD" in body
        and "raw classifier intent" in body.lower()
    )


@pytest.mark.unit
def test_breakout_signed_by_h4_can_qualify_as_e1_when_signed_ci_clears() -> None:
    """Counterpart: a SIGNED-direction breakout row IS allowed to be
    an E1 candidate when its signed CI clears zero."""
    import sys
    scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
    sys.path.insert(0, str(scripts_dir))
    try:
        from hedgerock_regime_opportunity_atlas import _fmt_trend_opportunity
    finally:
        sys.path.pop(0)

    # 100 records with H4 trend up + consistent +1.0% returns →
    # signed mean +1.0%, tiny CI, MFE > |MAE| → E1 candidate.
    pairs = [
        (
            _dec_breakout(h4_trend_bars=+2, idx=i),
            OutcomeLabel(
                decision_close=100.0,
                returns={1: 0.1, 4: 0.2, 12: 0.5, 24: 1.0},
                mae_24h=-0.2, mfe_24h=1.5, range_width_24h=1.7,
            ),
        )
        for i in range(100)
    ]
    out = aggregate_trend_opportunity(pairs)
    cfg = AtlasConfig(min_sample_for_signal=10)
    report = AtlasReport(
        config=cfg, records=[],
        waterfall=GraceWaterfall(stages=()),
        regime_x_bucket_count={},
        range_opportunity=[], trend_opportunity=out, halt_aftermath=[],
    )
    text = "\n".join(_fmt_trend_opportunity(report))
    # signed_by_h4 row gets the E1-candidate prose.
    assert "breakout_signed_by_h4" in text
    # The candidate prose specifically appears for signed_by_h4 (or at
    # least once on a non-magnitude row).
    assert "E1 momentum candidate" in text
