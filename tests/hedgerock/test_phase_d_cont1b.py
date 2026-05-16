"""Phase D-cont1b — tiny-normal halt-expiry release tests.

Adds a SECOND halt-auto-expiry release path:

    halt_auto_expiry_release = "observe"      ← D-cont1 (default, conservative)
    halt_auto_expiry_release = "tiny_normal"  ← D-cont1b (small new hedge)

Pinned guarantees:
    1. Baseline (default config, no experiment) is byte-identical —
       D-cont1b adds NO behaviour at default.
    2. D-cont1 observe-only release is preserved unchanged — explicitly
       passing release="observe" still produces lot_factor=0,
       hedgerock_enabled=False, and an "halt_auto_expiry" reason that
       does NOT contain "tiny_normal".
    3. D-cont1b tiny_normal release after expiry produces:
         mode="hedgerock", hedgerock_enabled=True, risk_tier="observe",
         lot_factor=0.1, max_next_lot ≤ 0.02, max_orders_buy=max_orders_sell=1,
         cooldown_until=None, reason containing "halt_auto_expiry_tiny_normal".
    4. D-cont1b tiny_normal NEVER produces aggressive — the override
       hard-codes risk_tier="observe", regardless of the inbound halt
       params or market state.
    5. CLI red-flag detection in scripts/run_phase_d_walk_forward.py
       triggers when:
         (a) variant return improves AND DD > baseline + 2pp, OR
         (b) variant near-stopout > baseline (any increase, no tolerance), OR
         (c) variant blew up while baseline survived.
    6. run_walk_forward 3-way: experiments list with two configs yields
       two ExperimentResult entries, baseline.dynamic_metrics is
       byte-identical to the no-experiments call.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from smc.hedgerock.decision_server import MarketFeatures
from smc.hedgerock.ea_state import build_ea_state
from smc.hedgerock.market_state import aggregate_market_state
from smc.hedgerock.phase_d_walk_forward import (
    ExperimentConfig,
    ExperimentResult,
    TradeMetrics,
    WalkForwardConfig,
    apply_experiment_overrides,
    run_walk_forward,
)
from smc.hedgerock.regime_classifier_v2 import RegimeAssessmentV2
from smc.hedgerock.rule_engine import DynamicParams


# ---------------------------------------------------------------------------
# Shared fixtures (mirror test_phase_d_cont1.py)
# ---------------------------------------------------------------------------


def _now() -> datetime:
    return datetime(2024, 6, 15, 12, 0, tzinfo=timezone.utc)


def _state(*, regime_v2: str, confidence: float, ea_state, now: datetime):
    assessment = RegimeAssessmentV2(
        regime=regime_v2,  # type: ignore[arg-type]
        confidence=confidence,
        reason="fixture",
        rule_votes=(),
    )
    return aggregate_market_state(
        symbol="XAUUSD", now=now,
        features=MarketFeatures(
            volatility_rank=0.45, hh_count=3, ll_count=3,
            h4_trend_bars=1, regime="CONSOLIDATION",
        ),
        regime_assessment=assessment,
        ea_state=ea_state,
        ea_state_recorded_at=now - timedelta(seconds=5),
    )


def _halt_params() -> DynamicParams:
    """A rule_engine output that landed in halt under the DD-severe branch."""
    return DynamicParams(
        mode="halt",
        hedgerock_enabled=False,
        risk_tier="observe",
        cooldown_until=_now() + timedelta(minutes=30),
        lot_factor=0.0,
        grid_multiplier=1.0,
        max_next_lot=0.05,
        takeprofit_points=600,
        stoploss_points=3900,
        recovery_multiplier=1.2,
        max_orders_buy=0,
        max_orders_sell=0,
        reason="DD=0.0600 ≥ 0.05 → halt + cooldown",
    )


# ---------------------------------------------------------------------------
# 1. Baseline byte-identical — D-cont1b adds NO default behaviour
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_default_release_value_is_observe() -> None:
    """ExperimentConfig() with no kwargs must default to release='observe'."""
    cfg = ExperimentConfig()
    assert cfg.halt_auto_expiry_release == "observe"
    assert cfg.is_baseline is True


@pytest.mark.unit
def test_baseline_with_explicit_observe_release_is_still_baseline() -> None:
    """Explicitly setting release='observe' (matching default) doesn't make
    it a non-baseline. is_baseline ignores release; what matters is whether
    the gating knobs (cold_start_grace / halt_auto_expiry_hours) are set."""
    cfg = ExperimentConfig(halt_auto_expiry_release="observe")
    assert cfg.is_baseline is True
    cfg2 = ExperimentConfig(halt_auto_expiry_release="tiny_normal")
    assert cfg2.is_baseline is True  # release alone doesn't enable anything


@pytest.mark.unit
def test_baseline_experiment_no_op_with_release_set() -> None:
    """Setting release without halt_auto_expiry_hours = no-op (no halt
    mechanism active)."""
    now = _now()
    ea = build_ea_state(equity=9400.0, balance=10000.0, dd_pct=0.06, spread_pts=20)
    state = _state(regime_v2="range", confidence=0.85, ea_state=ea, now=now)
    p_in = _halt_params()
    p_out, streak = apply_experiment_overrides(
        p_in, market_state=state, halt_streak_started_ts=None,
        now=now,
        experiment=ExperimentConfig(halt_auto_expiry_release="tiny_normal"),
    )
    # is_baseline → True since halt_auto_expiry_hours is None → returned untouched
    assert p_out == p_in
    assert streak is None


# ---------------------------------------------------------------------------
# 2. D-cont1 observe-only release is preserved unchanged
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_observe_release_after_expiry_unchanged_from_dcont1() -> None:
    """halt_auto_expiry_release='observe' after N hours → mode=observe,
    lot_factor=0, hedgerock_enabled=False (D-cont1 behaviour)."""
    now = _now()
    streak_start = now - timedelta(hours=4, minutes=5)  # past 4h threshold
    ea = build_ea_state(equity=9400.0, balance=10000.0, dd_pct=0.06, spread_pts=20)
    state = _state(regime_v2="range", confidence=0.85, ea_state=ea, now=now)
    p_in = _halt_params()
    exp = ExperimentConfig(
        halt_auto_expiry_hours=4.0,
        halt_auto_expiry_release="observe",
    )
    p_out, streak = apply_experiment_overrides(
        p_in, market_state=state,
        halt_streak_started_ts=streak_start,
        now=now, experiment=exp,
    )
    assert p_out.mode == "observe"
    assert p_out.hedgerock_enabled is False
    assert p_out.lot_factor == 0.0
    assert p_out.cooldown_until is None
    # Reason contains plain "halt_auto_expiry" but NOT the tiny_normal suffix.
    assert "halt_auto_expiry" in p_out.reason
    assert "tiny_normal" not in p_out.reason
    # Streak persists for next bar.
    assert streak == streak_start


# ---------------------------------------------------------------------------
# 3. D-cont1b tiny_normal release — opens new tiny positions after expiry
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_tiny_normal_release_after_expiry_produces_tiny_hedgerock() -> None:
    """halt_auto_expiry_release='tiny_normal' after N hours → opens
    a tiny hedge: mode=hedgerock, lot_factor=0.1, max_next_lot ≤ 0.02,
    max_orders_buy=max_orders_sell=1, cooldown_until=None."""
    now = _now()
    streak_start = now - timedelta(hours=4, minutes=5)
    ea = build_ea_state(equity=9400.0, balance=10000.0, dd_pct=0.06, spread_pts=20)
    state = _state(regime_v2="range", confidence=0.85, ea_state=ea, now=now)
    p_in = _halt_params()
    exp = ExperimentConfig(
        halt_auto_expiry_hours=4.0,
        halt_auto_expiry_release="tiny_normal",
    )
    p_out, streak = apply_experiment_overrides(
        p_in, market_state=state,
        halt_streak_started_ts=streak_start,
        now=now, experiment=exp,
    )
    assert p_out.mode == "hedgerock"
    assert p_out.hedgerock_enabled is True
    assert p_out.lot_factor == pytest.approx(0.1)
    assert p_out.max_next_lot <= 0.02 + 1e-9
    assert p_out.max_orders_buy == 1
    assert p_out.max_orders_sell == 1
    assert p_out.cooldown_until is None
    # Reason MUST contain the unique tag so the report can detect it.
    assert "halt_auto_expiry_tiny_normal" in p_out.reason
    # Streak persists for the next bar — DD may still be severe, and the
    # next rule_engine call will re-fire halt → we must keep auto-expiring.
    assert streak == streak_start


@pytest.mark.unit
def test_tiny_normal_does_not_fire_below_threshold() -> None:
    """Halt elapsed < N hours → still halt, no override."""
    now = _now()
    streak_start = now - timedelta(hours=2)  # only 2h elapsed
    ea = build_ea_state(equity=9400.0, balance=10000.0, dd_pct=0.06, spread_pts=20)
    state = _state(regime_v2="range", confidence=0.85, ea_state=ea, now=now)
    p_in = _halt_params()
    exp = ExperimentConfig(
        halt_auto_expiry_hours=4.0,
        halt_auto_expiry_release="tiny_normal",
    )
    p_out, streak = apply_experiment_overrides(
        p_in, market_state=state,
        halt_streak_started_ts=streak_start,
        now=now, experiment=exp,
    )
    assert p_out.mode == "halt"  # not yet expired
    assert streak == streak_start


@pytest.mark.unit
def test_tiny_normal_streak_resets_on_non_halt() -> None:
    """rule_engine returns non-halt → streak clears regardless of release."""
    now = _now()
    streak_start = now - timedelta(hours=5)
    ea = build_ea_state(equity=10000.0, balance=10000.0, dd_pct=0.0, spread_pts=20)
    state = _state(regime_v2="range", confidence=0.85, ea_state=ea, now=now)
    # Hedgerock params (rule_engine recovered).
    p_in = DynamicParams(
        mode="hedgerock", hedgerock_enabled=True, risk_tier="normal",
        cooldown_until=None, lot_factor=1.0, grid_multiplier=1.0,
        max_next_lot=0.05, takeprofit_points=600, stoploss_points=3900,
        recovery_multiplier=1.2, max_orders_buy=2, max_orders_sell=2,
        reason="range + normal",
    )
    exp = ExperimentConfig(
        halt_auto_expiry_hours=4.0,
        halt_auto_expiry_release="tiny_normal",
    )
    p_out, streak = apply_experiment_overrides(
        p_in, market_state=state,
        halt_streak_started_ts=streak_start,
        now=now, experiment=exp,
    )
    assert p_out == p_in  # untouched
    assert streak is None  # reset


# ---------------------------------------------------------------------------
# 4. D-cont1b NEVER aggressive — hard cap at observe tier
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_tiny_normal_never_aggressive_high_confidence() -> None:
    """Even with confidence well above the aggressive floor, tiny_normal
    must produce risk_tier='observe'. The cap is hard."""
    now = _now()
    streak_start = now - timedelta(hours=10)  # very stale halt
    # Heavy ea history, max confidence — would normally enable aggressive.
    ea = build_ea_state(
        equity=9400.0, balance=10000.0, dd_pct=0.06, spread_pts=20,
        consec_losses=0, recent_closed_pnl=200.0, recent_sample_count=20,
    )
    state = _state(regime_v2="range", confidence=0.95, ea_state=ea, now=now)
    # Pretend rule_engine itself returned aggressive halt (artificial).
    p_in = replace(_halt_params(), risk_tier="aggressive")
    exp = ExperimentConfig(
        halt_auto_expiry_hours=4.0,
        halt_auto_expiry_release="tiny_normal",
    )
    p_out, _ = apply_experiment_overrides(
        p_in, market_state=state,
        halt_streak_started_ts=streak_start,
        now=now, experiment=exp,
    )
    assert p_out.risk_tier == "observe"
    assert p_out.lot_factor == pytest.approx(0.1)
    assert p_out.max_next_lot <= 0.02 + 1e-9


@pytest.mark.unit
def test_tiny_normal_caps_max_next_lot_when_input_was_larger() -> None:
    """If the inbound halt_params somehow had max_next_lot > 0.02, the
    override clamps it back down. Same for max_orders."""
    now = _now()
    streak_start = now - timedelta(hours=5)
    ea = build_ea_state(equity=9400.0, balance=10000.0, dd_pct=0.06, spread_pts=20)
    state = _state(regime_v2="range", confidence=0.85, ea_state=ea, now=now)
    p_in = replace(
        _halt_params(),
        max_next_lot=0.50,        # large — pretend leaked in
        max_orders_buy=10,        # large — pretend leaked in
        max_orders_sell=10,
    )
    exp = ExperimentConfig(
        halt_auto_expiry_hours=4.0,
        halt_auto_expiry_release="tiny_normal",
    )
    p_out, _ = apply_experiment_overrides(
        p_in, market_state=state,
        halt_streak_started_ts=streak_start,
        now=now, experiment=exp,
    )
    assert p_out.max_next_lot == pytest.approx(0.02)
    assert p_out.max_orders_buy == 1
    assert p_out.max_orders_sell == 1


# ---------------------------------------------------------------------------
# 5. Red-flag logic (CLI helper)
# ---------------------------------------------------------------------------


def _make_metrics(
    *, return_pct: float, dd_pct: float, near_stopout: int = 0,
    blowup: bool = False, aggressive_bars: int = 0,
) -> TradeMetrics:
    """Minimal TradeMetrics fixture — only the fields red-flag reads."""
    return TradeMetrics(
        final_equity=10000.0 * (1 + return_pct / 100.0),
        total_return_pct=return_pct,
        max_dd_pct=dd_pct,
        monthly_returns={},
        worst_month=("", 0.0),
        blowup=blowup,
        margin_stopout_count=0,
        near_stopout_count=near_stopout,
        n_trades=0,
        win_rate=0.0,
        avg_lot=0.0,
        max_lot=0.0,
        aggressive_tier_bars=aggressive_bars,
    )


def _import_redflag():
    """Lazy import — script lives under /scripts which isn't an installed
    package. We add it to sys.path for this test only."""
    import sys
    from pathlib import Path
    scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
    sys.path.insert(0, str(scripts_dir))
    try:
        from run_phase_d_walk_forward import _evaluate_variant_redflag
    finally:
        sys.path.pop(0)
    return _evaluate_variant_redflag


@pytest.mark.unit
def test_redflag_tiny_normal_triggers_when_dd_exceeds_redline() -> None:
    """tiny_normal: return improved BUT DD > baseline + 2pp → RED."""
    redflag = _import_redflag()
    baseline = _make_metrics(return_pct=-5.0, dd_pct=10.0)
    variant = ExperimentResult(
        label="d-cont1b",
        config=ExperimentConfig(
            halt_auto_expiry_hours=4.0,
            halt_auto_expiry_release="tiny_normal",
        ),
        metrics=_make_metrics(return_pct=-2.0, dd_pct=12.5),  # +2.5pp DD
        envelope_log=[],
    )
    red, badge, reasons = redflag(baseline, variant)
    assert red is True
    assert "RED FLAG" in badge
    assert any("DD deteriorated" in r for r in reasons)


@pytest.mark.unit
def test_redflag_tiny_normal_triggers_when_near_stopout_increases() -> None:
    """tiny_normal: ANY increase in near_stopout_count → RED, no tolerance."""
    redflag = _import_redflag()
    baseline = _make_metrics(return_pct=-5.0, dd_pct=10.0, near_stopout=3)
    variant = ExperimentResult(
        label="d-cont1b",
        config=ExperimentConfig(
            halt_auto_expiry_hours=4.0,
            halt_auto_expiry_release="tiny_normal",
        ),
        # DD same, return same — but near_stopout went UP by 1.
        metrics=_make_metrics(return_pct=-5.0, dd_pct=10.0, near_stopout=4),
        envelope_log=[],
    )
    red, badge, reasons = redflag(baseline, variant)
    assert red is True
    assert "RED FLAG" in badge
    assert any("near-stopout" in r for r in reasons)


@pytest.mark.unit
def test_redflag_tiny_normal_no_red_when_safe_improvement() -> None:
    """tiny_normal: return improves AND DD shrinks AND near_stopout same →
    PROMOTE-CANDIDATE."""
    redflag = _import_redflag()
    baseline = _make_metrics(return_pct=-5.0, dd_pct=10.0, near_stopout=3)
    variant = ExperimentResult(
        label="d-cont1b",
        config=ExperimentConfig(
            halt_auto_expiry_hours=4.0,
            halt_auto_expiry_release="tiny_normal",
        ),
        metrics=_make_metrics(return_pct=-2.0, dd_pct=8.0, near_stopout=3),
        envelope_log=[],
    )
    red, badge, _ = redflag(baseline, variant)
    assert red is False
    assert "PROMOTE-CANDIDATE" in badge


@pytest.mark.unit
def test_redflag_blowup_regression_is_red() -> None:
    """variant blew up while baseline survived → RED, regardless of return."""
    redflag = _import_redflag()
    baseline = _make_metrics(return_pct=-5.0, dd_pct=10.0, blowup=False)
    variant = ExperimentResult(
        label="d-cont1b",
        config=ExperimentConfig(
            halt_auto_expiry_hours=4.0,
            halt_auto_expiry_release="tiny_normal",
        ),
        metrics=_make_metrics(return_pct=-50.0, dd_pct=80.0, blowup=True),
        envelope_log=[],
    )
    red, badge, reasons = redflag(baseline, variant)
    assert red is True
    assert "RED FLAG" in badge
    assert any("blowup" in r.lower() for r in reasons)


@pytest.mark.unit
def test_redflag_dcont1_observe_uses_lenient_dd_band() -> None:
    """observe-only variant: DD allowed up to baseline+2pp without RED."""
    redflag = _import_redflag()
    baseline = _make_metrics(return_pct=-5.0, dd_pct=10.0, near_stopout=3)
    variant = ExperimentResult(
        label="d-cont1",
        config=ExperimentConfig(
            halt_auto_expiry_hours=4.0,
            halt_auto_expiry_release="observe",
        ),
        # DD up by 1.5pp — below the 2pp redline → not RED.
        # near_stopout up — NOT a red trigger for observe (only tiny_normal).
        metrics=_make_metrics(return_pct=-3.0, dd_pct=11.5, near_stopout=4),
        envelope_log=[],
    )
    red, _, _ = redflag(baseline, variant)
    assert red is False


# ---------------------------------------------------------------------------
# 6. run_walk_forward 3-way — experiments list with two configs
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
def test_run_walk_forward_three_way_records_two_experiments(small_lake) -> None:
    """experiments=[d-cont1, d-cont1b] yields two ExperimentResult entries
    in result.experiments, AND baseline.dynamic_metrics is identical to a
    no-experiments call (no cross-pollination)."""
    cfg = WalkForwardConfig(
        instrument="XAUUSD",
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 1, 31, tzinfo=timezone.utc),
        h1_lookback=120, h4_lookback=20,
    )
    experiments = [
        (
            "d-cont1",
            ExperimentConfig(
                halt_auto_expiry_hours=4.0,
                halt_auto_expiry_release="observe",
            ),
        ),
        (
            "d-cont1b",
            ExperimentConfig(
                halt_auto_expiry_hours=4.0,
                halt_auto_expiry_release="tiny_normal",
            ),
        ),
    ]
    result = run_walk_forward(cfg, small_lake, experiments=experiments)

    assert len(result.experiments) == 2
    assert [v.label for v in result.experiments] == ["d-cont1", "d-cont1b"]
    assert result.experiments[0].config.halt_auto_expiry_release == "observe"
    assert result.experiments[1].config.halt_auto_expiry_release == "tiny_normal"

    # Baseline byte-identical to a no-experiments run.
    baseline_only = run_walk_forward(cfg, small_lake)
    assert (
        result.dynamic_metrics.final_equity
        == baseline_only.dynamic_metrics.final_equity
    )
    assert (
        result.dynamic_metrics.bars_in_hedgerock
        == baseline_only.dynamic_metrics.bars_in_hedgerock
    )

    # Backwards-compat: experiment_metrics returns the FIRST variant.
    assert result.experiment_metrics is result.experiments[0].metrics
    assert result.experiment_envelope_log is result.experiments[0].envelope_log


@pytest.mark.unit
def test_run_walk_forward_skips_baseline_in_experiments_list(small_lake) -> None:
    """Passing a baseline ExperimentConfig in experiments list (no knobs
    set) is silently dropped — we don't want to duplicate baseline."""
    cfg = WalkForwardConfig(
        instrument="XAUUSD",
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 1, 31, tzinfo=timezone.utc),
        h1_lookback=120, h4_lookback=20,
    )
    experiments = [
        ("baseline-noop", ExperimentConfig()),  # is_baseline → True
        (
            "d-cont1b",
            ExperimentConfig(
                halt_auto_expiry_hours=4.0,
                halt_auto_expiry_release="tiny_normal",
            ),
        ),
    ]
    result = run_walk_forward(cfg, small_lake, experiments=experiments)
    # Only the non-baseline variant should appear.
    assert len(result.experiments) == 1
    assert result.experiments[0].label == "d-cont1b"


@pytest.mark.unit
def test_run_walk_forward_baseline_only_byte_identical_to_pre_dcont1b(
    small_lake,
) -> None:
    """Three callsites (no exp, experiments=None, experiments=[]) all
    produce identical baseline dynamic metrics."""
    cfg = WalkForwardConfig(
        instrument="XAUUSD",
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 1, 31, tzinfo=timezone.utc),
        h1_lookback=120, h4_lookback=20,
    )
    a = run_walk_forward(cfg, small_lake)
    b = run_walk_forward(cfg, small_lake, experiments=None)
    c = run_walk_forward(cfg, small_lake, experiments=[])
    assert a.dynamic_metrics.final_equity == b.dynamic_metrics.final_equity
    assert a.dynamic_metrics.final_equity == c.dynamic_metrics.final_equity
    assert a.dynamic_metrics.n_trades == c.dynamic_metrics.n_trades
    assert len(a.experiments) == 0
    assert len(b.experiments) == 0
    assert len(c.experiments) == 0


# ---------------------------------------------------------------------------
# 7. Phase D-cont1b-closeout — calibration lock for --experiment-set
# ---------------------------------------------------------------------------
#
# The named experiment-sets carry a fixed calibration that must match
# the original D-cont1 report's definition (cold_start_grace=True AND
# halt_auto_expiry_hours=4.0). These tests fail loudly if anyone retunes
# the constants in-place — re-tuning requires a NEW label.


def _import_experiments_from_args():
    import sys
    from pathlib import Path
    scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
    sys.path.insert(0, str(scripts_dir))
    try:
        from run_phase_d_walk_forward import (
            _DCONT1_CONFIG,
            _DCONT1B_CONFIG,
            _experiments_from_args,
        )
    finally:
        sys.path.pop(0)
    return _DCONT1_CONFIG, _DCONT1B_CONFIG, _experiments_from_args


def _ns(**kwargs):
    """Argparse-Namespace stub. Defaults match the script's parser."""
    import argparse
    defaults = dict(
        experiment_set="custom",
        experiment_cold_start_grace=False,
        experiment_halt_auto_expiry_hours=None,
        experiment_halt_auto_expiry_release="observe",
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


@pytest.mark.unit
def test_dcont1_canonical_config_calibration_locked() -> None:
    """`--experiment-set d-cont1` must use the *full* D-cont1 calibration:
    cold_start_grace=True + halt_auto_expiry_hours=4.0 + release='observe'.
    Re-tuning these constants requires a NEW label."""
    DCONT1, _DCONT1B, _ = _import_experiments_from_args()
    assert DCONT1.cold_start_grace is True, (
        "d-cont1 must keep cold_start_grace=True to match original "
        "D-cont1 report; rename the label if you want a release-only "
        "variant"
    )
    assert DCONT1.halt_auto_expiry_hours == 4.0
    assert DCONT1.halt_auto_expiry_release == "observe"
    assert DCONT1.is_baseline is False


@pytest.mark.unit
def test_dcont1b_canonical_config_calibration_locked() -> None:
    """`--experiment-set d-cont1b` must use the same gating knobs as
    D-cont1, with release='tiny_normal' as the only difference."""
    _DCONT1, DCONT1B, _ = _import_experiments_from_args()
    assert DCONT1B.cold_start_grace is True
    assert DCONT1B.halt_auto_expiry_hours == 4.0
    assert DCONT1B.halt_auto_expiry_release == "tiny_normal"
    assert DCONT1B.is_baseline is False


@pytest.mark.unit
def test_dcont1_and_dcont1b_share_gating_knobs() -> None:
    """Sanity: the only allowed difference between d-cont1 and d-cont1b
    is the release mode. Anything else MUST stay identical so the A/B
    delta is attributable to the release mode alone."""
    DCONT1, DCONT1B, _ = _import_experiments_from_args()
    assert DCONT1.cold_start_grace == DCONT1B.cold_start_grace
    assert DCONT1.halt_auto_expiry_hours == DCONT1B.halt_auto_expiry_hours
    assert DCONT1.halt_auto_expiry_release != DCONT1B.halt_auto_expiry_release


@pytest.mark.unit
def test_experiments_from_args_dcont1_returns_canonical() -> None:
    DCONT1, _DCONT1B, run_args = _import_experiments_from_args()
    out = run_args(_ns(experiment_set="d-cont1"))
    assert len(out) == 1
    label, cfg = out[0]
    assert label == "d-cont1"
    assert cfg == DCONT1


@pytest.mark.unit
def test_experiments_from_args_dcont1b_runs_three_way() -> None:
    DCONT1, DCONT1B, run_args = _import_experiments_from_args()
    out = run_args(_ns(experiment_set="d-cont1b"))
    assert [label for label, _ in out] == ["d-cont1", "d-cont1b"]
    assert out[0][1] == DCONT1
    assert out[1][1] == DCONT1B


@pytest.mark.unit
def test_experiments_from_args_custom_no_flags_is_empty() -> None:
    _DCONT1, _DCONT1B, run_args = _import_experiments_from_args()
    assert run_args(_ns(experiment_set="custom")) == []


@pytest.mark.unit
def test_experiments_from_args_custom_honours_flags() -> None:
    """Custom mode keeps the per-flag escape hatch — used when caller
    intentionally wants release-only without cold_start_grace."""
    _DCONT1, _DCONT1B, run_args = _import_experiments_from_args()
    out = run_args(_ns(
        experiment_set="custom",
        experiment_cold_start_grace=False,
        experiment_halt_auto_expiry_hours=4.0,
        experiment_halt_auto_expiry_release="tiny_normal",
    ))
    assert len(out) == 1
    label, cfg = out[0]
    assert label == "experiment"
    assert cfg.cold_start_grace is False
    assert cfg.halt_auto_expiry_hours == 4.0
    assert cfg.halt_auto_expiry_release == "tiny_normal"
