"""Phase D-cont1 — A/B experiment tests.

Pinned guarantees:
    1. Baseline (default config) unchanged — apply_experiment_overrides
       is a no-op when experiment.is_baseline.
    2. cold_start_grace promotes observe → hedgerock@normal under the
       strict eligibility rule, never to aggressive.
    3. halt_auto_expiry: after N hours of continuous halt, mode flips
       to observe with lot_factor=0.
    4. WalkForwardResult carries experiment_metrics when an experiment
       was supplied; baseline-only runs leave it None.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from smc.hedgerock.decision_server import MarketFeatures
from smc.hedgerock.ea_state import EAState, build_ea_state
from smc.hedgerock.market_state import aggregate_market_state
from smc.hedgerock.phase_d_walk_forward import (
    ExperimentConfig,
    WalkForwardConfig,
    apply_experiment_overrides,
    run_walk_forward,
)
from smc.hedgerock.regime_classifier_v2 import RegimeAssessmentV2
from smc.hedgerock.rule_engine import DynamicParams


# ---------------------------------------------------------------------------
# Helpers
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


def _observe_params(reason: str = "confidence=0.50 < 0.55 → observe") -> DynamicParams:
    """A rule_engine output that landed in observe due to confidence floor."""
    return DynamicParams(
        mode="observe",
        hedgerock_enabled=False,
        risk_tier="observe",
        cooldown_until=None,
        lot_factor=0.0,
        grid_multiplier=1.0,
        max_next_lot=0.05,
        takeprofit_points=600,
        stoploss_points=3900,
        recovery_multiplier=1.2,
        max_orders_buy=2,
        max_orders_sell=2,
        reason=reason,
    )


def _halt_params() -> DynamicParams:
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


def _hedgerock_params(risk_tier: str = "normal", lot_factor: float = 1.0) -> DynamicParams:
    return DynamicParams(
        mode="hedgerock",
        hedgerock_enabled=True,
        risk_tier=risk_tier,  # type: ignore[arg-type]
        cooldown_until=None,
        lot_factor=lot_factor,
        grid_multiplier=1.0,
        max_next_lot=0.05,
        takeprofit_points=600,
        stoploss_points=3900,
        recovery_multiplier=1.2,
        max_orders_buy=2,
        max_orders_sell=2,
        reason="range + normal",
    )


# ---------------------------------------------------------------------------
# 1. Baseline experiment is no-op
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_baseline_experiment_is_no_op() -> None:
    now = _now()
    ea = build_ea_state(equity=10000.0, balance=10000.0, dd_pct=0.0, spread_pts=20)
    state = _state(regime_v2="range", confidence=0.50, ea_state=ea, now=now)
    p_in = _observe_params()
    p_out, streak = apply_experiment_overrides(
        p_in, market_state=state, halt_streak_started_ts=None,
        now=now, experiment=ExperimentConfig(),
    )
    assert p_out == p_in
    assert streak is None


@pytest.mark.unit
def test_baseline_experiment_property() -> None:
    assert ExperimentConfig().is_baseline is True
    assert ExperimentConfig(cold_start_grace=True).is_baseline is False
    assert ExperimentConfig(halt_auto_expiry_hours=4).is_baseline is False


# ---------------------------------------------------------------------------
# 2. cold_start_grace
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_cold_start_grace_promotes_observe_to_normal_in_eligible_window() -> None:
    """range regime + conf 0.50 + cold start (no recent samples) →
    grace promotes observe → hedgerock@normal (NOT aggressive)."""
    now = _now()
    ea = build_ea_state(
        equity=10000.0, balance=10000.0, dd_pct=0.0, spread_pts=20,
        # No recent_* fields — cold start
    )
    state = _state(regime_v2="range", confidence=0.50, ea_state=ea, now=now)
    p_in = _observe_params()
    p_out, _streak = apply_experiment_overrides(
        p_in, market_state=state, halt_streak_started_ts=None,
        now=now, experiment=ExperimentConfig(cold_start_grace=True),
    )
    assert p_out.mode == "hedgerock"
    assert p_out.hedgerock_enabled is True
    assert p_out.risk_tier == "normal"  # ← MUST NOT be aggressive
    assert p_out.lot_factor == pytest.approx(1.0)
    assert "cold_start_grace" in p_out.reason


@pytest.mark.unit
def test_cold_start_grace_does_not_fire_below_grace_floor() -> None:
    """conf < 0.45 → grace does NOT fire (still too uncertain)."""
    now = _now()
    ea = build_ea_state(equity=10000.0, balance=10000.0, dd_pct=0.0, spread_pts=20)
    state = _state(regime_v2="range", confidence=0.40, ea_state=ea, now=now)
    p_in = _observe_params()
    p_out, _ = apply_experiment_overrides(
        p_in, market_state=state, halt_streak_started_ts=None,
        now=now, experiment=ExperimentConfig(cold_start_grace=True),
    )
    assert p_out == p_in  # untouched


@pytest.mark.unit
def test_cold_start_grace_does_not_fire_with_recent_history() -> None:
    """recent_sample_count >= MIN → not cold start → grace doesn't fire."""
    now = _now()
    ea = build_ea_state(
        equity=10000.0, balance=10000.0, dd_pct=0.0, spread_pts=20,
        consec_losses=0, recent_closed_pnl=10.0, recent_sample_count=20,
    )
    state = _state(regime_v2="range", confidence=0.50, ea_state=ea, now=now)
    p_in = _observe_params()
    p_out, _ = apply_experiment_overrides(
        p_in, market_state=state, halt_streak_started_ts=None,
        now=now, experiment=ExperimentConfig(cold_start_grace=True),
    )
    assert p_out == p_in


@pytest.mark.unit
def test_cold_start_grace_does_not_fire_outside_range_regime() -> None:
    """regime=trend_up → grace doesn't fire (HedgeRock only for range)."""
    now = _now()
    ea = build_ea_state(equity=10000.0, balance=10000.0, dd_pct=0.0, spread_pts=20)
    state = _state(regime_v2="trend_up", confidence=0.50, ea_state=ea, now=now)
    p_in = _observe_params()
    p_out, _ = apply_experiment_overrides(
        p_in, market_state=state, halt_streak_started_ts=None,
        now=now, experiment=ExperimentConfig(cold_start_grace=True),
    )
    assert p_out == p_in


@pytest.mark.unit
def test_cold_start_grace_does_not_promote_to_aggressive() -> None:
    """No matter how high the confidence, grace caps at normal — never aggressive."""
    now = _now()
    ea = build_ea_state(equity=10000.0, balance=10000.0, dd_pct=0.0, spread_pts=20)
    # conf = 0.54 — just below OBSERVE_FLOOR. Eligible window.
    state = _state(regime_v2="range", confidence=0.54, ea_state=ea, now=now)
    p_in = _observe_params()
    p_out, _ = apply_experiment_overrides(
        p_in, market_state=state, halt_streak_started_ts=None,
        now=now, experiment=ExperimentConfig(cold_start_grace=True),
    )
    assert p_out.mode == "hedgerock"
    assert p_out.risk_tier == "normal"
    assert p_out.lot_factor == pytest.approx(1.0)
    assert p_out.risk_tier != "aggressive"


@pytest.mark.unit
def test_cold_start_grace_does_not_override_cooldown() -> None:
    """If params has a cooldown active, grace doesn't override it."""
    now = _now()
    ea = build_ea_state(equity=10000.0, balance=10000.0, dd_pct=0.0, spread_pts=20)
    state = _state(regime_v2="range", confidence=0.50, ea_state=ea, now=now)
    p_in = replace(_observe_params(),
                   cooldown_until=now + timedelta(minutes=30))
    p_out, _ = apply_experiment_overrides(
        p_in, market_state=state, halt_streak_started_ts=None,
        now=now, experiment=ExperimentConfig(cold_start_grace=True),
    )
    assert p_out == p_in


# ---------------------------------------------------------------------------
# 3. halt_auto_expiry
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_halt_auto_expiry_starts_streak_on_first_halt_bar() -> None:
    """First halt bar — streak begins, params unchanged this bar."""
    now = _now()
    ea = build_ea_state(equity=9400.0, balance=10000.0, dd_pct=0.06, spread_pts=20)
    state = _state(regime_v2="range", confidence=0.85, ea_state=ea, now=now)
    p_in = _halt_params()
    p_out, streak = apply_experiment_overrides(
        p_in, market_state=state, halt_streak_started_ts=None,
        now=now, experiment=ExperimentConfig(halt_auto_expiry_hours=4),
    )
    assert p_out.mode == "halt"  # not yet expired
    assert streak == now


@pytest.mark.unit
def test_halt_auto_expiry_releases_after_n_hours() -> None:
    """After N hours of continuous halt, override to observe."""
    now = _now()
    streak_start = now - timedelta(hours=4, minutes=5)  # 4h + 5min ago
    ea = build_ea_state(equity=9400.0, balance=10000.0, dd_pct=0.06, spread_pts=20)
    state = _state(regime_v2="range", confidence=0.85, ea_state=ea, now=now)
    p_in = _halt_params()
    p_out, streak = apply_experiment_overrides(
        p_in, market_state=state, halt_streak_started_ts=streak_start,
        now=now, experiment=ExperimentConfig(halt_auto_expiry_hours=4),
    )
    assert p_out.mode == "observe"
    assert p_out.hedgerock_enabled is False
    assert p_out.lot_factor == 0.0
    assert p_out.cooldown_until is None
    assert "halt_auto_expiry" in p_out.reason
    # Streak must persist — DD is still severe; next bar's rule_engine
    # will re-fire halt and we'll keep auto-expiring.
    assert streak == streak_start


@pytest.mark.unit
def test_halt_auto_expiry_streak_resets_when_rule_engine_returns_non_halt() -> None:
    """When rule_engine itself returns non-halt (= equity recovered),
    the streak clears so next halt event starts fresh."""
    now = _now()
    streak_start = now - timedelta(hours=2)
    ea = build_ea_state(equity=10000.0, balance=10000.0, dd_pct=0.0, spread_pts=20)
    state = _state(regime_v2="range", confidence=0.85, ea_state=ea, now=now)
    p_in = _hedgerock_params()
    p_out, streak = apply_experiment_overrides(
        p_in, market_state=state, halt_streak_started_ts=streak_start,
        now=now, experiment=ExperimentConfig(halt_auto_expiry_hours=4),
    )
    assert p_out == p_in  # untouched
    assert streak is None  # reset


@pytest.mark.unit
def test_halt_auto_expiry_stays_halt_below_threshold() -> None:
    """Halt elapsed < N hours → still halt (no override)."""
    now = _now()
    streak_start = now - timedelta(hours=1)  # only 1h elapsed
    ea = build_ea_state(equity=9400.0, balance=10000.0, dd_pct=0.06, spread_pts=20)
    state = _state(regime_v2="range", confidence=0.85, ea_state=ea, now=now)
    p_in = _halt_params()
    p_out, streak = apply_experiment_overrides(
        p_in, market_state=state, halt_streak_started_ts=streak_start,
        now=now, experiment=ExperimentConfig(halt_auto_expiry_hours=4),
    )
    assert p_out.mode == "halt"
    assert streak == streak_start


# ---------------------------------------------------------------------------
# 4. WalkForwardResult contract
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
def test_run_walk_forward_baseline_only_leaves_experiment_metrics_none(
    small_lake,
) -> None:
    cfg = WalkForwardConfig(
        instrument="XAUUSD",
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 1, 31, tzinfo=timezone.utc),
        h1_lookback=120, h4_lookback=20,
    )
    result = run_walk_forward(cfg, small_lake)
    assert result.experiment_metrics is None
    assert result.experiment_envelope_log == []
    assert result.experiment_config is None


@pytest.mark.unit
def test_run_walk_forward_with_experiment_records_both_runs(small_lake) -> None:
    cfg = WalkForwardConfig(
        instrument="XAUUSD",
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 1, 31, tzinfo=timezone.utc),
        h1_lookback=120, h4_lookback=20,
    )
    exp = ExperimentConfig(cold_start_grace=True, halt_auto_expiry_hours=4)
    result = run_walk_forward(cfg, small_lake, experiment=exp)
    assert result.experiment_metrics is not None
    assert result.experiment_config == exp
    # Baseline dynamic must NOT be affected by the experiment toggle.
    # (We check structural equality of the baseline by re-running.)
    baseline_only = run_walk_forward(cfg, small_lake)
    assert (
        result.dynamic_metrics.final_equity
        == baseline_only.dynamic_metrics.final_equity
    )
    assert (
        result.dynamic_metrics.bars_in_hedgerock
        == baseline_only.dynamic_metrics.bars_in_hedgerock
    )


@pytest.mark.unit
def test_baseline_run_byte_identical_to_pre_dcont1(small_lake) -> None:
    """Running with config.experiment defaulting to baseline must
    produce IDENTICAL dynamic metrics to a run that explicitly passes
    experiment=None — the no-op invariant."""
    cfg = WalkForwardConfig(
        instrument="XAUUSD",
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 1, 31, tzinfo=timezone.utc),
        h1_lookback=120, h4_lookback=20,
    )
    a = run_walk_forward(cfg, small_lake)
    b = run_walk_forward(cfg, small_lake, experiment=None)
    c = run_walk_forward(cfg, small_lake, experiment=ExperimentConfig())
    assert a.dynamic_metrics.final_equity == b.dynamic_metrics.final_equity
    assert a.dynamic_metrics.final_equity == c.dynamic_metrics.final_equity
    assert a.dynamic_metrics.n_trades == c.dynamic_metrics.n_trades
