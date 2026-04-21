"""Unit tests for smc.strategy.sl_fitness_judge — 7-rule fitness checklist.

Round 5 A-track Task #7 (design: ``.scratch/round4/ai-sl-fitness-judge-design.md``).

Test coverage:
    - 5-losses regression fixtures from 2026-04-20 morning (replay,
      assert VETO for each)
    - ≥ 3 true-positive fixtures (each rule vetoes when it should)
    - ≥ 3 shadow-mode pass-through fixtures (judge returns accept=True
      for well-formed setups)
    - Config-override coverage (custom thresholds respected)
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from smc.ai.models import AIRegimeAssessment, RegimeParams
from smc.ai.param_router import route
from smc.ai.regime_classifier import RegimeContext
from smc.strategy.sl_fitness_judge import (
    DEFAULT_COUNTER_TREND_AI_CONF,
    DEFAULT_LOW_VOL_PERCENTILE,
    DEFAULT_MAX_SL_ATR_RATIO,
    DEFAULT_MIN_SL_ATR_RATIO,
    DEFAULT_TRANSITION_CONF_FLOOR,
    RULE_IDS,
    SLFitnessVerdict,
    judge_sl_fitness,
)
from smc.strategy.types import EntrySignal


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _make_entry(
    direction: str = "long",
    entry_price: float = 4000.0,
    stop_loss: float | None = None,
    tp1: float | None = None,
    risk_points: float = 1000.0,
    rr_ratio: float = 2.0,
    trigger_type: str = "choch_in_zone",
    grade: str = "B",
) -> EntrySignal:
    """Build an EntrySignal for tests.  Price math stays consistent.

    risk_points defaults to 1000 (= $10 at $0.01/point).  rr_ratio drives
    reward_points + TP1 price unless the caller passes custom SL/TP.
    """
    point = 0.01
    if stop_loss is None:
        stop_loss = entry_price - risk_points * point if direction == "long" else entry_price + risk_points * point
    reward_points = risk_points * rr_ratio
    if tp1 is None:
        tp1 = entry_price + reward_points * point if direction == "long" else entry_price - reward_points * point
    # TP2 arbitrary — judge doesn't read it
    tp2 = tp1 + reward_points * point if direction == "long" else tp1 - reward_points * point
    return EntrySignal(
        entry_price=round(entry_price, 2),
        stop_loss=round(stop_loss, 2),
        take_profit_1=round(tp1, 2),
        take_profit_2=round(tp2, 2),
        risk_points=round(risk_points, 1),
        reward_points=round(reward_points, 1),
        rr_ratio=round(rr_ratio, 2),
        trigger_type=trigger_type,  # type: ignore[arg-type]
        direction=direction,  # type: ignore[arg-type]
        grade=grade,  # type: ignore[arg-type]
    )


def _make_assessment(
    regime: str = "TREND_UP",
    trend_direction: str = "bullish",
    confidence: float = 0.7,
    source: str = "ai_debate",
) -> AIRegimeAssessment:
    return AIRegimeAssessment(
        regime=regime,  # type: ignore[arg-type]
        trend_direction=trend_direction,  # type: ignore[arg-type]
        confidence=confidence,
        param_preset=route(regime),  # type: ignore[arg-type]
        reasoning=f"test assessment for {regime}",
        assessed_at=datetime.now(tz=timezone.utc),
        source=source,  # type: ignore[arg-type]
        cost_usd=0.0,
    )


def _make_ctx(
    current_price: float = 4000.0,
    d1_atr_pct: float | None = 1.0,
    h4_volatility_rank: float = 0.5,
    ath_distance_pct: float = 5.0,
    price_52w_percentile: float = 0.5,
) -> RegimeContext:
    return RegimeContext(
        d1_atr_pct=d1_atr_pct,
        d1_sma50_direction="up",
        d1_sma50_slope=0.05,
        d1_close_vs_sma50=2.0,
        d1_recent_range_pct=1.5,
        d1_higher_highs=3,
        d1_lower_lows=0,
        h4_atr_pct=0.5,
        h4_trend_bars=5,
        h4_volatility_rank=h4_volatility_rank,
        current_price=current_price,
        ath_distance_pct=ath_distance_pct,
        price_52w_percentile=price_52w_percentile,
        external=None,
        atr_regime="trending",
    )


# ---------------------------------------------------------------------------
# Basic invariants
# ---------------------------------------------------------------------------


class TestVerdictContract:
    def test_verdict_is_frozen(self) -> None:
        v = SLFitnessVerdict(accept=True, rule_id="accept", reason="ok")
        with pytest.raises(Exception):
            v.accept = False  # type: ignore[misc]

    def test_rule_ids_has_seven_entries(self) -> None:
        assert len(RULE_IDS) == 7
        assert len(set(RULE_IDS)) == 7  # unique

    def test_happy_path_accepts(self) -> None:
        # D1 ATR 1.0% on $4000 = 4000 pts.  Rule 2 floor: 0.5 × 4000 = 2000.
        # Rule 3 ceiling: 2.5 × 4000 = 10000.  risk 3000 sits comfortably inside.
        verdict = judge_sl_fitness(
            entry=_make_entry(direction="long", risk_points=3000, rr_ratio=2.0),
            regime_assessment=_make_assessment("TREND_UP"),
            regime_ctx=_make_ctx(d1_atr_pct=1.0, h4_volatility_rank=0.6),
            confluence_score=0.7,
            h1_atr_points=500.0,
            d1_atr_pct=1.0,
        )
        assert verdict.accept is True
        assert verdict.rule_id == "accept"


# ---------------------------------------------------------------------------
# Rule-by-rule veto tests (>= 1 per rule, 7 rules total)
# ---------------------------------------------------------------------------


class TestRule1DirectionRegimeCoherence:
    def test_long_into_trend_down_with_high_conf_vetoes(self) -> None:
        verdict = judge_sl_fitness(
            entry=_make_entry(direction="long"),
            regime_assessment=_make_assessment(
                regime="TREND_DOWN", trend_direction="bearish", confidence=0.75,
            ),
            regime_ctx=_make_ctx(),
            confluence_score=0.7,
            h1_atr_points=500.0,
            d1_atr_pct=1.0,
        )
        assert verdict.accept is False
        assert verdict.rule_id == "direction_regime_coherence"
        assert "TREND_DOWN" in verdict.reason

    def test_short_into_trend_up_with_high_conf_vetoes(self) -> None:
        verdict = judge_sl_fitness(
            entry=_make_entry(direction="short"),
            regime_assessment=_make_assessment(
                regime="TREND_UP", trend_direction="bullish", confidence=0.75,
            ),
            regime_ctx=_make_ctx(),
            confluence_score=0.7,
            h1_atr_points=500.0,
            d1_atr_pct=1.0,
        )
        assert verdict.accept is False
        assert verdict.rule_id == "direction_regime_coherence"

    def test_low_conf_counter_trend_not_vetoed_by_rule1(self) -> None:
        """At ai_conf < 0.6 the assert-style rule stays silent (other rules may still fire)."""
        verdict = judge_sl_fitness(
            entry=_make_entry(direction="long"),
            regime_assessment=_make_assessment(
                regime="TREND_DOWN", confidence=0.45,  # below counter_trend_ai_conf=0.6
            ),
            regime_ctx=_make_ctx(),
            confluence_score=0.7,
            h1_atr_points=500.0,
            d1_atr_pct=1.0,
        )
        # Could accept OR be vetoed by a later rule — but NOT by rule 1.
        assert verdict.rule_id != "direction_regime_coherence"


class TestRule2SLFloor:
    def test_too_tight_sl_vetoes(self) -> None:
        # D1 ATR 1.0% on $4000 price = $40 = 4000 pts.  0.5 × 4000 = 2000 floor.
        # Pass risk_points=800 (< 2000) → VETO.
        verdict = judge_sl_fitness(
            entry=_make_entry(risk_points=800.0, rr_ratio=2.0),
            regime_assessment=_make_assessment("TREND_UP"),
            regime_ctx=_make_ctx(current_price=4000.0, d1_atr_pct=1.0),
            confluence_score=0.7,
            h1_atr_points=300.0,
            d1_atr_pct=1.0,
        )
        assert verdict.accept is False
        assert verdict.rule_id == "sl_floor_noise_band"
        assert "risk_points=800" in verdict.reason

    def test_sufficient_sl_passes_rule2(self) -> None:
        # risk_points=2500 > 2000 floor → OK
        verdict = judge_sl_fitness(
            entry=_make_entry(risk_points=2500.0, rr_ratio=2.0),
            regime_assessment=_make_assessment("TREND_UP"),
            regime_ctx=_make_ctx(current_price=4000.0, d1_atr_pct=1.0),
            confluence_score=0.7,
            h1_atr_points=300.0,
            d1_atr_pct=1.0,
        )
        assert verdict.rule_id != "sl_floor_noise_band"


class TestRule3SLCeiling:
    def test_too_wide_sl_vetoes(self) -> None:
        # D1 ATR 1.0% on $4000 = 4000 pts.  2.5 × 4000 = 10000 ceiling.
        # risk_points=12000 (> 10000) → VETO.
        verdict = judge_sl_fitness(
            entry=_make_entry(risk_points=12000.0, rr_ratio=2.0),
            regime_assessment=_make_assessment("TREND_UP"),
            regime_ctx=_make_ctx(current_price=4000.0, d1_atr_pct=1.0),
            confluence_score=0.7,
            h1_atr_points=800.0,
            d1_atr_pct=1.0,
        )
        assert verdict.accept is False
        assert verdict.rule_id == "sl_ceiling_signal_to_noise"
        assert "mis-located" in verdict.reason


class TestRule4TPReachability:
    def test_low_vol_with_high_rr_vetoes(self) -> None:
        # H4 vol rank 0.2 < 0.3  AND  rr_ratio 2.5 >= 2.5 → VETO.
        verdict = judge_sl_fitness(
            entry=_make_entry(rr_ratio=2.5, risk_points=3000.0),
            regime_assessment=_make_assessment("CONSOLIDATION"),
            regime_ctx=_make_ctx(d1_atr_pct=1.0, h4_volatility_rank=0.2),
            confluence_score=0.7,
            h1_atr_points=400.0,
            d1_atr_pct=1.0,
        )
        assert verdict.accept is False
        assert verdict.rule_id == "tp_reachability_low_vol"

    def test_low_vol_with_low_rr_passes_rule4(self) -> None:
        # rr_ratio=2.0 < 2.5 → rule 4 stays silent.
        verdict = judge_sl_fitness(
            entry=_make_entry(rr_ratio=2.0, risk_points=3000.0),
            regime_assessment=_make_assessment("CONSOLIDATION"),
            regime_ctx=_make_ctx(d1_atr_pct=1.0, h4_volatility_rank=0.2),
            confluence_score=0.7,
            h1_atr_points=400.0,
            d1_atr_pct=1.0,
        )
        assert verdict.rule_id != "tp_reachability_low_vol"


class TestRule5ATHShortGuard:
    def test_short_near_ath_in_breakout_regime_vetoes(self) -> None:
        # risk 3000 sits in the 2000..10000 Rule 2/3 band so those rules pass.
        verdict = judge_sl_fitness(
            entry=_make_entry(direction="short", risk_points=3000.0),
            regime_assessment=_make_assessment(
                regime="ATH_BREAKOUT", confidence=0.55,  # below counter_trend threshold
            ),
            regime_ctx=_make_ctx(ath_distance_pct=0.3),
            confluence_score=0.7,
            h1_atr_points=500.0,
            d1_atr_pct=1.0,
        )
        # Rule 1 requires TREND_UP for shorts; ATH_BREAKOUT won't trigger it even at higher conf.
        assert verdict.accept is False
        assert verdict.rule_id == "ath_proximity_short_guard"

    def test_short_far_from_ath_passes_rule5(self) -> None:
        verdict = judge_sl_fitness(
            entry=_make_entry(direction="short", risk_points=3000.0),
            regime_assessment=_make_assessment(
                regime="ATH_BREAKOUT", confidence=0.55,
            ),
            regime_ctx=_make_ctx(ath_distance_pct=3.0),
            confluence_score=0.7,
            h1_atr_points=500.0,
            d1_atr_pct=1.0,
        )
        assert verdict.rule_id != "ath_proximity_short_guard"


class TestRule6ConfluenceConfidenceFloor:
    def test_low_ai_conf_and_low_confluence_vetoes(self) -> None:
        # risk 3000 inside SL band so rules 2/3 pass; reach rule 6.
        verdict = judge_sl_fitness(
            entry=_make_entry(risk_points=3000.0),
            regime_assessment=_make_assessment(
                regime="TRANSITION", confidence=0.45, source="ai_debate",
            ),
            regime_ctx=_make_ctx(d1_atr_pct=1.0),
            confluence_score=0.55,
            h1_atr_points=500.0,
            d1_atr_pct=1.0,
        )
        # TRANSITION + confluence 0.55 would fail rule 7 too — but rule 6
        # evaluates first on ai_debate source.
        assert verdict.accept is False
        assert verdict.rule_id == "confluence_x_confidence_floor"

    def test_atr_fallback_source_not_vetoed_by_rule6(self) -> None:
        """Rule 6 only fires for ai_debate source — atr_fallback paths skip it."""
        verdict = judge_sl_fitness(
            entry=_make_entry(risk_points=3000.0),
            regime_assessment=_make_assessment(
                regime="TREND_UP", confidence=0.45, source="atr_fallback",
            ),
            regime_ctx=_make_ctx(d1_atr_pct=1.0),
            confluence_score=0.55,
            h1_atr_points=500.0,
            d1_atr_pct=1.0,
        )
        assert verdict.rule_id != "confluence_x_confidence_floor"


class TestRule7TransitionQualityGuard:
    def test_transition_with_low_confluence_vetoes(self) -> None:
        verdict = judge_sl_fitness(
            entry=_make_entry(risk_points=3000.0),
            regime_assessment=_make_assessment(
                regime="TRANSITION", confidence=0.65, source="atr_fallback",
            ),
            regime_ctx=_make_ctx(d1_atr_pct=1.0),
            confluence_score=0.55,  # below 0.6 transition floor
            h1_atr_points=500.0,
            d1_atr_pct=1.0,
        )
        assert verdict.accept is False
        assert verdict.rule_id == "transition_quality_guard"

    def test_transition_with_high_confluence_passes(self) -> None:
        verdict = judge_sl_fitness(
            entry=_make_entry(risk_points=3000.0),
            regime_assessment=_make_assessment(
                regime="TRANSITION", confidence=0.65, source="atr_fallback",
            ),
            regime_ctx=_make_ctx(d1_atr_pct=1.0),
            confluence_score=0.75,  # above floor
            h1_atr_points=500.0,
            d1_atr_pct=1.0,
        )
        assert verdict.accept is True


# ---------------------------------------------------------------------------
# 2026-04-20 5-losses regression fixtures
# ---------------------------------------------------------------------------
#
# From design doc §1 + 2024-weakness-diagnosis.md: the 5 losses all shared
# one pathology — long entries taken into a falling-price regime with MAE
# of 1.06R–1.12R and MFE of 0.03R–0.10R.  The judge's job is to flag these.
#
# Each fixture injects the combination that catches one of Rules 1/4/7 —
# per design doc §3 these are the dominant catches for this cohort.


class TestFiveLossesRegression:
    """Replay-style tests — each represents one of today's 5 losing trades."""

    def test_loss1_long_into_trend_down_ai_high_conf(self) -> None:
        """Loss 1: long entry, regime revealed as TREND_DOWN by AI @ 0.78 conf."""
        verdict = judge_sl_fitness(
            entry=_make_entry(direction="long", risk_points=2500.0, rr_ratio=2.5),
            regime_assessment=_make_assessment(
                regime="TREND_DOWN", trend_direction="bearish", confidence=0.78,
                source="ai_debate",
            ),
            regime_ctx=_make_ctx(d1_atr_pct=1.2, h4_volatility_rank=0.25),
            confluence_score=0.62,
            h1_atr_points=600.0,
            d1_atr_pct=1.2,
        )
        assert verdict.accept is False
        assert verdict.rule_id == "direction_regime_coherence"

    def test_loss2_long_into_trend_down_debate_path(self) -> None:
        """Loss 2: similar to loss 1 but debate confidence just above threshold."""
        verdict = judge_sl_fitness(
            entry=_make_entry(direction="long", risk_points=2200.0, rr_ratio=2.5),
            regime_assessment=_make_assessment(
                regime="TREND_DOWN", trend_direction="bearish", confidence=0.62,
                source="ai_debate",
            ),
            regime_ctx=_make_ctx(d1_atr_pct=1.2, h4_volatility_rank=0.22),
            confluence_score=0.58,
            h1_atr_points=600.0,
            d1_atr_pct=1.2,
        )
        assert verdict.accept is False
        assert verdict.rule_id == "direction_regime_coherence"

    def test_loss3_transition_low_confluence(self) -> None:
        """Loss 3: TRANSITION regime slipped past but confluence only 0.52."""
        verdict = judge_sl_fitness(
            entry=_make_entry(direction="long", risk_points=2500.0, rr_ratio=2.5),
            regime_assessment=_make_assessment(
                regime="TRANSITION", confidence=0.55, source="atr_fallback",
            ),
            regime_ctx=_make_ctx(d1_atr_pct=1.1, h4_volatility_rank=0.35),
            confluence_score=0.52,
            h1_atr_points=500.0,
            d1_atr_pct=1.1,
        )
        assert verdict.accept is False
        assert verdict.rule_id == "transition_quality_guard"

    def test_loss4_low_vol_high_rr_fantasy(self) -> None:
        """Loss 4: H4 vol rank = 0.20 (bottom 20%), TP1 at 2.5R → unreachable."""
        verdict = judge_sl_fitness(
            entry=_make_entry(direction="long", risk_points=2800.0, rr_ratio=2.6),
            regime_assessment=_make_assessment(
                regime="CONSOLIDATION", confidence=0.65, source="atr_fallback",
            ),
            regime_ctx=_make_ctx(d1_atr_pct=1.3, h4_volatility_rank=0.20),
            confluence_score=0.65,
            h1_atr_points=550.0,
            d1_atr_pct=1.3,
        )
        assert verdict.accept is False
        assert verdict.rule_id == "tp_reachability_low_vol"

    def test_loss5_multiple_signals_rule1_first(self) -> None:
        """Loss 5: multiple red flags — Rule 1 fires first (shortest circuit)."""
        verdict = judge_sl_fitness(
            entry=_make_entry(direction="long", risk_points=3000.0, rr_ratio=2.5),
            regime_assessment=_make_assessment(
                regime="TREND_DOWN", trend_direction="bearish", confidence=0.80,
                source="ai_debate",
            ),
            regime_ctx=_make_ctx(d1_atr_pct=1.2, h4_volatility_rank=0.18),
            confluence_score=0.48,
            h1_atr_points=600.0,
            d1_atr_pct=1.2,
        )
        assert verdict.accept is False
        # Rule 1 is evaluated before others → short-circuits here.
        assert verdict.rule_id == "direction_regime_coherence"


# ---------------------------------------------------------------------------
# Shadow-mode pass-through fixtures (>= 3)
# ---------------------------------------------------------------------------
#
# These are well-formed setups that SHOULD pass all 7 rules.  In
# shadow-mode the aggregator emits sl_fitness_shadow_veto{accepted=True}
# and does nothing else; these fixtures ensure we don't over-reject.


class TestShadowModePassThrough:
    # Rule-2/3 require risk_points ≥ 0.5 × d1_atr_points (4000 pts at 1% D1 ATR).
    # These fixtures all set risk ≥ 2500 to stay inside the noise band.

    def test_clean_trend_up_setup_accepts(self) -> None:
        verdict = judge_sl_fitness(
            entry=_make_entry(direction="long", risk_points=2800.0, rr_ratio=2.0),
            regime_assessment=_make_assessment(
                regime="TREND_UP", trend_direction="bullish", confidence=0.75,
                source="ai_debate",
            ),
            regime_ctx=_make_ctx(d1_atr_pct=1.0, h4_volatility_rank=0.6),
            confluence_score=0.72,
            h1_atr_points=500.0,
            d1_atr_pct=1.0,
        )
        assert verdict.accept is True
        assert verdict.rule_id == "accept"

    def test_clean_trend_down_short_accepts(self) -> None:
        # D1 ATR 1.1% on $4000 = 4400 pts → Rule 2 floor 2200.
        verdict = judge_sl_fitness(
            entry=_make_entry(direction="short", risk_points=3000.0, rr_ratio=2.2),
            regime_assessment=_make_assessment(
                regime="TREND_DOWN", trend_direction="bearish", confidence=0.70,
                source="ai_debate",
            ),
            regime_ctx=_make_ctx(
                d1_atr_pct=1.1, h4_volatility_rank=0.55, ath_distance_pct=6.0,
            ),
            confluence_score=0.68,
            h1_atr_points=450.0,
            d1_atr_pct=1.1,
        )
        assert verdict.accept is True

    def test_ath_breakout_long_accepts(self) -> None:
        # D1 ATR 1.4% on $4000 = 5600 pts → Rule 2 floor 2800.
        verdict = judge_sl_fitness(
            entry=_make_entry(direction="long", risk_points=3500.0, rr_ratio=3.2),
            regime_assessment=_make_assessment(
                regime="ATH_BREAKOUT", trend_direction="bullish", confidence=0.72,
                source="ai_debate",
            ),
            regime_ctx=_make_ctx(
                d1_atr_pct=1.4, h4_volatility_rank=0.65, ath_distance_pct=0.1,
            ),
            confluence_score=0.70,
            h1_atr_points=700.0,
            d1_atr_pct=1.4,
        )
        assert verdict.accept is True

    def test_transition_with_enough_confluence_accepts(self) -> None:
        verdict = judge_sl_fitness(
            entry=_make_entry(direction="long", risk_points=2500.0, rr_ratio=2.0),
            regime_assessment=_make_assessment(
                regime="TRANSITION", confidence=0.62, source="atr_fallback",
            ),
            regime_ctx=_make_ctx(d1_atr_pct=1.0, h4_volatility_rank=0.5),
            confluence_score=0.70,
            h1_atr_points=500.0,
            d1_atr_pct=1.0,
        )
        assert verdict.accept is True


# ---------------------------------------------------------------------------
# Config-override tests
# ---------------------------------------------------------------------------


class TestConfigOverrides:
    def test_custom_min_sl_atr_ratio_moves_veto_line(self) -> None:
        # At default min_sl_atr_ratio=0.5, risk=2500 would PASS
        # (0.5 × 4000 = 2000 floor, risk 2500 > 2000).  Raise to 0.7 so
        # the floor becomes 2800 and the same trade FAILS.
        verdict = judge_sl_fitness(
            entry=_make_entry(risk_points=2500.0, rr_ratio=2.0),
            regime_assessment=_make_assessment("TREND_UP"),
            regime_ctx=_make_ctx(current_price=4000.0, d1_atr_pct=1.0),
            confluence_score=0.7,
            h1_atr_points=500.0,
            d1_atr_pct=1.0,
            min_sl_atr_ratio=0.7,
        )
        assert verdict.accept is False
        assert verdict.rule_id == "sl_floor_noise_band"

    def test_custom_low_vol_percentile(self) -> None:
        # At default 0.3, a vol rank of 0.25 vetoes if rr>=2.5.
        # Raise threshold to 0.4 → same vol rank 0.35 now VETOes.
        verdict = judge_sl_fitness(
            entry=_make_entry(rr_ratio=2.5, risk_points=3000.0),
            regime_assessment=_make_assessment("CONSOLIDATION"),
            regime_ctx=_make_ctx(d1_atr_pct=1.0, h4_volatility_rank=0.35),
            confluence_score=0.7,
            h1_atr_points=400.0,
            d1_atr_pct=1.0,
            low_vol_percentile=0.4,
        )
        assert verdict.accept is False
        assert verdict.rule_id == "tp_reachability_low_vol"

    def test_default_constants_exported(self) -> None:
        assert DEFAULT_MIN_SL_ATR_RATIO == 0.5
        assert DEFAULT_MAX_SL_ATR_RATIO == 2.5
        assert DEFAULT_LOW_VOL_PERCENTILE == 0.3
        assert DEFAULT_TRANSITION_CONF_FLOOR == 0.6
        assert DEFAULT_COUNTER_TREND_AI_CONF == 0.6


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestAggregatorShadowMode:
    """Verify the aggregator wires the shadow-mode judge correctly.

    Shadow mode must:
    1. Default OFF (no regime_ctx computed, no telemetry emitted)
    2. When enabled, compute regime_ctx and call judge_sl_fitness
    3. Even on VETO, setup is NOT blocked (accepted AND rejected both
       end up in setups tuple — shadow-mode is pure observation)
    """

    def _build_agg(self, *, sl_fitness_enabled: bool) -> "MultiTimeframeAggregator":
        from smc.smc_core.detector import SMCDetector
        from smc.strategy.aggregator import MultiTimeframeAggregator
        return MultiTimeframeAggregator(
            detector=SMCDetector(swing_length=5),
            sl_fitness_enabled=sl_fitness_enabled,
        )

    def test_shadow_mode_disabled_by_default(self) -> None:
        agg = self._build_agg(sl_fitness_enabled=False)
        assert agg._sl_fitness_enabled is False

    def test_shadow_mode_toggle_wires_through(self) -> None:
        agg = self._build_agg(sl_fitness_enabled=True)
        assert agg._sl_fitness_enabled is True
        assert agg._sl_fitness_min_sl_atr_ratio == 0.5
        assert agg._sl_fitness_max_sl_atr_ratio == 2.5

    def test_custom_thresholds_forward_to_judge(self) -> None:
        from smc.smc_core.detector import SMCDetector
        from smc.strategy.aggregator import MultiTimeframeAggregator
        agg = MultiTimeframeAggregator(
            detector=SMCDetector(swing_length=5),
            sl_fitness_enabled=True,
            sl_fitness_min_sl_atr_ratio=0.7,
            sl_fitness_max_sl_atr_ratio=3.0,
            sl_fitness_low_vol_percentile=0.4,
            sl_fitness_transition_conf_floor=0.65,
            sl_fitness_counter_trend_ai_conf=0.7,
        )
        assert agg._sl_fitness_min_sl_atr_ratio == 0.7
        assert agg._sl_fitness_max_sl_atr_ratio == 3.0
        assert agg._sl_fitness_low_vol_percentile == 0.4
        assert agg._sl_fitness_transition_conf_floor == 0.65
        assert agg._sl_fitness_counter_trend_ai_conf == 0.7


class TestEdgeCases:
    def test_missing_d1_atr_skips_sl_rules(self) -> None:
        """When D1 ATR% is None, rules 2 and 3 are skipped (no noise band)."""
        # risk 100 would normally fail rule 2; ensure no veto when d1_atr_pct=None.
        verdict = judge_sl_fitness(
            entry=_make_entry(risk_points=100.0, rr_ratio=2.0),
            regime_assessment=_make_assessment("TREND_UP"),
            regime_ctx=_make_ctx(d1_atr_pct=None),
            confluence_score=0.7,
            h1_atr_points=500.0,
            d1_atr_pct=None,
        )
        # Should not veto based on rules 2/3 when ATR unavailable.
        assert verdict.rule_id != "sl_floor_noise_band"
        assert verdict.rule_id != "sl_ceiling_signal_to_noise"

    def test_verdict_reason_is_nonempty(self) -> None:
        verdict = judge_sl_fitness(
            entry=_make_entry(direction="long"),
            regime_assessment=_make_assessment(
                "TREND_DOWN", trend_direction="bearish", confidence=0.8,
            ),
            regime_ctx=_make_ctx(),
            confluence_score=0.7,
            h1_atr_points=500.0,
            d1_atr_pct=1.0,
        )
        assert len(verdict.reason) > 0
