"""Unit tests for smc.strategy.mode_router — priority-based mode routing."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from smc.ai.models import AIRegimeAssessment, RegimeParams
from smc.strategy.hysteresis import HysteresisState
from smc.strategy.mode_router import route_trading_mode
from smc.strategy.range_types import RangeBounds


# ---------------------------------------------------------------------------
# Shared AIRegimeAssessment factory (Round 7 P0-1)
# ---------------------------------------------------------------------------


def _make_ai_assessment(
    regime: str,
    confidence: float,
    trend_direction: str | None = None,
) -> AIRegimeAssessment:
    """Build a minimal frozen AIRegimeAssessment for routing tests."""
    if trend_direction is None:
        if regime in ("TREND_UP", "ATH_BREAKOUT"):
            trend_direction = "bullish"
        elif regime == "TREND_DOWN":
            trend_direction = "bearish"
        else:
            trend_direction = "neutral"
    params = RegimeParams(
        sl_atr_multiplier=0.75, tp1_rr=2.5, confluence_floor=0.45,
        allowed_directions=("long", "short"), allowed_triggers=("fvg_fill_in_zone",),
        max_concurrent=3, zone_cooldown_hours=24,
        enable_ob_test=False, regime_label=regime,
    )
    return AIRegimeAssessment(
        regime=regime,  # type: ignore[arg-type]
        trend_direction=trend_direction,  # type: ignore[arg-type]
        confidence=confidence,
        param_preset=params,
        reasoning=f"test fixture for {regime}",
        assessed_at=datetime(2024, 6, 10, 12, 0, tzinfo=timezone.utc),
        source="ai_debate",
        cost_usd=0.0,
    )


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_range_bounds() -> RangeBounds:
    return RangeBounds(
        upper=2380.00,
        lower=2340.00,
        width_points=4000.0,
        midpoint=2360.00,
        detected_at=datetime(2024, 6, 10, 12, 0, tzinfo=timezone.utc),
        source="ob_boundaries",
        confidence=0.85,
        duration_bars=20,
    )


# ---------------------------------------------------------------------------
# ASIAN_CORE session: Round 4.5 — ranging ALLOWED, trending still blocked
# ---------------------------------------------------------------------------


class TestAsianCoreSession:
    def test_asian_core_neutral_no_range(self) -> None:
        result = route_trading_mode(
            ai_direction="neutral",
            ai_confidence=0.2,
            regime="ranging",
            session="ASIAN_CORE",
            range_bounds=None,
        )
        assert result.mode == "v1_passthrough"

    def test_asian_core_with_range_and_guards_enters_ranging(
        self, sample_range_bounds: RangeBounds
    ) -> None:
        """Round 4.5: ASIAN_CORE now allows ranging when range + guards pass."""
        result = route_trading_mode(
            ai_direction="neutral",
            ai_confidence=0.2,
            regime="ranging",
            session="ASIAN_CORE",
            range_bounds=sample_range_bounds,
            guards_passed=True,
        )
        assert result.mode == "ranging"

    def test_asian_core_bullish_high_conf_still_v1_passthrough(self) -> None:
        """ASIAN_CORE continues to block trending even at high confidence."""
        result = route_trading_mode(
            ai_direction="bullish",
            ai_confidence=0.9,
            regime="trending",
            session="ASIAN_CORE",
            range_bounds=None,
        )
        assert result.mode == "v1_passthrough"


# ---------------------------------------------------------------------------
# Priority 1: AI strong directional + confidence >= 0.5 + NOT ASIAN_CORE
# ---------------------------------------------------------------------------


class TestPriority1Trending:
    def test_london_bullish_0_6(self) -> None:
        result = route_trading_mode(
            ai_direction="bullish",
            ai_confidence=0.6,
            regime="trending",
            session="LONDON",
            range_bounds=None,
        )
        assert result.mode == "trending"
        assert "bullish" in result.reason

    def test_bullish_exactly_threshold(self) -> None:
        result = route_trading_mode(
            ai_direction="bullish",
            ai_confidence=0.5,
            regime="transitional",
            session="LONDON",
            range_bounds=None,
        )
        assert result.mode == "trending"

    def test_bearish_high_conf_new_york(self) -> None:
        result = route_trading_mode(
            ai_direction="bearish",
            ai_confidence=0.8,
            regime="trending",
            session="NEW YORK",
            range_bounds=None,
        )
        assert result.mode == "trending"
        assert "bearish" in result.reason

    def test_asian_non_core_bullish_allowed(self) -> None:
        """'ASIAN' session (not 'ASIAN_CORE') is permitted for trending."""
        result = route_trading_mode(
            ai_direction="bullish",
            ai_confidence=0.7,
            regime="trending",
            session="ASIAN",
            range_bounds=None,
        )
        assert result.mode == "trending"

    def test_trending_priority_over_range_and_guards(
        self, sample_range_bounds: RangeBounds
    ) -> None:
        """Even with range_bounds + guards_passed, high AI confidence wins."""
        result = route_trading_mode(
            ai_direction="bullish",
            ai_confidence=0.7,
            regime="ranging",
            session="LONDON",
            range_bounds=sample_range_bounds,
            guards_passed=True,
        )
        assert result.mode == "trending"

    def test_below_conf_threshold_does_not_trend(self) -> None:
        # Round 5 T5: threshold lowered from 0.50 to 0.45 so 0.47 edge
        # cases can enter trending. Update test to use 0.44 (just below).
        result = route_trading_mode(
            ai_direction="bullish",
            ai_confidence=0.44,
            regime="trending",
            session="LONDON",
            range_bounds=None,
        )
        assert result.mode != "trending"


# ---------------------------------------------------------------------------
# Priority 2: range + 5 guards pass + active session → ranging
# ---------------------------------------------------------------------------


class TestPriority2Ranging:
    def test_asian_london_transition_with_range_and_guards(
        self, sample_range_bounds: RangeBounds
    ) -> None:
        result = route_trading_mode(
            ai_direction="neutral",
            ai_confidence=0.2,
            regime="ranging",
            session="ASIAN_LONDON_TRANSITION",
            range_bounds=sample_range_bounds,
            guards_passed=True,
        )
        assert result.mode == "ranging"
        assert result.range_bounds is not None

    def test_london_with_range_and_guards(
        self, sample_range_bounds: RangeBounds
    ) -> None:
        result = route_trading_mode(
            ai_direction="neutral",
            ai_confidence=0.3,
            regime="ranging",
            session="LONDON",
            range_bounds=sample_range_bounds,
            guards_passed=True,
        )
        assert result.mode == "ranging"

    def test_new_york_with_range_and_guards(
        self, sample_range_bounds: RangeBounds
    ) -> None:
        result = route_trading_mode(
            ai_direction="neutral",
            ai_confidence=0.1,
            regime="transitional",
            session="NEW YORK",
            range_bounds=sample_range_bounds,
            guards_passed=True,
        )
        assert result.mode == "ranging"

    def test_guards_failed_gives_v1_passthrough(
        self, sample_range_bounds: RangeBounds
    ) -> None:
        result = route_trading_mode(
            ai_direction="neutral",
            ai_confidence=0.2,
            regime="ranging",
            session="LONDON",
            range_bounds=sample_range_bounds,
            guards_passed=False,
        )
        assert result.mode == "v1_passthrough"

    def test_no_range_bounds_gives_v1_passthrough(self) -> None:
        result = route_trading_mode(
            ai_direction="neutral",
            ai_confidence=0.2,
            regime="ranging",
            session="LONDON",
            range_bounds=None,
            guards_passed=True,
        )
        assert result.mode == "v1_passthrough"

    def test_low_conf_bullish_with_range_and_guards(
        self, sample_range_bounds: RangeBounds
    ) -> None:
        """Bullish but below threshold → falls to Priority 2 ranging."""
        result = route_trading_mode(
            ai_direction="bullish",
            ai_confidence=0.3,
            regime="ranging",
            session="LONDON",
            range_bounds=sample_range_bounds,
            guards_passed=True,
        )
        assert result.mode == "ranging"


# ---------------------------------------------------------------------------
# Fallback: v1_passthrough
# ---------------------------------------------------------------------------


class TestV1Passthrough:
    def test_neutral_no_range(self) -> None:
        result = route_trading_mode(
            ai_direction="neutral",
            ai_confidence=0.1,
            regime="ranging",
            session="LONDON",
            range_bounds=None,
        )
        assert result.mode == "v1_passthrough"
        assert "v1 pipeline" in result.reason

    def test_low_conf_bearish_no_range(self) -> None:
        result = route_trading_mode(
            ai_direction="bearish",
            ai_confidence=0.2,
            regime="transitional",
            session="NEW YORK",
            range_bounds=None,
        )
        assert result.mode == "v1_passthrough"
        assert "v1 pipeline" in result.reason

    def test_neutral_trending_regime_no_range(self) -> None:
        result = route_trading_mode(
            ai_direction="neutral",
            ai_confidence=0.3,
            regime="trending",
            session="LONDON",
            range_bounds=None,
        )
        assert result.mode == "v1_passthrough"
        assert "AI unsure" in result.reason


# ---------------------------------------------------------------------------
# Context passthrough
# ---------------------------------------------------------------------------


class TestContextFields:
    def test_fields_preserved(self, sample_range_bounds: RangeBounds) -> None:
        result = route_trading_mode(
            ai_direction="bullish",
            ai_confidence=0.9,
            regime="trending",
            session="LONDON",
            range_bounds=sample_range_bounds,
        )
        assert result.ai_direction == "bullish"
        assert result.ai_confidence == 0.9
        assert result.regime == "trending"
        assert result.range_bounds == sample_range_bounds

    def test_frozen(self) -> None:
        result = route_trading_mode(
            ai_direction="neutral",
            ai_confidence=0.1,
            regime="ranging",
            session="LONDON",
            range_bounds=None,
        )
        with pytest.raises(Exception):
            result.mode = "trending"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Hysteresis integration: route_trading_mode output fed through HysteresisState
# ---------------------------------------------------------------------------


class TestModeRouterWithHysteresis:
    def test_bar1_ranging_commits_enter_bar2_break_starts_exit(
        self, sample_range_bounds: RangeBounds
    ) -> None:
        """Audit R3 S5 asymmetric hysteresis:
        - bar1 proposes ranging → enter 1-bar commits immediately
        - bar2 proposes v1 (guards fail) → exit pending, not yet committed
        """
        state = HysteresisState()

        mode1 = route_trading_mode(
            ai_direction="neutral",
            ai_confidence=0.2,
            regime="ranging",
            session="LONDON",
            range_bounds=sample_range_bounds,
            guards_passed=True,
        )
        committed1 = state.update(mode1.mode)
        assert committed1 == "ranging"  # S5: enter commits in 1 bar

        mode2 = route_trading_mode(
            ai_direction="neutral",
            ai_confidence=0.2,
            regime="ranging",
            session="LONDON",
            range_bounds=sample_range_bounds,
            guards_passed=False,  # guards fail → v1_passthrough proposal
        )
        committed2 = state.update(mode2.mode)
        assert committed2 == "ranging"  # exit needs 2 bars, only 1 so far
        assert state.pending_mode == "v1_passthrough"

    def test_bar1_bar2_ranging_flips(
        self, sample_range_bounds: RangeBounds
    ) -> None:
        """Two consecutive ranging proposals commit to ranging."""
        state = HysteresisState()
        committed = "v1_passthrough"

        for _ in range(2):
            mode = route_trading_mode(
                ai_direction="neutral",
                ai_confidence=0.2,
                regime="ranging",
                session="LONDON",
                range_bounds=sample_range_bounds,
                guards_passed=True,
            )
            committed = state.update(mode.mode)

        assert committed == "ranging"

    def test_flipped_ranging_bar1_bar2_break_flips_back(
        self, sample_range_bounds: RangeBounds
    ) -> None:
        """Flipped to ranging; two consecutive break bars flip back to v1_passthrough."""
        state = HysteresisState(current_mode="ranging")
        committed = "ranging"

        for _ in range(2):
            mode = route_trading_mode(
                ai_direction="neutral",
                ai_confidence=0.2,
                regime="ranging",
                session="LONDON",
                range_bounds=sample_range_bounds,
                guards_passed=False,  # guards fail
            )
            committed = state.update(mode.mode)

        assert committed == "v1_passthrough"


# ---------------------------------------------------------------------------
# Priority 0 (Round 7 P0-1): AI regime assessment override
# ---------------------------------------------------------------------------


class TestAIRegimeIntegration:
    """Round 7 P0-1 — AI regime classifier output drives mode selection.

    Gate semantics:
      - When ``ai_regime_assessment`` is None OR confidence below threshold,
        the router's behaviour must be byte-identical to Round 4 v5.
      - When present AND confidence ≥ threshold, the AI regime overrides.
    """

    # -- Gate OFF: assessment is None or conf below threshold → legacy path --

    def test_gate_off_assessment_none_falls_through(
        self, sample_range_bounds: RangeBounds
    ) -> None:
        """Assessment None → behaves exactly as Priority 1-3 (legacy)."""
        result = route_trading_mode(
            ai_direction="neutral",
            ai_confidence=0.2,
            regime="ranging",
            session="LONDON",
            range_bounds=sample_range_bounds,
            guards_passed=True,
            ai_regime_assessment=None,
            ai_regime_trust_threshold=0.6,
        )
        # Matches TestPriority2Ranging.test_london_with_range_and_guards
        assert result.mode == "ranging"

    def test_gate_below_threshold_falls_through(
        self, sample_range_bounds: RangeBounds
    ) -> None:
        """Conf 0.55 < 0.6 threshold → legacy Priority 1-3 logic."""
        assessment = _make_ai_assessment("CONSOLIDATION", confidence=0.55)
        result = route_trading_mode(
            ai_direction="neutral",
            ai_confidence=0.2,
            regime="ranging",
            session="LONDON",
            range_bounds=sample_range_bounds,
            guards_passed=True,
            ai_regime_assessment=assessment,
            ai_regime_trust_threshold=0.6,
        )
        # Still satisfies Priority 2 → ranging (legacy behaviour preserved).
        assert result.mode == "ranging"
        assert "AI regime" not in result.reason  # legacy reason string

    # -- Trend regimes: force trending regardless of ai_direction/regime/range --

    def test_trend_up_forces_trending_over_neutral_direction(
        self, sample_range_bounds: RangeBounds
    ) -> None:
        assessment = _make_ai_assessment("TREND_UP", confidence=0.8)
        result = route_trading_mode(
            ai_direction="neutral",
            ai_confidence=0.2,
            regime="ranging",  # ATR says ranging
            session="LONDON",
            range_bounds=sample_range_bounds,  # range detected
            guards_passed=True,
            ai_regime_assessment=assessment,
        )
        assert result.mode == "trending"
        assert "TREND_UP" in result.reason
        assert "forcing trending path" in result.reason

    def test_trend_down_forces_trending(
        self, sample_range_bounds: RangeBounds
    ) -> None:
        assessment = _make_ai_assessment("TREND_DOWN", confidence=0.75)
        result = route_trading_mode(
            ai_direction="neutral",
            ai_confidence=0.3,
            regime="transitional",
            session="NEW YORK",
            range_bounds=sample_range_bounds,
            guards_passed=True,
            ai_regime_assessment=assessment,
        )
        assert result.mode == "trending"
        assert "TREND_DOWN" in result.reason

    def test_ath_breakout_forces_trending_at_threshold(
        self, sample_range_bounds: RangeBounds
    ) -> None:
        """ATH_BREAKOUT at exactly threshold (0.7 ≥ 0.7) → trending."""
        assessment = _make_ai_assessment("ATH_BREAKOUT", confidence=0.7)
        result = route_trading_mode(
            ai_direction="neutral",
            ai_confidence=0.2,
            regime="ranging",
            session="LONDON",
            range_bounds=sample_range_bounds,
            guards_passed=True,
            ai_regime_assessment=assessment,
            ai_regime_trust_threshold=0.7,
        )
        assert result.mode == "trending"
        assert "ATH_BREAKOUT" in result.reason

    # -- CONSOLIDATION: allow ranging only when guards + session + range align --

    def test_consolidation_allows_ranging_when_preconditions_met(
        self, sample_range_bounds: RangeBounds
    ) -> None:
        assessment = _make_ai_assessment("CONSOLIDATION", confidence=0.7)
        result = route_trading_mode(
            ai_direction="neutral",
            ai_confidence=0.2,
            regime="transitional",
            session="LONDON",
            range_bounds=sample_range_bounds,
            guards_passed=True,
            current_price=2360.0,  # in range
            ai_regime_assessment=assessment,
        )
        assert result.mode == "ranging"
        assert "CONSOLIDATION" in result.reason

    def test_consolidation_without_guards_to_v1(
        self, sample_range_bounds: RangeBounds
    ) -> None:
        """CONSOLIDATION but guards fail → v1_passthrough (not legacy fallback)."""
        assessment = _make_ai_assessment("CONSOLIDATION", confidence=0.7)
        result = route_trading_mode(
            ai_direction="neutral",
            ai_confidence=0.2,
            regime="transitional",
            session="LONDON",
            range_bounds=sample_range_bounds,
            guards_passed=False,  # guards fail
            ai_regime_assessment=assessment,
        )
        assert result.mode == "v1_passthrough"
        assert "CONSOLIDATION" in result.reason
        assert "preconditions missing" in result.reason

    # -- TRANSITION: default v1, exception when ATR trending + AI dir conviction --

    def test_transition_default_to_v1_passthrough(
        self, sample_range_bounds: RangeBounds
    ) -> None:
        """Regular TRANSITION → v1_passthrough blocks ranging AND trending."""
        assessment = _make_ai_assessment("TRANSITION", confidence=0.7)
        result = route_trading_mode(
            ai_direction="neutral",
            ai_confidence=0.2,
            regime="ranging",
            session="LONDON",
            range_bounds=sample_range_bounds,
            guards_passed=True,  # would trigger legacy ranging
            ai_regime_assessment=assessment,
        )
        assert result.mode == "v1_passthrough"
        assert "TRANSITION" in result.reason

    def test_transition_momentum_exception_allows_trending(
        self,
    ) -> None:
        """TRANSITION + ATR trending + AI bullish ≥ 0.45 → momentum-follow."""
        assessment = _make_ai_assessment("TRANSITION", confidence=0.7)
        result = route_trading_mode(
            ai_direction="bullish",
            ai_confidence=0.55,
            regime="trending",  # ATR agrees
            session="LONDON",
            range_bounds=None,
            ai_regime_assessment=assessment,
        )
        assert result.mode == "trending"
        assert "TRANSITION" in result.reason
        assert "momentum-follow" in result.reason

    # -- Regressions for the two live incidents the feature is meant to fix --

    def test_regression_2024_ath_bull_trend_mistaken_as_ranging(
        self, sample_range_bounds: RangeBounds
    ) -> None:
        """Sprint 11 review: 2024 ATH period routed to ranging (0% WR, n=18).

        Baseline: ATR regime "trending", AI direction neutral, range detected,
        guards pass → legacy Priority 2 picks ranging → disaster.
        Treatment: AI regime TREND_UP conf 0.85 at gate ON → Priority 0
        forces trending, avoiding the ranging trap.
        """
        # Baseline: gate OFF (assessment None)
        baseline = route_trading_mode(
            ai_direction="neutral",
            ai_confidence=0.3,
            regime="trending",  # ATR trending but neutral AI direction
            session="LONDON",
            range_bounds=sample_range_bounds,
            guards_passed=True,
            current_price=2360.0,  # inside Donchian range
            ai_regime_assessment=None,
        )
        # Legacy path: regime trending but price IN range → trending_suppress
        # is False (price_in_range=True), so Priority 2 fires → ranging.
        assert baseline.mode == "ranging"

        # Treatment: gate ON with confident TREND_UP
        treatment = route_trading_mode(
            ai_direction="neutral",
            ai_confidence=0.3,
            regime="trending",
            session="LONDON",
            range_bounds=sample_range_bounds,
            guards_passed=True,
            current_price=2360.0,
            ai_regime_assessment=_make_ai_assessment("TREND_UP", confidence=0.85),
        )
        assert treatment.mode == "trending"
        assert "TREND_UP" in treatment.reason

    def test_regression_monday_0200_utc_transition_stops_stack_disaster(
        self, sample_range_bounds: RangeBounds
    ) -> None:
        """2026-04-20 02:00 UTC post-mortem: 5 stacked BUY in ASIAN range.

        Baseline: ATR transitional + range_bounds + guards pass +
        ASIAN_LONDON_TRANSITION → legacy Priority 2 fires ranging →
        5 stacks of mean-reversion buys into a falling market.
        Treatment: AI regime TRANSITION conf 0.72 at gate ON → Priority 0
        forces v1_passthrough, no new range setups.
        """
        baseline = route_trading_mode(
            ai_direction="neutral",
            ai_confidence=0.2,
            regime="transitional",
            session="ASIAN_LONDON_TRANSITION",
            range_bounds=sample_range_bounds,
            guards_passed=True,
            ai_regime_assessment=None,
        )
        assert baseline.mode == "ranging"  # legacy picks ranging

        treatment = route_trading_mode(
            ai_direction="neutral",
            ai_confidence=0.2,
            regime="transitional",
            session="ASIAN_LONDON_TRANSITION",
            range_bounds=sample_range_bounds,
            guards_passed=True,
            ai_regime_assessment=_make_ai_assessment("TRANSITION", confidence=0.72),
        )
        assert treatment.mode == "v1_passthrough"
        assert "TRANSITION" in treatment.reason

    # -- Context passthrough preserved under new branch --

    def test_ai_regime_result_preserves_context_fields(
        self, sample_range_bounds: RangeBounds
    ) -> None:
        """Returned TradingMode must carry original ai_direction / regime."""
        assessment = _make_ai_assessment("TREND_UP", confidence=0.8)
        result = route_trading_mode(
            ai_direction="neutral",
            ai_confidence=0.37,
            regime="ranging",
            session="LONDON",
            range_bounds=sample_range_bounds,
            guards_passed=True,
            ai_regime_assessment=assessment,
        )
        assert result.ai_direction == "neutral"
        assert result.ai_confidence == 0.37
        assert result.regime == "ranging"
        assert result.range_bounds == sample_range_bounds
