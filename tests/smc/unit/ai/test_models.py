"""Unit tests for smc.ai.models — frozen Pydantic types."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from smc.ai.models import (
    AIRegimeAssessment,
    AnalystView,
    DebateRound,
    ExternalContext,
    JudgeVerdict,
    RegimeParams,
    RegimeState,
)


class TestRegimeParams:
    def test_frozen(self) -> None:
        params = RegimeParams(
            sl_atr_multiplier=0.75, tp1_rr=2.5, confluence_floor=0.45,
            allowed_directions=("long",), allowed_triggers=("fvg_fill_in_zone",),
            max_concurrent=3, zone_cooldown_hours=24,
            enable_ob_test=True, regime_label="Trend Up",
        )
        with pytest.raises(Exception):
            params.sl_atr_multiplier = 1.0  # type: ignore[misc]

    def test_allowed_directions_tuple(self) -> None:
        params = RegimeParams(
            sl_atr_multiplier=0.75, tp1_rr=2.5, confluence_floor=0.45,
            allowed_directions=("long", "short"), allowed_triggers=("fvg_fill_in_zone",),
            max_concurrent=3, zone_cooldown_hours=24,
            enable_ob_test=False, regime_label="Test",
        )
        assert "long" in params.allowed_directions
        assert "short" in params.allowed_directions

    def test_allowed_triggers_tuple(self) -> None:
        params = RegimeParams(
            sl_atr_multiplier=0.75, tp1_rr=2.5, confluence_floor=0.45,
            allowed_directions=("long",),
            allowed_triggers=("choch_in_zone", "fvg_fill_in_zone"),
            max_concurrent=3, zone_cooldown_hours=24,
            enable_ob_test=True, regime_label="Test",
        )
        assert "choch_in_zone" in params.allowed_triggers
        assert "fvg_fill_in_zone" in params.allowed_triggers
        assert "ob_test_rejection" not in params.allowed_triggers

    def test_enable_ob_test_flag(self) -> None:
        params_on = RegimeParams(
            sl_atr_multiplier=0.75, tp1_rr=2.5, confluence_floor=0.45,
            allowed_directions=("long",), allowed_triggers=("fvg_fill_in_zone",),
            max_concurrent=3, zone_cooldown_hours=24,
            enable_ob_test=True, regime_label="Test",
        )
        params_off = RegimeParams(
            sl_atr_multiplier=0.75, tp1_rr=2.5, confluence_floor=0.45,
            allowed_directions=("long",), allowed_triggers=("fvg_fill_in_zone",),
            max_concurrent=3, zone_cooldown_hours=24,
            enable_ob_test=False, regime_label="Test",
        )
        assert params_on.enable_ob_test is True
        assert params_off.enable_ob_test is False


class TestAIRegimeAssessment:
    def test_frozen(self) -> None:
        now = datetime.now(tz=timezone.utc)
        params = RegimeParams(
            sl_atr_multiplier=0.75, tp1_rr=2.5, confluence_floor=0.45,
            allowed_directions=("long",), allowed_triggers=("fvg_fill_in_zone",),
            max_concurrent=3, zone_cooldown_hours=24,
            enable_ob_test=True, regime_label="Trend Up",
        )
        assessment = AIRegimeAssessment(
            regime="TREND_UP", trend_direction="bullish", confidence=0.85,
            param_preset=params, reasoning="Strong uptrend",
            assessed_at=now, source="ai_debate", cost_usd=0.78,
        )
        assert assessment.regime == "TREND_UP"
        assert assessment.source == "ai_debate"
        with pytest.raises(Exception):
            assessment.regime = "TREND_DOWN"  # type: ignore[misc]

    def test_all_sources(self) -> None:
        now = datetime.now(tz=timezone.utc)
        params = RegimeParams(
            sl_atr_multiplier=0.85, tp1_rr=2.0, confluence_floor=0.55,
            allowed_directions=("long", "short"), allowed_triggers=("fvg_fill_in_zone",),
            max_concurrent=2, zone_cooldown_hours=24,
            enable_ob_test=False, regime_label="Transition",
        )
        for source in ("ai_debate", "atr_fallback", "default"):
            a = AIRegimeAssessment(
                regime="TRANSITION", trend_direction="neutral", confidence=0.5,
                param_preset=params, reasoning="test",
                assessed_at=now, source=source, cost_usd=0.0,  # type: ignore[arg-type]
            )
            assert a.source == source

    def test_zero_cost_for_fallback(self) -> None:
        now = datetime.now(tz=timezone.utc)
        params = RegimeParams(
            sl_atr_multiplier=0.85, tp1_rr=2.0, confluence_floor=0.55,
            allowed_directions=("long", "short"), allowed_triggers=("fvg_fill_in_zone",),
            max_concurrent=2, zone_cooldown_hours=24,
            enable_ob_test=False, regime_label="Transition",
        )
        a = AIRegimeAssessment(
            regime="TRANSITION", trend_direction="neutral", confidence=0.5,
            param_preset=params, reasoning="ATR fallback",
            assessed_at=now, source="atr_fallback", cost_usd=0.0,
        )
        assert a.cost_usd == 0.0


class TestRegimeState:
    def test_frozen(self) -> None:
        now = datetime.now(tz=timezone.utc)
        state = RegimeState(
            current_regime="TREND_UP", regime_since=now, bars_in_regime=5,
            previous_regime=None, transition_confidence=0.85,
        )
        with pytest.raises(Exception):
            state.bars_in_regime = 6  # type: ignore[misc]

    def test_default_consecutive_different_count(self) -> None:
        now = datetime.now(tz=timezone.utc)
        state = RegimeState(
            current_regime="TREND_UP", regime_since=now, bars_in_regime=0,
            previous_regime=None, transition_confidence=0.85,
        )
        assert state.consecutive_different_count == 0

    def test_with_previous_regime(self) -> None:
        now = datetime.now(tz=timezone.utc)
        state = RegimeState(
            current_regime="CONSOLIDATION", regime_since=now, bars_in_regime=0,
            previous_regime="TREND_UP", transition_confidence=0.7,
            consecutive_different_count=0,
        )
        assert state.previous_regime == "TREND_UP"
        assert state.current_regime == "CONSOLIDATION"

    def test_hold_pattern_creates_new_instance(self) -> None:
        """Simulates the HOLD case — new instance with incremented counters."""
        now = datetime.now(tz=timezone.utc)
        old = RegimeState(
            current_regime="TREND_UP", regime_since=now, bars_in_regime=2,
            previous_regime=None, transition_confidence=0.85,
            consecutive_different_count=0,
        )
        # Simulate HOLD with AI disagreement
        new = RegimeState(
            current_regime=old.current_regime,
            regime_since=old.regime_since,
            bars_in_regime=old.bars_in_regime + 1,
            previous_regime=old.previous_regime,
            transition_confidence=old.transition_confidence,
            consecutive_different_count=old.consecutive_different_count + 1,
        )
        assert new.bars_in_regime == 3
        assert new.consecutive_different_count == 1
        assert new.current_regime == "TREND_UP"


class TestAnalystView:
    def test_all_domains(self) -> None:
        for domain in ("trend", "zone", "macro", "risk"):
            view = AnalystView(
                domain=domain,  # type: ignore[arg-type]
                regime_vote="TREND_UP", confidence=0.8, reasoning="test",
            )
            assert view.domain == domain

    def test_frozen(self) -> None:
        view = AnalystView(
            domain="trend", regime_vote="TREND_UP",
            confidence=0.8, reasoning="test",
        )
        with pytest.raises(Exception):
            view.confidence = 0.5  # type: ignore[misc]


class TestDebateRound:
    def test_creation(self) -> None:
        r = DebateRound(
            round_num=1,
            bull_argument="Strong HH pattern",
            bear_argument="Overextended ATR",
        )
        assert r.round_num == 1
        assert r.bull_argument == "Strong HH pattern"


class TestJudgeVerdict:
    def test_decisive_factors_tuple(self) -> None:
        v = JudgeVerdict(
            regime="TREND_UP", confidence=0.85,
            decisive_factors=("d1_sma50_direction", "d1_higher_highs"),
            reasoning="Clear bullish structure",
        )
        assert len(v.decisive_factors) == 2
        assert "d1_sma50_direction" in v.decisive_factors

    def test_frozen(self) -> None:
        v = JudgeVerdict(
            regime="TREND_UP", confidence=0.85,
            decisive_factors=("d1_sma50_direction",),
            reasoning="test",
        )
        with pytest.raises(Exception):
            v.regime = "CONSOLIDATION"  # type: ignore[misc]


class TestExternalContext:
    def test_minimal_creation(self) -> None:
        now = datetime.now(tz=timezone.utc)
        ctx = ExternalContext(
            dxy_direction="flat", fetched_at=now,
            source_quality="unavailable",
        )
        assert ctx.dxy_value is None
        assert ctx.vix_level is None
        assert ctx.source_quality == "unavailable"

    def test_full_creation(self) -> None:
        now = datetime.now(tz=timezone.utc)
        ctx = ExternalContext(
            dxy_direction="weakening", dxy_value=103.5,
            vix_level=18.5, vix_regime="normal",
            real_rate_10y=2.1, cot_net_spec=150000.0,
            central_bank_stance="dovish",
            fetched_at=now, source_quality="live",
        )
        assert ctx.dxy_value == 103.5
        assert ctx.vix_regime == "normal"
        assert ctx.central_bank_stance == "dovish"

    def test_frozen(self) -> None:
        now = datetime.now(tz=timezone.utc)
        ctx = ExternalContext(
            dxy_direction="flat", fetched_at=now,
            source_quality="cached",
        )
        with pytest.raises(Exception):
            ctx.dxy_direction = "weakening"  # type: ignore[misc]
