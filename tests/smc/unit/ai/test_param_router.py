"""Unit tests for smc.ai.param_router — regime → parameter preset routing."""

from __future__ import annotations

import pytest

from smc.ai.models import MarketRegimeAI, RegimeParams
from smc.ai.param_router import PRESETS, route


class TestRoute:
    def test_all_five_regimes_exist(self) -> None:
        regimes: list[MarketRegimeAI] = [
            "TREND_UP", "TREND_DOWN", "CONSOLIDATION", "TRANSITION", "ATH_BREAKOUT",
        ]
        for regime in regimes:
            params = route(regime)
            assert isinstance(params, RegimeParams)

    def test_unknown_regime_raises(self) -> None:
        with pytest.raises(KeyError):
            route("INVALID_REGIME")  # type: ignore[arg-type]

    def test_presets_are_frozen(self) -> None:
        for regime, params in PRESETS.items():
            with pytest.raises(Exception):
                params.sl_atr_multiplier = 999.0  # type: ignore[misc]

    def test_route_returns_same_object(self) -> None:
        """route() should return the module-level constant, not a copy."""
        assert route("TREND_UP") is PRESETS["TREND_UP"]


class TestTrendUpPreset:
    def test_direction_long_only(self) -> None:
        params = route("TREND_UP")
        assert params.allowed_directions == ("long",)

    def test_all_triggers_allowed(self) -> None:
        """Trending regimes allow all 4 trigger types."""
        triggers = route("TREND_UP").allowed_triggers
        assert "choch_in_zone" in triggers
        assert "fvg_fill_in_zone" in triggers
        assert "ob_test_rejection" in triggers
        assert "bos_in_zone" in triggers

    def test_ob_test_enabled(self) -> None:
        assert route("TREND_UP").enable_ob_test is True

    def test_sl_atr_multiplier(self) -> None:
        assert route("TREND_UP").sl_atr_multiplier == 0.75  # base

    def test_tp1_rr_widened(self) -> None:
        """Round 4 v5: TREND_UP widens TP to 3.0 to ride momentum."""
        assert route("TREND_UP").tp1_rr == 3.0

    def test_max_concurrent(self) -> None:
        assert route("TREND_UP").max_concurrent == 3

    def test_confluence_floor(self) -> None:
        assert route("TREND_UP").confluence_floor == 0.45


class TestTrendDownPreset:
    def test_direction_short_only(self) -> None:
        assert route("TREND_DOWN").allowed_directions == ("short",)

    def test_ob_test_enabled(self) -> None:
        assert route("TREND_DOWN").enable_ob_test is True

    def test_mirrors_trend_up_except_direction(self) -> None:
        up = route("TREND_UP")
        down = route("TREND_DOWN")
        assert up.sl_atr_multiplier == down.sl_atr_multiplier
        assert up.tp1_rr == down.tp1_rr
        assert up.confluence_floor == down.confluence_floor
        assert up.max_concurrent == down.max_concurrent
        assert up.zone_cooldown_hours == down.zone_cooldown_hours


class TestConsolidationPreset:
    def test_both_directions(self) -> None:
        assert route("CONSOLIDATION").allowed_directions == ("long", "short")

    def test_ob_test_disabled(self) -> None:
        assert route("CONSOLIDATION").enable_ob_test is False

    def test_all_triggers_allowed(self) -> None:
        """v7c fix: CONSOLIDATION allows all triggers (choch has 40% WR in ranging)."""
        assert len(route("CONSOLIDATION").allowed_triggers) == 5

    def test_tightened_sl(self) -> None:
        """Round 4 v5: CONSOLIDATION tightens SL to 0.6 for range scalping."""
        assert route("CONSOLIDATION").sl_atr_multiplier == 0.6

    def test_uses_base_concurrent(self) -> None:
        """CONSOLIDATION uses Sprint 5 base concurrent (unchanged in v5)."""
        assert route("CONSOLIDATION").max_concurrent == 3

    def test_uses_base_confluence_floor(self) -> None:
        """CONSOLIDATION uses Sprint 5 base floor (unchanged in v5)."""
        assert route("CONSOLIDATION").confluence_floor == 0.45

    def test_uses_base_cooldown(self) -> None:
        """CONSOLIDATION uses Sprint 5 base cooldown (unchanged in v5)."""
        assert route("CONSOLIDATION").zone_cooldown_hours == 24

    def test_tightened_tp(self) -> None:
        """Round 4 v5: CONSOLIDATION tightens TP to 2.0 for quick scalps."""
        assert route("CONSOLIDATION").tp1_rr == 2.0


class TestTransitionPreset:
    def test_both_directions(self) -> None:
        assert route("TRANSITION").allowed_directions == ("long", "short")

    def test_ob_test_disabled(self) -> None:
        assert route("TRANSITION").enable_ob_test is False

    def test_allowed_triggers_no_choch_no_ob(self) -> None:
        """Transitional allows fvg_fill + bos, but not choch or ob_test."""
        triggers = route("TRANSITION").allowed_triggers
        assert "fvg_fill_in_zone" in triggers
        assert "bos_in_zone" in triggers
        assert "choch_in_zone" not in triggers
        assert "ob_test_rejection" not in triggers

    def test_base_params_except_confluence(self) -> None:
        """Transition uses Sprint 5 base params, only confluence raised to 0.55."""
        p = route("TRANSITION")
        assert p.sl_atr_multiplier == 0.75
        assert p.tp1_rr == 2.5
        assert p.confluence_floor == 0.55
        assert p.max_concurrent == 3
        assert p.zone_cooldown_hours == 24


class TestATHBreakoutPreset:
    def test_long_only(self) -> None:
        assert route("ATH_BREAKOUT").allowed_directions == ("long",)

    def test_ob_test_disabled(self) -> None:
        assert route("ATH_BREAKOUT").enable_ob_test is False

    def test_all_triggers_allowed(self) -> None:
        """ATH breakout allows all triggers (v7 fix: uses base triggers)."""
        triggers = route("ATH_BREAKOUT").allowed_triggers
        assert len(triggers) == 5

    def test_extended_tp(self) -> None:
        """Round 4 v5: ATH extends TP to 3.5 to capture breakout impulse."""
        assert route("ATH_BREAKOUT").tp1_rr == 3.5

    def test_widened_sl(self) -> None:
        """Round 4 v5: ATH widens SL to 0.9 to survive post-breakout vol."""
        assert route("ATH_BREAKOUT").sl_atr_multiplier == 0.9

    def test_uses_base_cooldown(self) -> None:
        """ATH uses Sprint 5 base cooldown (unchanged in v5)."""
        assert route("ATH_BREAKOUT").zone_cooldown_hours == 24


class TestPresetConsistency:
    def test_all_presets_have_labels(self) -> None:
        for regime, params in PRESETS.items():
            assert params.regime_label, f"{regime} has no label"

    def test_all_presets_have_allowed_triggers(self) -> None:
        for regime, params in PRESETS.items():
            assert len(params.allowed_triggers) >= 1, f"{regime} has no allowed triggers"

    def test_fvg_fill_always_allowed(self) -> None:
        """fvg_fill_in_zone is the safest trigger — allowed in every regime."""
        for regime, params in PRESETS.items():
            assert "fvg_fill_in_zone" in params.allowed_triggers, (
                f"{regime} missing fvg_fill_in_zone"
            )

    def test_trending_has_most_triggers(self) -> None:
        """Trending regimes should allow the most trigger types."""
        trend_count = len(route("TREND_UP").allowed_triggers)
        for regime in ("CONSOLIDATION", "TRANSITION"):
            assert trend_count >= len(route(regime).allowed_triggers)

    def test_all_regimes_share_base_concurrent_cooldown(self) -> None:
        """All regimes share concurrent/cooldown (only SL/TP differentiated in v5)."""
        for regime, params in PRESETS.items():
            assert params.max_concurrent == 3, f"{regime} concurrent != base"
            assert params.zone_cooldown_hours == 24, f"{regime} cooldown != base"

    def test_regime_specific_sl_tp_matrix(self) -> None:
        """Round 4 v5: SL/TP differentiated per regime.

        TRANSITION is the neutral baseline; trends widen TP, consolidation
        tightens both, ATH widens both. Deltas kept conservative (±20%
        SL, ±20-40% TP) until backtest validation extends the deltas.
        """
        expected: dict[MarketRegimeAI, tuple[float, float]] = {
            "TREND_UP":      (0.75, 3.0),
            "TREND_DOWN":    (0.75, 3.0),
            "CONSOLIDATION": (0.60, 2.0),
            "TRANSITION":    (0.75, 2.5),
            "ATH_BREAKOUT":  (0.90, 3.5),
        }
        for regime, (sl, tp) in expected.items():
            params = route(regime)
            assert params.sl_atr_multiplier == sl, f"{regime} sl_mult"
            assert params.tp1_rr == tp, f"{regime} tp1_rr"

    def test_only_transition_has_raised_floor(self) -> None:
        """Only TRANSITION has confluence_floor > 0.45 (raised to 0.55)."""
        for regime, params in PRESETS.items():
            if regime == "TRANSITION":
                assert params.confluence_floor == 0.55
            else:
                assert params.confluence_floor == 0.45, f"{regime} floor != base"


# ---------------------------------------------------------------------------
# Round 5 A-track Task #8 — Regime-dynamic trailing SL
# ---------------------------------------------------------------------------


class TestTrailParams:
    """Round 5 A-track Task #8: get_trail_params per-regime presets."""

    def test_trail_params_is_namedtuple(self) -> None:
        from smc.ai.param_router import TrailParams, get_trail_params
        p = get_trail_params("TREND_UP")
        assert isinstance(p, TrailParams)
        # NamedTuple supports both index + field access.
        assert p[0] == p.activate_r
        assert p[1] == p.distance_r

    def test_trend_up_locks_fast(self) -> None:
        from smc.ai.param_router import get_trail_params
        p = get_trail_params("TREND_UP")
        assert p.activate_r == 0.3
        assert p.distance_r == 0.5

    def test_trend_down_mirrors_trend_up(self) -> None:
        from smc.ai.param_router import get_trail_params
        up = get_trail_params("TREND_UP")
        down = get_trail_params("TREND_DOWN")
        assert up == down  # symmetric

    def test_consolidation_tight_scalp(self) -> None:
        from smc.ai.param_router import get_trail_params
        p = get_trail_params("CONSOLIDATION")
        assert p.activate_r == 0.5
        assert p.distance_r == 0.3

    def test_transition_baseline(self) -> None:
        from smc.ai.param_router import get_trail_params
        p = get_trail_params("TRANSITION")
        assert p.activate_r == 0.5
        assert p.distance_r == 0.5

    def test_ath_breakout_gives_room(self) -> None:
        """ATH breakouts should have the widest activate + distance."""
        from smc.ai.param_router import get_trail_params
        p = get_trail_params("ATH_BREAKOUT")
        assert p.activate_r == 0.8
        assert p.distance_r == 0.7

    def test_all_five_regimes_have_presets(self) -> None:
        from smc.ai.param_router import TRAIL_PRESETS
        expected: tuple[MarketRegimeAI, ...] = (
            "TREND_UP", "TREND_DOWN", "CONSOLIDATION", "TRANSITION", "ATH_BREAKOUT",
        )
        assert set(TRAIL_PRESETS.keys()) == set(expected)

    def test_trail_values_in_sensible_range(self) -> None:
        """All activate_r / distance_r ∈ (0, 1) — > 1R is never desirable."""
        from smc.ai.param_router import TRAIL_PRESETS
        for regime, params in TRAIL_PRESETS.items():
            assert 0.0 < params.activate_r <= 1.0, f"{regime} activate_r"
            assert 0.0 < params.distance_r <= 1.0, f"{regime} distance_r"

    def test_get_trail_params_is_total_over_regimes(self) -> None:
        """get_trail_params must succeed for every MarketRegimeAI literal."""
        from smc.ai.param_router import get_trail_params
        regimes: list[MarketRegimeAI] = [
            "TREND_UP", "TREND_DOWN", "CONSOLIDATION", "TRANSITION", "ATH_BREAKOUT",
        ]
        for r in regimes:
            p = get_trail_params(r)
            assert p.activate_r > 0.0
            assert p.distance_r > 0.0

    def test_unknown_regime_raises(self) -> None:
        from smc.ai.param_router import get_trail_params
        with pytest.raises(KeyError):
            get_trail_params("INVALID")  # type: ignore[arg-type]

    def test_trail_params_is_immutable(self) -> None:
        """NamedTuple is immutable — can't reassign fields."""
        from smc.ai.param_router import get_trail_params
        p = get_trail_params("TREND_UP")
        with pytest.raises(AttributeError):
            p.activate_r = 0.9  # type: ignore[misc]
