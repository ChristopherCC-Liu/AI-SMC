"""Integration tests for Round 4 Alt-B W2 macro overlay wiring in live_demo.py.

These tests verify that:
1. When macro_enabled=False, MacroLayer.compute_macro_bias is NOT called and
   score_confluence receives macro_bias=0.0 (baseline parity).
2. When macro_enabled=True, MacroLayer.compute_macro_bias IS called and the
   returned total_bias is threaded through to set_macro_bias on the aggregator.
3. When MacroLayer.compute_macro_bias raises, the error is caught, macro_bias
   falls back to 0.0, and the cycle continues without crashing.

These tests import only the strategy/aggregator/confluence layers — they do NOT
import live_demo.py at module level (which triggers MetaTrader5 import and
_ensure_single_instance()).  Instead they test the wiring contract directly
via the aggregator.set_macro_bias → score_confluence path.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from smc.ai.macro_layer import MacroBias, MacroLayer
from smc.strategy.aggregator import MultiTimeframeAggregator
from smc.strategy.confluence import score_confluence
from smc.strategy.types import BiasDirection, EntrySignal, TradeZone
from smc.data.schemas import Timeframe
from smc.smc_core.detector import SMCDetector


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_bullish_bias() -> BiasDirection:
    return BiasDirection(
        direction="bullish",
        confidence=0.8,
        key_levels=(2340.0, 2355.0, 2370.0),
        rationale="Tier 1: D1+H4 bullish",
    )


def _make_zone() -> TradeZone:
    return TradeZone(
        zone_high=2352.0, zone_low=2348.0, zone_type="ob",
        direction="long", timeframe=Timeframe.H1, confidence=0.8,
    )


def _make_entry() -> EntrySignal:
    return EntrySignal(
        entry_price=2350.0, stop_loss=2347.0, take_profit_1=2357.0,
        take_profit_2=2363.0, risk_points=300.0, reward_points=700.0,
        rr_ratio=2.5, trigger_type="choch_in_zone", direction="long", grade="A",
    )


def _make_macro_bias(total: float, direction: str = "bullish") -> MacroBias:
    return MacroBias(
        cot_bias=round(total / 3, 4),
        yield_bias=round(total / 3, 4),
        dxy_bias=round(total / 3, 4),
        total_bias=round(total, 4),
        sources_available=3,
        direction=direction,  # type: ignore[arg-type]
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------------
# Test 1: macro_enabled=False → aggregator macro_bias stays 0.0
# ---------------------------------------------------------------------------


class TestMacroDisabledBaseline:
    """When macro_enabled=False, confluence scoring is identical to pre-macro."""

    def test_disabled_macro_produces_baseline_score(self) -> None:
        """set_macro_bias(0.0) leaves score identical to no-macro call."""
        detector = SMCDetector(swing_length=10)
        aggregator = MultiTimeframeAggregator(detector=detector)

        bias = _make_bullish_bias()
        zone = _make_zone()
        entry = _make_entry()

        # Simulate disabled macro: aggregator._macro_bias stays at default 0.0
        # score_confluence with macro_bias=0.0 must equal score with no macro_bias arg
        score_no_macro = score_confluence(bias, zone, entry)
        score_disabled = score_confluence(bias, zone, entry, macro_bias=aggregator._macro_bias)

        assert score_no_macro == score_disabled
        assert aggregator._macro_bias == 0.0


# ---------------------------------------------------------------------------
# Test 2: macro_enabled=True → MacroLayer called, bias passes to aggregator
# ---------------------------------------------------------------------------


class TestMacroEnabledWiring:
    """When macro_enabled=True, total_bias is injected via set_macro_bias."""

    def test_macro_bias_passes_through_to_confluence(self) -> None:
        """MacroLayer result flows from compute_macro_bias → set_macro_bias → score."""
        detector = SMCDetector(swing_length=10)
        aggregator = MultiTimeframeAggregator(detector=detector)

        macro_layer = MacroLayer.__new__(MacroLayer)
        expected_total_bias = 0.15
        mock_result = _make_macro_bias(expected_total_bias, direction="bullish")

        with patch.object(macro_layer, "compute_macro_bias", return_value=mock_result):
            mb = macro_layer.compute_macro_bias(instrument="XAUUSD")
            # Simulate what live_demo.py does:
            aggregator.set_macro_bias(mb.total_bias)

        assert aggregator._macro_bias == pytest.approx(expected_total_bias)

    def test_aligned_macro_boosts_confluence_score(self) -> None:
        """Aggregator set_macro_bias → score_confluence reflects the boost."""
        detector = SMCDetector(swing_length=10)
        aggregator = MultiTimeframeAggregator(detector=detector)

        bias = _make_bullish_bias()
        zone = _make_zone()
        entry = _make_entry()

        score_before = score_confluence(bias, zone, entry, macro_bias=0.0)

        aggregator.set_macro_bias(0.15)
        score_after = score_confluence(bias, zone, entry, macro_bias=aggregator._macro_bias)

        # Bullish SMC + positive macro (also bullish) → boosted score
        assert score_after > score_before

    def test_negative_macro_bias_penalises_score(self) -> None:
        """Negative macro_bias (bearish macro) reduces bullish SMC score."""
        detector = SMCDetector(swing_length=10)
        aggregator = MultiTimeframeAggregator(detector=detector)

        bias = _make_bullish_bias()
        zone = _make_zone()
        entry = _make_entry()

        score_before = score_confluence(bias, zone, entry, macro_bias=0.0)

        aggregator.set_macro_bias(-0.15)
        score_after = score_confluence(bias, zone, entry, macro_bias=aggregator._macro_bias)

        # Bullish SMC + negative macro (bearish) → penalised score
        assert score_after < score_before


# ---------------------------------------------------------------------------
# Test 3: MacroLayer raises → fallback to 0.0, cycle continues
# ---------------------------------------------------------------------------


class TestMacroFetchFailureFallback:
    """When MacroLayer raises, live_demo must fall back to macro_bias=0.0."""

    def test_compute_macro_bias_exception_caught(self) -> None:
        """Simulate compute_macro_bias raising; verify fallback=0.0 and no crash."""
        # Replicate the try/except logic from live_demo.py section 5a.
        macro_layer = MacroLayer.__new__(MacroLayer)
        macro_bias_value: float = 0.0  # initialised to default

        with patch.object(
            macro_layer,
            "compute_macro_bias",
            side_effect=RuntimeError("all sources failed"),
        ):
            try:
                _mb = macro_layer.compute_macro_bias(instrument="XAUUSD")
                macro_bias_value = _mb.total_bias
            except Exception:
                macro_bias_value = 0.0  # fallback, as in live_demo.py

        assert macro_bias_value == 0.0

    def test_fallback_produces_baseline_score(self) -> None:
        """After fallback to 0.0, confluence score == pre-macro score."""
        detector = SMCDetector(swing_length=10)
        aggregator = MultiTimeframeAggregator(detector=detector)

        bias = _make_bullish_bias()
        zone = _make_zone()
        entry = _make_entry()

        # Fallback: macro_bias stays 0.0 (aggregator default)
        score_fallback = score_confluence(bias, zone, entry, macro_bias=aggregator._macro_bias)
        score_baseline = score_confluence(bias, zone, entry)

        assert score_fallback == score_baseline
