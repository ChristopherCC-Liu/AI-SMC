"""Unit tests for smc.ai.macro_layer — Alt-B macro overlay MVP.

Covers:
    - MacroLayer.compute_macro_bias aggregation math
    - Clamping to ±0.3
    - Graceful degradation when sources fail
    - DXY source wiring through ExternalContextFetcher
    - Direction classification thresholds

COT and TIPS yield sources are stubbed (NotImplementedError) in MVP;
dedicated tests land with those implementations (plan §2 / §3).
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from smc.ai.macro_layer import MacroBias, MacroLayer
from smc.ai.models import ExternalContext


def _make_external_context(
    dxy_direction: str = "flat",
    dxy_value: float | None = 105.0,
    quality: str = "live",
) -> ExternalContext:
    """Helper to build a valid ExternalContext for patching."""
    return ExternalContext(
        dxy_direction=dxy_direction,  # type: ignore[arg-type]
        dxy_value=dxy_value,
        vix_level=18.0,
        vix_regime="normal",
        real_rate_10y=2.1,
        cot_net_spec=None,
        central_bank_stance=None,
        fetched_at=datetime.now(timezone.utc),
        source_quality=quality,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# MacroLayer.compute_macro_bias — aggregation + edge cases
# ---------------------------------------------------------------------------


class TestComputeMacroBias:
    def test_all_sources_fail_returns_neutral(self, tmp_path) -> None:
        """When all 3 sources fail (COT/yield NotImplemented, DXY unavailable)
        the layer must return neutral bias with 0 sources, never raise."""
        layer = MacroLayer(cache_dir=tmp_path)
        with patch.object(
            layer._external,
            "fetch",
            return_value=_make_external_context(quality="unavailable", dxy_value=None),
        ):
            bias = layer.compute_macro_bias("XAUUSD")

        assert bias.sources_available == 0
        assert bias.total_bias == 0.0
        assert bias.direction == "neutral"
        assert bias.cot_bias == 0.0
        assert bias.yield_bias == 0.0
        assert bias.dxy_bias == 0.0

    def test_dxy_weakening_contributes_positive_bias(self, tmp_path) -> None:
        """USD weakening → gold bullish → positive bias."""
        layer = MacroLayer(cache_dir=tmp_path)
        with patch.object(
            layer._external,
            "fetch",
            return_value=_make_external_context(dxy_direction="weakening"),
        ):
            bias = layer.compute_macro_bias("XAUUSD")

        assert bias.sources_available == 1  # only DXY implemented
        assert bias.dxy_bias == 0.05
        assert bias.total_bias == 0.05
        # 0.05 is exactly at the direction threshold → neutral
        assert bias.direction == "neutral"

    def test_dxy_strengthening_contributes_negative_bias(self, tmp_path) -> None:
        """USD strengthening → gold bearish → negative bias."""
        layer = MacroLayer(cache_dir=tmp_path)
        with patch.object(
            layer._external,
            "fetch",
            return_value=_make_external_context(dxy_direction="strengthening"),
        ):
            bias = layer.compute_macro_bias("XAUUSD")

        assert bias.dxy_bias == -0.05
        assert bias.total_bias == -0.05
        assert bias.direction == "neutral"  # at threshold, not beyond

    def test_dxy_flat_contributes_zero(self, tmp_path) -> None:
        """Flat DXY → zero bias but source is still counted as available."""
        layer = MacroLayer(cache_dir=tmp_path)
        with patch.object(
            layer._external,
            "fetch",
            return_value=_make_external_context(dxy_direction="flat"),
        ):
            bias = layer.compute_macro_bias("XAUUSD")

        assert bias.dxy_bias == 0.0
        assert bias.total_bias == 0.0
        assert bias.direction == "neutral"
        assert bias.sources_available == 1  # DXY fetched successfully, even if 0

    def test_result_is_frozen_dataclass(self, tmp_path) -> None:
        """MacroBias must be immutable per project coding-style rules."""
        layer = MacroLayer(cache_dir=tmp_path)
        with patch.object(
            layer._external,
            "fetch",
            return_value=_make_external_context(dxy_direction="flat"),
        ):
            bias = layer.compute_macro_bias("XAUUSD")

        with pytest.raises((AttributeError, Exception)):
            bias.total_bias = 0.99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# DXY source — direct tests
# ---------------------------------------------------------------------------


class TestDXYBias:
    def test_dxy_bias_raises_when_fetcher_unavailable(self, tmp_path) -> None:
        """When ExternalContextFetcher reports unavailable, _dxy_bias raises."""
        layer = MacroLayer(cache_dir=tmp_path)
        with patch.object(
            layer._external,
            "fetch",
            return_value=_make_external_context(quality="unavailable", dxy_value=None),
        ):
            with pytest.raises(RuntimeError, match="DXY data unavailable"):
                layer._dxy_bias()

    def test_safe_dxy_bias_swallows_runtime_error(self, tmp_path) -> None:
        """_safe_dxy_bias must never raise — returns None on error."""
        layer = MacroLayer(cache_dir=tmp_path)
        with patch.object(
            layer._external,
            "fetch",
            return_value=_make_external_context(quality="unavailable", dxy_value=None),
        ):
            result = layer._safe_dxy_bias()
        assert result is None


# ---------------------------------------------------------------------------
# Stubbed sources — must not crash the aggregation
# ---------------------------------------------------------------------------


class TestStubbedSourcesGracefulDegrade:
    def test_cot_bias_raises_not_implemented(self, tmp_path) -> None:
        """COT source is intentionally stubbed in MVP."""
        layer = MacroLayer(cache_dir=tmp_path)
        with pytest.raises(NotImplementedError):
            layer._cot_bias("XAUUSD")

    def test_yield_bias_raises_not_implemented(self, tmp_path) -> None:
        """TIPS yield source is intentionally stubbed in MVP."""
        layer = MacroLayer(cache_dir=tmp_path)
        with pytest.raises(NotImplementedError):
            layer._yield_bias()

    def test_safe_cot_returns_none(self, tmp_path) -> None:
        """Safe wrapper converts NotImplementedError to None cleanly."""
        layer = MacroLayer(cache_dir=tmp_path)
        assert layer._safe_cot_bias("XAUUSD") is None

    def test_safe_yield_returns_none(self, tmp_path) -> None:
        layer = MacroLayer(cache_dir=tmp_path)
        assert layer._safe_yield_bias() is None


# ---------------------------------------------------------------------------
# Clamping — ensure ±0.3 cap holds even if future sources overshoot
# ---------------------------------------------------------------------------


class TestClamping:
    def test_total_clamped_to_positive_max(self, tmp_path, monkeypatch) -> None:
        """Even if future sources all return 0.5, total stays at +0.3."""
        layer = MacroLayer(cache_dir=tmp_path)
        monkeypatch.setattr(layer, "_safe_cot_bias", lambda _: 0.5)
        monkeypatch.setattr(layer, "_safe_yield_bias", lambda: 0.5)
        monkeypatch.setattr(layer, "_safe_dxy_bias", lambda: 0.5)

        bias = layer.compute_macro_bias("XAUUSD")
        assert bias.total_bias == 0.3
        assert bias.sources_available == 3
        assert bias.direction == "bullish"

    def test_total_clamped_to_negative_max(self, tmp_path, monkeypatch) -> None:
        layer = MacroLayer(cache_dir=tmp_path)
        monkeypatch.setattr(layer, "_safe_cot_bias", lambda _: -0.5)
        monkeypatch.setattr(layer, "_safe_yield_bias", lambda: -0.5)
        monkeypatch.setattr(layer, "_safe_dxy_bias", lambda: -0.5)

        bias = layer.compute_macro_bias("XAUUSD")
        assert bias.total_bias == -0.3
        assert bias.direction == "bearish"

    def test_mixed_sources_sum_correctly(self, tmp_path, monkeypatch) -> None:
        """COT bullish + yield bearish + DXY bullish → small positive net."""
        layer = MacroLayer(cache_dir=tmp_path)
        monkeypatch.setattr(layer, "_safe_cot_bias", lambda _: 0.10)
        monkeypatch.setattr(layer, "_safe_yield_bias", lambda: -0.05)
        monkeypatch.setattr(layer, "_safe_dxy_bias", lambda: 0.05)

        bias = layer.compute_macro_bias("XAUUSD")
        assert abs(bias.total_bias - 0.10) < 1e-9
        assert bias.sources_available == 3
        assert bias.direction == "bullish"  # 0.10 > 0.05 threshold


# ---------------------------------------------------------------------------
# Direction classification thresholds
# ---------------------------------------------------------------------------


class TestDirectionClassification:
    def test_direction_bullish_above_threshold(self, tmp_path, monkeypatch) -> None:
        layer = MacroLayer(cache_dir=tmp_path)
        monkeypatch.setattr(layer, "_safe_cot_bias", lambda _: 0.08)
        monkeypatch.setattr(layer, "_safe_yield_bias", lambda: None)
        monkeypatch.setattr(layer, "_safe_dxy_bias", lambda: None)

        bias = layer.compute_macro_bias("XAUUSD")
        assert bias.direction == "bullish"

    def test_direction_bearish_below_threshold(self, tmp_path, monkeypatch) -> None:
        layer = MacroLayer(cache_dir=tmp_path)
        monkeypatch.setattr(layer, "_safe_cot_bias", lambda _: -0.08)
        monkeypatch.setattr(layer, "_safe_yield_bias", lambda: None)
        monkeypatch.setattr(layer, "_safe_dxy_bias", lambda: None)

        bias = layer.compute_macro_bias("XAUUSD")
        assert bias.direction == "bearish"

    def test_direction_neutral_at_threshold(self, tmp_path, monkeypatch) -> None:
        """Boundary: exactly 0.05 should be neutral (strict >)."""
        layer = MacroLayer(cache_dir=tmp_path)
        monkeypatch.setattr(layer, "_safe_cot_bias", lambda _: 0.05)
        monkeypatch.setattr(layer, "_safe_yield_bias", lambda: None)
        monkeypatch.setattr(layer, "_safe_dxy_bias", lambda: None)

        bias = layer.compute_macro_bias("XAUUSD")
        assert bias.direction == "neutral"
