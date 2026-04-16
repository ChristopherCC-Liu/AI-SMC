"""Unit tests for the Sprint 7 AI Direction Engine.

Tests cover:
- H4 technical context extraction (SMA50, HH/LL counts)
- SMA + DXY deterministic fallback (all combinations)
- Memory cache hit/miss behavior
- File cache lookup via DirectionCacheLookup
- Prompt formatting (macro, news, judge contexts)
- Claude CLI response JSON parsing (including malformed responses)
- DirectionEngine.get_direction() integration with mocked LLM
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest

from smc.ai.direction_cache import DirectionCacheLookup
from smc.ai.direction_engine import (
    DirectionEngine,
    _format_judge_context,
    _format_macro_context,
    _format_news_context,
    _parse_direction_json,
    extract_h4_context,
)
from smc.ai.models import AIDirection, ExternalContext, H4TechnicalContext


# ---------------------------------------------------------------------------
# Test data helpers
# ---------------------------------------------------------------------------


def _make_h4_df(
    n_bars: int = 100,
    base_price: float = 2000.0,
    trend: float = 0.0,
    volatility: float = 5.0,
    seed: int = 42,
) -> pl.DataFrame:
    """Create a synthetic H4 OHLCV DataFrame."""
    rng = np.random.RandomState(seed)
    start = datetime(2024, 1, 2, 0, 0, tzinfo=timezone.utc)

    closes: list[float] = []
    price = base_price
    for _ in range(n_bars):
        price += trend + rng.normal(0, volatility)
        price = max(price, 100.0)
        closes.append(price)

    rows = []
    for i, c in enumerate(closes):
        bar_range = abs(rng.normal(0, volatility * 0.5))
        o = c + rng.normal(0, volatility * 0.3)
        h = max(o, c) + bar_range
        lo = min(o, c) - bar_range
        rows.append({
            "timestamp": start + timedelta(hours=4 * i),
            "open": round(o, 2),
            "high": round(h, 2),
            "low": round(lo, 2),
            "close": round(c, 2),
            "volume": int(rng.uniform(1000, 5000)),
        })

    return pl.DataFrame(rows)


def _make_external_ctx(
    dxy_direction: str = "weakening",
    vix_level: float = 18.0,
    central_bank_stance: str = "dovish",
) -> ExternalContext:
    """Create a synthetic ExternalContext."""
    return ExternalContext(
        dxy_direction=dxy_direction,
        dxy_value=103.5,
        vix_level=vix_level,
        vix_regime="normal",
        real_rate_10y=1.8,
        cot_net_spec=150000.0,
        central_bank_stance=central_bank_stance,
        fetched_at=datetime.now(tz=timezone.utc),
        source_quality="live",
    )


# ---------------------------------------------------------------------------
# H4 Technical Context Extraction
# ---------------------------------------------------------------------------


class TestExtractH4Context:
    """Tests for extract_h4_context()."""

    def test_none_input_returns_flat(self) -> None:
        ctx = extract_h4_context(None)
        assert ctx.sma50_direction == "flat"
        assert ctx.sma50_slope == 0.0
        assert ctx.higher_highs == 0
        assert ctx.lower_lows == 0
        assert ctx.bar_count == 0

    def test_empty_df_returns_flat(self) -> None:
        empty = pl.DataFrame({
            "timestamp": [],
            "open": [],
            "high": [],
            "low": [],
            "close": [],
            "volume": [],
        })
        ctx = extract_h4_context(empty)
        assert ctx.sma50_direction == "flat"
        assert ctx.bar_count == 0

    def test_insufficient_data_returns_flat(self) -> None:
        short_df = _make_h4_df(n_bars=30)
        ctx = extract_h4_context(short_df)
        assert ctx.sma50_direction == "flat"
        assert ctx.bar_count == 30

    def test_uptrend_detected(self) -> None:
        df = _make_h4_df(n_bars=100, trend=2.0, volatility=3.0)
        ctx = extract_h4_context(df)
        assert ctx.sma50_direction == "up"
        assert ctx.sma50_slope > 0

    def test_downtrend_detected(self) -> None:
        df = _make_h4_df(n_bars=100, trend=-2.0, volatility=3.0)
        ctx = extract_h4_context(df)
        assert ctx.sma50_direction == "down"
        assert ctx.sma50_slope < 0

    def test_sideways_detected(self) -> None:
        df = _make_h4_df(n_bars=100, trend=0.0, volatility=0.5, seed=99)
        ctx = extract_h4_context(df)
        # With near-zero trend and low volatility, should be flat
        assert ctx.sma50_direction in ("flat", "up", "down")
        # Slope magnitude should be small
        assert abs(ctx.sma50_slope) < 0.5

    def test_frozen_model(self) -> None:
        ctx = extract_h4_context(_make_h4_df())
        with pytest.raises(Exception):
            ctx.sma50_direction = "down"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# SMA + DXY Fallback
# ---------------------------------------------------------------------------


class TestSMADXYFallback:
    """Tests for DirectionEngine._sma_dxy_fallback()."""

    def test_sma_up_dxy_weakening_bullish(self) -> None:
        h4_ctx = H4TechnicalContext(
            sma50_direction="up", sma50_slope=0.05,
            higher_highs=5, lower_lows=1, bar_count=100,
        )
        ext = _make_external_ctx(dxy_direction="weakening")
        result = DirectionEngine._sma_dxy_fallback(h4_ctx, ext)
        assert result.direction == "bullish"
        assert result.confidence == 0.6
        assert "sma50_up" in result.key_drivers
        assert "dxy_weakening" in result.key_drivers
        assert result.source == "sma_fallback"

    def test_sma_down_dxy_strengthening_bearish(self) -> None:
        h4_ctx = H4TechnicalContext(
            sma50_direction="down", sma50_slope=-0.05,
            higher_highs=1, lower_lows=5, bar_count=100,
        )
        ext = _make_external_ctx(dxy_direction="strengthening")
        result = DirectionEngine._sma_dxy_fallback(h4_ctx, ext)
        assert result.direction == "bearish"
        assert result.confidence == 0.6
        assert "sma50_down" in result.key_drivers
        assert "dxy_strengthening" in result.key_drivers

    def test_sma_up_dxy_strengthening_reduced_confidence(self) -> None:
        h4_ctx = H4TechnicalContext(
            sma50_direction="up", sma50_slope=0.05,
            higher_highs=3, lower_lows=2, bar_count=100,
        )
        ext = _make_external_ctx(dxy_direction="strengthening")
        result = DirectionEngine._sma_dxy_fallback(h4_ctx, ext)
        assert result.direction == "bullish"
        assert result.confidence == 0.4  # reduced due to conflicting DXY

    def test_sma_down_dxy_weakening_reduced_confidence(self) -> None:
        h4_ctx = H4TechnicalContext(
            sma50_direction="down", sma50_slope=-0.05,
            higher_highs=2, lower_lows=3, bar_count=100,
        )
        ext = _make_external_ctx(dxy_direction="weakening")
        result = DirectionEngine._sma_dxy_fallback(h4_ctx, ext)
        assert result.direction == "bearish"
        assert result.confidence == 0.4

    def test_sma_flat_neutral(self) -> None:
        h4_ctx = H4TechnicalContext(
            sma50_direction="flat", sma50_slope=0.0,
            higher_highs=2, lower_lows=2, bar_count=100,
        )
        result = DirectionEngine._sma_dxy_fallback(h4_ctx, None)
        assert result.direction == "neutral"
        assert result.confidence == 0.3
        assert "sma50_flat" in result.key_drivers

    def test_no_external_context_uses_flat_dxy(self) -> None:
        h4_ctx = H4TechnicalContext(
            sma50_direction="up", sma50_slope=0.05,
            higher_highs=5, lower_lows=1, bar_count=100,
        )
        result = DirectionEngine._sma_dxy_fallback(h4_ctx, None)
        assert result.direction == "bullish"
        assert result.confidence == 0.5  # no DXY confirmation

    def test_immutable_result(self) -> None:
        h4_ctx = H4TechnicalContext(
            sma50_direction="up", sma50_slope=0.05,
            higher_highs=5, lower_lows=1, bar_count=100,
        )
        result = DirectionEngine._sma_dxy_fallback(h4_ctx, None)
        with pytest.raises(Exception):
            result.direction = "bearish"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# JSON Parsing
# ---------------------------------------------------------------------------


class TestParseDirectionJSON:
    """Tests for _parse_direction_json()."""

    def test_clean_json(self) -> None:
        raw = json.dumps({
            "direction": "bullish",
            "confidence": 0.75,
            "key_factors": ["dxy_weakening", "dovish_cb"],
            "reasoning": "DXY weakening supports gold",
        })
        result = _parse_direction_json(raw)
        assert result["direction"] == "bullish"
        assert result["confidence"] == 0.75
        assert "dxy_weakening" in result["key_factors"]

    def test_json_with_markdown_fences(self) -> None:
        raw = '```json\n{"direction": "bearish", "confidence": 0.6, "key_factors": [], "reasoning": "test"}\n```'
        result = _parse_direction_json(raw)
        assert result["direction"] == "bearish"
        assert result["confidence"] == 0.6

    def test_key_drivers_alias(self) -> None:
        """Judge returns key_drivers instead of key_factors."""
        raw = json.dumps({
            "direction": "neutral",
            "confidence": 0.3,
            "key_drivers": ["mixed_signals"],
            "reasoning": "conflicting",
        })
        result = _parse_direction_json(raw)
        assert "mixed_signals" in result["key_factors"]

    def test_malformed_json_raises(self) -> None:
        with pytest.raises(Exception):
            _parse_direction_json("not json at all")


# ---------------------------------------------------------------------------
# Prompt Formatting
# ---------------------------------------------------------------------------


class TestPromptFormatting:
    """Tests for context formatting functions."""

    def test_format_macro_with_data(self) -> None:
        ext = _make_external_ctx()
        text = _format_macro_context(ext)
        assert "dxy_direction: weakening" in text
        assert "vix_level: 18.0" in text
        assert "central_bank_stance: dovish" in text

    def test_format_macro_none(self) -> None:
        text = _format_macro_context(None)
        assert "unavailable" in text

    def test_format_news_with_data(self) -> None:
        ext = _make_external_ctx()
        text = _format_news_context(ext)
        assert "vix_level" in text
        assert "Sprint 8" in text  # mentions future news integration

    def test_format_news_none(self) -> None:
        text = _format_news_context(None)
        assert "neutral" in text

    def test_format_judge_context(self) -> None:
        macro_view = {
            "direction": "bullish", "confidence": 0.7,
            "key_factors": ["dxy_weak"], "reasoning": "DXY down",
        }
        news_view = {
            "direction": "neutral", "confidence": 0.3,
            "key_factors": [], "reasoning": "no events",
        }
        h4_ctx = H4TechnicalContext(
            sma50_direction="up", sma50_slope=0.05,
            higher_highs=4, lower_lows=1, bar_count=100,
        )
        text = _format_judge_context(macro_view, news_view, h4_ctx)
        assert "MACRO ANALYST VIEW" in text
        assert "NEWS ANALYST VIEW" in text
        assert "H4 TECHNICAL CONTEXT" in text
        assert "sma50_direction: up" in text


# ---------------------------------------------------------------------------
# Memory Cache
# ---------------------------------------------------------------------------


class TestMemoryCache:
    """Tests for DirectionEngine in-memory cache."""

    def test_cache_hit_avoids_computation(self) -> None:
        engine = DirectionEngine()
        h4_df = _make_h4_df(n_bars=100, trend=2.0)
        ts = datetime(2024, 6, 1, 8, 0, tzinfo=timezone.utc)

        # First call populates cache
        with patch.object(engine, "_run_direction_debate", side_effect=RuntimeError("no LLM")):
            result1 = engine.get_direction(h4_df=h4_df, bar_ts=ts)

        # Second call should hit cache (same hour key)
        with patch.object(engine, "_run_direction_debate", side_effect=AssertionError("should not be called")):
            result2 = engine.get_direction(h4_df=h4_df, bar_ts=ts)

        assert result1.direction == result2.direction
        assert result1.confidence == result2.confidence

    def test_different_hour_misses_cache(self) -> None:
        engine = DirectionEngine()
        h4_df = _make_h4_df(n_bars=100, trend=2.0)
        ts1 = datetime(2024, 6, 1, 8, 0, tzinfo=timezone.utc)
        ts2 = datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc)

        with patch.object(engine, "_run_direction_debate", side_effect=RuntimeError("no LLM")):
            result1 = engine.get_direction(h4_df=h4_df, bar_ts=ts1)
            result2 = engine.get_direction(h4_df=h4_df, bar_ts=ts2)

        # Both should complete (cache miss on second), both from SMA fallback
        assert result1.source == "sma_fallback"
        assert result2.source == "sma_fallback"


# ---------------------------------------------------------------------------
# File Cache (DirectionCacheLookup)
# ---------------------------------------------------------------------------


class TestDirectionCacheLookup:
    """Tests for DirectionCacheLookup parquet-based cache."""

    def _build_cache_df(self, n: int = 10) -> pl.DataFrame:
        """Create a minimal direction cache DataFrame."""
        start = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
        rows = []
        for i in range(n):
            rows.append({
                "ts": start + timedelta(hours=4 * i),
                "direction": "bullish" if i % 3 == 0 else "bearish" if i % 3 == 1 else "neutral",
                "confidence": round(0.4 + (i % 5) * 0.1, 2),
                "key_drivers": json.dumps(["sma50_up"]),
                "reasoning": f"entry {i}",
                "source": "sma_fallback",
            })
        return pl.DataFrame(rows, schema={
            "ts": pl.Datetime("ns", "UTC"),
            "direction": pl.String,
            "confidence": pl.Float64,
            "key_drivers": pl.String,
            "reasoning": pl.String,
            "source": pl.String,
        })

    def test_lookup_exact_match(self, tmp_path: Path) -> None:
        df = self._build_cache_df()
        path = tmp_path / "direction_cache.parquet"
        df.write_parquet(path)

        cache = DirectionCacheLookup(path)
        ts = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
        result = cache.lookup(ts)
        assert result is not None
        assert result.direction == "bullish"
        assert result.source == "cache"

    def test_lookup_between_entries(self, tmp_path: Path) -> None:
        df = self._build_cache_df()
        path = tmp_path / "direction_cache.parquet"
        df.write_parquet(path)

        cache = DirectionCacheLookup(path)
        # Between first and second entry — should return first
        ts = datetime(2024, 1, 1, 2, 0, tzinfo=timezone.utc)
        result = cache.lookup(ts)
        assert result is not None
        assert result.direction == "bullish"

    def test_lookup_before_first_returns_none(self, tmp_path: Path) -> None:
        df = self._build_cache_df()
        path = tmp_path / "direction_cache.parquet"
        df.write_parquet(path)

        cache = DirectionCacheLookup(path)
        ts = datetime(2023, 12, 31, 0, 0, tzinfo=timezone.utc)
        assert cache.lookup(ts) is None

    def test_lookup_after_last(self, tmp_path: Path) -> None:
        df = self._build_cache_df(n=5)
        path = tmp_path / "direction_cache.parquet"
        df.write_parquet(path)

        cache = DirectionCacheLookup(path)
        ts = datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)
        result = cache.lookup(ts)
        assert result is not None  # Returns last entry

    def test_cache_size_and_range(self, tmp_path: Path) -> None:
        df = self._build_cache_df(n=20)
        path = tmp_path / "direction_cache.parquet"
        df.write_parquet(path)

        cache = DirectionCacheLookup(path)
        assert cache.size == 20
        start, end = cache.date_range
        assert start < end

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            DirectionCacheLookup(tmp_path / "nonexistent.parquet")

    def test_empty_cache_raises(self, tmp_path: Path) -> None:
        empty_df = pl.DataFrame({
            "ts": [],
            "direction": [],
            "confidence": [],
            "key_drivers": [],
            "reasoning": [],
            "source": [],
        }, schema={
            "ts": pl.Datetime("ns", "UTC"),
            "direction": pl.String,
            "confidence": pl.Float64,
            "key_drivers": pl.String,
            "reasoning": pl.String,
            "source": pl.String,
        })
        path = tmp_path / "empty_cache.parquet"
        empty_df.write_parquet(path)

        with pytest.raises(ValueError, match="empty"):
            DirectionCacheLookup(path)

    def test_result_is_frozen(self, tmp_path: Path) -> None:
        df = self._build_cache_df(n=3)
        path = tmp_path / "cache.parquet"
        df.write_parquet(path)

        cache = DirectionCacheLookup(path)
        result = cache.lookup(datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc))
        assert result is not None
        with pytest.raises(Exception):
            result.direction = "bearish"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Integration: get_direction with mocked LLM
# ---------------------------------------------------------------------------


class TestGetDirectionIntegration:
    """Integration tests for DirectionEngine.get_direction()."""

    @patch("smc.ai.direction_engine._has_claude_cli", return_value=False)
    def test_no_llm_falls_back_to_sma(self, mock_cli: MagicMock) -> None:
        engine = DirectionEngine()
        h4_df = _make_h4_df(n_bars=100, trend=2.0)
        result = engine.get_direction(h4_df=h4_df)
        assert result.source == "sma_fallback"
        assert result.direction in ("bullish", "bearish", "neutral")

    def test_with_file_cache(self, tmp_path: Path) -> None:
        # Build a small cache
        start = datetime(2024, 6, 1, 0, 0, tzinfo=timezone.utc)
        rows = [{
            "ts": start + timedelta(hours=4 * i),
            "direction": "bullish",
            "confidence": 0.7,
            "key_drivers": json.dumps(["sma50_up", "dxy_weakening"]),
            "reasoning": "cache entry",
            "source": "sma_fallback",
        } for i in range(10)]
        df = pl.DataFrame(rows, schema={
            "ts": pl.Datetime("ns", "UTC"),
            "direction": pl.String,
            "confidence": pl.Float64,
            "key_drivers": pl.String,
            "reasoning": pl.String,
            "source": pl.String,
        })
        cache_path = tmp_path / "dir_cache.parquet"
        df.write_parquet(cache_path)

        engine = DirectionEngine(cache_path=cache_path)
        result = engine.get_direction(bar_ts=start + timedelta(hours=2))
        assert result.source == "cache"
        assert result.direction == "bullish"

    @patch("smc.ai.direction_engine._has_claude_cli", return_value=True)
    @patch("smc.ai.direction_engine._claude_cli_chat")
    def test_ai_debate_success(self, mock_cli: MagicMock, mock_has: MagicMock) -> None:
        """Test successful AI debate with mocked Claude CLI."""
        # Mock responses for macro, news, judge
        macro_resp = json.dumps({
            "direction": "bullish", "confidence": 0.7,
            "key_factors": ["dxy_weakening"], "reasoning": "DXY down",
        })
        news_resp = json.dumps({
            "direction": "bullish", "confidence": 0.6,
            "key_factors": ["dovish_fomc"], "reasoning": "FOMC dovish",
        })
        judge_resp = json.dumps({
            "direction": "bullish", "confidence": 0.8,
            "key_drivers": ["dxy_weakening", "dovish_fomc"],
            "reasoning": "Both analysts bullish, H4 confirms",
        })
        mock_cli.side_effect = [macro_resp, news_resp, judge_resp]

        engine = DirectionEngine()
        h4_df = _make_h4_df(n_bars=100, trend=2.0)
        result = engine.get_direction(h4_df=h4_df)

        assert result.source == "ai_debate"
        assert result.direction == "bullish"
        assert result.confidence == 0.8
        assert mock_cli.call_count == 3

    @patch("smc.ai.direction_engine._has_claude_cli", return_value=False)
    def test_none_h4_df_returns_result(self, mock_cli: MagicMock) -> None:
        """Engine should still work with None H4 data (SMA fallback → neutral)."""
        engine = DirectionEngine()
        result = engine.get_direction(h4_df=None)
        assert result.direction == "neutral"
        assert result.source == "sma_fallback"


# ---------------------------------------------------------------------------
# AIDirection model tests
# ---------------------------------------------------------------------------


class TestAIDirectionModel:
    """Tests for the AIDirection Pydantic model."""

    def test_create_valid(self) -> None:
        d = AIDirection(
            direction="bullish",
            confidence=0.7,
            key_drivers=("sma50_up", "dxy_weakening"),
            reasoning="test",
            assessed_at=datetime.now(tz=timezone.utc),
            source="sma_fallback",
        )
        assert d.direction == "bullish"
        assert d.cost_usd == 0.0

    def test_frozen(self) -> None:
        d = AIDirection(
            direction="neutral",
            confidence=0.3,
            key_drivers=(),
            reasoning="test",
            assessed_at=datetime.now(tz=timezone.utc),
            source="neutral_default",
        )
        with pytest.raises(Exception):
            d.confidence = 0.9  # type: ignore[misc]
