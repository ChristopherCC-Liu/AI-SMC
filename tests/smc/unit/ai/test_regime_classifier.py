"""Unit tests for smc.ai.regime_classifier — feature extraction and ATR fallback.

Tests cover:
- extract_regime_context() with various D1/H4 data scenarios
- classify_regime_ai() fallback chain (ATR → default)
- ATR regime → MarketRegimeAI mapping with SMA50 direction
- Edge cases: None/empty DataFrames, insufficient data
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl
import pytest

from smc.ai.models import MarketRegimeAI
from smc.ai.regime_classifier import (
    RegimeContext,
    _atr_fallback,
    _compute_atr_pct,
    _compute_sma,
    _count_hh_ll,
    _sma50_direction_and_slope,
    classify_regime_ai,
    extract_regime_context,
)


# ---------------------------------------------------------------------------
# Test data helpers
# ---------------------------------------------------------------------------


def _make_d1_df(
    n_bars: int = 100,
    base_price: float = 2000.0,
    trend: float = 0.0,
    volatility: float = 10.0,
    seed: int = 42,
) -> pl.DataFrame:
    """Create a synthetic D1 OHLCV DataFrame.

    trend > 0: uptrend, trend < 0: downtrend, trend == 0: sideways.
    """
    rng = np.random.RandomState(seed)
    start = datetime(2023, 1, 2, 0, 0, tzinfo=timezone.utc)

    closes = []
    price = base_price
    for _ in range(n_bars):
        price += trend + rng.normal(0, volatility)
        price = max(price, 100.0)  # floor
        closes.append(price)

    rows = []
    for i, c in enumerate(closes):
        bar_range = abs(rng.normal(0, volatility * 0.5))
        o = c + rng.normal(0, volatility * 0.3)
        h = max(o, c) + abs(rng.normal(0, bar_range))
        lo = min(o, c) - abs(rng.normal(0, bar_range))
        rows.append({
            "ts": start + timedelta(days=i),
            "open": round(o, 2),
            "high": round(h, 2),
            "low": round(lo, 2),
            "close": round(c, 2),
            "volume": 1000.0,
        })

    return pl.DataFrame(rows, schema={
        "ts": pl.Datetime("ns", "UTC"),
        "open": pl.Float64,
        "high": pl.Float64,
        "low": pl.Float64,
        "close": pl.Float64,
        "volume": pl.Float64,
    })


def _make_h4_df(n_bars: int = 400, **kwargs) -> pl.DataFrame:
    """Create a synthetic H4 OHLCV DataFrame."""
    rng = np.random.RandomState(kwargs.get("seed", 42))
    base_price = kwargs.get("base_price", 2000.0)
    trend = kwargs.get("trend", 0.0)
    volatility = kwargs.get("volatility", 5.0)
    start = datetime(2023, 1, 2, 0, 0, tzinfo=timezone.utc)

    closes = []
    price = base_price
    for _ in range(n_bars):
        price += trend + rng.normal(0, volatility)
        price = max(price, 100.0)
        closes.append(price)

    rows = []
    for i, c in enumerate(closes):
        bar_range = abs(rng.normal(0, volatility * 0.3))
        o = c + rng.normal(0, volatility * 0.2)
        h = max(o, c) + abs(rng.normal(0, bar_range))
        lo = min(o, c) - abs(rng.normal(0, bar_range))
        rows.append({
            "ts": start + timedelta(hours=4 * i),
            "open": round(o, 2),
            "high": round(h, 2),
            "low": round(lo, 2),
            "close": round(c, 2),
            "volume": 500.0,
        })

    return pl.DataFrame(rows, schema={
        "ts": pl.Datetime("ns", "UTC"),
        "open": pl.Float64,
        "high": pl.Float64,
        "low": pl.Float64,
        "close": pl.Float64,
        "volume": pl.Float64,
    })


# ---------------------------------------------------------------------------
# _compute_atr_pct
# ---------------------------------------------------------------------------


class TestComputeAtrPct:
    def test_returns_none_for_none(self):
        assert _compute_atr_pct(None) is None

    def test_returns_none_for_insufficient_data(self):
        df = _make_d1_df(n_bars=5)
        assert _compute_atr_pct(df) is None

    def test_returns_positive_float(self):
        df = _make_d1_df(n_bars=100)
        result = _compute_atr_pct(df)
        assert result is not None
        assert result > 0.0


# ---------------------------------------------------------------------------
# SMA50 direction and slope
# ---------------------------------------------------------------------------


class TestSma50Direction:
    def test_uptrend_detected(self):
        df = _make_d1_df(n_bars=100, trend=5.0, volatility=2.0)
        closes = df["close"].to_list()
        direction, slope = _sma50_direction_and_slope(closes)
        assert direction == "up"
        assert slope > 0

    def test_downtrend_detected(self):
        df = _make_d1_df(n_bars=100, trend=-5.0, volatility=2.0)
        closes = df["close"].to_list()
        direction, slope = _sma50_direction_and_slope(closes)
        assert direction == "down"
        assert slope < 0

    def test_flat_for_sideways(self):
        df = _make_d1_df(n_bars=100, trend=0.0, volatility=0.5)
        closes = df["close"].to_list()
        direction, _ = _sma50_direction_and_slope(closes)
        assert direction == "flat"

    def test_insufficient_data_returns_flat(self):
        direction, slope = _sma50_direction_and_slope([100.0] * 10)
        assert direction == "flat"
        assert slope == 0.0


# ---------------------------------------------------------------------------
# HH/LL counting
# ---------------------------------------------------------------------------


class TestCountHhLl:
    def test_uptrend_has_higher_highs(self):
        # Strong trend with enough bars for clear local extremes
        df = _make_d1_df(n_bars=100, trend=5.0, volatility=8.0)
        hh, ll = _count_hh_ll(df)
        assert hh >= ll  # uptrend should produce at least as many HH as LL

    def test_downtrend_has_lower_lows(self):
        df = _make_d1_df(n_bars=100, trend=-5.0, volatility=8.0)
        hh, ll = _count_hh_ll(df)
        assert ll >= hh  # downtrend should produce at least as many LL as HH

    def test_empty_df_returns_zeros(self):
        df = _make_d1_df(n_bars=3)
        hh, ll = _count_hh_ll(df)
        assert hh == 0
        assert ll == 0


# ---------------------------------------------------------------------------
# extract_regime_context
# ---------------------------------------------------------------------------


class TestExtractRegimeContext:
    def test_returns_frozen_dataclass(self):
        d1 = _make_d1_df()
        h4 = _make_h4_df()
        ctx = extract_regime_context(d1, h4)
        assert isinstance(ctx, RegimeContext)
        with pytest.raises(AttributeError):
            ctx.current_price = 999.0  # type: ignore[misc]

    def test_none_d1_returns_defaults(self):
        ctx = extract_regime_context(None, None)
        assert ctx.d1_atr_pct is None
        assert ctx.d1_sma50_direction == "flat"
        assert ctx.current_price == 0.0
        assert ctx.atr_regime == "transitional"  # default from classify_regime(None)

    def test_d1_only_without_h4(self):
        d1 = _make_d1_df()
        ctx = extract_regime_context(d1, None)
        assert ctx.d1_atr_pct is not None
        assert ctx.h4_atr_pct is None
        assert ctx.current_price > 0

    def test_atr_regime_populated(self):
        d1 = _make_d1_df(volatility=50.0)  # high vol → trending
        ctx = extract_regime_context(d1, None)
        assert ctx.atr_regime in ("trending", "transitional", "ranging")

    def test_external_context_threaded(self):
        from smc.ai.models import ExternalContext

        ext = ExternalContext(
            dxy_direction="weakening",
            fetched_at=datetime.now(tz=timezone.utc),
            source_quality="live",
        )
        ctx = extract_regime_context(_make_d1_df(), None, external_ctx=ext)
        assert ctx.external is ext
        assert ctx.external.dxy_direction == "weakening"


# ---------------------------------------------------------------------------
# classify_regime_ai — ATR fallback path
# ---------------------------------------------------------------------------


class TestClassifyRegimeAi:
    def test_none_d1_returns_default(self):
        result = classify_regime_ai(None, None)
        assert result.regime == "TRANSITION"
        assert result.source == "default"
        assert result.confidence == 0.3
        assert result.cost_usd == 0.0

    def test_empty_d1_returns_default(self):
        empty = pl.DataFrame(schema={
            "ts": pl.Datetime("ns", "UTC"),
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
        })
        result = classify_regime_ai(empty, None)
        assert result.regime == "TRANSITION"
        assert result.source == "default"

    def test_atr_fallback_uptrend(self):
        # Strong uptrend: high vol + positive slope
        d1 = _make_d1_df(n_bars=100, trend=8.0, volatility=30.0, base_price=2000.0)
        result = classify_regime_ai(d1, None)
        assert result.source == "atr_fallback"
        assert result.cost_usd == 0.0
        # With strong uptrend, should be TREND_UP or TRANSITION
        assert result.regime in ("TREND_UP", "TREND_DOWN", "TRANSITION")

    def test_atr_fallback_ranging(self):
        # Low vol sideways: should map to CONSOLIDATION
        d1 = _make_d1_df(n_bars=100, trend=0.0, volatility=2.0, base_price=2000.0)
        result = classify_regime_ai(d1, None)
        assert result.source == "atr_fallback"
        # Low vol → ranging → CONSOLIDATION
        assert result.regime in ("CONSOLIDATION", "TRANSITION")

    def test_param_preset_matches_regime(self):
        d1 = _make_d1_df()
        result = classify_regime_ai(d1, None)
        from smc.ai.param_router import route
        expected_params = route(result.regime)
        assert result.param_preset == expected_params

    def test_assessment_is_frozen(self):
        d1 = _make_d1_df()
        result = classify_regime_ai(d1, None)
        with pytest.raises(Exception):
            result.regime = "TREND_UP"  # type: ignore[misc]

    def test_ai_disabled_skips_debate(self):
        d1 = _make_d1_df()
        result = classify_regime_ai(d1, None, ai_enabled=False)
        assert result.source in ("atr_fallback", "default")

    def test_telemetry_emits_event_per_call(self, monkeypatch) -> None:
        """Every classify_regime_ai() call must emit one ai_regime_classified event."""
        captured: list[tuple[str, dict]] = []

        def _fake_info(event: str, **fields: object) -> None:
            captured.append((event, dict(fields)))

        import smc.monitor.structured_log as _slog
        monkeypatch.setattr(_slog, "info", _fake_info)

        d1 = _make_d1_df()
        result = classify_regime_ai(d1, None, ai_enabled=False)

        assert len(captured) == 1
        event_name, fields = captured[0]
        assert event_name == "ai_regime_classified"
        assert fields["regime"] == result.regime
        assert fields["source"] == result.source
        assert fields["ai_enabled"] is False
        assert fields["elapsed_ms"] >= 0
        assert fields["cost_usd"] == 0.0

    def test_telemetry_never_breaks_classifier(self, monkeypatch) -> None:
        """Broken telemetry must not propagate to the caller."""
        def _boom(*_a: object, **_kw: object) -> None:
            raise RuntimeError("telemetry dead")

        import smc.monitor.structured_log as _slog
        monkeypatch.setattr(_slog, "info", _boom)

        d1 = _make_d1_df()
        # Must not raise
        result = classify_regime_ai(d1, None, ai_enabled=False)
        assert result.source in ("atr_fallback", "default")

    def test_reasoning_under_300_chars(self):
        d1 = _make_d1_df()
        result = classify_regime_ai(d1, None)
        assert len(result.reasoning) <= 300

    def test_debate_exception_falls_back_to_atr(self, monkeypatch) -> None:
        """Round 4 v5 hotfix: any AttributeError / RuntimeError / etc. from
        the debate pipeline must not kill the cycle. classify_regime_ai
        falls through to ATR fallback instead of propagating.
        """
        def _boom(_features: object) -> dict:
            raise AttributeError("'RegimeContext' object has no attribute 'get'")

        import smc.ai.debate.pipeline as _pipe
        monkeypatch.setattr(_pipe, "run_regime_debate", _boom)

        d1 = _make_d1_df()
        # Must not raise — ATR fallback absorbs the failure.
        result = classify_regime_ai(d1, None, ai_enabled=True)
        assert result.source in ("atr_fallback", "default")


class TestCoerceDebateResult:
    """Ensure the debate dict → AIRegimeAssessment adapter is strict.

    The adapter must reject malformed dicts (returning None → ATR
    fallback) and only accept valid ``MarketRegimeAI`` labels. The
    ``ctx`` arg is accepted for future extensions but the body doesn't
    read from it, so tests pass None here.
    """

    def test_valid_dict_coerces(self) -> None:
        from smc.ai.regime_classifier import _coerce_debate_result_to_assessment
        raw = {
            "regime": "TREND_UP",
            "confidence": 0.72,
            "reasoning": "Strong uptrend + macro tailwind",
            "total_cost_usd": 0.42,
        }
        result = _coerce_debate_result_to_assessment(raw, None)  # type: ignore[arg-type]
        assert result is not None
        assert result.regime == "TREND_UP"
        assert result.trend_direction == "bullish"
        assert result.confidence == 0.72
        assert result.source == "ai_debate"
        assert result.cost_usd == 0.42

    def test_trend_down_maps_to_bearish(self) -> None:
        from smc.ai.regime_classifier import _coerce_debate_result_to_assessment
        raw = {"regime": "TREND_DOWN", "confidence": 0.6}
        result = _coerce_debate_result_to_assessment(raw, None)  # type: ignore[arg-type]
        assert result is not None and result.trend_direction == "bearish"

    def test_consolidation_maps_to_neutral(self) -> None:
        from smc.ai.regime_classifier import _coerce_debate_result_to_assessment
        raw = {"regime": "CONSOLIDATION", "confidence": 0.6}
        result = _coerce_debate_result_to_assessment(raw, None)  # type: ignore[arg-type]
        assert result is not None and result.trend_direction == "neutral"

    def test_unknown_regime_rejected(self) -> None:
        from smc.ai.regime_classifier import _coerce_debate_result_to_assessment
        raw = {"regime": "SUPER_BULL", "confidence": 0.9}
        assert _coerce_debate_result_to_assessment(raw, None) is None  # type: ignore[arg-type]

    def test_non_dict_rejected(self) -> None:
        from smc.ai.regime_classifier import _coerce_debate_result_to_assessment
        assert _coerce_debate_result_to_assessment("not a dict", None) is None  # type: ignore[arg-type]
        assert _coerce_debate_result_to_assessment(None, None) is None  # type: ignore[arg-type]

    def test_confidence_clamped(self) -> None:
        from smc.ai.regime_classifier import _coerce_debate_result_to_assessment
        raw = {"regime": "TRANSITION", "confidence": 1.5}
        result = _coerce_debate_result_to_assessment(raw, None)  # type: ignore[arg-type]
        assert result is not None and result.confidence == 1.0

    def test_missing_reasoning_defaults_empty(self) -> None:
        from smc.ai.regime_classifier import _coerce_debate_result_to_assessment
        raw = {"regime": "CONSOLIDATION", "confidence": 0.6}
        result = _coerce_debate_result_to_assessment(raw, None)  # type: ignore[arg-type]
        assert result is not None and result.reasoning == ""


# ---------------------------------------------------------------------------
# ATR fallback mapping specifics
# ---------------------------------------------------------------------------


class TestAtrFallbackMapping:
    def _ctx(self, **overrides) -> RegimeContext:
        defaults = dict(
            d1_atr_pct=1.5,
            d1_sma50_direction="up",
            d1_sma50_slope=0.1,
            d1_close_vs_sma50=2.0,
            d1_recent_range_pct=5.0,
            d1_higher_highs=5,
            d1_lower_lows=1,
            h4_atr_pct=0.8,
            h4_trend_bars=10,
            h4_volatility_rank=0.6,
            current_price=2100.0,
            ath_distance_pct=5.0,
            price_52w_percentile=0.85,
            external=None,
            atr_regime="trending",
        )
        defaults.update(overrides)
        return RegimeContext(**defaults)

    def test_trending_sma_up_maps_to_trend_up(self):
        ctx = self._ctx(atr_regime="trending", d1_sma50_direction="up")
        result = _atr_fallback(ctx)
        assert result.regime == "TREND_UP"
        assert result.trend_direction == "bullish"

    def test_trending_sma_down_maps_to_trend_down(self):
        ctx = self._ctx(atr_regime="trending", d1_sma50_direction="down", d1_sma50_slope=-0.1)
        result = _atr_fallback(ctx)
        assert result.regime == "TREND_DOWN"
        assert result.trend_direction == "bearish"

    def test_trending_sma_flat_maps_to_transition(self):
        ctx = self._ctx(atr_regime="trending", d1_sma50_direction="flat", d1_sma50_slope=0.0)
        result = _atr_fallback(ctx)
        assert result.regime == "TRANSITION"
        assert result.trend_direction == "neutral"

    def test_ranging_maps_to_consolidation(self):
        ctx = self._ctx(atr_regime="ranging", d1_atr_pct=0.7)
        result = _atr_fallback(ctx)
        assert result.regime == "CONSOLIDATION"
        assert result.trend_direction == "neutral"

    def test_transitional_maps_to_transition(self):
        ctx = self._ctx(atr_regime="transitional", d1_atr_pct=1.1)
        result = _atr_fallback(ctx)
        assert result.regime == "TRANSITION"

    def test_transitional_with_sma_up_is_bullish(self):
        ctx = self._ctx(atr_regime="transitional", d1_sma50_direction="up")
        result = _atr_fallback(ctx)
        assert result.trend_direction == "bullish"

    def test_all_fallback_results_have_zero_cost(self):
        for atr_regime in ("trending", "transitional", "ranging"):
            ctx = self._ctx(atr_regime=atr_regime)
            result = _atr_fallback(ctx)
            assert result.cost_usd == 0.0
            assert result.source == "atr_fallback"

    def test_confidence_bounded(self):
        for atr_regime in ("trending", "transitional", "ranging"):
            ctx = self._ctx(atr_regime=atr_regime)
            result = _atr_fallback(ctx)
            assert 0.0 <= result.confidence <= 1.0


# ---------------------------------------------------------------------------
# Round 5 A-track Task #7 refinement — precomputed_ctx kwarg
# ---------------------------------------------------------------------------


class TestPrecomputedCtx:
    """classify_regime_ai accepts a precomputed RegimeContext so the
    aggregator (SL fitness judge path) can extract once and forward."""

    def test_precomputed_ctx_yields_identical_result(self):
        """Passing precomputed_ctx must produce the same regime/source
        as the default code path that extracts internally."""
        from smc.ai.regime_classifier import extract_regime_context
        d1 = _make_d1_df(n_bars=100, trend=5.0, volatility=20.0, base_price=4000.0)
        h4 = _make_d1_df(n_bars=300, trend=4.0, volatility=18.0, base_price=4000.0)

        # Default path
        result_default = classify_regime_ai(d1, h4)
        # Precomputed path
        ctx = extract_regime_context(d1, h4)
        result_pre = classify_regime_ai(d1, h4, precomputed_ctx=ctx)

        assert result_default.regime == result_pre.regime
        assert result_default.source == result_pre.source
        assert result_default.trend_direction == result_pre.trend_direction

    def test_precomputed_ctx_none_falls_back_to_extraction(self):
        """Passing precomputed_ctx=None must not change behaviour."""
        d1 = _make_d1_df()
        result_a = classify_regime_ai(d1, None)
        result_b = classify_regime_ai(d1, None, precomputed_ctx=None)
        assert result_a.regime == result_b.regime
        assert result_a.source == result_b.source

    def test_precomputed_ctx_cached_path_still_hits(self, monkeypatch):
        """When a cache hit occurs, precomputed_ctx is irrelevant — the
        cached assessment wins before ctx is ever consulted."""
        from smc.ai.regime_classifier import extract_regime_context
        from smc.ai.models import AIRegimeAssessment
        from smc.ai.param_router import route
        from datetime import datetime, timezone

        class _FakeCache:
            def lookup(self, ts):
                return AIRegimeAssessment(
                    regime="ATH_BREAKOUT",
                    trend_direction="bullish",
                    confidence=0.9,
                    param_preset=route("ATH_BREAKOUT"),
                    reasoning="fake cache",
                    assessed_at=datetime.now(tz=timezone.utc),
                    source="ai_debate",
                    cost_usd=0.0,
                )

        d1 = _make_d1_df()
        ctx = extract_regime_context(d1, None)
        result = classify_regime_ai(
            d1, None,
            cache=_FakeCache(),
            cache_ts=datetime.now(tz=timezone.utc),
            precomputed_ctx=ctx,
        )
        # Cache path wins regardless of ctx — regime comes from fake cache.
        assert result.regime == "ATH_BREAKOUT"
