"""Tests for the Phase 5.2 Stage C production-wiring adapters.

Three adapters under test:

1. :class:`MT5BrokerExposureProvider` — sums signed lots from a
   :class:`BrokerPort` stub.
2. :class:`NewsEngineFeaturesProvider` — orchestrates ``NewsEngine``
   + exposure lookup + ``classify_for_xauusd``; degrades to
   ``NewsUnavailable`` on crawler errors.
3. :class:`ForexDataLakeReplaySource` — walks ``ForexDataLake`` for
   each window edge and emits :class:`ReplayObservation`.

Plus a smoke integration test that wires the three adapters into
``decision_server.create_app`` to confirm the contracts compose.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Final

import polars as pl
import pytest
from fastapi.testclient import TestClient

from smc.execution.types import PositionState
from smc.hedgerock.decision_server import (
    FeaturesUnavailable,
    MarketFeatures,
    NewsUnavailable,
    PrevRegimeStore,
    create_app,
)
from smc.hedgerock.exposure_provider_impl import (
    MT5BrokerExposureProvider,
    signed_lots_for_symbol,
)
from smc.hedgerock.news_classifier import NewsClassification
from smc.hedgerock.news_engine import NewsEvent
from smc.hedgerock.news_features_provider_impl import (
    DEFAULT_FLAT_THRESHOLD_LOTS,
    NewsEngineFeaturesProvider,
    exposure_lots_to_direction,
)
from smc.hedgerock.replay_data_source_impl import ForexDataLakeReplaySource


_SYMBOL: Final[str] = "XAUUSD"
_NOW: Final[datetime] = datetime(2024, 3, 6, 13, 30, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_position(
    *,
    instrument: str = _SYMBOL,
    direction: str = "long",
    lots: float = 0.5,
    ticket: int = 1,
) -> PositionState:
    return PositionState(
        ticket=ticket,
        instrument=instrument,
        direction=direction,  # type: ignore[arg-type]
        lots=lots,
        open_price=2300.0,
        current_price=2305.0,
        sl=2280.0,
        tp=2330.0,
        pnl_usd=50.0,
        open_time=_NOW,
    )


class _StubBroker:
    """Minimal :class:`BrokerPort`-compatible stub."""

    def __init__(self, positions: tuple[PositionState, ...]) -> None:
        self._positions = positions

    def get_positions(self) -> tuple[PositionState, ...]:
        return self._positions

    # Unused interface members — adapter only touches get_positions.
    def send_order(self, request) -> None:  # pragma: no cover
        raise NotImplementedError

    def modify_order(self, ticket, *, sl=None, tp=None) -> None:  # pragma: no cover
        raise NotImplementedError

    def close_position(self, ticket, lots=None) -> None:  # pragma: no cover
        raise NotImplementedError

    def get_account_info(self):  # pragma: no cover
        raise NotImplementedError


# ---------------------------------------------------------------------------
# MT5BrokerExposureProvider
# ---------------------------------------------------------------------------


def test_signed_lots_long_only() -> None:
    positions = (_make_position(direction="long", lots=0.7),)
    assert signed_lots_for_symbol(positions, _SYMBOL) == pytest.approx(0.7)


def test_signed_lots_short_negates() -> None:
    positions = (_make_position(direction="short", lots=0.5),)
    assert signed_lots_for_symbol(positions, _SYMBOL) == pytest.approx(-0.5)


def test_signed_lots_aggregates_mixed() -> None:
    positions = (
        _make_position(direction="long", lots=1.0, ticket=1),
        _make_position(direction="short", lots=0.4, ticket=2),
        _make_position(direction="long", lots=0.2, ticket=3),
    )
    # 1.0 - 0.4 + 0.2 = +0.8
    assert signed_lots_for_symbol(positions, _SYMBOL) == pytest.approx(0.8)


def test_signed_lots_filters_other_symbols() -> None:
    positions = (
        _make_position(instrument="XAUUSD", direction="long", lots=0.5),
        _make_position(instrument="EURUSD", direction="long", lots=2.0, ticket=2),
    )
    # EURUSD is excluded.
    assert signed_lots_for_symbol(positions, _SYMBOL) == pytest.approx(0.5)


def test_signed_lots_empty_returns_zero() -> None:
    assert signed_lots_for_symbol((), _SYMBOL) == 0.0


def test_mt5_broker_exposure_provider_uses_broker_get_positions() -> None:
    broker = _StubBroker((_make_position(direction="long", lots=0.3),))
    provider = MT5BrokerExposureProvider(broker)
    assert provider.get_exposure_lots(_SYMBOL) == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# Helper: exposure_lots_to_direction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "lots,expected",
    [
        (1.0, "long"),
        (-1.0, "short"),
        (0.0, "flat"),
        (0.001, "flat"),  # below default threshold
        (-0.001, "flat"),
    ],
)
def test_exposure_lots_to_direction(lots: float, expected: str) -> None:
    assert exposure_lots_to_direction(lots) == expected


def test_exposure_lots_to_direction_custom_threshold() -> None:
    # 0.05 is below 0.1 threshold → flat
    assert exposure_lots_to_direction(0.05, flat_threshold=0.1) == "flat"
    # 0.5 is above
    assert exposure_lots_to_direction(0.5, flat_threshold=0.1) == "long"


def test_exposure_threshold_constant_matches_exit_decider() -> None:
    """DEFAULT_FLAT_THRESHOLD_LOTS must agree with exit_decider._exposure_sign.

    Otherwise the news classifier sees a different exposure direction
    than the exit decider's hard rule — silent contract drift.
    """
    # exit_decider uses 0.005; we mirror it.
    assert DEFAULT_FLAT_THRESHOLD_LOTS == 0.005


# ---------------------------------------------------------------------------
# NewsEngineFeaturesProvider
# ---------------------------------------------------------------------------


class _StubExposureProvider:
    def __init__(self, lots: float) -> None:
        self._lots = lots

    def get_exposure_lots(self, symbol: str) -> float:
        return self._lots


class _StubNewsEngine:
    """Async ``find_active_event``-only stub."""

    def __init__(self, event: NewsEvent | None, *, raise_exc: Exception | None = None) -> None:
        self._event = event
        self._raise = raise_exc

    async def find_active_event(
        self,
        *,
        now: datetime,
        window_before_seconds: int = 30 * 60,
        window_after_seconds: int = 30 * 60,
        currencies=None,
    ) -> NewsEvent | None:
        if self._raise is not None:
            raise self._raise
        return self._event


def _make_event(
    *,
    name: str = "Non-Farm Payrolls",
    currency: str = "USD",
    intensity: str = "high",
    actual: float = 300_000.0,
    forecast: float = 200_000.0,
) -> NewsEvent:
    return NewsEvent(
        event_id="test-evt",
        name=name,
        currency=currency,
        intensity=intensity,  # type: ignore[arg-type]
        scheduled_at=_NOW,
        actual=actual,
        forecast=forecast,
        previous=180_000.0,
    )


def test_news_provider_returns_none_when_no_active_event() -> None:
    engine = _StubNewsEngine(None)
    exposure = _StubExposureProvider(0.0)
    provider = NewsEngineFeaturesProvider(
        engine, exposure, clock=lambda: _NOW,
    )
    assert provider.get_news_classification(_SYMBOL) is None


def test_news_provider_classifies_when_event_active() -> None:
    engine = _StubNewsEngine(_make_event())
    exposure = _StubExposureProvider(1.0)  # long
    provider = NewsEngineFeaturesProvider(
        engine, exposure, clock=lambda: _NOW,
    )
    classification = provider.get_news_classification(_SYMBOL)
    assert isinstance(classification, NewsClassification)
    assert classification.event.name == "Non-Farm Payrolls"
    # USD-strong NFP + long XAU → expected XAU down → against.
    assert classification.direction == "against"


def test_news_provider_classifies_neutral_when_flat_exposure() -> None:
    engine = _StubNewsEngine(_make_event())
    exposure = _StubExposureProvider(0.0)  # flat
    provider = NewsEngineFeaturesProvider(
        engine, exposure, clock=lambda: _NOW,
    )
    classification = provider.get_news_classification(_SYMBOL)
    assert classification is not None
    assert classification.direction == "neutral"


def test_news_provider_raises_news_unavailable_on_crawler_error() -> None:
    engine = _StubNewsEngine(None, raise_exc=RuntimeError("network down"))
    exposure = _StubExposureProvider(0.0)
    provider = NewsEngineFeaturesProvider(
        engine, exposure, clock=lambda: _NOW,
    )
    with pytest.raises(NewsUnavailable, match="network down"):
        provider.get_news_classification(_SYMBOL)


def test_news_provider_falls_back_flat_when_exposure_provider_fails() -> None:
    """Robustness: if the exposure provider crashes, default to flat
    (==> direction=neutral) rather than dropping the entire event."""

    class _BrokenExposure:
        def get_exposure_lots(self, symbol: str) -> float:
            raise RuntimeError("broker unreachable")

    engine = _StubNewsEngine(_make_event())
    provider = NewsEngineFeaturesProvider(
        engine, _BrokenExposure(), clock=lambda: _NOW,
    )
    classification = provider.get_news_classification(_SYMBOL)
    assert classification is not None
    # Flat → neutral (no directional impact recorded).
    assert classification.direction == "neutral"


# ---------------------------------------------------------------------------
# ForexDataLakeReplaySource
# ---------------------------------------------------------------------------


class _StubLake:
    """ForexDataLake-shaped stub returning a fixed DataFrame per slice."""

    def __init__(self, df: pl.DataFrame) -> None:
        self._df = df

    def query(self, instrument, timeframe, start, end):
        return self._df.filter(
            (pl.col("ts") >= start) & (pl.col("ts") < end)
        )

    def available_range(self, instrument, timeframe):
        return None  # unused by ReplaySource


def _ohlcv_df(start: datetime, *, n_bars: int) -> pl.DataFrame:
    """Build a minimal OHLCV DataFrame with the schema the lake emits."""
    return pl.DataFrame(
        {
            "ts": pl.Series(
                [start + timedelta(minutes=5 * i) for i in range(n_bars)],
                dtype=pl.Datetime("ns", "UTC"),
            ),
            "open": [2300.0 + i for i in range(n_bars)],
            "high": [2305.0 + i for i in range(n_bars)],
            "low": [2295.0 + i for i in range(n_bars)],
            "close": [2302.0 + i for i in range(n_bars)],
            "volume": [100.0] * n_bars,
        }
    )


def test_replay_source_iterates_one_observation_per_test_window() -> None:
    """7-train/1-test/1-step over 14 days → 7 observations."""
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = start + timedelta(days=14)
    df = _ohlcv_df(start, n_bars=14 * 288)  # 14 days × 288 M5 bars/day
    lake = _StubLake(df)

    fixed_features = MarketFeatures(
        volatility_rank=0.4,
        hh_count=2,
        ll_count=1,
        h4_trend_bars=4,
        regime="TREND_UP",
    )

    def extractor(bars: pl.DataFrame) -> MarketFeatures:
        assert not bars.is_empty()
        return fixed_features

    source = ForexDataLakeReplaySource(
        lake, "XAUUSD", extractor,
    )
    obs = source.iter_observations(
        start=start,
        end=end,
        grain="day",
        train_grains=7,
        test_grains=1,
        step_grains=1,
    )
    assert len(obs) == 7
    # First obs has prev_regime=None (cold start).
    assert obs[0].prev_regime is None
    # Subsequent obs carry forward.
    assert obs[1].prev_regime == "TREND_UP"
    # All observations carry the fixed features.
    assert all(o.features == fixed_features for o in obs)


def test_replay_source_skips_empty_data_windows() -> None:
    """Window with no bars → skipped entirely (don't call extractor)."""
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = start + timedelta(days=10)
    # Only first 3 days have bars; later windows return empty.
    df = _ohlcv_df(start, n_bars=3 * 288)
    lake = _StubLake(df)

    extractor_calls: list[int] = []

    def extractor(bars: pl.DataFrame) -> MarketFeatures:
        extractor_calls.append(len(bars))
        return MarketFeatures(
            volatility_rank=0.3,
            hh_count=1,
            ll_count=1,
            h4_trend_bars=2,
            regime="CONSOLIDATION",
        )

    source = ForexDataLakeReplaySource(lake, "XAUUSD", extractor)
    obs = source.iter_observations(
        start=start,
        end=end,
        grain="day",
        train_grains=1,
        test_grains=1,
        step_grains=1,
    )
    # Only days 1-2 have bars in the test window after train_grains=1.
    assert len(obs) > 0
    assert len(obs) <= 3
    # Extractor only called for non-empty test slices.
    assert all(n > 0 for n in extractor_calls)


def test_replay_source_passes_news_and_exposure_lookups() -> None:
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = start + timedelta(days=10)
    df = _ohlcv_df(start, n_bars=10 * 288)
    lake = _StubLake(df)

    features = MarketFeatures(
        volatility_rank=0.5,
        hh_count=2,
        ll_count=2,
        h4_trend_bars=3,
        regime="TREND_UP",
    )

    def news_lookup(ts: datetime) -> None:
        # Always None — the test confirms the wiring path, not the lookup logic.
        return None

    def exposure_lookup(ts: datetime) -> float:
        return 0.5

    source = ForexDataLakeReplaySource(
        lake,
        "XAUUSD",
        lambda bars: features,
        news_lookup=news_lookup,
        exposure_lookup=exposure_lookup,
    )
    obs = source.iter_observations(
        start=start,
        end=end,
        grain="day",
        train_grains=7,
        test_grains=1,
        step_grains=1,
    )
    assert len(obs) > 0
    for o in obs:
        assert o.current_exposure_lots == 0.5
        assert o.news_classification is None


# ---------------------------------------------------------------------------
# Integration: three adapters wired into decision_server.create_app
# ---------------------------------------------------------------------------


class _StubFeaturesProvider:
    def __init__(self, features: MarketFeatures) -> None:
        self._features = features

    def get_features(self, symbol: str) -> MarketFeatures:
        return self._features


def test_production_wiring_smoke_three_adapters_compose() -> None:
    """Three adapters wired into decision_server → /signal returns 200
    with full envelope (proves no signature mismatch)."""
    features = MarketFeatures(
        volatility_rank=0.4,
        hh_count=2,
        ll_count=1,
        h4_trend_bars=4,
        regime="TREND_UP",
    )
    market = _StubFeaturesProvider(features)

    broker = _StubBroker(())  # flat
    exposure_provider = MT5BrokerExposureProvider(broker)

    engine = _StubNewsEngine(None)
    news_provider = NewsEngineFeaturesProvider(
        engine, exposure_provider, clock=lambda: _NOW,
    )

    store = PrevRegimeStore()
    app = create_app(
        market,
        store=store,
        news_provider=news_provider,
        exposure_provider=exposure_provider,
        enable_debate=False,  # keep test hermetic
    )

    with TestClient(app) as client:
        resp = client.get("/signal", params={"symbol": "XAUUSD"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["symbol"] == "XAUUSD"
        assert body["regime"] == "trend_up"
        # No active news event (engine returned None) → safe default.
        assert body["news_intensity"] == "none"
        # No positions in broker → no exposure → exit_directive 'none'.
        assert body["exit_directive"] == "none"


def test_production_wiring_status_endpoint_reports_attached_providers() -> None:
    market = _StubFeaturesProvider(
        MarketFeatures(
            volatility_rank=0.4,
            hh_count=2,
            ll_count=1,
            h4_trend_bars=4,
            regime="TREND_UP",
        )
    )
    broker = _StubBroker(())
    exposure_provider = MT5BrokerExposureProvider(broker)
    news_provider = NewsEngineFeaturesProvider(
        _StubNewsEngine(None), exposure_provider, clock=lambda: _NOW,
    )
    app = create_app(
        market,
        news_provider=news_provider,
        exposure_provider=exposure_provider,
        enable_debate=False,
    )
    with TestClient(app) as client:
        resp = client.get("/status")
        assert resp.status_code == 200
        body = resp.json()
        assert body["news_provider_attached"] is True
        assert body["exposure_provider_attached"] is True
        assert body["debate_enabled"] is False


# ---------------------------------------------------------------------------
# LiquiditySweepProvider (Phase 5 议题 1) integration
# ---------------------------------------------------------------------------


class _StubLiquidityProvider:
    """Returns a canned LiquiditySweepReversal-shaped object."""

    def __init__(self, payload) -> None:
        self._payload = payload

    def get_active_sweep(self, symbol: str):
        return self._payload


class _SweepPayload:
    """Lightweight stand-in for LiquiditySweepReversal — duck-typed."""

    def __init__(self, *, active, direction, distance_pts, confidence) -> None:
        self.active = active
        self.direction = direction
        self.distance_pts = distance_pts
        self.confidence = confidence


def test_liquidity_provider_active_sweep_fills_envelope_fields() -> None:
    market = _StubFeaturesProvider(
        MarketFeatures(
            volatility_rank=0.4,
            hh_count=2,
            ll_count=1,
            h4_trend_bars=4,
            regime="TREND_UP",
        )
    )
    sweep = _SweepPayload(
        active=True,
        direction="bullish_reversal",
        distance_pts=42.5,
        confidence=0.71,
    )
    liquidity_provider = _StubLiquidityProvider(sweep)
    app = create_app(
        market,
        liquidity_provider=liquidity_provider,
        enable_debate=False,
    )
    with TestClient(app) as client:
        resp = client.get("/signal", params={"symbol": "XAUUSD"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["liquidity_sweep_active"] is True
        assert body["liquidity_sweep_direction"] == "bullish_reversal"
        assert body["liquidity_sweep_distance_pts"] == pytest.approx(42.5)
        assert body["liquidity_sweep_confidence"] == pytest.approx(0.71)


def test_liquidity_provider_inactive_sweep_keeps_fields_null() -> None:
    market = _StubFeaturesProvider(
        MarketFeatures(
            volatility_rank=0.4,
            hh_count=2,
            ll_count=1,
            h4_trend_bars=4,
            regime="TREND_UP",
        )
    )
    inactive = _SweepPayload(
        active=False,
        direction=None,
        distance_pts=None,
        confidence=None,
    )
    app = create_app(
        market,
        liquidity_provider=_StubLiquidityProvider(inactive),
        enable_debate=False,
    )
    with TestClient(app) as client:
        body = client.get("/signal", params={"symbol": "XAUUSD"}).json()
        # active=False → envelope fields stay None (safe default).
        assert body["liquidity_sweep_active"] is None
        assert body["liquidity_sweep_direction"] is None


def test_liquidity_provider_none_returns_null_fields() -> None:
    """provider returning None → envelope fields None."""
    market = _StubFeaturesProvider(
        MarketFeatures(
            volatility_rank=0.4,
            hh_count=2,
            ll_count=1,
            h4_trend_bars=4,
            regime="TREND_UP",
        )
    )
    app = create_app(
        market,
        liquidity_provider=_StubLiquidityProvider(None),
        enable_debate=False,
    )
    with TestClient(app) as client:
        body = client.get("/signal", params={"symbol": "XAUUSD"}).json()
        assert body["liquidity_sweep_active"] is None
        assert body["liquidity_sweep_direction"] is None
        assert body["liquidity_sweep_distance_pts"] is None
        assert body["liquidity_sweep_confidence"] is None


def test_liquidity_provider_crash_degrades_to_null() -> None:
    """If the provider raises, the envelope still ships with safe defaults."""

    class _CrashingProvider:
        def get_active_sweep(self, symbol):
            raise RuntimeError("detector crashed")

    market = _StubFeaturesProvider(
        MarketFeatures(
            volatility_rank=0.4,
            hh_count=2,
            ll_count=1,
            h4_trend_bars=4,
            regime="TREND_UP",
        )
    )
    app = create_app(
        market,
        liquidity_provider=_CrashingProvider(),
        enable_debate=False,
    )
    with TestClient(app) as client:
        resp = client.get("/signal", params={"symbol": "XAUUSD"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["liquidity_sweep_active"] is None


def test_status_reports_liquidity_provider_attached() -> None:
    market = _StubFeaturesProvider(
        MarketFeatures(
            volatility_rank=0.4,
            hh_count=2,
            ll_count=1,
            h4_trend_bars=4,
            regime="TREND_UP",
        )
    )
    app = create_app(
        market,
        liquidity_provider=_StubLiquidityProvider(None),
        enable_debate=False,
    )
    with TestClient(app) as client:
        body = client.get("/status").json()
        assert body["liquidity_provider_attached"] is True


def test_status_liquidity_provider_default_false_when_unattached() -> None:
    market = _StubFeaturesProvider(
        MarketFeatures(
            volatility_rank=0.4,
            hh_count=2,
            ll_count=1,
            h4_trend_bars=4,
            regime="TREND_UP",
        )
    )
    app = create_app(market, enable_debate=False)
    with TestClient(app) as client:
        body = client.get("/status").json()
        assert body["liquidity_provider_attached"] is False


# ---------------------------------------------------------------------------
# FilterInputsProvider — Phase 1 Injection #3a externalized
# ---------------------------------------------------------------------------


class _StubFilterInputsProvider:
    """Test double for FilterInputsProvider — returns a fixed payload."""

    def __init__(self, payload) -> None:
        self._payload = payload
        self.calls: list[str] = []

    def get_filter_inputs(self, symbol: str):
        self.calls.append(symbol)
        return self._payload


def _trend_features() -> MarketFeatures:
    return MarketFeatures(
        volatility_rank=0.4,
        hh_count=2,
        ll_count=1,
        h4_trend_bars=4,
        regime="TREND_UP",
    )


def test_filter_inputs_provider_safe_inputs_keep_lock_unchanged() -> None:
    """Safe FilterInputs (= no triggers) → transition_lock not pushed forward."""
    from smc.hedgerock.mock_provider import default_safe_filter_inputs

    market = _StubFeaturesProvider(_trend_features())
    provider = _StubFilterInputsProvider(default_safe_filter_inputs())
    app = create_app(
        market,
        filter_inputs_provider=provider,
        enable_debate=False,
    )
    with TestClient(app) as client:
        resp = client.get("/signal", params={"symbol": "XAUUSD"})
        assert resp.status_code == 200
        body = resp.json()
        # No regime change yet (first call) and filters silent → no lock.
        assert body["transition_lock_until_ts"] is None
        assert provider.calls == ["XAUUSD"]


def test_filter_inputs_provider_halt_extends_transition_lock() -> None:
    """Halting FilterInputs → transition_lock_until_ts is in the future."""
    from smc.hedgerock.regime_filters import FilterInputs

    # h4_trend > 2% threshold → triggers Filter A.
    halt_inputs = FilterInputs(
        h4_close_now=110.0,
        h4_close_lookback=100.0,  # 10% trend (well above 2%)
        h1_atr_now=0.10,
        h1_atr_lookback_avg=0.10,
        h1_recent_high=100.25,
        h1_recent_low=99.75,
        h1_reference_high=100.50,
        h1_reference_low=99.50,
    )
    market = _StubFeaturesProvider(_trend_features())
    app = create_app(
        market,
        filter_inputs_provider=_StubFilterInputsProvider(halt_inputs),
        enable_debate=False,
    )
    with TestClient(app) as client:
        resp = client.get("/signal", params={"symbol": "XAUUSD"})
        assert resp.status_code == 200
        body = resp.json()
        lock = body["transition_lock_until_ts"]
        assert lock is not None, "filter halt must push the transition_lock forward"
        # The lock should be ≈ now + 5min — well above the response timestamp.
        from datetime import datetime as _dt
        lock_dt = _dt.fromisoformat(lock.replace("Z", "+00:00"))
        gen_dt = _dt.fromisoformat(body["generated_at"].replace("Z", "+00:00"))
        delta = (lock_dt - gen_dt).total_seconds()
        assert 4 * 60 <= delta <= 6 * 60, f"expected ~5min lock, got {delta}s"


def test_filter_inputs_provider_none_returns_no_halt() -> None:
    """provider=None → endpoint behaves identically to pre-#3a path."""
    market = _StubFeaturesProvider(_trend_features())
    app = create_app(market, enable_debate=False)  # no filter provider
    with TestClient(app) as client:
        body = client.get("/signal", params={"symbol": "XAUUSD"}).json()
        assert body["transition_lock_until_ts"] is None


def test_filter_inputs_provider_crash_degrades_to_no_halt() -> None:
    """Provider raises → endpoint still ships envelope (no halt)."""

    class _CrashingProvider:
        def get_filter_inputs(self, symbol):
            raise RuntimeError("data lake crashed")

    market = _StubFeaturesProvider(_trend_features())
    app = create_app(
        market,
        filter_inputs_provider=_CrashingProvider(),
        enable_debate=False,
    )
    with TestClient(app) as client:
        resp = client.get("/signal", params={"symbol": "XAUUSD"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["transition_lock_until_ts"] is None


def test_filter_inputs_provider_returns_none_keeps_lock_unchanged() -> None:
    """provider.get_filter_inputs returns None → no halt evaluation."""
    market = _StubFeaturesProvider(_trend_features())
    app = create_app(
        market,
        filter_inputs_provider=_StubFilterInputsProvider(None),
        enable_debate=False,
    )
    with TestClient(app) as client:
        body = client.get("/signal", params={"symbol": "XAUUSD"}).json()
        assert body["transition_lock_until_ts"] is None


def test_status_reports_filter_inputs_provider_attached() -> None:
    market = _StubFeaturesProvider(_trend_features())
    app = create_app(
        market,
        filter_inputs_provider=_StubFilterInputsProvider(None),
        enable_debate=False,
    )
    with TestClient(app) as client:
        body = client.get("/status").json()
        assert body["filter_inputs_provider_attached"] is True


def test_status_filter_inputs_provider_default_false_when_unattached() -> None:
    market = _StubFeaturesProvider(_trend_features())
    app = create_app(market, enable_debate=False)
    with TestClient(app) as client:
        body = client.get("/status").json()
        assert body["filter_inputs_provider_attached"] is False
