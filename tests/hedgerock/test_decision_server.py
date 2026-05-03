"""Tests for hedgerock.decision_server."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from smc.ai.models import MarketRegimeAI
from smc.hedgerock.decision_server import (
    DEFAULT_PORT,
    FeaturesUnavailable,
    MarketFeatures,
    PrevRegimeStore,
    build_envelope,
    build_strategy_id,
    create_app,
)
from smc.hedgerock.schemas import SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeProvider:
    """Test double for `MarketFeaturesProvider`."""

    def __init__(self, features: MarketFeatures | None = None) -> None:
        self.features = features
        self.unavailable = False
        self.calls: list[str] = []

    def get_features(self, symbol: str) -> MarketFeatures:
        self.calls.append(symbol)
        if self.unavailable:
            raise FeaturesUnavailable(f"forced unavailable for {symbol}")
        if self.features is None:
            raise FeaturesUnavailable("no features set in fake")
        return self.features


@pytest.fixture
def trend_features() -> MarketFeatures:
    return MarketFeatures(
        volatility_rank=0.5,
        hh_count=8,
        ll_count=0,
        h4_trend_bars=5,
        regime="TREND_UP",
    )


@pytest.fixture
def now_utc() -> datetime:
    return datetime(2026, 4, 26, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# build_strategy_id
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_strategy_id_format() -> None:
    assert build_strategy_id("XAUUSD", "H1", "TREND_UP") == "xauusd_h1_trend_up"


@pytest.mark.unit
def test_strategy_id_lowercases_everything() -> None:
    sid = build_strategy_id("XAUUSD", "M15", "ATH_BREAKOUT")
    assert sid == sid.lower()


# ---------------------------------------------------------------------------
# build_envelope
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_build_envelope_basic(trend_features: MarketFeatures, now_utc: datetime) -> None:
    env = build_envelope(
        "XAUUSD", trend_features, prev_regime=None, now=now_utc
    )
    assert env.symbol == "XAUUSD"
    assert env.regime == "trend_up"
    assert env.prev_regime is None
    assert env.transition_lock_until_ts is None  # no prev = no lock
    assert env.active_timeframe == "H1"
    assert env.active_strategy_id == "xauusd_h1_trend_up"
    assert env.exit_directive == "none"
    assert env.grid_multiplier == 1.0
    assert env.lot_factor == 1.0
    assert env.confidence > 0


@pytest.mark.unit
def test_build_envelope_with_prev_same_regime_no_lock(
    trend_features: MarketFeatures, now_utc: datetime
) -> None:
    env = build_envelope(
        "XAUUSD", trend_features, prev_regime="TREND_UP", now=now_utc
    )
    assert env.transition_lock_until_ts is None


@pytest.mark.unit
def test_build_envelope_with_prev_extreme_regime_change_locks(
    trend_features: MarketFeatures, now_utc: datetime
) -> None:
    env = build_envelope(
        "XAUUSD", trend_features, prev_regime="TREND_DOWN", now=now_utc
    )
    assert env.transition_lock_until_ts is not None
    delta = (env.transition_lock_until_ts - now_utc).total_seconds()
    assert delta == pytest.approx(7200.0)


@pytest.mark.unit
def test_build_envelope_extreme_volatility_picks_m5() -> None:
    feats = MarketFeatures(
        volatility_rank=0.95,
        hh_count=0,
        ll_count=0,
        h4_trend_bars=0,
        regime="ATH_BREAKOUT",
    )
    env = build_envelope(
        "XAUUSD", feats, prev_regime=None, now=datetime.now(timezone.utc)
    )
    assert env.active_timeframe == "M5"
    # Phase C-hotfix #3: active_strategy_id uses v2 regime enum (lowercase
    # "breakout") — was legacy MarketRegimeAI "ath_breakout" before.
    assert "breakout" in env.active_strategy_id


# ---------------------------------------------------------------------------
# PrevRegimeStore
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_store_returns_none_when_unset() -> None:
    store = PrevRegimeStore()
    assert store.get("XAUUSD") is None


@pytest.mark.unit
def test_store_round_trip() -> None:
    store = PrevRegimeStore()
    store.set("XAUUSD", "TREND_UP")
    assert store.get("XAUUSD") == "TREND_UP"


@pytest.mark.unit
def test_store_overwrites() -> None:
    store = PrevRegimeStore()
    store.set("XAUUSD", "TREND_UP")
    store.set("XAUUSD", "CONSOLIDATION")
    assert store.get("XAUUSD") == "CONSOLIDATION"


@pytest.mark.unit
def test_store_isolates_symbols() -> None:
    store = PrevRegimeStore()
    store.set("XAUUSD", "TREND_UP")
    store.set("BTCUSD", "TREND_DOWN")
    assert store.get("XAUUSD") == "TREND_UP"
    assert store.get("BTCUSD") == "TREND_DOWN"


# ---------------------------------------------------------------------------
# FastAPI: /healthz
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_healthz_returns_ok(trend_features: MarketFeatures) -> None:
    app = create_app(FakeProvider(trend_features))
    client = TestClient(app)
    resp = client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["schema_version"] == SCHEMA_VERSION


# ---------------------------------------------------------------------------
# FastAPI: /signal happy path
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_signal_returns_envelope(trend_features: MarketFeatures) -> None:
    provider = FakeProvider(trend_features)
    app = create_app(provider)
    client = TestClient(app)
    resp = client.get("/signal", params={"symbol": "XAUUSD"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["symbol"] == "XAUUSD"
    assert body["regime"] == "trend_up"
    assert body["active_timeframe"] == "H1"
    assert body["schema_version"] == SCHEMA_VERSION
    assert provider.calls == ["XAUUSD"]


@pytest.mark.integration
def test_signal_lowercase_symbol_normalised(trend_features: MarketFeatures) -> None:
    provider = FakeProvider(trend_features)
    app = create_app(provider)
    client = TestClient(app)
    resp = client.get("/signal", params={"symbol": "xauusd"})
    assert resp.status_code == 200
    assert resp.json()["symbol"] == "XAUUSD"
    assert provider.calls == ["XAUUSD"]


@pytest.mark.integration
def test_signal_persists_regime_for_next_call(trend_features: MarketFeatures) -> None:
    provider = FakeProvider(trend_features)
    store = PrevRegimeStore()
    app = create_app(provider, store)
    client = TestClient(app)

    # First call → no prev, no lock.
    r1 = client.get("/signal", params={"symbol": "XAUUSD"}).json()
    assert r1["prev_regime"] is None
    assert r1["transition_lock_until_ts"] is None
    assert store.get("XAUUSD") == "TREND_UP"

    # Switch regime to DOWN.
    provider.features = MarketFeatures(
        volatility_rank=0.5,
        hh_count=0,
        ll_count=8,
        h4_trend_bars=5,
        regime="TREND_DOWN",
    )
    # Second call should see prev=TREND_UP, regime=TREND_DOWN, extreme lock.
    r2 = client.get("/signal", params={"symbol": "XAUUSD"}).json()
    assert r2["prev_regime"] == "trend_up"
    assert r2["regime"] == "trend_down"
    assert r2["transition_lock_until_ts"] is not None


# ---------------------------------------------------------------------------
# FastAPI: /signal error paths
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_signal_unknown_symbol_404(trend_features: MarketFeatures) -> None:
    app = create_app(FakeProvider(trend_features))
    client = TestClient(app)
    resp = client.get("/signal", params={"symbol": "UNKNOWN"})
    assert resp.status_code == 404


@pytest.mark.integration
def test_signal_features_unavailable_503() -> None:
    provider = FakeProvider()
    provider.unavailable = True
    app = create_app(provider)
    client = TestClient(app)
    resp = client.get("/signal", params={"symbol": "XAUUSD"})
    assert resp.status_code == 503


@pytest.mark.integration
def test_signal_missing_symbol_param_422() -> None:
    app = create_app(FakeProvider())
    client = TestClient(app)
    resp = client.get("/signal")
    assert resp.status_code == 422  # pydantic validation


@pytest.mark.integration
def test_signal_empty_symbol_422() -> None:
    app = create_app(FakeProvider())
    client = TestClient(app)
    resp = client.get("/signal", params={"symbol": ""})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# FastAPI: /status
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_status_lists_whitelist(trend_features: MarketFeatures) -> None:
    app = create_app(
        FakeProvider(trend_features),
        symbol_whitelist=frozenset({"XAUUSD", "BTCUSD"}),
    )
    client = TestClient(app)
    body = client.get("/status").json()
    assert body["schema_version"] == SCHEMA_VERSION
    assert sorted(body["symbols"]) == ["BTCUSD", "XAUUSD"]
    assert body["tracked_regimes"] == {"BTCUSD": None, "XAUUSD": None}


@pytest.mark.integration
def test_status_reflects_recent_signals(trend_features: MarketFeatures) -> None:
    provider = FakeProvider(trend_features)
    store = PrevRegimeStore()
    app = create_app(provider, store)
    client = TestClient(app)
    client.get("/signal", params={"symbol": "XAUUSD"})
    body = client.get("/status").json()
    assert body["tracked_regimes"]["XAUUSD"] == "TREND_UP"


# ---------------------------------------------------------------------------
# Sanity
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_default_port_distinct_from_strategy_server() -> None:
    """8788 must not collide with strategy_server.py's 8080."""
    assert DEFAULT_PORT != 8080
    assert 1024 < DEFAULT_PORT < 65536


@pytest.mark.unit
def test_unsupported_regime_in_features_maps_to_unknown() -> None:
    """v2.0.0: unrecognized legacy regime → envelope.regime='unknown' (safe).

    Previous v1 contract raised ValidationError; v2 prefers degrading to
    'unknown' so a flaky upstream classifier cannot bring the EA's polling
    down. The EA treats regime='unknown' as observe-mode (no new entries).
    """
    bad = MarketFeatures(
        volatility_rank=0.5,
        hh_count=0,
        ll_count=0,
        h4_trend_bars=0,
        regime="NOT_A_REGIME",  # type: ignore[arg-type]
    )
    env = build_envelope(
        "XAUUSD", bad, prev_regime=None, now=datetime.now(timezone.utc)
    )
    assert env.regime == "unknown"


# ---------------------------------------------------------------------------
# Phase 3.4 — news_classification + exit_decision wiring (build_envelope)
# ---------------------------------------------------------------------------


from smc.hedgerock.exit_decider import ExitDecision  # noqa: E402
from smc.hedgerock.news_classifier import NewsClassification  # noqa: E402
from smc.hedgerock.news_engine import NewsEvent  # noqa: E402


def _ev(name: str = "Non-Farm Payrolls", *, intensity: str = "high") -> NewsEvent:
    return NewsEvent(
        event_id="ev1",
        name=name,
        currency="USD",
        intensity=intensity,  # type: ignore[arg-type]
        scheduled_at=datetime(2026, 4, 26, 12, 0, tzinfo=timezone.utc),
        actual=272_000.0,
        forecast=200_000.0,
    )


def _cls(*, intensity: str = "high", direction: str = "against") -> NewsClassification:
    return NewsClassification(
        event=_ev(intensity=intensity),
        direction=direction,  # type: ignore[arg-type]
        surprise_score=0.36,
        impact_currency="USD",
        classifier_version="v1.0.0",
    )


def _exit(directive: str = "urgent_take_profit") -> ExitDecision:
    return ExitDecision(
        directive=directive,  # type: ignore[arg-type]
        source="hard_rule",
        rationale="test",
        cost_usd=0.0,
        elapsed_ms=1,
    )


@pytest.mark.unit
def test_build_envelope_news_classification_propagates_into_envelope(
    trend_features: MarketFeatures, now_utc: datetime
) -> None:
    cls = _cls(intensity="high", direction="against")
    env = build_envelope(
        "XAUUSD",
        trend_features,
        prev_regime="TREND_UP",
        now=now_utc,
        news_classification=cls,
    )
    assert env.news_intensity == "high"
    assert env.news_direction == "against"
    assert env.news_event_name == "Non-Farm Payrolls"


@pytest.mark.unit
def test_build_envelope_no_news_keeps_safe_defaults(
    trend_features: MarketFeatures, now_utc: datetime
) -> None:
    """Phase-1 backwards compatibility: no news_classification → defaults."""
    env = build_envelope(
        "XAUUSD",
        trend_features,
        prev_regime="TREND_UP",
        now=now_utc,
        news_classification=None,
    )
    assert env.news_intensity == "none"
    assert env.news_direction is None
    assert env.news_event_name is None


@pytest.mark.unit
def test_build_envelope_high_against_long_yields_urgent_take_profit(
    trend_features: MarketFeatures, now_utc: datetime
) -> None:
    """Lead [GO] now wires decide_exit *inside* build_envelope. With
    high+against news + long exposure the hard-rule path fires
    urgent_take_profit without ever calling chat_fn."""

    def _no_chat(*_a: object, **_kw: object) -> tuple[str, int, float]:
        raise AssertionError("chat must not be reached on hard-rule path")

    cls = _cls(intensity="high", direction="against")
    env = build_envelope(
        "XAUUSD",
        trend_features,
        prev_regime="TREND_UP",
        now=now_utc,
        news_classification=cls,
        current_exposure_lots=1.0,
        exit_decider_chat_fn=_no_chat,
    )
    assert env.exit_directive == "urgent_take_profit"


@pytest.mark.unit
def test_build_envelope_no_news_no_exposure_yields_none_directive(
    trend_features: MarketFeatures, now_utc: datetime
) -> None:
    """Phase-1 backwards compatibility: no news_classification, flat
    exposure → hard-rule emits 'none' without ever calling chat_fn."""

    def _no_chat(*_a: object, **_kw: object) -> tuple[str, int, float]:
        raise AssertionError("chat must not be reached on hard-rule path")

    env = build_envelope(
        "XAUUSD",
        trend_features,
        prev_regime="TREND_UP",
        now=now_utc,
        exit_decider_chat_fn=_no_chat,
    )
    assert env.exit_directive == "none"


# ---------------------------------------------------------------------------
# Phase 3.4 — endpoint orchestration with NewsFeaturesProvider + ExposureProvider
# ---------------------------------------------------------------------------


from smc.hedgerock.decision_server import (  # noqa: E402
    ExposureProvider,
    NewsFeaturesProvider,
    NewsUnavailable,
    _safe_get_exposure,
    _safe_get_news_classification,
)


class FakeNewsProvider:
    """Test double for ``NewsFeaturesProvider``.

    Per [GO] contract returns a pre-built ``NewsClassification`` rather
    than a raw event.
    """

    def __init__(self, classification: NewsClassification | None = None) -> None:
        self.classification = classification
        self.calls: list[str] = []
        self.raise_unavailable = False
        self.raise_other = False

    def get_news_classification(self, symbol: str) -> NewsClassification | None:
        self.calls.append(symbol)
        if self.raise_unavailable:
            raise NewsUnavailable(f"forced unavailable for {symbol}")
        if self.raise_other:
            raise RuntimeError("crashed")
        return self.classification


class FakeExposureProvider:
    """Test double for ``ExposureProvider``."""

    def __init__(self, lots: float = 0.0) -> None:
        self.lots = lots
        self.raise_other = False

    def get_exposure_lots(self, symbol: str) -> float:
        if self.raise_other:
            raise RuntimeError("crashed")
        return self.lots


@pytest.mark.unit
def test_safe_get_news_classification_returns_none_when_provider_missing() -> None:
    assert _safe_get_news_classification(None, "XAUUSD") is None


@pytest.mark.unit
def test_safe_get_news_classification_passes_through() -> None:
    fake = FakeNewsProvider(classification=_cls())
    out = _safe_get_news_classification(fake, "XAUUSD")
    assert out is not None
    assert out.event.name == "Non-Farm Payrolls"
    assert fake.calls == ["XAUUSD"]


@pytest.mark.unit
def test_safe_get_news_classification_swallows_news_unavailable() -> None:
    fake = FakeNewsProvider()
    fake.raise_unavailable = True
    assert _safe_get_news_classification(fake, "XAUUSD") is None


@pytest.mark.unit
def test_safe_get_news_classification_swallows_arbitrary_exception() -> None:
    fake = FakeNewsProvider()
    fake.raise_other = True
    assert _safe_get_news_classification(fake, "XAUUSD") is None


@pytest.mark.unit
def test_safe_get_exposure_returns_zero_when_provider_missing() -> None:
    assert _safe_get_exposure(None, "XAUUSD") == 0.0


@pytest.mark.unit
def test_safe_get_exposure_returns_provider_lots() -> None:
    fake = FakeExposureProvider(lots=2.5)
    assert _safe_get_exposure(fake, "XAUUSD") == 2.5


@pytest.mark.unit
def test_safe_get_exposure_swallows_provider_failure() -> None:
    fake = FakeExposureProvider(lots=2.5)
    fake.raise_other = True
    assert _safe_get_exposure(fake, "XAUUSD") == 0.0


# ---------------------------------------------------------------------------
# /signal endpoint with full Phase 3.4 wiring
# ---------------------------------------------------------------------------


def _failing_chat(*_a: object, **_kw: object) -> tuple[str, int, float]:
    raise AssertionError("chat must not be reached on hard-rule path")


@pytest.mark.integration
def test_signal_with_no_news_provider_keeps_legacy_envelope_shape(
    trend_features: MarketFeatures,
) -> None:
    """No news_provider injected → envelope falls back to defaults exactly
    as Phase 1 deployments expect."""
    provider = FakeProvider(features=trend_features)
    app = create_app(provider, PrevRegimeStore())
    client = TestClient(app)
    rsp = client.get("/signal", params={"symbol": "XAUUSD"})
    assert rsp.status_code == 200
    body = rsp.json()
    assert body["news_intensity"] == "none"
    assert body["news_direction"] is None
    assert body["news_event_name"] is None
    assert body["exit_directive"] == "none"


@pytest.mark.integration
def test_signal_with_active_news_and_long_exposure_sets_urgent_tp(
    trend_features: MarketFeatures,
) -> None:
    """High-intensity USD beat + long exposure → exit_directive
    'urgent_take_profit' via the hard-rule path (no LLM)."""
    provider = FakeProvider(features=trend_features)
    news = FakeNewsProvider(
        classification=_cls(intensity="high", direction="against")
    )
    expo = FakeExposureProvider(lots=1.0)
    app = create_app(
        provider,
        PrevRegimeStore(),
        news_provider=news,
        exposure_provider=expo,
        exit_decider_chat_fn=_failing_chat,  # must not be reached
    )
    client = TestClient(app)
    rsp = client.get("/signal", params={"symbol": "XAUUSD"})
    assert rsp.status_code == 200
    body = rsp.json()
    assert body["news_intensity"] == "high"
    assert body["news_direction"] == "against"
    assert body["news_event_name"] == "Non-Farm Payrolls"
    assert body["exit_directive"] == "urgent_take_profit"


@pytest.mark.integration
def test_signal_news_provider_failure_does_not_break_envelope(
    trend_features: MarketFeatures,
) -> None:
    """A flaky news provider degrades the envelope gracefully."""
    provider = FakeProvider(features=trend_features)
    news = FakeNewsProvider()
    news.raise_unavailable = True
    app = create_app(
        provider,
        PrevRegimeStore(),
        news_provider=news,
        exposure_provider=FakeExposureProvider(lots=1.0),
        exit_decider_chat_fn=_failing_chat,  # not reached without news
    )
    client = TestClient(app)
    rsp = client.get("/signal", params={"symbol": "XAUUSD"})
    assert rsp.status_code == 200
    body = rsp.json()
    assert body["news_intensity"] == "none"
    # No news + long exposure + same regime → hard rule emits 'none'.
    assert body["exit_directive"] == "none"


@pytest.mark.integration
def test_signal_exposure_provider_failure_assumes_flat(
    trend_features: MarketFeatures,
) -> None:
    """Broker failure should not crash polling; we fall back to flat."""
    provider = FakeProvider(features=trend_features)
    expo = FakeExposureProvider(lots=1.0)
    expo.raise_other = True
    app = create_app(
        provider,
        PrevRegimeStore(),
        exposure_provider=expo,
        exit_decider_chat_fn=_failing_chat,
    )
    client = TestClient(app)
    rsp = client.get("/signal", params={"symbol": "XAUUSD"})
    assert rsp.status_code == 200
    # Flat exposure + no news → exit directive defaults to 'none'.
    assert rsp.json()["exit_directive"] == "none"


@pytest.mark.integration
def test_signal_extreme_regime_flip_forces_halt() -> None:
    """Pre-warm store with TREND_DOWN, then send TREND_UP features + short
    exposure → exit_decider hard-rule fires halt_and_close_all."""
    feats = MarketFeatures(
        volatility_rank=0.5,
        hh_count=8,
        ll_count=0,
        h4_trend_bars=5,
        regime="TREND_UP",
    )
    provider = FakeProvider(features=feats)
    store = PrevRegimeStore()
    store.set("XAUUSD", "TREND_DOWN")
    expo = FakeExposureProvider(lots=-1.0)  # short — opposed to new TREND_UP
    app = create_app(
        provider,
        store,
        exposure_provider=expo,
        exit_decider_chat_fn=_failing_chat,
    )
    client = TestClient(app)
    rsp = client.get("/signal", params={"symbol": "XAUUSD"})
    assert rsp.status_code == 200
    assert rsp.json()["exit_directive"] == "halt_and_close_all"


@pytest.mark.integration
def test_status_reports_provider_attachment_flags(
    trend_features: MarketFeatures,
) -> None:
    """The /status endpoint surfaces whether providers are wired in."""
    provider = FakeProvider(features=trend_features)
    news = FakeNewsProvider(classification=None)
    app = create_app(
        provider,
        PrevRegimeStore(),
        news_provider=news,
        enable_debate=False,
    )
    client = TestClient(app)
    body = client.get("/status").json()
    assert body["news_provider_attached"] is True
    assert body["exposure_provider_attached"] is False
    assert body["debate_enabled"] is False


@pytest.mark.integration
def test_status_default_no_providers_attached(
    trend_features: MarketFeatures,
) -> None:
    provider = FakeProvider(features=trend_features)
    app = create_app(provider, PrevRegimeStore())
    client = TestClient(app)
    body = client.get("/status").json()
    assert body["news_provider_attached"] is False
    assert body["exposure_provider_attached"] is False
    assert body["debate_enabled"] is True
