"""HedgeRock decision server — HTTP endpoint the MQL5 EA polls.

Reconstructed from consumer signatures + test contracts after the
P0/P1 merge accidentally overwrote the original untracked file. The
public surface is the union of:

  * Tests in ``tests/hedgerock/test_decision_server.py`` (most
    complete behavioural contract).
  * Consumer imports across ``src/smc/hedgerock/`` (market_state,
    mock_provider, decision_replay, replay_data_source_impl,
    news_features_provider_impl, forex_data_lake_provider,
    regime_opportunity_atlas).
  * The Tier-1 unseal additions from the self-evolution sidecar
    (:func:`get_live_parameters`, :data:`LIVE_PARAMETER_KEYS`).

The reconstruction stays faithful to the documented behaviour but
cannot guarantee bit-for-bit equality with the lost source. Anywhere
in doubt, behaviour follows the test contract.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Callable, Literal, Mapping, Protocol

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from smc.ai.cost_tracker import CostTracker
from smc.ai.models import MarketRegimeAI
from smc.hedgerock.ea_state import EAState, EAStateRecord, EAStateStore, build_ea_state
from smc.hedgerock.exit_decider import ChatFn, ExitDecision, decide_exit
from smc.hedgerock.news_classifier import NewsClassification
from smc.hedgerock.regime_filters import FilterInputs, FilterResult, compute_filters
from smc.hedgerock.schemas import (
    SCHEMA_VERSION,
    Mode,
    NewsDirection,
    NewsIntensity,
    RegimeV2,
    RiskTier,
    SignalEnvelope,
    Timeframe,
    regime_v1_to_v2,
)
from smc.hedgerock.tf_router import TimeframeRoute, route_timeframe
from smc.hedgerock.transition_lock import compute_lock_until
from types import MappingProxyType


__all__ = [
    "DEFAULT_PORT",
    "EAState",
    "EAStateRecord",
    "EAStateStore",
    "ExposureProvider",
    "FeaturesUnavailable",
    "FilterInputsProvider",
    "LIVE_PARAMETER_KEYS",
    "LiquiditySweepProvider",
    "MarketFeatures",
    "MarketFeaturesProvider",
    "NewsFeaturesProvider",
    "NewsUnavailable",
    "PrevRegimeStore",
    "PrevRegimeV2Store",
    "build_ea_state",
    "build_envelope",
    "build_strategy_id",
    "create_app",
    "get_live_parameters",
]


_LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


DEFAULT_PORT: int = 8788
"""HTTP port the EA polls. Distinct from strategy_server's 8080."""


# ---------------------------------------------------------------------------
# Public dataclasses + exceptions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MarketFeatures:
    """Current-market snapshot consumed by the decision pipeline.

    Intentionally minimal — feature enrichment happens upstream in
    ``ForexDataLakeMarketFeaturesProvider`` (or its mock). Keeping this
    class flat lets ``decision_server`` stay decoupled from the data
    source and lets tests construct a ``MarketFeatures`` inline.
    """

    volatility_rank: float
    hh_count: int
    ll_count: int
    h4_trend_bars: int
    regime: MarketRegimeAI


class FeaturesUnavailable(RuntimeError):
    """Raised by a :class:`MarketFeaturesProvider` when the underlying
    data lake / cache cannot serve a feature snapshot for the given
    symbol. ``decision_server`` returns 503 in response."""


class NewsUnavailable(RuntimeError):
    """Raised by a :class:`NewsFeaturesProvider` when the news source
    is temporarily unreachable. ``decision_server`` swallows this and
    proceeds without a news classification."""


# ---------------------------------------------------------------------------
# Protocols (provider seams)
# ---------------------------------------------------------------------------


class MarketFeaturesProvider(Protocol):
    def get_features(self, symbol: str) -> MarketFeatures: ...


class NewsFeaturesProvider(Protocol):
    def get_news_classification(
        self, symbol: str,
    ) -> NewsClassification | None: ...


class ExposureProvider(Protocol):
    def get_exposure_lots(self, symbol: str) -> float: ...


class LiquiditySweepProvider(Protocol):
    def get_liquidity_sweep(
        self, symbol: str,
    ) -> tuple[bool, str | None, float | None, float | None] | None: ...


class FilterInputsProvider(Protocol):
    def get_filter_inputs(self, symbol: str) -> FilterInputs | None: ...


# ---------------------------------------------------------------------------
# Per-symbol regime stores
# ---------------------------------------------------------------------------


class PrevRegimeStore:
    """Thread-safe per-symbol legacy ``MarketRegimeAI`` store.

    Holds the *previous* regime read by the EA on the last poll. The
    rule engine uses ``(prev, current)`` to compute the transition
    lock; the store persists across HTTP calls so transitions are
    detected across polls.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._regimes: dict[str, MarketRegimeAI] = {}

    def get(self, symbol: str) -> MarketRegimeAI | None:
        with self._lock:
            return self._regimes.get(symbol)

    def set(self, symbol: str, regime: MarketRegimeAI) -> None:
        with self._lock:
            self._regimes[symbol] = regime

    def keys(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._regimes.keys()))


class PrevRegimeV2Store:
    """Same shape as :class:`PrevRegimeStore` but keyed on the v2
    lowercase ``RegimeV2`` enum. Used by Phase D consumers that have
    already migrated to the v2 contract."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._regimes: dict[str, RegimeV2] = {}

    def get(self, symbol: str) -> RegimeV2 | None:
        with self._lock:
            return self._regimes.get(symbol)

    def set(self, symbol: str, regime: RegimeV2) -> None:
        with self._lock:
            self._regimes[symbol] = regime

    def keys(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._regimes.keys()))


# ---------------------------------------------------------------------------
# Strategy id + envelope construction
# ---------------------------------------------------------------------------


def build_strategy_id(
    symbol: str, timeframe: str | Timeframe, regime: str | MarketRegimeAI,
) -> str:
    """Compose a stable lowercase strategy id from the trio."""
    return f"{str(symbol).lower()}_{str(timeframe).lower()}_{str(regime).lower()}"


_DEFAULT_GENERATED_AT = datetime(1970, 1, 1, tzinfo=timezone.utc)


def build_envelope(
    symbol: str,
    features: MarketFeatures,
    *,
    prev_regime: MarketRegimeAI | None = None,
    now: datetime | None = None,
    news_classification: NewsClassification | None = None,
    current_exposure_lots: float = 0.0,
    liquidity_sweep: tuple[bool, str | None, float | None, float | None] | None = None,
    filter_result: FilterResult | None = None,
    enable_debate: bool = False,
    exit_decider_chat_fn: ChatFn | None = None,
    cost_tracker: CostTracker | None = None,
    # Dynamic-params overrides (rule_engine.DynamicParams.to_kwargs()).
    mode: Mode | None = None,
    hedgerock_enabled: bool | None = None,
    risk_tier: RiskTier | None = None,
    cooldown_until: datetime | None = None,
    lot_factor: float | None = None,
    grid_multiplier: float | None = None,
    max_next_lot: float | None = None,
    takeprofit_points: int | None = None,
    stoploss_points: int | None = None,
    recovery_multiplier: float | None = None,
    max_orders_buy: int | None = None,
    max_orders_sell: int | None = None,
    reason: str | None = None,
) -> SignalEnvelope:
    """Build a :class:`SignalEnvelope` from a market snapshot.

    All non-``symbol`` / ``features`` arguments are keyword-only —
    every one carries dimensional weight (timestamps, news context,
    exposure, …) and positional args risked silent argument-mixing.
    """
    ts = now if now is not None else datetime.now(timezone.utc)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)

    sym = symbol.upper()
    regime_v2: RegimeV2 = regime_v1_to_v2(features.regime) or "unknown"
    prev_v2: RegimeV2 | None = regime_v1_to_v2(prev_regime)

    route: TimeframeRoute = route_timeframe(
        volatility_rank=features.volatility_rank,
        hh_count=features.hh_count,
        ll_count=features.ll_count,
        h4_trend_bars=features.h4_trend_bars,
    )

    lock_until = compute_lock_until(prev_regime, features.regime, ts)

    # News context with safe defaults.
    news_intensity: NewsIntensity = "none"
    news_direction: NewsDirection | None = None
    news_event_name: str | None = None
    if news_classification is not None:
        news_intensity = news_classification.event.intensity
        news_direction = news_classification.direction
        news_event_name = news_classification.event.name

    # Exit directive — hard-rule first, optional debate.
    exit_decision: ExitDecision = decide_exit(
        regime=features.regime,
        prev_regime=prev_regime,
        news_classification=news_classification,
        current_exposure_lots=current_exposure_lots,
        enable_debate=enable_debate,
        chat_fn=exit_decider_chat_fn,
        cost_tracker=cost_tracker,
    )

    # Liquidity-sweep + filter-result are advisory; surface verbatim.
    sweep_active: bool | None = None
    sweep_direction: Literal["bullish_reversal", "bearish_reversal"] | None = None
    sweep_distance: float | None = None
    sweep_confidence: float | None = None
    if liquidity_sweep is not None:
        sweep_active, sweep_dir_raw, sweep_distance, sweep_confidence = liquidity_sweep
        if sweep_dir_raw in ("bullish_reversal", "bearish_reversal"):
            sweep_direction = sweep_dir_raw  # type: ignore[assignment]

    # Confidence — tf_router gives a stable [0,1] readout; clamp.
    confidence = max(0.0, min(1.0, float(route.confidence)))

    return SignalEnvelope(
        symbol=sym,
        schema_version=SCHEMA_VERSION,
        generated_at=ts,
        active_timeframe=route.timeframe,
        active_strategy_id=build_strategy_id(sym, route.timeframe, regime_v2),
        regime=regime_v2,
        prev_regime=prev_v2,
        confidence=confidence,
        mode=mode if mode is not None else "observe",
        hedgerock_enabled=(
            hedgerock_enabled if hedgerock_enabled is not None else False
        ),
        risk_tier=risk_tier if risk_tier is not None else "observe",
        transition_lock_until_ts=lock_until,
        cooldown_until=cooldown_until,
        exit_directive=exit_decision.directive,
        lot_factor=lot_factor if lot_factor is not None else 1.0,
        grid_multiplier=(
            grid_multiplier if grid_multiplier is not None else 1.0
        ),
        max_next_lot=max_next_lot if max_next_lot is not None else 0.05,
        takeprofit_points=(
            takeprofit_points if takeprofit_points is not None else 600
        ),
        stoploss_points=(
            stoploss_points if stoploss_points is not None else 3900
        ),
        recovery_multiplier=(
            recovery_multiplier if recovery_multiplier is not None else 1.2
        ),
        max_orders_buy=(
            max_orders_buy if max_orders_buy is not None else 2
        ),
        max_orders_sell=(
            max_orders_sell if max_orders_sell is not None else 2
        ),
        news_intensity=news_intensity,
        news_direction=news_direction,
        news_event_name=news_event_name,
        liquidity_sweep_active=sweep_active,
        liquidity_sweep_direction=sweep_direction,
        liquidity_sweep_distance_pts=sweep_distance,
        liquidity_sweep_confidence=sweep_confidence,
        reason=reason if reason is not None else route.reason,
    )


# ---------------------------------------------------------------------------
# Safe-getter helpers — providers can be ``None`` or flaky; never crash.
# ---------------------------------------------------------------------------


def _safe_get_news_classification(
    provider: NewsFeaturesProvider | None, symbol: str,
) -> NewsClassification | None:
    if provider is None:
        return None
    try:
        return provider.get_news_classification(symbol)
    except NewsUnavailable:
        return None
    except Exception:  # pragma: no cover - defensive
        _LOG.exception("news provider crashed for %s; ignoring", symbol)
        return None


def _safe_get_liquidity_sweep(
    provider: LiquiditySweepProvider | None, symbol: str,
) -> tuple[bool, str | None, float | None, float | None] | None:
    if provider is None:
        return None
    try:
        return provider.get_liquidity_sweep(symbol)
    except Exception:  # pragma: no cover
        _LOG.exception("liquidity_sweep provider crashed for %s", symbol)
        return None


def _safe_get_exposure(
    provider: ExposureProvider | None, symbol: str,
) -> float:
    if provider is None:
        return 0.0
    try:
        return float(provider.get_exposure_lots(symbol))
    except Exception:
        _LOG.warning("exposure provider crashed for %s; assuming flat", symbol)
        return 0.0


def _safe_get_filter_inputs(
    provider: FilterInputsProvider | None, symbol: str,
) -> FilterInputs | None:
    if provider is None:
        return None
    try:
        return provider.get_filter_inputs(symbol)
    except Exception:  # pragma: no cover
        _LOG.exception("filter_inputs provider crashed for %s", symbol)
        return None


# ---------------------------------------------------------------------------
# FastAPI factory
# ---------------------------------------------------------------------------


def create_app(
    market_features_provider: MarketFeaturesProvider,
    prev_regime_store: PrevRegimeStore | None = None,
    *,
    news_provider: NewsFeaturesProvider | None = None,
    exposure_provider: ExposureProvider | None = None,
    liquidity_sweep_provider: LiquiditySweepProvider | None = None,
    liquidity_provider: Any | None = None,
    filter_inputs_provider: FilterInputsProvider | None = None,
    enable_debate: bool = True,
    enable_rule_engine: bool = False,
    ea_state_store: EAStateStore | None = None,
    exit_decider_chat_fn: ChatFn | None = None,
    cost_tracker: CostTracker | None = None,
    symbol_whitelist: frozenset[str] | None = None,
    market_provider_label: str = "unknown",
) -> FastAPI:
    """Build the FastAPI app the EA polls.

    The factory is dependency-injection friendly so tests can wire
    fakes for every external seam (features / news / exposure /
    liquidity / chat).
    """
    app = FastAPI(title="HedgeRock Decision Server", version=SCHEMA_VERSION)
    store = prev_regime_store or PrevRegimeStore()
    ea_store = ea_state_store or EAStateStore()
    whitelist = symbol_whitelist or frozenset({"XAUUSD"})

    @app.get("/healthz")
    def _healthz() -> JSONResponse:
        return JSONResponse(
            {
                "status": "ok",
                "schema_version": SCHEMA_VERSION,
            }
        )

    @app.get("/status")
    def _status() -> JSONResponse:
        symbols = sorted(whitelist)
        tracked = {sym: store.get(sym) for sym in symbols}
        latest_ea: dict[str, dict] = {}
        records = ea_store.snapshot_records()
        now_ts = datetime.now(timezone.utc)
        for sym, rec in records.items():
            try:
                state_dict = rec.state.to_dict()
                latest_ea[sym] = {
                    "state": state_dict,
                    "recorded_at": rec.recorded_at.isoformat(),
                    "age_seconds": max(
                        0.0,
                        (now_ts - rec.recorded_at).total_seconds(),
                    ),
                }
            except Exception:  # pragma: no cover
                continue
        return JSONResponse(
            {
                "schema_version": SCHEMA_VERSION,
                "symbols": symbols,
                "tracked_regimes": tracked,
                "news_provider_attached": news_provider is not None,
                "exposure_provider_attached": exposure_provider is not None,
                "liquidity_sweep_provider_attached":
                    liquidity_sweep_provider is not None,
                "liquidity_provider_attached":
                    liquidity_provider is not None,
                "filter_inputs_provider_attached":
                    filter_inputs_provider is not None,
                "debate_enabled": bool(enable_debate),
                "rule_engine_enabled": bool(enable_rule_engine),
                "market_provider_label": market_provider_label,
                "ea_state_store_attached": True,
                "latest_ea_states": latest_ea,
            }
        )

    @app.get("/signal")
    def _signal(
        request: Request,
        symbol: str = Query(..., min_length=1),
    ) -> JSONResponse:
        sym = symbol.upper()
        if sym not in whitelist:
            raise HTTPException(status_code=404, detail=f"unknown symbol {sym}")

        try:
            features = market_features_provider.get_features(sym)
        except FeaturesUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from None

        # Pull EA-state query params (extra params beyond ``symbol``).
        ea_kwargs: dict[str, Any] = {}
        for key in (
            "equity", "balance", "dd_pct", "free_margin", "margin_level",
            "open_lots", "open_positions", "floating_pnl", "spread_pts",
            "consec_losses", "recent_closed_pnl", "recent_sample_count",
        ):
            raw = request.query_params.get(key)
            if raw is None or raw == "":
                continue
            try:
                if key in ("open_positions", "spread_pts", "consec_losses",
                           "recent_sample_count"):
                    ea_kwargs[key] = int(raw)
                else:
                    ea_kwargs[key] = float(raw)
            except (TypeError, ValueError):
                continue
        if ea_kwargs:
            ea_state = build_ea_state(**ea_kwargs)
            if ea_state is not None:
                ea_store.set(sym, ea_state)

        prev = store.get(sym)
        news = _safe_get_news_classification(news_provider, sym)
        exposure_lots = _safe_get_exposure(exposure_provider, sym)
        sweep = _safe_get_liquidity_sweep(liquidity_sweep_provider, sym)

        env = build_envelope(
            sym, features,
            prev_regime=prev,
            now=datetime.now(timezone.utc),
            news_classification=news,
            current_exposure_lots=exposure_lots,
            liquidity_sweep=sweep,
            enable_debate=enable_debate,
            exit_decider_chat_fn=exit_decider_chat_fn,
            cost_tracker=cost_tracker,
        )

        # Persist this regime for next poll's transition computation.
        store.set(sym, features.regime)

        return JSONResponse(env.model_dump(mode="json"))

    return app


# ---------------------------------------------------------------------------
# Tier-1 unseal — read-only live-parameter snapshot consumed by the
# self-evolution sidecar (replay_validator + candidate_generator).
# ---------------------------------------------------------------------------


_CURRENT_LIVE_PARAMETERS: Mapping[str, float] = MappingProxyType(
    {
        "confidence_threshold_observe": 0.55,
        "confidence_threshold_aggressive": 0.80,
        "confidence_threshold_range_2": 0.65,
        "halt_expiry_observe_hours": 4.0,
    }
)


LIVE_PARAMETER_KEYS: frozenset[str] = frozenset(
    _CURRENT_LIVE_PARAMETERS.keys()
)


def get_live_parameters() -> dict[str, float]:
    """Return a fresh dict copy of the live parameter snapshot.

    The returned object is safe to mutate locally; mutation does not
    propagate back to the module-level constant.
    """
    return dict(_CURRENT_LIVE_PARAMETERS)
