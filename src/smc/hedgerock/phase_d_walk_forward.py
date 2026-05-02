"""Phase D walk-forward harness — reconstructed from test contracts.

The original module implemented a full walk-forward simulator
(static vs dynamic legs, ATR-based sizing, dynamic strategy
overrides via :class:`ExperimentConfig`). It was lost when the
P0/P1 merge overwrote the untracked file. This reconstruction
covers:

  * The public API surface every consumer imports
    (:class:`WalkForwardConfig`, :class:`WalkForwardResult`,
    :class:`TradeMetrics`, :class:`ExperimentConfig`,
    :class:`ExperimentResult`, :func:`run_walk_forward`,
    :func:`apply_experiment_overrides`, ``DEFAULT_INIT_EQUITY``,
    ``DEFAULT_SPREAD_PTS``, ``_HALT_AUTO_EXPIRY_HOURS_OBSERVE``,
    ``_load_data``, ``_prepare_atr_d1``).
  * The Tier-1 read-only backtest interface used by the evolution
    sidecar (:func:`run_walk_forward_backtest`,
    :class:`BacktestResult`, :class:`BacktestWindowResult`,
    ``PUBLIC_BACKTEST_PARAMETERS``).

The simulator's outer shape is faithful — it walks the H1 timeline
window-by-window after a lookback warm-up, emits an envelope per
bar with the keys consumers expect, and produces a
:class:`TradeMetrics` snapshot. The detailed strategy semantics
(static-vs-dynamic differential, cold_start_grace promotion,
halt_auto_expiry streak) are NOT reconstructed faithfully —
:func:`apply_experiment_overrides` preserves the
``baseline-is-no-op`` invariant and otherwise returns inputs
unchanged. Tests that exercise the override semantics will surface
as known regressions until the original logic is rebuilt from
spec.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, Iterable, Mapping


__all__ = [
    "BacktestResult",
    "BacktestWindowResult",
    "DEFAULT_INIT_EQUITY",
    "DEFAULT_SPREAD_PTS",
    "_COLD_START_GRACE_CONF_FLOOR",
    "ExperimentConfig",
    "ExperimentResult",
    "PUBLIC_BACKTEST_PARAMETERS",
    "TradeMetrics",
    "WalkForwardConfig",
    "WalkForwardResult",
    "_HALT_AUTO_EXPIRY_HOURS_OBSERVE",
    "_load_data",
    "_prepare_atr_d1",
    "apply_experiment_overrides",
    "run_walk_forward",
    "run_walk_forward_backtest",
]


DEFAULT_INIT_EQUITY: float = 10000.0
DEFAULT_SPREAD_PTS: int = 20
_BLOWUP_DD_PCT: float = 0.80
_HALT_AUTO_EXPIRY_HOURS_OBSERVE: float = 4.0
# Cold-start grace eligibility: minimum observe-mode confidence floor
# under which the (lost) cold_start_grace branch could promote to
# hedgerock@normal. Reconstruction note: kept as a named constant so
# regime_opportunity_atlas can import it; the actual grace logic is
# not reconstructed.
_COLD_START_GRACE_CONF_FLOOR: float = 0.45


PUBLIC_BACKTEST_PARAMETERS: frozenset[str] = frozenset(
    {
        "confidence_threshold_observe",
        "confidence_threshold_aggressive",
        "confidence_threshold_range_2",
        "halt_expiry_observe_hours",
    }
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WalkForwardConfig:
    """Run-time configuration for :func:`run_walk_forward`.

    Defaults match the canonical 2024 XAUUSD H1 run.
    """

    instrument: str = "XAUUSD"
    start: datetime = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end: datetime = datetime(2024, 12, 31, tzinfo=timezone.utc)
    h1_lookback: int = 240
    h4_lookback: int = 60
    init_equity: float = DEFAULT_INIT_EQUITY
    spread_pts: int = DEFAULT_SPREAD_PTS


@dataclass(frozen=True)
class ExperimentConfig:
    """Optional dynamic-strategy overrides."""

    cold_start_grace: bool = False
    halt_auto_expiry_hours: float | None = None
    confidence_threshold_observe: float | None = None
    confidence_threshold_aggressive: float | None = None

    @property
    def is_baseline(self) -> bool:
        return (
            self.cold_start_grace is False
            and self.halt_auto_expiry_hours is None
            and self.confidence_threshold_observe is None
            and self.confidence_threshold_aggressive is None
        )


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TradeMetrics:
    final_equity: float
    n_trades: int
    blowup: bool
    bars_in_hedgerock: int = 0
    bars_in_observe: int = 0
    bars_in_halt: int = 0
    bars_in_momentum: int = 0


@dataclass(frozen=True)
class ExperimentResult:
    config: ExperimentConfig
    metrics: TradeMetrics


@dataclass(frozen=True)
class WalkForwardResult:
    static_metrics: TradeMetrics
    dynamic_metrics: TradeMetrics
    envelope_log: tuple[dict[str, Any], ...]
    experiment_metrics: ExperimentResult | None = None


# ---------------------------------------------------------------------------
# Internal helpers — re-exposed so existing tests keep importing them.
# ---------------------------------------------------------------------------


def _prepare_atr_d1(d1_df, period: int = 14) -> dict[datetime, float]:
    """Build a {bar-close-ts → ATR} map from a D1 OHLC DataFrame.

    Reconstruction note: returns a rolling mean of ``high - low``
    (a stable proxy for true range that's adequate for the fake-lake
    smoke tests). Returns an empty dict for missing / empty input.
    """
    try:
        if d1_df is None or d1_df.is_empty():
            return {}
    except AttributeError:
        return {}
    rows = d1_df.to_dicts()
    out: dict[datetime, float] = {}
    window: list[float] = []
    for r in rows:
        try:
            tr = float(r["high"]) - float(r["low"])
        except (KeyError, TypeError, ValueError):
            continue
        window.append(tr)
        if len(window) > period:
            window.pop(0)
        if window:
            out[r["ts"]] = sum(window) / len(window)
    return out


def _load_data(lake, config: WalkForwardConfig):
    """Pull H1/H4/D1 frames from the lake for the configured window.

    Returns ``(h1_df, h4_df, d1_df)``.
    """
    instrument = config.instrument
    h1 = lake.query(instrument, "H1", config.start, config.end)
    h4 = lake.query(instrument, "H4", config.start, config.end)
    d1 = lake.query(instrument, "D1", config.start, config.end)
    return h1, h4, d1


# ---------------------------------------------------------------------------
# apply_experiment_overrides
# ---------------------------------------------------------------------------


def apply_experiment_overrides(
    params,
    *,
    market_state,
    halt_streak_started_ts: datetime | None,
    now: datetime,
    experiment: ExperimentConfig,
) -> tuple[Any, datetime | None]:
    """Apply :class:`ExperimentConfig` overrides to a base
    :class:`DynamicParams` (from rule_engine).

    Reconstruction note: the original implemented cold_start_grace
    promotion, halt_auto_expiry streak handling, and threshold
    overrides. Those branches were lost. This reconstruction
    preserves the **baseline-is-no-op** invariant and returns the
    inputs unchanged for non-baseline configs (tests exercising the
    override semantics will fail until the original logic is
    rebuilt from spec).
    """
    if experiment.is_baseline:
        return params, halt_streak_started_ts
    return params, halt_streak_started_ts


# ---------------------------------------------------------------------------
# Walk-forward simulator (minimal, deterministic)
# ---------------------------------------------------------------------------


def _bar_iter(h1_df) -> list[dict]:
    try:
        if h1_df is None or h1_df.is_empty():
            return []
    except AttributeError:
        return []
    return h1_df.sort("ts").to_dicts()


def _envelope_entry(*, ts, equity: float, reason: str = "") -> dict[str, Any]:
    """Per-bar envelope log row. Schema matches the test contract."""
    return {
        "ts": ts,
        "equity": equity,
        "regime_v2": "unknown",
        "confidence": 0.5,
        "mode": "observe",
        "risk_tier": "observe",
        "lot_factor": 0.0,
        "max_next_lot": 0.05,
        "takeprofit_points": 600,
        "recovery_multiplier": 1.2,
        "max_orders_buy": 2,
        "max_orders_sell": 2,
        "cooldown_until": None,
        "reason": reason or "phase_d_walk_forward_stub",
    }


def run_walk_forward(
    config: WalkForwardConfig,
    lake,
    *,
    experiment: ExperimentConfig | None = None,
) -> WalkForwardResult:
    """Walk the configured H1 window, emit one envelope row per bar
    after the lookback warm-up.

    Reconstruction note: the original ran a full simulator (static
    vs dynamic legs with PnL accounting). This minimal stub keeps
    the public shape — both legs return the initial equity with no
    trades — which is enough for the smoke tests in
    ``test_phase_d_walk_forward.py``. Deeper PnL-comparison tests
    cannot pass against this stub without rebuilding the simulator.
    """
    h1, _h4, _d1 = _load_data(lake, config)
    bars = _bar_iter(h1)
    warm = max(0, int(config.h1_lookback))
    log_bars = bars[warm:] if len(bars) > warm else []

    log: list[dict[str, Any]] = []
    eq = float(config.init_equity)
    for b in log_bars:
        log.append(_envelope_entry(ts=b["ts"], equity=eq))

    n = len(log)
    static = TradeMetrics(
        final_equity=eq, n_trades=0, blowup=False, bars_in_observe=n,
    )
    dynamic = TradeMetrics(
        final_equity=eq, n_trades=0, blowup=False, bars_in_observe=n,
    )

    experiment_metrics: ExperimentResult | None = None
    if experiment is not None:
        experiment_metrics = ExperimentResult(
            config=experiment, metrics=dynamic,
        )

    return WalkForwardResult(
        static_metrics=static,
        dynamic_metrics=dynamic,
        envelope_log=tuple(log),
        experiment_metrics=experiment_metrics,
    )


# ---------------------------------------------------------------------------
# Tier-1 read-only backtest — preserved verbatim from the unseal patch.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BacktestWindowResult:
    window_id: str
    pnl_pp: float
    dd_pp: float
    n_signals: int
    regime_bucket: str


@dataclass(frozen=True)
class BacktestResult:
    parameters: Mapping[str, float]
    windows: tuple[BacktestWindowResult, ...]
    aggregate_pnl_pp: float
    aggregate_dd_pp_worst: float
    n_windows: int


def _confidence_factor(parameters: Mapping[str, float], default: float) -> float:
    obs = float(parameters.get("confidence_threshold_observe", default))
    agg = float(parameters.get("confidence_threshold_aggressive", default))
    rng2 = float(parameters.get("confidence_threshold_range_2", default))
    drift = (obs - 0.55) + (agg - 0.80) + (rng2 - 0.65)
    return 1.0 + max(-0.5, min(0.5, drift))


def _halt_factor(parameters: Mapping[str, float]) -> float:
    hours = float(
        parameters.get(
            "halt_expiry_observe_hours", _HALT_AUTO_EXPIRY_HOURS_OBSERVE,
        )
    )
    return 1.0 + (hours - _HALT_AUTO_EXPIRY_HOURS_OBSERVE) / 48.0


def run_walk_forward_backtest(
    *,
    parameters: Mapping[str, float],
    history: Iterable[Mapping[str, Any]],
) -> BacktestResult:
    """Replay ``history`` under proposed ``parameters``. Pure,
    side-effect-free. Used by the self-evolution sidecar."""
    cf = _confidence_factor(parameters, default=0.55)
    hf = _halt_factor(parameters)

    rendered: list[BacktestWindowResult] = []
    for raw in history:
        wid = str(raw.get("window_id", "")).strip()
        if not wid:
            continue
        base_pnl = float(raw.get("pnl_pp", 0.0))
        base_dd = float(raw.get("dd_pp", 0.0))
        n_sig = int(raw.get("n_signals", 0))
        bucket = str(raw.get("regime_bucket", ""))
        rendered.append(
            BacktestWindowResult(
                window_id=wid,
                pnl_pp=round(base_pnl * cf * hf, 6),
                dd_pp=round(base_dd * (2.0 - hf), 6),
                n_signals=n_sig,
                regime_bucket=bucket,
            )
        )

    agg_pnl = sum(w.pnl_pp for w in rendered)
    agg_dd_worst = max((w.dd_pp for w in rendered), default=0.0)

    return BacktestResult(
        parameters=MappingProxyType(dict(parameters)),
        windows=tuple(rendered),
        aggregate_pnl_pp=round(agg_pnl, 6),
        aggregate_dd_pp_worst=round(agg_dd_worst, 6),
        n_windows=len(rendered),
    )
