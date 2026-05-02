"""Phase D walk-forward — public read-only backtest interface.

This module exposes a deterministic, side-effect-free historical
backtest that the evolution layer (``replay_validator``) is allowed
to call. The contract is intentionally narrow:

  * :func:`run_walk_forward_backtest` — pure function. Given a
    parameter mapping and an iterable of historical window dicts,
    return per-window deltas vs the live baseline.
  * :data:`_HALT_AUTO_EXPIRY_HOURS_OBSERVE` — frozen constant the
    candidate generator references for one of its parameter classes.
  * :data:`PUBLIC_BACKTEST_PARAMETERS` — the set of parameter keys
    this backtest will honour. Anything outside this set is ignored.

The function does not write to disk, does not mutate inputs, and
does not import the live trade-decision path.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterable, Mapping


__all__ = [
    "BacktestResult",
    "BacktestWindowResult",
    "PUBLIC_BACKTEST_PARAMETERS",
    "_HALT_AUTO_EXPIRY_HOURS_OBSERVE",
    "run_walk_forward_backtest",
]


_HALT_AUTO_EXPIRY_HOURS_OBSERVE: float = 4.0

PUBLIC_BACKTEST_PARAMETERS: frozenset[str] = frozenset(
    {
        "confidence_threshold_observe",
        "confidence_threshold_aggressive",
        "confidence_threshold_range_2",
        "halt_expiry_observe_hours",
    }
)


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
    """Return a deterministic scaling factor in (0.5, 1.5) based on
    how far the proposed thresholds drift from the canonical mid-band.
    Pure arithmetic — no IO, no randomness."""
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
    """Replay the supplied ``history`` under the proposed
    ``parameters``, returning per-window deltas.

    ``history`` items are expected to be plain dicts with the keys
    ``window_id``, ``pnl_pp``, ``dd_pp``, ``n_signals``,
    ``regime_bucket``. Missing keys default to neutral values.
    Items lacking ``window_id`` are skipped (no exception — pure
    aggregation).

    Returns a frozen :class:`BacktestResult`. Inputs are never
    mutated.
    """
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
