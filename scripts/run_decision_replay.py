"""Run a decision_replay over a synthetic XAUUSD time slice.

Usage:
    python scripts/run_decision_replay.py --instrument XAUUSD \
        --start 2024-01-01 --end 2024-12-31

The default source is a deterministic synthetic stream — useful for
sanity-checking the decision pipeline before plumbing the real
ForexDataLake-backed source (Phase 4.2). With ``--enable-debate`` the
LLM micro-debate runs at every window where the hard rule yields
``"none"`` — costs are tracked via :class:`CostTracker` and reported.

The script prints an ASCII directive distribution table at the end, the
same format that ``format_directive_distribution_table`` emits for
test snapshots / journal logs.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone

from smc.ai.cost_tracker import CostTracker
from smc.ai.models import MarketRegimeAI
from smc.backtest.walk_forward import Grain
from smc.hedgerock.decision_replay import (
    DecisionReplayConfig,
    ReplayDataSource,
    ReplayObservation,
    format_directive_distribution_table,
    run_decision_replay,
)
from smc.hedgerock.decision_server import MarketFeatures


# Stable rotation of regimes so the synthetic stream exercises every
# routing path without relying on an external feature store.
_REGIME_ROTATION: tuple[MarketRegimeAI, ...] = (
    "TREND_UP",
    "CONSOLIDATION",
    "TREND_DOWN",
    "ATH_BREAKOUT",
    "TRANSITION",
)


class _SyntheticReplaySource:
    """Deterministic ``ReplayDataSource`` producing one observation per day.

    Volatility / hh / ll cycle through a small set so the timeframe
    router sees varied inputs without needing real OHLCV. The regime
    rotation injects one ATH_BREAKOUT → TREND_DOWN flip every 5 days,
    which is enough to surface ``halt_and_close_all`` for any window
    where exposure is non-trivial.
    """

    def __init__(self, *, exposure_lots: float = 0.0) -> None:
        self._exposure = exposure_lots

    def iter_observations(
        self,
        *,
        start: datetime,
        end: datetime,
        grain: Grain,
        train_grains: int,
        test_grains: int,
        step_grains: int,
    ) -> Sequence[ReplayObservation]:
        observations: list[ReplayObservation] = []
        cursor = start
        prev_regime: MarketRegimeAI | None = None
        i = 0
        while cursor < end:
            regime = _REGIME_ROTATION[i % len(_REGIME_ROTATION)]
            features = MarketFeatures(
                volatility_rank=0.3 + 0.05 * (i % 5),
                hh_count=2 + (i % 3),
                ll_count=1 + ((i + 1) % 3),
                h4_trend_bars=4,
                regime=regime,
            )
            observations.append(
                ReplayObservation(
                    ts=cursor,
                    features=features,
                    prev_regime=prev_regime,
                    news_classification=None,
                    current_exposure_lots=self._exposure,
                )
            )
            prev_regime = regime
            cursor = cursor + timedelta(days=1)
            i += 1
        return tuple(observations)


def _parse_iso_date(value: str) -> datetime:
    """Permissive ISO-8601 parser — accepts YYYY-MM-DD or full ISO timestamps."""
    if "T" in value or " " in value:
        dt = datetime.fromisoformat(value)
    else:
        dt = datetime.fromisoformat(f"{value}T00:00:00+00:00")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instrument", default="XAUUSD")
    parser.add_argument(
        "--start",
        type=_parse_iso_date,
        required=True,
        help="UTC ISO date, e.g. 2024-01-01",
    )
    parser.add_argument(
        "--end",
        type=_parse_iso_date,
        required=True,
        help="UTC ISO date (exclusive)",
    )
    parser.add_argument(
        "--exposure-lots",
        type=float,
        default=0.5,
        help="signed lot count to feed every window (default: 0.5 long)",
    )
    parser.add_argument(
        "--enable-debate",
        action="store_true",
        help="run LLM micro-debate when hard rule yields 'none' "
        "(costs $; default off)",
    )
    parser.add_argument(
        "--daily-budget-usd",
        type=float,
        default=10.0,
        help="CostTracker daily budget when --enable-debate is set",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cost_tracker: CostTracker | None = None
    if args.enable_debate:
        cost_tracker = CostTracker(
            daily_budget_usd=args.daily_budget_usd,
            burst_budget_usd=2.0,
        )

    config = DecisionReplayConfig(
        instrument=args.instrument,
        start=args.start,
        end=args.end,
        grain="day",
        train_grains=7,
        test_grains=1,
        step_grains=1,
        enable_debate=args.enable_debate,
        cost_tracker=cost_tracker,
    )
    source = _SyntheticReplaySource(exposure_lots=args.exposure_lots)
    result = run_decision_replay(config, source)
    print(format_directive_distribution_table(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
