"""Phase 4.1 — historical replay of the HedgeRock decision pipeline.

The Phase 4 problem: now that ``decision_server.build_envelope`` produces
a real ``exit_directive`` (Phase 3.4), we want to know *empirically* how
often each directive fires on historical XAUUSD data. The replay is an
envelope-flow accountant: it walks the requested time slice in walk-forward
windows, calls :func:`build_envelope` once per window, and tallies the
``directive`` distribution + LLM cost + decision latency.

This is **not** a PnL backtest. There are no fills, no positions, no
Sharpe — only directive counts. PnL backtesting (Phase 4.2) consumes
this output and pairs each directive with the trades that would follow.

Design points worth noting:

- **Walk-forward window edges via :mod:`smc.backtest.walk_forward`**:
  reuses ``_advance`` + ``_resolve_window_grains`` so the replay shares
  exactly the same window arithmetic as ``walk_forward_oos`` /
  ``run_short_backtest``. No fresh window logic.
- **Features sequence is injected**: the replay does not couple to
  ``ForexDataLake``. Production wires a lake-backed
  :class:`ReplayDataSource`, tests inject a list. This is the same
  Protocol-injection pattern that decision_server uses.
- **enable_debate=False is the default**: an "envelope-flow accountant"
  has no need to burn LLM budget — the hard rule shows directive
  trends already. Set ``enable_debate=True`` only when measuring the
  debate-path directive distribution explicitly.
- **All public types frozen**: same immutability discipline as the
  rest of hedgerock.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

from smc.ai.cost_tracker import CostTracker
from smc.ai.models import MarketRegimeAI
from smc.backtest.walk_forward import (
    Grain,
    _advance,
    _resolve_window_grains,
)
from smc.hedgerock.decision_server import (
    MarketFeatures,
    build_envelope,
)
from smc.hedgerock.exit_decider import ChatFn, ExitDecision, decide_exit
from smc.hedgerock.news_classifier import NewsClassification
from smc.hedgerock.schemas import EXIT_DIRECTIVES, ExitDirective, SignalEnvelope


__all__ = [
    "DecisionReplayConfig",
    "DecisionReplayResult",
    "ReplayDataSource",
    "ReplayObservation",
    "format_directive_distribution_table",
    "run_decision_replay",
]


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReplayObservation:
    """Inputs for one replay window.

    The replay caller builds a sequence of these (one per walk-forward
    test window). Each carries everything ``build_envelope`` needs for
    that point in time — features, prior regime, news classification
    snapshot, current exposure.

    ``ts`` is the test-window-end timestamp; the recorded
    :class:`SignalEnvelope` carries ``generated_at = ts``.
    """

    ts: datetime
    features: MarketFeatures
    prev_regime: MarketRegimeAI | None
    news_classification: NewsClassification | None = None
    current_exposure_lots: float = 0.0


class ReplayDataSource(Protocol):
    """Anything that can stream replay observations for ``[start, end)``.

    Production: queries ``ForexDataLake`` for OHLCV, derives
    :class:`MarketFeatures` per window via the existing regime
    classifier, optionally walks ``NewsEngine`` for the relevant time
    slice. Tests: a thin wrapper around an in-memory list.
    """

    def iter_observations(
        self,
        *,
        start: datetime,
        end: datetime,
        grain: Grain,
        train_grains: int,
        test_grains: int,
        step_grains: int,
    ) -> Sequence[ReplayObservation]: ...  # pragma: no cover


@dataclass(frozen=True)
class DecisionReplayConfig:
    """Knobs for one replay run. Field set verbatim from lead [GO]."""

    instrument: str
    start: datetime
    end: datetime
    grain: Grain = "day"
    train_grains: int = 7
    test_grains: int = 1
    step_grains: int = 1
    enable_debate: bool = False
    cost_tracker: CostTracker | None = None


@dataclass(frozen=True)
class DecisionReplayResult:
    """Aggregate output of one replay run.

    ``directive_distribution`` and ``directive_pct`` are aligned dicts
    keyed by every value in :data:`EXIT_DIRECTIVES`, including those
    that fired zero times — this lets downstream dashboards render a
    fixed-shape table without conditional logic.
    """

    config: DecisionReplayConfig
    total_decisions: int
    directive_distribution: dict[ExitDirective, int]
    directive_pct: dict[ExitDirective, float]
    avg_latency_ms: float
    total_cost_usd: float
    windows_processed: int
    raw_decisions: tuple[ExitDecision, ...] = field(default_factory=tuple)
    raw_envelopes: tuple[SignalEnvelope, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _empty_distribution() -> dict[ExitDirective, int]:
    """Zero-initialised counter for every legal directive.

    Pre-populating the keys keeps ``format_directive_distribution_table``
    output stable when a directive happens to fire zero times.
    """
    return {d: 0 for d in EXIT_DIRECTIVES}


def _empty_pct() -> dict[ExitDirective, float]:
    """Zero-initialised percentage table parallel to the count dict."""
    return {d: 0.0 for d in EXIT_DIRECTIVES}


def _expected_window_count(
    *,
    start: datetime,
    end: datetime,
    grain: Grain,
    train_grains: int,
    test_grains: int,
    step_grains: int,
) -> int:
    """How many test windows fit in ``[start, end)`` for this rhythm.

    Mirrors the loop in :func:`walk_forward_oos` exactly: window starts at
    ``start``, and a window is admitted iff ``test_end <= end``.
    """
    count = 0
    cursor = start
    while True:
        train_end = _advance(cursor, grain, train_grains)
        test_end = _advance(train_end, grain, test_grains)
        if test_end > end:
            break
        count += 1
        cursor = _advance(cursor, grain, step_grains)
    return count


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_decision_replay(
    config: DecisionReplayConfig,
    source: ReplayDataSource,
    *,
    chat_fn: ChatFn | None = None,
) -> DecisionReplayResult:
    """Replay the decision pipeline window-by-window over ``[start, end)``.

    Args:
        config: replay knobs (instrument, time range, walk-forward rhythm,
            debate flag, optional cost tracker).
        source: yields one :class:`ReplayObservation` per test window.
            Production wires a ``ForexDataLake``-backed source; tests
            inject an in-memory list.
        chat_fn: optional LLM dispatcher forwarded to
            ``build_envelope``. Only consulted when
            ``config.enable_debate`` is True.

    Returns:
        :class:`DecisionReplayResult` with directive counts + percentages,
        latency / cost aggregates, and the raw envelope/decision tuples
        for downstream analysis.

    Raises:
        ValueError: if ``config.start >= config.end`` or any window-grain
            kwarg is non-positive (rejected upstream by
            ``_resolve_window_grains``).
    """
    if config.start >= config.end:
        raise ValueError(
            f"start ({config.start.isoformat()}) must be earlier than "
            f"end ({config.end.isoformat()})"
        )

    # Validate window arithmetic up-front; raises ValueError on garbage.
    _resolve_window_grains(
        grain=config.grain,
        train_grains=config.train_grains,
        test_grains=config.test_grains,
        step_grains=config.step_grains,
        train_months=None,
        test_months=None,
        step_months=None,
    )

    observations = source.iter_observations(
        start=config.start,
        end=config.end,
        grain=config.grain,
        train_grains=config.train_grains,
        test_grains=config.test_grains,
        step_grains=config.step_grains,
    )

    distribution = _empty_distribution()
    decisions: list[ExitDecision] = []
    envelopes: list[SignalEnvelope] = []
    total_latency_ns = 0
    total_cost = 0.0

    for obs in observations:
        t0 = time.perf_counter_ns()
        envelope = build_envelope(
            symbol=config.instrument,
            features=obs.features,
            prev_regime=obs.prev_regime,
            now=obs.ts,
            news_classification=obs.news_classification,
            current_exposure_lots=obs.current_exposure_lots,
            exit_decider_chat_fn=chat_fn,
            cost_tracker=config.cost_tracker,
            enable_debate=config.enable_debate,
        )
        elapsed_ns = time.perf_counter_ns() - t0
        total_latency_ns += elapsed_ns

        # Re-derive the ExitDecision so we can persist source / cost / etc.
        # The envelope only carries the directive, not the full decision —
        # but that detail is what the journal-style raw_decisions tuple
        # is meant to expose. Re-deriving costs nothing in the
        # debate-disabled hot path because hard_rule_directive runs the
        # same calc twice (cheap, deterministic, no IO).
        decision = decide_exit(
            regime=obs.features.regime,
            prev_regime=obs.prev_regime,
            news_classification=obs.news_classification,
            current_exposure_lots=obs.current_exposure_lots,
            recent_equity=(),
            enable_debate=config.enable_debate,
            chat_fn=chat_fn,
            cost_tracker=config.cost_tracker,
        )

        distribution[envelope.exit_directive] += 1
        envelopes.append(envelope)
        decisions.append(decision)
        total_cost += decision.cost_usd

    total = len(envelopes)
    pct = _empty_pct()
    if total > 0:
        for directive, count in distribution.items():
            pct[directive] = 100.0 * count / total

    avg_latency_ms = (
        (total_latency_ns / total) / 1_000_000.0 if total > 0 else 0.0
    )

    return DecisionReplayResult(
        config=config,
        total_decisions=total,
        directive_distribution=distribution,
        directive_pct=pct,
        avg_latency_ms=avg_latency_ms,
        total_cost_usd=total_cost,
        windows_processed=total,
        raw_decisions=tuple(decisions),
        raw_envelopes=tuple(envelopes),
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def format_directive_distribution_table(result: DecisionReplayResult) -> str:
    """Render an ASCII table of directive counts + %.

    Stable column order (matching :data:`EXIT_DIRECTIVES`) so test
    snapshots stay byte-stable across runs and CLI output reads
    deterministically.
    """
    lines: list[str] = []
    lines.append(
        f"Decision Replay — {result.config.instrument} "
        f"{result.config.start.isoformat()} → {result.config.end.isoformat()}"
    )
    lines.append(
        f"  windows: {result.windows_processed}   "
        f"avg_latency: {result.avg_latency_ms:.2f} ms   "
        f"total_cost: ${result.total_cost_usd:.4f}"
    )
    lines.append("")
    lines.append(f"  {'directive':<22}  {'count':>6}  {'pct':>7}")
    lines.append(f"  {'-' * 22}  {'-' * 6}  {'-' * 7}")
    for directive in EXIT_DIRECTIVES:
        count = result.directive_distribution[directive]
        pct = result.directive_pct[directive]
        lines.append(f"  {directive:<22}  {count:>6}  {pct:>6.2f}%")
    return "\n".join(lines)
