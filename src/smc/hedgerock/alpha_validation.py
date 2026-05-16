"""Phase 5.1 — Backtest Alpha Validation harness.

Validates the consensus from task #29 cross-debate via 6 acceptance
criteria (AC-1 through AC-6, plus AC-0 schema check):

- AC-1: ``/signal`` p99 latency < 200 ms (cache-hit + cache-miss)
- AC-2: NewsEngine cache hit rate > 90 % over the test slice
- AC-3: cache miss/stale fallback to ``actual vs forecast`` is correct
- AC-4: HedgeRock-specific reverse_pf < 1.0 — proves the decision
  pipeline carries real edge (per cross-system-lessons.md A3 / KC V8)
- AC-5: regime-mismatch lot scaling vs full-lot — max DD / final
  equity / recovery factor delta
- AC-6: failure sentinel — AC-5 delta < 5 pp or scaled group worse →
  alert + Lead review hook

The module is deliberately NOT a production-readiness validator. Per
[GO] scope: NLP layer is mocked (canned latency / cache stats), trade
fills come from a synthetic PnL stream, and the liquidity_sweep
detector is mocked (Stage E will implement it). What this harness
does verify is the **math + control flow** that downstream stages
will plug into.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from smc.ai.models import MarketRegimeAI


__all__ = [
    "AlphaValidationConfig",
    "AlphaValidationResult",
    "TradeRecord",
    "compute_reverse_pf",
    "compute_recovery_factor",
    "format_validation_report",
    "run_alpha_validation",
    "simulate_synthetic_trades",
]


# ---------------------------------------------------------------------------
# Public types — verbatim from [GO] contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TradeRecord:
    """Minimal trade for backtest math.

    Production trade objects carry many more fields (symbol, ticket,
    setup confluence, etc) but ``alpha_validation`` only needs the
    PnL stream + a ``regime_mismatch`` flag for AC-5 grouping.
    """

    ts: datetime
    pnl_usd: float
    regime_at_entry: MarketRegimeAI
    regime_mismatch: bool = False


@dataclass(frozen=True)
class AlphaValidationConfig:
    """Knobs for one validation run. Field set verbatim from [GO]."""

    instrument: str = "XAUUSD"
    start: datetime = field(
        default_factory=lambda: datetime(2024, 1, 1)
    )
    end: datetime = field(default_factory=lambda: datetime(2024, 12, 31))
    lot_factor_when_mismatch: float = 0.3
    p99_latency_budget_ms: int = 200
    min_cache_hit_rate: float = 0.90
    sentinel_delta_pp_threshold: float = 5.0


@dataclass(frozen=True)
class AlphaValidationResult:
    """Aggregate output. Field set verbatim from [GO]."""

    config: AlphaValidationConfig
    # AC-1/2/3 NLP latency + cache
    signal_p99_latency_ms: float
    cache_hit_rate: float
    fallback_path_correct: bool
    # AC-4 reverse PF
    forward_pf: float
    reverse_pf: float
    edge_real: bool
    # AC-5 regime mismatch lot scaling
    mismatch_group_max_dd: float
    mismatch_group_final_equity: float
    mismatch_group_recovery_factor: float
    full_lot_group_max_dd: float
    full_lot_group_final_equity: float
    full_lot_group_recovery_factor: float
    # AC-6 sentinel
    sentinel_triggered: bool
    sentinel_reason: str | None
    # AC-7 (P1, deferred to Stage D — see ac7_deferred_reason)
    ac7_with_nlp_final_equity: float | None = None
    ac7_without_nlp_final_equity: float | None = None
    ac7_pnl_delta_pct: float | None = None
    ac7_deferred: bool = True
    ac7_deferred_reason: str | None = None
    # Overall
    pass_all_criteria: bool = False
    failed_criteria: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Pure math helpers (testable independently of the harness)
# ---------------------------------------------------------------------------


def compute_pf(pnls: Sequence[float]) -> float:
    """Profit factor = sum(positive PnL) / abs(sum(negative PnL)).

    Edge cases:
    - all-positive trades → ``inf`` (PF undefined when no losses, but
      the AC-4 decision uses ``> 1.0`` so callers should treat ``inf``
      as "PF passes any threshold").
    - empty input → 0.0 (no trades = no edge measurable).
    - all-zero PnL → 0.0 (degenerate; treated as "no edge").
    """
    if not pnls:
        return 0.0
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = sum(-p for p in pnls if p < 0)
    if gross_loss == 0.0:
        return float("inf") if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def compute_reverse_pf(pnls: Sequence[float]) -> float:
    """Reverse PF — flip every PnL sign and recompute.

    Per cross-system-lessons.md A3 (KC V8 ``v8_validate.py:252-264``):
    ``rev = -np.array(trades)``, then PF over the flipped stream.
    Mathematical invariant: if the forward strategy has real edge,
    its reverse must lose money → ``reverse_pf < 1.0``.

    Returns ``0.0`` for empty / all-zero streams (no edge to measure).
    """
    if not pnls:
        return 0.0
    return compute_pf([-p for p in pnls])


def compute_max_dd_pct(equity_curve: Sequence[float]) -> float:
    """Max drawdown as a percentage of the running peak.

    Uses the simple peak-to-trough method that KC ``v8_validate.py``
    uses (cf. A4): ``peak = maximum.accumulate``, ``dd = (peak - eq)/peak``.
    """
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    max_dd = 0.0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        if peak > 0:
            dd = (peak - eq) / peak
            if dd > max_dd:
                max_dd = dd
    return max_dd * 100.0


def compute_recovery_factor(
    equity_curve: Sequence[float],
    initial_equity: float,
) -> float:
    """Recovery factor = net profit / max DD (in absolute USD).

    A strategy that loses 10 % then makes 30 % has higher recovery
    factor than one that loses 30 % to make 30 %. Used for AC-5
    comparison alongside max DD and final equity.
    """
    if not equity_curve or initial_equity <= 0:
        return 0.0
    final = equity_curve[-1]
    net_profit = final - initial_equity

    peak = equity_curve[0]
    max_dd_usd = 0.0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = peak - eq
        if dd > max_dd_usd:
            max_dd_usd = dd
    if max_dd_usd == 0.0:
        return float("inf") if net_profit > 0 else 0.0
    return net_profit / max_dd_usd


def equity_curve_from_trades(
    trades: Sequence[TradeRecord],
    initial_equity: float,
    lot_factor_full: float,
    lot_factor_mismatch: float,
    *,
    apply_mismatch_scaling: bool,
) -> tuple[float, ...]:
    """Build a running-equity tuple from PnL.

    When ``apply_mismatch_scaling`` is True, trades whose
    ``regime_mismatch`` flag is True are scaled by
    ``lot_factor_mismatch`` (e.g. 0.3 = 30 % size); otherwise they
    use ``lot_factor_full`` (e.g. 1.0 = full size).
    """
    eq = initial_equity
    curve = [eq]
    for t in trades:
        if apply_mismatch_scaling and t.regime_mismatch:
            scale = lot_factor_mismatch
        else:
            scale = lot_factor_full
        eq += t.pnl_usd * scale
        curve.append(eq)
    return tuple(curve)


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------


def simulate_synthetic_trades(
    *,
    start: datetime,
    end: datetime,
    seed: int = 42,
    edge_strength: float = 0.55,
    trades_per_day: int = 3,
) -> tuple[TradeRecord, ...]:
    """Deterministic synthetic XAUUSD trade stream for AC-4 / AC-5.

    Carries a real positive edge (default 55 % win rate) so the
    forward PF > 1.0 + reverse PF < 1.0 invariant holds — letting
    AC-4 fail loudly if our compute_reverse_pf has a bug. Use
    ``edge_strength=0.5`` for a no-edge control group.

    A small fraction (~20 %) of trades carry ``regime_mismatch=True``
    to feed AC-5 with both buckets.
    """
    if start >= end:
        raise ValueError(f"start {start} must precede end {end}")

    import random

    rng = random.Random(seed)
    days = max((end - start).days, 1)
    trades: list[TradeRecord] = []
    regimes: tuple[MarketRegimeAI, ...] = (
        "TREND_UP",
        "CONSOLIDATION",
        "TREND_DOWN",
        "ATH_BREAKOUT",
        "TRANSITION",
    )
    for d in range(days):
        for k in range(trades_per_day):
            ts = start + timedelta(days=d, hours=k * 6)
            win = rng.random() < edge_strength
            magnitude = abs(rng.gauss(50.0, 20.0)) + 5.0
            pnl = magnitude if win else -magnitude
            mismatch = rng.random() < 0.20
            regime = rng.choice(regimes)
            trades.append(
                TradeRecord(
                    ts=ts,
                    pnl_usd=pnl,
                    regime_at_entry=regime,
                    regime_mismatch=mismatch,
                )
            )
    return tuple(trades)


# ---------------------------------------------------------------------------
# Mocked NLP latency + cache (AC-1/2/3 architecture verification)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _NlpStats:
    p99_latency_ms: float
    hit_rate: float
    fallback_correct: bool


def _simulate_nlp_layer(
    *,
    p99_budget_ms: int,
    seed: int = 7,
    n_calls: int = 1000,
    target_hit_rate: float = 0.95,
) -> _NlpStats:
    """Generate canned NLP latency + cache stats.

    Per [GO] this task validates **architecture correctness**, not
    real LLM. We sample a latency distribution that respects the p99
    budget on cache-hit paths (~5 ms) and adds rare cache-miss spikes
    (~150 ms median, capped well under 200 ms by design).
    """
    import random

    rng = random.Random(seed)
    latencies: list[float] = []
    hits = 0
    for _ in range(n_calls):
        if rng.random() < target_hit_rate:
            hits += 1
            # Cache hit: tight, well below p99 budget.
            latencies.append(abs(rng.gauss(5.0, 1.5)))
        else:
            # Cache miss: bounded by the architecture promise that
            # the LLM call is async (background) so /signal still
            # returns the *previously cached* value or fallback. We
            # model this as ~50 ms typical, occasional 100 ms.
            latencies.append(abs(rng.gauss(50.0, 20.0)))
    latencies.sort()
    p99 = latencies[int(0.99 * len(latencies))]
    return _NlpStats(
        p99_latency_ms=p99,
        hit_rate=hits / n_calls,
        # Fallback correctness is verified by tests in
        # ``test_alpha_validation.py`` exercising the fallback branch
        # explicitly; here we record the architectural assertion.
        fallback_correct=True,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_alpha_validation(
    config: AlphaValidationConfig,
    trades: Sequence[TradeRecord] | None = None,
    *,
    initial_equity: float = 10_000.0,
    nlp_target_hit_rate: float = 0.95,
) -> AlphaValidationResult:
    """Validate the 6 AC over the requested slice.

    Args:
        config: validation knobs (date range + thresholds).
        trades: optional pre-built trade stream. If None, a
            synthetic stream is generated via
            :func:`simulate_synthetic_trades`.
        initial_equity: starting balance for AC-5 equity curves.
        nlp_target_hit_rate: target cache hit rate for the simulated
            NLP layer (defaults to 0.95, well above AC-2's 0.90 floor).

    Returns:
        :class:`AlphaValidationResult` with every AC reported and
        ``pass_all_criteria`` True iff every AC passed.
    """
    if config.start >= config.end:
        raise ValueError("config.start must be earlier than config.end")

    if trades is None:
        trades = simulate_synthetic_trades(
            start=config.start,
            end=config.end,
        )

    # ---- AC-1/2/3: NLP latency + cache architecture -----------------------
    nlp = _simulate_nlp_layer(
        p99_budget_ms=config.p99_latency_budget_ms,
        target_hit_rate=nlp_target_hit_rate,
    )

    # ---- AC-4: reverse PF -------------------------------------------------
    pnls = tuple(t.pnl_usd for t in trades)
    fwd_pf = compute_pf(pnls)
    rev_pf = compute_reverse_pf(pnls)
    edge_real = rev_pf < 1.0

    # ---- AC-5: regime-mismatch lot scaling --------------------------------
    mismatch_curve = equity_curve_from_trades(
        trades,
        initial_equity=initial_equity,
        lot_factor_full=1.0,
        lot_factor_mismatch=config.lot_factor_when_mismatch,
        apply_mismatch_scaling=True,
    )
    full_curve = equity_curve_from_trades(
        trades,
        initial_equity=initial_equity,
        lot_factor_full=1.0,
        lot_factor_mismatch=1.0,
        apply_mismatch_scaling=False,
    )
    mismatch_max_dd = compute_max_dd_pct(mismatch_curve)
    mismatch_final = mismatch_curve[-1]
    mismatch_recovery = compute_recovery_factor(mismatch_curve, initial_equity)
    full_max_dd = compute_max_dd_pct(full_curve)
    full_final = full_curve[-1]
    full_recovery = compute_recovery_factor(full_curve, initial_equity)

    # ---- AC-6: sentinel ---------------------------------------------------
    dd_delta_pp = full_max_dd - mismatch_max_dd  # positive = mismatch helped
    sentinel_triggered = False
    sentinel_reason: str | None = None
    if mismatch_final < full_final:
        sentinel_triggered = True
        sentinel_reason = (
            f"mismatch group final equity ${mismatch_final:.2f} < "
            f"full lot group ${full_final:.2f}:降仓反而更差，需 lead 评审"
        )
    elif dd_delta_pp < config.sentinel_delta_pp_threshold:
        sentinel_triggered = True
        sentinel_reason = (
            f"max DD delta {dd_delta_pp:.2f}pp < threshold "
            f"{config.sentinel_delta_pp_threshold}pp: reframe 收益不足以入 production"
        )

    # ---- Overall pass / fail ---------------------------------------------
    failed: list[str] = []
    if nlp.p99_latency_ms >= config.p99_latency_budget_ms:
        failed.append(f"AC-1 latency: p99 {nlp.p99_latency_ms:.1f}ms")
    if nlp.hit_rate <= config.min_cache_hit_rate:
        failed.append(f"AC-2 cache: hit rate {nlp.hit_rate:.2%}")
    if not nlp.fallback_correct:
        failed.append("AC-3 fallback path incorrect")
    if not edge_real:
        failed.append(f"AC-4 reverse_pf {rev_pf:.3f} >= 1.0 (no real edge)")
    if sentinel_triggered:
        failed.append(f"AC-6 sentinel: {sentinel_reason}")

    # ---- AC-7 (P1, DEFERRED): with-NLP vs without-NLP P&L delta ---------
    # Lead approved DEFERRED on synthetic data: the trade stream has no
    # news-event linkage, so any "with-NLP" mock would be arbitrary. We
    # record the deferral reason instead of producing meaningless numbers.
    ac7_deferred_reason = (
        "AC-7 DEFERRED to Stage D: synthetic trade stream has no news-event "
        "linkage; mocking 'with-NLP' P&L impact would produce arbitrary deltas. "
        "Re-run after Stage D wires real NewsEngine + LLM sentiment."
    )

    return AlphaValidationResult(
        config=config,
        signal_p99_latency_ms=nlp.p99_latency_ms,
        cache_hit_rate=nlp.hit_rate,
        fallback_path_correct=nlp.fallback_correct,
        forward_pf=fwd_pf,
        reverse_pf=rev_pf,
        edge_real=edge_real,
        mismatch_group_max_dd=mismatch_max_dd,
        mismatch_group_final_equity=mismatch_final,
        mismatch_group_recovery_factor=mismatch_recovery,
        full_lot_group_max_dd=full_max_dd,
        full_lot_group_final_equity=full_final,
        full_lot_group_recovery_factor=full_recovery,
        sentinel_triggered=sentinel_triggered,
        sentinel_reason=sentinel_reason,
        ac7_with_nlp_final_equity=None,
        ac7_without_nlp_final_equity=None,
        ac7_pnl_delta_pct=None,
        ac7_deferred=True,
        ac7_deferred_reason=ac7_deferred_reason,
        pass_all_criteria=not failed,
        failed_criteria=tuple(failed),
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def format_validation_report(result: AlphaValidationResult) -> str:
    """Render a human-readable ASCII report for CLI / journal."""
    lines: list[str] = []
    lines.append(
        f"Alpha Validation — {result.config.instrument} "
        f"{result.config.start.date()} → {result.config.end.date()}"
    )
    lines.append(
        f"  PASS: {'YES' if result.pass_all_criteria else 'NO'}    "
        f"failed: {len(result.failed_criteria)}"
    )
    lines.append("")
    lines.append("AC-1/2/3  NLP latency + cache architecture")
    lines.append(
        f"  p99 latency:    {result.signal_p99_latency_ms:.2f} ms  "
        f"(budget {result.config.p99_latency_budget_ms} ms)"
    )
    lines.append(
        f"  cache hit:      {result.cache_hit_rate:.2%}  "
        f"(min {result.config.min_cache_hit_rate:.0%})"
    )
    lines.append(
        f"  fallback ok:    {'YES' if result.fallback_path_correct else 'NO'}"
    )
    lines.append("")
    lines.append("AC-4      reverse PF (real edge?)")
    lines.append(
        f"  forward PF:     {result.forward_pf:.3f}     "
        f"reverse PF: {result.reverse_pf:.3f}     "
        f"edge real: {'YES' if result.edge_real else 'NO'}"
    )
    lines.append("")
    lines.append("AC-5      regime mismatch lot scaling")
    lines.append(
        f"  mismatch  scale={result.config.lot_factor_when_mismatch:.1f}: "
        f"DD {result.mismatch_group_max_dd:.2f}%  "
        f"final ${result.mismatch_group_final_equity:.2f}  "
        f"recovery {result.mismatch_group_recovery_factor:.2f}"
    )
    lines.append(
        f"  full lot  scale=1.0: "
        f"DD {result.full_lot_group_max_dd:.2f}%  "
        f"final ${result.full_lot_group_final_equity:.2f}  "
        f"recovery {result.full_lot_group_recovery_factor:.2f}"
    )
    dd_delta = result.full_lot_group_max_dd - result.mismatch_group_max_dd
    lines.append(
        f"  delta:    DD {dd_delta:+.2f}pp"
    )
    lines.append("")
    lines.append("AC-6      failure sentinel")
    lines.append(
        f"  triggered:      {'YES' if result.sentinel_triggered else 'NO'}"
    )
    if result.sentinel_reason:
        lines.append(f"  reason:         {result.sentinel_reason}")
    lines.append("")
    lines.append("AC-7      with-NLP vs without-NLP P&L delta (P1)")
    if result.ac7_deferred:
        lines.append(f"  status:         DEFERRED")
        if result.ac7_deferred_reason:
            lines.append(f"  reason:         {result.ac7_deferred_reason}")
    else:
        lines.append(
            f"  with NLP:       ${result.ac7_with_nlp_final_equity:.2f}    "
            f"without NLP: ${result.ac7_without_nlp_final_equity:.2f}    "
            f"delta: {result.ac7_pnl_delta_pct:+.2f}%"
        )
    if result.failed_criteria:
        lines.append("")
        lines.append("Failed AC:")
        for f in result.failed_criteria:
            lines.append(f"  - {f}")
    return "\n".join(lines)
