"""Two-layer ``ExitDirective`` decision: hard rule first, micro debate second.

Wired into the decision_server's OnTimer cycle. Lead spec for task #25:

1. **Hard rule layer** (~µs, no LLM, $0):
   - intensity=high + direction=against → ``urgent_take_profit``
   - intensity=high + direction=with + same-side exposure → ``news_trade_window``
   - extreme regime transition (distance ≥ 3) + exposure against new regime →
     ``halt_and_close_all``
   - everything else → ``"none"`` (may escalate to debate)

2. **Micro debate layer** (only when hard rule emits ``"none"`` and
   there is something arguable):
   - 2-agent: Bull defends keeping position, Bear advocates exit/halt
   - 256 max tokens per agent, fast brain (Sonnet)
   - reuses ``smc.ai.debate.pipeline._chat`` backend dispatch
   - reuses ``smc.ai.cost_tracker.CostTracker`` budget gating
   - hard 2-second timeout — exceeding it falls back to ``"none"``

The decision_server folds the result into
:class:`smc.hedgerock.schemas.SignalEnvelope.exit_directive` on every
poll; ``cost_usd`` and ``elapsed_ms`` are journaled.

The IO seam is :data:`ChatFn`. Production wires ``_chat(brain="fast")``
via :func:`_default_chat_fn`; tests inject canned responses (and
optional ``time.sleep``) to exercise timeout / cost / parse paths
without touching the real LLM.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final, Literal

from smc.ai.cost_tracker import CostTracker
from smc.ai.models import MarketRegimeAI
from smc.hedgerock.news_classifier import NewsClassification
from smc.hedgerock.schemas import EXIT_DIRECTIVES, ExitDirective


__all__ = [
    "AGENT_MAX_TOKENS",
    "DEBATE_TIMEOUT_S",
    "DECIDER_VERSION",
    "DEFAULT_PER_CALL_COST_USD",
    "EXTREME_TRANSITION_DISTANCE",
    "ChatFn",
    "ExitDecision",
    "decide_exit",
    "hard_rule_directive",
    "regime_transition_distance",
]


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------


DECIDER_VERSION: Final[str] = "v1.0.0"
"""Bump on rule / prompt changes — journals can be matched to a ruleset."""


DEBATE_TIMEOUT_S: Final[float] = 2.0
"""Hard cap on the entire micro-debate slot.

Lead spec: "延迟 < 2s 强制 (OnTimer 10s 节奏)". The decider returns
inside this budget even if the LLM is hung, falling back to ``"none"``.
"""


AGENT_MAX_TOKENS: Final[int] = 256
"""Per-agent max output tokens (lead spec)."""


DEFAULT_PER_CALL_COST_USD: Final[float] = 0.012
"""Conservative ceiling recorded in ``CostTracker`` for one full debate.

Two Sonnet calls × 256 output tokens cost roughly $0.008 in the worst
case (Sonnet output is ~$15/Mtok). $0.012 leaves headroom so we never
under-record. Hard-rule and fallback paths record $0.
"""


EXTREME_TRANSITION_DISTANCE: Final[int] = 3
"""Regime-change distance at/above which we treat the move as extreme.

Distance is computed via :func:`regime_transition_distance` over the
fixed regime ladder ``[TREND_UP, ATH_BREAKOUT, CONSOLIDATION,
TRANSITION, TREND_DOWN]``: a flip from TREND_UP to TREND_DOWN crosses
the whole ladder (distance 4); CONSOLIDATION → TREND_DOWN is distance
2 (not extreme by default).
"""


# Stable regime ordering used to compute "transition distance". The list
# is in approximate market-stance order; flipping ends counts as
# "extreme regime change".
_REGIME_LADDER: Final[tuple[MarketRegimeAI, ...]] = (
    "TREND_UP",
    "ATH_BREAKOUT",
    "CONSOLIDATION",
    "TRANSITION",
    "TREND_DOWN",
)
_REGIME_INDEX: Final[dict[MarketRegimeAI, int]] = {
    r: i for i, r in enumerate(_REGIME_LADDER)
}


# Source of the final directive (lead spec: 3 values).
_DecisionSource = Literal["hard_rule", "micro_debate", "fallback"]


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExitDecision:
    """Result of :func:`decide_exit`. Fields verbatim from lead [GO]."""

    directive: ExitDirective
    source: _DecisionSource
    rationale: str
    cost_usd: float
    elapsed_ms: int
    decider_version: str = DECIDER_VERSION


# ---------------------------------------------------------------------------
# Backend protocol — only IO seam
# ---------------------------------------------------------------------------


ChatFn = Callable[[str, str, int], tuple[str, int, float]]
"""(system, user, max_tokens) -> (content, total_tokens, cost_usd).

Mirrors ``smc.ai.debate.pipeline._chat`` (with brain="fast"). Tests
inject a fake; production lazy-loads the real ``_chat``.
"""


def _default_chat_fn(system: str, user: str, max_tokens: int) -> tuple[str, int, float]:
    """Production chat dispatcher — Claude CLI > Anthropic API > error.

    Lazy import keeps the ai/debate module out of unit-test import
    graphs that inject their own ``chat_fn``.
    """
    from smc.ai.debate.pipeline import _chat  # noqa: PLC0415 — intentional lazy

    return _chat(system, user, max_tokens, brain="fast")


# ---------------------------------------------------------------------------
# Regime transition distance
# ---------------------------------------------------------------------------


def regime_transition_distance(
    prev_regime: MarketRegimeAI | None, regime: MarketRegimeAI
) -> int:
    """Absolute distance between two regimes on :data:`_REGIME_LADDER`.

    ``prev_regime=None`` (first call after EA boot) returns 0 — there
    is no transition to measure yet, so no extreme flag fires.
    """
    if prev_regime is None or prev_regime == regime:
        return 0
    return abs(_REGIME_INDEX[regime] - _REGIME_INDEX[prev_regime])


def _exposure_sign(lots: float, *, flat_threshold: float = 0.005) -> int:
    """``+1`` long, ``-1`` short, ``0`` flat.

    The threshold filters tiny residual positions left over from
    reconciliation drift; signs flipping at ``±0.001`` lots would
    create flapping ``with``/``against`` in the news classifier path.
    """
    if abs(lots) <= flat_threshold:
        return 0
    return 1 if lots > 0 else -1


def _regime_directional_bias(regime: MarketRegimeAI) -> int:
    """Map regime to expected price direction sign for XAUUSD.

    +1 = up bias, -1 = down bias, 0 = mixed/neutral. Used to detect
    "exposure against new regime" cases where a regime flip would
    leave the EA holding a directly opposed position.
    """
    if regime == "TREND_UP":
        return 1
    if regime == "TREND_DOWN":
        return -1
    if regime == "ATH_BREAKOUT":
        return 1
    return 0  # CONSOLIDATION / TRANSITION are direction-neutral


# ---------------------------------------------------------------------------
# Hard-rule layer
# ---------------------------------------------------------------------------


def hard_rule_directive(
    *,
    regime: MarketRegimeAI,
    prev_regime: MarketRegimeAI | None,
    news_classification: NewsClassification | None,
    current_exposure_lots: float,
) -> tuple[ExitDirective, str]:
    """Return ``(directive, rationale)`` from deterministic rules.

    Rules — checked top-down, first match wins:

    1. **Extreme regime transition + opposed exposure** → ``halt_and_close_all``.
    2. **High intensity news + against** → ``urgent_take_profit``.
    3. **High intensity news + with + matching exposure** → ``news_trade_window``.
    4. Otherwise → ``"none"`` (caller may escalate to micro-debate).

    A ``rationale`` string suitable for the journal is returned
    alongside.

    ``news_classification=None`` is treated as "no news context" — only
    rule (1) can still fire.
    """
    distance = regime_transition_distance(prev_regime, regime)
    exposure_sign = _exposure_sign(current_exposure_lots)
    new_regime_bias = _regime_directional_bias(regime)

    # Rule 1: extreme regime flip with opposed exposure.
    if (
        distance >= EXTREME_TRANSITION_DISTANCE
        and exposure_sign != 0
        and new_regime_bias != 0
        and exposure_sign != new_regime_bias
    ):
        return (
            "halt_and_close_all",
            (
                f"extreme regime flip {prev_regime}→{regime} "
                f"(distance={distance}≥{EXTREME_TRANSITION_DISTANCE}); "
                f"exposure {current_exposure_lots:+.2f} lots opposes "
                f"new bias"
            ),
        )

    # Without news context, no further hard rule applies.
    if news_classification is None:
        return "none", f"no news context (regime={regime})"

    intensity = news_classification.event.intensity
    direction = news_classification.direction

    # Rule 2: high-intensity adverse news → tighten trailing close.
    if intensity == "high" and direction == "against" and exposure_sign != 0:
        return (
            "urgent_take_profit",
            (
                f"high-intensity {news_classification.event.name!r} against "
                f"{current_exposure_lots:+.2f} lots — tighten trailing close"
            ),
        )

    # Rule 3: high-intensity favourable news + matching exposure → relax
    # the entry-confirm chain so the EA can ride the move.
    if intensity == "high" and direction == "with" and exposure_sign != 0:
        return (
            "news_trade_window",
            (
                f"high-intensity {news_classification.event.name!r} with "
                f"{current_exposure_lots:+.2f} lots — open news window"
            ),
        )

    return "none", f"hard rule emits 'none' (intensity={intensity}, direction={direction})"


# ---------------------------------------------------------------------------
# Debate prompt builders
# ---------------------------------------------------------------------------


_BULL_SYSTEM: Final[str] = """\
You are the BULL agent in a HedgeRock micro-debate.
Your job: argue why the EA should KEEP its current position or EXPAND
its grid. Be terse — 3-4 sentences max, no bullet points.

Context invariants:
- The EA trades XAUUSD with grid + recovery management.
- "with"/"against"/"neutral" describes the news vs current exposure.
- Cite at least one factual element from the context (regime, news
  rationale, exposure direction, recent equity trend).

End your message with EXACTLY:
``RECOMMEND: <directive>``
where <directive> is one of: none / urgent_take_profit /
halt_and_close_all / news_trade_window. Nothing after that line.
"""


_BEAR_SYSTEM: Final[str] = """\
You are the BEAR agent in a HedgeRock micro-debate.
Your job: argue why the EA should EXIT, tighten, or halt its current
position. Be terse — 3-4 sentences max, no bullet points.

Context invariants:
- The EA trades XAUUSD with grid + recovery management.
- "with"/"against"/"neutral" describes the news vs current exposure.
- Be specific about which directive you propose and why the bull case
  is wrong.

End your message with EXACTLY:
``RECOMMEND: <directive>``
where <directive> is one of: none / urgent_take_profit /
halt_and_close_all / news_trade_window. Nothing after that line.
"""


def _format_debate_context(
    *,
    regime: MarketRegimeAI,
    prev_regime: MarketRegimeAI | None,
    news_classification: NewsClassification | None,
    current_exposure_lots: float,
    recent_equity: Sequence[float],
) -> str:
    """Compact context block fed to both Bull and Bear agents."""
    lines: list[str] = [
        f"regime: {regime}",
        f"prev_regime: {prev_regime if prev_regime is not None else '(none)'}",
        f"current_exposure_lots: {current_exposure_lots:+.2f}",
    ]
    if news_classification is not None:
        ev = news_classification.event
        lines.append(f"news_event: {ev.name}")
        lines.append(f"news_intensity: {ev.intensity}")
        lines.append(f"news_direction: {news_classification.direction}")
        if news_classification.surprise_score is not None:
            lines.append(
                f"news_surprise_score: {news_classification.surprise_score:+.3f}"
            )
        lines.append(f"news_currency: {news_classification.impact_currency}")
    else:
        lines.append("news: (no event)")
    if recent_equity:
        formatted = ", ".join(f"{v:.2f}" for v in recent_equity)
        lines.append(f"recent_equity (last {len(recent_equity)}): [{formatted}]")
    else:
        lines.append("recent_equity: (unavailable)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------


_RECOMMEND_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"RECOMMEND:\s*(?P<dir>[a-z_]+)", re.IGNORECASE
)


def _extract_recommendation(content: str) -> ExitDirective | None:
    """Pull a ``RECOMMEND: ...`` directive token from a Bull/Bear reply."""
    match = _RECOMMEND_PATTERN.search(content)
    if match is None:
        return None
    candidate = match.group("dir").lower()
    if candidate in EXIT_DIRECTIVES:
        return candidate  # type: ignore[return-value]
    return None


# ---------------------------------------------------------------------------
# Debate primitives
# ---------------------------------------------------------------------------


def _run_one_agent(
    chat_fn: ChatFn,
    system: str,
    user: str,
) -> tuple[str, float]:
    """Run a single agent, return ``(content, cost_usd)``.

    Catches every exception so the calling pipeline can fall back
    instead of propagating an LLM hiccup.
    """
    try:
        content, _tokens, cost = chat_fn(system, user, AGENT_MAX_TOKENS)
        return content, max(0.0, cost)
    except Exception:  # noqa: BLE001 — backend errors swallowed deliberately
        logger.exception("exit_decider agent call failed")
        return "", 0.0


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def decide_exit(
    *,
    regime: MarketRegimeAI,
    prev_regime: MarketRegimeAI | None,
    news_classification: NewsClassification | None,
    current_exposure_lots: float,
    recent_equity: tuple[float, ...] = (),
    enable_debate: bool = True,
    chat_fn: ChatFn | None = None,
    cost_tracker: CostTracker | None = None,
    debate_timeout_s: float = DEBATE_TIMEOUT_S,
    per_call_cost_usd: float = DEFAULT_PER_CALL_COST_USD,
) -> ExitDecision:
    """Decide an :class:`ExitDirective`. Hard rule first, debate on demand.

    Args:
        regime: Current ``MarketRegimeAI``.
        prev_regime: Previous regime (``None`` on first call after boot).
        news_classification: Output of ``classify_for_xauusd``, or
            ``None`` if there is no current news event.
        current_exposure_lots: Signed lot count: ``+`` long, ``-``
            short, ``0`` flat.
        recent_equity: Recent equity samples for debate context. May be
            empty.
        enable_debate: If ``False``, never invoke the LLM (used for
            backtest / dev mode).
        chat_fn: Inject a chat backend. Defaults to lazy-loaded
            ``_default_chat_fn``.
        cost_tracker: Optional :class:`CostTracker` for budget gating.
        debate_timeout_s: Hard wall on the debate slot. Lead default 2s.
        per_call_cost_usd: Conservative ceiling recorded against the
            cost tracker on a successful debate.

    Returns:
        :class:`ExitDecision` — always populated, never raises (caller-
        facing exceptions are caught and converted to a fallback).

    Raises:
        ValueError: ``debate_timeout_s`` non-positive.
    """
    if debate_timeout_s <= 0:
        raise ValueError(
            f"debate_timeout_s must be positive, got {debate_timeout_s}"
        )

    started = time.perf_counter()

    # ----- 1. Hard rule fast path -----------------------------------------
    rule_directive, rule_rationale = hard_rule_directive(
        regime=regime,
        prev_regime=prev_regime,
        news_classification=news_classification,
        current_exposure_lots=current_exposure_lots,
    )
    if rule_directive != "none":
        elapsed = int((time.perf_counter() - started) * 1000)
        return ExitDecision(
            directive=rule_directive,
            source="hard_rule",
            rationale=rule_rationale,
            cost_usd=0.0,
            elapsed_ms=elapsed,
        )

    # ----- 2. Decide whether to escalate to debate -------------------------
    # Skip the debate when there is nothing arguable: no news event,
    # exposure flat, or the caller explicitly disabled it (backtest mode).
    exposure_flat = _exposure_sign(current_exposure_lots) == 0
    no_news = (
        news_classification is None
        or news_classification.event.intensity == "none"
    )
    if not enable_debate or (no_news and exposure_flat):
        elapsed = int((time.perf_counter() - started) * 1000)
        return ExitDecision(
            directive="none",
            source="hard_rule",
            rationale=rule_rationale,
            cost_usd=0.0,
            elapsed_ms=elapsed,
        )

    # ----- 3. Cost tracker gate -------------------------------------------
    if cost_tracker is not None and not cost_tracker.can_classify():
        if not cost_tracker.can_burst_classify():
            elapsed = int((time.perf_counter() - started) * 1000)
            return ExitDecision(
                directive="none",
                source="fallback",
                rationale="cost tracker exhausted; debate skipped",
                cost_usd=0.0,
                elapsed_ms=elapsed,
            )
        logger.info("exit_decider using burst budget for micro-debate")

    chat = chat_fn if chat_fn is not None else _default_chat_fn
    context = _format_debate_context(
        regime=regime,
        prev_regime=prev_regime,
        news_classification=news_classification,
        current_exposure_lots=current_exposure_lots,
        recent_equity=recent_equity,
    )

    def _run_debate() -> tuple[str, str, float]:
        bull_content, bull_cost = _run_one_agent(chat, _BULL_SYSTEM, context)
        bear_content, bear_cost = _run_one_agent(chat, _BEAR_SYSTEM, context)
        return bull_content, bear_content, bull_cost + bear_cost

    # ----- 4. Run debate inside the latency-guarded slot -------------------
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(_run_debate)
            try:
                bull, bear, run_cost = future.result(timeout=debate_timeout_s)
            except concurrent.futures.TimeoutError:
                future.cancel()
                logger.warning(
                    "exit_decider debate timed out after %.2fs", debate_timeout_s
                )
                elapsed = int((time.perf_counter() - started) * 1000)
                return ExitDecision(
                    directive="none",
                    source="fallback",
                    rationale=f"debate timed out after {debate_timeout_s:.1f}s",
                    cost_usd=0.0,
                    elapsed_ms=elapsed,
                )
    except RuntimeError as exc:
        elapsed = int((time.perf_counter() - started) * 1000)
        return ExitDecision(
            directive="none",
            source="fallback",
            rationale=f"chat backend error: {exc!s:.120}",
            cost_usd=0.0,
            elapsed_ms=elapsed,
        )

    # ----- 5. Combine bull / bear via majority vote ------------------------
    bull_rec = _extract_recommendation(bull)
    bear_rec = _extract_recommendation(bear)
    if bull_rec is not None and bear_rec is not None and bull_rec == bear_rec:
        directive: ExitDirective = bull_rec
        rationale = f"debate consensus: {directive}"
    elif bear_rec is not None and bull_rec is None:
        directive = bear_rec
        rationale = f"only bear recommended: {directive}"
    elif bull_rec is not None and bear_rec is None:
        directive = bull_rec
        rationale = f"only bull recommended: {directive}"
    elif bull_rec is not None and bear_rec is not None:
        # Both spoke but disagreed — bear escalation wins for safety,
        # but never escalate to halt_and_close_all from a tie.
        if bear_rec == "halt_and_close_all":
            directive = "urgent_take_profit"
            rationale = "debate split; bear suggested halt — softening to urgent TP"
        else:
            directive = bear_rec
            rationale = f"debate split; deferring to bear recommendation: {directive}"
    else:
        # Neither parsed — take the safest no-op.
        directive = "none"
        rationale = "debate unparseable; default 'none'"

    final_cost = max(per_call_cost_usd, run_cost)
    if cost_tracker is not None and directive != "none":
        cost_tracker.record_spend(final_cost)
    elif cost_tracker is not None:
        # Even a "none" answer cost real LLM calls — record it.
        cost_tracker.record_spend(final_cost)

    elapsed = int((time.perf_counter() - started) * 1000)
    return ExitDecision(
        directive=directive,
        source="micro_debate",
        rationale=rationale,
        cost_usd=final_cost,
        elapsed_ms=elapsed,
    )
