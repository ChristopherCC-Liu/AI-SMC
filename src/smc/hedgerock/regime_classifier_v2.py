"""v2.0.0 regime classifier — produces ``RegimeV2`` + confidence.

Phase B step 3. The legacy ``smc.ai.regime_classifier`` outputs the
5-value ``MarketRegimeAI`` enum (TREND_UP / TREND_DOWN / CONSOLIDATION
/ TRANSITION / ATH_BREAKOUT) and is gated behind the AI debate budget.

This module is the v2 layer: it consumes the same OHLCV frames the
``ForexDataLakeMarketFeaturesProvider`` reads, plus optional news /
liquidity hints, and emits the v2 7-value enum
(range / trend_up / trend_down / news / breakout / crisis / unknown)
together with a ``confidence`` float and a human-readable ``reason``.

Design rules:
    1. Deterministic, no LLM. Pure rules over volatility / trend /
       swing counts. Phase C may layer the LLM on top via a separate
       ``hybrid_regime_classifier`` if budget allows.
    2. Each output rule contributes a confidence ∈ [0, 1] and a reason
       string. Multiple rules vote; the *highest-priority* matching
       rule wins.
    3. ``crisis`` and ``news`` regimes are NEVER inferred from price
       alone — they require an explicit external signal (news event /
       extreme volatility marker). Without those signals we fall
       through to the price-based set (range / trend / breakout /
       unknown).
    4. ``unknown`` is the fallback when no rule has confidence ≥ the
       floor — the EA's safe-mode posture should treat unknown as
       observe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from smc.hedgerock.schemas import REGIMES_V2, RegimeV2

__all__ = [
    "RegimeAssessmentV2",
    "classify_regime_v2",
    "DEFAULT_CONFIDENCE_FLOOR",
]


DEFAULT_CONFIDENCE_FLOOR: float = 0.55
"""Below this confidence the classifier returns 'unknown' regardless
of which rule matched. Phase C rule_engine should treat unknown as
``mode=observe`` (no new entries)."""

# Rule priority — higher numbers fire first when conditions overlap.
# crisis > news > breakout > trending > range > unknown.
_PRIORITY: dict[RegimeV2, int] = {
    "crisis":     6,
    "news":       5,
    "breakout":   4,
    "trend_up":   3,
    "trend_down": 3,
    "range":      2,
    "unknown":    1,
}


@dataclass(frozen=True)
class RegimeAssessmentV2:
    """Output of :func:`classify_regime_v2`.

    Always frozen; callers mutate by ``model_copy`` style replacement
    (or ``dataclasses.replace``).
    """

    regime: RegimeV2
    confidence: float
    reason: str
    # All sub-rule votes for diagnostics/logging.
    rule_votes: tuple[tuple[RegimeV2, float, str], ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Rule helpers
# ---------------------------------------------------------------------------


def _rule_crisis(
    *,
    volatility_rank: float,
    news_intensity: Literal["none", "low", "medium", "high"] | None,
    spread_pts: int | None,
) -> tuple[float, str] | None:
    """Crisis = extreme volatility + news high (or grossly elevated spread).

    Returns (confidence, reason) when fires, else None.
    """
    if volatility_rank >= 0.95 and (news_intensity == "high"):
        return (0.92, f"vol_rank={volatility_rank:.2f} ≥ 0.95 + news=high")
    if spread_pts is not None and spread_pts >= 200:
        return (0.85, f"spread_pts={spread_pts} ≥ 200 (crisis-tier wide spread)")
    return None


def _rule_news(
    *,
    news_intensity: Literal["none", "low", "medium", "high"] | None,
) -> tuple[float, str] | None:
    """News window = explicit medium/high news event in flight."""
    if news_intensity == "high":
        return (0.88, "news=high")
    if news_intensity == "medium":
        return (0.70, "news=medium")
    return None


def _rule_breakout(
    *,
    volatility_rank: float,
    h4_trend_bars: int,
    hh_count: int,
    ll_count: int,
) -> tuple[float, str] | None:
    """Breakout = top-decile volatility + clean directional swing dominance.

    Distinct from trend_up / trend_down: breakout fires on the *initial*
    impulse (vol just spiked) before the trend has matured.
    """
    if volatility_rank >= 0.85 and h4_trend_bars >= 4 and abs(hh_count - ll_count) >= 4:
        side = "up" if hh_count > ll_count else "down"
        return (0.78,
                f"breakout {side}: vol_rank={volatility_rank:.2f}, "
                f"trend_bars={h4_trend_bars}, hh={hh_count} ll={ll_count}")
    return None


def _rule_trend(
    *,
    volatility_rank: float,
    h4_trend_bars: int,
    hh_count: int,
    ll_count: int,
) -> tuple[RegimeV2, float, str] | None:
    """Trending — directional swing dominance at moderate-or-higher vol."""
    if volatility_rank < 0.40:
        return None
    if h4_trend_bars < 3:
        return None
    if hh_count > ll_count + 2:
        conf = min(0.85, 0.55 + 0.05 * (hh_count - ll_count) + 0.02 * h4_trend_bars)
        return ("trend_up", conf,
                f"trend_up: trend_bars={h4_trend_bars}, hh={hh_count} ll={ll_count}, "
                f"vol_rank={volatility_rank:.2f}")
    if ll_count > hh_count + 2:
        conf = min(0.85, 0.55 + 0.05 * (ll_count - hh_count) + 0.02 * h4_trend_bars)
        return ("trend_down", conf,
                f"trend_down: trend_bars={h4_trend_bars}, ll={ll_count} hh={hh_count}, "
                f"vol_rank={volatility_rank:.2f}")
    return None


def _rule_range(
    *,
    volatility_rank: float,
    hh_count: int,
    ll_count: int,
) -> tuple[float, str] | None:
    """Range = compressed volatility OR balanced HH/LL even at higher vol."""
    if volatility_rank < 0.30:
        return (0.80, f"range: vol_rank={volatility_rank:.2f} < 0.30 (compressed)")
    if abs(hh_count - ll_count) <= 1 and volatility_rank < 0.65:
        return (0.65,
                f"range: balanced swings (hh={hh_count} ll={ll_count}), "
                f"vol_rank={volatility_rank:.2f}")
    return None


# ---------------------------------------------------------------------------
# Public classifier
# ---------------------------------------------------------------------------


def classify_regime_v2(
    *,
    volatility_rank: float,
    h4_trend_bars: int,
    hh_count: int,
    ll_count: int,
    news_intensity: Literal["none", "low", "medium", "high"] | None = None,
    spread_pts: int | None = None,
    confidence_floor: float = DEFAULT_CONFIDENCE_FLOOR,
) -> RegimeAssessmentV2:
    """Run the rule pipeline and return the highest-priority match.

    Inputs:
        volatility_rank: 0..1 percentile rank of current vs lookback ATR.
        h4_trend_bars:   consecutive H4 bars on the same side of mid.
        hh_count / ll_count: higher-high / lower-low counts in lookback.
        news_intensity:  optional ``NewsIntensity`` value from the news
                         pipeline (``None`` if not wired).
        spread_pts:      optional current bid/ask spread in points.
        confidence_floor: rules below this confidence fall through to
                         ``unknown``.

    Returns:
        :class:`RegimeAssessmentV2`. ``rule_votes`` lists every rule
        that fired (regardless of priority) for downstream diagnostics
        and decision_log inspection.
    """
    votes: list[tuple[RegimeV2, float, str]] = []

    crisis = _rule_crisis(
        volatility_rank=volatility_rank,
        news_intensity=news_intensity,
        spread_pts=spread_pts,
    )
    if crisis is not None:
        votes.append(("crisis", crisis[0], crisis[1]))

    news = _rule_news(news_intensity=news_intensity)
    if news is not None:
        votes.append(("news", news[0], news[1]))

    breakout = _rule_breakout(
        volatility_rank=volatility_rank,
        h4_trend_bars=h4_trend_bars,
        hh_count=hh_count,
        ll_count=ll_count,
    )
    if breakout is not None:
        votes.append(("breakout", breakout[0], breakout[1]))

    trend = _rule_trend(
        volatility_rank=volatility_rank,
        h4_trend_bars=h4_trend_bars,
        hh_count=hh_count,
        ll_count=ll_count,
    )
    if trend is not None:
        votes.append(trend)

    rng = _rule_range(
        volatility_rank=volatility_rank,
        hh_count=hh_count,
        ll_count=ll_count,
    )
    if rng is not None:
        votes.append(("range", rng[0], rng[1]))

    # Pick the top-priority above-floor vote.
    eligible = [v for v in votes if v[1] >= confidence_floor]
    if not eligible:
        return RegimeAssessmentV2(
            regime="unknown",
            confidence=max((v[1] for v in votes), default=0.0),
            reason=(
                "no rule above confidence_floor "
                f"({confidence_floor:.2f}) — defaulted to unknown"
            ),
            rule_votes=tuple(votes),
        )

    eligible.sort(key=lambda v: (_PRIORITY[v[0]], v[1]), reverse=True)
    chosen = eligible[0]
    # Sanity: chosen regime must be in the v2 enum (defensive).
    assert chosen[0] in REGIMES_V2

    return RegimeAssessmentV2(
        regime=chosen[0],
        confidence=chosen[1],
        reason=chosen[2],
        rule_votes=tuple(votes),
    )
