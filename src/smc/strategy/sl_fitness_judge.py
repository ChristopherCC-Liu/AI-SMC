"""SL fitness judge — rule-based veto for trade setups whose SL/TP don't fit the regime.

Round 5 A-track Task #7 (design doc: ``.scratch/round4/ai-sl-fitness-judge-design.md``).

The judge evaluates a pre-built ``EntrySignal`` against the active
``AIRegimeAssessment`` + ``RegimeContext`` + confluence score.  It runs a
deterministic 7-rule checklist and returns a frozen verdict:

    - ``accept=True``   → setup passes all rules
    - ``accept=False``  → ``rule_id`` + ``reason`` explain which rule vetoed

Design rationale (see §3 of the design doc): The 7-agent regime debate
already paid the cost of domain reasoning; the judge composes its output
with deterministic microstructure features to answer *"does this trade
plan fit what the market can actually deliver right now?"*.  No extra LLM
call, zero per-candidate latency, fully replayable.

Integration:
    - ``MultiTimeframeAggregator`` calls :func:`judge_sl_fitness` inside
      its setup loop.  In **shadow mode** (Round 5 default) the verdict
      is logged as ``sl_fitness_shadow_veto`` but does NOT block the
      setup — we measure first, enforce later.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from smc.ai.models import AIRegimeAssessment
from smc.ai.regime_classifier import RegimeContext
from smc.strategy.types import EntrySignal

__all__ = [
    "SLFitnessVerdict",
    "judge_sl_fitness",
    "RULE_IDS",
    "DEFAULT_MIN_SL_ATR_RATIO",
    "DEFAULT_MAX_SL_ATR_RATIO",
    "DEFAULT_LOW_VOL_PERCENTILE",
    "DEFAULT_TRANSITION_CONF_FLOOR",
    "DEFAULT_COUNTER_TREND_AI_CONF",
]


# ---------------------------------------------------------------------------
# Constants — defaults from design doc §5 config schema
# ---------------------------------------------------------------------------

# Rule 2: SL tighter than half the day's noise gets wicked out before thesis plays.
DEFAULT_MIN_SL_ATR_RATIO: float = 0.5

# Rule 3: SL wider than 2.5 × D1 ATR means the zone itself was mis-located.
DEFAULT_MAX_SL_ATR_RATIO: float = 2.5

# Rule 4: H4 vol rank below this percentile = low-vol regime unable to deliver large RR.
DEFAULT_LOW_VOL_PERCENTILE: float = 0.3

# Rule 7: TRANSITION regime demands higher setup quality.
DEFAULT_TRANSITION_CONF_FLOOR: float = 0.6

# Rule 1: AI confidence bar for asserting counter-trend VETO despite Tier bypass.
DEFAULT_COUNTER_TREND_AI_CONF: float = 0.6

# Rule 5: ATH-short guard — distance below which fresh ATH shorts are rejected.
_ATH_SHORT_DISTANCE_PCT: float = 1.0

# Rule 4: TP1 RR threshold above which a "big move" is being asked of a low-vol regime.
_HIGH_RR_THRESHOLD: float = 2.5

# Rule 6: floors for the two independent-axis uncertainty check.
_AI_CONF_UNCERTAIN: float = 0.5
_CONFLUENCE_UNCERTAIN: float = 0.6


# ---------------------------------------------------------------------------
# Rule identifiers (stable — used in telemetry keys, grep-friendly)
# ---------------------------------------------------------------------------

RULE_IDS: tuple[str, ...] = (
    "direction_regime_coherence",
    "sl_floor_noise_band",
    "sl_ceiling_signal_to_noise",
    "tp_reachability_low_vol",
    "ath_proximity_short_guard",
    "confluence_x_confidence_floor",
    "transition_quality_guard",
)


# ---------------------------------------------------------------------------
# Verdict type
# ---------------------------------------------------------------------------


class SLFitnessVerdict(BaseModel):
    """Frozen result of the 7-rule fitness check.

    ``accept=True`` → setup is compatible with the regime.  ``rule_id`` is
    ``"accept"`` and ``reason`` summarises the pass.

    ``accept=False`` → the named rule vetoed.  ``reason`` carries the
    concrete numeric violation so post-mortems don't require re-running
    the judge.
    """

    model_config = ConfigDict(frozen=True)

    accept: bool
    rule_id: Literal[
        "accept",
        "direction_regime_coherence",
        "sl_floor_noise_band",
        "sl_ceiling_signal_to_noise",
        "tp_reachability_low_vol",
        "ath_proximity_short_guard",
        "confluence_x_confidence_floor",
        "transition_quality_guard",
    ]
    reason: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _d1_atr_points(d1_atr_pct: float | None, current_price: float) -> float:
    """Convert D1 ATR% (e.g. 1.4 means 1.4% of price) to points for XAUUSD.

    XAUUSD point size is $0.01, so 1 point == $0.01.  A D1 ATR of 1.4%
    on a $4,000 price = $56 = 5,600 points.  Callers may pass a
    pre-computed points value via the ``d1_atr_points`` kwarg in
    :func:`judge_sl_fitness` when they already have it — this helper is
    a fallback for when only the % form is available.
    """
    if d1_atr_pct is None or d1_atr_pct <= 0.0 or current_price <= 0.0:
        return 0.0
    # 1 XAUUSD point = $0.01; price in USD → ATR in USD / 0.01 = points
    return (d1_atr_pct / 100.0) * current_price / 0.01


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def judge_sl_fitness(
    entry: EntrySignal,
    regime_assessment: AIRegimeAssessment,
    regime_ctx: RegimeContext,
    confluence_score: float,
    h1_atr_points: float,
    d1_atr_pct: float | None,
    *,
    min_sl_atr_ratio: float = DEFAULT_MIN_SL_ATR_RATIO,
    max_sl_atr_ratio: float = DEFAULT_MAX_SL_ATR_RATIO,
    low_vol_percentile: float = DEFAULT_LOW_VOL_PERCENTILE,
    transition_conf_floor: float = DEFAULT_TRANSITION_CONF_FLOOR,
    counter_trend_ai_conf: float = DEFAULT_COUNTER_TREND_AI_CONF,
) -> SLFitnessVerdict:
    """Evaluate a setup against the 7-rule fitness checklist.

    All rules are evaluated in order; the first VETO short-circuits and
    returns.  When all rules pass, an ``accept=True`` verdict is returned.

    Parameters
    ----------
    entry:
        The candidate ``EntrySignal`` (already has SL/TP/RR computed).
    regime_assessment:
        The ``AIRegimeAssessment`` returned by ``classify_regime_ai``.
    regime_ctx:
        The pre-computed ``RegimeContext`` feature snapshot.
    confluence_score:
        The setup's confluence score (0.0–1.0).
    h1_atr_points:
        H1 ATR(14) in XAUUSD points (used for risk_points / ATR ratio
        sanity if D1 ATR is missing).  Optional — passed through to
        logging hooks but not used in the default rule set.
    d1_atr_pct:
        D1 ATR(14) as % of price (e.g. 1.4 means 1.4% of price).  Used
        to compute the noise band for Rule 2 + Rule 3.

    Other keyword arguments override the module defaults (wired from
    ``SMCConfig.sl_fitness_*`` fields at aggregator init).

    Returns
    -------
    SLFitnessVerdict
        Frozen result.  ``accept=False`` triggers the veto in
        non-shadow-mode aggregator paths.
    """
    # Pre-compute D1 ATR points for SL-band rules.
    d1_atr_points = _d1_atr_points(d1_atr_pct, regime_ctx.current_price)

    # ---- Rule 1: Direction-regime coherence ---------------------------
    # ASSERT-style redundant check against ``allowed_directions``.  Catches
    # race conditions where a Tier bypass let a counter-trend trade slip
    # through the upstream filter despite a high-confidence bearish regime.
    regime = regime_assessment.regime
    ai_conf = regime_assessment.confidence
    if (
        entry.direction == "long"
        and regime == "TREND_DOWN"
        and ai_conf >= counter_trend_ai_conf
    ):
        return SLFitnessVerdict(
            accept=False,
            rule_id="direction_regime_coherence",
            reason=(
                f"long entry against TREND_DOWN with ai_conf={ai_conf:.2f}"
                f">={counter_trend_ai_conf:.2f}"
            ),
        )
    if (
        entry.direction == "short"
        and regime == "TREND_UP"
        and ai_conf >= counter_trend_ai_conf
    ):
        return SLFitnessVerdict(
            accept=False,
            rule_id="direction_regime_coherence",
            reason=(
                f"short entry against TREND_UP with ai_conf={ai_conf:.2f}"
                f">={counter_trend_ai_conf:.2f}"
            ),
        )

    # ---- Rule 2: SL floor — noise band --------------------------------
    # A stop tighter than half the day's noise will be wicked out before
    # the thesis plays.  Only runs when D1 ATR is available (skip when not).
    if d1_atr_points > 0 and entry.risk_points < min_sl_atr_ratio * d1_atr_points:
        return SLFitnessVerdict(
            accept=False,
            rule_id="sl_floor_noise_band",
            reason=(
                f"risk_points={entry.risk_points:.0f} < "
                f"{min_sl_atr_ratio:.2f} × d1_atr_points={d1_atr_points:.0f}"
            ),
        )

    # ---- Rule 3: SL ceiling — signal-to-noise -------------------------
    # Anything past 2.5 ATR means the zone itself was mis-located.
    if d1_atr_points > 0 and entry.risk_points > max_sl_atr_ratio * d1_atr_points:
        return SLFitnessVerdict(
            accept=False,
            rule_id="sl_ceiling_signal_to_noise",
            reason=(
                f"risk_points={entry.risk_points:.0f} > "
                f"{max_sl_atr_ratio:.2f} × d1_atr_points={d1_atr_points:.0f} "
                f"(zone mis-located)"
            ),
        )

    # ---- Rule 4: TP reachability in low-vol ---------------------------
    # If current H4 vol is in the bottom 30 percent of recent bars and the
    # planned TP1 RR >= 2.5, that's a "fantasy TP" — low-vol regimes do
    # not deliver 2.5R moves.  Uses the ``rr_ratio`` already computed by
    # ``check_entry`` (reward_1 / risk_points).
    h4_vol_rank = regime_ctx.h4_volatility_rank
    if h4_vol_rank < low_vol_percentile and entry.rr_ratio >= _HIGH_RR_THRESHOLD:
        return SLFitnessVerdict(
            accept=False,
            rule_id="tp_reachability_low_vol",
            reason=(
                f"h4_vol_rank={h4_vol_rank:.2f} < {low_vol_percentile:.2f} "
                f"AND rr_ratio={entry.rr_ratio:.2f} >= {_HIGH_RR_THRESHOLD:.1f}"
            ),
        )

    # ---- Rule 5: ATH-proximity short guard ----------------------------
    # Don't fade fresh ATH breakouts — supply/demand is already broken.
    ath_distance = regime_ctx.ath_distance_pct
    if (
        entry.direction == "short"
        and regime == "ATH_BREAKOUT"
        and ath_distance < _ATH_SHORT_DISTANCE_PCT
    ):
        return SLFitnessVerdict(
            accept=False,
            rule_id="ath_proximity_short_guard",
            reason=(
                f"short in ATH_BREAKOUT with ath_distance={ath_distance:.2f}% "
                f"< {_ATH_SHORT_DISTANCE_PCT:.1f}%"
            ),
        )

    # ---- Rule 6: Confluence × confidence floor ------------------------
    # Uncertainty on two independent axes (AI debate + confluence scoring)
    # means the system does not have enough signal to risk capital.
    if (
        regime_assessment.source == "ai_debate"
        and ai_conf < _AI_CONF_UNCERTAIN
        and confluence_score < _CONFLUENCE_UNCERTAIN
    ):
        return SLFitnessVerdict(
            accept=False,
            rule_id="confluence_x_confidence_floor",
            reason=(
                f"ai_debate with ai_conf={ai_conf:.2f} < {_AI_CONF_UNCERTAIN:.2f} "
                f"AND confluence={confluence_score:.2f} < {_CONFLUENCE_UNCERTAIN:.2f}"
            ),
        )

    # ---- Rule 7: TRANSITION quality guard -----------------------------
    # Transitions are ambiguous — demand extra setup quality before taking.
    if regime == "TRANSITION" and confluence_score < transition_conf_floor:
        return SLFitnessVerdict(
            accept=False,
            rule_id="transition_quality_guard",
            reason=(
                f"TRANSITION regime with confluence={confluence_score:.2f} "
                f"< {transition_conf_floor:.2f}"
            ),
        )

    # All rules passed — ACCEPT.
    return SLFitnessVerdict(
        accept=True,
        rule_id="accept",
        reason=(
            f"all 7 rules passed (regime={regime}, dir={entry.direction}, "
            f"rr={entry.rr_ratio:.2f}, conf={confluence_score:.2f})"
        ),
    )
