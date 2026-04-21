"""Pure-function mode router: decides trending vs ranging vs v1-passthrough.

Priority logic:
  0. (Round 7 P0-1, optional) AI regime classifier (TREND_UP / TREND_DOWN /
     ATH_BREAKOUT / CONSOLIDATION / TRANSITION) — only fires when the caller
     passes ``ai_regime_assessment`` AND its confidence ≥ trust threshold.
     Otherwise falls through to the legacy 1-3 logic untouched.
  1. AI bullish/bearish + confidence >= 0.5 + session != ASIAN_CORE → trending (v1 5-gate)
  2. range_bounds exists + 5 guards pass + session in active sessions → ranging
  3. Fallback → v1_passthrough (allowed in ALL sessions including ASIAN_CORE)

Round 4.5 hotfix (用户指令 UTC 03:30):
  ASIAN_CORE 加入 _RANGING_SESSIONS (推翻 Phase 1a/1b 切分)
  用户原话: "现在就应该使用亚洲盘套利的策略"
  风险已知接受: Phase 1b backtest n=16 PF=0.69 (skeptic 历史数据)
  风控完整保留: 5 guards + CircuitBreaker + RangeQuotaTracker 不变

Round 4.6-Z: parameterized — cfg kwarg dispatches per-instrument sessions.
  _RANGING_SESSIONS removed; callers pass cfg=InstrumentConfig or default=XAUUSD.

Round 7 P0-1: AI-aware priority-0 branch added behind a caller-gated flag.
  The caller (scripts/live_demo.py::determine_action) reads
  ``SMCConfig.ai_mode_router_enabled`` and passes the aggregator's last
  ``AIRegimeAssessment`` only when the flag is ON.  When OFF (default) or
  assessment is None, the router's behaviour is byte-identical to Round 4.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from smc.instruments.types import InstrumentConfig
from smc.strategy.range_types import RangeBounds, TradingMode

if TYPE_CHECKING:
    from smc.ai.models import AIRegimeAssessment

__all__ = ["route_trading_mode"]


# Trend-class regimes that force the trending path when AI is confident.
_TREND_REGIMES: frozenset[str] = frozenset({"TREND_UP", "TREND_DOWN", "ATH_BREAKOUT"})

# TRANSITION exception: allow trending only when ATR regime agrees AND the
# AI direction engine has ≥ this much conviction.  Intentionally aligned
# with Priority 1's post-Round-5-T5 threshold (0.45) so the two trending
# paths use a single directional floor — consistency beats defensive
# layering here (Lead ACK-READY 7433c19: "0.45 is the correct final value
# — it matches current Priority 1's threshold so the two paths align").
_TRANSITION_TREND_DIR_CONF: float = 0.45


def route_trading_mode(
    ai_direction: str,
    ai_confidence: float,
    regime: str,
    session: str,
    range_bounds: RangeBounds | None,
    guards_passed: bool = False,
    current_price: float | None = None,
    *,
    cfg: InstrumentConfig | None = None,
    ai_regime_assessment: "AIRegimeAssessment | None" = None,
    ai_regime_trust_threshold: float = 0.6,
) -> TradingMode:
    """Decide between trending, ranging, and v1-passthrough trading mode.

    Parameters
    ----------
    ai_direction:
        "bullish", "bearish", or "neutral" from the AI direction engine.
    ai_confidence:
        Confidence score 0.0–1.0 from the AI direction engine.
    regime:
        Market regime from ``classify_regime()``: "trending", "transitional",
        or "ranging".
    session:
        Trading session from ``get_session_info()``.
    range_bounds:
        Detected range boundaries from ``RangeTrader.detect_range()``, or
        None if no range was detected.
    guards_passed:
        True if all 5 range guards pass (computed by caller via
        ``check_range_guards()``).
    ai_regime_assessment:
        Round 7 P0-1: optional AIRegimeAssessment from
        ``classify_regime_ai``.  When provided AND its confidence is at
        or above ``ai_regime_trust_threshold``, the AI regime drives a
        new Priority-0 branch (see module docstring).  When None or
        below threshold, the function behaves identically to Round 4 v5.
    ai_regime_trust_threshold:
        Round 7 P0-1: minimum confidence required on the assessment for
        the AI-aware branch to fire.  Typically sourced from
        ``SMCConfig.ai_regime_trust_threshold`` by the caller.

    Returns
    -------
    TradingMode
        Frozen model with mode, reason, and context fields.
    """
    if cfg is None:
        from smc.instruments import get_instrument_config
        cfg = get_instrument_config("XAUUSD")

    # ------------------------------------------------------------------
    # Priority 0 (Round 7 P0-1): AI regime classifier override
    #
    # Only fires when the caller opted in by passing a non-None
    # ``ai_regime_assessment`` AND its confidence clears the trust gate.
    # Below-threshold and None assessments fall through to Priority 1-3
    # unchanged; we still tag the decision so downstream telemetry can
    # count how often the AI path actually fired.
    # ------------------------------------------------------------------
    fell_through_tag: str | None = None
    if ai_regime_assessment is not None:
        ai_regime = ai_regime_assessment.regime
        ai_regime_conf = ai_regime_assessment.confidence

        if ai_regime_conf < ai_regime_trust_threshold:
            fell_through_tag = f"fell_through_low_conf_{ai_regime_conf:.2f}"
        elif ai_regime in _TREND_REGIMES:
            # --- Trend regimes: force trending path (v1 5-gate) ---
            # Sprint 11 bug fixed here: ATR "transitional" + AI TREND_UP/ATH
            # had been routing to ranging; now we trust the AI regime when
            # it is confident and push straight to v1 trend-following.
            return TradingMode(
                mode="trending",
                reason=(
                    f"AI regime {ai_regime} (conf={ai_regime_conf:.2f}) "
                    f"— forcing trending path"
                ),
                ai_direction=ai_direction,
                ai_confidence=ai_confidence,
                regime=regime,
                range_bounds=range_bounds,
                ai_regime_decision=(
                    f"forced_trending_by_{ai_regime}_conf_{ai_regime_conf:.2f}"
                ),
            )
        elif ai_regime == "CONSOLIDATION":
            # --- P0-1d: CONSOLIDATION defers to strong ai_direction ---
            # When the DirectionEngine shows strong conviction (≥ 0.5),
            # two AI components disagree: regime says "consolidating" but
            # direction says "clearly trending".  Trust the higher-
            # conviction signal — fall through to Priority 1-3 so Priority
            # 1 can route trending.  2023 backtest evidence: 14 bars with
            # valid range + guards routed to ranging under pure regime
            # view, but Priority-1 trending would have been more profitable
            # (Δ PF −0.23 unchanged by P0-1c fall-through because those
            # bars still passed the CONSOLIDATION→ranging preconditions).
            #
            # Threshold 0.5 is DELIBERATELY stricter than Priority 1's 0.45
            # — CONSOLIDATION override demands higher conviction.  At
            # ai_confidence < 0.5 the regime still gets the vote (ranging
            # if preconditions hold).
            if (
                ai_direction in ("bullish", "bearish")
                and ai_confidence >= 0.5
            ):
                fell_through_tag = (
                    f"consolidation_deferred_to_direction_conf_"
                    f"{ai_confidence:.2f}"
                )
                # Flow continues to Priority 1-3 below.
            else:
                # --- CONSOLIDATION: allow ranging when guards + session align ---
                price_in_range = (
                    current_price is None
                    or range_bounds is None
                    or (range_bounds.lower <= current_price <= range_bounds.upper)
                )
                if (
                    range_bounds is not None
                    and guards_passed
                    and session in cfg.ranging_sessions
                    and price_in_range
                ):
                    return TradingMode(
                        mode="ranging",
                        reason=(
                            f"AI regime CONSOLIDATION (conf={ai_regime_conf:.2f}) "
                            f"— mean-reversion mode, session={session}"
                        ),
                        ai_direction=ai_direction,
                        ai_confidence=ai_confidence,
                        regime=regime,
                        range_bounds=range_bounds,
                        ai_regime_decision=(
                            f"consolidation_to_ranging_conf_{ai_regime_conf:.2f}"
                        ),
                    )
                # P0-1c fix: CONSOLIDATION + range preconditions missing must NOT
                # force v1_passthrough — that starves Priority 1 of AI-bullish/
                # bearish trending entries.  Tag the fell-through state and let
                # Priority 1-3 decide.  The AI's CONSOLIDATION view stays
                # informational via the telemetry tag below.
                fell_through_tag = (
                    f"consolidation_fell_through_no_range_conf_{ai_regime_conf:.2f}"
                )
        elif ai_regime == "TRANSITION":
            # --- TRANSITION: default v1_passthrough, exception for ATR-trending
            #     + directional AI conviction (conservative momentum-follow).
            if (
                regime == "trending"
                and ai_direction in ("bullish", "bearish")
                and ai_confidence >= _TRANSITION_TREND_DIR_CONF
            ):
                return TradingMode(
                    mode="trending",
                    reason=(
                        f"AI regime TRANSITION (conf={ai_regime_conf:.2f}) "
                        f"+ ATR trending + AI direction {ai_direction} "
                        f"(conf={ai_confidence:.2f}) — momentum-follow"
                    ),
                    ai_direction=ai_direction,
                    ai_confidence=ai_confidence,
                    regime=regime,
                    range_bounds=range_bounds,
                    ai_regime_decision=(
                        f"transition_momentum_follow_conf_{ai_regime_conf:.2f}"
                    ),
                )
            return TradingMode(
                mode="v1_passthrough",
                reason=(
                    f"AI regime TRANSITION (conf={ai_regime_conf:.2f}) "
                    f"— waiting for clarity, v1-passthrough"
                ),
                ai_direction=ai_direction,
                ai_confidence=ai_confidence,
                regime=regime,
                range_bounds=range_bounds,
                ai_regime_decision=(
                    f"transition_to_v1_conf_{ai_regime_conf:.2f}"
                ),
            )
        else:
            # Unknown regime label (future-proofing) — tag and fall through.
            fell_through_tag = f"fell_through_unknown_regime_{ai_regime}"

    # ------------------------------------------------------------------
    # Priority 1: AI strong directional + NOT ASIAN_CORE → trending (v1 5-gate)
    # Round 5 T5 tweak: threshold 0.5 → 0.45. decision-reviewer observed
    # today's XAU ai_confidence=0.47 (bearish) was stuck below 0.5 → never
    # entered trending path. 0.45 lets edge-case convictions through; v1
    # 5-gate aggregator still filters false trends downstream.
    # ------------------------------------------------------------------
    if (
        ai_direction in ("bullish", "bearish")
        and ai_confidence >= 0.45
        and (cfg.asian_core_session_name is None or session != cfg.asian_core_session_name)
    ):
        return TradingMode(
            mode="trending",
            reason=f"AI {ai_direction} (conf={ai_confidence:.2f}) — trend-following mode",
            ai_direction=ai_direction,
            ai_confidence=ai_confidence,
            regime=regime,
            range_bounds=range_bounds,
            ai_regime_decision=fell_through_tag,
        )

    # Priority 2: range detected + 5 guards pass + active session → ranging
    # Round 4.6-R: regime=trending + breakout OUT of range → v1_passthrough (trend-following)
    # Round 4.6-T-v3 (USER "解决到开仓"): 4.6-R 过度 — only suppress ranging when
    # price has actually BROKEN OUT of range. 若 price 仍在 range 内, mean-reversion
    # 依然有效（price 往中点回归概率高于继续 trend out of range）.
    # ASIAN_CORE 保留例外 (Asian 低波反转力强, regime trending 多假信号).
    price_in_range = (
        current_price is None
        or range_bounds is None
        or (range_bounds.lower <= current_price <= range_bounds.upper)
    )
    trending_suppress = (
        regime == "trending"
        and (cfg.asian_core_session_name is None or session != cfg.asian_core_session_name)
        and not price_in_range  # only suppress when price OUT of range
    )
    if (
        range_bounds is not None
        and guards_passed
        and session in cfg.ranging_sessions
        and not trending_suppress
    ):
        return TradingMode(
            mode="ranging",
            reason=(
                f"Range detected (guards passed), session={session} — mean-reversion mode"
            ),
            ai_direction=ai_direction,
            ai_confidence=ai_confidence,
            regime=regime,
            range_bounds=range_bounds,
            ai_regime_decision=fell_through_tag,
        )

    # Priority 3: fallback — v1 pipeline runs in all sessions including ASIAN_CORE
    return TradingMode(
        mode="v1_passthrough",
        reason=(
            f"AI unsure (dir={ai_direction}, conf={ai_confidence:.2f}) "
            f"— v1 pipeline runs with HTF bias only"
        ),
        ai_direction=ai_direction,
        ai_confidence=ai_confidence,
        regime=regime,
        range_bounds=range_bounds,
        ai_regime_decision=fell_through_tag,
    )
