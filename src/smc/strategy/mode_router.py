"""Pure-function mode router: decides trending vs ranging vs v1-passthrough.

Priority logic:
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
"""

from __future__ import annotations

from smc.instruments.types import InstrumentConfig
from smc.strategy.range_types import RangeBounds, TradingMode

__all__ = ["route_trading_mode"]


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

    Returns
    -------
    TradingMode
        Frozen model with mode, reason, and context fields.
    """
    if cfg is None:
        from smc.instruments import get_instrument_config
        cfg = get_instrument_config("XAUUSD")

    # Priority 1: AI strong directional + NOT ASIAN_CORE → trending (v1 5-gate)
    if (
        ai_direction in ("bullish", "bearish")
        and ai_confidence >= 0.5
        and (cfg.asian_core_session_name is None or session != cfg.asian_core_session_name)
    ):
        return TradingMode(
            mode="trending",
            reason=f"AI {ai_direction} (conf={ai_confidence:.2f}) — trend-following mode",
            ai_direction=ai_direction,
            ai_confidence=ai_confidence,
            regime=regime,
            range_bounds=range_bounds,
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
    )
