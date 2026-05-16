"""ExecutionLayer —— 把 fused direction + perception 喂给 rule_engine，
得到 DynamicParams，再合成 SignalEnvelope。

这一层是 fusion 跟旧管道的真正接口：rule_engine 已经成熟，我们不重
写它，只是用更智能的输入"喂"它，并把 fused.confidence / fused.direction
通过对参数做后置增益（如 lot_factor 微调）的方式把融合信号"渗"进去。
"""

from __future__ import annotations

import logging
from dataclasses import replace
from datetime import datetime, timedelta
from typing import Any, Mapping

from smc.fusion.contracts import (
    FusedDirection,
    FusionConfig,
    PerceptionSnapshot,
    ValidationVerdict,
)
from smc.hedgerock.decision_server import MarketFeatures
from smc.hedgerock.ea_state import EAState, EAStateStore
from smc.hedgerock.market_state import aggregate_market_state
from smc.hedgerock.regime_classifier_v2 import RegimeAssessmentV2
from smc.hedgerock.rule_engine import (
    DynamicParams,
    derive_envelope_params,
)
from smc.hedgerock.schemas import RegimeV2, SignalEnvelope

_LOG = logging.getLogger(__name__)

__all__ = ["ExecutionLayer"]


class ExecutionLayer:
    """Wraps rule_engine + applies fusion-aware overlays."""

    def __init__(self, *, config: FusionConfig) -> None:
        self._config = config

    # ------------------------------------------------------------------
    # 主入口 —— 在 fusion_controller 里调用
    # ------------------------------------------------------------------

    def derive_params(
        self,
        *,
        symbol: str,
        features: MarketFeatures,
        snapshot: PerceptionSnapshot,
        fused: FusedDirection,
        validation: ValidationVerdict,
        ea_state_store: EAStateStore,
        prev_envelope: SignalEnvelope | None,
        now: datetime,
    ) -> tuple[DynamicParams, RegimeV2, float]:
        """返回 (DynamicParams, regime_v2_override, confidence_override)。

        - regime_v2 来自 snapshot.regime_v2（已由 perception 计算过）
        - 当 validation 不允许时，强制走 observe（fail-safe）
        - 否则把 fused.confidence 注入到 DynamicParams.lot_factor 上的
          后置增益（保持在 schema 边界内）
        """
        # 1) Validation 拒绝 → safe observe
        if not validation.allowed:
            cooldown = self._cooldown_from_validation(validation, now)
            params = _safe_observe(
                reason=f"validation_block:{validation.reason}",
                now=now,
                cooldown_until=cooldown,
            )
            return params, snapshot.regime_v2, snapshot.regime_confidence

        # 2) 构造 RegimeAssessmentV2 (rule_engine 的入参形态)
        assessment = RegimeAssessmentV2(
            regime=snapshot.regime_v2,
            confidence=snapshot.regime_confidence,
            reason=snapshot.regime_reason,
            rule_votes=(),
        )

        # 3) 用 ea_state_store 聚合 MarketState
        record = ea_state_store.get_record(symbol.upper())
        ea_state: EAState | None = record.state if record else None
        ea_recorded_at: datetime | None = record.recorded_at if record else None

        try:
            market_state = aggregate_market_state(
                symbol=symbol,
                now=now,
                features=features,
                regime_assessment=assessment,
                ea_state=ea_state,
                ea_state_recorded_at=ea_recorded_at,
            )
        except Exception:
            _LOG.exception("aggregate_market_state crashed for %s", symbol)
            params = _safe_observe(
                reason="aggregate_market_state_failed",
                now=now,
            )
            return params, snapshot.regime_v2, snapshot.regime_confidence

        # 4) 跑 rule_engine
        try:
            params = derive_envelope_params(market_state, prev_envelope=prev_envelope)
        except Exception:
            _LOG.exception("derive_envelope_params crashed for %s", symbol)
            params = _safe_observe(
                reason="rule_engine_crash",
                now=now,
            )
            return params, snapshot.regime_v2, snapshot.regime_confidence

        # 5) Fusion overlay —— 把 fused signal 信号注入 lot_factor + reason
        params = self._apply_fusion_overlay(
            params=params,
            fused=fused,
            validation=validation,
            snapshot=snapshot,
        )

        return params, snapshot.regime_v2, snapshot.regime_confidence

    # ------------------------------------------------------------------
    # Overlays
    # ------------------------------------------------------------------

    def _apply_fusion_overlay(
        self,
        *,
        params: DynamicParams,
        fused: FusedDirection,
        validation: ValidationVerdict,
        snapshot: PerceptionSnapshot,
    ) -> DynamicParams:
        """让 fused.confidence/direction 微调 rule_engine 的输出。

        - fused.confidence < confidence_floor → demote to observe (safety);
        - fused.direction == 'neutral' → 不限方向，但 lot_factor *= 0.8;
        - fused.direction 与 snapshot.regime_v2 同向 → lot_factor 微增益;
        - fused.confidence ≥ aggressive_floor 且 mode=hedgerock → 允许 risk_tier 升 'aggressive'.

        所有修改都在 schema 边界内（lot_factor ≤ 5.0, max_next_lot ≤ 1.0）。
        """
        if params.mode != "hedgerock":
            # observe / halt / momentum 路径不做融合放大
            reason = (
                f"{params.reason} | fusion={fused.direction}@{fused.confidence:.2f}"
            )
            return replace(params, reason=_clip_reason(reason))

        # 同向 boost / 反向 stepdown
        regime_dir = _regime_to_direction(snapshot.regime_v2)
        align = (
            +1 if (regime_dir is not None and regime_dir == fused.direction) else
            (-1 if (regime_dir is not None and fused.direction != "neutral"
                    and regime_dir != fused.direction) else 0)
        )
        lot_mult = 1.0 + 0.30 * align * fused.confidence
        lot_mult = max(0.5, min(1.5, lot_mult))
        new_lot_factor = max(0.0, min(5.0, params.lot_factor * lot_mult))

        new_risk_tier = params.risk_tier
        if (
            fused.confidence >= self._config.confidence_aggressive_floor
            and align >= 0
            and params.risk_tier == "normal"
        ):
            new_risk_tier = "aggressive"

        # confidence 太低 —— 即使 rule_engine 决定 hedgerock，也保留观望
        if fused.confidence < self._config.confidence_floor:
            return _safe_observe(
                reason=(
                    f"fusion_confidence_below_floor:{fused.confidence:.2f}<"
                    f"{self._config.confidence_floor:.2f}"
                ),
                now=params.cooldown_until or datetime.now(tz=None),
            )

        new_reason = (
            f"{params.reason} | fusion={fused.direction}@{fused.confidence:.2f} "
            f"align={align:+d} lot_mult={lot_mult:.2f} "
            f"sources={','.join(fused.sources_used)}"
        )
        return replace(
            params,
            lot_factor=new_lot_factor,
            risk_tier=new_risk_tier,
            reason=_clip_reason(new_reason),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _cooldown_from_validation(
        self,
        validation: ValidationVerdict,
        now: datetime,
    ) -> datetime | None:
        if validation.cooldown_extension_minutes <= 0:
            return None
        return now + timedelta(minutes=validation.cooldown_extension_minutes)


# ---------------------------------------------------------------------------
# 模块级 helpers
# ---------------------------------------------------------------------------


def _safe_observe(
    *,
    reason: str,
    now: datetime,
    cooldown_until: datetime | None = None,
) -> DynamicParams:
    """复用 rule_engine 的 safe-observe profile，但带 fusion-prefixed reason。"""
    return DynamicParams(
        mode="observe",
        hedgerock_enabled=False,
        risk_tier="observe",
        cooldown_until=cooldown_until,
        lot_factor=0.0,
        grid_multiplier=1.0,
        max_next_lot=0.05,
        takeprofit_points=600,
        stoploss_points=3900,
        recovery_multiplier=1.2,
        max_orders_buy=2,
        max_orders_sell=2,
        reason=_clip_reason(f"[fusion] {reason}"),
    )


def _clip_reason(reason: str) -> str:
    """SignalEnvelope.reason 限 500 chars."""
    return reason[:480]


def _regime_to_direction(
    regime: RegimeV2,
) -> str | None:
    """range/news/unknown/crisis → None；trend_up/breakout → bullish；
    trend_down → bearish。"""
    if regime in ("trend_up", "breakout"):
        return "bullish"
    if regime == "trend_down":
        return "bearish"
    return None


