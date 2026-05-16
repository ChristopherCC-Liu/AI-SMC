"""FusionController —— Tier 0~5 全链路主编排器。

调用顺序（单次 /signal poll 内）：

    1. PerceptionLayer.snapshot()    → PerceptionSnapshot
    2. DecisionLayer.decide()        → FusedDirection
    3. ValidationLayer.validate()    → ValidationVerdict
    4. ExecutionLayer.derive_params()→ DynamicParams + regime + conf
    5. build_envelope()              → SignalEnvelope
    6. ValidationLayer.record_evidence() → FusionTrace 落盘
    7. EvolutionSidecar.maybe_run_daily_cycle()  (非阻塞)

任一层失败时永远返回一个安全的 observe envelope；error 仅写日志。

设计原则（与 docs/fusion-architecture.md 一致）：
- 不抛异常 —— 上层 HTTP server 不应因为 fusion bug 而 500；
- 不持久化敏感数据 —— 仅证据链 jsonl（路径来自 FusionConfig）；
- 完全幂等 —— 同一 features+state 给同一结果；
- 向后兼容 —— SignalEnvelope schema 零改动，EA 无感升级。
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from smc.fusion.contracts import (
    FusedDirection,
    FusionConfig,
    FusionOutcome,
    FusionTrace,
    ValidationVerdict,
)
from smc.fusion.decision import DecisionLayer
from smc.fusion.evolution_sidecar import EvolutionSidecar
from smc.fusion.execution import ExecutionLayer
from smc.fusion.perception import PerceptionLayer
from smc.fusion.validation import ValidationLayer
from smc.hedgerock.decision_server import MarketFeatures, build_envelope
from smc.hedgerock.ea_state import EAStateStore
from smc.hedgerock.news_classifier import NewsClassification
from smc.hedgerock.schemas import RegimeV2, SignalEnvelope
from smc.hedgerock.transition_lock import compute_lock_until_v2

_LOG = logging.getLogger(__name__)

__all__ = ["FusionController"]


class FusionController:
    """单一入口 —— decision_server._signal 端点把整个 fusion 链委托给它。"""

    def __init__(
        self,
        *,
        perception: PerceptionLayer,
        decision: DecisionLayer,
        validation: ValidationLayer,
        execution: ExecutionLayer,
        evolution: EvolutionSidecar | None = None,
        config: FusionConfig | None = None,
    ) -> None:
        self._perception = perception
        self._decision = decision
        self._validation = validation
        self._execution = execution
        self._evolution = evolution
        self._config = config or FusionConfig()

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def on_signal_request(
        self,
        *,
        symbol: str,
        features: MarketFeatures,
        ea_state_store: EAStateStore,
        prev_envelope: SignalEnvelope | None,
        news_classification: NewsClassification | None,
        spread_pts: int | None = None,
        exposure_lots: float = 0.0,
        liquidity_sweep: Any | None = None,
        filter_result: Any | None = None,
        now: datetime | None = None,
    ) -> FusionOutcome:
        """单次 /signal poll 的完整 fusion 决策。

        永不抛异常 —— 任何子步骤异常都降级到 safe observe。
        """
        ts = now or datetime.now(timezone.utc)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        sym = symbol.upper()

        degradation: list[str] = []

        # ---- 进化 sidecar 冻结检查 ----
        evolution_frozen = self._safe_evolution_frozen()
        if evolution_frozen:
            degradation.append("evolution_circuit_breaker_frozen")

        # ---- 后台调度（非阻塞）----
        if self._evolution is not None:
            try:
                self._evolution.maybe_run_daily_cycle(now=ts)
            except Exception:  # pragma: no cover
                _LOG.exception("evolution.maybe_run_daily_cycle crashed")

        # ---- 1) Perception ----
        try:
            snapshot = self._perception.snapshot(
                symbol=sym,
                features=features,
                news_classification=news_classification,
                spread_pts=spread_pts,
                now=ts,
            )
        except Exception:
            _LOG.exception("PerceptionLayer.snapshot crashed for %s", sym)
            return self._safe_outcome(
                sym=sym,
                features=features,
                ts=ts,
                reason="perception_layer_crash",
                prev_envelope=prev_envelope,
                news_classification=news_classification,
                exposure_lots=exposure_lots,
                liquidity_sweep=liquidity_sweep,
                filter_result=filter_result,
                degradation_notes=("perception_crash", *degradation),
            )

        # ---- 2) Decision (fusion scorer) ----
        try:
            fused = self._decision.decide(snapshot=snapshot, now=ts)
        except Exception:
            _LOG.exception("DecisionLayer.decide crashed for %s", sym)
            fused = FusedDirection(
                direction="neutral",
                confidence=0.0,
                fused_score=0.0,
                component_agreement=0.0,
                reasoning="decision_layer_crash",
            )
            degradation.append("decision_layer_crash")

        # ---- 3) Validation ----
        try:
            verdict = self._validation.validate(
                snapshot=snapshot, fused=fused, now=ts,
            )
        except Exception:
            _LOG.exception("ValidationLayer.validate crashed for %s", sym)
            verdict = ValidationVerdict(
                allowed=False,
                reason="validation_layer_crash",
                full_lockdown=False,
            )
            degradation.append("validation_layer_crash")

        # 进化层冻结时强制 not allowed
        if evolution_frozen and verdict.allowed:
            verdict = ValidationVerdict(
                allowed=False,
                reason="evolution_circuit_breaker_frozen",
                gate_status=dict(verdict.gate_status),
                confidence_threshold_multiplier=verdict.confidence_threshold_multiplier,
                cooldown_extension_minutes=max(15, verdict.cooldown_extension_minutes),
                full_lockdown=False,
            )

        # ---- 4) Execution ----
        try:
            params, regime_v2, regime_conf = self._execution.derive_params(
                symbol=sym,
                features=features,
                snapshot=snapshot,
                fused=fused,
                validation=verdict,
                ea_state_store=ea_state_store,
                prev_envelope=prev_envelope,
                now=ts,
            )
        except Exception:
            _LOG.exception("ExecutionLayer.derive_params crashed for %s", sym)
            return self._safe_outcome(
                sym=sym,
                features=features,
                ts=ts,
                reason="execution_layer_crash",
                prev_envelope=prev_envelope,
                news_classification=news_classification,
                exposure_lots=exposure_lots,
                liquidity_sweep=liquidity_sweep,
                filter_result=filter_result,
                degradation_notes=("execution_crash", *degradation),
            )

        # ---- 5) Build envelope ----
        rule_kwargs = params.to_kwargs()
        prev_v2: RegimeV2 | None = (
            prev_envelope.regime if prev_envelope is not None else None
        )
        v2_lock = compute_lock_until_v2(prev_v2, regime_v2, ts)
        if (
            prev_envelope is not None
            and prev_envelope.transition_lock_until_ts is not None
            and prev_envelope.transition_lock_until_ts > ts
        ):
            prev_lock = prev_envelope.transition_lock_until_ts
            if v2_lock is None or prev_lock > v2_lock:
                v2_lock = prev_lock

        try:
            envelope = build_envelope(
                sym,
                features,
                prev_regime=features.regime,
                now=ts,
                news_classification=news_classification,
                current_exposure_lots=exposure_lots,
                liquidity_sweep=liquidity_sweep,
                filter_result=filter_result,
                enable_debate=False,
                regime_override=regime_v2,
                confidence_override=regime_conf,
                prev_regime_v2=prev_v2,
                transition_lock_until_override=v2_lock,
                transition_lock_override_provided=True,
                **rule_kwargs,
            )
        except Exception:
            _LOG.exception("build_envelope crashed for %s", sym)
            return self._safe_outcome(
                sym=sym,
                features=features,
                ts=ts,
                reason="build_envelope_crash",
                prev_envelope=prev_envelope,
                news_classification=news_classification,
                exposure_lots=exposure_lots,
                liquidity_sweep=liquidity_sweep,
                filter_result=filter_result,
                degradation_notes=("envelope_crash", *degradation),
            )

        # ---- 6) Evidence chain (audit) ----
        trace = FusionTrace(
            ts=ts,
            symbol=sym,
            perception=snapshot,
            fused_direction=fused,
            validation=verdict,
            rule_kwargs=rule_kwargs,
            final_envelope_summary=_summarize_envelope(envelope),
            degradation_notes=tuple(degradation + list(snapshot.degradation_notes)),
        )
        if self._config.enable_evidence_chain:
            try:
                self._validation.record_evidence(
                    trace_payload=_serialize_trace(trace),
                    now=ts,
                )
            except Exception:  # pragma: no cover
                _LOG.exception("record_evidence crashed for %s", sym)

        return FusionOutcome(
            envelope=envelope,
            rule_kwargs=rule_kwargs,
            regime_override=regime_v2,
            confidence_override=regime_conf,
            post_v2_regime=regime_v2,
            transition_lock_override=v2_lock,
            transition_lock_override_provided=True,
            trace=trace,
        )

    # ------------------------------------------------------------------
    # 安全降级路径
    # ------------------------------------------------------------------

    def _safe_outcome(
        self,
        *,
        sym: str,
        features: MarketFeatures,
        ts: datetime,
        reason: str,
        prev_envelope: SignalEnvelope | None,
        news_classification: NewsClassification | None,
        exposure_lots: float,
        liquidity_sweep: Any | None,
        filter_result: Any | None,
        degradation_notes: tuple[str, ...],
    ) -> FusionOutcome:
        """当任一阶段异常时返回一个保守的 observe envelope。

        永不抛异常 —— 此函数本身也要稳定。
        """
        cooldown_until = ts + timedelta(minutes=5)
        safe_kwargs: dict[str, Any] = dict(
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
            reason=f"[fusion-safe] {reason}",
        )
        try:
            envelope = build_envelope(
                sym,
                features,
                prev_regime=features.regime,
                now=ts,
                news_classification=news_classification,
                current_exposure_lots=exposure_lots,
                liquidity_sweep=liquidity_sweep,
                filter_result=filter_result,
                enable_debate=False,
                **safe_kwargs,
            )
        except Exception:
            _LOG.exception("safe build_envelope also crashed; emitting minimal")
            # 最终兜底：直接构造一个 SignalEnvelope（绕过 build_envelope）
            envelope = SignalEnvelope(
                symbol=sym,
                generated_at=ts,
                active_timeframe="M15",
                active_strategy_id=f"{sym.lower()}_m15_unknown",
                regime="unknown",
                prev_regime=None,
                confidence=0.0,
                mode="observe",
                hedgerock_enabled=False,
                risk_tier="observe",
                cooldown_until=cooldown_until,
                exit_directive="none",
                lot_factor=0.0,
                grid_multiplier=1.0,
                max_next_lot=0.05,
                takeprofit_points=600,
                stoploss_points=3900,
                recovery_multiplier=1.2,
                max_orders_buy=0,
                max_orders_sell=0,
                news_intensity="none",
                reason=f"[fusion-minimal] {reason}",
            )
        from smc.fusion.contracts import PerceptionSnapshot
        stub_snapshot = PerceptionSnapshot(
            symbol=sym,
            generated_at=ts,
            regime_v2=envelope.regime,
            regime_confidence=envelope.confidence,
            regime_reason="safe_fallback",
            degradation_notes=degradation_notes,
        )
        stub_validation = ValidationVerdict(
            allowed=False,
            reason=reason,
            full_lockdown=False,
        )
        trace = FusionTrace(
            ts=ts,
            symbol=sym,
            perception=stub_snapshot,
            fused_direction=None,
            validation=stub_validation,
            rule_kwargs=safe_kwargs,
            final_envelope_summary=_summarize_envelope(envelope),
            degradation_notes=degradation_notes,
        )
        return FusionOutcome(
            envelope=envelope,
            rule_kwargs=safe_kwargs,
            regime_override=envelope.regime,
            confidence_override=envelope.confidence,
            post_v2_regime=envelope.regime,
            transition_lock_override=None,
            transition_lock_override_provided=False,
            trace=trace,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _safe_evolution_frozen(self) -> bool:
        if self._evolution is None:
            return False
        try:
            return self._evolution.is_frozen()
        except Exception:
            _LOG.exception("evolution.is_frozen crashed")
            return False


# ---------------------------------------------------------------------------
# Trace serialization helpers
# ---------------------------------------------------------------------------


def _summarize_envelope(env: SignalEnvelope) -> dict[str, Any]:
    return {
        "regime": env.regime,
        "mode": env.mode,
        "risk_tier": env.risk_tier,
        "lot_factor": env.lot_factor,
        "max_next_lot": env.max_next_lot,
        "tp_points": env.takeprofit_points,
        "sl_points": env.stoploss_points,
        "confidence": env.confidence,
        "reason": env.reason,
    }


def _serialize_trace(trace: FusionTrace) -> dict[str, Any]:
    snap = trace.perception
    fused = trace.fused_direction
    return {
        "ts": trace.ts.isoformat(),
        "symbol": trace.symbol,
        "perception": {
            "regime_v2": snap.regime_v2,
            "regime_confidence": snap.regime_confidence,
            "regime_reason": snap.regime_reason,
            "smc_score": snap.smc_score,
            "smc_evidence": list(snap.smc_evidence),
            "anomaly_level": snap.anomaly_level,
            "anomaly_triggers": list(snap.anomaly_triggers),
            "news_intensity": snap.news_intensity,
            "news_direction": snap.news_direction,
            "news_event_name": snap.news_event_name,
            "degradation_notes": list(snap.degradation_notes),
        },
        "fused_direction": (
            None if fused is None else {
                "direction": fused.direction,
                "confidence": fused.confidence,
                "fused_score": fused.fused_score,
                "component_agreement": fused.component_agreement,
                "components": dict(fused.components),
                "sources_used": list(fused.sources_used),
                "reasoning": fused.reasoning,
            }
        ),
        "validation": {
            "allowed": trace.validation.allowed,
            "reason": trace.validation.reason,
            "gate_status": dict(trace.validation.gate_status),
            "confidence_threshold_multiplier":
                trace.validation.confidence_threshold_multiplier,
            "cooldown_extension_minutes":
                trace.validation.cooldown_extension_minutes,
            "full_lockdown": trace.validation.full_lockdown,
        },
        "rule_kwargs": _stringify_kwargs(dict(trace.rule_kwargs)),
        "envelope": trace.final_envelope_summary,
        "degradation_notes": list(trace.degradation_notes),
    }


def _stringify_kwargs(d: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, datetime):
            out[k] = v.isoformat()
        else:
            out[k] = v
    return out
