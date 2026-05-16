"""Fusion 层的纯数据契约 —— 所有 dataclass 都是 frozen 的。

放在独立文件，避免循环依赖：每个 Layer 只 import contracts，
不互相 import 实现细节。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Mapping

from smc.hedgerock.schemas import RegimeV2, SignalEnvelope

__all__ = [
    "FusedDirection",
    "FusionConfig",
    "FusionOutcome",
    "FusionTrace",
    "PerceptionSnapshot",
    "ValidationVerdict",
]

# ---------------------------------------------------------------------------
# 融合方向 —— DecisionLayer 输出
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FusedDirection:
    """加权融合后的方向 + 置信度。

    direction: bullish/bearish/neutral —— 用于 SMC 入场过滤
    confidence: ∈ [0,1] —— 用于 rule_engine 的 confidence 路由
    fused_score: ∈ [-1,1] —— 原始加权得分（debug/audit）
    component_agreement: ∈ [0,1] —— 多组件同向度
    components: 各子组件原始读数（NamedTuple 形式留 trace）
    """

    direction: Literal["bullish", "bearish", "neutral"]
    confidence: float
    fused_score: float
    component_agreement: float
    components: Mapping[str, float] = field(default_factory=dict)
    sources_used: tuple[str, ...] = field(default_factory=tuple)
    reasoning: str = ""


# ---------------------------------------------------------------------------
# 感知层快照 —— PerceptionLayer 输出
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PerceptionSnapshot:
    """单 tick 的感知快照。

    每个字段都是 Optional —— 任何子组件失败时设 None，下游用
    `is not None` 判断是否启用对应权重。
    """

    symbol: str
    generated_at: datetime
    # regime —— 主路径 regime_classifier_v2 结果
    regime_v2: RegimeV2
    regime_confidence: float
    regime_reason: str
    # SMC 结构性方向打分 (+ 看多 / − 看空 / 0 无)
    smc_score: float | None = None
    smc_evidence: tuple[str, ...] = field(default_factory=tuple)
    # 异常状态 —— AnomalyDetector
    anomaly_level: Literal["NORMAL", "ELEVATED", "CRITICAL", "LOCKDOWN"] | None = None
    anomaly_triggers: tuple[str, ...] = field(default_factory=tuple)
    # 新闻状态 —— NewsClassifier
    news_intensity: Literal["none", "low", "medium", "high"] | None = None
    news_direction: Literal["with", "against", "neutral"] | None = None
    news_event_name: str | None = None
    # 降级备注 —— "ai_off", "smc_empty_bars", ...
    degradation_notes: tuple[str, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# 验证层判决 —— ValidationLayer 输出
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationVerdict:
    """验证层是否允许信号通过到 rule_engine。

    allowed=False 时 ExecutionLayer 必须 force observe / halt。
    """

    allowed: bool
    reason: str
    gate_status: Mapping[str, bool] = field(default_factory=dict)
    confidence_threshold_multiplier: float = 1.0
    cooldown_extension_minutes: int = 0
    full_lockdown: bool = False
    evidence_path: str | None = None  # ShadowArtefact 文件路径


# ---------------------------------------------------------------------------
# 配置 —— 所有权重 / 阈值 / 开关集中在一个 dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FusionConfig:
    """融合层的运行时配置。

    所有可调参数集中在此；SASL AutoAdjuster 也读这里。
    Frozen + 通过 dataclasses.replace() 生成新版本。
    """

    # 融合权重 —— α/β/γ/δ/ω in docs/fusion-architecture.md §6
    weight_ai: float = 0.35
    weight_macro: float = 0.25
    weight_smc: float = 0.30
    weight_news: float = 0.10
    weight_anomaly_penalty: float = 0.50

    # 阈值
    direction_threshold: float = 0.15  # |fused_score| 小于则 neutral
    confidence_floor: float = 0.30  # 低于该值强制 observe
    confidence_aggressive_floor: float = 0.80

    # 组件开关 —— 按 layer 粒度降级
    enable_ai_direction: bool = True
    enable_macro_layer: bool = True
    enable_smc_detection: bool = True
    enable_anomaly_shield: bool = True
    enable_promotion_gates: bool = True
    enable_evidence_chain: bool = True

    # 资源限制
    ai_cache_ttl_hours: int = 4
    macro_cache_ttl_hours: int = 24
    perception_cache_ttl_seconds: int = 60

    # 证据链落盘目录（None = 不落盘，仅返回）
    evidence_dir: str | None = None


# ---------------------------------------------------------------------------
# Fusion trace —— 完整审计载荷
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FusionTrace:
    """Single signal request 的完整审计 trace。

    序列化后可以直接写到 ShadowArtefact 或 fusion_<ts>.jsonl。
    """

    ts: datetime
    symbol: str
    perception: PerceptionSnapshot
    fused_direction: FusedDirection | None
    validation: ValidationVerdict
    rule_kwargs: Mapping[str, Any]
    final_envelope_summary: Mapping[str, Any]
    degradation_notes: tuple[str, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# FusionOutcome —— FusionController.on_signal_request 的返回类型
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FusionOutcome:
    """FusionController 返回给 decision_server 的完整结果。

    envelope: 直接 publish 到 RegimeCache 的 SignalEnvelope
    rule_kwargs: build_envelope() 的扩展 kwargs（兼容性留存）
    regime_override / confidence_override / post_v2_regime:
        与 decision_server._run_rule_engine 的返回兼容
    transition_lock_override / transition_lock_override_provided:
        允许 fusion 强制重置 transition lock
    trace: 完整 audit 信息
    """

    envelope: SignalEnvelope
    rule_kwargs: Mapping[str, Any]
    regime_override: RegimeV2
    confidence_override: float
    post_v2_regime: RegimeV2
    transition_lock_override: datetime | None
    transition_lock_override_provided: bool
    trace: FusionTrace
