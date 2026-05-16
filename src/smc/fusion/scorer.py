"""FusionScorer —— 把 AI direction / Macro / SMC / News 几个异质信号
合并成单一的 ``FusedDirection``。

数学很简单（加权求和 + 阈值），但是把它放在独立模块里：
1. 单元测试方便（纯函数）；
2. 未来想换成 sklearn 训练的 stacking model，只换这一个文件；
3. SASL AutoAdjuster 只需调整 FusionConfig 的几个权重就能自适应。
"""

from __future__ import annotations

from typing import Literal

from smc.fusion.contracts import FusedDirection, FusionConfig
from smc.fusion.perception import anomaly_confidence_penalty

__all__ = ["score_fusion"]


def _direction_to_score(
    direction: Literal["bullish", "bearish", "neutral"] | None,
    confidence: float,
) -> float:
    """把 (direction, confidence) 编码成 ∈ [-1, +1] 的标量。

    bullish → +confidence
    bearish → -confidence
    neutral / None → 0
    """
    if direction == "bullish":
        return max(0.0, min(1.0, confidence))
    if direction == "bearish":
        return -max(0.0, min(1.0, confidence))
    return 0.0


def _news_score(
    news_intensity: str | None,
    news_direction: str | None,
) -> float:
    """news_direction ∈ {with, against, neutral}, news_intensity ∈
    {none, low, medium, high}.

    把 ``with`` 视作 +intensity（顺势利好），``against`` 视作 -intensity，
    映射为 ∈ [-1, +1]。注意 fusion 上层并不知道当前持仓方向 —— 这里
    给出的是相对"既有 exposure"的 with/against，仅当上层把 news 当成
    短期事件冲击（不直接决定 long/short）时才有意义。

    上层会把 score 乘以 weight_news（默认 0.10），影响很小，作为微调。
    """
    intensity_map = {"none": 0.0, "low": 0.25, "medium": 0.6, "high": 1.0}
    base = intensity_map.get(news_intensity or "none", 0.0)
    if news_direction == "with":
        return +base
    if news_direction == "against":
        return -base
    return 0.0


def _component_agreement(scores: list[float]) -> float:
    """方向一致度 ∈ [0, 1] —— 同号组件 / 非零组件。"""
    nonzero = [s for s in scores if abs(s) > 1e-9]
    if not nonzero:
        return 0.0
    pos = sum(1 for s in nonzero if s > 0)
    neg = len(nonzero) - pos
    return max(pos, neg) / len(nonzero)


def score_fusion(
    *,
    config: FusionConfig,
    ai_direction: Literal["bullish", "bearish", "neutral"] | None,
    ai_confidence: float,
    macro_direction: Literal["bullish", "bearish", "neutral"] | None,
    macro_total_bias: float,
    smc_score: float | None,
    news_intensity: str | None,
    news_direction: str | None,
    anomaly_level: str | None,
) -> FusedDirection:
    """合成最终 FusedDirection。

    输入完全 nullable —— 任何组件为 None / 0 时其权重相当于 0。
    """
    sources: list[str] = []
    components: dict[str, float] = {}

    # AI —— 关闭时 raw=0，权重相当于 0
    ai_score_raw = 0.0
    if ai_direction is not None and config.enable_ai_direction:
        ai_score_raw = _direction_to_score(ai_direction, ai_confidence)
        components["ai"] = ai_score_raw
        if abs(ai_score_raw) > 1e-9:
            sources.append("ai")

    # Macro
    macro_score_raw = 0.0
    if macro_direction is not None and config.enable_macro_layer:
        # MacroBias.total_bias 范围 ≈ [-0.30, +0.30] —— 放大到 [-1, +1]
        macro_score_raw = max(-1.0, min(1.0, macro_total_bias / 0.30))
        components["macro"] = macro_score_raw
        if abs(macro_score_raw) > 1e-9:
            sources.append("macro")

    # SMC
    smc_score_raw = 0.0
    if smc_score is not None and config.enable_smc_detection:
        smc_score_raw = smc_score
        components["smc"] = smc_score_raw
        if abs(smc_score_raw) > 1e-9:
            sources.append("smc")

    # News
    news_score_raw = _news_score(news_intensity, news_direction)
    components["news"] = news_score_raw
    if abs(news_score_raw) > 1e-9:
        sources.append("news")

    # 加权
    weighted = (
        config.weight_ai * ai_score_raw
        + config.weight_macro * macro_score_raw
        + config.weight_smc * smc_score_raw
        + config.weight_news * news_score_raw
    )

    # Anomaly penalty —— 乘性衰减
    anomaly_mult = anomaly_confidence_penalty(anomaly_level)
    fused_score = weighted * anomaly_mult
    fused_score = max(-1.0, min(1.0, fused_score))

    # 方向
    direction: Literal["bullish", "bearish", "neutral"]
    if fused_score > config.direction_threshold:
        direction = "bullish"
    elif fused_score < -config.direction_threshold:
        direction = "bearish"
    else:
        direction = "neutral"

    # 一致度 + confidence
    agreement = _component_agreement([
        ai_score_raw, macro_score_raw, smc_score_raw, news_score_raw,
    ])
    base_confidence = abs(fused_score)
    confidence = max(0.0, min(1.0, base_confidence * (0.5 + 0.5 * agreement)))

    reasoning = (
        f"ai={ai_score_raw:+.2f} macro={macro_score_raw:+.2f} "
        f"smc={smc_score_raw:+.2f} news={news_score_raw:+.2f} "
        f"anomaly_mult={anomaly_mult:.2f} fused={fused_score:+.3f}"
    )

    return FusedDirection(
        direction=direction,
        confidence=confidence,
        fused_score=fused_score,
        component_agreement=agreement,
        components=components,
        sources_used=tuple(sources),
        reasoning=reasoning,
    )
