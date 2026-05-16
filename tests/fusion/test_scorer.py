"""FusionScorer 单元测试。

只测纯函数 ``score_fusion`` —— 不依赖外部状态、不读 IO。
覆盖：方向、置信度、组件一致度、anomaly penalty。
"""

from __future__ import annotations

import pytest

from smc.fusion.contracts import FusionConfig
from smc.fusion.scorer import score_fusion


@pytest.fixture
def default_config() -> FusionConfig:
    return FusionConfig()


@pytest.mark.unit
def test_all_components_bullish_yield_bullish(default_config):
    """AI+Macro+SMC+News 全部看多 → 融合方向 bullish + 高置信度。"""
    result = score_fusion(
        config=default_config,
        ai_direction="bullish",
        ai_confidence=0.80,
        macro_direction="bullish",
        macro_total_bias=0.20,
        smc_score=0.75,
        news_intensity="medium",
        news_direction="with",
        anomaly_level="NORMAL",
    )
    assert result.direction == "bullish"
    assert result.confidence > 0.4
    assert result.component_agreement == pytest.approx(1.0)
    assert "ai" in result.sources_used
    assert "macro" in result.sources_used
    assert "smc" in result.sources_used


@pytest.mark.unit
def test_all_components_bearish_yield_bearish(default_config):
    result = score_fusion(
        config=default_config,
        ai_direction="bearish",
        ai_confidence=0.80,
        macro_direction="bearish",
        macro_total_bias=-0.20,
        smc_score=-0.75,
        news_intensity="high",
        news_direction="with",
        anomaly_level="NORMAL",
    )
    assert result.direction == "bearish"
    assert result.fused_score < -0.15


@pytest.mark.unit
def test_split_components_yield_neutral_or_low_conf(default_config):
    """AI 看多但 SMC 看空 → 方向分歧 → component_agreement < 1。"""
    result = score_fusion(
        config=default_config,
        ai_direction="bullish",
        ai_confidence=0.70,
        macro_direction="neutral",
        macro_total_bias=0.0,
        smc_score=-0.6,
        news_intensity="none",
        news_direction=None,
        anomaly_level="NORMAL",
    )
    assert result.component_agreement < 1.0


@pytest.mark.unit
def test_anomaly_lockdown_neutralizes(default_config):
    """LOCKDOWN 时 anomaly_mult=0，最终 fused_score=0 → neutral。"""
    result = score_fusion(
        config=default_config,
        ai_direction="bullish",
        ai_confidence=0.95,
        macro_direction="bullish",
        macro_total_bias=0.25,
        smc_score=0.9,
        news_intensity="high",
        news_direction="with",
        anomaly_level="LOCKDOWN",
    )
    assert result.direction == "neutral"
    assert result.fused_score == pytest.approx(0.0)


@pytest.mark.unit
def test_anomaly_elevated_dampens_but_keeps_direction(default_config):
    result = score_fusion(
        config=default_config,
        ai_direction="bullish",
        ai_confidence=0.90,
        macro_direction="bullish",
        macro_total_bias=0.25,
        smc_score=0.9,
        news_intensity="medium",
        news_direction="with",
        anomaly_level="ELEVATED",
    )
    assert result.direction == "bullish"


@pytest.mark.unit
def test_disable_ai_layer_via_config():
    cfg = FusionConfig(enable_ai_direction=False)
    result = score_fusion(
        config=cfg,
        ai_direction="bullish",
        ai_confidence=0.99,
        macro_direction="neutral",
        macro_total_bias=0.0,
        smc_score=0.0,
        news_intensity="none",
        news_direction=None,
        anomaly_level="NORMAL",
    )
    # AI 被关闭 → 没有任何信号 → neutral
    assert result.direction == "neutral"
    assert "ai" not in result.sources_used


@pytest.mark.unit
def test_score_clamped_to_unit_range(default_config):
    result = score_fusion(
        config=default_config,
        ai_direction="bullish",
        ai_confidence=10.0,  # 故意越界
        macro_direction="bullish",
        macro_total_bias=10.0,
        smc_score=10.0,
        news_intensity="high",
        news_direction="with",
        anomaly_level="NORMAL",
    )
    assert -1.0 <= result.fused_score <= 1.0
    assert 0.0 <= result.confidence <= 1.0
