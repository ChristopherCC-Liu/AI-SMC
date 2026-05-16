"""PerceptionLayer —— 把 SMC 结构检测、regime 分类、异常检测、新闻
聚合成一个 PerceptionSnapshot。

设计原则：
1. 永不抛异常 —— 任一子组件失败仅在 degradation_notes 留痕；
2. 输入是已经预聚合好的 ``MarketFeatures``（来自 decision_server 的
   ``MarketFeaturesProvider``），加上若干 OHLCV bars 用于 anomaly；
3. 输出是 frozen PerceptionSnapshot，可作为字典 key、可序列化。
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any, Protocol

from smc.fusion.contracts import FusionConfig, PerceptionSnapshot
from smc.hedgerock.decision_server import MarketFeatures
from smc.hedgerock.evolution.anomaly_shield import (
    AnomalyDetector,
    AnomalyLevel,
    AnomalyState,
)
from smc.hedgerock.news_classifier import NewsClassification
from smc.hedgerock.regime_classifier_v2 import classify_regime_v2
from smc.hedgerock.schemas import RegimeV2

_LOG = logging.getLogger(__name__)

__all__ = ["BarsProvider", "PerceptionLayer", "SMCScoreFn"]


# ---------------------------------------------------------------------------
# Protocols —— 让测试可以注入 fake
# ---------------------------------------------------------------------------


class BarsProvider(Protocol):
    """提供历史 OHLCV bars 用于 anomaly 检测和 SMC 评分。"""

    def get_recent_bars(
        self, symbol: str, timeframe: str, count: int,
    ) -> Sequence[Mapping[str, float]]: ...


class SMCScoreFn(Protocol):
    """SMC 结构方向评分函数。

    返回：(score, evidence_tuple)
        score ∈ [-1.0, +1.0]，正数看多，负数看空
        evidence 是人类可读的依据字符串 tuple
    """

    def __call__(
        self,
        symbol: str,
        bars: Sequence[Mapping[str, float]],
    ) -> tuple[float, tuple[str, ...]]: ...


# ---------------------------------------------------------------------------
# 默认 SMC 评分器 —— 包装 smc_core 检测器
# ---------------------------------------------------------------------------


def _default_smc_score(
    symbol: str,
    bars: Sequence[Mapping[str, float]],
) -> tuple[float, tuple[str, ...]]:
    """轻量级 SMC 评分实现 —— 不依赖 Polars，只看 swing/OB 倾向。

    完整版应该 import smc.smc_core.* 检测器，这里给一个 stub fallback
    供没有 Polars DataFrame 的运行时路径使用。真正的 SMC 评分由调用方
    通过 ``smc_score_fn`` 注入。
    """
    if not bars:
        return 0.0, ()
    closes = [float(b.get("close", b.get("c", 0.0))) for b in bars if b]
    closes = [c for c in closes if c > 0.0]
    if len(closes) < 5:
        return 0.0, ("smc_insufficient_bars",)
    recent = closes[-5:]
    momentum = (recent[-1] - recent[0]) / max(recent[0], 1e-9)
    # 把 [-0.5%, +0.5%] 线性映射到 [-1, +1]
    score = max(-1.0, min(1.0, momentum / 0.005))
    evidence = (f"momentum_5bar={momentum:.4f}",)
    return score, evidence


# ---------------------------------------------------------------------------
# PerceptionLayer
# ---------------------------------------------------------------------------


class PerceptionLayer:
    """聚合所有感知组件的状态。

    构造参数全部可选，缺省时退化为 features-only 模式。
    """

    def __init__(
        self,
        *,
        config: FusionConfig,
        bars_provider: BarsProvider | None = None,
        anomaly_detector: AnomalyDetector | None = None,
        smc_score_fn: SMCScoreFn | None = None,
    ) -> None:
        self._config = config
        self._bars_provider = bars_provider
        self._anomaly_detector = anomaly_detector or AnomalyDetector()
        self._smc_score_fn = smc_score_fn or _default_smc_score
        # 上一次的 AnomalyState —— 用于 graded recovery
        self._prev_anomaly: dict[str, AnomalyState] = {}

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def snapshot(
        self,
        *,
        symbol: str,
        features: MarketFeatures,
        news_classification: NewsClassification | None,
        spread_pts: int | None,
        now: datetime,
    ) -> PerceptionSnapshot:
        """构造单 tick 感知快照。

        永不抛异常 —— 每个子步骤独立 try/except，失败处记 degradation。
        """
        degradation: list[str] = []

        # ---- 1) regime v2 ----
        regime_v2, regime_conf, regime_reason = self._classify_regime(
            features=features,
            news_classification=news_classification,
            spread_pts=spread_pts,
            degradation=degradation,
        )

        # ---- 2) SMC 评分 ----
        smc_score, smc_evidence = self._compute_smc_score(
            symbol=symbol,
            degradation=degradation,
        )

        # ---- 3) 异常检测 ----
        anomaly_level, anomaly_triggers = self._detect_anomaly(
            symbol=symbol,
            now=now,
            degradation=degradation,
        )

        # ---- 4) 新闻字段 ----
        news_intensity, news_direction, news_event_name = self._news_fields(
            news_classification,
        )

        return PerceptionSnapshot(
            symbol=symbol.upper(),
            generated_at=now,
            regime_v2=regime_v2,
            regime_confidence=regime_conf,
            regime_reason=regime_reason,
            smc_score=smc_score,
            smc_evidence=smc_evidence,
            anomaly_level=anomaly_level,
            anomaly_triggers=anomaly_triggers,
            news_intensity=news_intensity,
            news_direction=news_direction,
            news_event_name=news_event_name,
            degradation_notes=tuple(degradation),
        )

    # ------------------------------------------------------------------
    # 子步骤
    # ------------------------------------------------------------------

    def _classify_regime(
        self,
        *,
        features: MarketFeatures,
        news_classification: NewsClassification | None,
        spread_pts: int | None,
        degradation: list[str],
    ) -> tuple[RegimeV2, float, str]:
        news_intensity = (
            news_classification.event.intensity
            if news_classification is not None
            else None
        )
        try:
            assessment = classify_regime_v2(
                volatility_rank=features.volatility_rank,
                h4_trend_bars=features.h4_trend_bars,
                hh_count=features.hh_count,
                ll_count=features.ll_count,
                news_intensity=news_intensity,
                spread_pts=spread_pts,
            )
        except Exception:  # pragma: no cover - defensive
            _LOG.exception("classify_regime_v2 failed; fallback to unknown")
            degradation.append("regime_classifier_v2_failed")
            return "unknown", self._config.confidence_floor, "fallback:exception"
        return assessment.regime, float(assessment.confidence), assessment.reason

    def _compute_smc_score(
        self,
        *,
        symbol: str,
        degradation: list[str],
    ) -> tuple[float | None, tuple[str, ...]]:
        if not self._config.enable_smc_detection:
            degradation.append("smc_disabled")
            return None, ()
        if self._bars_provider is None:
            degradation.append("smc_no_bars_provider")
            return None, ()
        try:
            bars = self._bars_provider.get_recent_bars(symbol, "M15", 200)
        except Exception:
            _LOG.exception("bars_provider crashed for %s", symbol)
            degradation.append("smc_bars_provider_failed")
            return None, ()
        if not bars:
            degradation.append("smc_empty_bars")
            return 0.0, ()
        try:
            return self._smc_score_fn(symbol, bars)
        except Exception:
            _LOG.exception("smc_score_fn crashed for %s", symbol)
            degradation.append("smc_score_failed")
            return None, ()

    def _detect_anomaly(
        self,
        *,
        symbol: str,
        now: datetime,
        degradation: list[str],
    ) -> tuple[
        str | None,
        tuple[str, ...],
    ]:
        if not self._config.enable_anomaly_shield:
            degradation.append("anomaly_disabled")
            return None, ()
        if self._bars_provider is None:
            degradation.append("anomaly_no_bars_provider")
            return None, ()
        try:
            bars = self._bars_provider.get_recent_bars(symbol, "M15", 200)
        except Exception:
            _LOG.exception("bars_provider crashed (anomaly path) for %s", symbol)
            degradation.append("anomaly_bars_failed")
            return None, ()
        try:
            prev = self._prev_anomaly.get(symbol.upper())
            state = self._anomaly_detector.detect(
                bars=bars, previous_state=prev, now=now,
            )
            self._prev_anomaly[symbol.upper()] = state
        except Exception:
            _LOG.exception("AnomalyDetector.detect crashed for %s", symbol)
            degradation.append("anomaly_detect_failed")
            return None, ()
        # AnomalyLevel 是 str enum，直接 .value
        return state.level.value, tuple(state.triggers)

    @staticmethod
    def _news_fields(
        news: NewsClassification | None,
    ) -> tuple[str | None, str | None, str | None]:
        if news is None:
            return None, None, None
        return news.event.intensity, news.direction, news.event.name


# ---------------------------------------------------------------------------
# 工具 —— 把 anomaly level 映射成 confidence 调整因子
# ---------------------------------------------------------------------------


def anomaly_confidence_penalty(level: str | None) -> float:
    """根据异常级别返回置信度乘数 ∈ (0,1]。

    LOCKDOWN → 0.0（直接屏蔽）
    CRITICAL → 0.4
    ELEVATED → 0.7
    NORMAL / None → 1.0
    """
    if level == AnomalyLevel.LOCKDOWN.value:
        return 0.0
    if level == AnomalyLevel.CRITICAL.value:
        return 0.4
    if level == AnomalyLevel.ELEVATED.value:
        return 0.7
    return 1.0
