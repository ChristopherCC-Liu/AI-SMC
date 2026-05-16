"""PerceptionLayer 单元测试。

测点：
- regime_v2 来自 classify_regime_v2；
- bars_provider 缺省时降级 graceful，degradation_notes 有标记；
- bars_provider crash 时返回 None smc_score、记 degradation；
- AnomalyDetector 联动通过。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone

import pytest

from smc.fusion.contracts import FusionConfig
from smc.fusion.perception import PerceptionLayer
from smc.hedgerock.decision_server import MarketFeatures


@pytest.fixture
def now() -> datetime:
    return datetime(2026, 5, 16, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def features() -> MarketFeatures:
    return MarketFeatures(
        volatility_rank=0.45,
        hh_count=5,
        ll_count=2,
        h4_trend_bars=4,
        regime="TREND_UP",
    )


class _FakeBars:
    """生成稳定 trending up 的 200 根 bar，让 anomaly NORMAL + SMC bullish."""

    def __init__(self, base_price: float = 2000.0, step: float = 0.5) -> None:
        self._base = base_price
        self._step = step

    def get_recent_bars(
        self, symbol: str, timeframe: str, count: int,
    ) -> Sequence[Mapping[str, float]]:
        return [
            {
                "open": self._base + i * self._step,
                "high": self._base + i * self._step + 0.3,
                "low": self._base + i * self._step - 0.3,
                "close": self._base + i * self._step + 0.1,
                "volume": 100.0,
            }
            for i in range(count)
        ]


class _CrashingBars:
    def get_recent_bars(self, *args, **kwargs):
        raise RuntimeError("simulated bars provider failure")


@pytest.mark.unit
def test_perception_without_bars_provider_marks_degradation(features, now):
    layer = PerceptionLayer(config=FusionConfig())
    snap = layer.snapshot(
        symbol="XAUUSD", features=features, news_classification=None,
        spread_pts=None, now=now,
    )
    assert snap.regime_v2 in {"trend_up", "trend_down", "range", "news",
                              "breakout", "crisis", "unknown"}
    assert "smc_no_bars_provider" in snap.degradation_notes
    assert "anomaly_no_bars_provider" in snap.degradation_notes


@pytest.mark.unit
def test_perception_with_uptrend_bars_yields_positive_smc(features, now):
    layer = PerceptionLayer(
        config=FusionConfig(), bars_provider=_FakeBars(),
    )
    snap = layer.snapshot(
        symbol="XAUUSD", features=features, news_classification=None,
        spread_pts=None, now=now,
    )
    # bars 都在涨 → smc_score > 0
    assert snap.smc_score is not None
    assert snap.smc_score > 0.0
    # 平稳 trending → anomaly 不应到 LOCKDOWN/CRITICAL
    assert snap.anomaly_level not in ("LOCKDOWN", "CRITICAL")


@pytest.mark.unit
def test_perception_bars_crash_is_degradation_not_exception(features, now):
    layer = PerceptionLayer(
        config=FusionConfig(), bars_provider=_CrashingBars(),
    )
    snap = layer.snapshot(
        symbol="XAUUSD", features=features, news_classification=None,
        spread_pts=None, now=now,
    )
    # 不应 raise，且 degradation 有记录
    assert snap.smc_score is None
    assert any("failed" in note or "crash" in note
               for note in snap.degradation_notes)


@pytest.mark.unit
def test_perception_disabled_components(features, now):
    cfg = FusionConfig(enable_smc_detection=False, enable_anomaly_shield=False)
    layer = PerceptionLayer(config=cfg, bars_provider=_FakeBars())
    snap = layer.snapshot(
        symbol="XAUUSD", features=features, news_classification=None,
        spread_pts=None, now=now,
    )
    assert snap.smc_score is None
    assert snap.anomaly_level is None
    assert "smc_disabled" in snap.degradation_notes
    assert "anomaly_disabled" in snap.degradation_notes
