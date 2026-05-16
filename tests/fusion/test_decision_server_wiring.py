"""decision_server + fusion_controller 接线测试。

验证：
- create_app(fusion_controller=fc) 时，/signal 走 fusion 路径；
- 不传 fusion_controller 时，老路径仍工作（这一点已经被
  tests/hedgerock/test_decision_server.py 覆盖，这里不重复）；
- fusion crash 时 fallthrough 到 legacy path（即仍能返回 envelope，
  而不是 500）。
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from smc.ai.models import AIDirection
from smc.fusion import FusionConfig, FusionController
from smc.fusion.decision import DecisionLayer
from smc.fusion.execution import ExecutionLayer
from smc.fusion.perception import PerceptionLayer
from smc.fusion.validation import ValidationLayer
from smc.hedgerock.decision_server import (
    MarketFeatures,
    MarketFeaturesProvider,
    create_app,
)


class _StaticFeatures(MarketFeaturesProvider):
    def get_features(self, symbol: str) -> MarketFeatures:
        return MarketFeatures(
            volatility_rank=0.25,
            hh_count=2,
            ll_count=2,
            h4_trend_bars=1,
            regime="CONSOLIDATION",
        )


class _UptrendBars:
    def get_recent_bars(self, symbol, timeframe, count):
        return [
            {"open": 2000.0 + i * 0.3, "high": 2000.3 + i * 0.3,
             "low": 1999.7 + i * 0.3, "close": 2000.2 + i * 0.3,
             "volume": 100.0}
            for i in range(count)
        ]


class _BullishAI:
    def get_direction(self, h4_df=None, bar_ts=None):
        return AIDirection(
            direction="bullish",
            confidence=0.70,
            key_drivers=("d",),
            reasoning="r",
            assessed_at=bar_ts or datetime.now(timezone.utc),
            source="ai_debate",
        )


@pytest.fixture
def fusion_controller(tmp_path: Path) -> FusionController:
    cfg = FusionConfig(
        evidence_dir=str(tmp_path / "evidence"),
        confidence_floor=0.10,
    )
    return FusionController(
        perception=PerceptionLayer(config=cfg, bars_provider=_UptrendBars()),
        decision=DecisionLayer(config=cfg, direction_engine=_BullishAI()),
        validation=ValidationLayer(config=cfg),
        execution=ExecutionLayer(config=cfg),
        config=cfg,
    )


@pytest.mark.integration
def test_signal_endpoint_via_fusion_returns_envelope(fusion_controller):
    app = create_app(
        market_features_provider=_StaticFeatures(),
        fusion_controller=fusion_controller,
    )
    client = TestClient(app)
    resp = client.get("/signal", params={"symbol": "XAUUSD"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["symbol"] == "XAUUSD"
    assert body["schema_version"]
    assert "mode" in body
    assert body["regime"] in (
        "range", "trend_up", "trend_down", "news",
        "breakout", "crisis", "unknown",
    )


@pytest.mark.integration
def test_fusion_crash_falls_back_to_legacy(fusion_controller):
    """fusion_controller.on_signal_request 抛异常 → server 不应 500。"""

    class _Boom:
        def on_signal_request(self, **kwargs):
            raise RuntimeError("boom")

    app = create_app(
        market_features_provider=_StaticFeatures(),
        fusion_controller=_Boom(),
        enable_rule_engine=False,
    )
    client = TestClient(app)
    resp = client.get("/signal", params={"symbol": "XAUUSD"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["symbol"] == "XAUUSD"


@pytest.mark.integration
def test_signal_endpoint_no_fusion_still_works():
    """不传 fusion_controller → 老路径正常返回（backward-compat 哨兵）。"""
    app = create_app(
        market_features_provider=_StaticFeatures(),
    )
    client = TestClient(app)
    resp = client.get("/signal", params={"symbol": "XAUUSD"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["symbol"] == "XAUUSD"
