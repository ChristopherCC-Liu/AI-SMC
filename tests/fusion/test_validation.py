"""ValidationLayer 单元测试。

测点：
- LOCKDOWN 时 allowed=False；
- gate snapshot critical fail 时 allowed=False；
- confidence 低于 floor 时 allowed=False；
- 证据链写入到 evidence_dir。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from smc.fusion.contracts import (
    FusedDirection,
    FusionConfig,
    PerceptionSnapshot,
)
from smc.fusion.validation import ValidationLayer


@pytest.fixture
def now() -> datetime:
    return datetime(2026, 5, 16, 9, 0, 0, tzinfo=timezone.utc)


def _make_snapshot(
    *,
    regime_v2: str = "trend_up",
    confidence: float = 0.80,
    anomaly_level: str | None = "NORMAL",
) -> PerceptionSnapshot:
    return PerceptionSnapshot(
        symbol="XAUUSD",
        generated_at=datetime.now(timezone.utc),
        regime_v2=regime_v2,
        regime_confidence=confidence,
        regime_reason="fixture",
        anomaly_level=anomaly_level,
    )


def _make_fused(
    *,
    direction: str = "bullish",
    confidence: float = 0.55,
) -> FusedDirection:
    return FusedDirection(
        direction=direction,
        confidence=confidence,
        fused_score=confidence if direction == "bullish" else -confidence,
        component_agreement=1.0,
    )


@pytest.mark.unit
def test_validation_normal_path_allows(now):
    layer = ValidationLayer(config=FusionConfig())
    verdict = layer.validate(
        snapshot=_make_snapshot(),
        fused=_make_fused(confidence=0.60),
        now=now,
    )
    assert verdict.allowed is True
    assert verdict.full_lockdown is False


@pytest.mark.unit
def test_validation_lockdown_blocks(now):
    layer = ValidationLayer(config=FusionConfig())
    verdict = layer.validate(
        snapshot=_make_snapshot(anomaly_level="LOCKDOWN"),
        fused=_make_fused(confidence=0.95),
        now=now,
    )
    assert verdict.allowed is False
    assert verdict.full_lockdown is True
    assert "lockdown" in verdict.reason


@pytest.mark.unit
def test_validation_critical_gate_failure_blocks(now):
    layer = ValidationLayer(config=FusionConfig())
    layer.set_gate_status({"G1": False, "G2": True, "G3": True, "G4": True,
                           "G5": True, "G6": True, "G7": True, "G8": True})
    verdict = layer.validate(
        snapshot=_make_snapshot(),
        fused=_make_fused(confidence=0.70),
        now=now,
    )
    assert verdict.allowed is False
    assert "critical_promotion_gate" in verdict.reason


@pytest.mark.unit
def test_validation_low_confidence_blocks(now):
    cfg = FusionConfig(confidence_floor=0.50)
    layer = ValidationLayer(config=cfg)
    verdict = layer.validate(
        snapshot=_make_snapshot(),
        fused=_make_fused(direction="bullish", confidence=0.30),
        now=now,
    )
    assert verdict.allowed is False
    assert "confidence_below_floor" in verdict.reason


@pytest.mark.unit
def test_validation_neutral_direction_bypasses_confidence_floor(now):
    cfg = FusionConfig(confidence_floor=0.99)
    layer = ValidationLayer(config=cfg)
    verdict = layer.validate(
        snapshot=_make_snapshot(),
        fused=_make_fused(direction="neutral", confidence=0.05),
        now=now,
    )
    # neutral 方向 → 不在 confidence floor 检查范围（无方向无所谓置信度）
    assert verdict.allowed is True


@pytest.mark.unit
def test_validation_records_evidence_to_jsonl(tmp_path: Path, now):
    cfg = FusionConfig(evidence_dir=str(tmp_path))
    layer = ValidationLayer(config=cfg)
    path = layer.record_evidence(
        trace_payload={"hello": "world"}, now=now,
    )
    assert path is not None
    assert Path(path).exists()
    line = Path(path).read_text(encoding="utf-8").strip()
    assert json.loads(line) == {"hello": "world"}


@pytest.mark.unit
def test_validation_evidence_disabled_returns_none(now):
    cfg = FusionConfig(enable_evidence_chain=False)
    layer = ValidationLayer(config=cfg)
    assert layer.record_evidence(
        trace_payload={"x": 1}, now=now,
    ) is None
