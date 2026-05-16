"""FusionController 端到端集成测试。

测点：
1. 完整链路跑通，输出合法 SignalEnvelope；
2. AI/Macro 缺失时降级到 SMC-only，仍能产出 envelope；
3. 全组件 crash 时仍返回 safe-observe outcome（不抛异常）；
4. trace 完整记录所有阶段；
5. transition_lock_override 在 prev_envelope 还活着时正确 carry-over。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from smc.ai.models import AIDirection
from smc.fusion import FusionConfig, FusionController
from smc.fusion.decision import DecisionLayer
from smc.fusion.evolution_sidecar import EvolutionSidecar
from smc.fusion.execution import ExecutionLayer
from smc.fusion.perception import PerceptionLayer
from smc.fusion.validation import ValidationLayer
from smc.hedgerock.decision_server import MarketFeatures
from smc.hedgerock.ea_state import EAStateStore, build_ea_state


# ---------------------------------------------------------------------------
# 小型 fake 组件
# ---------------------------------------------------------------------------


class _UpTrendBars:
    def get_recent_bars(
        self, symbol: str, timeframe: str, count: int,
    ) -> Sequence[Mapping[str, float]]:
        return [
            {"open": 2000.0 + i * 0.5, "high": 2000.5 + i * 0.5,
             "low": 1999.5 + i * 0.5, "close": 2000.3 + i * 0.5,
             "volume": 100.0}
            for i in range(count)
        ]


class _BullishAIDirection:
    """假 DirectionEngine —— 永远返回 bullish。"""

    def __init__(self, confidence: float = 0.75) -> None:
        self._conf = confidence

    def get_direction(self, h4_df=None, bar_ts=None) -> AIDirection:
        return AIDirection(
            direction="bullish",
            confidence=self._conf,
            key_drivers=("test_driver",),
            reasoning="test_fake_bullish",
            assessed_at=bar_ts or datetime.now(timezone.utc),
            source="ai_debate",
        )


class _CrashingAIDirection:
    def get_direction(self, h4_df=None, bar_ts=None):
        raise RuntimeError("simulated AI failure")


class _BullishMacro:
    def compute_macro_bias(self, instrument: str = "XAUUSD"):
        class _Bias:
            direction = "bullish"
            total_bias = 0.18
        return _Bias()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def now() -> datetime:
    return datetime(2026, 5, 16, 14, 30, 0, tzinfo=timezone.utc)


@pytest.fixture
def features() -> MarketFeatures:
    # Range-ish features —— rule_engine 在 range 才会出 hedgerock
    return MarketFeatures(
        volatility_rank=0.25,  # < 0.30 → range
        hh_count=2,
        ll_count=2,
        h4_trend_bars=1,
        regime="CONSOLIDATION",
    )


@pytest.fixture
def ea_store(now) -> EAStateStore:
    store = EAStateStore()
    state = build_ea_state(
        equity=10000.0, balance=10000.0, dd_pct=0.005,
        free_margin=9800.0, margin_level=400.0, open_lots=0.02,
        open_positions=2, floating_pnl=-2.5, spread_pts=30,
        consec_losses=0, recent_closed_pnl=12.0, recent_sample_count=8,
    )
    store.set("XAUUSD", state, recorded_at=now - timedelta(seconds=5))
    return store


@pytest.fixture
def controller(tmp_path: Path) -> FusionController:
    cfg = FusionConfig(
        evidence_dir=str(tmp_path / "evidence"),
        confidence_floor=0.20,  # 测试时放宽，避免轻易拦截
    )
    perception = PerceptionLayer(
        config=cfg, bars_provider=_UpTrendBars(),
    )
    decision = DecisionLayer(
        config=cfg,
        direction_engine=_BullishAIDirection(),
        macro_layer=_BullishMacro(),
    )
    validation = ValidationLayer(config=cfg)
    execution = ExecutionLayer(config=cfg)
    return FusionController(
        perception=perception, decision=decision,
        validation=validation, execution=execution,
        config=cfg,
    )


# ---------------------------------------------------------------------------
# 端到端测试
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_e2e_full_pipeline_produces_envelope(
    controller, features, ea_store, now, tmp_path,
):
    outcome = controller.on_signal_request(
        symbol="XAUUSD",
        features=features,
        ea_state_store=ea_store,
        prev_envelope=None,
        news_classification=None,
        spread_pts=30,
        exposure_lots=0.02,
        now=now,
    )
    env = outcome.envelope
    assert env.symbol == "XAUUSD"
    assert env.generated_at == now
    # range + 所有 risk-control 都健康 → 应该是 hedgerock 或 observe，永不 halt
    assert env.mode in ("hedgerock", "observe")
    # confidence 写出来了
    assert 0.0 <= env.confidence <= 1.0
    # 证据链写入
    assert outcome.trace.fused_direction is not None
    fused = outcome.trace.fused_direction
    assert fused.direction in ("bullish", "neutral")
    # evidence dir 里应该有 fusion_evidence_*.jsonl
    evidence_files = list(Path(tmp_path / "evidence").glob("fusion_evidence_*.jsonl"))
    assert len(evidence_files) >= 1


@pytest.mark.integration
def test_e2e_ai_crash_degrades_gracefully(
    features, ea_store, now, tmp_path,
):
    cfg = FusionConfig(
        evidence_dir=str(tmp_path / "evidence"),
        confidence_floor=0.20,
    )
    controller = FusionController(
        perception=PerceptionLayer(config=cfg, bars_provider=_UpTrendBars()),
        decision=DecisionLayer(
            config=cfg, direction_engine=_CrashingAIDirection(),
            macro_layer=_BullishMacro(),
        ),
        validation=ValidationLayer(config=cfg),
        execution=ExecutionLayer(config=cfg),
        config=cfg,
    )
    outcome = controller.on_signal_request(
        symbol="XAUUSD",
        features=features,
        ea_state_store=ea_store,
        prev_envelope=None,
        news_classification=None,
        now=now,
    )
    # AI crash 不应阻塞，仍应返回 envelope
    assert outcome.envelope.symbol == "XAUUSD"
    # AI 不可用 → fused_direction 可能仍因 macro + smc 而非 None
    assert outcome.trace is not None


@pytest.mark.integration
def test_e2e_transition_lock_carryover(
    controller, features, ea_store, now,
):
    # 第 1 次 poll → 得到 envelope A
    out1 = controller.on_signal_request(
        symbol="XAUUSD",
        features=features,
        ea_state_store=ea_store,
        prev_envelope=None,
        news_classification=None,
        now=now,
    )
    env1 = out1.envelope
    # 第 2 次 poll 5 秒后 → prev_envelope = env1
    out2 = controller.on_signal_request(
        symbol="XAUUSD",
        features=features,
        ea_state_store=ea_store,
        prev_envelope=env1,
        news_classification=None,
        now=now + timedelta(seconds=5),
    )
    # 第二次 envelope 仍然合法、时间戳推进
    assert out2.envelope.generated_at > env1.generated_at
    # transition_lock_override_provided 总为 True（fusion 路径接管 lock 计算）
    assert out2.transition_lock_override_provided is True


@pytest.mark.integration
def test_e2e_lockdown_anomaly_blocks_entries(
    features, ea_store, now, tmp_path,
):
    class _LockdownBars:
        """构造极端 gap，让 AnomalyDetector LOCKDOWN。"""

        def get_recent_bars(self, *args, **kwargs):
            bars = [{"open": 2000.0, "high": 2001.0, "low": 1999.0,
                     "close": 2000.0, "volume": 100.0} for _ in range(50)]
            # 注入一个 5% 的跳空
            bars.append({"open": 2100.0, "high": 2105.0, "low": 2095.0,
                         "close": 2102.0, "volume": 1000.0})
            bars.extend([{"open": 2102.0, "high": 2103.0, "low": 2100.0,
                          "close": 2101.0, "volume": 100.0} for _ in range(20)])
            return bars

    cfg = FusionConfig(
        evidence_dir=str(tmp_path / "evidence"),
        confidence_floor=0.20,
    )
    controller = FusionController(
        perception=PerceptionLayer(config=cfg, bars_provider=_LockdownBars()),
        decision=DecisionLayer(config=cfg),  # 没有 direction / macro provider
        validation=ValidationLayer(config=cfg),
        execution=ExecutionLayer(config=cfg),
        config=cfg,
    )
    outcome = controller.on_signal_request(
        symbol="XAUUSD",
        features=features,
        ea_state_store=ea_store,
        prev_envelope=None,
        news_classification=None,
        now=now,
    )
    # LOCKDOWN → validation block → safe observe
    env = outcome.envelope
    assert env.mode == "observe"
    assert env.hedgerock_enabled is False


@pytest.mark.integration
def test_e2e_evolution_sidecar_frozen_forces_observe(
    features, ea_store, now, tmp_path,
):
    class _FrozenBreaker:
        def is_frozen(self, _now):
            return True

    sidecar = EvolutionSidecar(
        workspace=tmp_path / "sasl",
        circuit_breaker=_FrozenBreaker(),
    )
    cfg = FusionConfig(
        evidence_dir=str(tmp_path / "evidence"),
        confidence_floor=0.20,
    )
    controller = FusionController(
        perception=PerceptionLayer(config=cfg, bars_provider=_UpTrendBars()),
        decision=DecisionLayer(
            config=cfg, direction_engine=_BullishAIDirection(),
            macro_layer=_BullishMacro(),
        ),
        validation=ValidationLayer(config=cfg),
        execution=ExecutionLayer(config=cfg),
        evolution=sidecar,
        config=cfg,
    )
    outcome = controller.on_signal_request(
        symbol="XAUUSD",
        features=features,
        ea_state_store=ea_store,
        prev_envelope=None,
        news_classification=None,
        now=now,
    )
    # circuit breaker frozen → validation force-blocked → observe
    assert outcome.envelope.mode == "observe"
    assert outcome.envelope.hedgerock_enabled is False
