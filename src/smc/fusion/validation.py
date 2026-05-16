"""ValidationLayer —— 在 rule_engine 之前对 fused signal 做 gate 检查。

包含：
1. Anomaly shield 转换 —— level → confidence_threshold_multiplier；
2. Promotion gate snapshot —— 启动时一次性加载最近通过的 candidate
   ID 集合（运行时只是查 dict）；
3. Evidence chain —— 把这一次决策的 trace 写到 ShadowArtefact JSONL。

设计原则：
- ValidationLayer 不阻止信号 —— 它只输出一个 ValidationVerdict，
  ExecutionLayer 决定如何降级；
- 离线 candidate 评估（G1–G8 完整跑一遍）是慢操作，**不**在
  on_signal_request 路径上跑 —— 那个由 evolution sidecar 离线做，
  这里只读最新结果。
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from smc.fusion.contracts import (
    FusedDirection,
    FusionConfig,
    PerceptionSnapshot,
    ValidationVerdict,
)
from smc.fusion.perception import anomaly_confidence_penalty
from smc.hedgerock.evolution.anomaly_shield import (
    AnomalyLevel,
    AnomalyState,
    ShieldAction,
    shield_action,
)

_LOG = logging.getLogger(__name__)

__all__ = ["ValidationLayer"]


class ValidationLayer:
    """运行时验证 —— gates / anomaly shield / evidence chain。"""

    def __init__(
        self,
        *,
        config: FusionConfig,
        gate_snapshot_path: Path | str | None = None,
    ) -> None:
        self._config = config
        self._gate_snapshot_path = (
            Path(gate_snapshot_path) if gate_snapshot_path else None
        )
        self._evidence_dir = (
            Path(config.evidence_dir) if config.evidence_dir else None
        )
        if self._evidence_dir is not None:
            self._evidence_dir.mkdir(parents=True, exist_ok=True)
        self._gate_status: dict[str, bool] = self._load_gate_snapshot()

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def validate(
        self,
        *,
        snapshot: PerceptionSnapshot,
        fused: FusedDirection,
        now: datetime,
    ) -> ValidationVerdict:
        # 1) Anomaly shield —— 把 perception 的 level 转成 ShieldAction
        shield = self._compute_shield(snapshot=snapshot, now=now)

        if shield.full_lockdown:
            verdict_reason = f"anomaly_lockdown:{','.join(shield.detail_reasons)}"
            return ValidationVerdict(
                allowed=False,
                reason=verdict_reason,
                gate_status=dict(self._gate_status),
                confidence_threshold_multiplier=(
                    shield.confidence_threshold_multiplier
                ),
                cooldown_extension_minutes=shield.cooldown_extension_minutes,
                full_lockdown=True,
            )

        # 2) Promotion gates snapshot —— 任一关键 gate 失败时降级
        critical_gates_ok = self._critical_gates_ok()
        if not critical_gates_ok:
            return ValidationVerdict(
                allowed=False,
                reason="critical_promotion_gate_failed",
                gate_status=dict(self._gate_status),
                confidence_threshold_multiplier=(
                    shield.confidence_threshold_multiplier
                ),
                cooldown_extension_minutes=shield.cooldown_extension_minutes,
                full_lockdown=False,
            )

        # 3) Confidence floor —— fused.confidence 太低就不放行
        effective_floor = self._config.confidence_floor * max(
            1.0, shield.confidence_threshold_multiplier,
        )
        if fused.direction != "neutral" and fused.confidence < effective_floor:
            return ValidationVerdict(
                allowed=False,
                reason=(
                    f"confidence_below_floor:{fused.confidence:.2f}<"
                    f"{effective_floor:.2f}"
                ),
                gate_status=dict(self._gate_status),
                confidence_threshold_multiplier=(
                    shield.confidence_threshold_multiplier
                ),
                cooldown_extension_minutes=shield.cooldown_extension_minutes,
                full_lockdown=False,
            )

        return ValidationVerdict(
            allowed=True,
            reason=shield.banner or "ok",
            gate_status=dict(self._gate_status),
            confidence_threshold_multiplier=(
                shield.confidence_threshold_multiplier
            ),
            cooldown_extension_minutes=shield.cooldown_extension_minutes,
            full_lockdown=False,
        )

    # ------------------------------------------------------------------
    # 证据链
    # ------------------------------------------------------------------

    def record_evidence(
        self,
        *,
        trace_payload: Mapping[str, Any],
        now: datetime,
    ) -> str | None:
        """把 fusion trace 追加写到 evidence JSONL。返回写入路径或 None。"""
        if not self._config.enable_evidence_chain or self._evidence_dir is None:
            return None
        bucket = now.strftime("%Y%m%d")
        path = self._evidence_dir / f"fusion_evidence_{bucket}.jsonl"
        try:
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(trace_payload, default=str) + "\n")
        except Exception:
            _LOG.exception("failed to write evidence chain to %s", path)
            return None
        return str(path)

    # ------------------------------------------------------------------
    # 内部 helpers
    # ------------------------------------------------------------------

    def _compute_shield(
        self,
        *,
        snapshot: PerceptionSnapshot,
        now: datetime,
    ) -> ShieldAction:
        """根据 snapshot 重建 AnomalyState 并调用 shield_action。

        我们这里没有 AnomalyDetector 实例 —— 而是从 perception
        snapshot 还原一个最小化 AnomalyState 调用 shield_action。
        """
        level_str = snapshot.anomaly_level
        if level_str is None:
            level = AnomalyLevel.NORMAL
        else:
            try:
                level = AnomalyLevel(level_str)
            except ValueError:
                level = AnomalyLevel.NORMAL
        state = AnomalyState(
            level=level,
            triggers=snapshot.anomaly_triggers,
            short_window_vol=0.0,
            historical_vol_p90=0.0,
            historical_vol_p95=0.0,
            historical_vol_p99=0.0,
            max_gap_pct=0.0,
            n_bars_observed=0,
            last_anomaly_at=None,
            next_recovery_at=None,
            blocking_conditions=(),
            generated_at=now.isoformat(),
        )
        return shield_action(state)

    def _load_gate_snapshot(self) -> dict[str, bool]:
        """加载离线 promotion gate 评估结果（如果存在）。

        Schema: {"G1": true, "G2": true, ..., "G8": false}
        """
        if (
            not self._config.enable_promotion_gates
            or self._gate_snapshot_path is None
        ):
            # 缺省全部通过，避免冷启动时屏蔽所有信号
            return {f"G{i}": True for i in range(1, 9)}
        if not self._gate_snapshot_path.exists():
            _LOG.warning(
                "gate snapshot %s not found; defaulting all gates to True",
                self._gate_snapshot_path,
            )
            return {f"G{i}": True for i in range(1, 9)}
        try:
            data = json.loads(self._gate_snapshot_path.read_text(encoding="utf-8"))
            return {k: bool(v) for k, v in data.items()}
        except Exception:
            _LOG.exception(
                "failed to parse gate snapshot %s; defaulting to True",
                self._gate_snapshot_path,
            )
            return {f"G{i}": True for i in range(1, 9)}

    def _critical_gates_ok(self) -> bool:
        """G1 (min evidence), G6 (safety bounds), G7 (interface stability)
        视为 critical —— 任一失败就 force observe。"""
        critical = ("G1", "G6", "G7")
        for gate in critical:
            if not self._gate_status.get(gate, True):
                return False
        return True

    # ------------------------------------------------------------------
    # 测试钩子 —— 让单测可以 monkey-patch gate 状态
    # ------------------------------------------------------------------

    def set_gate_status(self, status: Mapping[str, bool]) -> None:
        self._gate_status = {str(k): bool(v) for k, v in status.items()}
