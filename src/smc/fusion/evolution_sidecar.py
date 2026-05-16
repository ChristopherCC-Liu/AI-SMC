"""EvolutionSidecar —— 把 SASL(漂移检测→自调参→熔断→自清洁) 包装成
一个生命周期化对象，让 fusion_controller 可以：

1. on tick: ``maybe_run_daily_cycle()`` —— 非阻塞，如果距上一次 cycle
   >= 24h 则 schedule 后台运行；
2. on critical event: ``trigger_event(event_type, payload)`` —— 立即跑
   event-triggered cycle（drift_spike / breaker_alert 等）；
3. on shutdown: ``flush()`` —— 等待后台任务完成并写最终报告。

设计要点：
- Sidecar 本身永不阻塞主线信号路径 —— 所有重活在 threading.Thread 内
  跑；主线只看 ``circuit_breaker.is_frozen()`` 是否触发；
- 如果运行时没有 evidence registry / market bars，cycle 会用空输入
  跑通（输出全是"insufficient data"的 stage report，安全降级）。
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

_LOG = logging.getLogger(__name__)

__all__ = ["EvolutionSidecar"]


class EvolutionSidecar:
    """生命周期化的 SASL 编排器代理。"""

    DEFAULT_CYCLE_INTERVAL = timedelta(hours=24)

    def __init__(
        self,
        *,
        workspace: Path | str,
        circuit_breaker: object | None = None,
        cycle_interval: timedelta | None = None,
        market_bars_provider: Callable[[], Sequence[Mapping[str, float]]] | None = None,
        baseline_bars_provider: Callable[[], Sequence[Mapping[str, float]]] | None = None,
        evidence_registry_provider: Callable[[], Mapping[str, Any]] | None = None,
        current_params_provider: Callable[[], Mapping[str, float]] | None = None,
        baseline_params_provider: Callable[[], Mapping[str, float]] | None = None,
        recent_outcomes_provider: Callable[[], Sequence[Mapping[str, float]]] | None = None,
        baseline_outcomes_provider: Callable[[], Sequence[Mapping[str, float]]] | None = None,
        apply_adjustments: bool = False,
    ) -> None:
        self._workspace = Path(workspace)
        self._workspace.mkdir(parents=True, exist_ok=True)
        self._circuit_breaker = circuit_breaker
        self._cycle_interval = cycle_interval or self.DEFAULT_CYCLE_INTERVAL
        self._apply_adjustments = apply_adjustments

        self._market_bars = market_bars_provider or (lambda: [])
        self._baseline_bars = baseline_bars_provider or (lambda: [])
        self._evidence = evidence_registry_provider or (lambda: {})
        self._current_params = current_params_provider or (lambda: {})
        self._baseline_params = baseline_params_provider or (lambda: {})
        self._recent_outcomes = recent_outcomes_provider or (lambda: [])
        self._baseline_outcomes = baseline_outcomes_provider or (lambda: [])

        self._last_cycle_at: datetime | None = None
        self._lock = threading.Lock()
        self._active_thread: threading.Thread | None = None
        # 最近一次 cycle 报告 —— 让 fusion_controller 在 fusion trace 里
        # 嵌入一行 cycle health 摘要。
        self._last_report_summary: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------

    def is_frozen(self) -> bool:
        """委托给 SASL CircuitBreaker —— 锁住时上层 force observe。"""
        cb = self._circuit_breaker
        if cb is None:
            return False
        try:
            return bool(cb.is_frozen(datetime.now(timezone.utc)))
        except Exception:
            _LOG.exception("circuit_breaker.is_frozen crashed")
            return False

    def maybe_run_daily_cycle(self, now: datetime | None = None) -> bool:
        """如果距上一次 >= cycle_interval，则在后台启动一次 daily cycle。

        返回是否启动了新 cycle。
        """
        ts = now or datetime.now(timezone.utc)
        with self._lock:
            if (
                self._last_cycle_at is not None
                and ts - self._last_cycle_at < self._cycle_interval
            ):
                return False
            if self._active_thread is not None and self._active_thread.is_alive():
                return False
            self._last_cycle_at = ts
            self._active_thread = threading.Thread(
                target=self._run_daily_cycle_sync,
                name="fusion-evolution-daily",
                args=(ts,),
                daemon=True,
            )
            self._active_thread.start()
            return True

    def trigger_event(
        self,
        *,
        event_type: str,
        payload: Mapping[str, Any] | None = None,
        now: datetime | None = None,
    ) -> None:
        """事件触发 cycle —— 后台异步运行，永不阻塞调用者。"""
        ts = now or datetime.now(timezone.utc)
        thread = threading.Thread(
            target=self._run_event_cycle_sync,
            name=f"fusion-evolution-event-{event_type}",
            args=(event_type, dict(payload or {}), ts),
            daemon=True,
        )
        thread.start()

    def flush(self, timeout_seconds: float = 30.0) -> None:
        """等待后台 cycle 结束 —— 用于 graceful shutdown。"""
        with self._lock:
            thread = self._active_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout_seconds)

    def latest_summary(self) -> Mapping[str, Any]:
        """读 latest cycle 摘要 —— fusion trace 里可以嵌一行。"""
        return dict(self._last_report_summary)

    # ------------------------------------------------------------------
    # 后台任务
    # ------------------------------------------------------------------

    def _run_daily_cycle_sync(self, ts: datetime) -> None:
        try:
            from smc.hedgerock.evolution.sasl_orchestrator import SASLOrchestrator
        except Exception:
            _LOG.exception("SASL import failed; skipping cycle")
            return
        try:
            orch = SASLOrchestrator(
                workspace=self._workspace,
                circuit_breaker=self._circuit_breaker,
                apply_adjustments=self._apply_adjustments,
            )
            report = orch.run_daily_cycle(
                market_bars=self._safe(self._market_bars),
                baseline_bars=self._safe(self._baseline_bars),
                evidence_registry=self._safe(self._evidence),
                current_params=self._safe(self._current_params),
                baseline_params=self._safe(self._baseline_params),
                recent_outcomes=self._safe(self._recent_outcomes),
                baseline_outcomes=self._safe(self._baseline_outcomes),
                now=ts,
            )
            self._last_report_summary = _summarize_report(report)
        except Exception:
            _LOG.exception("daily SASL cycle crashed")

    def _run_event_cycle_sync(
        self,
        event_type: str,
        payload: Mapping[str, Any],
        ts: datetime,
    ) -> None:
        try:
            from smc.hedgerock.evolution.sasl_orchestrator import SASLOrchestrator
        except Exception:
            _LOG.exception("SASL import failed; skipping event cycle")
            return
        try:
            orch = SASLOrchestrator(
                workspace=self._workspace,
                circuit_breaker=self._circuit_breaker,
                apply_adjustments=self._apply_adjustments,
            )
            orch.run_event_triggered(
                event_type=event_type,
                event_data=payload,
                market_bars=self._safe(self._market_bars),
                baseline_bars=self._safe(self._baseline_bars),
                evidence_registry=self._safe(self._evidence),
                current_params=self._safe(self._current_params),
                baseline_params=self._safe(self._baseline_params),
                recent_outcomes=self._safe(self._recent_outcomes),
                baseline_outcomes=self._safe(self._baseline_outcomes),
                now=ts,
            )
        except Exception:
            _LOG.exception("event SASL cycle crashed: %s", event_type)

    @staticmethod
    def _safe(provider: Callable[[], Any]) -> Any:
        try:
            return provider()
        except Exception:
            _LOG.exception("evolution provider crashed; using empty")
            return [] if provider.__name__ != "<lambda>" else []


def _summarize_report(report: Any) -> dict[str, Any]:
    """SASLCycleReport → 一个简短的 dict 摘要 (用于 fusion trace)。"""
    if report is None:
        return {}
    stages = getattr(report, "stages", ())
    return {
        "trigger": getattr(report, "trigger", "unknown"),
        "now": str(getattr(report, "now", "")),
        "n_stages": len(stages) if hasattr(stages, "__len__") else 0,
        "overall_status": getattr(report, "overall_status", "unknown"),
    }
