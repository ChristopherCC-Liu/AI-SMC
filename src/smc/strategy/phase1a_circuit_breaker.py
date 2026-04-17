"""Asian circuit breaker for ASIAN_CORE + ASIAN_LONDON_TRANSITION ranging.

Round 4.5 hotfix (用户指令 UTC 03:30):
  扩展到全 Asian 覆盖（UTC 0-8）— 原仅 UTC 6-8 ASIAN_LONDON_TRANSITION。
  类名保留 Phase1aCircuitBreaker 避免 import 波及。
  live_demo.py 调用点已扩展 session in ("ASIAN_CORE", "ASIAN_LONDON_TRANSITION").

Round 4.5 addendum (skeptic Gate 1 catch):
  Daily reset — trip 后每 UTC 00:00 自动清空，不再需要 operator 手动干预。
  语义: "当日保险" 而非 "永久终结"。
  实现: is_tripped() / record_trade_close() 入口处 check `last_reset_date`。

Tracks ranging trades opened in the UTC 0-8 Asian session. Trips when:
- cumulative_losses >= 3 (lose>2), OR
- cumulative_pnl_usd <= -20.0 (PnL<-$20)

Once tripped, live_demo treats both ASIAN_CORE and ASIAN_LONDON_TRANSITION
as if ranging is disabled (fall through to v1_passthrough) until UTC 00:00
rolls over, at which point the breaker auto-resets for the new day.

State is persisted to `data/phase1a_breaker_state.json` so the breaker
survives process restarts.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import date, datetime, timezone
from pathlib import Path

__all__ = ["Phase1aCircuitBreaker", "BreakerState"]

DEFAULT_STATE_PATH = Path("data/phase1a_breaker_state.json")
_LOSS_LIMIT = 3          # lose > 2 means 3+ losses
_PNL_LIMIT_USD = -20.0   # PnL < -$20


@dataclass(frozen=True)
class BreakerState:
    """Immutable snapshot of circuit breaker state."""
    losses: int = 0
    pnl_usd: float = 0.0
    tripped: bool = False
    tripped_at: str | None = None  # ISO datetime UTC
    last_updated: str | None = None
    last_reset_date: str | None = None  # ISO date (UTC) — Round 4.5 daily reset


class Phase1aCircuitBreaker:
    """Stateful circuit breaker for Phase 1a ranging trades.

    Call `record_trade_close(pnl_usd)` after every ASIAN_LONDON_TRANSITION
    ranging trade closes. Call `is_tripped()` before deciding to open.

    Round 4.5: daily auto-reset every UTC 00:00 rollover.
    """
    def __init__(self, state_path: Path | str = DEFAULT_STATE_PATH) -> None:
        self._state_path = Path(state_path)
        self._state = self._load_state()

    def _load_state(self) -> BreakerState:
        if not self._state_path.exists():
            return BreakerState(last_reset_date=_today_utc_iso())
        try:
            raw = json.loads(self._state_path.read_text())
            return BreakerState(**raw)
        except Exception:
            return BreakerState(last_reset_date=_today_utc_iso())

    def _save_state(self) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        self._state_path.write_text(json.dumps(asdict(self._state), indent=2))

    def _apply_daily_reset_if_needed(self) -> None:
        """Auto-reset state when UTC date rolls over."""
        today_iso = _today_utc_iso()
        if self._state.last_reset_date != today_iso:
            self._state = BreakerState(last_reset_date=today_iso)
            self._save_state()

    def is_tripped(self) -> bool:
        self._apply_daily_reset_if_needed()
        return self._state.tripped

    def snapshot(self) -> BreakerState:
        self._apply_daily_reset_if_needed()
        return self._state

    def record_trade_close(self, pnl_usd: float) -> BreakerState:
        """Called after an ASIAN_LONDON_TRANSITION ranging trade closes.

        Updates internal state, persists, and trips the breaker if limits hit.
        Returns the new state. Applies daily reset first so stale trip state
        from yesterday does not suppress today's accounting.
        """
        self._apply_daily_reset_if_needed()

        if self._state.tripped:
            # Already tripped today, no-op further recording
            return self._state

        new_losses = self._state.losses + (1 if pnl_usd < 0 else 0)
        new_pnl = self._state.pnl_usd + pnl_usd
        now_iso = datetime.now(tz=timezone.utc).isoformat()

        trip = new_losses >= _LOSS_LIMIT or new_pnl <= _PNL_LIMIT_USD

        self._state = BreakerState(
            losses=new_losses,
            pnl_usd=round(new_pnl, 2),
            tripped=trip,
            tripped_at=now_iso if trip else None,
            last_updated=now_iso,
            last_reset_date=self._state.last_reset_date,
        )
        self._save_state()
        return self._state

    def reset(self) -> BreakerState:
        """Manual reset — operator-only override (daily reset handles normal case)."""
        self._state = BreakerState(last_reset_date=_today_utc_iso())
        self._save_state()
        return self._state


def _today_utc_iso() -> str:
    return datetime.now(tz=timezone.utc).date().isoformat()
