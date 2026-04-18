"""Live-path position sizing helper — single source of truth.

audit-r2 R1 + ops-sustain round-2 review:

- Pulled out of scripts/live_demo.py so both pre-write gate and save_state
  call the SAME helper; previously margin_cap read `best.position_size_lots`
  as an attribute (never existed on TradeSetup/RangeSetup instances) and
  silently fell back to cfg.min_lot, disabling the gate for any lot > 0.01.

- balance_usd is intentionally Optional: callers must supply a concrete
  value or accept 0.0 lots (fail-closed).  Never default to a literal
  (e.g. 10_000) because that magnitude on a $1k demo account over-sizes
  by 10x.

- Pure function, no MT5 / Python side effects → directly importable by
  tests without the MetaTrader5 module at scripts/live_demo.py top-level.
"""
from __future__ import annotations

from typing import Any

from smc.risk.position_sizer import compute_position_size


def compute_live_position_size(
    best_setup: Any,
    *,
    cfg: Any,
    balance_usd: float | None,
    risk_pct: float,
    blocked_reason: str | None,
) -> float:
    """Return lot size for live_state.json / pre_write_gate.

    Returns 0.0 (fail-closed) when:
      - blocked_reason is set (risk gate already tripped)
      - best_setup is None
      - balance_usd is None or non-positive (fail-closed on unknown equity)
      - cfg missing pip_value_per_lot / point_size
      - entry or stop_loss missing / invalid (<=0)
      - SL distance in points is <=0
      - compute_position_size raises (log externally if caller wants)
    """
    if blocked_reason or best_setup is None:
        return 0.0
    if balance_usd is None or balance_usd <= 0:
        return 0.0

    # RangeSetup: .entry_price / .stop_loss directly.
    # TradeSetup: via .entry_signal.
    if hasattr(best_setup, "entry_signal"):
        e = best_setup.entry_signal
        entry = getattr(e, "entry_price", None)
        sl = getattr(e, "stop_loss", None)
    else:
        entry = getattr(best_setup, "entry_price", None)
        sl = getattr(best_setup, "stop_loss", None)

    if entry is None or sl is None or entry <= 0 or sl <= 0:
        return 0.0

    point_size = getattr(cfg, "point_size", None)
    pip_value_per_lot = getattr(cfg, "pip_value_per_lot", None)
    min_lot = getattr(cfg, "min_lot", 0.01)
    if not point_size or not pip_value_per_lot:
        return 0.0

    sl_distance_points = abs(entry - sl) / point_size
    if sl_distance_points <= 0:
        return 0.0

    try:
        size = compute_position_size(
            balance_usd=balance_usd,
            risk_pct=risk_pct,
            sl_distance_points=sl_distance_points,
            pip_value_per_lot=pip_value_per_lot,
            min_lot_size=min_lot,
        )
        return float(size.lots)
    except Exception:
        return 0.0
