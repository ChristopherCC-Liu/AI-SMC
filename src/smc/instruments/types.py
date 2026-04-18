"""InstrumentConfig dataclass — single source of truth for per-symbol parameters."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InstrumentConfig:
    # Identity
    symbol: str
    mt5_path: str
    magic: int

    # Contract economics
    point_size: float
    contract_size: float
    leverage_ratio: int
    min_lot: float

    # Range detection
    donchian_lookback: int
    min_range_width_points: float | None
    min_range_width_pct: float | None
    max_range_width_points: float | None
    max_range_width_pct: float | None
    boundary_pct_default: float
    boundary_pct_wide: float

    # Guards
    guard_width_low: float
    guard_width_high: float
    guard_duration_low: int
    guard_duration_high: int
    guard_rr_min: float

    # Regime
    regime_trending_pct: float
    regime_ranging_pct: float

    # SL/TP
    sl_atr_multiplier: float
    sl_min_buffer_points: float | None
    sl_min_buffer_pct: float | None
    tp1_rr_ratio: float
    tp2_rr_ratio: float

    # Session
    sessions: dict[str, tuple[int, int]]
    ranging_sessions: frozenset[str]
    asian_sessions: frozenset[str]
    asian_core_session_name: str | None  # literal session name used in mode_router priority gates
    wide_band_sessions: frozenset[str]
    weekend_flag_active: bool

    # Quota / halt
    use_asian_quota: bool
    consec_loss_limit: int
