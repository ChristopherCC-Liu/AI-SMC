"""BTCUSD InstrumentConfig — percentage-based range detection, 24/7 market."""
from __future__ import annotations

from smc.instruments.registry import SYMBOL_REGISTRY
from smc.instruments.types import InstrumentConfig

BTCUSD_CONFIG = InstrumentConfig(
    symbol="BTCUSD",
    mt5_path="Bitcoin\\BTCUSD",
    magic=19760419,
    point_size=0.01,
    contract_size=1.0,
    leverage_ratio=100,
    min_lot=0.01,
    donchian_lookback=24,
    min_range_width_points=None,
    min_range_width_pct=2.0,
    max_range_width_points=None,
    max_range_width_pct=8.0,
    boundary_pct_default=0.15,
    boundary_pct_wide=0.25,
    guard_width_low=1500.0,
    guard_width_high=1500.0,
    guard_duration_low=6,
    guard_duration_high=8,
    guard_rr_min=1.5,
    regime_trending_pct=5.0,
    regime_ranging_pct=2.0,
    sl_atr_multiplier=1.0,
    sl_min_buffer_points=None,
    sl_min_buffer_pct=0.3,
    # BTC uses same TP rr ratios as XAU initially; tune after paper week
    tp1_rr_ratio=2.5,
    tp2_rr_ratio=4.0,
    sessions={
        "HIGH_VOL": (12, 22),
        "LOW_VOL": (22, 12),
    },
    ranging_sessions=frozenset({"HIGH_VOL", "LOW_VOL"}),
    asian_sessions=frozenset(),
    # BTC has no Asian-core gate (24/7 crypto); None signals "skip Asian-specific logic"
    asian_core_session_name=None,
    wide_band_sessions=frozenset({"HIGH_VOL", "LOW_VOL"}),
    weekend_flag_active=True,
    use_asian_quota=False,
    consec_loss_limit=3,
)

SYMBOL_REGISTRY["BTCUSD"] = BTCUSD_CONFIG
