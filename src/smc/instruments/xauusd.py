"""XAUUSD InstrumentConfig — values sourced from live codebase.

Cross-checked against:
  - src/smc/smc_core/constants.py        (point_size=0.01)
  - src/smc/strategy/range_trader.py     (donchian, guards, boundary)
  - src/smc/strategy/regime.py           (regime thresholds)
  - src/smc/config.py                    (sl_atr_multiplier, sl_min_buffer_points)
  - scripts/live_demo.py                 (session hour boundaries)

NOTE: Session hour ranges here match scripts/live_demo.py:get_session_info()
exactly.  The prompt specification used different ranges (0-7, 7-8, 8-12,
12-17, 17-20, 20-24); the authoritative source file has (0-6, 6-8, 8-13,
13-16, 16-21, 21-24).  Source-file values are used here.
"""
from __future__ import annotations

from smc.instruments.registry import SYMBOL_REGISTRY
from smc.instruments.types import InstrumentConfig

XAUUSD_CONFIG = InstrumentConfig(
    symbol="XAUUSD",
    mt5_path="XAUUSD",
    magic=19760418,
    # constants.py:13 — XAUUSD_POINT_SIZE = 0.01
    point_size=0.01,
    contract_size=100.0,
    leverage_ratio=100,
    min_lot=0.01,
    # range_trader.py:51 — _DONCHIAN_LOOKBACK_BARS = 48
    donchian_lookback=48,
    # range_trader.py:319 — min_range_width default 200.0
    min_range_width_points=200.0,
    min_range_width_pct=None,
    # range_trader.py:324 — max_range_width default 20000.0
    max_range_width_points=20000.0,
    max_range_width_pct=None,
    # range_trader.py:73 — _BOUNDARY_PCT_DEFAULT = 0.15
    boundary_pct_default=0.15,
    # range_trader.py:75 — _BOUNDARY_PCT_WIDE = 0.30
    boundary_pct_wide=0.30,
    # range_trader.py:62 — _GUARD_WIDTH_MIN_ASIAN = 400.0 (low / Asian)
    guard_width_low=400.0,
    # range_trader.py:60 — _GUARD_WIDTH_MIN_DEFAULT = 800.0 (high / default)
    guard_width_high=800.0,
    # range_trader.py:63 — _GUARD_DURATION_MIN_ASIAN = 8 (low / Asian)
    guard_duration_low=8,
    # range_trader.py:61 — _GUARD_DURATION_MIN_DEFAULT = 12 (high / default)
    guard_duration_high=12,
    # range_trader.py:204 — rr_pass = setup.rr_ratio >= 1.2
    guard_rr_min=1.2,
    # regime.py:27 — _TRENDING_THRESHOLD = 1.4
    regime_trending_pct=1.4,
    # regime.py:28 — _RANGING_THRESHOLD = 1.0
    regime_ranging_pct=1.0,
    # config.py:182 — sl_atr_multiplier default 0.75
    sl_atr_multiplier=0.75,
    # config.py:188 — sl_min_buffer_points default 200.0
    sl_min_buffer_points=200.0,
    sl_min_buffer_pct=None,
    # entry_trigger.py:37,38 — _TP1_RR_RATIO / _TP2_RR_RATIO
    tp1_rr_ratio=2.5,
    tp2_rr_ratio=4.0,
    # live_demo.py:229-245 — get_session_info() hour boundaries
    sessions={
        "ASIAN_CORE": (0, 6),
        "ASIAN_LONDON_TRANSITION": (6, 8),
        "LONDON": (8, 13),
        "LONDON/NY OVERLAP": (13, 16),
        "NEW YORK": (16, 21),
        "LATE NY": (21, 24),
    },
    ranging_sessions=frozenset({
        "ASIAN_CORE",
        "ASIAN_LONDON_TRANSITION",
        "LONDON",
        "LONDON/NY OVERLAP",
        "NEW YORK",
        "LATE NY",
    }),
    # range_trader.py:57-59 — _ASIAN_SESSIONS
    asian_sessions=frozenset({"ASIAN_CORE", "ASIAN_LONDON_TRANSITION"}),
    # mode_router.py:69,93 — literal "ASIAN_CORE" gate for Priority 1/2
    asian_core_session_name="ASIAN_CORE",
    # range_trader.py:77-86 — _WIDE_BAND_SESSIONS
    wide_band_sessions=frozenset({
        "ASIAN_CORE",
        "ASIAN_LONDON_TRANSITION",
        "LONDON",
        "LONDON/NY OVERLAP",
        "NEW YORK",
        "LATE NY",
    }),
    weekend_flag_active=False,
    use_asian_quota=True,
    consec_loss_limit=3,
)

SYMBOL_REGISTRY["XAUUSD"] = XAUUSD_CONFIG
