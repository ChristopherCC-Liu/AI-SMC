"""SMC Core detection layer for the AI-SMC trading system.

Public re-exports for convenient imports:

    from smc.smc_core import SMCDetector, SMCSnapshot, OrderBlock, ...
"""

from __future__ import annotations

from smc.smc_core.constants import XAUUSD_PIP_SIZE, XAUUSD_POINT_SIZE
from smc.smc_core.detector import SMCDetector
from smc.smc_core.fvg import detect_fvgs, update_fill_status
from smc.smc_core.liquidity import detect_liquidity_levels, detect_liquidity_sweep
from smc.smc_core.order_block import detect_order_blocks, update_mitigation
from smc.smc_core.structure import current_trend, detect_structure
from smc.smc_core.swing import detect_swings, filter_significant_swings
from smc.smc_core.types import (
    FairValueGap,
    LiquidityLevel,
    OrderBlock,
    SMCSnapshot,
    StructureBreak,
    SwingPoint,
    Timeframe,
)

__all__ = [
    # Constants
    "XAUUSD_POINT_SIZE",
    "XAUUSD_PIP_SIZE",
    # Detector
    "SMCDetector",
    # Types
    "FairValueGap",
    "LiquidityLevel",
    "OrderBlock",
    "SMCSnapshot",
    "StructureBreak",
    "SwingPoint",
    "Timeframe",
    # Sub-detector functions
    "detect_swings",
    "filter_significant_swings",
    "detect_order_blocks",
    "update_mitigation",
    "detect_fvgs",
    "update_fill_status",
    "detect_structure",
    "current_trend",
    "detect_liquidity_levels",
    "detect_liquidity_sweep",
]
