"""Risk management module for the AI-SMC trading system.

Public API
----------
- :class:`PositionSize` / :class:`RiskBudget` — immutable data types
- :func:`compute_position_size` — lot sizing from balance + SL distance
- :class:`DrawdownGuard` — stateless circuit breaker
- :func:`check_margin` / :func:`check_max_exposure` — margin & exposure checks
"""

from smc.risk.drawdown_guard import DrawdownGuard
from smc.risk.exposure import check_margin, check_max_exposure
from smc.risk.position_sizer import compute_position_size
from smc.risk.types import PositionSize, RiskBudget

__all__ = [
    "DrawdownGuard",
    "PositionSize",
    "RiskBudget",
    "check_margin",
    "check_max_exposure",
    "compute_position_size",
]
