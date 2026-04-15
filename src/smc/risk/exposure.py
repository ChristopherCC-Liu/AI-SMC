"""Exposure and margin checks for live trading.

Stateless utility functions that verify margin adequacy and position
exposure before placing a new order.

XAUUSD margin mechanics
-----------------------
- Margin level = (equity / margin_used) * 100
- Margin level 200 % = safe; below 100 % triggers margin call
- We use 200 % as the conservative floor to leave headroom
"""

from __future__ import annotations


def check_margin(
    balance: float,
    margin_used: float,
    margin_level_min: float = 200.0,
) -> bool:
    """Return True if the margin level is above the minimum threshold.

    Parameters
    ----------
    balance:
        Current account equity in USD.
    margin_used:
        Total margin currently consumed by open positions.
    margin_level_min:
        Minimum acceptable margin level as a percentage (default 200 %).

    Returns
    -------
    bool
        True if margin level >= margin_level_min, or if margin_used is zero.
    """
    if margin_used <= 0:
        # No margin consumed — always safe
        return True
    if balance <= 0:
        return False

    margin_level = (balance / margin_used) * 100.0
    return margin_level >= margin_level_min


def check_max_exposure(
    open_lots: float,
    new_lots: float,
    max_total_lots: float = 1.0,
) -> bool:
    """Return True if adding *new_lots* stays within the maximum exposure.

    Parameters
    ----------
    open_lots:
        Total lots currently held across all open positions.
    new_lots:
        Lot size of the proposed new order.
    max_total_lots:
        Hard ceiling on aggregate lot exposure.

    Returns
    -------
    bool
        True if ``open_lots + new_lots <= max_total_lots``.
    """
    return (open_lots + new_lots) <= max_total_lots
