"""Position sizing for XAUUSD trades.

Calculates lot size so that a stop-loss hit risks exactly ``risk_pct`` of
the account balance.

XAUUSD contract mechanics
-------------------------
- 1 standard lot = 100 troy oz
- 1 pip  = $0.10 price move = $10  per pip per standard lot
- 1 point = $0.01 price move = $1.00 per point per standard lot
- Typical margin: ~$2 000 per lot at 1:50 leverage

Formula::

    risk_usd = balance_usd * (risk_pct / 100)
    point_value_per_lot = pip_value_per_lot / 10
    lots = risk_usd / (sl_distance_points * point_value_per_lot)
    lots = clamp(lots, min_lot_size, max_lot_size)
"""

from __future__ import annotations

from smc.risk.types import PositionSize

# XAUUSD defaults
_DEFAULT_PIP_VALUE_PER_LOT: float = 10.0  # $10 per pip per standard lot
_DEFAULT_MAX_LOT: float = 1.0
_DEFAULT_MIN_LOT: float = 0.01
_DEFAULT_MARGIN_PER_LOT: float = 2_000.0  # ~$2 000 at 1:50 leverage


def compute_position_size(
    *,
    balance_usd: float,
    risk_pct: float,
    sl_distance_points: float,
    pip_value_per_lot: float = _DEFAULT_PIP_VALUE_PER_LOT,
    max_lot_size: float = _DEFAULT_MAX_LOT,
    min_lot_size: float = _DEFAULT_MIN_LOT,
    margin_per_lot: float = _DEFAULT_MARGIN_PER_LOT,
) -> PositionSize:
    """Calculate lot size to risk exactly *risk_pct* of *balance_usd*.

    Parameters
    ----------
    balance_usd:
        Current account balance in USD.
    risk_pct:
        Percentage of balance to risk (e.g. 1.0 = 1 %).
    sl_distance_points:
        Stop-loss distance in **points** (1 point = $0.01 price move).
    pip_value_per_lot:
        Dollar value of a single pip move per standard lot.
        Default 10.0 for XAUUSD.
    max_lot_size:
        Hard cap on lots per order.
    min_lot_size:
        Minimum tradeable lot size (broker constraint).
    margin_per_lot:
        Estimated margin requirement per standard lot.

    Returns
    -------
    PositionSize
        Immutable result with ``lots`` clamped to [min_lot_size, max_lot_size].

    Raises
    ------
    ValueError
        If any numeric input is non-positive.
    """
    if balance_usd <= 0:
        raise ValueError(f"balance_usd must be positive, got {balance_usd}")
    if risk_pct <= 0:
        raise ValueError(f"risk_pct must be positive, got {risk_pct}")
    if sl_distance_points <= 0:
        raise ValueError(f"sl_distance_points must be positive, got {sl_distance_points}")
    if pip_value_per_lot <= 0:
        raise ValueError(f"pip_value_per_lot must be positive, got {pip_value_per_lot}")

    risk_usd = balance_usd * (risk_pct / 100.0)
    point_value_per_lot = pip_value_per_lot / 10.0  # $1.00 for XAUUSD

    raw_lots = risk_usd / (sl_distance_points * point_value_per_lot)
    clamped_lots = max(min_lot_size, min(raw_lots, max_lot_size))
    # Round to 2 decimal places (standard broker precision)
    clamped_lots = round(clamped_lots, 2)

    margin_required = clamped_lots * margin_per_lot

    return PositionSize(
        lots=clamped_lots,
        risk_usd=risk_usd,
        risk_points=sl_distance_points,
        margin_required_usd=margin_required,
    )
