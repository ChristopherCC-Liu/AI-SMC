"""XAUUSD instrument constants — single source of truth.

MT5 terminology for XAUUSD (5-digit pricing):
  - 1 point = $0.01 (minimum price increment)
  - 1 pip = 10 points = $0.10 (standard trading unit)

This codebase uses POINTS (not pips) as the base unit for all
price distance calculations. This matches MT5's internal representation.
"""
from __future__ import annotations

# XAUUSD 5-digit pricing: 1 point = $0.01
XAUUSD_POINT_SIZE: float = 0.01

# For reference: 1 pip = 10 points = $0.10
XAUUSD_PIP_SIZE: float = 0.10
