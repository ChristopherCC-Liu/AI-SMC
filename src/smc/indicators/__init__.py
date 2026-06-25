"""Deterministic, no-look-ahead indicator engine.

``core`` holds the primitive indicators (moving averages, ATR, RSI, ADX,
Donchian, VWAP, NRB, fractals, MA-slope cascade) shared by the Python
backtester/engine and the MQL5 EA.  Composite proprietary indicators and the
IC validation harness build on these (added in later phases).
"""

from __future__ import annotations

from smc.indicators import core

__all__ = ["core"]
