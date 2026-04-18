"""Symbol registry — maps symbol strings to InstrumentConfig instances."""
from __future__ import annotations

from smc.instruments.types import InstrumentConfig

SYMBOL_REGISTRY: dict[str, InstrumentConfig] = {}


def get_instrument_config(symbol: str) -> InstrumentConfig:
    if symbol not in SYMBOL_REGISTRY:
        raise KeyError(
            f"Unknown symbol: {symbol}. Registered: {list(SYMBOL_REGISTRY)}"
        )
    return SYMBOL_REGISTRY[symbol]
