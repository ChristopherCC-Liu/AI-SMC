"""Instrument configs (XAU/BTC/...) + registry."""
from smc.instruments.types import InstrumentConfig
from smc.instruments.registry import SYMBOL_REGISTRY, get_instrument_config
# Import symbol modules for side-effect registry population
from smc.instruments import xauusd as _xauusd  # noqa: F401
from smc.instruments import btcusd as _btcusd  # noqa: F401

__all__ = ["InstrumentConfig", "SYMBOL_REGISTRY", "get_instrument_config"]
