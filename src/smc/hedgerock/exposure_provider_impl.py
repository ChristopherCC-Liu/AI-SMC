"""Production :class:`ExposureProvider` wrapping :class:`BrokerPort`.

Phase 5.2 Stage C — minimal adapter that turns the existing
``smc.execution.executor.BrokerPort`` (which the rest of AI-SMC
already plumbs to MT5 / sim brokers) into the signed-lots-per-symbol
shape that ``decision_server.ExposureProvider`` expects.

The Phase 1 ``decision_server`` happily ran with ``current_exposure_lots=0.0``
hard-coded; this lets us flip on broker-truth in production without
touching the decision_server contract.
"""

from __future__ import annotations

from collections.abc import Iterable

from smc.execution.executor import BrokerPort
from smc.execution.types import PositionState

__all__ = ["MT5BrokerExposureProvider", "signed_lots_for_symbol"]


def signed_lots_for_symbol(
    positions: Iterable[PositionState],
    symbol: str,
) -> float:
    """Sum signed lots across positions matching ``symbol``.

    ``PositionState.direction`` is ``"long"`` | ``"short"`` and
    ``lots`` is unsigned, so we apply the sign here. Symbol matching
    is **case-sensitive equality** — broker reports MT5 instrument
    name verbatim (``"XAUUSD"`` not ``"xauusd"``).
    """
    total = 0.0
    for p in positions:
        if p.instrument != symbol:
            continue
        sign = 1.0 if p.direction == "long" else -1.0
        total += sign * p.lots
    return total


class MT5BrokerExposureProvider:
    """``ExposureProvider`` impl that queries a :class:`BrokerPort`.

    The Protocol-based decision_server expects ``get_exposure_lots``;
    we satisfy that by querying ``broker.get_positions()`` (which is
    sync — both ``MT5BrokerPort`` and ``SimBrokerPort`` implement it
    synchronously) and summing signed lots.

    Tests inject a stub ``BrokerPort`` that returns canned
    ``PositionState`` tuples; production wiring uses the real
    ``MT5BrokerPort`` already configured by ``smc.execution.executor``.
    """

    def __init__(self, broker: BrokerPort) -> None:
        self._broker = broker

    def get_exposure_lots(self, symbol: str) -> float:
        positions = self._broker.get_positions()
        return signed_lots_for_symbol(positions, symbol)
