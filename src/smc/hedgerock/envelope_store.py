"""Per-symbol latest :class:`SignalEnvelope` cache.

Phase C-hotfix #1: cooldown_until must persist across /signal calls.
The rule_engine receives ``prev_envelope`` and uses
``max(prev.cooldown_until, newly_requested_cooldown)`` so a transient
spike (e.g. spread spike) that triggered cooldown N seconds ago is not
wiped by the very next poll where conditions look fine.

Mirrors the :class:`EAStateStore` shape — symbol case-insensitive,
thread-safe, decoupled snapshot.
"""

from __future__ import annotations

from threading import Lock

from smc.hedgerock.schemas import SignalEnvelope

__all__ = ["EnvelopeStore"]


class EnvelopeStore:
    """Thread-safe cache of the most recent :class:`SignalEnvelope` per symbol.

    The decision_server is single-process so an in-memory dict
    suffices. /signal writes after each successful build; rule_engine
    reads before deriving params on the next call.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._cache: dict[str, SignalEnvelope] = {}

    @staticmethod
    def _canon(symbol: str) -> str:
        return symbol.upper()

    def get(self, symbol: str) -> SignalEnvelope | None:
        with self._lock:
            return self._cache.get(self._canon(symbol))

    def set(self, symbol: str, envelope: SignalEnvelope) -> None:
        with self._lock:
            self._cache[self._canon(symbol)] = envelope

    def snapshot(self) -> dict[str, SignalEnvelope]:
        with self._lock:
            return dict(self._cache)
