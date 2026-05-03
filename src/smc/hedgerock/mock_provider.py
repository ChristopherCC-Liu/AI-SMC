"""Mock `MarketFeaturesProvider` — Phase 1 PoC, no real data.

In Phase 2 a `LakeBackedProvider` will replace this; it pulls ATR,
swing counts, and trend bars from ForexDataLake + regime_classifier.

For Phase 1 we only need *something* deterministic that exercises the
HTTP/cache pipeline end to end. Two flavours:

- `StaticMockProvider`: returns the same features on every call. Good
  for smoke tests where you want to verify the EA polls correctly.
- `ScriptedMockProvider`: cycles through a list of features. Good
  for testing transition_lock behaviour over multiple polls.
"""

from __future__ import annotations

from collections.abc import Sequence
from threading import Lock

from smc.ai.models import MarketRegimeAI
from smc.hedgerock.decision_server import (
    FeaturesUnavailable,
    MarketFeatures,
)
from smc.hedgerock.regime_filters import FilterInputs

__all__ = [
    "ScriptedMockProvider",
    "StaticFilterInputsProvider",
    "StaticMockProvider",
    "default_safe_filter_inputs",
    "default_static_features",
]


def default_static_features(regime: MarketRegimeAI = "TREND_UP") -> MarketFeatures:
    """A reasonable mid-volatility, trending fixture used by smoke tests."""
    return MarketFeatures(
        volatility_rank=0.5,
        hh_count=8,
        ll_count=0,
        h4_trend_bars=5,
        regime=regime,
    )


class StaticMockProvider:
    """Always returns the same `MarketFeatures` regardless of symbol."""

    def __init__(self, features: MarketFeatures | None = None) -> None:
        self.features = features or default_static_features()

    def get_features(self, symbol: str) -> MarketFeatures:
        return self.features


class ScriptedMockProvider:
    """Cycles through a fixed sequence of features, one per call.

    Wraps around to the first entry once exhausted, so long-running
    polling doesn't run out.
    """

    def __init__(self, sequence: Sequence[MarketFeatures]) -> None:
        if not sequence:
            raise ValueError("ScriptedMockProvider needs at least one feature snapshot")
        self._sequence: tuple[MarketFeatures, ...] = tuple(sequence)
        self._index = 0
        self._lock = Lock()

    def get_features(self, symbol: str) -> MarketFeatures:
        with self._lock:
            if self._index >= len(self._sequence):
                # Wrap rather than crash — operator can extend the script
                # later without redeploying the server.
                self._index = 0
            feat = self._sequence[self._index]
            self._index += 1
        return feat

    def remaining(self) -> int:
        with self._lock:
            return len(self._sequence) - self._index


# ---------------------------------------------------------------------------
# FilterInputs mocks
# ---------------------------------------------------------------------------


def default_safe_filter_inputs() -> FilterInputs:
    """A FilterInputs snapshot calibrated to never trigger any filter.

    Phase 1 placeholder for the production ``serve_decision_8788.py``
    until the ForexDataLake-backed provider lands. Values land all three
    diagnostic ratios safely below their thresholds:

    - H4 trend: 0%       (threshold 2%)
    - H1 ATR ratio: 1.0  (threshold 1.5×)
    - Range ratio: 0.5   (threshold 0.6)
    """
    return FilterInputs(
        h4_close_now=100.0,
        h4_close_lookback=100.0,
        h1_atr_now=0.10,
        h1_atr_lookback_avg=0.10,
        h1_recent_high=100.25,
        h1_recent_low=99.75,
        h1_reference_high=100.50,
        h1_reference_low=99.50,
    )


class StaticFilterInputsProvider:
    """Always returns a safe (= no halt) ``FilterInputs`` regardless of symbol.

    Phase 1 placeholder for the real ForexDataLake-backed provider.
    Useful for:

    - Smoke tests where you want the EA's ``transition_lock_until_ts``
      to be driven by regime-transition logic alone.
    - Production startup against a cold lake — swap in the real
      provider once the lake is warm.

    Pass a custom ``inputs`` to deliberately exercise the halt path
    (e.g., set ``h4_close_now`` 5% above ``h4_close_lookback``).
    """

    def __init__(self, inputs: FilterInputs | None = None) -> None:
        self.inputs = inputs or default_safe_filter_inputs()

    def get_filter_inputs(self, symbol: str) -> FilterInputs:
        return self.inputs


__all__ += ["FeaturesUnavailable"]  # re-export for convenience
