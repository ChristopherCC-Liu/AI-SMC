"""Decision server — public read-only parameter snapshot.

The evolution layer is allowed to import this module to read the
current live parameter values. The contract is **read-only**:

  * :func:`get_live_parameters` — return a defensive copy of the
    live parameter mapping.
  * :data:`LIVE_PARAMETER_KEYS` — the canonical set of parameter
    class ids that this module exposes.

There is intentionally no setter, no writer, and no method that
mutates the live parameters from outside this module. Mutating
production parameters at runtime is reserved for the human
promotion flow, which lives outside this file.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping


__all__ = [
    "LIVE_PARAMETER_KEYS",
    "get_live_parameters",
]


# Live values — kept in sync with the canonical manifest baselines
# in CANDIDATE_MENU_V0. When the human promotion flow updates a
# parameter, both this snapshot and the manifest move together.
_CURRENT_LIVE_PARAMETERS: Mapping[str, float] = MappingProxyType(
    {
        "confidence_threshold_observe": 0.55,
        "confidence_threshold_aggressive": 0.80,
        "confidence_threshold_range_2": 0.65,
        "halt_expiry_observe_hours": 4.0,
    }
)


LIVE_PARAMETER_KEYS: frozenset[str] = frozenset(_CURRENT_LIVE_PARAMETERS.keys())


def get_live_parameters() -> dict[str, float]:
    """Return a fresh dict copy of the live parameter snapshot.

    The returned object is safe to mutate locally; mutation does not
    propagate back to the module-level constant.
    """
    return dict(_CURRENT_LIVE_PARAMETERS)
