"""Fusion layer — wires every existing AI-SMC component into a single
end-to-end decision pipeline.

The fusion package is purely orchestration: it owns no detection logic
of its own, only adapters + a scorer + a controller that composes the
existing perception/decision/validation/execution/evolution modules.

Public re-exports kept narrow on purpose so external callers (e.g.
``decision_server.create_app``) depend only on :class:`FusionController`
and the data-class contracts.
"""

from __future__ import annotations

from smc.fusion.contracts import (
    FusedDirection,
    FusionConfig,
    FusionOutcome,
    FusionTrace,
)
from smc.fusion.fusion_controller import FusionController

__all__ = [
    "FusedDirection",
    "FusionConfig",
    "FusionController",
    "FusionOutcome",
    "FusionTrace",
]
