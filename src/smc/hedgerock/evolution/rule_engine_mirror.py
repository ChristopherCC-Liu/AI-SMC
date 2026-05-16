"""Ticket 2 Step 4 — Class A mirror: rule_engine module-level constants.

**Read-only mirror.** The sidecar uses this module to:

  1. Discover which production rule_engine constants are exposed
     for shadow-comparison patching (the whitelist).
  2. Read live values from production at runtime
     (:func:`snapshot_params`) without ever mutating them.
  3. Detect drift between recorded baselines and live production
     (:func:`check_mirror_drift`) — at sidecar startup AND at every
     shadow run, fail-closed per R3.

**No mutation.** This module reads ``smc.hedgerock.rule_engine`` via
``importlib.import_module`` + ``getattr``; it never calls
``setattr``, never monkeypatches, never reassigns module attributes.

**Whitelist scope.** Only constants that are *actually present in
production rule_engine as module-level scalars* may join the
whitelist. Hypothetical knobs (e.g. c4's
``_RANGE_2_CONFIDENCE_BASELINE``, which is hardcoded inline in
production rather than exposed as a constant) are deliberately
absent → callers receive ``ABSTAIN: unsupported_target`` from the
runner downstream.

Adding a new whitelist entry is a code change with full review:
the new entry must be paired with mirror_drift_test coverage, the
production constant must exist with the recorded type/value, and a
new RFC ticket can introduce its own candidate menu entry.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from typing import Any


__all__ = [
    "CLASS_A_TARGET_WHITELIST",
    "MirrorDriftError",
    "check_mirror_drift",
    "compute_mirror_version",
    "is_target_in_whitelist",
    "snapshot_params",
]


class MirrorDriftError(RuntimeError):
    """Raised when production values diverge from recorded baselines.

    The sidecar must NOT proceed to produce PASS/FAIL artefacts when
    drift is detected. Acceptable responses:
      - abort the shadow run before any artefact is written, OR
      - emit an artefact with verdict ABSTAIN +
        mirror_consistency_check=FAIL.
    """


# ---------------------------------------------------------------------------
# Whitelist with recorded baselines
#
# Layout: target dotted path → (expected_type, expected_value)
# ---------------------------------------------------------------------------


_CLASS_A_BASELINES: dict[str, tuple[type, Any]] = {
    "smc.hedgerock.rule_engine._CONFIDENCE_OBSERVE_FLOOR": (float, 0.55),
    "smc.hedgerock.rule_engine._CONFIDENCE_AGGRESSIVE_FLOOR": (float, 0.8),
}

CLASS_A_TARGET_WHITELIST: tuple[str, ...] = tuple(_CLASS_A_BASELINES.keys())


def is_target_in_whitelist(target: str) -> bool:
    return target in _CLASS_A_BASELINES


# ---------------------------------------------------------------------------
# Reading production state (read-only)
# ---------------------------------------------------------------------------


def _split_module_attr(target: str) -> tuple[str, str]:
    """Split 'a.b.c.NAME' → ('a.b.c', 'NAME')."""
    parts = target.rsplit(".", 1)
    if len(parts) != 2:
        raise ValueError(f"target must be a dotted path: {target!r}")
    return parts[0], parts[1]


def _read_production_value(target: str) -> tuple[bool, Any, type | None]:
    """Read live production value for ``target``. Returns
    ``(found, value, type_)``. ``found=False`` when the module has
    no such attribute (drift = removed)."""
    module_path, attr = _split_module_attr(target)
    module = importlib.import_module(module_path)
    if not hasattr(module, attr):
        return False, None, None
    val = getattr(module, attr)
    return True, val, type(val)


def snapshot_params() -> dict[str, Any]:
    """Return a flat dict ``{target: live_production_value}`` for
    every whitelisted target. This is the input to
    :func:`policy_overlay.apply_overlay`. **Read-only**: no
    setattr, no monkeypatch, no global mutation."""
    out: dict[str, Any] = {}
    for target in CLASS_A_TARGET_WHITELIST:
        found, val, _ = _read_production_value(target)
        if not found:
            # Drift mid-flight; surfaced by check_mirror_drift().
            continue
        out[target] = val
    return out


# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------


def check_mirror_drift() -> tuple[bool, list[str]]:
    """Compare every whitelisted target's live production value
    against the recorded baseline. Returns (is_consistent, reasons).

    Drift modes detected:
      - production attribute removed
      - production value differs (any equality difference)
      - production value type differs
    """
    reasons: list[str] = []
    for target, (expected_type, expected_value) in _CLASS_A_BASELINES.items():
        found, val, actual_type = _read_production_value(target)
        if not found:
            reasons.append(
                f"{target}: production attribute missing (mirror has "
                f"baseline {expected_value!r}, type {expected_type.__name__})"
            )
            continue
        if not isinstance(val, expected_type):
            reasons.append(
                f"{target}: type drift — expected {expected_type.__name__}, "
                f"got {type(val).__name__} (value {val!r})"
            )
            continue
        if val != expected_value:
            reasons.append(
                f"{target}: value drift — mirror baseline "
                f"{expected_value!r}, production {val!r}"
            )
    return (len(reasons) == 0, reasons)


def compute_mirror_version() -> str:
    """Deterministic SHA-256 of the recorded whitelist + baselines.
    Embedded in every shadow artefact's ``mirror_version`` field so
    G8 can detect "artefact produced under mirror v1, evaluated by
    mirror v2"."""
    payload = {
        "class": "rule_engine_mirror_class_A",
        "baselines": [
            {
                "target": target,
                "type": type_.__name__,
                "value": value,
            }
            for target, (type_, value) in sorted(_CLASS_A_BASELINES.items())
        ],
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()
