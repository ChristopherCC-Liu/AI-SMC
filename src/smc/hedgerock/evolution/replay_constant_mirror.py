"""Ticket 2 Step 4 — Class B mirror: phase_d_walk_forward replay constants.

**Read-only mirror, separate from Class A.** Plan R3 calls for
splitting "rule_engine decision constants" (Class A) from
"phase_d_walk_forward replay constants" (Class B) so candidates that
patch the replay layer are routed through their own mirror with its
own drift detection.

In v1 the production ``phase_d_walk_forward`` module does NOT
expose any module-level scalar that candidates need to patch — the
halt-expiry knobs are runtime fields on ``ExperimentConfig``, not
module constants. Therefore Class B's whitelist is empty in v1;
candidates whose target points here (e.g. c2's hypothetical
``_HALT_AUTO_EXPIRY_HOURS_OBSERVE``) fall to ``ABSTAIN:
unsupported_target`` from the runner.

The module exists so Step 5 can route candidate targets through a
single ``is_target_in_whitelist`` check that consults both mirrors
and so future production work that exposes a scalar replay knob
can extend Class B without touching Class A.

Same hard rules as Class A:
  - read-only (no setattr / monkeypatch / module-level mutation),
  - drift detection via ``check_mirror_drift()``,
  - deterministic ``compute_mirror_version()``.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from typing import Any


__all__ = [
    "CLASS_B_TARGET_WHITELIST",
    "check_mirror_drift",
    "compute_mirror_version",
    "is_target_in_whitelist",
    "snapshot_params",
]


# v1: empty. Adding entries requires:
#   1. Production phase_d_walk_forward exposes the constant as a
#      module-level scalar.
#   2. Mirror baseline + drift test in this file.
#   3. Candidate menu entry references the new target.
# All three steps must land together; otherwise the candidate falls
# to ABSTAIN at runtime.
_CLASS_B_BASELINES: dict[str, tuple[type, Any]] = {}

CLASS_B_TARGET_WHITELIST: tuple[str, ...] = tuple(_CLASS_B_BASELINES.keys())


def is_target_in_whitelist(target: str) -> bool:
    return target in _CLASS_B_BASELINES


def _split_module_attr(target: str) -> tuple[str, str]:
    parts = target.rsplit(".", 1)
    if len(parts) != 2:
        raise ValueError(f"target must be a dotted path: {target!r}")
    return parts[0], parts[1]


def _read_production_value(target: str) -> tuple[bool, Any, type | None]:
    module_path, attr = _split_module_attr(target)
    module = importlib.import_module(module_path)
    if not hasattr(module, attr):
        return False, None, None
    val = getattr(module, attr)
    return True, val, type(val)


def snapshot_params() -> dict[str, Any]:
    """Empty dict in v1 (whitelist is empty)."""
    out: dict[str, Any] = {}
    for target in CLASS_B_TARGET_WHITELIST:
        found, val, _ = _read_production_value(target)
        if found:
            out[target] = val
    return out


def check_mirror_drift() -> tuple[bool, list[str]]:
    """v1 returns (True, []) since the whitelist is empty. As future
    targets are added, drift detection follows Class A's pattern."""
    reasons: list[str] = []
    for target, (expected_type, expected_value) in _CLASS_B_BASELINES.items():
        found, val, _ = _read_production_value(target)
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
    payload = {
        "class": "replay_constant_mirror_class_B",
        "baselines": [
            {
                "target": target,
                "type": type_.__name__,
                "value": value,
            }
            for target, (type_, value) in sorted(_CLASS_B_BASELINES.items())
        ],
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()
