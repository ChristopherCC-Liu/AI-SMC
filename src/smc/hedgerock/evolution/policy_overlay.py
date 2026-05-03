"""Ticket 2 Step 3 — policy overlay (candidate parameter patch).

**Pure dataclasses + functional dict transform.** No imports from
production runtime; no setattr / monkeypatch / global mutation.

Design (per Ticket 2 plan §4):

  - :class:`PolicyOverlay` is a frozen description of one
    parameter substitution: target dotted path + proposed value +
    baseline value.
  - :func:`apply_overlay` takes a **mapping snapshot** of the
    relevant parameters (built earlier by the runner from the rule
    engine / replay-constant mirrors) and returns a NEW mapping
    with the target value replaced. The input mapping is never
    mutated; the production rule_engine module is never touched.

Why a mapping snapshot rather than the live module:

  1. The snapshot is the only thing the shadow runner reads or
     writes. Production module state is opaque to this module by
     design.
  2. Tests can verify byte-equality of the production module's
     constants before and after :func:`apply_overlay` to confirm
     no global pollution.
  3. The mirror files (Step 4) own the snapshot construction; this
     module only knows how to apply one overlay to one snapshot.

Hard rules (encoded by static AST tests):

  - No ``setattr`` / ``delattr`` / ``exec`` / ``eval`` / ``compile``.
  - No ``monkeypatch`` / ``.patch(...)`` / pytest fixture imports.
  - No imports of ``smc.hedgerock.rule_engine``,
    ``smc.hedgerock.decision_server``, or
    ``smc.hedgerock.phase_d_walk_forward``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping


__all__ = [
    "OVERLAY_SCHEMA_VERSION",
    "PolicyOverlay",
    "apply_overlay",
    "compute_overlay_id",
]


OVERLAY_SCHEMA_VERSION: str = "1.0.0"


@dataclass(frozen=True)
class PolicyOverlay:
    """One candidate parameter patch.

    Fields mirror the on-disk ``CandidateDiff`` shape but live in
    sidecar memory only. ``target`` is a dotted path (e.g.
    ``smc.hedgerock.rule_engine._CONFIDENCE_OBSERVE_FLOOR``) used as
    the key into the parameter snapshot dict that the mirror builds.
    """

    candidate_id: str
    target: str
    proposed_value: float | int | str | None
    baseline_value: float | int | str | None


def compute_overlay_id(overlay: PolicyOverlay) -> str:
    """Deterministic SHA-256 of the overlay's content. Used by the
    artefact's ``candidate_overlay_id`` field so the same overlay
    in two different runs hashes to the same id."""
    payload = json.dumps(
        asdict(overlay), indent=2, sort_keys=True, ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def apply_overlay(
    params: Mapping[str, Any],
    overlay: PolicyOverlay,
) -> dict[str, Any]:
    """Return a NEW dict equal to ``params`` with
    ``params[overlay.target]`` replaced by ``overlay.proposed_value``.

    Raises:
        KeyError: when ``overlay.target`` is not in ``params``. A
            missing key indicates the snapshot was built from the
            wrong mirror or the candidate's target isn't in the
            mirror whitelist; the runner is expected to surface this
            as ABSTAIN: unsupported_target.

    The input mapping is never mutated. The production rule_engine
    module is never touched (this module never imports it).
    """
    if overlay.target not in params:
        raise KeyError(
            f"overlay target {overlay.target!r} is not present in the "
            "parameter snapshot — likely outside the mirror whitelist"
        )
    out = dict(params)
    out[overlay.target] = overlay.proposed_value
    return out
