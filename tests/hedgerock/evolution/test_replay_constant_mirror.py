"""Ticket 2 Step 4 — Class B mirror tests (replay-side constants).

Class B handles candidate targets that live in
``phase_d_walk_forward`` (replay constants), not in
``rule_engine`` (decision constants). Per Ticket 2 plan §R3 c2's
target conceptually belongs here.

The current production phase_d_walk_forward module does NOT expose
``_HALT_AUTO_EXPIRY_HOURS_OBSERVE`` as a module-level constant
(halt-expiry hours are passed via ExperimentConfig at run time).
Therefore Class B's whitelist starts empty in v1; c2 lands in
"unsupported_target" → G8 ABSTAIN at runtime, exactly per the R4
expectation.

Tests pinned:
  - Whitelist exists and is a tuple (possibly empty in v1).
  - is_target_in_whitelist returns False for every Class A target
    (separation of concerns).
  - is_target_in_whitelist returns False for c2's hypothetical target
    in v1.
  - check_mirror_drift on an empty whitelist returns clean PASS.
  - The module is read-only (does not mutate phase_d_walk_forward).
"""

from __future__ import annotations

import importlib

import pytest

from smc.hedgerock.evolution.replay_constant_mirror import (
    CLASS_B_TARGET_WHITELIST,
    check_mirror_drift,
    compute_mirror_version,
    is_target_in_whitelist,
    snapshot_params,
)


@pytest.mark.unit
def test_whitelist_is_a_tuple() -> None:
    assert isinstance(CLASS_B_TARGET_WHITELIST, tuple)


@pytest.mark.unit
def test_whitelist_v1_is_empty() -> None:
    """v1: phase_d_walk_forward exposes no module-level scalar that
    candidates need to patch. The c2 candidate target
    ``_HALT_AUTO_EXPIRY_HOURS_OBSERVE`` is hypothetical (the real
    knob is an ExperimentConfig field, not a module constant), so
    no Class B target makes it into the whitelist in v1."""
    assert CLASS_B_TARGET_WHITELIST == ()


@pytest.mark.unit
def test_class_a_targets_are_not_in_class_b() -> None:
    """Class A and Class B whitelists must be disjoint."""
    assert not is_target_in_whitelist(
        "smc.hedgerock.rule_engine._CONFIDENCE_OBSERVE_FLOOR"
    )
    assert not is_target_in_whitelist(
        "smc.hedgerock.rule_engine._CONFIDENCE_AGGRESSIVE_FLOOR"
    )


@pytest.mark.unit
def test_c2_hypothetical_target_not_in_v1_whitelist() -> None:
    """c2 candidate target not present → unsupported_target path."""
    assert not is_target_in_whitelist(
        "smc.hedgerock.phase_d_walk_forward._HALT_AUTO_EXPIRY_HOURS_OBSERVE"
    )


@pytest.mark.unit
def test_check_mirror_drift_clean_with_empty_whitelist() -> None:
    is_consistent, reasons = check_mirror_drift()
    assert is_consistent is True
    assert reasons == []


@pytest.mark.unit
def test_snapshot_params_is_empty_with_empty_whitelist() -> None:
    assert snapshot_params() == {}


@pytest.mark.unit
def test_compute_mirror_version_deterministic() -> None:
    a = compute_mirror_version()
    b = compute_mirror_version()
    assert a == b
    assert isinstance(a, str)
    assert len(a) == 64


@pytest.mark.unit
def test_class_b_mirror_does_not_mutate_phase_d_walk_forward() -> None:
    pdwf = importlib.import_module("smc.hedgerock.phase_d_walk_forward")
    # Capture every module-level scalar / tuple before mirror import.
    before = {
        name: getattr(pdwf, name)
        for name in dir(pdwf)
        if not name.startswith("__")
        and isinstance(getattr(pdwf, name, None), (int, float, str, tuple))
    }
    importlib.import_module("smc.hedgerock.evolution.replay_constant_mirror")
    after = {
        name: getattr(pdwf, name)
        for name in dir(pdwf)
        if not name.startswith("__")
        and isinstance(getattr(pdwf, name, None), (int, float, str, tuple))
    }
    assert before == after
