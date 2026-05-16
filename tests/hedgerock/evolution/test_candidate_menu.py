"""Phase D-cont3 / Ticket 1 — candidate_menu tests.

Locks the v0 candidate list (4 entries per Plan §4) so that anyone
silently expanding it to 5+ trips a test rather than a review.
"""

from __future__ import annotations

import pytest

from smc.hedgerock.evolution.candidate_menu import CANDIDATE_MENU_V0
from smc.hedgerock.evolution.policy_manifest import (
    CandidateManifest,
    CandidateState,
)


@pytest.mark.unit
def test_menu_has_exactly_four_entries() -> None:
    """Plan §4: exact count is part of the contract."""
    assert len(CANDIDATE_MENU_V0) == 4


@pytest.mark.unit
def test_menu_entries_are_candidate_manifests_in_draft_state() -> None:
    for c in CANDIDATE_MENU_V0:
        assert isinstance(c, CandidateManifest)
        assert c.state == CandidateState.DRAFT


@pytest.mark.unit
def test_menu_ids_are_unique() -> None:
    ids = [c.candidate_id for c in CANDIDATE_MENU_V0]
    assert len(ids) == len(set(ids))


@pytest.mark.unit
def test_menu_includes_each_named_pattern_id() -> None:
    """Every named pattern from Plan §4 is present."""
    ids = [c.candidate_id for c in CANDIDATE_MENU_V0]
    assert "c1-lower-observe-floor-0.50" in ids
    assert "c2-halt-expiry-observe-6h" in ids
    assert "c3-aggressive-floor-0.78" in ids
    assert "c4-range2-conf-0.70" in ids


@pytest.mark.unit
def test_c2_marks_affects_halt_mode() -> None:
    c2 = next(c for c in CANDIDATE_MENU_V0 if c.candidate_id == "c2-halt-expiry-observe-6h")
    assert c2.diff.scope.affects_halt_mode is True


@pytest.mark.unit
def test_c3_marks_raises_gross_exposure() -> None:
    """c3 is the exposure-class candidate per Plan §4."""
    c3 = next(c for c in CANDIDATE_MENU_V0 if c.candidate_id == "c3-aggressive-floor-0.78")
    assert c3.diff.scope.raises_gross_exposure is True


@pytest.mark.unit
def test_other_candidates_do_not_raise_exposure() -> None:
    for c in CANDIDATE_MENU_V0:
        if c.candidate_id == "c3-aggressive-floor-0.78":
            continue
        assert c.diff.scope.raises_gross_exposure is False
        assert c.diff.scope.raises_leverage is False
        assert c.diff.scope.raises_max_open_positions is False
        assert c.diff.scope.raises_max_recovery_multiplier is False
        assert c.diff.scope.raises_max_grid_density is False


@pytest.mark.unit
def test_menu_proposed_values_are_numeric_or_none_only() -> None:
    """No free-form code: every diff target must be a typed numeric
    knob or a flag, never a code path."""
    for c in CANDIDATE_MENU_V0:
        v = c.diff.proposed_value
        assert v is None or isinstance(v, (int, float, str)), (
            f"{c.candidate_id}: proposed_value must be primitive, got {type(v)}"
        )
