"""Ticket 2 Step 4 — Class A mirror tests (rule_engine constants).

Pinned guarantees (per R3 + R6 step 4):
  - The mirror declares an explicit target whitelist; targets
    outside it are NOT supported (callers receive a clear "not in
    whitelist" error).
  - For each whitelisted target, mirror records the expected
    production value + type. At sidecar startup,
    `check_mirror_drift()` compares the recorded values against
    live production values; mismatch → MirrorDriftError.
  - `snapshot_params()` returns a dict {target: production_value}
    keyed by dotted path; this is what apply_overlay() consumes.
  - `compute_mirror_version()` is deterministic and changes when
    the whitelist or expected baselines change.
  - The mirror does NOT mutate production state (only reads).
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from smc.hedgerock.evolution.rule_engine_mirror import (
    CLASS_A_TARGET_WHITELIST,
    MirrorDriftError,
    check_mirror_drift,
    compute_mirror_version,
    is_target_in_whitelist,
    snapshot_params,
)


# ---------------------------------------------------------------------------
# 1. Whitelist semantics
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_whitelist_is_a_tuple_of_dotted_paths() -> None:
    assert isinstance(CLASS_A_TARGET_WHITELIST, tuple)
    for t in CLASS_A_TARGET_WHITELIST:
        assert isinstance(t, str)
        assert t.startswith("smc.hedgerock.rule_engine.")


@pytest.mark.unit
def test_known_real_targets_are_in_whitelist() -> None:
    """c1 and c3 candidate targets must be present (they're the real
    rule_engine module-level constants)."""
    assert (
        "smc.hedgerock.rule_engine._CONFIDENCE_OBSERVE_FLOOR"
        in CLASS_A_TARGET_WHITELIST
    )
    assert (
        "smc.hedgerock.rule_engine._CONFIDENCE_AGGRESSIVE_FLOOR"
        in CLASS_A_TARGET_WHITELIST
    )


@pytest.mark.unit
def test_hypothetical_targets_not_in_whitelist() -> None:
    """c4's target doesn't exist as a production constant — must NOT
    be silently registered. The mirror only registers what really
    exists."""
    assert (
        "smc.hedgerock.rule_engine._RANGE_2_CONFIDENCE_BASELINE"
        not in CLASS_A_TARGET_WHITELIST
    )


@pytest.mark.unit
def test_is_target_in_whitelist_reports_correctly() -> None:
    assert is_target_in_whitelist(
        "smc.hedgerock.rule_engine._CONFIDENCE_OBSERVE_FLOOR"
    ) is True
    assert is_target_in_whitelist(
        "smc.hedgerock.rule_engine._RANGE_2_CONFIDENCE_BASELINE"
    ) is False
    assert is_target_in_whitelist(
        "smc.hedgerock.phase_d_walk_forward._HALT_AUTO_EXPIRY_HOURS_OBSERVE"
    ) is False  # belongs to Class B


# ---------------------------------------------------------------------------
# 2. snapshot_params reads live production values
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_snapshot_returns_keys_for_all_whitelisted_targets() -> None:
    snapshot = snapshot_params()
    assert set(snapshot.keys()) == set(CLASS_A_TARGET_WHITELIST)


@pytest.mark.unit
def test_snapshot_values_match_production_constants() -> None:
    """The snapshot's value for each target must equal the live
    production module's value at the time of the call."""
    rule_engine = importlib.import_module("smc.hedgerock.rule_engine")
    snapshot = snapshot_params()
    for target in CLASS_A_TARGET_WHITELIST:
        attr = target.split(".")[-1]
        assert snapshot[target] == getattr(rule_engine, attr), (
            f"snapshot[{target}] != production value"
        )


# ---------------------------------------------------------------------------
# 3. check_mirror_drift — clean state passes; injected drift fails
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_check_mirror_drift_clean_state_returns_pass() -> None:
    """With production values matching the recorded baselines, the
    drift check returns (True, [])."""
    is_consistent, reasons = check_mirror_drift()
    assert is_consistent is True, f"unexpected drift reasons: {reasons}"
    assert reasons == []


@pytest.mark.unit
def test_check_mirror_drift_detects_value_change(monkeypatch) -> None:
    """If a production value diverges from the recorded baseline,
    drift check fails. We simulate this with monkeypatch (test
    ONLY — never used by sidecar code)."""
    rule_engine = importlib.import_module("smc.hedgerock.rule_engine")
    monkeypatch.setattr(
        rule_engine, "_CONFIDENCE_OBSERVE_FLOOR", 0.99, raising=True,
    )
    is_consistent, reasons = check_mirror_drift()
    assert is_consistent is False
    assert any("_CONFIDENCE_OBSERVE_FLOOR" in r for r in reasons)


@pytest.mark.unit
def test_check_mirror_drift_detects_type_change(monkeypatch) -> None:
    rule_engine = importlib.import_module("smc.hedgerock.rule_engine")
    # Replace the float with a string — type drift.
    monkeypatch.setattr(
        rule_engine, "_CONFIDENCE_OBSERVE_FLOOR", "0.55-string", raising=True,
    )
    is_consistent, reasons = check_mirror_drift()
    assert is_consistent is False
    assert any("type" in r.lower() or "_CONFIDENCE_OBSERVE_FLOOR" in r
               for r in reasons)


@pytest.mark.unit
def test_check_mirror_drift_detects_missing_attr(monkeypatch) -> None:
    """If production removes the constant entirely, the mirror MUST
    report drift."""
    rule_engine = importlib.import_module("smc.hedgerock.rule_engine")
    monkeypatch.delattr(rule_engine, "_CONFIDENCE_AGGRESSIVE_FLOOR")
    is_consistent, reasons = check_mirror_drift()
    assert is_consistent is False


# ---------------------------------------------------------------------------
# 4. MirrorDriftError type
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_mirror_drift_error_is_runtime_error_subclass() -> None:
    assert issubclass(MirrorDriftError, RuntimeError)


# ---------------------------------------------------------------------------
# 5. compute_mirror_version determinism
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_compute_mirror_version_is_deterministic() -> None:
    a = compute_mirror_version()
    b = compute_mirror_version()
    assert a == b
    assert isinstance(a, str)
    assert len(a) == 64  # sha256 hex


# ---------------------------------------------------------------------------
# 6. Mirror module does not mutate production state
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_mirror_does_not_mutate_rule_engine_when_imported() -> None:
    """Importing the mirror twice is fine and does not change
    production module-level constants."""
    rule_engine = importlib.import_module("smc.hedgerock.rule_engine")
    before = {
        attr: getattr(rule_engine, attr)
        for attr in ("_CONFIDENCE_OBSERVE_FLOOR", "_CONFIDENCE_AGGRESSIVE_FLOOR")
    }
    importlib.import_module("smc.hedgerock.evolution.rule_engine_mirror")
    after = {
        attr: getattr(rule_engine, attr)
        for attr in ("_CONFIDENCE_OBSERVE_FLOOR", "_CONFIDENCE_AGGRESSIVE_FLOOR")
    }
    assert before == after
