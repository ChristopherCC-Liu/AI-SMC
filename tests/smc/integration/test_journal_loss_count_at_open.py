"""R10 cal-0b: every journal row carries `loss_count_in_window_at_open`.

Calibration runner (commit 2) consumes per-arm consec_loss histograms to
inform P3.1 loss-aware position sizing decay schedules. Reconstructing
the snapshot value from post-trade outcomes alone is lossy (operator
restarts mid-day reset the rolling deque), so we capture the value at
the moment the trade-decision is made.

Verifies, by static AST inspection of ``scripts/live_demo.py``:
    - every ``log_entry = {...}`` journal-row dict carries the new key
    - the captured value sources from ``consec_halt.snapshot()`` (not
      a stale module-level constant or a hardcoded zero)

Cross-checked against P3.2 V3 invariant: the field name maps 1:1 to
``ConsecLossSnapshot.loss_count_in_window`` (the rolling-window total,
NOT trailing-streak ``consec_losses``). Future maintainers must not
silently swap one for the other — calibration math depends on the
window-total semantic specifically.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


_DEMO_PATH = Path(__file__).parents[3] / "scripts" / "live_demo.py"
_FIELD_NAME = "loss_count_in_window_at_open"


# ---------------------------------------------------------------------------
# Helpers (mirrors test_journal_execution_path.py for cross-suite consistency)
# ---------------------------------------------------------------------------


def _extract_module_source() -> str:
    return _DEMO_PATH.read_text(encoding="utf-8")


def _find_log_entry_dicts(source: str) -> list[ast.Dict]:
    """Return every ``log_entry = {...}`` journal-row dict in live_demo.py."""
    tree = ast.parse(source)
    found: list[ast.Dict] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)):
            continue
        if node.targets[0].id != "log_entry":
            continue
        if not isinstance(node.value, ast.Dict):
            continue
        keys = {
            k.value
            for k in node.value.keys
            if isinstance(k, ast.Constant) and isinstance(k.value, str)
        }
        # Only journal rows have both ``time`` and ``cycle`` string keys.
        if {"time", "cycle"}.issubset(keys):
            found.append(node.value)
    return found


def _dict_keys(node: ast.Dict) -> set[str]:
    return {
        k.value
        for k in node.keys
        if isinstance(k, ast.Constant) and isinstance(k.value, str)
    }


def _value_for_key(node: ast.Dict, key_name: str) -> ast.expr | None:
    """Return the AST value node for ``key_name`` in ``node`` (or None)."""
    for k, v in zip(node.keys, node.values):
        if isinstance(k, ast.Constant) and k.value == key_name:
            return v
    return None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLossCountInWindowAtOpenCapture:
    """All 3 journal write sites carry ``loss_count_in_window_at_open``."""

    @pytest.fixture(scope="class")
    def journal_rows(self) -> list[ast.Dict]:
        rows = _find_log_entry_dicts(_extract_module_source())
        # 3 distinct write points: MARGIN_GATED, range (main), trending.
        # Mirrors the guard in test_journal_execution_path.py — if the
        # journal-writer count diverges we want a loud failure.
        assert len(rows) >= 3, (
            f"Expected >=3 log_entry dicts in live_demo.py, found {len(rows)}. "
            "Did the journal writer refactor?"
        )
        return rows

    def test_every_journal_row_has_loss_count_field(self, journal_rows):
        """Every journal row MUST carry the new calibration field.

        Future-proofing: when calibration runner consumes journals, missing
        rows degrade the histogram silently. The classifier helper defaults
        to None for legacy records (pre-cal-0b), but post-cal-0b every row
        must populate the field.
        """
        missing = [
            row for row in journal_rows
            if _FIELD_NAME not in _dict_keys(row)
        ]
        assert not missing, (
            f"{len(missing)} journal log_entry dict(s) missing "
            f"`{_FIELD_NAME}` at line(s) {[row.lineno for row in missing]}. "
            "Every journal row must carry the rolling-window snapshot per "
            "R10 cal-0b for calibration runner consumption."
        )

    def test_value_sources_from_consec_halt_snapshot(self, journal_rows):
        """The value MUST be `consec_halt.snapshot().loss_count_in_window`.

        Guards against a future refactor that silently swaps in
        `consec_halt.consec_losses` (trailing-streak semantic) — calibration
        math depends on the window-total semantic specifically. The split
        between trailing and window-total was the contentious part of
        P3.2 V3 cross-review and we do NOT want to lose that distinction.
        """
        offenders = []
        for row in journal_rows:
            value = _value_for_key(row, _FIELD_NAME)
            if value is None:
                # Caught by the previous test; skip here.
                continue
            # Expected shape: ``consec_halt.snapshot().loss_count_in_window``
            # which is Attribute(Call(Attribute(Name(consec_halt), snapshot), ...), loss_count_in_window).
            ok = (
                isinstance(value, ast.Attribute)
                and value.attr == "loss_count_in_window"
                and isinstance(value.value, ast.Call)
                and isinstance(value.value.func, ast.Attribute)
                and value.value.func.attr == "snapshot"
                and isinstance(value.value.func.value, ast.Name)
                and value.value.func.value.id == "consec_halt"
            )
            if not ok:
                offenders.append(row.lineno)
        assert not offenders, (
            f"journal rows at line(s) {offenders} have an unexpected "
            f"value shape for `{_FIELD_NAME}`. Must be exactly "
            "`consec_halt.snapshot().loss_count_in_window` to preserve "
            "the window-total semantic (NOT consec_losses trailing-streak)."
        )

    def test_field_name_is_window_specific_not_consec_losses(
        self, journal_rows
    ):
        """Schema-level lock: NEVER name the field `consec_losses_at_open`.

        That would suggest the trailing-streak semantic, which is the wrong
        input for calibration's loss-distribution histogram. The whole
        point of P3.2 V3's split into two accessors was to disambiguate;
        this test pins the disambiguation at the journal schema layer.
        """
        for row in journal_rows:
            keys = _dict_keys(row)
            assert "consec_losses_at_open" not in keys, (
                f"row at line {row.lineno} uses the trailing-streak field "
                "name. Use `loss_count_in_window_at_open` (window-total)."
            )


class TestRollingWindowSemanticInvariant:
    """Cross-suite invariant: the V3 P3.2 split semantic is preserved.

    Defense's V3 P3.2 introduced two accessors on ConsecLossSnapshot:
        - consec_losses (trailing): scans backward, stops at first win
        - loss_count_in_window (total): counts all losses in the deque

    Foundation flagged during cross-review that "deque + monotonic counter
    is two truths" which led to the split. Calibration consumes the
    window-total. This invariant test protects against either:
        - A future regression that conflates the two accessors
        - A live_demo.py change that captures the wrong one
    """

    def test_consec_loss_state_property_still_exists(self):
        """The property must exist on the state dataclass."""
        from smc.risk.consec_loss_halt import ConsecLossState
        # Cheap structural assertion: instances of the state must expose
        # loss_count_in_window. If the property is removed or renamed,
        # this import-time check fails before the schema test has a
        # chance to mislead a reader with a "field present" pass.
        assert hasattr(ConsecLossState, "loss_count_in_window"), (
            "ConsecLossState.loss_count_in_window property is missing. "
            "Cal-0b's journal field captures this; if the source property "
            "is gone, the journal value will silently break."
        )

    def test_consec_loss_state_keeps_consec_losses_too(self):
        """consec_losses must still exist (operator-display string uses it).

        This pins the V3 P3.2 split: BOTH accessors live on the state,
        each with its specific consumer. We do not want a future cleanup
        to drop ``consec_losses`` thinking it's redundant.
        """
        from smc.risk.consec_loss_halt import ConsecLossState
        assert hasattr(ConsecLossState, "consec_losses"), (
            "ConsecLossState.consec_losses property is missing. "
            "Operator-facing displays use it; do NOT remove."
        )
