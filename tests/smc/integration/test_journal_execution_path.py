"""Round 6 B3: journal rows must carry execution_path alongside mode.

Round 4 v5 made the EA the sole executor — Python's ``order_send`` is
permanently disabled.  As a result every live journal row carries
``mode=PAPER`` even though the EA successfully executes.  ``execution_path``
disambiguates by recording *who actually places the trade*:

    "ea"     → EA is the executor (Round 4 v5 default)
    "paper"  → --paper / --no-execute: no execution at all
    "python" → reserved (Python order_send, dormant)

This module verifies, by static inspection of ``scripts/live_demo.py``,
that every ``log_entry = {"time": ..., "cycle": ..., ..., "mode": ...}``
dict that is subsequently appended to the journal also carries an
``execution_path`` key.  We do not exec live_demo — the module runs the
trading loop at import time via ``_ensure_single_instance()``.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest


_DEMO_PATH = Path(__file__).parents[3] / "scripts" / "live_demo.py"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_module_source() -> str:
    return _DEMO_PATH.read_text(encoding="utf-8")


def _find_log_entry_dicts(source: str) -> list[ast.Dict]:
    """Return every ``ast.Dict`` node assigned to the name ``log_entry``.

    Matches patterns like::

        log_entry = {
            "time": ...,
            "cycle": ...,
            ...,
        }

    We only collect dicts that look like journal rows (have both ``time``
    and ``cycle`` string keys) so unrelated ``log_entry = {...}`` helpers
    elsewhere don't pollute the assertion.
    """
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
        if {"time", "cycle"}.issubset(keys):
            found.append(node.value)
    return found


def _dict_keys(node: ast.Dict) -> set[str]:
    return {
        k.value
        for k in node.keys
        if isinstance(k, ast.Constant) and isinstance(k.value, str)
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestJournalExecutionPath:
    """Every journal row must carry ``execution_path`` alongside ``mode``."""

    @pytest.fixture(scope="class")
    def journal_rows(self) -> list[ast.Dict]:
        rows = _find_log_entry_dicts(_extract_module_source())
        # Round 6 B3: three distinct journal write points in live_demo.py —
        # MARGIN_GATED, range (main), trending.  If that structure changes,
        # this guard prevents silent regressions.
        assert len(rows) >= 3, (
            f"Expected ≥3 journal log_entry dicts in live_demo.py, "
            f"found {len(rows)}. Did the journal writer refactor?"
        )
        return rows

    def test_every_journal_row_has_mode(self, journal_rows):
        missing = [row for row in journal_rows if "mode" not in _dict_keys(row)]
        assert not missing, (
            f"{len(missing)} journal log_entry dict(s) missing `mode` field "
            f"at line(s) {[row.lineno for row in missing]}"
        )

    def test_every_journal_row_has_execution_path(self, journal_rows):
        """Round 6 B3 contract: new field `execution_path` on every row."""
        missing = [
            row for row in journal_rows
            if "execution_path" not in _dict_keys(row)
        ]
        assert not missing, (
            f"{len(missing)} journal log_entry dict(s) missing "
            f"`execution_path` at line(s) {[row.lineno for row in missing]}. "
            "Every journal row must carry execution_path per Round 6 B3."
        )

    def test_mode_field_still_present_for_reverse_compat(self, journal_rows):
        """`mode` must not be dropped — legacy consumers still grep it."""
        for row in journal_rows:
            keys = _dict_keys(row)
            assert "mode" in keys, (
                f"journal row at line {row.lineno} dropped `mode` — "
                "breaks reverse compat with pre-Round-6 consumers."
            )


class TestExecutionPathInitialised:
    """``_execution_path`` must be assigned before any journal write.

    Static AST walk: the Name ``_execution_path`` must be the target of at
    least one Assign node, and none of the journal ``log_entry`` dicts may
    reference an unassigned form like ``"execution_path": "python"`` (we
    forbid ``python`` in live_demo because Python order_send is dormant).
    """

    @pytest.fixture(scope="class")
    def source(self) -> str:
        return _extract_module_source()

    def test_execution_path_variable_assigned(self, source):
        tree = ast.parse(source)
        assigns = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "_execution_path"
                for t in node.targets
            )
        ]
        assert assigns, (
            "`_execution_path` must be assigned at least once in "
            "scripts/live_demo.py before the first journal write."
        )

    def test_no_python_execution_path_literal(self, source):
        """Round 4 v5: Python order_send is DISABLED — don't emit `"python"`."""
        rows = _find_log_entry_dicts(source)
        offenders = []
        for row in rows:
            for key, value in zip(row.keys, row.values):
                if (
                    isinstance(key, ast.Constant)
                    and key.value == "execution_path"
                    and isinstance(value, ast.Constant)
                    and value.value == "python"
                ):
                    offenders.append(row.lineno)
        assert not offenders, (
            f"journal rows at line(s) {offenders} set "
            "execution_path='python' — Python order_send is dormant in "
            "Round 4 v5. Use 'ea' (default) or 'paper' (--paper)."
        )


class TestMarginGatedRow:
    """The MARGIN_GATED journal row must tag execution_path as EA.

    Rationale: the gate blocks an *EA-intended* order.  The intent was EA,
    only the margin guard stopped it — so the row should stay "ea" for
    post-hoc audit ("how many EA signals did margin-cap kill?").
    """

    def test_margin_gated_row_tagged_ea(self):
        source = _extract_module_source()
        rows = _find_log_entry_dicts(source)
        mg_rows = [
            row for row in rows
            if "margin_gated" in _dict_keys(row)
        ]
        assert mg_rows, "Could not locate the MARGIN_GATED log_entry row."
        assert len(mg_rows) == 1, (
            f"Expected exactly one MARGIN_GATED row, got {len(mg_rows)}."
        )
        row = mg_rows[0]
        # The value attached to "execution_path" should be the
        # _execution_path variable (Name node), not a hard-coded string —
        # because --paper still routes through gate logic with "paper".
        ep_value: ast.AST | None = None
        for key, value in zip(row.keys, row.values):
            if isinstance(key, ast.Constant) and key.value == "execution_path":
                ep_value = value
                break
        assert ep_value is not None, "MARGIN_GATED row missing execution_path."
        assert isinstance(ep_value, ast.Name), (
            "MARGIN_GATED row's execution_path should reference the "
            "`_execution_path` variable (which honours --paper), not a "
            "hard-coded literal."
        )
        assert ep_value.id == "_execution_path"
