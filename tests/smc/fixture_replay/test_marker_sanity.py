"""Sanity tests for the ``fixture_replay`` pytest marker plumbing (R10 P4.1).

Two checks:
- The marker is registered (no ``PytestUnknownMarkWarning`` when applied).
- The marker correctly gates collection: the suite is opt-in via
  ``pytest -m fixture_replay`` and is not picked up by the default
  ``pytest`` invocation in CI.

The second guarantee is asserted indirectly by re-running the test runner
in a subprocess. We keep the subprocess scope narrow so the test stays
under a second.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_REPLAY_DIR = Path(__file__).resolve().parent


@pytest.mark.fixture_replay
def test_marker_registered_and_collected_under_replay() -> None:
    """A trivial passing test that proves the marker is collectable."""
    assert True


def test_default_pytest_run_excludes_fixture_replay_via_marker_filter() -> None:
    """When invoked without ``-m fixture_replay`` and with ``-m 'not fixture_replay'``,
    the replay-tagged tests are excluded.

    This is the contract the CI default suite relies on: developers running
    ``pytest`` for fast iteration shouldn't pay the cost of the replay
    fixtures unless they explicitly opt in.
    """
    # Run pytest in --collect-only mode, restricted to this directory, with
    # the inverse marker filter. We expect ZERO collected items because every
    # test in this package other than the current one is fixture_replay-marked.
    cmd = [
        sys.executable, "-m", "pytest",
        str(_REPLAY_DIR),
        "--collect-only", "-q",
        "-m", "not fixture_replay",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    # Pytest exit code 5 = "no tests collected"; that is the desired outcome
    # because every other test in the package carries the marker. The current
    # test itself is excluded because pytest skips the file we're inside when
    # we restrict to --collect-only of children only? No — collect-only still
    # discovers this file. So the current test (NOT marked fixture_replay)
    # WILL collect, giving exit code 0 with at least 1 item. We accept either:
    #   exit 0 with the only collected item being THIS test, or
    #   exit 5 if pytest's marker filter is even stricter than expected.
    assert result.returncode in (0, 5), (
        f"unexpected pytest exit {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    # Whichever path: NONE of the marked replay tests should appear in the
    # collected output. Spot-check by scanning stdout for known marked test
    # names.
    forbidden = (
        "test_marker_registered_and_collected_under_replay",
        "test_apr_20_blocked_with_fix_enabled",
        "test_apr_20_baseline_without_fix_documents_bug",
        "test_apr_21_blocked_with_fix_enabled",
        "test_apr_21_baseline_without_fix_documents_bug",
    )
    for name in forbidden:
        assert name not in result.stdout, (
            f"marker filter leaked: {name} should be excluded\n"
            f"stdout:\n{result.stdout}"
        )
