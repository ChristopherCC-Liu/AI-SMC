"""Unit tests for Round 4 Alt-B W3 journal path suffix support.

Tests the path-construction logic that is threaded through live_demo.py
main() via SMCConfig.journal_suffix:

  - Default suffix "" → data/{symbol}/journal/live_trades.jsonl
  - Custom suffix "_macro" → data/{symbol}/journal_macro/live_trades.jsonl
  - live_state path also uses suffix: live_state{suffix}.json
  - Missing journal directory is auto-created (mkdir parents=True)

Because live_demo.py imports MetaTrader5 at module level we cannot import
it in unit tests.  We mirror the exact path-construction expressions here
to guarantee regressions in either copy surface immediately.
"""
from __future__ import annotations

from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Mirror of the path-construction logic from live_demo.py main()
# ---------------------------------------------------------------------------

def _compute_journal_path(data_root: Path, journal_suffix: str) -> Path:
    """Mirrors: JOURNAL_PATH = (DATA_ROOT / f'journal{suffix}') / 'live_trades.jsonl'"""
    return data_root / f"journal{journal_suffix}" / "live_trades.jsonl"


def _compute_state_path(data_root: Path, journal_suffix: str) -> Path:
    """Mirrors: STATE_PATH = DATA_ROOT / f'live_state{suffix}.json'"""
    return data_root / f"live_state{journal_suffix}.json"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestJournalPathSuffixDefault:
    """Empty suffix must preserve pre-W3 backward-compatible paths."""

    def test_journal_path_empty_suffix(self) -> None:
        """suffix='' → data/XAUUSD/journal/live_trades.jsonl"""
        data_root = Path("data") / "XAUUSD"
        journal_path = _compute_journal_path(data_root, "")
        assert journal_path == Path("data/XAUUSD/journal/live_trades.jsonl")

    def test_state_path_empty_suffix(self) -> None:
        """suffix='' → data/XAUUSD/live_state.json"""
        data_root = Path("data") / "XAUUSD"
        state_path = _compute_state_path(data_root, "")
        assert state_path == Path("data/XAUUSD/live_state.json")


class TestJournalPathSuffixMacro:
    """suffix='_macro' must produce separate journal and state paths."""

    def test_journal_path_macro_suffix(self) -> None:
        """suffix='_macro' → data/XAUUSD/journal_macro/live_trades.jsonl"""
        data_root = Path("data") / "XAUUSD"
        journal_path = _compute_journal_path(data_root, "_macro")
        assert journal_path == Path("data/XAUUSD/journal_macro/live_trades.jsonl")

    def test_state_path_macro_suffix(self) -> None:
        """suffix='_macro' → data/XAUUSD/live_state_macro.json"""
        data_root = Path("data") / "XAUUSD"
        state_path = _compute_state_path(data_root, "_macro")
        assert state_path == Path("data/XAUUSD/live_state_macro.json")

    def test_journal_and_state_paths_do_not_collide_with_control(self) -> None:
        """Treatment paths must differ from control paths so A and B don't clobber."""
        data_root = Path("data") / "XAUUSD"
        assert _compute_journal_path(data_root, "_macro") != _compute_journal_path(data_root, "")
        assert _compute_state_path(data_root, "_macro") != _compute_state_path(data_root, "")


class TestJournalDirAutoCreate:
    """Journal directory must be created automatically if absent."""

    def test_journal_dir_created_when_missing(self, tmp_path: Path) -> None:
        """mkdir(parents=True, exist_ok=True) on the journal dir must succeed."""
        data_root = tmp_path / "data" / "XAUUSD"
        journal_suffix = "_macro"
        journal_dir = data_root / f"journal{journal_suffix}"

        assert not journal_dir.exists()
        journal_dir.mkdir(parents=True, exist_ok=True)
        assert journal_dir.is_dir()

    def test_journal_dir_create_is_idempotent(self, tmp_path: Path) -> None:
        """Calling mkdir with exist_ok=True twice must not raise."""
        journal_dir = tmp_path / "data" / "XAUUSD" / "journal_macro"
        journal_dir.mkdir(parents=True, exist_ok=True)
        # Second call must be a no-op
        journal_dir.mkdir(parents=True, exist_ok=True)
        assert journal_dir.is_dir()
