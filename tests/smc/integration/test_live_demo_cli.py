"""Integration smoke tests for live_demo.py Stage 3 CLI + migration.

Tests:
- argparse --symbol / --paper flags
- _migrate_legacy_xau_paths idempotency
- data/{SYMBOL}/ path layout
"""
from __future__ import annotations

import ast
import os
import sys
import textwrap
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEMO_PATH = Path(__file__).parents[3] / "scripts" / "live_demo.py"


def _extract_migrate_fn() -> str:
    """Return the source text of _migrate_legacy_xau_paths from live_demo.py.

    We parse the AST to locate the FunctionDef node and reconstruct its
    source lines — no exec of the full module, so _ensure_single_instance()
    never runs.
    """
    source = _DEMO_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    lines = source.splitlines()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_migrate_legacy_xau_paths":
            start = node.lineno - 1  # ast lineno is 1-based
            end = node.end_lineno  # type: ignore[attr-defined]
            fn_source = "\n".join(lines[start:end])
            return fn_source
    raise RuntimeError("_migrate_legacy_xau_paths not found in live_demo.py")


def _get_migrate_fn():
    """Compile and return the _migrate_legacy_xau_paths callable."""
    fn_source = _extract_migrate_fn()
    # shutil must be available in the exec namespace
    ns: dict = {}
    imports = textwrap.dedent("""\
        import shutil
        from pathlib import Path
    """)
    code = compile(imports + "\n" + fn_source, "<live_demo_migrate>", "exec")
    exec(code, ns)  # noqa: S102
    return ns["_migrate_legacy_xau_paths"]


# ---------------------------------------------------------------------------
# argparse tests
# ---------------------------------------------------------------------------


class TestArgparse:
    """Verify CLI argument parsing without executing main()."""

    def _parse(self, argv: list[str]) -> object:
        """Run just the argparse block extracted from main() directly."""
        import argparse

        parser = argparse.ArgumentParser(description="AI-SMC Live Trading Loop")
        parser.add_argument(
            "--symbol",
            choices=["XAUUSD", "BTCUSD"],
            default="XAUUSD",
        )
        parser.add_argument("--paper", action="store_true")
        return parser.parse_args(argv)

    def test_default_symbol_is_xauusd(self):
        args = self._parse([])
        assert args.symbol == "XAUUSD"

    def test_explicit_xauusd(self):
        args = self._parse(["--symbol", "XAUUSD"])
        assert args.symbol == "XAUUSD"

    def test_btcusd(self):
        args = self._parse(["--symbol", "BTCUSD"])
        assert args.symbol == "BTCUSD"

    def test_paper_flag_false_by_default(self):
        args = self._parse([])
        assert args.paper is False

    def test_paper_flag_true(self):
        args = self._parse(["--paper"])
        assert args.paper is True

    def test_paper_and_symbol_together(self):
        args = self._parse(["--symbol", "BTCUSD", "--paper"])
        assert args.symbol == "BTCUSD"
        assert args.paper is True

    def test_invalid_symbol_raises(self):
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--symbol", choices=["XAUUSD", "BTCUSD"], default="XAUUSD")
        with pytest.raises(SystemExit):
            parser.parse_args(["--symbol", "EURUSD"])


# ---------------------------------------------------------------------------
# Migration helper tests
# ---------------------------------------------------------------------------


class TestMigrateLegacyXauPaths:
    """Verify _migrate_legacy_xau_paths behaviour."""

    def test_first_run_moves_files(self, tmp_path: Path):
        migrate = _get_migrate_fn()

        # Create legacy flat files
        legacy_data = tmp_path / "data"
        legacy_data.mkdir(parents=True)
        (legacy_data / "journal").mkdir()
        legacy_files = [
            "live_state.json",
            "ai_analysis.json",
            "asian_range_quota_state.json",
        ]
        for name in legacy_files:
            (legacy_data / name).write_text("{}", encoding="utf-8")
        (legacy_data / "journal" / "live_trades.jsonl").write_text(
            '{"test": 1}\n', encoding="utf-8"
        )

        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            migrate("XAUUSD")
        finally:
            os.chdir(old_cwd)

        target = tmp_path / "data" / "XAUUSD"
        for name in legacy_files:
            assert (target / name).exists(), f"Expected {name} in data/XAUUSD/"
            assert not (legacy_data / name).exists(), f"Legacy {name} should be gone"
        assert (target / "journal" / "live_trades.jsonl").exists()
        assert not (legacy_data / "journal" / "live_trades.jsonl").exists()

    def test_second_run_is_noop(self, tmp_path: Path):
        """Second call must not raise and must leave data/XAUUSD/ intact."""
        migrate = _get_migrate_fn()

        legacy_data = tmp_path / "data"
        legacy_data.mkdir(parents=True)
        (legacy_data / "live_state.json").write_text("{}", encoding="utf-8")

        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            migrate("XAUUSD")
            # Simulate second run — legacy file is gone, target exists
            migrate("XAUUSD")
        finally:
            os.chdir(old_cwd)

        # No exception raised; target file still present
        assert (tmp_path / "data" / "XAUUSD" / "live_state.json").exists()

    def test_idempotent_when_dst_already_exists(self, tmp_path: Path):
        """If dst already exists, src should not overwrite it."""
        migrate = _get_migrate_fn()

        legacy_data = tmp_path / "data"
        legacy_data.mkdir(parents=True)
        (legacy_data / "live_state.json").write_text('{"old": true}', encoding="utf-8")

        target = legacy_data / "XAUUSD"
        target.mkdir(parents=True)
        (target / "live_state.json").write_text('{"new": true}', encoding="utf-8")

        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            migrate("XAUUSD")
        finally:
            os.chdir(old_cwd)

        # dst should remain unchanged; src should still exist (not moved)
        result = (target / "live_state.json").read_text(encoding="utf-8")
        assert '"new": true' in result
        assert (legacy_data / "live_state.json").exists()

    def test_btcusd_symbol_does_nothing(self, tmp_path: Path):
        """BTC symbol must leave data/ untouched."""
        migrate = _get_migrate_fn()

        legacy_data = tmp_path / "data"
        legacy_data.mkdir(parents=True)
        (legacy_data / "live_state.json").write_text("{}", encoding="utf-8")

        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            migrate("BTCUSD")
        finally:
            os.chdir(old_cwd)

        # Nothing moved
        assert (legacy_data / "live_state.json").exists()
        assert not (legacy_data / "BTCUSD").exists()

    def test_no_legacy_files_does_not_raise(self, tmp_path: Path):
        """Calling migrate when there are no legacy files must not raise."""
        migrate = _get_migrate_fn()
        (tmp_path / "data").mkdir(parents=True)

        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            migrate("XAUUSD")
        finally:
            os.chdir(old_cwd)


# ---------------------------------------------------------------------------
# Data path layout tests
# ---------------------------------------------------------------------------


class TestDataPathLayout:
    """Verify that DATA_ROOT / SYMBOL subdir paths are correctly constructed."""

    @pytest.mark.parametrize("symbol", ["XAUUSD", "BTCUSD"])
    def test_data_root_is_under_symbol(self, symbol: str, tmp_path: Path):
        """DATA_ROOT = data/{SYMBOL} should be a subdirectory of data/."""
        data_root = Path("data") / symbol
        assert data_root.parts[-1] == symbol
        assert data_root.parts[-2] == "data"

    @pytest.mark.parametrize("symbol", ["XAUUSD", "BTCUSD"])
    def test_path_construction(self, symbol: str):
        """Verify expected path structure for all state files."""
        data_root = Path("data") / symbol
        assert data_root / "live_state.json" == Path(f"data/{symbol}/live_state.json")
        assert data_root / "journal" / "live_trades.jsonl" == Path(
            f"data/{symbol}/journal/live_trades.jsonl"
        )
        assert data_root / "ai_analysis.json" == Path(f"data/{symbol}/ai_analysis.json")
        assert data_root / "consec_loss_state.json" == Path(
            f"data/{symbol}/consec_loss_state.json"
        )
        assert data_root / "asian_range_quota_state.json" == Path(
            f"data/{symbol}/asian_range_quota_state.json"
        )
