"""Unit tests for atomic JSON state persistence utility (P0-4).

Covers:
- atomic_write_json: round-trip read/write
- atomic_write_json: creates parent directory
- atomic_write_json: no partial file on write failure (tmp cleanup)
- load_json: returns default when file is missing
- load_json: returns default when file is corrupt JSON
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from smc.monitor.state_io import atomic_write_json, load_json


class TestAtomicWriteJson:
    def test_atomic_write_and_read(self, tmp_path: Path) -> None:
        """Write a dict and read it back — values survive the round-trip."""
        target = tmp_path / "state.json"
        data = {"direction": "buy", "ts": "2026-01-01T00:00:00+00:00"}

        atomic_write_json(target, data)

        raw = json.loads(target.read_text(encoding="utf-8"))
        assert raw == data

    def test_atomic_write_creates_parent_dir(self, tmp_path: Path) -> None:
        """Parent directory is created automatically if it does not exist."""
        target = tmp_path / "nested" / "deep" / "state.json"
        assert not target.parent.exists()

        atomic_write_json(target, {"x": 1})

        assert target.exists()
        assert json.loads(target.read_text(encoding="utf-8")) == {"x": 1}

    def test_atomic_write_no_partial_on_error(self, tmp_path: Path) -> None:
        """If write_text raises, the tmp file is cleaned up and target unchanged.

        Simulate failure by patching Path.write_text to raise after the tmp
        file would be created.  The destination file must not exist (or remain
        unchanged if it existed before).
        """
        target = tmp_path / "state.json"
        # Pre-create with known content so we can verify it isn't changed.
        target.write_text(json.dumps({"safe": True}), encoding="utf-8")

        original_content = target.read_text(encoding="utf-8")
        tmp_path_expected = target.with_suffix(target.suffix + ".tmp")

        # Patch write_text on Path instances to raise after first call would
        # write the tmp file.  We need to intercept only the tmp write.
        _real_write = Path.write_text

        call_count = {"n": 0}

        def _failing_write(self: Path, data: str, encoding: str = "utf-8") -> None:
            call_count["n"] += 1
            if self == tmp_path_expected:
                raise OSError("simulated disk full")
            _real_write(self, data, encoding=encoding)

        with patch.object(Path, "write_text", _failing_write):
            with pytest.raises(OSError, match="simulated disk full"):
                atomic_write_json(target, {"bad": True})

        # tmp file must be cleaned up
        assert not tmp_path_expected.exists()
        # Target must retain its original content
        assert target.read_text(encoding="utf-8") == original_content

    def test_atomic_write_overwrites_existing(self, tmp_path: Path) -> None:
        """Subsequent writes overwrite the previous content atomically."""
        target = tmp_path / "state.json"
        atomic_write_json(target, {"v": 1})
        atomic_write_json(target, {"v": 2})

        assert json.loads(target.read_text())["v"] == 2

    def test_atomic_write_default_str_serialiser(self, tmp_path: Path) -> None:
        """Non-serialisable types (e.g. datetime) are coerced to str."""
        from datetime import datetime, timezone

        target = tmp_path / "ts.json"
        now = datetime.now(tz=timezone.utc)
        atomic_write_json(target, {"ts": now})

        raw = json.loads(target.read_text())
        assert isinstance(raw["ts"], str)


class TestLoadJson:
    def test_load_missing_returns_default(self, tmp_path: Path) -> None:
        """Missing file returns the specified default value."""
        missing = tmp_path / "does_not_exist.json"
        result = load_json(missing, default={"fallback": True})
        assert result == {"fallback": True}

    def test_load_missing_returns_empty_dict_when_default_none(
        self, tmp_path: Path
    ) -> None:
        """When default=None (omitted), fall back returns {} not None."""
        missing = tmp_path / "does_not_exist.json"
        result = load_json(missing)
        assert result == {}

    def test_load_corrupt_returns_default(self, tmp_path: Path) -> None:
        """Corrupt JSON returns the specified default rather than raising."""
        bad = tmp_path / "corrupt.json"
        bad.write_text("not valid json }{", encoding="utf-8")

        result = load_json(bad, default={"ok": False})
        assert result == {"ok": False}

    def test_load_valid_json(self, tmp_path: Path) -> None:
        """Well-formed JSON is loaded and returned as a Python object."""
        f = tmp_path / "good.json"
        payload = {"a": 1, "b": [2, 3]}
        f.write_text(json.dumps(payload), encoding="utf-8")

        assert load_json(f) == payload

    def test_load_default_none_returns_empty_dict_on_error(
        self, tmp_path: Path
    ) -> None:
        """Default is None → load_json falls back to {} on any error."""
        missing = tmp_path / "nope.json"
        assert load_json(missing, default=None) == {}
