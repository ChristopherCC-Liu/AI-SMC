"""Tests for hedgerock.cache_writer."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from smc.hedgerock.cache_writer import (
    DEFAULT_CACHE_FILENAME,
    ENV_CACHE_PATH,
    resolve_cache_path,
    write_envelope,
)
from smc.hedgerock.schemas import SCHEMA_VERSION, SignalEnvelope


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def envelope() -> SignalEnvelope:
    return SignalEnvelope(
        symbol="XAUUSD",
        generated_at=datetime(2026, 4, 26, 10, 30, 0, tzinfo=timezone.utc),
        active_timeframe="H1",
        active_strategy_id="xauusd_h1_trend",
        regime="trend_up",  # v2.0.0 lowercase enum
    )


# ---------------------------------------------------------------------------
# resolve_cache_path
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_resolve_cache_path_explicit_wins(tmp_path: Path) -> None:
    explicit = tmp_path / DEFAULT_CACHE_FILENAME
    with patch.dict(os.environ, {ENV_CACHE_PATH: "/tmp/should-be-ignored.json"}):
        assert resolve_cache_path(explicit) == explicit


@pytest.mark.unit
def test_resolve_cache_path_falls_back_to_env(tmp_path: Path) -> None:
    target = tmp_path / "from_env.json"
    with patch.dict(os.environ, {ENV_CACHE_PATH: str(target)}):
        assert resolve_cache_path() == target


@pytest.mark.unit
def test_resolve_cache_path_unset_raises() -> None:
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop(ENV_CACHE_PATH, None)
        with pytest.raises(RuntimeError, match="Cache path not configured"):
            resolve_cache_path()


# ---------------------------------------------------------------------------
# write_envelope happy path
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_write_envelope_creates_file(envelope: SignalEnvelope, tmp_path: Path) -> None:
    target = tmp_path / DEFAULT_CACHE_FILENAME
    written = write_envelope(envelope, target)
    assert written == target
    assert target.exists()


@pytest.mark.unit
def test_write_envelope_round_trips_json(envelope: SignalEnvelope, tmp_path: Path) -> None:
    target = tmp_path / DEFAULT_CACHE_FILENAME
    write_envelope(envelope, target)
    loaded = json.loads(target.read_text(encoding="utf-8"))
    assert loaded["symbol"] == "XAUUSD"
    assert loaded["regime"] == "trend_up"
    assert loaded["active_timeframe"] == "H1"
    assert loaded["schema_version"] == SCHEMA_VERSION
    # Defaults serialised
    assert loaded["exit_directive"] == "none"
    assert loaded["grid_multiplier"] == 1.0
    assert loaded["lot_factor"] == 1.0


@pytest.mark.unit
def test_write_envelope_creates_parent_dirs(envelope: SignalEnvelope, tmp_path: Path) -> None:
    target = tmp_path / "deep" / "nested" / "path" / DEFAULT_CACHE_FILENAME
    write_envelope(envelope, target)
    assert target.exists()


@pytest.mark.unit
def test_write_envelope_overwrites_existing(envelope: SignalEnvelope, tmp_path: Path) -> None:
    target = tmp_path / DEFAULT_CACHE_FILENAME
    target.write_text("stale data", encoding="utf-8")
    write_envelope(envelope, target)
    assert json.loads(target.read_text(encoding="utf-8"))["symbol"] == "XAUUSD"


@pytest.mark.unit
def test_write_envelope_uses_env_var_when_path_omitted(
    envelope: SignalEnvelope, tmp_path: Path
) -> None:
    target = tmp_path / "env_target.json"
    with patch.dict(os.environ, {ENV_CACHE_PATH: str(target)}):
        write_envelope(envelope)
    assert target.exists()
    assert json.loads(target.read_text(encoding="utf-8"))["symbol"] == "XAUUSD"


# ---------------------------------------------------------------------------
# Atomicity: no partial file ever visible
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_write_envelope_does_not_leave_temp_files_on_success(
    envelope: SignalEnvelope, tmp_path: Path
) -> None:
    target = tmp_path / DEFAULT_CACHE_FILENAME
    write_envelope(envelope, target)
    siblings = list(tmp_path.iterdir())
    # Only the final file remains; no leftover .tmp
    assert len(siblings) == 1
    assert siblings[0] == target


@pytest.mark.unit
def test_write_envelope_cleans_up_temp_on_replace_failure(
    envelope: SignalEnvelope, tmp_path: Path
) -> None:
    target = tmp_path / DEFAULT_CACHE_FILENAME
    boom = OSError("simulated rename failure")

    with patch("smc.hedgerock.cache_writer.os.replace", side_effect=boom):
        with pytest.raises(OSError, match="simulated rename failure"):
            write_envelope(envelope, target)

    leftover_tmp = [p for p in tmp_path.iterdir() if p.name.startswith(".")]
    assert leftover_tmp == [], f"Temp file not cleaned up: {leftover_tmp}"


@pytest.mark.unit
def test_write_envelope_atomic_target_never_partial(
    envelope: SignalEnvelope, tmp_path: Path
) -> None:
    """If replace fails, the target should either be the OLD file or absent
    — never a half-written new payload.
    """
    target = tmp_path / DEFAULT_CACHE_FILENAME
    target.write_text('{"symbol":"OLD"}', encoding="utf-8")

    with patch("smc.hedgerock.cache_writer.os.replace", side_effect=OSError("fail")):
        with pytest.raises(OSError):
            write_envelope(envelope, target)

    # Target untouched (the OLD payload survives)
    assert json.loads(target.read_text(encoding="utf-8")) == {"symbol": "OLD"}


# ---------------------------------------------------------------------------
# JSON shape contract (the EA depends on this)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_written_json_is_indented_for_human_inspection(
    envelope: SignalEnvelope, tmp_path: Path
) -> None:
    target = tmp_path / DEFAULT_CACHE_FILENAME
    write_envelope(envelope, target)
    text = target.read_text(encoding="utf-8")
    # Indented (2 spaces) JSON has newlines; minified JSON does not.
    assert "\n" in text
    assert text.count("\n") > 5


@pytest.mark.unit
def test_written_json_keys_are_sorted(envelope: SignalEnvelope, tmp_path: Path) -> None:
    target = tmp_path / DEFAULT_CACHE_FILENAME
    write_envelope(envelope, target)
    loaded = json.loads(target.read_text(encoding="utf-8"))
    assert list(loaded.keys()) == sorted(loaded.keys())


@pytest.mark.unit
def test_written_json_uses_iso8601_for_datetime(
    envelope: SignalEnvelope, tmp_path: Path
) -> None:
    target = tmp_path / DEFAULT_CACHE_FILENAME
    write_envelope(envelope, target)
    loaded = json.loads(target.read_text(encoding="utf-8"))
    # Pydantic JSON-mode serialises datetime as ISO 8601 string with offset
    assert loaded["generated_at"].startswith("2026-04-26T10:30:00")
