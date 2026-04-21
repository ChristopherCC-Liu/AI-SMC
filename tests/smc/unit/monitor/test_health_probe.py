"""Unit tests for smc.monitor.health_probe — Round 5 R2 cycle health event.

Covers:

- ``_compute_data_ok`` flag semantics across enum vs str keys, empty
  frames, missing timeframes.
- ``build_probe`` shape / type preservation.
- ``emit`` emits an INFO-severity ``health_probe`` event with exactly
  the specified fields (the daily digest depends on this schema).
"""
from __future__ import annotations

import io
import json
import re
from dataclasses import FrozenInstanceError

import pytest

from smc.monitor.health_probe import (
    HealthProbeFields,
    _compute_data_ok,
    build_probe,
    emit,
)


_EVENT_LINE = re.compile(r"^\[(?P<sev>[A-Z]+)\]\s+(?P<body>\{.*\})\s*$")


def _parse_events(captured_err: str) -> list[dict]:
    events: list[dict] = []
    for line in captured_err.splitlines():
        m = _EVENT_LINE.match(line)
        if not m:
            continue
        payload = json.loads(m.group("body"))
        payload["_severity"] = m.group("sev")
        events.append(payload)
    return events


# ---------------------------------------------------------------------------
# _compute_data_ok
# ---------------------------------------------------------------------------


class _FakeTf:
    """Stand-in for the Timeframe enum — exposes .name like an IntEnum."""

    def __init__(self, name: str) -> None:
        self.name = name


@pytest.mark.unit
def test_data_ok_all_timeframes_present() -> None:
    data = {
        _FakeTf("M15"): [0] * 10,
        _FakeTf("H1"): [0] * 5,
        _FakeTf("H4"): [0] * 3,
        _FakeTf("D1"): [0] * 2,
    }
    assert _compute_data_ok(data) is True


@pytest.mark.unit
def test_data_ok_accepts_string_keys() -> None:
    data = {"M15": [0], "H1": [0], "H4": [0], "D1": [0]}
    assert _compute_data_ok(data) is True


@pytest.mark.unit
def test_data_ok_missing_timeframe_returns_false() -> None:
    data = {"M15": [0], "H1": [0], "H4": [0]}  # missing D1
    assert _compute_data_ok(data) is False


@pytest.mark.unit
def test_data_ok_empty_frame_counted_missing() -> None:
    data = {"M15": [], "H1": [0], "H4": [0], "D1": [0]}
    assert _compute_data_ok(data) is False


@pytest.mark.unit
def test_data_ok_none_frame_counted_missing() -> None:
    data = {"M15": None, "H1": [0], "H4": [0], "D1": [0]}
    assert _compute_data_ok(data) is False


@pytest.mark.unit
@pytest.mark.parametrize("bad", [None, {}])
def test_data_ok_empty_or_none_data(bad) -> None:
    assert _compute_data_ok(bad) is False


@pytest.mark.unit
def test_data_ok_treats_opaque_len_less_frame_as_present() -> None:
    class _OpaqueFrame:
        pass

    data = {
        "M15": _OpaqueFrame(),
        "H1": _OpaqueFrame(),
        "H4": _OpaqueFrame(),
        "D1": _OpaqueFrame(),
    }
    assert _compute_data_ok(data) is True


# ---------------------------------------------------------------------------
# build_probe
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_build_probe_preserves_fields() -> None:
    data = {"M15": [0], "H1": [0], "H4": [0], "D1": [0]}
    p = build_probe(
        cycle=42,
        cycle_ts_iso="2026-04-21T03:15:00+00:00",
        leg="_macro",
        tick_ok=True,
        data=data,
        handle_age_sec=1234,
        handle_reset_count=2,
        debate_elapsed_ms_last=9100,
        macro_bias_fresh=True,
        balance_usd=1000.50,
        equity_usd=1015.25,
        floating_usd=14.75,
    )
    assert p.cycle == 42
    assert p.cycle_ts_iso == "2026-04-21T03:15:00+00:00"
    assert p.leg == "_macro"
    assert p.tick_ok is True
    assert p.data_ok is True
    assert p.handle_age_sec == 1234
    assert p.handle_reset_count == 2
    assert p.debate_elapsed_ms_last == 9100
    assert p.macro_bias_fresh is True
    assert p.balance_usd == 1000.50
    assert p.equity_usd == 1015.25
    assert p.floating_usd == 14.75


@pytest.mark.unit
def test_build_probe_nones_passed_through() -> None:
    p = build_probe(
        cycle=1,
        cycle_ts_iso="2026-04-21T03:15:00+00:00",
        tick_ok=False,
        data=None,
        handle_age_sec=None,
        debate_elapsed_ms_last=None,
        macro_bias_fresh=False,
        balance_usd=None,
    )
    assert p.leg == ""
    assert p.handle_reset_count == 0  # default
    assert p.data_ok is False
    assert p.handle_age_sec is None
    assert p.debate_elapsed_ms_last is None
    assert p.balance_usd is None
    assert p.equity_usd is None
    assert p.floating_usd is None


@pytest.mark.unit
def test_build_probe_defaults_leg_when_none_passed() -> None:
    """None → coerced to empty string (dataclass field is non-optional str)."""
    p = build_probe(
        cycle=1,
        cycle_ts_iso="2026-04-21T03:15:00+00:00",
        leg=None,  # type: ignore[arg-type]
        tick_ok=True,
        data={"M15": [0], "H1": [0], "H4": [0], "D1": [0]},
        handle_age_sec=0,
        debate_elapsed_ms_last=None,
        macro_bias_fresh=False,
        balance_usd=None,
    )
    assert p.leg == ""


@pytest.mark.unit
def test_health_probe_fields_is_immutable() -> None:
    p = HealthProbeFields(
        cycle=1, cycle_ts_iso="", leg="", tick_ok=True, data_ok=True,
        handle_age_sec=0, handle_reset_count=0,
        debate_elapsed_ms_last=None, macro_bias_fresh=False,
        balance_usd=None, equity_usd=None, floating_usd=None,
    )
    with pytest.raises(FrozenInstanceError):
        p.cycle = 99  # type: ignore[misc]


@pytest.mark.unit
def test_to_event_kwargs_includes_only_expected_keys() -> None:
    p = build_probe(
        cycle=7,
        cycle_ts_iso="2026-04-21T03:15:00+00:00",
        leg="_macro", tick_ok=True, data=None,
        handle_age_sec=10, handle_reset_count=0,
        debate_elapsed_ms_last=500,
        macro_bias_fresh=True, balance_usd=999.99,
        equity_usd=1002.22, floating_usd=2.23,
    )
    kwargs = p.to_event_kwargs()
    assert set(kwargs.keys()) == {
        "cycle", "cycle_ts_iso", "leg", "tick_ok", "data_ok",
        "handle_age_sec", "handle_reset_count",
        "debate_elapsed_ms_last", "macro_bias_fresh",
        "balance_usd", "equity_usd", "floating_usd",
    }


# ---------------------------------------------------------------------------
# emit
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_emit_writes_info_event_with_expected_fields(
    capsys: pytest.CaptureFixture[str],
) -> None:
    p = build_probe(
        cycle=3,
        cycle_ts_iso="2026-04-21T03:15:00+00:00",
        leg="_macro",
        tick_ok=True,
        data={"M15": [0], "H1": [0], "H4": [0], "D1": [0]},
        handle_age_sec=42,
        handle_reset_count=1,
        debate_elapsed_ms_last=8500,
        macro_bias_fresh=True,
        balance_usd=1234.56,
        equity_usd=1240.00,
        floating_usd=5.44,
    )
    emit(p)
    events = _parse_events(capsys.readouterr().err)
    assert len(events) == 1
    ev = events[0]
    assert ev["_severity"] == "INFO"
    assert ev["event"] == "health_probe"
    assert ev["cycle"] == 3
    assert ev["cycle_ts_iso"] == "2026-04-21T03:15:00+00:00"
    assert ev["leg"] == "_macro"
    assert ev["tick_ok"] is True
    assert ev["data_ok"] is True
    assert ev["handle_age_sec"] == 42
    assert ev["handle_reset_count"] == 1
    assert ev["debate_elapsed_ms_last"] == 8500
    assert ev["macro_bias_fresh"] is True
    assert ev["balance_usd"] == 1234.56
    assert ev["equity_usd"] == 1240.00
    assert ev["floating_usd"] == 5.44


@pytest.mark.unit
def test_emit_survives_logger_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """emit() must never raise, even if the structured logger blows up."""
    import smc.monitor.structured_log as sl

    def _explode(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(sl, "info", _explode)
    p = build_probe(
        cycle=1,
        cycle_ts_iso="2026-04-21T03:15:00+00:00",
        tick_ok=False, data=None,
        handle_age_sec=None, debate_elapsed_ms_last=None,
        macro_bias_fresh=False, balance_usd=None,
    )
    emit(p)  # should not raise
