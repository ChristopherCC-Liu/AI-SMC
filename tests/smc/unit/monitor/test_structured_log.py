"""Unit tests for smc.monitor.structured_log."""
from __future__ import annotations

import io
import json
from datetime import datetime, timezone

import pytest

from smc.monitor.structured_log import crit, info, log_event, warn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _capture(severity: str, event: str, **fields) -> dict:
    """Call log_event into a StringIO and return the parsed JSON payload."""
    buf = io.StringIO()
    log_event(severity, event, stream=buf, **fields)
    line = buf.getvalue()
    # Strip the "[SEV] " prefix, then parse JSON
    prefix = f"[{severity}] "
    assert line.startswith(prefix), f"Expected prefix {prefix!r}, got: {line!r}"
    payload_str = line[len(prefix) :].rstrip("\n")
    return json.loads(payload_str)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_log_event_emits_tag_and_json():
    """log_event writes a [SEV] prefix followed by valid JSON to the stream."""
    buf = io.StringIO()
    log_event("INFO", "test_event", stream=buf, foo="bar", num=42)
    line = buf.getvalue()

    assert line.startswith("[INFO] ")
    payload = json.loads(line[len("[INFO] ") :])
    assert payload["event"] == "test_event"
    assert payload["foo"] == "bar"
    assert payload["num"] == 42
    assert "ts" in payload


def test_log_event_severity_tag():
    """[CRIT] tag must appear at the very start of the emitted line."""
    buf = io.StringIO()
    log_event("CRIT", "critical_event", stream=buf)
    assert buf.getvalue().startswith("[CRIT]")


def test_log_event_default_str_serializes_datetime():
    """Passing a datetime value must not raise; it serialises via str."""
    dt = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
    buf = io.StringIO()
    # Should not raise
    log_event("INFO", "with_datetime", stream=buf, ts_field=dt)
    line = buf.getvalue()
    payload = json.loads(line[len("[INFO] ") :])
    # The datetime should be present as a string
    assert "ts_field" in payload
    assert isinstance(payload["ts_field"], str)


def test_log_event_ts_field_is_iso8601():
    """The auto-generated 'ts' field must be a parseable ISO-8601 string."""
    payload = _capture("WARN", "ts_check")
    # datetime.fromisoformat will raise if not valid ISO-8601
    parsed = datetime.fromisoformat(payload["ts"])
    assert parsed.tzinfo is not None  # must be timezone-aware


def test_crit_warn_info_shortcuts():
    """Each shortcut helper must emit the correct severity tag."""
    for func, expected_tag in [(crit, "CRIT"), (warn, "WARN"), (info, "INFO")]:
        buf = io.StringIO()
        func("shortcut_event", stream=buf)
        line = buf.getvalue()
        assert line.startswith(f"[{expected_tag}] "), (
            f"{func.__name__} emitted wrong tag: {line!r}"
        )
        payload = json.loads(line[len(f"[{expected_tag}] ") :])
        assert payload["event"] == "shortcut_event"


def test_log_event_never_raises_on_broken_stream():
    """log_event must not propagate exceptions from a closed/broken stream."""
    buf = io.StringIO()
    buf.close()
    # Should not raise even though the stream is closed
    log_event("CRIT", "should_not_raise", stream=buf)


def test_log_event_extra_fields_included():
    """Arbitrary keyword fields are included verbatim in the JSON body."""
    payload = _capture("DEBUG", "extra_fields", symbol="XAUUSD", price=2350.5, lots=0.01)
    assert payload["symbol"] == "XAUUSD"
    assert payload["price"] == pytest.approx(2350.5)
    assert payload["lots"] == pytest.approx(0.01)


def test_log_event_line_ends_with_newline():
    """Each log line must end with a newline for stream concatenation."""
    buf = io.StringIO()
    log_event("INFO", "newline_check", stream=buf)
    assert buf.getvalue().endswith("\n")
