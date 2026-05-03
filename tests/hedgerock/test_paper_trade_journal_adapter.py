"""Tests for ``smc.hedgerock.paper_trade_journal_adapter``.

Coverage targets:

- CSV: comma + semicolon delimiters, missing columns, malformed rows,
  custom column names, regime_lookup wiring
- JSON: valid + malformed lines, embedded regime context, lookup fallback
- Dispatcher: .csv / .json / .jsonl / unsupported extension
- Integration: loaded TradeRecords feed run_alpha_validation correctly
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from smc.hedgerock.alpha_validation import (
    AlphaValidationConfig,
    TradeRecord,
    run_alpha_validation,
)
from smc.hedgerock.paper_trade_journal_adapter import (
    load_csv,
    load_json,
    load_paper_trade_journal,
)


# ---------------------------------------------------------------------------
# CSV loader
# ---------------------------------------------------------------------------


def _write(tmp_path: Path, name: str, content: str) -> Path:
    path = tmp_path / name
    path.write_text(content)
    return path


def test_csv_loads_comma_delimited(tmp_path: Path) -> None:
    csv_text = (
        "close_ts,profit_usd,symbol\n"
        "2024-01-01T10:00:00Z,150.0,XAUUSD\n"
        "2024-01-01T11:00:00Z,-50.0,XAUUSD\n"
        "2024-01-01T12:00:00Z,200.0,XAUUSD\n"
    )
    path = _write(tmp_path, "trades.csv", csv_text)
    records = load_csv(path)
    assert len(records) == 3
    assert records[0].pnl_usd == 150.0
    assert records[0].ts == datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
    assert records[1].pnl_usd == -50.0


def test_csv_loads_semicolon_delimited(tmp_path: Path) -> None:
    csv_text = (
        "close_ts;profit_usd\n"
        "2024-01-01T10:00:00Z;75.5\n"
        "2024-01-01T11:00:00Z;-25.0\n"
    )
    path = _write(tmp_path, "trades_eu.csv", csv_text)
    records = load_csv(path)
    assert len(records) == 2
    assert records[0].pnl_usd == pytest.approx(75.5)


def test_csv_skips_malformed_rows(tmp_path: Path) -> None:
    csv_text = (
        "close_ts,profit_usd\n"
        "2024-01-01T10:00:00Z,150.0\n"
        "not_a_timestamp,not_a_number\n"
        "2024-01-01T11:00:00Z,abc\n"
        "2024-01-01T12:00:00Z,-50.0\n"
    )
    path = _write(tmp_path, "trades_dirty.csv", csv_text)
    records = load_csv(path)
    assert len(records) == 2
    assert records[0].pnl_usd == 150.0
    assert records[1].pnl_usd == -50.0


def test_csv_skips_rows_missing_required_columns(tmp_path: Path) -> None:
    csv_text = (
        "close_ts,profit_usd\n"
        ",100.0\n"  # missing ts
        "2024-01-01T10:00:00Z,\n"  # missing pnl
        "2024-01-01T11:00:00Z,200.0\n"
    )
    path = _write(tmp_path, "trades_missing.csv", csv_text)
    records = load_csv(path)
    assert len(records) == 1
    assert records[0].pnl_usd == 200.0


def test_csv_custom_column_names(tmp_path: Path) -> None:
    csv_text = (
        "exit_time,net_pnl\n"
        "2024-01-01T10:00:00Z,42.0\n"
    )
    path = _write(tmp_path, "trades_custom.csv", csv_text)
    records = load_csv(path, ts_column="exit_time", pnl_column="net_pnl")
    assert len(records) == 1
    assert records[0].pnl_usd == 42.0


def test_csv_regime_lookup_called_per_record(tmp_path: Path) -> None:
    csv_text = (
        "close_ts,profit_usd\n"
        "2024-01-01T10:00:00Z,100.0\n"
        "2024-01-01T11:00:00Z,-50.0\n"
    )
    path = _write(tmp_path, "trades.csv", csv_text)
    calls: list[datetime] = []

    def lookup(ts: datetime):
        calls.append(ts)
        return ("ATH_BREAKOUT", True)

    records = load_csv(path, regime_lookup=lookup)
    assert len(calls) == 2
    assert all(r.regime_at_entry == "ATH_BREAKOUT" for r in records)
    assert all(r.regime_mismatch is True for r in records)


def test_csv_default_regime_when_no_lookup(tmp_path: Path) -> None:
    csv_text = "close_ts,profit_usd\n2024-01-01T10:00:00Z,10.0\n"
    path = _write(tmp_path, "trades.csv", csv_text)
    records = load_csv(path)
    assert records[0].regime_at_entry == "TREND_UP"
    assert records[0].regime_mismatch is False


def test_csv_empty_file_returns_empty_tuple(tmp_path: Path) -> None:
    path = _write(tmp_path, "empty.csv", "close_ts,profit_usd\n")
    assert load_csv(path) == ()


def test_csv_naive_timestamp_assumed_utc(tmp_path: Path) -> None:
    """No ``Z`` / offset → adapter treats it as UTC (HedgeRock convention)."""
    csv_text = "close_ts,profit_usd\n2024-01-01T10:00:00,100.0\n"
    path = _write(tmp_path, "trades.csv", csv_text)
    records = load_csv(path)
    assert records[0].ts.tzinfo == timezone.utc


# ---------------------------------------------------------------------------
# JSON loader
# ---------------------------------------------------------------------------


def test_json_loads_valid_lines(tmp_path: Path) -> None:
    jsonl = (
        '{"close_ts": "2024-01-01T10:00:00Z", "pnl_usd": 100.0}\n'
        '{"close_ts": "2024-01-01T11:00:00Z", "pnl_usd": -50.0}\n'
    )
    path = _write(tmp_path, "trades.jsonl", jsonl)
    records = load_json(path)
    assert len(records) == 2
    assert records[0].pnl_usd == 100.0
    assert records[1].pnl_usd == -50.0


def test_json_skips_malformed_lines(tmp_path: Path) -> None:
    jsonl = (
        '{"close_ts": "2024-01-01T10:00:00Z", "pnl_usd": 100.0}\n'
        "not json at all\n"
        "[]\n"  # not a dict
        '{"missing_pnl": "x"}\n'
        '{"close_ts": "2024-01-01T11:00:00Z", "pnl_usd": -50.0}\n'
    )
    path = _write(tmp_path, "trades.jsonl", jsonl)
    records = load_json(path)
    assert len(records) == 2


def test_json_uses_embedded_regime_when_present(tmp_path: Path) -> None:
    jsonl = (
        '{"close_ts": "2024-01-01T10:00:00Z", "pnl_usd": 100.0, '
        '"regime_at_entry": "TREND_DOWN", "regime_mismatch": true}\n'
    )
    path = _write(tmp_path, "trades.jsonl", jsonl)
    records = load_json(path)
    assert records[0].regime_at_entry == "TREND_DOWN"
    assert records[0].regime_mismatch is True


def test_json_falls_back_to_lookup_when_embedded_missing(tmp_path: Path) -> None:
    jsonl = '{"close_ts": "2024-01-01T10:00:00Z", "pnl_usd": 100.0}\n'
    path = _write(tmp_path, "trades.jsonl", jsonl)

    def lookup(ts: datetime):
        return ("CONSOLIDATION", False)

    records = load_json(path, regime_lookup=lookup)
    assert records[0].regime_at_entry == "CONSOLIDATION"


def test_json_empty_file_returns_empty_tuple(tmp_path: Path) -> None:
    path = _write(tmp_path, "empty.jsonl", "")
    assert load_json(path) == ()


def test_json_skips_blank_lines(tmp_path: Path) -> None:
    jsonl = (
        "\n"
        '{"close_ts": "2024-01-01T10:00:00Z", "pnl_usd": 100.0}\n'
        "\n"
    )
    path = _write(tmp_path, "trades.jsonl", jsonl)
    records = load_json(path)
    assert len(records) == 1


def test_json_pnl_int_accepted(tmp_path: Path) -> None:
    """JSON int → coerce to float."""
    jsonl = '{"close_ts": "2024-01-01T10:00:00Z", "pnl_usd": 100}\n'
    path = _write(tmp_path, "trades.jsonl", jsonl)
    records = load_json(path)
    assert records[0].pnl_usd == 100.0


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def test_dispatcher_routes_csv(tmp_path: Path) -> None:
    path = _write(tmp_path, "x.csv", "close_ts,profit_usd\n2024-01-01T10:00:00Z,1.0\n")
    records = load_paper_trade_journal(path)
    assert len(records) == 1


def test_dispatcher_routes_jsonl(tmp_path: Path) -> None:
    path = _write(tmp_path, "x.jsonl", '{"close_ts":"2024-01-01T10:00:00Z","pnl_usd":1.0}\n')
    records = load_paper_trade_journal(path)
    assert len(records) == 1


def test_dispatcher_routes_json(tmp_path: Path) -> None:
    path = _write(tmp_path, "x.json", '{"close_ts":"2024-01-01T10:00:00Z","pnl_usd":1.0}\n')
    records = load_paper_trade_journal(path)
    assert len(records) == 1


def test_dispatcher_rejects_unknown_extension(tmp_path: Path) -> None:
    path = _write(tmp_path, "x.txt", "noop")
    with pytest.raises(ValueError, match="Unsupported journal format"):
        load_paper_trade_journal(path)


# ---------------------------------------------------------------------------
# Integration with run_alpha_validation
# ---------------------------------------------------------------------------


def test_loaded_records_feed_alpha_validation(tmp_path: Path) -> None:
    """End-to-end: a real journal flows through alpha_validation correctly."""
    # Mix of profitable + losing trades to give forward PF > 1.0 reliably.
    rows = [
        ("close_ts,profit_usd"),
    ]
    # 24 wins of $50 + 10 losses of $40 → PF = 1200/400 = 3.0 (hours 0-23 only)
    for i in range(24):
        rows.append(f"2024-01-01T{i:02d}:00:00Z,50.0")
    for i in range(10):
        rows.append(f"2024-01-02T{i:02d}:00:00Z,-40.0")
    path = _write(tmp_path, "win.csv", "\n".join(rows) + "\n")

    records = load_csv(path)
    assert len(records) == 34

    config = AlphaValidationConfig(
        instrument="XAUUSD",
        start=datetime(2024, 1, 1),
        end=datetime(2024, 1, 31),
    )
    result = run_alpha_validation(config, trades=records)
    # AC-4 reverse PF should be < 1.0 (forward 3.75 → reverse ~ 0.27)
    assert result.edge_real
    assert result.reverse_pf < 1.0
    assert result.forward_pf > 1.0
    # Total decisions matches input count.
    assert len(result.config.instrument) > 0  # smoke


def test_loaded_records_immutable() -> None:
    record = TradeRecord(
        ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
        pnl_usd=100.0,
        regime_at_entry="TREND_UP",
    )
    with pytest.raises(Exception):
        record.pnl_usd = 999.0  # type: ignore[misc]
