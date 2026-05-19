"""Unit tests for the pure metrics layer of ``scripts/ab_test_monitor.py``."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import ab_test_monitor as mod  # noqa: E402  (path adjustment above)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _deal(
    *,
    ticket: int = 1,
    entry: int = 1,
    profit: float = 0.0,
    swap: float = 0.0,
    commission: float = 0.0,
    magic: int = 30333333,
    position_id: int = 1,
) -> mod.Deal:
    return mod.Deal(
        ticket=ticket,
        time_utc="2026-05-20 00:00:00",
        entry=entry,
        volume=0.01,
        price=2400.0,
        profit=profit,
        swap=swap,
        commission=commission,
        position_id=position_id,
        magic=magic,
    )


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_compute_metrics_empty() -> None:
    m = mod.compute_metrics([], magic=42)
    assert m.magic == 42
    assert m.n_deals == 0
    assert m.n_closes == 0
    assert m.wins == 0
    assert m.losses == 0
    assert m.win_rate == 0.0
    assert m.net_pnl == 0.0
    assert m.max_drawdown == 0.0


@pytest.mark.unit
def test_compute_metrics_only_entries_no_closes() -> None:
    # entry=0 means DEAL_ENTRY_IN — should not count toward closes
    deals = [_deal(entry=0, profit=99.0) for _ in range(3)]
    m = mod.compute_metrics(deals, magic=42)
    assert m.n_deals == 3
    assert m.n_closes == 0
    assert m.net_pnl == 0.0
    assert m.win_rate == 0.0


@pytest.mark.unit
def test_compute_metrics_win_rate_and_pf() -> None:
    deals = [
        _deal(ticket=1, entry=1, profit=10.0),
        _deal(ticket=2, entry=1, profit=20.0),
        _deal(ticket=3, entry=1, profit=-5.0),
    ]
    m = mod.compute_metrics(deals, magic=99)
    assert m.wins == 2
    assert m.losses == 1
    assert m.win_rate == pytest.approx(2 / 3)
    assert m.gross_profit == pytest.approx(30.0)
    assert m.gross_loss == pytest.approx(-5.0)
    assert m.profit_factor == pytest.approx(6.0)
    assert m.net_pnl == pytest.approx(25.0)


@pytest.mark.unit
def test_compute_metrics_profit_factor_infinite_when_no_losses() -> None:
    deals = [_deal(entry=1, profit=10.0)]
    m = mod.compute_metrics(deals, magic=1)
    assert m.profit_factor == float("inf")


@pytest.mark.unit
def test_compute_metrics_includes_swap_and_commission() -> None:
    # net = profit + swap + commission = +5 - 1 - 2 = +2 → win
    deals = [_deal(entry=1, profit=5.0, swap=-1.0, commission=-2.0)]
    m = mod.compute_metrics(deals, magic=1)
    assert m.wins == 1
    assert m.net_pnl == pytest.approx(2.0)


@pytest.mark.unit
def test_compute_metrics_max_drawdown_tracks_peak_to_trough() -> None:
    deals = [
        _deal(ticket=1, entry=1, profit=10.0),   # cum=10  peak=10
        _deal(ticket=2, entry=1, profit=-3.0),   # cum=7   dd=3
        _deal(ticket=3, entry=1, profit=-5.0),   # cum=2   dd=8
        _deal(ticket=4, entry=1, profit=20.0),   # cum=22  peak=22
        _deal(ticket=5, entry=1, profit=-15.0),  # cum=7   dd=15
    ]
    m = mod.compute_metrics(deals, magic=1)
    assert m.max_drawdown == pytest.approx(15.0)


@pytest.mark.unit
def test_compute_metrics_zero_pnl_close_does_not_skew_win_rate() -> None:
    # A break-even close (BE stop, e.g.) should not be counted as win or loss
    deals = [
        _deal(ticket=1, entry=1, profit=10.0),
        _deal(ticket=2, entry=1, profit=0.0),
        _deal(ticket=3, entry=1, profit=-10.0),
    ]
    m = mod.compute_metrics(deals, magic=1)
    assert m.wins == 1
    assert m.losses == 1
    assert m.win_rate == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# parse_csv
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_parse_csv_missing_file(tmp_path: Path) -> None:
    assert mod.parse_csv(tmp_path / "missing.csv") == []


@pytest.mark.unit
def test_parse_csv_empty_file(tmp_path: Path) -> None:
    p = tmp_path / "empty.csv"
    p.write_text("")
    assert mod.parse_csv(p) == []


@pytest.mark.unit
def test_parse_csv_round_trip(tmp_path: Path) -> None:
    p = tmp_path / "ok.csv"
    p.write_text(
        "ticket,time_utc,symbol,type,entry,volume,price,profit,swap,"
        "commission,magic,position_id,deal_reason\n"
        "1001,2026-05-20 12:00:00,XAUUSD,1,1,0.01,2400.5,12.34,0,-0.10,"
        "30333333,5001,0\n"
        "1002,2026-05-20 13:00:00,XAUUSD,0,0,0.01,2401.0,0,0,0,"
        "30333333,5002,0\n"
    )
    deals = mod.parse_csv(p)
    assert len(deals) == 2
    assert deals[0].ticket == 1001
    assert deals[0].profit == pytest.approx(12.34)
    assert deals[0].entry == 1
    assert deals[1].entry == 0
    assert deals[1].magic == 30333333


@pytest.mark.unit
def test_parse_csv_skips_malformed_rows(tmp_path: Path) -> None:
    p = tmp_path / "mixed.csv"
    p.write_text(
        "ticket,time_utc,symbol,type,entry,volume,price,profit,swap,"
        "commission,magic,position_id,deal_reason\n"
        "1001,t,XAUUSD,1,1,0.01,2400,5,0,0,1,1,0\n"
        "BAD ROW WITH WRONG SHAPE\n"
        "1002,t,XAUUSD,1,1,0.01,2400,-3,0,0,1,1,0\n"
    )
    deals = mod.parse_csv(p)
    assert [d.ticket for d in deals] == [1001, 1002]


# ---------------------------------------------------------------------------
# format_comparison
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_format_comparison_contains_both_labels_and_metrics() -> None:
    m_a = mod.compute_metrics(
        [_deal(entry=1, profit=10.0), _deal(ticket=2, entry=1, profit=-5.0)],
        magic=20222222,
    )
    m_b = mod.compute_metrics(
        [_deal(entry=1, profit=8.0, magic=30333333)], magic=30333333,
    )
    out = mod.format_comparison(m_a, m_b, label_a="HedgeRock_v2", label_b="HedgeRock_Lite")
    assert "HedgeRock_v2" in out
    assert "HedgeRock_Lite" in out
    assert "win-rate" in out
    assert "5.00" in out  # net PnL of A = +5
    assert "8.00" in out  # net PnL of B = +8


# ---------------------------------------------------------------------------
# remote_csv_path — structural smoke test (no I/O)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_remote_csv_path_format() -> None:
    p = mod.remote_csv_path("ABCDEF", 30333333)
    assert p.endswith(r"\MQL5\Files\ab_deals_30333333.csv")
    assert "ABCDEF" in p
