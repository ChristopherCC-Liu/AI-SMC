"""Ticket 4 Step 2 — window_coverage tests (XAUUSD-only).

Pinned guarantees:
  - CoverageThresholds defaults match RFC §1.2 floors.
  - WindowSpec dataclass is frozen.
  - check_window_coverage:
      * insufficient windows → ABSTAIN reason starts with
        `insufficient_xauusd_window_coverage:`
      * insufficient regime buckets → reason
        `insufficient_regime_bucket_coverage:`
      * candidate-affects-halt + zero halt windows → reason
        `halt_corpus_insufficient:`
      * windows below MIN_BAR_COUNT_PER_WINDOW → reason
        `window_too_thin:`
      * data_gap_above_floor in any window → reason
        `data_gap_in_window_<id>:`
      * NEVER emits the legacy `single_symbol shadow window` text.
  - load_gold_profile reads YAML read-only; missing file returns
    empty dict; runtime never writes the file.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from smc.hedgerock.evolution.window_coverage import (
    CoverageReport,
    CoverageThresholds,
    REGIME_BUCKETS,
    WindowSpec,
    check_window_coverage,
    load_gold_profile,
)


def _spec(window_id: str, *, start: str = "2024-01-01", end: str = "2024-04-01",
          bucket: str = "range_low_vol") -> WindowSpec:
    return WindowSpec(
        window_id=window_id,
        start=datetime.fromisoformat(start).replace(tzinfo=timezone.utc),
        end=datetime.fromisoformat(end).replace(tzinfo=timezone.utc),
        declared_regime_bucket=bucket,
    )


def _stats(window_id: str, *, n_bars: int = 1500, n_decided: int = 1000,
           n_trades: int = 5, max_h1_gap_bars: int = 0,
           halt_event_count: int = 0,
           observed_buckets: tuple[str, ...] = ("range_low_vol",)) -> dict:
    return {
        "window_id": window_id,
        "n_bars": n_bars,
        "n_decided_bars": n_decided,
        "n_trades": n_trades,
        "max_h1_gap_bars": max_h1_gap_bars,
        "halt_event_count": halt_event_count,
        "observed_buckets": tuple(observed_buckets),
    }


# ---------------------------------------------------------------------------
# 1. Defaults match RFC §1.2
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_thresholds_defaults_match_rfc() -> None:
    t = CoverageThresholds()
    assert t.min_windows >= 6
    assert t.min_bar_count_per_window >= 1000
    assert t.min_decided_bars_per_window >= 500
    assert t.min_trade_count_per_window >= 4
    assert t.min_regime_buckets_covered >= 4
    assert t.min_halt_event_windows >= 1
    assert t.max_closed_bar_gap_h1 >= 168


@pytest.mark.unit
def test_regime_buckets_set_matches_rfc_seven() -> None:
    expected = {
        "range_low_vol", "range_high_vol", "trend_up", "trend_down",
        "breakout", "news_crisis", "weekend_gap",
    }
    assert set(REGIME_BUCKETS) == expected


@pytest.mark.unit
def test_window_spec_is_frozen() -> None:
    s = _spec("y2024_q1")
    from dataclasses import FrozenInstanceError
    with pytest.raises(FrozenInstanceError):
        s.window_id = "tampered"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 2. Insufficient window count → ABSTAIN
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_insufficient_window_count_abstain() -> None:
    specs = [_spec("y2024_q1"), _spec("y2024_q2")]
    stats = [_stats("y2024_q1"), _stats("y2024_q2")]
    out = check_window_coverage(
        specs=specs, per_window_stats=stats,
        candidate_affects_halt_mode=False,
    )
    assert out.coverage_pass is False
    assert any("insufficient_xauusd_window_coverage" in r
               for r in out.shortfall_reasons)


@pytest.mark.unit
def test_does_not_emit_legacy_single_symbol_blocker() -> None:
    """Critical RFC v2 invariant: never use the v1-era reason."""
    specs = [_spec(f"w{i}") for i in range(2)]
    stats = [_stats(f"w{i}") for i in range(2)]
    out = check_window_coverage(
        specs=specs, per_window_stats=stats,
        candidate_affects_halt_mode=False,
    )
    for r in out.shortfall_reasons:
        assert "single_symbol" not in r
        assert "cross_symbol" not in r
        assert "single symbol" not in r.lower()


# ---------------------------------------------------------------------------
# 3. Insufficient regime buckets → ABSTAIN
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_insufficient_regime_buckets_abstain() -> None:
    """6 windows but all in only 2 buckets → still ABSTAIN."""
    specs = [_spec(f"w{i}", bucket="range_low_vol") for i in range(6)]
    stats = [_stats(f"w{i}", observed_buckets=("range_low_vol",))
             for i in range(6)]
    out = check_window_coverage(
        specs=specs, per_window_stats=stats,
        candidate_affects_halt_mode=False,
    )
    assert out.coverage_pass is False
    assert any("insufficient_regime_bucket_coverage" in r
               for r in out.shortfall_reasons)


# ---------------------------------------------------------------------------
# 4. Halt corpus insufficient when candidate affects halt mode
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_halt_corpus_insufficient_when_candidate_affects_halt() -> None:
    """6 windows × 4 buckets but ZERO halt-event windows → ABSTAIN
    only when candidate.affects_halt_mode == True."""
    specs = [_spec(f"w{i}", bucket=b) for i, b in enumerate(
        ("range_low_vol", "range_high_vol", "trend_up", "trend_down",
         "breakout", "weekend_gap"),
    )]
    stats = [_stats(f"w{i}", observed_buckets=(b,), halt_event_count=0)
             for i, b in enumerate(
                 ("range_low_vol", "range_high_vol", "trend_up",
                  "trend_down", "breakout", "weekend_gap"),
             )]
    out = check_window_coverage(
        specs=specs, per_window_stats=stats,
        candidate_affects_halt_mode=True,
    )
    assert out.coverage_pass is False
    assert any("halt_corpus_insufficient" in r for r in out.shortfall_reasons)


@pytest.mark.unit
def test_halt_corpus_irrelevant_when_candidate_does_not_affect_halt() -> None:
    """Same coverage; halt mode irrelevant for non-halt candidate."""
    specs = [_spec(f"w{i}", bucket=b) for i, b in enumerate(
        ("range_low_vol", "range_high_vol", "trend_up", "trend_down",
         "breakout", "weekend_gap"),
    )]
    stats = [_stats(f"w{i}", observed_buckets=(b,), halt_event_count=0)
             for i, b in enumerate(
                 ("range_low_vol", "range_high_vol", "trend_up",
                  "trend_down", "breakout", "weekend_gap"),
             )]
    out = check_window_coverage(
        specs=specs, per_window_stats=stats,
        candidate_affects_halt_mode=False,
    )
    # Should not produce halt_corpus_insufficient reason.
    assert all("halt_corpus_insufficient" not in r
               for r in out.shortfall_reasons)


# ---------------------------------------------------------------------------
# 5. Window too thin / data gap
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_window_too_thin_recorded() -> None:
    specs = [_spec(f"w{i}") for i in range(6)]
    stats = [_stats(f"w{i}", n_bars=200) for i in range(6)]  # all thin
    out = check_window_coverage(
        specs=specs, per_window_stats=stats,
        candidate_affects_halt_mode=False,
    )
    assert out.coverage_pass is False
    assert any("window_too_thin" in r for r in out.shortfall_reasons)


@pytest.mark.unit
def test_insufficient_trade_coverage_in_any_window_fails_coverage() -> None:
    """RFC v2 §1.2 — min_trade_count_per_window is a HARD floor.
    Any window below it adds an
    `insufficient_trade_coverage_in_window_<id>` reason; coverage_pass
    flips to False. Average-over-windows must NOT mask weak windows."""
    specs = [_spec(f"w{i}", bucket=b) for i, b in enumerate(
        ("range_low_vol", "range_high_vol", "trend_up", "trend_down",
         "breakout", "weekend_gap"),
    )]
    # 5 strong windows + 1 thin-trade window.
    stats = []
    for i, b in enumerate(
        ("range_low_vol", "range_high_vol", "trend_up", "trend_down",
         "breakout", "weekend_gap"),
    ):
        n_trades = 1 if i == 5 else 8   # last window thin
        stats.append(_stats(
            f"w{i}", observed_buckets=(b,), halt_event_count=2,
            n_trades=n_trades,
        ))
    out = check_window_coverage(
        specs=specs, per_window_stats=stats,
        candidate_affects_halt_mode=False,
    )
    assert out.coverage_pass is False, out.shortfall_reasons
    assert any("insufficient_trade_coverage_in_window_w5" in r
               for r in out.shortfall_reasons)
    # The window_id is also surfaced via the existing no_trade_windows
    # field for backward-compat reporting.
    assert "w5" in out.no_trade_windows


@pytest.mark.unit
def test_data_gap_in_window_recorded() -> None:
    specs = [_spec(f"w{i}") for i in range(6)]
    stats = [_stats(f"w{i}", max_h1_gap_bars=500)  # >> floor 168
             for i in range(6)]
    out = check_window_coverage(
        specs=specs, per_window_stats=stats,
        candidate_affects_halt_mode=False,
    )
    assert out.coverage_pass is False
    assert any("data_gap_in_window" in r for r in out.shortfall_reasons)


# ---------------------------------------------------------------------------
# 6. Clean coverage → coverage_pass True (synthetic)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_clean_coverage_passes_synthetic() -> None:
    buckets = ("range_low_vol", "range_high_vol", "trend_up", "trend_down",
               "breakout", "news_crisis")
    specs = [_spec(f"w{i}", bucket=b) for i, b in enumerate(buckets)]
    stats = [_stats(f"w{i}", observed_buckets=(b,), halt_event_count=2,
                    n_bars=2000, n_decided=1500, n_trades=10,
                    max_h1_gap_bars=0)
             for i, b in enumerate(buckets)]
    out = check_window_coverage(
        specs=specs, per_window_stats=stats,
        candidate_affects_halt_mode=False,
    )
    assert out.coverage_pass is True, f"reasons: {out.shortfall_reasons}"
    assert tuple(sorted(out.regime_buckets_covered)) == tuple(sorted(buckets))
    assert out.halt_event_windows == 6


# ---------------------------------------------------------------------------
# 7. load_gold_profile — read-only YAML
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_load_gold_profile_missing_file_returns_empty(tmp_path: Path) -> None:
    p = tmp_path / "no_such_file.yaml"
    out = load_gold_profile(p)
    assert out == {}


@pytest.mark.unit
def test_load_gold_profile_reads_known_fields(tmp_path: Path) -> None:
    p = tmp_path / "gp.yaml"
    p.write_text("""
version: v0
operator_curated: true
windows_to_buckets:
  y2024_q1:
    - range_low_vol
  y2024_q3:
    - trend_up
    - news_crisis
""")
    out = load_gold_profile(p)
    assert out["version"] == "v0"
    assert out["operator_curated"] is True
    assert out["windows_to_buckets"]["y2024_q3"] == ["trend_up", "news_crisis"]


@pytest.mark.unit
def test_load_gold_profile_does_not_write(tmp_path: Path) -> None:
    """Read-only — calling load_gold_profile must never create the file."""
    p = tmp_path / "absent.yaml"
    load_gold_profile(p)
    assert not p.exists()
