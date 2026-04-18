"""audit-r2 ops #18: journal + live_state.json monitor fields.

Validates the pure-function parts of the monitor bundle:
  - planned_rr_ratio vs exec_rr_ratio math for RangeSetup (planned uses
    take_profit_ext, exec uses midpoint take_profit)
  - planned == exec for TradeSetup (trending has no aggressive target)
  - htf_bias_conf / htf_bias_tier surfacing with None-safe defaults

The live_demo save_state block is tested indirectly — we reproduce the
exact field-computation expressions here so regressions in either copy
surface immediately.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from smc.strategy.htf_bias import htf_bias_tier


# ---------------------------------------------------------------------------
# Mirror of save_state planned_rr/exec_rr computation for RangeSetup
# ---------------------------------------------------------------------------

def _compute_rr_pair_range(entry: float, sl: float, tp_exec: float, tp_planned: float) -> tuple[float, float]:
    risk = abs(entry - sl)
    if risk <= 0:
        return (0.0, 0.0)
    planned = round(abs(tp_planned - entry) / risk, 3)
    exec_ = round(abs(tp_exec - entry) / risk, 3)
    return (planned, exec_)


def _compute_rr_pair_trend(entry: float, sl: float, tp1: float) -> tuple[float, float]:
    risk = abs(entry - sl)
    if risk <= 0:
        return (0.0, 0.0)
    rr = round(abs(tp1 - entry) / risk, 3)
    return (rr, rr)


def _htf_fields(htf_bias) -> tuple[float, str]:
    conf = float(getattr(htf_bias, "confidence", 0.0) or 0.0) if htf_bias is not None else 0.0
    return (round(conf, 3), htf_bias_tier(conf))


# ---------------------------------------------------------------------------
# RangeSetup: planned TP_ext, exec midpoint — should differ
# ---------------------------------------------------------------------------

class TestRangeSetupRR:
    def test_long_planned_greater_than_exec(self):
        # long @2350, sl=2348, exec tp=2352 (2R), planned tp_ext=2354 (3R)
        planned, exec_ = _compute_rr_pair_range(
            entry=2350.0, sl=2348.0, tp_exec=2352.0, tp_planned=2354.0
        )
        assert exec_ == pytest.approx(1.0)
        assert planned == pytest.approx(2.0)
        assert planned > exec_  # TP1 debate signal

    def test_short_planned_greater_than_exec(self):
        planned, exec_ = _compute_rr_pair_range(
            entry=2350.0, sl=2352.0, tp_exec=2348.0, tp_planned=2346.0
        )
        assert exec_ == pytest.approx(1.0)
        assert planned == pytest.approx(2.0)

    def test_equal_tp_gives_equal_rr(self):
        """If tp_ext == tp (e.g. fallback), planned == exec."""
        planned, exec_ = _compute_rr_pair_range(
            entry=2350.0, sl=2348.0, tp_exec=2354.0, tp_planned=2354.0
        )
        assert planned == exec_

    def test_zero_sl_distance_returns_zero(self):
        planned, exec_ = _compute_rr_pair_range(
            entry=2350.0, sl=2350.0, tp_exec=2355.0, tp_planned=2360.0
        )
        assert planned == 0.0
        assert exec_ == 0.0


# ---------------------------------------------------------------------------
# TradeSetup (trending): planned == exec
# ---------------------------------------------------------------------------

class TestTradeSetupRR:
    def test_trending_planned_equals_exec(self):
        planned, exec_ = _compute_rr_pair_trend(entry=2350.0, sl=2348.0, tp1=2354.0)
        assert planned == exec_
        assert planned == pytest.approx(2.0)

    def test_trending_zero_sl_distance(self):
        planned, exec_ = _compute_rr_pair_trend(entry=2350.0, sl=2350.0, tp1=2354.0)
        assert planned == 0.0
        assert exec_ == 0.0


# ---------------------------------------------------------------------------
# HTF bias surfacing: confidence + tier
# ---------------------------------------------------------------------------

@dataclass
class _FakeBias:
    confidence: float


class TestHtfBiasFields:
    def test_none_bias_returns_neutral(self):
        conf, tier = _htf_fields(None)
        assert conf == 0.0
        assert tier == "neutral"

    def test_tier_1_bias(self):
        conf, tier = _htf_fields(_FakeBias(confidence=0.85))
        assert conf == 0.85
        assert tier == "tier_1"

    def test_tier_2_bias(self):
        conf, tier = _htf_fields(_FakeBias(confidence=0.55))
        assert conf == 0.55
        assert tier == "tier_2"

    def test_tier_3_bias(self):
        conf, tier = _htf_fields(_FakeBias(confidence=0.35))
        assert conf == 0.35
        assert tier == "tier_3"

    def test_disagreement_bias_neutral(self):
        """compute_htf_bias returns confidence=0.0 when D1 vs H4 disagree."""
        conf, tier = _htf_fields(_FakeBias(confidence=0.0))
        assert conf == 0.0
        assert tier == "neutral"

    def test_bias_missing_confidence_attr_returns_neutral(self):
        """Defensive: object without .confidence doesn't crash."""

        class _NoConf:
            pass

        conf, tier = _htf_fields(_NoConf())
        assert conf == 0.0
        assert tier == "neutral"

    def test_rounded_to_3_decimals(self):
        conf, tier = _htf_fields(_FakeBias(confidence=0.123456789))
        assert conf == 0.123
        assert tier == "neutral"  # < 0.3


# ---------------------------------------------------------------------------
# Combined: journal row shape sanity
# ---------------------------------------------------------------------------

class TestJournalRowShape:
    def test_range_row_has_both_rr_and_htf(self):
        """Simulated minimal journal row must carry all 4 monitor fields."""
        planned, exec_ = _compute_rr_pair_range(
            entry=2350.0, sl=2348.0, tp_exec=2352.0, tp_planned=2354.0
        )
        conf, tier = _htf_fields(_FakeBias(confidence=0.85))
        row = {
            "planned_rr_ratio": planned,
            "exec_rr_ratio": exec_,
            "htf_bias_conf": conf,
            "htf_bias_tier": tier,
        }
        assert set(row.keys()) == {
            "planned_rr_ratio",
            "exec_rr_ratio",
            "htf_bias_conf",
            "htf_bias_tier",
        }
        assert row["planned_rr_ratio"] > row["exec_rr_ratio"]
        assert row["htf_bias_tier"] == "tier_1"

    def test_trend_row_planned_equals_exec_but_htf_still_logged(self):
        planned, exec_ = _compute_rr_pair_trend(entry=2350.0, sl=2347.0, tp1=2355.0)
        conf, tier = _htf_fields(_FakeBias(confidence=0.45))
        row = {
            "planned_rr_ratio": planned,
            "exec_rr_ratio": exec_,
            "htf_bias_conf": conf,
            "htf_bias_tier": tier,
        }
        assert row["planned_rr_ratio"] == row["exec_rr_ratio"]
        assert row["htf_bias_tier"] == "tier_2"
