"""Unit tests for ``smc.eval.promotion_gate`` (R10 P4.2).

Gate rule (per team-lead [DECISION]):
- PROMOTE iff (n_baseline >= 30 AND n_treatment >= 30 AND
              PF_diff_ci 95% lower_bound > 0)
- chi-square p-value is INFORMATIONAL ONLY — included in the verdict's
  rationale string but does NOT participate in the PROMOTE/HOLD decision.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from smc.eval.promotion_gate import (
    bootstrap_pf_diff_ci,
    chi_square_winrate,
    evaluate_promotion,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers — synthetic trade dataclass that satisfies the duck-typed protocol
# the gate consumes (just needs ``.pnl_usd``).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FakeTrade:
    """Minimal trade duck — only ``pnl_usd`` is read by the gate."""

    pnl_usd: float


def _seeded_pnl_series(
    n: int, *, win_rate: float, win_size: float, loss_size: float, seed: int
) -> list[float]:
    """Generate a reproducible synthetic PnL series."""
    rng = np.random.default_rng(seed)
    wins = rng.random(n) < win_rate
    return [
        float(win_size if w else -loss_size) for w in wins
    ]


# ---------------------------------------------------------------------------
# bootstrap_pf_diff_ci — direct unit tests
# ---------------------------------------------------------------------------


def test_bootstrap_ci_zero_diff_overlaps_zero() -> None:
    """When both arms have identical pnl distributions the CI must straddle 0."""
    pnl = _seeded_pnl_series(n=200, win_rate=0.55, win_size=10.0, loss_size=8.0, seed=1)
    lo, hi = bootstrap_pf_diff_ci(pnl, pnl, n_iter=2000, rng_seed=42)
    assert lo <= 0.0 <= hi, (
        f"Identical inputs should produce a CI that brackets zero — got [{lo}, {hi}]"
    )


def test_bootstrap_ci_strong_treatment_excludes_zero() -> None:
    """When treatment dominates, the CI lower bound must be > 0."""
    baseline = _seeded_pnl_series(n=200, win_rate=0.50, win_size=10.0, loss_size=10.0, seed=2)
    treatment = _seeded_pnl_series(n=200, win_rate=0.65, win_size=15.0, loss_size=8.0, seed=3)
    lo, hi = bootstrap_pf_diff_ci(baseline, treatment, n_iter=2000, rng_seed=42)
    assert lo > 0.0, (
        f"Strong treatment effect should give CI lower bound > 0 — got [{lo}, {hi}]"
    )


def test_seed_determinism() -> None:
    """Same inputs + same seed → identical CI bounds across runs."""
    baseline = _seeded_pnl_series(n=100, win_rate=0.5, win_size=10.0, loss_size=10.0, seed=4)
    treatment = _seeded_pnl_series(n=100, win_rate=0.55, win_size=11.0, loss_size=9.0, seed=5)
    ci_a = bootstrap_pf_diff_ci(baseline, treatment, n_iter=1000, rng_seed=99)
    ci_b = bootstrap_pf_diff_ci(baseline, treatment, n_iter=1000, rng_seed=99)
    assert ci_a == ci_b, f"deterministic seed should give identical CIs — {ci_a} vs {ci_b}"


# ---------------------------------------------------------------------------
# chi_square_winrate — direct unit tests
# ---------------------------------------------------------------------------


def test_chi_square_extreme_imbalance() -> None:
    """50% vs 90% win rates with n=100 each → p-value near 0."""
    p = chi_square_winrate(
        wins_baseline=50, n_baseline=100,
        wins_treatment=90, n_treatment=100,
    )
    assert p < 0.001, f"extreme WR difference should give p < 0.001, got {p}"


def test_chi_square_zero_trades_in_one_arm_returns_nan_or_one() -> None:
    """Degenerate case: an arm has 0 trades — chi-square is undefined.

    Implementation must handle this without raising; conventional answer is
    p=1.0 (no signal) so downstream code treats it as "no evidence either way".
    """
    p = chi_square_winrate(
        wins_baseline=10, n_baseline=20,
        wins_treatment=0, n_treatment=0,
    )
    assert p == 1.0 or np.isnan(p), (
        f"degenerate case should return p=1.0 or NaN (no signal), got {p}"
    )


# ---------------------------------------------------------------------------
# evaluate_promotion — full verdict integration
# ---------------------------------------------------------------------------


def test_promote_with_strong_signal_n_30() -> None:
    """n=30 each + clear treatment edge → PROMOTE."""
    baseline = [FakeTrade(pnl) for pnl in _seeded_pnl_series(
        n=30, win_rate=0.50, win_size=10.0, loss_size=10.0, seed=10,
    )]
    treatment = [FakeTrade(pnl) for pnl in _seeded_pnl_series(
        n=30, win_rate=0.75, win_size=20.0, loss_size=8.0, seed=11,
    )]
    verdict = evaluate_promotion(baseline, treatment, n_iter=2000, rng_seed=42)
    assert verdict.promote is True, (
        f"strong-signal n=30 should PROMOTE — verdict={verdict}"
    )
    assert verdict.pf_diff_ci[0] > 0.0


def test_hold_under_min_n() -> None:
    """n=10 each + strong effect → HOLD due to insufficient sample."""
    baseline = [FakeTrade(pnl) for pnl in _seeded_pnl_series(
        n=10, win_rate=0.50, win_size=10.0, loss_size=10.0, seed=20,
    )]
    treatment = [FakeTrade(pnl) for pnl in _seeded_pnl_series(
        n=10, win_rate=0.90, win_size=30.0, loss_size=5.0, seed=21,
    )]
    verdict = evaluate_promotion(baseline, treatment, n_iter=1000, rng_seed=42, min_n=30)
    assert verdict.promote is False
    assert "min_n" in verdict.rationale.lower() or "sample" in verdict.rationale.lower()


def test_hold_when_ci_crosses_zero() -> None:
    """n=100 each with tiny effect → CI crosses zero → HOLD."""
    # Identical distributions so the CI is centered on 0
    baseline = [FakeTrade(pnl) for pnl in _seeded_pnl_series(
        n=100, win_rate=0.55, win_size=10.0, loss_size=8.0, seed=30,
    )]
    treatment = [FakeTrade(pnl) for pnl in _seeded_pnl_series(
        n=100, win_rate=0.55, win_size=10.0, loss_size=8.0, seed=31,
    )]
    verdict = evaluate_promotion(baseline, treatment, n_iter=2000, rng_seed=42)
    assert verdict.promote is False
    assert verdict.pf_diff_ci[0] <= 0.0 <= verdict.pf_diff_ci[1]


def test_pf_inf_handling() -> None:
    """Treatment arm has zero losses → PF=inf; bootstrap must not crash;
    verdict must HOLD with explicit rationale flagging the undefined PF."""
    baseline = [FakeTrade(pnl) for pnl in _seeded_pnl_series(
        n=50, win_rate=0.5, win_size=10.0, loss_size=10.0, seed=40,
    )]
    # Treatment: all wins → no losses → PF undefined (inf)
    treatment = [FakeTrade(pnl_usd=10.0) for _ in range(50)]
    verdict = evaluate_promotion(baseline, treatment, n_iter=1000, rng_seed=42)
    assert verdict.promote is False
    assert "undefined" in verdict.rationale.lower() or "inf" in verdict.rationale.lower()


def test_protocol_decoupling() -> None:
    """Any object with a ``.pnl_usd`` attribute should work — not just FakeTrade."""

    @dataclass
    class OtherShape:
        pnl_usd: float
        # extra fields the gate must ignore
        ticket_id: str = "x"

    baseline = [OtherShape(pnl_usd=p) for p in _seeded_pnl_series(
        n=40, win_rate=0.5, win_size=10.0, loss_size=10.0, seed=50,
    )]
    treatment = [OtherShape(pnl_usd=p) for p in _seeded_pnl_series(
        n=40, win_rate=0.7, win_size=15.0, loss_size=8.0, seed=51,
    )]
    verdict = evaluate_promotion(baseline, treatment, n_iter=1000, rng_seed=42)
    # Just asserting it runs and produces a verdict — the structural duck
    # typing is what matters here.
    assert verdict.n_baseline == 40
    assert verdict.n_treatment == 40


def test_unicode_safe_rationale() -> None:
    """rationale must be ASCII so VPS Windows console doesn't choke on cp1252."""
    baseline = [FakeTrade(pnl) for pnl in _seeded_pnl_series(
        n=40, win_rate=0.5, win_size=10.0, loss_size=10.0, seed=60,
    )]
    treatment = [FakeTrade(pnl) for pnl in _seeded_pnl_series(
        n=40, win_rate=0.6, win_size=12.0, loss_size=9.0, seed=61,
    )]
    verdict = evaluate_promotion(baseline, treatment, n_iter=1000, rng_seed=42)
    # encode('ascii') raises UnicodeEncodeError if any non-ASCII chars present.
    verdict.rationale.encode("ascii")


def test_chi2_p_value_is_informational_only_does_not_block_promote() -> None:
    """Per team-lead [DECISION]: chi2_p high but PF CI lower > 0 still PROMOTEs.

    Construct a scenario where treatment PF is materially better than baseline
    (PF CI excludes 0) but win-rate counts are similar enough that chi2 p-value
    is non-significant (e.g., > 0.10). Gate must STILL PROMOTE.
    """
    # Baseline: 50 trades, 25 wins, small wins + small losses → PF ≈ 1
    baseline = [FakeTrade(pnl_usd=5.0) for _ in range(25)] + [
        FakeTrade(pnl_usd=-5.0) for _ in range(25)
    ]
    # Treatment: 50 trades, 26 wins (only +1), but BIG wins + tiny losses → PF ≈ 6
    # WR diff (52 vs 50) gives chi2_p > 0.5; PF diff is huge.
    treatment = [FakeTrade(pnl_usd=30.0) for _ in range(26)] + [
        FakeTrade(pnl_usd=-5.0) for _ in range(24)
    ]
    verdict = evaluate_promotion(baseline, treatment, n_iter=2000, rng_seed=42)
    assert verdict.promote is True, (
        f"PF-driven promote should not be blocked by non-significant chi2 — verdict={verdict}"
    )
    # Sanity: chi2 should indeed be non-significant (> 0.05) for this scenario
    assert verdict.wr_chi2_p > 0.05, (
        f"test setup error — chi2 p was unexpectedly small: {verdict.wr_chi2_p}"
    )


def test_evaluate_with_realistic_r7_data() -> None:
    """Golden test: synthesised data with a clear treatment edge → PROMOTE.

    Uses large n (300 per arm) and a substantial effect size so that the
    bootstrap CI clearly excludes 0. PF-on-deterministic-vectors has wider
    CIs than typical real-world data (no within-arm variance from the
    synthetic structure), so we deliberately choose parameters where the
    edge is large enough to PROMOTE under bootstrap.
    """
    # Baseline: 50% WR with even win/loss size → PF=1.0
    baseline = (
        [FakeTrade(pnl_usd=10.0) for _ in range(150)]
        + [FakeTrade(pnl_usd=-10.0) for _ in range(150)]
    )
    # Treatment: 60% WR with bigger wins / smaller losses → PF=180/120=1.5
    treatment = (
        [FakeTrade(pnl_usd=20.0) for _ in range(180)]
        + [FakeTrade(pnl_usd=-10.0) for _ in range(120)]
    )
    verdict = evaluate_promotion(baseline, treatment, n_iter=3000, rng_seed=42)
    assert verdict.n_baseline == 300
    assert verdict.n_treatment == 300
    assert verdict.promote is True, f"strong-effect n=300 should PROMOTE — {verdict}"
    assert verdict.pf_treatment > verdict.pf_baseline
    assert verdict.pf_diff > 0
    assert verdict.pf_diff_ci[0] > 0.0


def test_empty_inputs_returns_hold_verdict() -> None:
    """Both arms empty → must not raise; returns HOLD with low n."""
    verdict = evaluate_promotion([], [], n_iter=100, rng_seed=42)
    assert verdict.n_baseline == 0
    assert verdict.n_treatment == 0
    assert verdict.promote is False


def test_rationale_contains_chi2_p_value() -> None:
    """rationale string must include the chi2 p-value (informational disclosure)."""
    baseline = [FakeTrade(pnl) for pnl in _seeded_pnl_series(
        n=50, win_rate=0.5, win_size=10.0, loss_size=10.0, seed=70,
    )]
    treatment = [FakeTrade(pnl) for pnl in _seeded_pnl_series(
        n=50, win_rate=0.7, win_size=15.0, loss_size=8.0, seed=71,
    )]
    verdict = evaluate_promotion(baseline, treatment, n_iter=1000, rng_seed=42)
    assert "chi2" in verdict.rationale.lower() or "wr_p" in verdict.rationale.lower()
