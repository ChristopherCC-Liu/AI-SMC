"""Statistical promotion gate for A/B backtest harnesses (R10 P4.2).

When a backtest produces baseline + treatment arms, this module decides
whether the treatment is statistically promotable to live based on:

PROMOTE iff
    n_baseline >= min_n  AND  n_treatment >= min_n
    AND  PF_diff_ci 95% lower_bound > 0  (paired-resample bootstrap)

Chi-square p-value on win-rate is included in the verdict for operator
visibility (rationale string) but does NOT participate in the PROMOTE
decision — PF and WR measure different things, and a treatment with
materially better PF can have a non-significant WR shift if the edge
comes from larger wins / smaller losses rather than win frequency.

All public functions are pure: no IO, no global state. Bootstrap RNG
takes an explicit ``rng_seed`` so verdicts are reproducible across
machines and across pytest runs.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.stats import chi2_contingency

__all__ = [
    "PromotionVerdict",
    "TradeLike",
    "bootstrap_pf_diff_ci",
    "chi_square_winrate",
    "evaluate_promotion",
]


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class TradeLike(Protocol):
    """Minimal duck for trade records — only ``pnl_usd`` is read."""

    pnl_usd: float


@dataclass(frozen=True)
class PromotionVerdict:
    """Output of :func:`evaluate_promotion`.

    All numeric fields are rounded for stable rationale string formatting;
    raw bootstrap CI bounds are preserved at full precision for downstream
    consumers (markdown report, logging) that want them.
    """

    n_baseline: int
    n_treatment: int
    pf_baseline: float
    pf_treatment: float
    pf_diff: float
    pf_diff_ci: tuple[float, float]
    wr_baseline: float
    wr_treatment: float
    wr_chi2_p: float
    promote: bool
    rationale: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _profit_factor(pnl: Sequence[float]) -> float:
    """PF = sum(wins) / |sum(losses)|.

    Returns float('inf') when there are wins but no losses (mathematically
    undefined). Returns 0.0 when there are no wins. Empty input → 0.0.
    """
    if not pnl:
        return 0.0
    wins = sum(p for p in pnl if p > 0)
    losses = abs(sum(p for p in pnl if p < 0))
    if losses == 0.0:
        return float("inf") if wins > 0 else 0.0
    return wins / losses


def _win_rate(pnl: Sequence[float]) -> float:
    if not pnl:
        return 0.0
    wins = sum(1 for p in pnl if p > 0)
    return wins / len(pnl)


def _wins_count(pnl: Sequence[float]) -> int:
    return sum(1 for p in pnl if p > 0)


# ---------------------------------------------------------------------------
# Public statistical primitives
# ---------------------------------------------------------------------------


def bootstrap_pf_diff_ci(
    pnl_baseline: Sequence[float],
    pnl_treatment: Sequence[float],
    *,
    n_iter: int = 10_000,
    confidence: float = 0.95,
    rng_seed: int = 0,
) -> tuple[float, float]:
    """Resample-based confidence interval on (PF_treatment - PF_baseline).

    Each iteration draws |arm| samples WITH REPLACEMENT from each arm
    independently, computes PF on the resamples, and records the diff.
    The (1-confidence)/2 and 1-(1-confidence)/2 quantiles of the diff
    distribution form the CI bounds.

    Reproducible via ``rng_seed``. Handles PF=inf gracefully — any
    iteration that produces inf in either arm is replaced with NaN and
    excluded from the quantile computation. If too many iterations
    yield inf the CI may be degenerate; callers should handle inf in
    ``evaluate_promotion``.
    """
    if not pnl_baseline or not pnl_treatment:
        return (0.0, 0.0)
    rng = np.random.default_rng(rng_seed)
    base_arr = np.asarray(pnl_baseline, dtype=float)
    treat_arr = np.asarray(pnl_treatment, dtype=float)
    n_b = len(base_arr)
    n_t = len(treat_arr)

    diffs = np.empty(n_iter, dtype=float)
    for i in range(n_iter):
        b_resample = base_arr[rng.integers(0, n_b, size=n_b)]
        t_resample = treat_arr[rng.integers(0, n_t, size=n_t)]
        pf_b = _profit_factor(list(b_resample))
        pf_t = _profit_factor(list(t_resample))
        if not np.isfinite(pf_b) or not np.isfinite(pf_t):
            diffs[i] = np.nan
        else:
            diffs[i] = pf_t - pf_b

    finite = diffs[np.isfinite(diffs)]
    if len(finite) == 0:
        return (0.0, 0.0)
    alpha = 1.0 - confidence
    lo = float(np.quantile(finite, alpha / 2.0))
    hi = float(np.quantile(finite, 1.0 - alpha / 2.0))
    return (lo, hi)


def chi_square_winrate(
    *,
    wins_baseline: int,
    n_baseline: int,
    wins_treatment: int,
    n_treatment: int,
) -> float:
    """Two-by-two chi-square contingency test on win counts.

    Returns the p-value. Returns 1.0 (no signal) when either arm is empty
    or the contingency table is degenerate, so callers can treat
    "insufficient data" the same as "no significant difference".
    """
    if n_baseline == 0 or n_treatment == 0:
        return 1.0
    losses_baseline = n_baseline - wins_baseline
    losses_treatment = n_treatment - wins_treatment
    table = [
        [wins_baseline, losses_baseline],
        [wins_treatment, losses_treatment],
    ]
    # If a whole row or column is zero, scipy raises — return 1.0 for safety.
    try:
        _, p, _, _ = chi2_contingency(table)
    except ValueError:
        return 1.0
    return float(p)


# ---------------------------------------------------------------------------
# Top-level decision function
# ---------------------------------------------------------------------------


def evaluate_promotion(
    baseline_trades: Iterable[TradeLike],
    treatment_trades: Iterable[TradeLike],
    *,
    min_n: int = 30,
    confidence: float = 0.95,
    n_iter: int = 10_000,
    rng_seed: int = 0,
) -> PromotionVerdict:
    """Decide whether ``treatment`` should be promoted over ``baseline``.

    Rule (per R10 P4.2):
    - PROMOTE iff n_baseline >= min_n AND n_treatment >= min_n
                  AND PF_diff_ci 95% lower_bound > 0
    - chi2_p is informational only, included in rationale.

    Verdict's ``rationale`` is plain ASCII so VPS Windows console
    encodes it without surprises.
    """
    baseline_pnl = [float(t.pnl_usd) for t in baseline_trades]
    treatment_pnl = [float(t.pnl_usd) for t in treatment_trades]

    n_b = len(baseline_pnl)
    n_t = len(treatment_pnl)

    pf_b = _profit_factor(baseline_pnl)
    pf_t = _profit_factor(treatment_pnl)
    wr_b = _win_rate(baseline_pnl)
    wr_t = _win_rate(treatment_pnl)
    wins_b = _wins_count(baseline_pnl)
    wins_t = _wins_count(treatment_pnl)

    chi2_p = chi_square_winrate(
        wins_baseline=wins_b,
        n_baseline=n_b,
        wins_treatment=wins_t,
        n_treatment=n_t,
    )

    pf_diff = (pf_t - pf_b) if (np.isfinite(pf_b) and np.isfinite(pf_t)) else float("nan")

    # Sample-size gate first — short-circuit before the (expensive) bootstrap.
    if n_b < min_n or n_t < min_n:
        rationale = (
            f"HOLD: insufficient sample (n_baseline={n_b}, n_treatment={n_t}, "
            f"min_n={min_n}). PF_b={_pf_str(pf_b)}, PF_t={_pf_str(pf_t)}, "
            f"WR_chi2_p={chi2_p:.3f} (informational)."
        )
        return PromotionVerdict(
            n_baseline=n_b,
            n_treatment=n_t,
            pf_baseline=round(pf_b, 4) if np.isfinite(pf_b) else pf_b,
            pf_treatment=round(pf_t, 4) if np.isfinite(pf_t) else pf_t,
            pf_diff=round(pf_diff, 4) if np.isfinite(pf_diff) else pf_diff,
            pf_diff_ci=(0.0, 0.0),
            wr_baseline=round(wr_b, 4),
            wr_treatment=round(wr_t, 4),
            wr_chi2_p=round(chi2_p, 6),
            promote=False,
            rationale=rationale,
        )

    # Undefined PF in either arm → cannot bootstrap reliably.
    if not np.isfinite(pf_b) or not np.isfinite(pf_t):
        rationale = (
            f"HOLD: PF undefined in at least one arm "
            f"(PF_b={_pf_str(pf_b)}, PF_t={_pf_str(pf_t)}). Bootstrap CI "
            f"cannot be computed reliably. WR_chi2_p={chi2_p:.3f} (informational)."
        )
        return PromotionVerdict(
            n_baseline=n_b,
            n_treatment=n_t,
            pf_baseline=pf_b,
            pf_treatment=pf_t,
            pf_diff=pf_diff,
            pf_diff_ci=(0.0, 0.0),
            wr_baseline=round(wr_b, 4),
            wr_treatment=round(wr_t, 4),
            wr_chi2_p=round(chi2_p, 6),
            promote=False,
            rationale=rationale,
        )

    ci_lo, ci_hi = bootstrap_pf_diff_ci(
        baseline_pnl,
        treatment_pnl,
        n_iter=n_iter,
        confidence=confidence,
        rng_seed=rng_seed,
    )
    promote = ci_lo > 0.0

    if promote:
        rationale = (
            f"PROMOTE: PF_diff={pf_diff:+.3f} 95%CI=[{ci_lo:+.3f}, {ci_hi:+.3f}] "
            f"excludes 0; n_baseline={n_b}, n_treatment={n_t} both >= min_n={min_n}. "
            f"WR: baseline={wr_b:.1%} treatment={wr_t:.1%} chi2_p={chi2_p:.3f} (informational)."
        )
    else:
        rationale = (
            f"HOLD: PF_diff={pf_diff:+.3f} 95%CI=[{ci_lo:+.3f}, {ci_hi:+.3f}] "
            f"crosses or sits below 0 (n_baseline={n_b}, n_treatment={n_t}). "
            f"WR: baseline={wr_b:.1%} treatment={wr_t:.1%} chi2_p={chi2_p:.3f} (informational)."
        )

    return PromotionVerdict(
        n_baseline=n_b,
        n_treatment=n_t,
        pf_baseline=round(pf_b, 4),
        pf_treatment=round(pf_t, 4),
        pf_diff=round(pf_diff, 4),
        pf_diff_ci=(ci_lo, ci_hi),
        wr_baseline=round(wr_b, 4),
        wr_treatment=round(wr_t, 4),
        wr_chi2_p=round(chi2_p, 6),
        promote=promote,
        rationale=rationale,
    )


def _pf_str(pf: float) -> str:
    if not np.isfinite(pf):
        return "inf" if pf == float("inf") else "undefined"
    return f"{pf:.3f}"
