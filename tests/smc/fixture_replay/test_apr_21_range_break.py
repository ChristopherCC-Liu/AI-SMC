"""Apr 21 2026 09:47 UTC range-break BUY incident replay (R10 P4.1).

RangeTrader emitted a long support_bounce setup at $4779 while the D1
SMA50 slope was -0.057 %/bar and the close was 2.1% below SMA50 — both
clearly bearish. The trade SL'd within an hour. R9 P0-A added the
`_trend_filter_should_block` defensive layer in `range_trader.py` to
catch longs in materially down-trending markets.

This module pins that behaviour. Outcome assertions are rule-agnostic:
we assert the long is blocked, not which threshold caught it.
"""
from __future__ import annotations

import pytest
from tests.smc.fixture_replay.conftest import load_bundle

from smc.strategy.range_trader import _trend_filter_should_block

pytestmark = pytest.mark.fixture_replay

_SLUG = "apr_21_2026_range_break"


def _evaluate_trend_filter(state: dict, *, fix_enabled: bool) -> bool:
    """Return True iff a new long BUY would be BLOCKED by the trend filter.

    When ``fix_enabled`` is False the filter is bypassed (mirrors the
    pre-R9-P0-A behaviour); the long always proceeds, so we return False.
    """
    if not fix_enabled:
        # Pre-R9-P0-A: no trend filter ran, so nothing blocks.
        return False
    direction = state["candidate_setup"]["direction"]
    slope = float(state["trend_signals"]["slope_pct_per_bar"])
    close_vs_sma = float(state["trend_signals"]["close_vs_sma50_pct"])
    return _trend_filter_should_block(
        direction=direction,
        slope_pct=slope,
        close_vs_sma_pct=close_vs_sma,
    )


def test_apr_21_blocked_with_fix_enabled() -> None:
    """With R9 P0-A trend filter enabled, the long BUY MUST be blocked.

    Outcome assertion only — does not name the specific threshold that
    fired, so future renames or value tweaks don't break the suite.
    """
    bundle = load_bundle(_SLUG)
    state = bundle.state
    blocked = _evaluate_trend_filter(state, fix_enabled=True)
    signals = state["trend_signals"]
    assert blocked is True, (
        "Apr 21 09:47 UTC scenario: a long BUY must be blocked when D1 trend "
        "is materially down. Reproduction inputs: "
        f"slope={signals['slope_pct_per_bar']} %/bar, "
        f"close-vs-SMA50={signals['close_vs_sma50_pct']} %, "
        f"direction={state['candidate_setup']['direction']}."
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Documents the pre-R9-P0-A bug: with the trend filter disabled, the long "
        "BUY proceeds despite the bearish D1 trend, reproducing the range-break "
        "incident. Strict xfail flips to XPASS if a future round inadvertently "
        "re-blocks via some other defensive layer — surfacing the change for review."
    ),
)
def test_apr_21_baseline_without_fix_documents_bug() -> None:
    """Sanity: without the trend filter, the long is NOT blocked.

    The assertion that the long is blocked therefore fails; xfail captures
    the expected failure. If a future round causes this to start passing
    (long blocked by some other path), strict xfail flips to XPASS and
    pytest fails — surfacing the change for review.
    """
    bundle = load_bundle(_SLUG)
    state = bundle.state
    blocked = _evaluate_trend_filter(state, fix_enabled=False)
    # We INTEND to assert the long is blocked — without the filter, it isn't.
    assert blocked is True
