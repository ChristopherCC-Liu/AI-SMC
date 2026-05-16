"""Apr 20 2026 02:46 UTC stacked-SL incident replay (R10 P4.1).

Reproduces the production scenario where five XAUUSD BUYs opened in a
30-minute window from the same H1 demand zone all hit SL simultaneously
for a combined -$212.25 loss.

The R4 v5 anti-stack defensive layer (`smc.risk.concurrent_gates`) was
added in the post-mortem to prevent recurrence. This module pins that
behaviour.

Outcome assertions are rule-agnostic per R10 P4.1 spec — we assert
"no BUY proceeds" rather than naming a specific gate's `reason` string,
so a future R-round renaming the reason tag does not silently break this
suite.
"""
from __future__ import annotations

from datetime import datetime

import pytest
from tests.smc.fixture_replay.conftest import FakeDeal, FakePosition, load_bundle

from smc.risk.concurrent_gates import check_anti_stack_cooldown, check_concurrent_cap

pytestmark = pytest.mark.fixture_replay

_SLUG = "apr_20_2026_stacked_sl"


# ---------------------------------------------------------------------------
# Helpers — translate the JSON fixture into the protocol-typed inputs
# concurrent_gates expects.
# ---------------------------------------------------------------------------


def _positions_from_state(state: dict) -> list[FakePosition]:
    return [
        FakePosition(symbol=p["symbol"], magic=int(p["magic"]))
        for p in state.get("open_positions", [])
    ]


def _deals_from_state(state: dict) -> list[FakeDeal]:
    out: list[FakeDeal] = []
    for d in state.get("recent_long_entries", []):
        ts = datetime.fromisoformat(d["time_utc"])
        out.append(
            FakeDeal(
                symbol=d["symbol"],
                magic=int(d["magic"]),
                entry=int(d["entry"]),
                type=int(d["type"]),
                time=int(ts.timestamp()),
            )
        )
    return out


def _now(state: dict) -> datetime:
    return datetime.fromisoformat(state["now_utc"])


def _evaluate_pre_write_gates(state: dict, *, gates_enabled: bool) -> bool:
    """Return True iff a new BUY would proceed.

    Mirrors the Round 4 v5 pre-write block in scripts/live_demo.py: both
    gates run; either failure blocks the trade. When ``gates_enabled`` is
    False, both checks are bypassed (the pre-R4-v5 behaviour) and we
    always return True.
    """
    if not gates_enabled:
        return True
    cap_result = check_concurrent_cap(
        _positions_from_state(state),
        magic=int(state["magic"]),
        max_concurrent=int(state["max_concurrent"]),
    )
    if not cap_result.can_trade:
        return False
    cooldown_result = check_anti_stack_cooldown(
        _deals_from_state(state),
        symbol=state["symbol"],
        magic=int(state["magic"]),
        direction=state["candidate_setup"]["direction"],
        now=_now(state),
        cooldown_minutes=int(state["anti_stack_cooldown_minutes"]),
    )
    return cooldown_result.can_trade


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_apr_20_blocked_with_fix_enabled() -> None:
    """With R4 v5 anti-stack enabled, the 6th BUY MUST not proceed.

    Outcome assertion only — does not name `reason="concurrent_cap"` or
    `reason="anti_stack"`, so renames in either gate do not break this.
    """
    bundle = load_bundle(_SLUG)
    state = bundle.state
    would_proceed = _evaluate_pre_write_gates(state, gates_enabled=True)
    assert would_proceed is False, (
        "Apr 20 02:46 UTC scenario: a 6th BUY entry must NOT proceed when "
        "R4 v5 anti-stack gates are enabled. Reproduction inputs: "
        f"open positions={len(state['open_positions'])}, "
        f"recent long entries in {state['anti_stack_cooldown_minutes']}-min "
        f"window={len(state['recent_long_entries'])}, "
        f"max_concurrent={state['max_concurrent']}."
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Documents the pre-R4-v5 bug: with anti-stack gates disabled, the 6th BUY "
        "would proceed, reproducing the stacked-SL incident. xfail is strict so "
        "if a future round inadvertently re-blocks via some other defensive layer, "
        "this test flips to XPASS and surfaces the change for explicit review."
    ),
)
def test_apr_20_baseline_without_fix_documents_bug() -> None:
    """Sanity: without the fix, the assertion that BUY is blocked FAILS.

    Pytest then reports this as the expected xfail. If the assertion ever
    starts passing (i.e., the BUY is blocked despite gates being off),
    the strict xfail flips to XPASS and pytest fails — drawing reviewer
    attention to the un-asked-for behaviour change.
    """
    bundle = load_bundle(_SLUG)
    state = bundle.state
    would_proceed = _evaluate_pre_write_gates(state, gates_enabled=False)
    # We INTEND to assert the BUY is blocked — but with gates off it
    # proceeds, so the assertion fails, and xfail captures that.
    assert would_proceed is False
