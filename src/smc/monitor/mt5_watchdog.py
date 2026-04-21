"""MT5 IPC handle auto-heal watchdog (Round 5 stability R1).

Root cause: ``mt5.symbol_info_tick(sym)`` silently returns ``None`` after
~17h of uptime on the VPS — the Python → MetaTrader5 IPC handle rots. The
process continues spinning; no orders flow; no reconciliation fires; BTC
went 17h dark before Telegram caught up.

This watchdog:

1. Tracks ``consecutive_tick_none`` per process (resets on any success).
2. After **3** consecutive failures, attempts ``mt5.shutdown()`` +
   ``mt5.initialize()`` once — logs ``mt5_handle_reset_attempt`` then
   either ``mt5_handle_reset_success`` (streak reset) or
   ``mt5_handle_reset_failed`` (streak continues climbing).
3. After **5** consecutive failures, logs ``mt5_handle_reset_giveup`` at
   CRIT severity and the caller ``sys.exit(1)``.  Windows Task Scheduler
   (or ``smc-live.bat``) respawns the process; fresh interpreter → fresh
   IPC handle.
4. A critical-section flag prevents resetting mid-``order_send`` — the
   cycle loop sets it around margin check + retry-send and clears on
   exit, so a tick_none from a post-send poll never yanks the handle
   while a ticket is in flight.

Design notes
------------
- State is kept as an immutable ``@dataclass(frozen=True)`` per project
  coding-style rules; every transition returns a new copy.
- Side-effects (``mt5.shutdown`` / ``mt5.initialize``) are isolated to
  :func:`try_reset_handle` which takes the ``mt5`` module by reference so
  unit tests can inject a ``Mock``.
- ``structured_log.crit`` / ``info`` are lazy-imported inside functions
  so the module is import-safe on machines without the logging config
  (tests construct state directly).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Any, Callable

DEFAULT_RESET_THRESHOLD = 3
DEFAULT_GIVEUP_THRESHOLD = 5


@dataclass(frozen=True)
class Mt5WatchdogState:
    """Immutable snapshot of MT5 handle health for the current process.

    Attributes
    ----------
    consecutive_tick_none:
        Number of consecutive ``symbol_info_tick → None`` outcomes since
        the last success.  Reset to 0 on any successful tick or any
        successful handle reinit.
    reset_attempts:
        Total handle-reset attempts over the process lifetime.  Diagnostic
        only — does not gate behaviour.
    last_init_monotonic:
        ``time.monotonic()`` at the last successful ``mt5.initialize()``.
        Used by the ``health_probe`` event in R2 to compute
        ``handle_age_sec``.  Defaults to 0.0 pre-init; the caller is
        expected to set this via :func:`mark_initialized` right after
        the very first ``mt5.initialize`` succeeds.
    in_critical_section:
        When True, :func:`should_reset` returns False so the watchdog
        never yanks the handle mid-``order_send``.  The cycle loop sets
        this around margin check + ``send_with_retry``.
    reset_threshold:
        Streak at which :func:`should_reset` returns True.
    giveup_threshold:
        Streak at which :func:`should_giveup` returns True.  The caller
        ``sys.exit(1)`` once this trips — Task Scheduler respawns.
    """

    consecutive_tick_none: int = 0
    reset_attempts: int = 0
    last_init_monotonic: float = 0.0
    in_critical_section: bool = False
    reset_threshold: int = DEFAULT_RESET_THRESHOLD
    giveup_threshold: int = DEFAULT_GIVEUP_THRESHOLD


def new_state(
    *,
    reset_threshold: int = DEFAULT_RESET_THRESHOLD,
    giveup_threshold: int = DEFAULT_GIVEUP_THRESHOLD,
) -> Mt5WatchdogState:
    """Return a fresh watchdog state with the given thresholds.

    Thresholds are configurable primarily for unit testing — production
    always uses the defaults (3 → reset, 5 → giveup).
    """
    if reset_threshold < 1:
        raise ValueError("reset_threshold must be >= 1")
    if giveup_threshold <= reset_threshold:
        raise ValueError("giveup_threshold must be > reset_threshold")
    return Mt5WatchdogState(
        reset_threshold=reset_threshold,
        giveup_threshold=giveup_threshold,
    )


def record_tick_result(state: Mt5WatchdogState, tick_ok: bool) -> Mt5WatchdogState:
    """Return a new state reflecting the tick result.

    ``tick_ok=True`` resets the streak; ``False`` increments it.
    """
    if tick_ok:
        if state.consecutive_tick_none == 0:
            return state
        return replace(state, consecutive_tick_none=0)
    return replace(state, consecutive_tick_none=state.consecutive_tick_none + 1)


def should_reset(state: Mt5WatchdogState) -> bool:
    """True iff the streak is at or above the reset threshold and we are
    NOT inside a critical section.  Giveup takes precedence — once the
    streak hits ``giveup_threshold`` the caller should ``sys.exit(1)``
    instead of attempting another reset.
    """
    if state.in_critical_section:
        return False
    if state.consecutive_tick_none >= state.giveup_threshold:
        return False
    return state.consecutive_tick_none >= state.reset_threshold


def should_giveup(state: Mt5WatchdogState) -> bool:
    """True iff the streak is at or above the giveup threshold."""
    return state.consecutive_tick_none >= state.giveup_threshold


def enter_critical(state: Mt5WatchdogState) -> Mt5WatchdogState:
    """Mark the cycle as being in an order_send critical section."""
    if state.in_critical_section:
        return state
    return replace(state, in_critical_section=True)


def exit_critical(state: Mt5WatchdogState) -> Mt5WatchdogState:
    """Leave the critical section.  Safe to call even when not inside."""
    if not state.in_critical_section:
        return state
    return replace(state, in_critical_section=False)


def mark_initialized(
    state: Mt5WatchdogState,
    *,
    monotonic_now: float | None = None,
) -> Mt5WatchdogState:
    """Record that ``mt5.initialize()`` just succeeded.

    Clears the tick-none streak and stamps ``last_init_monotonic`` so
    the health probe can compute handle age.
    """
    mono = monotonic_now if monotonic_now is not None else time.monotonic()
    return replace(
        state,
        consecutive_tick_none=0,
        last_init_monotonic=mono,
    )


def handle_age_sec(
    state: Mt5WatchdogState,
    *,
    monotonic_now: float | None = None,
) -> int | None:
    """Return ``int(now - last_init)`` in seconds, or None if the handle
    has never been initialised in this process.
    """
    if state.last_init_monotonic <= 0.0:
        return None
    mono = monotonic_now if monotonic_now is not None else time.monotonic()
    return max(0, int(mono - state.last_init_monotonic))


def try_reset_handle(
    mt5_module: Any,
    state: Mt5WatchdogState,
    *,
    monotonic_now: Callable[[], float] = time.monotonic,
) -> Mt5WatchdogState:
    """Attempt one ``mt5.shutdown() → mt5.initialize()`` cycle.

    Emits three structured events:

    - ``mt5_handle_reset_attempt`` — always, before touching MT5.
    - ``mt5_handle_reset_success`` — initialize() returned truthy.
    - ``mt5_handle_reset_failed`` — initialize() returned falsy OR an
      exception fired; the streak is NOT reset and the caller continues
      climbing toward the giveup threshold.

    Parameters
    ----------
    mt5_module:
        The ``MetaTrader5`` module (or a unit-test mock).  Must expose
        ``shutdown()`` and ``initialize()``; optionally ``last_error()``
        for diagnostic logging.
    state:
        Current watchdog state.  Returned unchanged on failure except
        for ``reset_attempts`` increment.

    Returns
    -------
    A new state: streak cleared + ``last_init_monotonic`` bumped on
    success; ``reset_attempts`` incremented unconditionally so operators
    can audit how often the handle rots per day.
    """
    from smc.monitor.structured_log import crit as log_crit
    from smc.monitor.structured_log import info as log_info
    from smc.monitor.structured_log import warn as log_warn

    attempt_state = replace(state, reset_attempts=state.reset_attempts + 1)
    log_info(
        "mt5_handle_reset_attempt",
        streak=state.consecutive_tick_none,
        attempt_no=attempt_state.reset_attempts,
    )

    # Step 1: shutdown (best-effort, never raise).
    try:
        mt5_module.shutdown()
    except Exception as shutdown_exc:  # pragma: no cover — MT5 shutdown rarely raises
        log_warn(
            "mt5_handle_reset_shutdown_error",
            exc=str(shutdown_exc)[:200],
        )

    # Step 2: re-initialize.
    init_ok = False
    last_error: Any = None
    try:
        init_ok = bool(mt5_module.initialize())
    except Exception as init_exc:
        log_crit(
            "mt5_handle_reset_failed",
            streak=state.consecutive_tick_none,
            attempt_no=attempt_state.reset_attempts,
            exception_type=type(init_exc).__name__,
            exception_msg=str(init_exc)[:200],
        )
        return attempt_state

    if not init_ok:
        try:
            last_error = mt5_module.last_error()
        except Exception:  # pragma: no cover — diagnostic only
            last_error = None
        log_crit(
            "mt5_handle_reset_failed",
            streak=state.consecutive_tick_none,
            attempt_no=attempt_state.reset_attempts,
            last_error=str(last_error)[:200] if last_error is not None else None,
        )
        return attempt_state

    reset_ok_state = replace(
        attempt_state,
        consecutive_tick_none=0,
        last_init_monotonic=float(monotonic_now()),
    )
    log_info(
        "mt5_handle_reset_success",
        attempt_no=reset_ok_state.reset_attempts,
        handle_age_cleared=True,
    )
    return reset_ok_state


__all__ = [
    "DEFAULT_RESET_THRESHOLD",
    "DEFAULT_GIVEUP_THRESHOLD",
    "Mt5WatchdogState",
    "new_state",
    "record_tick_result",
    "should_reset",
    "should_giveup",
    "enter_critical",
    "exit_critical",
    "mark_initialized",
    "handle_age_sec",
    "try_reset_handle",
]
