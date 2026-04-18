"""Unified critical event alerting: stderr log + Telegram (opt-in).

Provides a single synchronous entry point ``alert_critical`` that:
1. Always writes a ``[CRIT]`` structured-JSON line to stderr.
2. Optionally fires a Telegram message if the bot env vars are set.

Telegram failures are silently swallowed — this module must never
propagate exceptions into the main trading loop.

Note on async: ``TelegramAlerter._send`` is a coroutine.  ``alert_critical``
is intentionally *synchronous* (live_demo calls it from sync code).  We use
``asyncio.run()`` to drive the coroutine, guarded by a try/except so that a
running event loop (e.g. pytest-asyncio) doesn't crash the caller.
"""
from __future__ import annotations

import asyncio
import os
from typing import Any, Optional

from smc.monitor.alerter import TelegramAlerter
from smc.monitor.structured_log import log_event

_cached_alerter: Optional[TelegramAlerter] = None
_alerter_init_attempted: bool = False


def _get_alerter() -> Optional[TelegramAlerter]:
    """Lazy singleton; returns None if env is not configured, never raises."""
    global _cached_alerter, _alerter_init_attempted
    if _alerter_init_attempted:
        return _cached_alerter
    _alerter_init_attempted = True
    token = os.getenv("SMC_TELEGRAM_BOT_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("SMC_TELEGRAM_CHAT_ID") or os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return None
    try:
        _cached_alerter = TelegramAlerter(bot_token=token, chat_id=chat_id)
    except Exception:
        _cached_alerter = None
    return _cached_alerter


def alert_critical(event: str, *, send_telegram: bool = True, **context: Any) -> None:
    """Log [CRIT] JSON to stderr and optionally send a Telegram message.

    Parameters
    ----------
    event:
        Short event name / description (appears as ``"event"`` in JSON).
    send_telegram:
        When ``False``, only logs to stderr — skips the Telegram push.
        Useful for high-frequency noise you don't want pinging your phone.
    **context:
        Arbitrary key-value fields included in the JSON log line and
        appended to the Telegram message body.
    """
    log_event("CRIT", event, **context)

    if not send_telegram:
        return

    alerter = _get_alerter()
    if alerter is None:
        return

    try:
        msg = f"[CRIT] {event}"
        if context:
            msg += " | " + " | ".join(f"{k}={v}" for k, v in context.items())
        _run_async(alerter.send_text(msg))
    except Exception:
        pass  # best-effort; never raise


def _run_async(coro: Any) -> None:
    """Drive a coroutine to completion from sync code, swallowing all errors.

    Tries ``asyncio.run()`` first (clean slate).  If a loop is already running
    (e.g. inside an async test), schedules the coro as a fire-and-forget task
    instead.
    """
    try:
        asyncio.run(coro)
    except RuntimeError:
        # A running event loop exists (e.g. pytest-asyncio, uvicorn).
        # Schedule as a background task instead.
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(coro)
        except Exception:
            pass
    except Exception:
        pass


def reset_alerter_cache() -> None:
    """Force re-initialisation of the alerter singleton.

    Call this in tests after manipulating env vars so that the next
    ``alert_critical`` picks up the new configuration.
    """
    global _cached_alerter, _alerter_init_attempted
    _cached_alerter = None
    _alerter_init_attempted = False


__all__ = ["alert_critical", "reset_alerter_cache"]
