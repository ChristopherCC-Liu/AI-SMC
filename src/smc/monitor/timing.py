"""Bar close timing utilities for the AI-SMC live trading loop.

Provides :func:`wait_for_bar_close` which sleeps until the next M15/H1/H4/D1
bar boundary, and :func:`next_bar_close` which computes the next close time
without sleeping.

M15 bar boundaries:  :00, :15, :30, :45 of each hour.
H1  bar boundaries:  :00 of each hour.
H4  bar boundaries:  00:00, 04:00, 08:00, 12:00, 16:00, 20:00 UTC.
D1  bar boundaries:  00:00 UTC daily.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from smc.data.schemas import Timeframe

# Mapping from Timeframe to bar duration in minutes
_BAR_MINUTES: dict[Timeframe, int] = {
    Timeframe.M1: 1,
    Timeframe.M5: 5,
    Timeframe.M15: 15,
    Timeframe.H1: 60,
    Timeframe.H4: 240,
    Timeframe.D1: 1440,
}


def next_bar_close(timeframe: Timeframe, now: datetime | None = None) -> datetime:
    """Compute the next bar close time for the given timeframe.

    Parameters
    ----------
    timeframe:
        The timeframe to compute the next close for.
    now:
        Current time.  Defaults to ``datetime.now(tz=timezone.utc)``.

    Returns
    -------
    datetime
        The next bar close time (tz-aware, UTC).
    """
    if now is None:
        now = datetime.now(tz=timezone.utc)

    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    bar_minutes = _BAR_MINUTES[timeframe]

    # Truncate to the start of the current bar
    total_minutes = now.hour * 60 + now.minute
    bar_start_minute = (total_minutes // bar_minutes) * bar_minutes

    bar_start = now.replace(
        hour=bar_start_minute // 60,
        minute=bar_start_minute % 60,
        second=0,
        microsecond=0,
    )

    # For D1, the bar starts at 00:00 and closes at next day's 00:00
    if timeframe == Timeframe.D1:
        bar_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

    # Next bar close = bar_start + bar_duration
    next_close = bar_start + timedelta(minutes=bar_minutes)

    # If we're exactly at the close time, the bar just closed — return the
    # *following* bar close.
    if now >= next_close:
        next_close = next_close + timedelta(minutes=bar_minutes)

    return next_close


async def wait_for_bar_close(
    timeframe: Timeframe,
    now: datetime | None = None,
    buffer_seconds: float = 2.0,
) -> datetime:
    """Sleep until the next bar close time, then return it.

    Adds a small buffer (default 2 s) after the close to allow the broker
    to finalize the bar data.

    Parameters
    ----------
    timeframe:
        The timeframe to wait for.
    now:
        Current time (for testing — avoids sleeping when provided with a
        future time).
    buffer_seconds:
        Extra seconds to wait after the bar close.

    Returns
    -------
    datetime
        The bar close time that was waited for.
    """
    target = next_bar_close(timeframe, now)
    current = datetime.now(tz=timezone.utc) if now is None else now

    wait_seconds = (target - current).total_seconds() + buffer_seconds
    if wait_seconds > 0:
        await asyncio.sleep(wait_seconds)

    return target


__all__ = ["next_bar_close", "wait_for_bar_close"]
