"""Telegram alerter for the AI-SMC live trading system.

Sends trade alerts and health warnings to a configured Telegram chat.
Rate-limited to 20 messages per minute to respect Telegram's bot API limits.

Usage::

    from smc.monitor.alerter import TelegramAlerter

    alerter = TelegramAlerter(bot_token="...", chat_id="...")
    await alerter.send_trade_alert(
        action="open",
        instrument="XAUUSD",
        direction="long",
        price=2350.0,
        lots=0.01,
        sl=2340.0,
        tp=2370.0,
    )

When ``bot_token`` is empty, the alerter becomes a no-op (safe for dev/testing).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque

import httpx

logger = logging.getLogger(__name__)

# Telegram rate limit: 20 messages per minute
_RATE_LIMIT_WINDOW_SECONDS = 60.0
_RATE_LIMIT_MAX_MESSAGES = 20


class TelegramAlerter:
    """Send trade alerts to Telegram with rate limiting.

    Parameters
    ----------
    bot_token:
        Telegram Bot API token.  Empty string disables sending.
    chat_id:
        Telegram chat or channel ID.
    timeout:
        HTTP request timeout in seconds.
    """

    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        timeout: float = 10.0,
    ) -> None:
        self._bot_token = bot_token
        self._chat_id = chat_id
        self._timeout = timeout
        self._enabled = bool(bot_token and chat_id)
        self._send_timestamps: deque[float] = deque()

    @property
    def enabled(self) -> bool:
        """True if the alerter is configured and will send messages."""
        return self._enabled

    async def send_trade_alert(
        self,
        *,
        action: str,
        instrument: str,
        direction: str,
        price: float,
        lots: float,
        sl: float,
        tp: float,
        pnl: float = 0.0,
        confluence: float = 0.0,
    ) -> bool:
        """Send a formatted trade alert message.

        Returns True if the message was sent successfully, False otherwise.
        """
        emoji = _action_emoji(action)
        dir_arrow = "UP" if direction == "long" else "DOWN"

        lines = [
            f"{emoji} {action.upper()} {instrument}",
            f"Direction: {dir_arrow} {direction.upper()}",
            f"Price: {price:.2f}",
            f"Lots: {lots}",
            f"SL: {sl:.2f}  |  TP: {tp:.2f}",
        ]
        if pnl != 0.0:
            pnl_sign = "+" if pnl > 0 else ""
            lines.append(f"PnL: {pnl_sign}{pnl:.2f} USD")
        if confluence > 0.0:
            lines.append(f"Confluence: {confluence:.1%}")

        text = "\n".join(lines)
        return await self._send(text)

    async def send_health_alert(self, *, failed_checks: list[str]) -> bool:
        """Send a health warning when checks fail.

        Parameters
        ----------
        failed_checks:
            List of human-readable descriptions of failed checks.
        """
        lines = ["WARNING: Health Check Failed"]
        for check in failed_checks:
            lines.append(f"  - {check}")
        text = "\n".join(lines)
        return await self._send(text)

    async def send_text(self, text: str) -> bool:
        """Send a raw text message."""
        return await self._send(text)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    async def _send(self, text: str) -> bool:
        """Send a message via Telegram Bot API with rate limiting."""
        if not self._enabled:
            logger.debug("Telegram alerter disabled — message not sent")
            return False

        if not self._check_rate_limit():
            logger.warning("Telegram rate limit reached — dropping message")
            return False

        url = f"https://api.telegram.org/bot{self._bot_token}/sendMessage"
        # Round 4 v5 (2026-04-20 post-mortem): parse_mode="HTML" breaks when
        # messages contain '<' chars from Python tracebacks ("<frozen
        # importlib._bootstrap>", "<module>"), inequalities ("MFE <0.10R"),
        # or literal angle brackets in broker strings. Telegram's HTML parser
        # then returns 400 "Unsupported start tag" and the whole alert
        # silently drops. Plain text has no such footguns.
        payload = {
            "chat_id": self._chat_id,
            "text": text[:4000],  # Telegram hard limit 4096, leave margin
        }

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(url, json=payload)
                if resp.status_code == 200:
                    self._record_send()
                    return True
                logger.warning(
                    "Telegram API returned %d: %s",
                    resp.status_code,
                    resp.text[:200],
                )
                return False
        except httpx.HTTPError as exc:
            logger.warning("Telegram send failed: %s", exc)
            return False

    def _check_rate_limit(self) -> bool:
        """Return True if we are within the rate limit window."""
        now = time.monotonic()
        # Remove timestamps older than the window
        while self._send_timestamps and (now - self._send_timestamps[0]) > _RATE_LIMIT_WINDOW_SECONDS:
            self._send_timestamps.popleft()
        return len(self._send_timestamps) < _RATE_LIMIT_MAX_MESSAGES

    def _record_send(self) -> None:
        """Record that a message was sent."""
        self._send_timestamps.append(time.monotonic())


def _action_emoji(action: str) -> str:
    """Return a text indicator for the trade action."""
    mapping = {
        "open": "[OPEN]",
        "close": "[CLOSE]",
        "sl_hit": "[SL]",
        "tp_hit": "[TP]",
        "modify": "[MOD]",
        "partial_close": "[PARTIAL]",
        "cycle": "[CYCLE]",
    }
    return mapping.get(action, f"[{action.upper()}]")


__all__ = ["TelegramAlerter"]
