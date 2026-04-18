"""Unit tests for smc.monitor.critical_alerter."""
from __future__ import annotations

import asyncio
import io
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import smc.monitor.critical_alerter as _mod
from smc.monitor.critical_alerter import alert_critical, reset_alerter_cache


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_cache():
    """Always reset the alerter singleton before and after each test."""
    reset_alerter_cache()
    yield
    reset_alerter_cache()


@pytest.fixture()
def clean_env(monkeypatch: pytest.MonkeyPatch):
    """Remove Telegram env vars so the alerter is not initialised."""
    for var in (
        "SMC_TELEGRAM_BOT_TOKEN",
        "TELEGRAM_BOT_TOKEN",
        "SMC_TELEGRAM_CHAT_ID",
        "TELEGRAM_CHAT_ID",
    ):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture()
def telegram_env(monkeypatch: pytest.MonkeyPatch):
    """Set Telegram env vars so the alerter *would* initialise."""
    monkeypatch.setenv("SMC_TELEGRAM_BOT_TOKEN", "test-token-123")
    monkeypatch.setenv("SMC_TELEGRAM_CHAT_ID", "99999999")


# ---------------------------------------------------------------------------
# Test: no-telegram path (only logs)
# ---------------------------------------------------------------------------


def test_alert_critical_without_telegram_only_logs(clean_env, capsys):
    """When no Telegram env vars are set, alert_critical logs to stderr and does not raise."""
    alert_critical("connection_lost", instrument="XAUUSD", error="timeout")

    captured = capsys.readouterr()
    assert "[CRIT]" in captured.err
    assert "connection_lost" in captured.err


def test_alert_critical_no_telegram_does_not_raise(clean_env):
    """alert_critical must never raise, even with no Telegram configured."""
    # Should complete without exception
    alert_critical("some_event", foo="bar")


# ---------------------------------------------------------------------------
# Test: mocked Telegram path
# ---------------------------------------------------------------------------


def _make_mock_alerter() -> MagicMock:
    """Build a mock TelegramAlerter with an async send_text."""
    mock = MagicMock()
    mock.send_text = AsyncMock(return_value=True)
    return mock


def test_alert_critical_with_mocked_telegram(clean_env, monkeypatch):
    """When _get_alerter returns a mock, send_text is called exactly once."""
    mock_alerter = _make_mock_alerter()
    monkeypatch.setattr(_mod, "_get_alerter", lambda: mock_alerter)

    alert_critical("order_rejected", ticket=12345)

    mock_alerter.send_text.assert_called_once()
    call_arg = mock_alerter.send_text.call_args[0][0]
    assert "order_rejected" in call_arg
    assert "ticket" in call_arg


def test_alert_critical_send_telegram_false_skips_bot(clean_env, monkeypatch):
    """send_telegram=False must skip the Telegram push entirely."""
    mock_alerter = _make_mock_alerter()
    monkeypatch.setattr(_mod, "_get_alerter", lambda: mock_alerter)

    alert_critical("high_freq_event", send_telegram=False, count=100)

    mock_alerter.send_text.assert_not_called()


def test_alert_critical_send_telegram_false_still_logs(clean_env, capsys, monkeypatch):
    """send_telegram=False must still emit the [CRIT] log line."""
    mock_alerter = _make_mock_alerter()
    monkeypatch.setattr(_mod, "_get_alerter", lambda: mock_alerter)

    alert_critical("noisy_event", send_telegram=False)

    captured = capsys.readouterr()
    assert "[CRIT]" in captured.err


# ---------------------------------------------------------------------------
# Test: Telegram exception swallowed
# ---------------------------------------------------------------------------


def test_telegram_exception_swallowed(clean_env, monkeypatch):
    """alert_critical must not propagate exceptions thrown by the Telegram send."""
    mock_alerter = MagicMock()
    mock_alerter.send_text = AsyncMock(side_effect=RuntimeError("network error"))
    monkeypatch.setattr(_mod, "_get_alerter", lambda: mock_alerter)

    # Must not raise
    alert_critical("should_not_propagate", detail="network error")


def test_telegram_send_exception_still_logs(clean_env, capsys, monkeypatch):
    """Even when Telegram send raises, the [CRIT] log line must be present."""
    mock_alerter = MagicMock()
    mock_alerter.send_text = AsyncMock(side_effect=ConnectionError("refused"))
    monkeypatch.setattr(_mod, "_get_alerter", lambda: mock_alerter)

    alert_critical("connectivity_event")

    captured = capsys.readouterr()
    assert "[CRIT]" in captured.err


# ---------------------------------------------------------------------------
# Test: alerter cache env-change behaviour
# ---------------------------------------------------------------------------


def test_alerter_cache_no_env_returns_none(clean_env):
    """Without env vars, _get_alerter must return None."""
    result = _mod._get_alerter()
    assert result is None


def test_alerter_cache_env_change(clean_env, monkeypatch):
    """After reset_alerter_cache + env set, _get_alerter returns a live alerter.

    Phase 1: no env vars → None.
    Phase 2: set env, reset cache → alerter constructed successfully.
    """
    # Phase 1: cache was reset by autouse fixture; no env → None
    assert _mod._get_alerter() is None

    # Phase 2: set env vars, reset cache, verify alerter is created
    monkeypatch.setenv("SMC_TELEGRAM_BOT_TOKEN", "test-token-123")
    monkeypatch.setenv("SMC_TELEGRAM_CHAT_ID", "99999999")
    reset_alerter_cache()

    # Patch TelegramAlerter constructor to avoid real HTTP initialisation
    with patch("smc.monitor.critical_alerter.TelegramAlerter") as MockAlerter:
        mock_instance = MagicMock()
        MockAlerter.return_value = mock_instance

        result = _mod._get_alerter()

    assert result is mock_instance
    MockAlerter.assert_called_once_with(
        bot_token="test-token-123",
        chat_id="99999999",
    )


def test_alerter_cache_is_singleton(clean_env, monkeypatch):
    """_get_alerter returns the same object on repeated calls (singleton)."""
    mock_alerter = _make_mock_alerter()
    monkeypatch.setattr(_mod, "_get_alerter", lambda: mock_alerter)

    # Multiple alert_critical calls — _get_alerter (our lambda) returns same mock
    alert_critical("ev1")
    alert_critical("ev2")
    # Both calls used the same alerter (no reset in between)
    assert mock_alerter.send_text.call_count == 2


def test_alerter_constructor_failure_returns_none(telegram_env):
    """If TelegramAlerter constructor raises, _get_alerter silently returns None."""
    with patch(
        "smc.monitor.critical_alerter.TelegramAlerter",
        side_effect=ValueError("bad config"),
    ):
        result = _mod._get_alerter()

    assert result is None


# ---------------------------------------------------------------------------
# Test: SMC_ prefix env vars
# ---------------------------------------------------------------------------


def test_smc_prefix_env_vars_used(monkeypatch):
    """SMC_TELEGRAM_* env vars are preferred over bare TELEGRAM_* vars."""
    for var in ("SMC_TELEGRAM_BOT_TOKEN", "TELEGRAM_BOT_TOKEN", "SMC_TELEGRAM_CHAT_ID", "TELEGRAM_CHAT_ID"):
        monkeypatch.delenv(var, raising=False)

    monkeypatch.setenv("SMC_TELEGRAM_BOT_TOKEN", "smc-token")
    monkeypatch.setenv("SMC_TELEGRAM_CHAT_ID", "smc-chat")

    with patch("smc.monitor.critical_alerter.TelegramAlerter") as MockAlerter:
        MockAlerter.return_value = MagicMock()
        result = _mod._get_alerter()

    MockAlerter.assert_called_once_with(bot_token="smc-token", chat_id="smc-chat")
    assert result is not None


def test_bare_env_vars_fallback(monkeypatch):
    """Bare TELEGRAM_* vars work when SMC_ prefixed vars are absent."""
    for var in ("SMC_TELEGRAM_BOT_TOKEN", "SMC_TELEGRAM_CHAT_ID"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "bare-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "bare-chat")

    with patch("smc.monitor.critical_alerter.TelegramAlerter") as MockAlerter:
        MockAlerter.return_value = MagicMock()
        result = _mod._get_alerter()

    MockAlerter.assert_called_once_with(bot_token="bare-token", chat_id="bare-chat")
    assert result is not None
