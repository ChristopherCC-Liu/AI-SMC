"""Unit tests for scripts/notify_ea_deploy.py (O4)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_PATH = REPO_ROOT / "scripts" / "notify_ea_deploy.py"


def _load_module():
    """Load notify_ea_deploy.py as a module despite its script-style location."""
    spec = importlib.util.spec_from_file_location("notify_ea_deploy", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None, f"cannot spec {SCRIPT_PATH}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules["notify_ea_deploy"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def notify_mod():
    return _load_module()


# ---------------------------------------------------------------------------
# _build_message
# ---------------------------------------------------------------------------


def test_build_message_includes_version(notify_mod):
    msg = notify_mod._build_message(version="2.01", ex5_size_kb=142.3, compile_sec=1.7)
    assert "v2.01" in msg
    assert "AISMCReceiver" in msg
    assert "142.3 KB" in msg
    assert "1.7s" in msg


def test_build_message_has_newlines(notify_mod):
    # Telegram formatting relies on \n not being collapsed.
    msg = notify_mod._build_message(version="2.01", ex5_size_kb=142.3, compile_sec=1.7)
    assert msg.count("\n") >= 3
    assert "re-attach" in msg.lower()


def test_build_message_is_plain_text_no_html(notify_mod):
    # After the 2026-04-20 parse_mode fix, we must not emit HTML/Markdown.
    msg = notify_mod._build_message(version="2.01", ex5_size_kb=142.3, compile_sec=1.7)
    for bad in ("<b>", "<i>", "*", "_", "`"):
        assert bad not in msg, f"plain-text rule broken: contains {bad!r}"


# ---------------------------------------------------------------------------
# main() — dry run & env-missing path
# ---------------------------------------------------------------------------


def test_main_dry_run_prints_and_exits_zero(notify_mod, capsys):
    rc = notify_mod.main(["--version", "2.01", "--ex5-size-kb", "142.3", "--compile-sec", "1.7", "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "v2.01" in out
    assert "dry-run" in out.lower()


def test_main_without_telegram_env_prints_fallback(notify_mod, monkeypatch, capsys):
    for k in ("SMC_TELEGRAM_BOT_TOKEN", "TELEGRAM_BOT_TOKEN", "SMC_TELEGRAM_CHAT_ID", "TELEGRAM_CHAT_ID"):
        monkeypatch.delenv(k, raising=False)
    rc = notify_mod.main(["--version", "2.01", "--ex5-size-kb", "142.3", "--compile-sec", "1.7"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Telegram env vars missing" in out
    assert "v2.01" in out


def test_main_invokes_alerter_when_env_set(notify_mod, monkeypatch):
    monkeypatch.setenv("SMC_TELEGRAM_BOT_TOKEN", "dummy_token_xyz")
    monkeypatch.setenv("SMC_TELEGRAM_CHAT_ID", "-100123")
    sent: list[str] = []

    class _FakeAlerter:
        def __init__(self, *a, **kw):
            pass

        async def send_text(self, text):
            sent.append(text)
            return True

    with patch("smc.monitor.alerter.TelegramAlerter", _FakeAlerter):
        rc = notify_mod.main(["--version", "3.14", "--ex5-size-kb", "99.9", "--compile-sec", "2.5"])

    assert rc == 0
    assert len(sent) == 1
    assert "v3.14" in sent[0]
    # Plain-text invariant (post 2026-04-20 fix)
    assert "<" not in sent[0]


def test_main_returns_nonzero_when_telegram_fails(notify_mod, monkeypatch):
    monkeypatch.setenv("SMC_TELEGRAM_BOT_TOKEN", "dummy_token_xyz")
    monkeypatch.setenv("SMC_TELEGRAM_CHAT_ID", "-100123")

    class _FailingAlerter:
        def __init__(self, *a, **kw):
            pass

        async def send_text(self, text):
            return False

    with patch("smc.monitor.alerter.TelegramAlerter", _FailingAlerter):
        rc = notify_mod.main(["--version", "2.01", "--ex5-size-kb", "142.3", "--compile-sec", "1.7"])

    assert rc == 3
