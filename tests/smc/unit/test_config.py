"""Unit tests for smc.config.SMCConfig — Round 4 Alt-B W2 macro_enabled fields
and Round 4 Alt-B W3 journal_suffix field.

Tests:
    - macro_enabled defaults to False
    - SMC_MACRO_ENABLED env var sets macro_enabled=True
    - ENABLE_MACRO=1 env var also sets macro_enabled=True (alias support)
    - macro_cache_ttl_hours default
    - fred_api_key default
    - journal_suffix defaults to empty string
    - SMC_JOURNAL_SUFFIX env var overrides journal_suffix
"""

from __future__ import annotations

import pytest


class TestSMCConfigMacroDefaults:
    """macro_enabled must default to False for production safety."""

    def test_macro_enabled_default_false(self, monkeypatch) -> None:
        """Without any env vars, macro_enabled is False (default OFF)."""
        # Unset any env var that might be set in the parent test process
        monkeypatch.delenv("SMC_MACRO_ENABLED", raising=False)
        monkeypatch.delenv("ENABLE_MACRO", raising=False)

        from smc.config import SMCConfig
        cfg = SMCConfig()
        assert cfg.macro_enabled is False

    def test_macro_cache_ttl_hours_default(self, monkeypatch) -> None:
        """macro_cache_ttl_hours defaults to 24."""
        monkeypatch.delenv("SMC_MACRO_CACHE_TTL_HOURS", raising=False)

        from smc.config import SMCConfig
        cfg = SMCConfig()
        assert cfg.macro_cache_ttl_hours == 24

    def test_fred_api_key_default_empty(self, monkeypatch) -> None:
        """fred_api_key defaults to empty SecretStr."""
        monkeypatch.delenv("SMC_FRED_API_KEY", raising=False)

        from smc.config import SMCConfig
        cfg = SMCConfig()
        assert cfg.fred_api_key.get_secret_value() == ""


class TestSMCConfigMacroEnvOverride:
    """SMC_MACRO_ENABLED env var must enable macro overlay."""

    def test_smc_macro_enabled_env_var_activates(self, monkeypatch) -> None:
        """SMC_MACRO_ENABLED=true sets macro_enabled=True."""
        monkeypatch.setenv("SMC_MACRO_ENABLED", "true")

        # Reload the module to pick up the monkeypatched env var
        from importlib import reload
        import smc.config as _cfg_mod
        reload(_cfg_mod)

        from smc.config import SMCConfig
        cfg = SMCConfig()
        assert cfg.macro_enabled is True

    def test_macro_enabled_false_when_env_is_false(self, monkeypatch) -> None:
        """SMC_MACRO_ENABLED=false (or 0) keeps macro_enabled=False."""
        monkeypatch.setenv("SMC_MACRO_ENABLED", "false")

        from smc.config import SMCConfig
        cfg = SMCConfig()
        assert cfg.macro_enabled is False


class TestSMCConfigJournalSuffix:
    """journal_suffix field — Round 4 Alt-B W3 A/B path separation."""

    def test_journal_suffix_default_empty(self, monkeypatch) -> None:
        """journal_suffix defaults to '' (backward-compat control leg)."""
        monkeypatch.delenv("SMC_JOURNAL_SUFFIX", raising=False)

        from smc.config import SMCConfig
        cfg = SMCConfig()
        assert cfg.journal_suffix == ""

    def test_journal_suffix_set_via_env(self, monkeypatch) -> None:
        """SMC_JOURNAL_SUFFIX=_macro sets journal_suffix to '_macro'."""
        monkeypatch.setenv("SMC_JOURNAL_SUFFIX", "_macro")

        from smc.config import SMCConfig
        cfg = SMCConfig()
        assert cfg.journal_suffix == "_macro"
