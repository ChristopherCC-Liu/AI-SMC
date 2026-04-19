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


class TestSMCConfigMacroMagic:
    """audit-r4 v5 Option B: macro_magic treatment-leg magic override."""

    def test_macro_magic_default(self, monkeypatch) -> None:
        """macro_magic defaults to 19760428 (treatment leg on TMGM Demo)."""
        monkeypatch.delenv("SMC_MACRO_MAGIC", raising=False)
        from smc.config import SMCConfig
        cfg = SMCConfig()
        assert cfg.macro_magic == 19760428

    def test_macro_magic_env_override(self, monkeypatch) -> None:
        """SMC_MACRO_MAGIC env var overrides the default."""
        monkeypatch.setenv("SMC_MACRO_MAGIC", "12345678")
        from smc.config import SMCConfig
        cfg = SMCConfig()
        assert cfg.macro_magic == 12345678

    def test_magic_for_control_leg_uses_instrument_magic(self) -> None:
        """Control leg (suffix="") always returns cfg.magic (XAU=19760418)."""
        from smc.config import SMCConfig
        cfg = SMCConfig()
        assert cfg.magic_for(19760418, "") == 19760418
        assert cfg.magic_for(19760419, "") == 19760419  # BTC

    def test_magic_for_treatment_leg_uses_macro_magic(self, monkeypatch) -> None:
        """Treatment leg (suffix="_macro") returns macro_magic regardless of instrument."""
        monkeypatch.delenv("SMC_MACRO_MAGIC", raising=False)
        from smc.config import SMCConfig
        cfg = SMCConfig()
        assert cfg.magic_for(19760418, "_macro") == 19760428
        # BTC + treatment also lands on same macro magic (single treatment leg)
        assert cfg.magic_for(19760419, "_macro") == 19760428


class TestSMCConfigVirtualBalanceSplit:
    """audit-r4 v5 Option B: virtual_balance_split for dual-magic sizing."""

    def test_virtual_balance_split_default(self, monkeypatch) -> None:
        """Default split is 50/50 for control and treatment."""
        monkeypatch.delenv("SMC_VIRTUAL_BALANCE_SPLIT", raising=False)
        from smc.config import SMCConfig
        cfg = SMCConfig()
        assert cfg.virtual_balance_split == {"": 0.5, "_macro": 0.5}

    def test_virtual_balance_for_control_leg(self) -> None:
        """Control (suffix="") sees 50% of MT5 balance by default."""
        from smc.config import SMCConfig
        cfg = SMCConfig()
        assert cfg.virtual_balance_for("", 1000.0) == 500.0

    def test_virtual_balance_for_treatment_leg(self) -> None:
        """Treatment (suffix="_macro") sees the other 50% by default."""
        from smc.config import SMCConfig
        cfg = SMCConfig()
        assert cfg.virtual_balance_for("_macro", 1000.0) == 500.0

    def test_virtual_balance_for_none_balance_returns_zero(self) -> None:
        """Fail-closed: None balance → 0 virtual."""
        from smc.config import SMCConfig
        cfg = SMCConfig()
        assert cfg.virtual_balance_for("", None) == 0.0

    def test_virtual_balance_for_negative_balance_returns_zero(self) -> None:
        """Fail-closed: non-positive balance → 0 virtual."""
        from smc.config import SMCConfig
        cfg = SMCConfig()
        assert cfg.virtual_balance_for("", -100.0) == 0.0
        assert cfg.virtual_balance_for("_macro", 0.0) == 0.0

    def test_unknown_suffix_defaults_to_half(self) -> None:
        """Unknown suffixes default to 0.5 split so a typo doesn't over-size."""
        from smc.config import SMCConfig
        cfg = SMCConfig()
        # A suffix that isn't in the map still gets 50% — safer than 100%.
        assert cfg.virtual_balance_for("_typo", 1000.0) == 500.0

    def test_virtual_balance_split_env_override(self, monkeypatch) -> None:
        """SMC_VIRTUAL_BALANCE_SPLIT can override splits via JSON string."""
        monkeypatch.setenv(
            "SMC_VIRTUAL_BALANCE_SPLIT",
            '{"": 0.7, "_macro": 0.3}',
        )
        from smc.config import SMCConfig
        cfg = SMCConfig()
        assert cfg.virtual_balance_split.get("") == 0.7
        assert cfg.virtual_balance_split.get("_macro") == 0.3
        assert cfg.virtual_balance_for("", 1000.0) == 700.0
        assert cfg.virtual_balance_for("_macro", 1000.0) == 300.0

    def test_virtual_balance_split_env_invalid_falls_back(self, monkeypatch) -> None:
        """Invalid JSON in env falls back to the safe 50/50 default."""
        monkeypatch.setenv("SMC_VIRTUAL_BALANCE_SPLIT", "not-valid-json")
        from smc.config import SMCConfig
        cfg = SMCConfig()
        # Falls back to default
        assert cfg.virtual_balance_split == {"": 0.5, "_macro": 0.5}

    def test_virtual_balance_split_clamps_invalid_values(self, monkeypatch) -> None:
        """Out-of-range splits (0, >1, negative) get clamped to 0.5."""
        monkeypatch.setenv(
            "SMC_VIRTUAL_BALANCE_SPLIT",
            '{"": 2.5, "_macro": -0.1}',
        )
        from smc.config import SMCConfig
        cfg = SMCConfig()
        assert cfg.virtual_balance_split.get("") == 0.5
        assert cfg.virtual_balance_split.get("_macro") == 0.5
