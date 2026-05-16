"""Application configuration for the AI-SMC trading system.

All settings are loaded from environment variables (prefix ``SMC_``) or from
a ``.env`` file in the working directory.  Sensitive fields (passwords, API
keys, tokens) are never logged by default.

Usage::

    from smc.config import SMCConfig

    cfg = SMCConfig()          # reads env / .env
    cfg = SMCConfig(env="paper", mt5_mock=False)  # override in code

The config object is intentionally *not* frozen so that test fixtures can
construct instances with keyword overrides.  Treat it as effectively immutable
in production code — never mutate the singleton config after startup.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class SMCConfig(BaseSettings):
    """AI-SMC application settings.

    All fields can be overridden via environment variables with the ``SMC_``
    prefix (e.g. ``SMC_ENV=paper``, ``SMC_MT5_LOGIN=12345``).
    """

    model_config = SettingsConfigDict(
        env_prefix="SMC_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
        # Windows .bat `set VAR=VAL` historically ships trailing whitespace
        # up to `&&` line continuation or EOL, producing 'false ' (with space)
        # that pydantic strict bool_parser rejects. Strip all leading/trailing
        # whitespace on string inputs before type coercion.
        str_strip_whitespace=True,
    )

    # ------------------------------------------------------------------
    # Runtime
    # ------------------------------------------------------------------
    env: Literal["dev", "paper", "live"] = Field(
        default="dev",
        description="Execution environment. 'live' enables real order submission.",
    )
    data_dir: Path = Field(
        default=Path("./data"),
        description="Root directory for the local data lake (parquet files).",
    )
    log_level: str = Field(
        default="INFO",
        description="Python logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
    )
    mt5_mock: bool = Field(
        default=True,
        description="When True, MT5 calls are stubbed out. Required on macOS/Linux dev.",
    )

    # ------------------------------------------------------------------
    # MT5 Connection
    # ------------------------------------------------------------------
    mt5_login: int = Field(
        default=0,
        description="MetaTrader 5 account login number.",
    )

    @field_validator(
        "macro_enabled",
        "mt5_mock",
        "ai_regime_enabled",
        "range_reversal_confirm_enabled",
        "sl_fitness_judge_enabled",
        "synthetic_zones_enabled",
        "ai_mode_router_enabled",
        "range_trend_filter_enabled",
        "range_ai_regime_gate_enabled",
        "range_require_regime_valid",
        "mode_router_trending_dominance_enabled",
        "range_ai_direction_entry_gate_enabled",
        "spread_gate_enabled",
        mode="before",
    )
    @classmethod
    def _parse_bool_lenient(cls, v: object) -> object:
        """Strip whitespace + tolerate loose boolean strings from bat env.

        Windows `set VAR=false` ships with trailing whitespace/CR (`'false '`)
        which pydantic's strict bool parser rejects. This validator strips
        first, then normalises common truthy/falsy strings.
        """
        if isinstance(v, str):
            stripped = v.strip().lower()
            if stripped in {"true", "1", "yes", "on"}:
                return True
            if stripped in {"false", "0", "no", "off", ""}:
                return False
        return v

    @field_validator("journal_suffix", mode="before")
    @classmethod
    def _parse_suffix_lenient(cls, v: object) -> object:
        """Strip whitespace from journal_suffix env to handle Windows bat trailing space."""
        if isinstance(v, str):
            return v.strip()
        return v

    @field_validator("mt5_login", mode="before")
    @classmethod
    def _parse_mt5_login_lenient(cls, v: object) -> object:
        """Treat comment placeholders or non-numeric strings as 0.

        `.env.production` template historically shipped with a comment
        like `# Your MT5 demo account number` as the literal value,
        which pydantic would reject. This validator normalises such
        placeholders to 0 so the rest of the config loads; callers
        should use `has_mt5_credentials()` to check whether MT5 is
        actually configured before attempting connection.
        """
        if isinstance(v, str):
            stripped = v.strip()
            if not stripped or stripped.startswith("#"):
                return 0
            try:
                return int(stripped)
            except ValueError:
                return 0
        return v
    mt5_password: SecretStr = Field(
        default=SecretStr(""),
        description="MetaTrader 5 account password.",
    )
    mt5_server: str = Field(
        default="",
        description="MetaTrader 5 broker server address.",
    )

    # ------------------------------------------------------------------
    # Trading
    # ------------------------------------------------------------------
    instrument: str = Field(
        default="XAUUSD",
        description="Primary trading instrument symbol as used by MT5.",
    )
    risk_per_trade_pct: float = Field(
        default=1.0,
        ge=0.01,
        le=10.0,
        description="Account equity to risk per trade, as a percentage (e.g. 1.0 = 1%).",
    )
    max_daily_loss_pct: float = Field(
        default=3.0,
        ge=0.01,
        le=20.0,
        description="Maximum daily loss as a percentage of account equity before halting.",
    )
    max_drawdown_pct: float = Field(
        default=10.0,
        ge=0.1,
        le=50.0,
        description="Maximum drawdown from equity peak before halting all trading.",
    )
    max_lot_size: float = Field(
        default=0.01,
        ge=0.01,
        description="Hard cap on lot size per order. Prevents accidental overleveraging.",
    )

    # ------------------------------------------------------------------
    # SMC Detection Parameters
    # ------------------------------------------------------------------
    swing_length: int = Field(
        default=10,
        ge=1,
        description="Default number of candles on each side to confirm a swing high/low.",
    )
    swing_length_d1: int = Field(
        default=5,
        ge=1,
        description="Swing confirmation window for D1 timeframe.",
    )
    swing_length_h4: int = Field(
        default=7,
        ge=1,
        description="Swing confirmation window for H4 timeframe.",
    )
    swing_length_h1: int = Field(
        default=10,
        ge=1,
        description="Swing confirmation window for H1 timeframe.",
    )
    swing_length_m15: int = Field(
        default=10,
        ge=1,
        description="Swing confirmation window for M15 timeframe.",
    )
    liquidity_tolerance_points: float = Field(
        default=5.0,
        ge=0.0,
        description="Tolerance in points for clustering equal-high/low liquidity levels.",
    )
    ob_lookback: int = Field(
        default=50,
        ge=1,
        description="Number of bars to look back when scanning for order blocks.",
    )
    min_confluence_score: float = Field(
        default=0.45,
        ge=0.0,
        le=1.0,
        description="Minimum confluence score to accept a trade setup.",
    )
    max_concurrent_setups: int = Field(
        default=3,
        ge=1,
        description="Maximum number of concurrent active trade setups.",
    )
    min_rr_ratio: float = Field(
        default=2.0,
        ge=0.5,
        description="Minimum reward-to-risk ratio to qualify a setup.",
    )

    # ------------------------------------------------------------------
    # Regime Filter (Sprint 3)
    # ------------------------------------------------------------------
    regime_atr_trending_pct: float = Field(
        default=1.4,
        ge=0.1,
        description="D1 ATR(14) as % of price above which market is 'trending'.",
    )
    regime_atr_ranging_pct: float = Field(
        default=1.0,
        ge=0.1,
        description="D1 ATR(14) as % of price below which market is 'ranging'.",
    )

    # ------------------------------------------------------------------
    # ATR-Adaptive Stop-Loss (Sprint 4)
    # ------------------------------------------------------------------
    sl_atr_multiplier: float = Field(
        default=0.75,
        ge=0.1,
        le=3.0,
        description="Multiplier applied to H1 ATR(14) for adaptive SL buffer.",
    )
    sl_min_buffer_points: float = Field(
        default=200.0,
        ge=50.0,
        description="Minimum SL buffer in points, floor for the ATR-adaptive computation.",
    )
    enable_ob_test_trigger: bool = Field(
        default=False,
        description="Enable the ob_test_rejection entry trigger. Disabled in Sprint 5 due to 18.2% WR.",
    )
    zone_cooldown_hours: int = Field(
        default=24,
        ge=0,
        description="Hours to lock out a zone after it produces a losing trade.",
    )
    ai_regime_enabled: bool = Field(
        default=False,
        description="Enable AI-powered regime classification. When False, uses Sprint 5 ATR fallback.",
    )
    ai_regime_min_confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum AI confidence to use AI regime. Below this, ATR fallback activates.",
    )
    ai_mode_router_enabled: bool = Field(
        default=False,
        description=(
            "Round 7 P0-1: when True, mode_router consults the AI regime "
            "classifier output (AIRegimeAssessment) for regime-aware mode "
            "selection — forces 'trending' on TREND_UP/TREND_DOWN/"
            "ATH_BREAKOUT, allows 'ranging' on CONSOLIDATION, forces "
            "'v1_passthrough' on TRANSITION.  Default False preserves "
            "Round 4 v5 Priority 1-3 behaviour.  Toggle via "
            "SMC_AI_MODE_ROUTER_ENABLED after backtest validates on "
            "2020-2024 dual-mode A/B."
        ),
    )
    ai_regime_trust_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description=(
            "Round 7 P0-1: minimum AIRegimeAssessment.confidence required "
            "for the new AI-aware branch to fire.  Below this, the router "
            "falls through to the legacy Priority 1-3 logic unchanged."
        ),
    )
    max_concurrent_per_symbol: int = Field(
        default=3,
        ge=1,
        le=10,
        description=(
            "Hard cap on concurrent open positions per (symbol, magic) pair. "
            "Round 4 v5: prevents the 5-stacked-BUY disaster from 2026-04-20 02:46. "
            "Set to 1 to force one-position-per-symbol strict mode for A/B experiments."
        ),
    )
    anti_stack_cooldown_minutes: int = Field(
        default=30,
        ge=0,
        le=240,
        description=(
            "After opening a position, block further same-direction entries in "
            "this (symbol, magic) for N minutes. 0 disables. Prevents rapid "
            "same-direction stacking when max_concurrent_per_symbol > 1."
        ),
    )
    range_reversal_confirm_enabled: bool = Field(
        default=False,
        description=(
            "Round 4 v5 Task #52: require last closed M15 bar to confirm "
            "intended direction (bullish close for long, bearish for short) "
            "in addition to CHoCH/3-bar soft reversal. Addresses the "
            "2026-04-20 post-mortem where 5 stacked BUYs passed existing "
            "checks but had MFE < 0.10R (still-falling bars)."
        ),
    )
    range_trend_filter_enabled: bool = Field(
        default=False,
        description=(
            "Round 9 P0-A: when True, RangeTrader._build_setup blocks long "
            "support_bounce in a confirmed D1 downtrend (SMA50 slope <= "
            "-0.05%/bar AND close <= -1.0% below SMA50), and mirror for "
            "short. Default False preserves Round 8 production behavior. "
            "Toggle via SMC_RANGE_TREND_FILTER_ENABLED after backtest "
            "validates the regression on Tuesday 2026-04-21 fixture."
        ),
    )
    range_ai_regime_gate_enabled: bool = Field(
        default=False,
        description=(
            "Round 9 P0-B: when True, RangeTrader.generate_range_setups "
            "consults AIRegimeAssessment to per-direction gate setups. "
            "TREND_DOWN blocks range BUY, TREND_UP/ATH_BREAKOUT block "
            "range SELL, CONSOLIDATION allows both, TRANSITION tightens "
            "RR floor to 2.0. Activated only when assessment confidence "
            ">= 0.6. Default False. Toggle via "
            "SMC_RANGE_AI_REGIME_GATE_ENABLED."
        ),
    )
    range_require_regime_valid: bool = Field(
        default=False,
        description=(
            "Round 9 P0-C: when True, RangeTrader.detect_range invalidates "
            "any Donchian/OB/swing range when the AI regime classifier "
            "returns TREND_UP/TREND_DOWN/ATH_BREAKOUT with confidence "
            ">= 0.6 — a Donchian on a trend is a false range and "
            "range_trader should not operate. Default False preserves "
            "current production behavior. Toggle via "
            "SMC_RANGE_REQUIRE_REGIME_VALID."
        ),
    )
    mode_router_trending_dominance_enabled: bool = Field(
        default=False,
        description=(
            "R10 P2.1: when True, mode_router (a) lowers Priority-1 "
            "directional floor 0.45→0.30 when D1 SMA50 |slope| >= 0.05%/bar, "
            "and (b) demotes ranging→trending when D1 slope is confirmed AND "
            "ai_regime_assessment in (TREND_UP/TREND_DOWN/ATH_BREAKOUT) with "
            "confidence >= 0.6. Default False preserves R9 behavior. Toggle "
            "via SMC_MODE_ROUTER_TRENDING_DOMINANCE_ENABLED post-bake once "
            "gate_funnel.json validates the Phase-2 calibration."
        ),
    )
    range_ai_direction_entry_gate_enabled: bool = Field(
        default=False,
        description=(
            "R10 P2.2: when True, RangeTrader._build_setup vetoes a setup "
            "when the DirectionEngine's ai_direction confidently opposes it "
            "(confidence >= 0.55 AND opposite-direction). Default False "
            "preserves R9 behavior. Toggle via "
            "SMC_RANGE_AI_DIRECTION_ENTRY_GATE_ENABLED post-bake."
        ),
    )
    spread_gate_enabled: bool = Field(
        default=False,
        description=(
            "R10 P3.2: when True, live_demo blocks new opens whenever the "
            "live tick spread is >= 1.5x the rolling 20-bar median of the "
            "M15 bar spreads. Catches news spikes / illiquid windows. "
            "Default False — staged rollout: control leg keeps OFF while "
            "treatment leg flips ON via start_live_macro.bat once Phase 3 "
            "bake validates no false-blocks. Toggle via "
            "SMC_SPREAD_GATE_ENABLED."
        ),
    )
    # ------------------------------------------------------------------
    # Round 5 A-track — SL Fitness Judge (Task #7)
    # ------------------------------------------------------------------
    sl_fitness_judge_enabled: bool = Field(
        default=False,
        description=(
            "Round 5 Task #7: when True, the SL fitness judge runs in "
            "shadow mode inside the aggregator — emits "
            "sl_fitness_shadow_veto telemetry without actually vetoing "
            "any setup.  Default False keeps behaviour identical to "
            "Round 4 v5 live_demo.  Toggle via SMC_SL_FITNESS_JUDGE_ENABLED."
        ),
    )
    sl_fitness_min_sl_atr_ratio: float = Field(
        default=0.5,
        ge=0.0,
        le=2.0,
        description="Rule 2 floor: risk_points must be ≥ this × d1_atr_points.",
    )
    sl_fitness_max_sl_atr_ratio: float = Field(
        default=2.5,
        ge=0.5,
        le=10.0,
        description="Rule 3 ceiling: risk_points must be ≤ this × d1_atr_points.",
    )
    sl_fitness_low_vol_percentile: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Rule 4: H4 vol rank below this = low-vol (vetoes RR ≥ 2.5).",
    )
    sl_fitness_transition_conf_floor: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Rule 7: TRANSITION regime demands confluence ≥ this.",
    )
    sl_fitness_counter_trend_ai_conf: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Rule 1: AI confidence threshold for asserting counter-trend VETO.",
    )
    # ------------------------------------------------------------------
    # Round 5 A-track — SMC ATH synthetic zones (Task #9)
    # ------------------------------------------------------------------
    synthetic_zones_enabled: bool = Field(
        default=False,
        description=(
            "Round 5 Task #9: when True and historical zones < "
            "synthetic_zones_min_historical, the aggregator augments "
            "with synthetic zones (VWAP bands, session H/L, round "
            "numbers, prev-week H/L) during ATH regimes.  Addresses "
            "the 2024 W14+W15 setup drought (6 months zero setups "
            "during XAU's $2,300→$2,685 rally).  Default False for "
            "safety.  Toggle via SMC_SYNTHETIC_ZONES_ENABLED."
        ),
    )
    synthetic_zones_min_historical: int = Field(
        default=2,
        ge=0,
        le=10,
        description=(
            "Augment with synthetic zones when historical zone count "
            "is strictly less than this.  Default 2 — kick in when "
            "only 0 or 1 historical zones detected."
        ),
    )
    ath_reference_price: float = Field(
        default=3500.0,
        ge=100.0,
        description="All-time high reference for XAUUSD. Updated manually when ATH changes.",
    )
    tier2_confluence_floor: float = Field(
        default=0.55,
        ge=0.0,
        le=1.0,
        description="Minimum confluence for Tier 2 (H4-only) bias trades.",
    )
    tier3_confluence_floor: float = Field(
        default=0.55,
        ge=0.0,
        le=1.0,
        description="Minimum confluence for Tier 3 (D1-only) bias trades.",
    )

    # ------------------------------------------------------------------
    # Macro Overlay (Round 4 Alt-B)
    # ------------------------------------------------------------------
    macro_enabled: bool = Field(
        default=False,
        description=(
            "Enable macro overlay (COT/TIPS/DXY) in confluence scoring.  "
            "Default False for production safety during paper A/B testing.  "
            "Set SMC_MACRO_ENABLED=true (or ENABLE_MACRO=1 via shell alias) "
            "to activate the treatment leg."
        ),
    )
    macro_cache_ttl_hours: int = Field(
        default=24,
        ge=1,
        description="TTL for macro data cache in hours. 24h suits weekly COT + daily yields.",
    )
    fred_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="FRED API key for TIPS real-yield source (DFII10 series).",
    )

    # ------------------------------------------------------------------
    # A/B Paper Trading (Round 4 Alt-B W3)
    # ------------------------------------------------------------------
    journal_suffix: str = Field(
        default="",
        description=(
            "Suffix appended to the journal directory name and live_state filename.  "
            "Empty string (default) → data/{symbol}/journal/ and live_state.json.  "
            "Set SMC_JOURNAL_SUFFIX=_macro for the treatment leg so both processes "
            "write to separate journals: journal/ vs journal_macro/."
        ),
    )

    # ------------------------------------------------------------------
    # Dual-Magic A/B (audit-r4 v5 Option B)
    # ------------------------------------------------------------------
    macro_magic: int = Field(
        default=19760428,
        description=(
            "Magic number used by the treatment (macro) leg on the same TMGM Demo "
            "account.  Control leg uses cfg.magic (XAU=19760418, BTC=19760419); "
            "treatment leg overrides to this value so broker reconcile can split "
            "closed deals per-leg.  Set SMC_MACRO_MAGIC to override."
        ),
    )
    # NB: declared as plain str so pydantic-settings does not pre-parse it
    # as JSON (which would crash on malformed values).  We parse + validate
    # in the public accessor instead, yielding the sanitised dict.
    # Field name directly matches env var: SMC_VIRTUAL_BALANCE_SPLIT.
    virtual_balance_split_raw: str = Field(
        default='{"": 0.5, "_macro": 0.5}',
        validation_alias="SMC_VIRTUAL_BALANCE_SPLIT",
        description=(
            "Virtual balance split per journal_suffix for dual-magic sizing (JSON "
            "dict).  Each leg sees mt5_balance * split[suffix] when computing "
            "position size, preventing treatment leg from over-sizing using the "
            "full account equity shared with control.  Default 50/50.  Invalid "
            "JSON / out-of-range values fall back to the 50/50 default."
        ),
    )

    @property
    def virtual_balance_split(self) -> dict[str, float]:
        """Parse + sanitise the split dict.  Invalid values fail back to 50/50."""
        raw = self.virtual_balance_split_raw
        fallback = {"": 0.5, "_macro": 0.5}
        if isinstance(raw, dict):
            parsed = raw  # already a dict (e.g. init-time override)
        else:
            import json as _json
            stripped = str(raw).strip() if raw is not None else ""
            if not stripped:
                return fallback
            try:
                parsed = _json.loads(stripped)
            except (ValueError, TypeError):
                return fallback
        if not isinstance(parsed, dict) or not parsed:
            return fallback
        out: dict[str, float] = {}
        for key, val in parsed.items():
            try:
                f = float(val)
            except (TypeError, ValueError):
                f = 0.5
            if f <= 0.0 or f > 1.0:
                f = 0.5
            out[str(key)] = f
        return out if out else fallback

    def virtual_balance_for(self, suffix: str, mt5_balance: float) -> float:
        """Compute virtual balance for the given suffix.

        Looks up ``virtual_balance_split[suffix]`` (defaults to 0.5 for
        unknown suffixes so a typo never silently over-sizes), then returns
        ``mt5_balance * split``.  Returns 0.0 when mt5_balance is None or
        non-positive (fail-closed).
        """
        if mt5_balance is None or mt5_balance <= 0:
            return 0.0
        split = self.virtual_balance_split.get(suffix, 0.5)
        return float(mt5_balance) * float(split)

    def magic_for(self, instrument_magic: int, suffix: str) -> int:
        """Resolve the magic number to use for the given (instrument, suffix).

        Control leg (suffix="") uses the instrument-specific magic from cfg.magic.
        Treatment leg (suffix != "") uses macro_magic so broker reconcile can
        split deals per leg on the same TMGM Demo account.
        """
        if not suffix:
            return int(instrument_magic)
        return int(self.macro_magic)

    # ------------------------------------------------------------------
    # LLM (Phase 3+)
    # ------------------------------------------------------------------
    anthropic_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="Anthropic API key for LLM-assisted trade reasoning.",
    )
    anthropic_model: str = Field(
        default="claude-opus-4-6",
        description="Anthropic model ID to use for LLM reasoning steps.",
    )
    llm_daily_budget_usd: float = Field(
        default=5.0,
        ge=0.0,
        description="Maximum daily spend on LLM API calls in USD. 0 = unlimited.",
    )

    # ------------------------------------------------------------------
    # Alerting (Phase 4)
    # ------------------------------------------------------------------
    telegram_bot_token: SecretStr = Field(
        default=SecretStr(""),
        description="Telegram Bot API token for trade alert notifications.",
    )
    telegram_chat_id: str = Field(
        default="",
        description="Telegram chat / channel ID to send alerts to.",
    )

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        """Normalise to uppercase and assert it is a valid Python logging level."""
        upper = v.upper()
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if upper not in valid:
            raise ValueError(f"log_level must be one of {valid}, got {v!r}")
        return upper

    @field_validator("data_dir")
    @classmethod
    def _resolve_data_dir(cls, v: Path) -> Path:
        """Expand user home (~) and make the path absolute."""
        return v.expanduser().resolve()

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def is_live(self) -> bool:
        """Return True only when the system is in live trading mode."""
        return self.env == "live"

    def is_paper(self) -> bool:
        """Return True when running paper (simulated) trading."""
        return self.env == "paper"

    def is_dev(self) -> bool:
        """Return True in development / backtesting mode."""
        return self.env == "dev"

    def has_mt5_credentials(self) -> bool:
        """Return True if all three MT5 connection fields are set."""
        return bool(self.mt5_login and self.mt5_password.get_secret_value() and self.mt5_server)

    def has_llm(self) -> bool:
        """Return True if an Anthropic API key is configured."""
        return bool(self.anthropic_api_key.get_secret_value())

    def has_telegram(self) -> bool:
        """Return True if Telegram alerting is configured."""
        return bool(self.telegram_bot_token.get_secret_value() and self.telegram_chat_id)
