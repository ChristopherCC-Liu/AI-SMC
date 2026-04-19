"""Unit tests for smc.ai.macro_layer — Alt-B macro overlay W1D1-D5.

Covers:
    - MacroLayer.compute_macro_bias aggregation math
    - Clamping to ±0.3
    - Graceful degradation when sources fail
    - DXY source wiring through ExternalContextFetcher
    - Direction classification thresholds
    - COT fetcher: parse logic, signal math, network fallback (plan §2)
    - TIPS real-yield fetcher: FRED parse, 20d formula, signal mapping,
      network fallback (plan §3)

All 3 macro sources are now fully implemented.
"""

from __future__ import annotations

import datetime as _dt
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from smc.ai.cot_fetcher import COTFetcher, compute_cot_bias, compute_cot_net_long_pct
from smc.ai.macro_layer import MacroBias, MacroLayer
from smc.ai.models import ExternalContext
from smc.ai.tips_fetcher import TIPSFetcher, compute_tips_bias, compute_tips_yield_change


def _make_external_context(
    dxy_direction: str = "flat",
    dxy_value: float | None = 105.0,
    quality: str = "live",
) -> ExternalContext:
    """Helper to build a valid ExternalContext for patching."""
    return ExternalContext(
        dxy_direction=dxy_direction,  # type: ignore[arg-type]
        dxy_value=dxy_value,
        vix_level=18.0,
        vix_regime="normal",
        real_rate_10y=2.1,
        cot_net_spec=None,
        central_bank_stance=None,
        fetched_at=datetime.now(timezone.utc),
        source_quality=quality,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# MacroLayer.compute_macro_bias — aggregation + edge cases
# ---------------------------------------------------------------------------


class TestComputeMacroBias:
    def test_all_sources_fail_returns_neutral(self, tmp_path, monkeypatch) -> None:
        """When all 3 sources fail (COT fetch returns None, TIPS returns [],
        DXY unavailable) the layer must return neutral bias with 0 sources,
        never raise.

        COT and TIPS are explicitly stubbed to None/[] so this is a genuine
        unit test — no network dependency.
        """
        layer = MacroLayer(cache_dir=tmp_path)
        monkeypatch.setattr(layer, "_safe_cot_bias", lambda _: None)
        monkeypatch.setattr(layer, "_safe_yield_bias", lambda: None)
        with patch.object(
            layer._external,
            "fetch",
            return_value=_make_external_context(quality="unavailable", dxy_value=None),
        ):
            bias = layer.compute_macro_bias("XAUUSD")

        assert bias.sources_available == 0
        assert bias.total_bias == 0.0
        assert bias.direction == "neutral"
        assert bias.cot_bias == 0.0
        assert bias.yield_bias == 0.0
        assert bias.dxy_bias == 0.0

    def test_dxy_weakening_contributes_positive_bias(self, tmp_path, monkeypatch) -> None:
        """USD weakening → gold bullish → positive bias.

        COT and TIPS stubbed so only DXY contributes.
        """
        layer = MacroLayer(cache_dir=tmp_path)
        monkeypatch.setattr(layer, "_safe_cot_bias", lambda _: None)
        monkeypatch.setattr(layer, "_safe_yield_bias", lambda: None)
        with patch.object(
            layer._external,
            "fetch",
            return_value=_make_external_context(dxy_direction="weakening"),
        ):
            bias = layer.compute_macro_bias("XAUUSD")

        assert bias.sources_available == 1  # only DXY in this scenario
        assert bias.dxy_bias == 0.05
        assert bias.total_bias == 0.05
        # 0.05 is exactly at the direction threshold → neutral
        assert bias.direction == "neutral"

    def test_dxy_strengthening_contributes_negative_bias(self, tmp_path, monkeypatch) -> None:
        """USD strengthening → gold bearish → negative bias."""
        layer = MacroLayer(cache_dir=tmp_path)
        monkeypatch.setattr(layer, "_safe_cot_bias", lambda _: None)
        monkeypatch.setattr(layer, "_safe_yield_bias", lambda: None)
        with patch.object(
            layer._external,
            "fetch",
            return_value=_make_external_context(dxy_direction="strengthening"),
        ):
            bias = layer.compute_macro_bias("XAUUSD")

        assert bias.dxy_bias == -0.05
        assert bias.total_bias == -0.05
        assert bias.direction == "neutral"  # at threshold, not beyond

    def test_dxy_flat_contributes_zero(self, tmp_path, monkeypatch) -> None:
        """Flat DXY → zero bias but source is still counted as available."""
        layer = MacroLayer(cache_dir=tmp_path)
        monkeypatch.setattr(layer, "_safe_cot_bias", lambda _: None)
        monkeypatch.setattr(layer, "_safe_yield_bias", lambda: None)
        with patch.object(
            layer._external,
            "fetch",
            return_value=_make_external_context(dxy_direction="flat"),
        ):
            bias = layer.compute_macro_bias("XAUUSD")

        assert bias.dxy_bias == 0.0
        assert bias.total_bias == 0.0
        assert bias.direction == "neutral"
        assert bias.sources_available == 1  # DXY fetched successfully, even if 0

    def test_result_is_frozen_dataclass(self, tmp_path, monkeypatch) -> None:
        """MacroBias must be immutable per project coding-style rules."""
        layer = MacroLayer(cache_dir=tmp_path)
        monkeypatch.setattr(layer, "_safe_cot_bias", lambda _: None)
        monkeypatch.setattr(layer, "_safe_yield_bias", lambda: None)
        with patch.object(
            layer._external,
            "fetch",
            return_value=_make_external_context(dxy_direction="flat"),
        ):
            bias = layer.compute_macro_bias("XAUUSD")

        with pytest.raises((AttributeError, Exception)):
            bias.total_bias = 0.99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# DXY source — direct tests
# ---------------------------------------------------------------------------


class TestDXYBias:
    def test_dxy_bias_raises_when_fetcher_unavailable(self, tmp_path) -> None:
        """When ExternalContextFetcher reports unavailable, _dxy_bias raises."""
        layer = MacroLayer(cache_dir=tmp_path)
        with patch.object(
            layer._external,
            "fetch",
            return_value=_make_external_context(quality="unavailable", dxy_value=None),
        ):
            with pytest.raises(RuntimeError, match="DXY data unavailable"):
                layer._dxy_bias()

    def test_safe_dxy_bias_swallows_runtime_error(self, tmp_path) -> None:
        """_safe_dxy_bias must never raise — returns None on error."""
        layer = MacroLayer(cache_dir=tmp_path)
        with patch.object(
            layer._external,
            "fetch",
            return_value=_make_external_context(quality="unavailable", dxy_value=None),
        ):
            result = layer._safe_dxy_bias()
        assert result is None


# ---------------------------------------------------------------------------
# Stubbed sources — must not crash the aggregation
# ---------------------------------------------------------------------------


class TestStubbedSourcesGracefulDegrade:
    def test_cot_bias_raises_runtime_on_empty_data(self, tmp_path) -> None:
        """COT source raises RuntimeError when fetcher returns no data.

        No network in unit tests — COTFetcher.fetch() returns None with
        an empty cache dir, which _cot_bias converts to RuntimeError.
        """
        layer = MacroLayer(cache_dir=tmp_path)
        with patch("smc.ai.macro_layer.COTFetcher") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.fetch.return_value = None
            mock_cls.return_value = mock_instance
            with pytest.raises(RuntimeError, match="COT fetch returned no data"):
                layer._cot_bias("XAUUSD")

    def test_yield_bias_raises_runtime_on_empty_data(self, tmp_path) -> None:
        """TIPS source raises RuntimeError when fetcher returns empty list.

        No network + no FRED key + empty cache dir → TIPSFetcher returns []
        → _yield_bias raises RuntimeError.
        """
        layer = MacroLayer(cache_dir=tmp_path)
        with patch("smc.ai.macro_layer.TIPSFetcher") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.fetch_history.return_value = []
            mock_cls.return_value = mock_instance
            with pytest.raises(RuntimeError, match="TIPS fetch returned no data"):
                layer._yield_bias()

    def test_safe_cot_returns_none_on_fetch_failure(self, tmp_path) -> None:
        """Safe wrapper converts RuntimeError (no COT data) to None cleanly."""
        layer = MacroLayer(cache_dir=tmp_path)
        with patch("smc.ai.macro_layer.COTFetcher") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.fetch.return_value = None
            mock_cls.return_value = mock_instance
            result = layer._safe_cot_bias("XAUUSD")
        assert result is None

    def test_safe_yield_returns_none_when_no_data(self, tmp_path) -> None:
        """_safe_yield_bias returns None when TIPSFetcher returns empty list."""
        layer = MacroLayer(cache_dir=tmp_path)
        with patch("smc.ai.macro_layer.TIPSFetcher") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.fetch_history.return_value = []
            mock_cls.return_value = mock_instance
            assert layer._safe_yield_bias() is None


# ---------------------------------------------------------------------------
# Clamping — ensure ±0.3 cap holds even if future sources overshoot
# ---------------------------------------------------------------------------


class TestClamping:
    def test_total_clamped_to_positive_max(self, tmp_path, monkeypatch) -> None:
        """Even if future sources all return 0.5, total stays at +0.3."""
        layer = MacroLayer(cache_dir=tmp_path)
        monkeypatch.setattr(layer, "_safe_cot_bias", lambda _: 0.5)
        monkeypatch.setattr(layer, "_safe_yield_bias", lambda: 0.5)
        monkeypatch.setattr(layer, "_safe_dxy_bias", lambda: 0.5)

        bias = layer.compute_macro_bias("XAUUSD")
        assert bias.total_bias == 0.3
        assert bias.sources_available == 3
        assert bias.direction == "bullish"

    def test_total_clamped_to_negative_max(self, tmp_path, monkeypatch) -> None:
        layer = MacroLayer(cache_dir=tmp_path)
        monkeypatch.setattr(layer, "_safe_cot_bias", lambda _: -0.5)
        monkeypatch.setattr(layer, "_safe_yield_bias", lambda: -0.5)
        monkeypatch.setattr(layer, "_safe_dxy_bias", lambda: -0.5)

        bias = layer.compute_macro_bias("XAUUSD")
        assert bias.total_bias == -0.3
        assert bias.direction == "bearish"

    def test_mixed_sources_sum_correctly(self, tmp_path, monkeypatch) -> None:
        """COT bullish + yield bearish + DXY bullish → small positive net."""
        layer = MacroLayer(cache_dir=tmp_path)
        monkeypatch.setattr(layer, "_safe_cot_bias", lambda _: 0.10)
        monkeypatch.setattr(layer, "_safe_yield_bias", lambda: -0.05)
        monkeypatch.setattr(layer, "_safe_dxy_bias", lambda: 0.05)

        bias = layer.compute_macro_bias("XAUUSD")
        assert abs(bias.total_bias - 0.10) < 1e-9
        assert bias.sources_available == 3
        assert bias.direction == "bullish"  # 0.10 > 0.05 threshold


# ---------------------------------------------------------------------------
# Direction classification thresholds
# ---------------------------------------------------------------------------


class TestDirectionClassification:
    def test_direction_bullish_above_threshold(self, tmp_path, monkeypatch) -> None:
        layer = MacroLayer(cache_dir=tmp_path)
        monkeypatch.setattr(layer, "_safe_cot_bias", lambda _: 0.08)
        monkeypatch.setattr(layer, "_safe_yield_bias", lambda: None)
        monkeypatch.setattr(layer, "_safe_dxy_bias", lambda: None)

        bias = layer.compute_macro_bias("XAUUSD")
        assert bias.direction == "bullish"

    def test_direction_bearish_below_threshold(self, tmp_path, monkeypatch) -> None:
        layer = MacroLayer(cache_dir=tmp_path)
        monkeypatch.setattr(layer, "_safe_cot_bias", lambda _: -0.08)
        monkeypatch.setattr(layer, "_safe_yield_bias", lambda: None)
        monkeypatch.setattr(layer, "_safe_dxy_bias", lambda: None)

        bias = layer.compute_macro_bias("XAUUSD")
        assert bias.direction == "bearish"

    def test_direction_neutral_at_threshold(self, tmp_path, monkeypatch) -> None:
        """Boundary: exactly 0.05 should be neutral (strict >)."""
        layer = MacroLayer(cache_dir=tmp_path)
        monkeypatch.setattr(layer, "_safe_cot_bias", lambda _: 0.05)
        monkeypatch.setattr(layer, "_safe_yield_bias", lambda: None)
        monkeypatch.setattr(layer, "_safe_dxy_bias", lambda: None)

        bias = layer.compute_macro_bias("XAUUSD")
        assert bias.direction == "neutral"


# ---------------------------------------------------------------------------
# COT Fetcher — parse logic, signal math, network fallback (plan §2)
# ---------------------------------------------------------------------------


def _make_cot_row(
    date_str: str,
    noncomm_long: int = 250_000,
    noncomm_short: int = 80_000,
    open_interest: int = 500_000,
) -> dict:
    """Helper to build a mock CFTC Socrata JSON row."""
    return {
        "report_date_as_yyyy_mm_dd": date_str,
        "noncomm_positions_long_all": str(noncomm_long),
        "noncomm_positions_short_all": str(noncomm_short),
        "open_interest_all": str(open_interest),
    }


def _make_cot_history_df(net_long_pcts: list[float]) -> pl.DataFrame:
    """Build a synthetic COT history DataFrame from a list of net long pct values."""
    n = len(net_long_pcts)
    base_date = _dt.date(2022, 1, 4)
    dates = [base_date + _dt.timedelta(weeks=i) for i in range(n)]
    return pl.DataFrame(
        {
            "report_date": dates,
            "noncomm_long": [100_000] * n,
            "noncomm_short": [100_000] * n,
            "open_interest": [500_000] * n,
            "cot_net_long_pct": net_long_pcts,
        }
    ).with_columns(pl.col("report_date").cast(pl.Date))


class TestCOTFetcherParsing:
    def test_cot_fetcher_parses_xau_row(self, tmp_path) -> None:
        """Mock Socrata JSON response with a valid gold row — verify net long pct."""
        mock_rows = [
            _make_cot_row("2026-04-15", noncomm_long=245_123, noncomm_short=89_321, open_interest=512_876),
        ]
        fetcher = COTFetcher(cache_path=tmp_path / "cot.parquet")
        df = fetcher._parse_rows(mock_rows)

        assert df is not None
        assert len(df) == 1
        expected_pct = (245_123 - 89_321) / 512_876 * 100.0
        assert abs(float(df["cot_net_long_pct"][0]) - expected_pct) < 1e-6

    def test_cot_fetcher_handles_empty_response(self, tmp_path) -> None:
        """Empty row list → _parse_rows returns None."""
        fetcher = COTFetcher(cache_path=tmp_path / "cot.parquet")
        result = fetcher._parse_rows([])
        assert result is None

    def test_cot_fetcher_handles_http_error(self, tmp_path) -> None:
        """Network failure (ConnectionError) → fetch() returns None.

        ``requests`` is imported lazily inside _fetch_from_cftc, so we patch
        it via the standard ``requests.get`` path in the requests module.
        """
        import requests as _requests
        fetcher = COTFetcher(cache_path=tmp_path / "cot.parquet")
        with patch.object(_requests, "get", side_effect=ConnectionError("timeout")):
            result = fetcher.fetch()
        assert result is None


class TestCOTNetLongFormula:
    def test_cot_net_long_formula_basic(self) -> None:
        """Verify the net long pct formula: (long - short) / oi * 100."""
        result = compute_cot_net_long_pct(
            noncomm_long=300_000,
            noncomm_short=100_000,
            open_interest=500_000,
        )
        assert result is not None
        assert abs(result - 40.0) < 1e-9  # (300k - 100k) / 500k * 100 = 40.0

    def test_cot_net_long_formula_zero_oi_returns_none(self) -> None:
        """Division by zero guard: zero open interest → None."""
        result = compute_cot_net_long_pct(noncomm_long=100_000, noncomm_short=50_000, open_interest=0)
        assert result is None

    def test_cot_net_long_formula_negative_net(self) -> None:
        """Short-heavy positioning → negative value."""
        result = compute_cot_net_long_pct(
            noncomm_long=80_000,
            noncomm_short=200_000,
            open_interest=500_000,
        )
        assert result is not None
        assert result < 0.0


class TestCOTPercentileRolling104w:
    def test_cot_percentile_rolling_104w_sufficient_data(self) -> None:
        """With 120 weeks of synthetic data, signal must use the most recent 104 entries."""
        # Generate a smooth range so we know exactly where each value ranks
        # Values from -30 to +60 (range of 90 across 120 points)
        n = 120
        pcts = [-30.0 + (i * 90.0 / (n - 1)) for i in range(n)]
        df = _make_cot_history_df(pcts)

        # Latest value is +60 — the maximum — should be at the very top of window
        bias = compute_cot_bias(df)
        # 60.0 is max of the last 104 values, percentile ≈ 1.0 → > 0.90 → -0.10
        assert bias == -0.10

    def test_cot_percentile_min_value_bullish(self) -> None:
        """Minimum value in window (≈ 0th percentile) → contrarian bullish +0.10."""
        n = 50
        # Last value is the minimum
        pcts = [float(i) for i in range(1, n + 1)]
        pcts[-1] = 0.0  # replace last with minimum
        df = _make_cot_history_df(pcts)

        bias = compute_cot_bias(df)
        # Latest is 0.0 — below all others → percentile ≈ 0 → < 0.10 → +0.10
        assert bias == +0.10


class TestCOTBiasSignal:
    def test_cot_extreme_long_returns_bearish_bias(self) -> None:
        """Feed a dataset where the latest value is at the 95th+ percentile → -0.10."""
        # 20 values from 0 to 19; latest=19 is max → percentile 1.0 → > 0.90
        pcts = [float(i) for i in range(20)]
        df = _make_cot_history_df(pcts)
        assert compute_cot_bias(df) == -0.10

    def test_cot_extreme_short_returns_bullish_bias(self) -> None:
        """Feed a dataset where the latest value is at the sub-10th percentile → +0.10."""
        # 20 values from 1 to 20; replace latest with 0 (minimum)
        pcts = [float(i) for i in range(1, 21)]
        pcts[-1] = 0.0
        df = _make_cot_history_df(pcts)
        assert compute_cot_bias(df) == +0.10

    def test_cot_mid_range_returns_zero_bias(self) -> None:
        """Latest value at the 50th percentile → neutral mid-range → 0.0 bias.

        Build 21 evenly-spaced values from 0..20.  The latest value is 10.0
        which sits at rank 10/20 = 50th percentile → within 30–70 band → 0.0.
        """
        pcts = [float(i) for i in range(21)]  # 0, 1, 2, … 20; latest = 20 is index -1
        # We want the latest to be at the exact middle.  Reconstruct so latest = 10.
        # Use 20 values in ascending order, then the median in the 21st slot.
        base = list(range(0, 20))  # 0..19, latest = 19 (max) — that's wrong.
        # Instead: put a mid-value (10) at the end of a uniform spread.
        # 20 values from 0..19, then append 10 as the "latest".
        pcts = list(range(20)) + [10]  # latest = 10
        df = _make_cot_history_df([float(v) for v in pcts])
        # Window = 21 values. Values below 10: count = 10 (i.e. 0..9).
        # pct_rank = 10/21 ≈ 0.476 → in [0.30, 0.70] neutral band.
        assert compute_cot_bias(df) == 0.0


class TestCOTNetworkFailFallback:
    def test_cot_network_fail_returns_zero_via_macro_layer(self, tmp_path) -> None:
        """When CFTC network fails and cache is empty, _cot_bias raises RuntimeError
        which _safe_cot_bias converts to None → compute_macro_bias contributes 0.0."""
        layer = MacroLayer(cache_dir=tmp_path)

        # Patch COTFetcher.fetch to simulate total network failure (no cache, no data)
        with patch("smc.ai.macro_layer.COTFetcher") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.fetch.return_value = None
            mock_cls.return_value = mock_instance

            # Also patch DXY and yield so they don't interfere
            with patch.object(layer, "_safe_dxy_bias", return_value=None):
                with patch.object(layer, "_safe_yield_bias", return_value=None):
                    bias = layer.compute_macro_bias("XAUUSD")

        assert bias.cot_bias == 0.0
        assert bias.sources_available == 0
        assert bias.total_bias == 0.0


# ---------------------------------------------------------------------------
# TIPS real-yield fetcher — parse, formula, signal mapping, fallback (plan §3)
# ---------------------------------------------------------------------------


def _make_fred_observations(values: list[float | str]) -> dict:
    """Build a mock FRED API JSON response from a list of values.

    Parameters
    ----------
    values:
        List of values (newest-first).  Use ``"."`` to simulate FRED
        missing-value markers.
    """
    obs = []
    for i, v in enumerate(values):
        obs.append({
            "realtime_start": "2026-01-01",
            "realtime_end": "2026-04-19",
            "date": f"2026-04-{19 - i:02d}",
            "value": str(v),
        })
    return {"observations": obs}


class TestTIPSFetcherParsesFredJson:
    def test_tips_fetcher_parses_fred_json(self, tmp_path) -> None:
        """Mock FRED JSON with 25 observations; verify parse returns expected floats.

        Tests that:
        - ``"."`` missing-value markers are skipped
        - Values are returned newest-first as floats
        - List length matches number of valid (non-``"."``) entries
        """
        raw_values: list[float | str] = [2.5, 2.4, ".", 2.3, 2.2,  # indices 0-4 (5 valid)
                                          2.1, 2.0, 1.9, 1.8, 1.7,  # indices 5-9
                                          1.6, 1.5, 1.4, 1.3, 1.2,  # indices 10-14
                                          1.1, 1.0, 0.9, 0.8, 0.7,  # indices 15-19
                                          0.6, 0.5, 0.4, 0.3, 0.2]  # indices 20-24
        mock_data = _make_fred_observations(raw_values)
        # 25 raw entries - 1 "." = 24 valid
        result = TIPSFetcher._parse_fred_response(mock_data)
        assert result is not None
        assert len(result) == 24
        # First value should be the most recent non-"." value (2.5)
        assert abs(result[0] - 2.5) < 1e-9
        # All values are floats
        assert all(isinstance(v, float) for v in result)


class TestTIPSRealYieldChange20d:
    def test_tips_real_yield_change_20d_formula(self) -> None:
        """Verify the 20d change formula: recent_5d_avg - older_20d_avg.

        Build a synthetic 30-observation series (newest-first):
        - indices 0–4  (recent 5d):  all 3.0 → avg = 3.0
        - indices 15–24 (older 10d): all 2.0 → avg = 2.0
        - expected change: 3.0 - 2.0 = +1.0 pp (yields rose → bearish)
        """
        history = (
            [3.0] * 5   # recent indices 0–4
            + [2.5] * 10  # intermediate indices 5–14
            + [2.0] * 10  # older indices 15–24
            + [1.5] * 5   # extra beyond window
        )
        change = compute_tips_yield_change(history)
        assert change is not None
        assert abs(change - 1.0) < 1e-9

    def test_tips_yield_change_returns_none_when_history_too_short(self) -> None:
        """Fewer than 25 values → older window (indices 15-24) is incomplete → None."""
        history = [2.0] * 10  # only 10 values; older window starts at idx 15
        change = compute_tips_yield_change(history)
        assert change is None


class TestTIPSStrongRiseReturnsBearishBias:
    def test_tips_strong_rise_returns_bearish_bias(self) -> None:
        """Real yield rising ≥ 0.25 pp → -0.10 (gold bearish).

        Construct a history where recent avg = 2.50 and older avg = 2.00,
        giving change = +0.50 pp ≥ +0.25 threshold → -0.10.
        """
        history = (
            [2.50] * 5   # recent 5d
            + [2.25] * 10  # intermediate
            + [2.00] * 10  # older 10d
        )
        bias = compute_tips_bias(history)
        assert bias == -0.10

    def test_tips_mild_rise_returns_mild_bearish_bias(self) -> None:
        """Real yield rising ≥ 0.10 pp but < 0.25 pp → -0.05."""
        history = (
            [2.15] * 5
            + [2.05] * 10
            + [2.00] * 10
        )
        bias = compute_tips_bias(history)
        assert bias == -0.05


class TestTIPSStrongFallReturnsBullishBias:
    def test_tips_strong_fall_returns_bullish_bias(self) -> None:
        """Real yield falling ≤ -0.25 pp → +0.10 (gold bullish).

        Construct a history where recent avg = 1.50 and older avg = 2.00,
        giving change = -0.50 pp ≤ -0.25 threshold → +0.10.
        """
        history = (
            [1.50] * 5   # recent 5d
            + [1.75] * 10  # intermediate
            + [2.00] * 10  # older 10d
        )
        bias = compute_tips_bias(history)
        assert bias == +0.10

    def test_tips_mild_fall_returns_mild_bullish_bias(self) -> None:
        """Real yield falling ≤ -0.10 pp but > -0.25 pp → +0.05."""
        history = (
            [1.85] * 5
            + [1.92] * 10
            + [2.00] * 10
        )
        bias = compute_tips_bias(history)
        assert bias == +0.05


class TestTIPSNetworkFailReturnsZero:
    def test_tips_network_fail_returns_zero(self, tmp_path) -> None:
        """Network failure + no FRED key + empty cache → _yield_bias raises
        RuntimeError → _safe_yield_bias returns None → macro_layer contributes 0.0.

        End-to-end path through MacroLayer with COT and DXY also patched out.
        """
        layer = MacroLayer(cache_dir=tmp_path, fred_api_key=None)

        with patch("smc.ai.macro_layer.TIPSFetcher") as mock_tips_cls:
            mock_tips = MagicMock()
            mock_tips.fetch_history.return_value = []
            mock_tips_cls.return_value = mock_tips

            with patch.object(layer, "_safe_cot_bias", return_value=None):
                with patch.object(layer, "_safe_dxy_bias", return_value=None):
                    bias = layer.compute_macro_bias("XAUUSD")

        assert bias.yield_bias == 0.0
        assert bias.sources_available == 0
        assert bias.total_bias == 0.0
