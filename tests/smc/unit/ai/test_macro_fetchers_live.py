"""Integration-style tests for the 3 macro fetchers after Round 4 Alt-B fixes.

These tests **patch** the underlying network libraries (``yfinance`` and
``requests``) with realistic synthetic responses, then assert that each
fetcher parses / falls back / logs correctly.  They are NOT live network
tests — the patches make them fully deterministic and CI-safe.

Covered scenarios (per Round 4 urgent-fix spec):

1. DXY primary ticker (``DX-Y.NYB``) returns a usable history → signal
   produced; verify real-world-plausible value.
2. DXY primary empty → fallback to ``DX=F`` → then ``UUP`` ETF proxy.
3. COT fetcher consumes the corrected CFTC Legacy-Futures-Only endpoint
   (``6dca-aqww``) with ``cftc_contract_market_code='088691'`` and sends
   the polite ``User-Agent`` header.
4. TIPS: FRED 200 response → signal via primary path.
5. TIPS: FRED returns 401 (bad / missing key) → yfinance TIP ETF fallback
   kicks in and produces a non-empty proxy series.
6. All paths fail silently (never raise) — ``compute_macro_bias`` returns
   a neutral ``MacroBias`` with ``sources_available=0``.

The tests avoid hitting live endpoints so they run in < 1s and can run
offline.  Round 4 Lead runs the probe script (``scripts/probe_macro.py``)
on the VPS for live verification.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from smc.ai.cot_fetcher import (
    _CFTC_URL,
    _GOLD_MARKET_CODE,
    _USER_AGENT,
    COTFetcher,
)
from smc.ai.external_context import (
    _DXY_TICKERS,
    _fetch_dxy,
)
from smc.ai.macro_layer import MacroLayer
from smc.ai.tips_fetcher import (
    _SOURCE_FRED,
    _SOURCE_TIP_PROXY,
    _TIP_ETF_TICKER,
    TIPSFetcher,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_yf_hist(closes: list[float]) -> MagicMock:
    """Return a MagicMock Yahoo Finance Ticker backed by the given closes."""
    import pandas as pd

    mock_ticker = MagicMock()
    mock_ticker.history.return_value = pd.DataFrame({"Close": closes})
    return mock_ticker


def _make_yf_empty() -> MagicMock:
    """Return a MagicMock Ticker whose ``history`` is empty."""
    import pandas as pd

    mock_ticker = MagicMock()
    mock_ticker.history.return_value = pd.DataFrame({"Close": []})
    return mock_ticker


def _make_cftc_response(num_rows: int = 104) -> list[dict]:
    """Fabricate a Socrata response shaped like the live CFTC endpoint."""
    rows = []
    for i in range(num_rows):
        # Alternate long/short pressure so the percentile logic has structure
        long_ = 200_000 + (i * 500)
        short_ = 60_000 + ((num_rows - i) * 300)
        oi = 500_000
        rows.append(
            {
                "report_date_as_yyyy_mm_dd": f"2024-{((i % 12) + 1):02d}-{((i % 27) + 1):02d}",
                "noncomm_positions_long_all": str(long_),
                "noncomm_positions_short_all": str(short_),
                "open_interest_all": str(oi),
                "cftc_contract_market_code": _GOLD_MARKET_CODE,
                "market_and_exchange_names": "GOLD - COMMODITY EXCHANGE INC.",
            }
        )
    return rows


def _make_fred_response(values: list[float]) -> dict:
    """Build a minimal FRED observations payload (newest-first)."""
    return {
        "observations": [
            {"date": f"2026-04-{19 - i:02d}", "value": f"{v:.3f}"}
            for i, v in enumerate(values)
        ]
    }


# ---------------------------------------------------------------------------
# 1. DXY fetcher — primary + fallback chain
# ---------------------------------------------------------------------------


class TestDXYFetcherLive:
    def test_dxy_primary_ticker_succeeds(self) -> None:
        """Primary ticker DX-Y.NYB returns plausible closes → strengthening signal."""
        closes = [103.0, 103.2, 103.4, 103.8, 104.0, 104.2, 104.3, 104.5, 104.7, 104.9]
        with patch("yfinance.Ticker", return_value=_make_yf_hist(closes)) as mock_t:
            value, direction = _fetch_dxy()

        # Only one ticker was attempted (primary succeeded)
        assert mock_t.call_count == 1
        assert mock_t.call_args.args[0] == _DXY_TICKERS[0]  # "DX-Y.NYB"
        # Value is the last close
        assert value == pytest.approx(104.9)
        # 1.8% rise over 5 days → strengthening
        assert direction == "strengthening"
        # Real-world DXY index typically trades 90–115 → sanity bound
        assert 50.0 < value < 200.0

    def test_dxy_falls_back_to_second_ticker_when_primary_empty(self) -> None:
        """Empty history on DX-Y.NYB → fallback to DX=F."""
        # First call empty; second call populated
        empty = _make_yf_empty()
        populated = _make_yf_hist([102.0, 102.1, 102.2, 102.3, 102.4, 102.5, 102.6, 102.7, 102.8, 102.9])

        with patch(
            "yfinance.Ticker",
            side_effect=[empty, populated, populated],  # 3 slots just in case
        ) as mock_t:
            value, direction = _fetch_dxy()

        # At least 2 tickers attempted — primary empty, fallback succeeded
        assert mock_t.call_count >= 2
        assert mock_t.call_args_list[0].args[0] == _DXY_TICKERS[0]
        assert mock_t.call_args_list[1].args[0] == _DXY_TICKERS[1]
        assert value == pytest.approx(102.9)
        # 0.88% rise → strengthening
        assert direction == "strengthening"

    def test_dxy_all_tickers_fail_returns_none_flat(self) -> None:
        """All 3 tickers throw → (None, 'flat'), no exception leaks out."""
        with patch("yfinance.Ticker", side_effect=Exception("network")):
            value, direction = _fetch_dxy()

        assert value is None
        assert direction == "flat"


# ---------------------------------------------------------------------------
# 2. COT fetcher — corrected CFTC endpoint + UA header
# ---------------------------------------------------------------------------


class TestCOTFetcherLive:
    def test_cot_fetcher_hits_legacy_futures_endpoint_with_ua(self, tmp_path: Path) -> None:
        """Verify the fetcher targets the Legacy Futures-Only dataset with
        ``cftc_contract_market_code`` filter and a polite User-Agent."""
        import requests as _requests

        captured: dict = {}

        def fake_get(url: str, *, params: dict, headers: dict, timeout: int):
            captured["url"] = url
            captured["params"] = params
            captured["headers"] = headers
            captured["timeout"] = timeout
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            mock_resp.json.return_value = _make_cftc_response(num_rows=104)
            return mock_resp

        fetcher = COTFetcher(cache_path=tmp_path / "cot.parquet")
        with patch.object(_requests, "get", side_effect=fake_get):
            df = fetcher.fetch()

        # URL must be the corrected Socrata endpoint
        assert captured["url"] == _CFTC_URL
        assert "6dca-aqww" in _CFTC_URL  # defence against silent regression

        # The $where clause must filter by the correct field name
        where = captured["params"]["$where"]
        assert "cftc_contract_market_code" in where
        assert f"'{_GOLD_MARKET_CODE}'" in where

        # Polite User-Agent header must be set
        assert captured["headers"].get("User-Agent") == _USER_AGENT
        assert captured["headers"].get("Accept") == "application/json"

        # DataFrame parsed successfully
        assert df is not None
        assert len(df) == 104
        assert "cot_net_long_pct" in df.columns
        # Net long pct should be in a sensible range (-100% .. +100%)
        first_val = float(df["cot_net_long_pct"][0])
        assert -100.0 < first_val < 100.0

    def test_cot_fetcher_returns_none_on_http_error(self, tmp_path: Path) -> None:
        """HTTP errors must not raise — return None so safe wrapper degrades."""
        import requests as _requests

        fetcher = COTFetcher(cache_path=tmp_path / "cot.parquet")
        with patch.object(_requests, "get", side_effect=_requests.ConnectionError("boom")):
            result = fetcher.fetch()

        assert result is None


# ---------------------------------------------------------------------------
# 3. TIPS fetcher — FRED primary + yfinance TIP fallback
# ---------------------------------------------------------------------------


class TestTIPSFetcherLive:
    def test_tips_fetcher_fred_primary_success(self, tmp_path: Path) -> None:
        """FRED 200 with usable observations → series returned, cache marked 'fred'."""
        import polars as pl
        import requests as _requests

        values = [2.10, 2.09, 2.08, 2.07, 2.06, 2.05, 2.04, 2.03, 2.02, 2.01,
                  2.00, 1.99, 1.98, 1.97, 1.96, 1.95, 1.94, 1.93, 1.92, 1.91,
                  1.90, 1.89, 1.88, 1.87, 1.86, 1.85, 1.84, 1.83, 1.82, 1.81]
        fake_resp = MagicMock()
        fake_resp.raise_for_status = MagicMock()
        fake_resp.json.return_value = _make_fred_response(values)

        fetcher = TIPSFetcher(
            cache_path=tmp_path / "tips.parquet",
            fred_api_key="test-key-abcdef",
        )
        with patch.object(_requests, "get", return_value=fake_resp):
            history = fetcher.fetch_history()

        assert history == pytest.approx(values, rel=1e-9)
        # Cache marked with primary source
        cached = pl.read_parquet(tmp_path / "tips.parquet")
        assert "source" in cached.columns
        assert cached["source"][0] == _SOURCE_FRED

    def test_tips_fetcher_no_fred_key_uses_tip_etf_fallback(self, tmp_path: Path) -> None:
        """No FRED key → TIP ETF fallback produces a non-empty synthetic series
        and cache marks source as ``tip_etf_proxy``."""
        import polars as pl

        # 30 days of slowly rising TIP closes → synthetic yields declining →
        # downstream signal would be bullish (we only verify shape here).
        closes = [110.0 + 0.05 * i for i in range(30)]
        mock_ticker = _make_yf_hist(closes)

        fetcher = TIPSFetcher(cache_path=tmp_path / "tips.parquet", fred_api_key=None)
        with patch("yfinance.Ticker", return_value=mock_ticker) as mock_t:
            history = fetcher.fetch_history()

        mock_t.assert_called_once_with(_TIP_ETF_TICKER)
        assert len(history) > 0
        # Every delta is a signed percentage point — finite floats
        assert all(isinstance(v, float) for v in history)
        # Rising TIP (our input) → inverted deltas → negative values (yields falling)
        assert history[0] < 0.0  # newest-first; most recent delta should be negative

        # Cache marked with fallback provenance
        cached = pl.read_parquet(tmp_path / "tips.parquet")
        assert cached["source"][0] == _SOURCE_TIP_PROXY

    def test_tips_fetcher_fred_401_then_tip_etf_fallback(self, tmp_path: Path) -> None:
        """FRED returns HTTP 401 (bad key) → fetcher falls through to TIP ETF."""
        import polars as pl
        import requests as _requests

        # HTTPError on raise_for_status()
        err = _requests.HTTPError("401 Client Error: Unauthorized")
        fake_resp = MagicMock()
        fake_resp.raise_for_status = MagicMock(side_effect=err)

        closes = [108.0, 108.2, 108.4, 108.6, 108.8, 109.0,
                  109.1, 109.2, 109.3, 109.4, 109.5, 109.6,
                  109.7, 109.8, 109.9, 110.0, 110.1, 110.2,
                  110.3, 110.4, 110.5, 110.6, 110.7, 110.8,
                  110.9, 111.0, 111.1, 111.2, 111.3, 111.4]
        mock_ticker = _make_yf_hist(closes)

        fetcher = TIPSFetcher(
            cache_path=tmp_path / "tips.parquet",
            fred_api_key="bad-key",
        )
        with (
            patch.object(_requests, "get", return_value=fake_resp),
            patch("yfinance.Ticker", return_value=mock_ticker),
        ):
            history = fetcher.fetch_history()

        assert len(history) > 0
        cached = pl.read_parquet(tmp_path / "tips.parquet")
        assert cached["source"][0] == _SOURCE_TIP_PROXY

    def test_tips_fetcher_all_sources_fail_returns_empty(self, tmp_path: Path) -> None:
        """No key + yfinance down + no cache → empty list, no exception."""
        fetcher = TIPSFetcher(cache_path=tmp_path / "tips.parquet", fred_api_key=None)
        with patch("yfinance.Ticker", side_effect=Exception("offline")):
            history = fetcher.fetch_history()

        assert history == []


# ---------------------------------------------------------------------------
# 4. End-to-end MacroLayer graceful degradation
# ---------------------------------------------------------------------------


class TestMacroLayerGracefulDegradation:
    def test_all_fetchers_offline_neutral_bias(self, tmp_path: Path) -> None:
        """When every network backend fails, MacroLayer must still return a
        valid ``MacroBias`` with ``sources_available=0`` and never raise.

        This covers the Round 4 incident: VPS had yfinance missing, CFTC
        endpoint wrong, and no FRED key.  After the fix, the same total
        outage still produces a clean neutral bias (no stack traces).
        """
        import requests as _requests

        layer = MacroLayer(cache_dir=tmp_path, fred_api_key=None)
        with (
            patch("yfinance.Ticker", side_effect=Exception("no yfinance")),
            patch.object(_requests, "get", side_effect=_requests.ConnectionError("blocked")),
        ):
            bias = layer.compute_macro_bias("XAUUSD")

        assert bias.sources_available == 0
        assert bias.total_bias == 0.0
        assert bias.direction == "neutral"
        assert bias.cot_bias == 0.0
        assert bias.yield_bias == 0.0
        assert bias.dxy_bias == 0.0
