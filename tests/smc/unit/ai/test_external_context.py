"""Unit tests for smc.ai.external_context — macro data fetching + caching."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from smc.ai.external_context import (
    ExternalContextFetcher,
    _classify_vix_regime,
    _fetch_dxy,
    _fetch_real_rate,
    _fetch_vix,
)
from smc.ai.models import ExternalContext


# ---------------------------------------------------------------------------
# VIX regime classification
# ---------------------------------------------------------------------------


class TestClassifyVixRegime:
    def test_low(self) -> None:
        assert _classify_vix_regime(12.0) == "low"

    def test_normal(self) -> None:
        assert _classify_vix_regime(18.0) == "normal"

    def test_elevated(self) -> None:
        assert _classify_vix_regime(25.0) == "elevated"

    def test_extreme(self) -> None:
        assert _classify_vix_regime(35.0) == "extreme"

    def test_boundary_low_normal(self) -> None:
        assert _classify_vix_regime(15.0) == "normal"

    def test_boundary_normal_elevated(self) -> None:
        assert _classify_vix_regime(20.0) == "elevated"

    def test_boundary_elevated_extreme(self) -> None:
        assert _classify_vix_regime(30.0) == "extreme"


# ---------------------------------------------------------------------------
# VIX fetcher (mocked)
# ---------------------------------------------------------------------------


class TestFetchVix:
    @patch("smc.ai.external_context.yf", create=True)
    def test_success(self, mock_yf: MagicMock) -> None:
        import pandas as pd

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame({"Close": [18.5]})

        with patch("yfinance.Ticker", return_value=mock_ticker):
            vix, regime = _fetch_vix()

        assert vix == pytest.approx(18.5)
        assert regime == "normal"

    def test_failure_returns_none(self) -> None:
        with patch("yfinance.Ticker", side_effect=Exception("network error")):
            vix, regime = _fetch_vix()
        assert vix is None
        assert regime is None

    def test_empty_history_returns_none(self) -> None:
        import pandas as pd

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()

        with patch("yfinance.Ticker", return_value=mock_ticker):
            vix, regime = _fetch_vix()

        assert vix is None
        assert regime is None


# ---------------------------------------------------------------------------
# DXY fetcher (mocked)
# ---------------------------------------------------------------------------


class TestFetchDxy:
    def test_strengthening(self) -> None:
        import pandas as pd

        mock_ticker = MagicMock()
        # 10 days of data, price goes from 103 to 104 (~1% rise)
        closes = [103.0] * 5 + [103.5, 103.8, 104.0, 104.1, 104.2]
        mock_ticker.history.return_value = pd.DataFrame({"Close": closes})

        with patch("yfinance.Ticker", return_value=mock_ticker):
            value, direction = _fetch_dxy()

        assert value == pytest.approx(104.2)
        assert direction == "strengthening"

    def test_weakening(self) -> None:
        import pandas as pd

        mock_ticker = MagicMock()
        closes = [105.0] * 5 + [104.5, 104.0, 103.5, 103.0, 102.5]
        mock_ticker.history.return_value = pd.DataFrame({"Close": closes})

        with patch("yfinance.Ticker", return_value=mock_ticker):
            value, direction = _fetch_dxy()

        assert value == pytest.approx(102.5)
        assert direction == "weakening"

    def test_flat(self) -> None:
        import pandas as pd

        mock_ticker = MagicMock()
        closes = [103.0] * 10
        mock_ticker.history.return_value = pd.DataFrame({"Close": closes})

        with patch("yfinance.Ticker", return_value=mock_ticker):
            value, direction = _fetch_dxy()

        assert value == pytest.approx(103.0)
        assert direction == "flat"

    def test_failure_returns_flat(self) -> None:
        with patch("yfinance.Ticker", side_effect=Exception("network")):
            value, direction = _fetch_dxy()
        assert value is None
        assert direction == "flat"


# ---------------------------------------------------------------------------
# FRED real rate (mocked)
# ---------------------------------------------------------------------------


class TestFetchRealRate:
    def test_no_api_key_returns_none(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            assert _fetch_real_rate() is None

    @patch("requests.get")
    def test_success(self, mock_get: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "observations": [{"value": "2.15"}],
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        with patch.dict("os.environ", {"FRED_API_KEY": "test_key"}):
            rate = _fetch_real_rate()

        assert rate == pytest.approx(2.15)

    @patch("requests.get")
    def test_missing_value_returns_none(self, mock_get: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "observations": [{"value": "."}],
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        with patch.dict("os.environ", {"FRED_API_KEY": "test_key"}):
            rate = _fetch_real_rate()

        assert rate is None

    @patch("requests.get", side_effect=Exception("timeout"))
    def test_network_error_returns_none(self, mock_get: MagicMock) -> None:
        with patch.dict("os.environ", {"FRED_API_KEY": "test_key"}):
            rate = _fetch_real_rate()
        assert rate is None


# ---------------------------------------------------------------------------
# ExternalContextFetcher (caching + integration)
# ---------------------------------------------------------------------------


class TestExternalContextFetcher:
    def test_unavailable_when_all_fail(self) -> None:
        fetcher = ExternalContextFetcher(cache_ttl_minutes=60)
        with (
            patch("yfinance.Ticker", side_effect=Exception("fail")),
            patch.dict("os.environ", {}, clear=True),
        ):
            ctx = fetcher.fetch()
        assert ctx.source_quality == "unavailable"
        assert ctx.dxy_direction == "flat"
        assert ctx.vix_level is None

    def test_live_when_vix_succeeds(self) -> None:
        import pandas as pd

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame({"Close": [22.0]})

        fetcher = ExternalContextFetcher(cache_ttl_minutes=60)
        with (
            patch("yfinance.Ticker", return_value=mock_ticker),
            patch.dict("os.environ", {}, clear=True),
        ):
            ctx = fetcher.fetch()

        assert ctx.source_quality == "live"
        assert ctx.vix_level == pytest.approx(22.0)
        assert ctx.vix_regime == "elevated"

    def test_caching_returns_cached(self) -> None:
        import pandas as pd

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame({"Close": [18.0]})

        fetcher = ExternalContextFetcher(cache_ttl_minutes=60)

        with (
            patch("yfinance.Ticker", return_value=mock_ticker),
            patch.dict("os.environ", {}, clear=True),
        ):
            first = fetcher.fetch()
            assert first.source_quality == "live"

        # Second call should return cached (even if network would fail)
        with (
            patch("yfinance.Ticker", side_effect=Exception("network down")),
            patch.dict("os.environ", {}, clear=True),
        ):
            second = fetcher.fetch()
            assert second.source_quality == "cached"
            assert second.vix_level == first.vix_level

    def test_cache_expiry(self) -> None:
        import pandas as pd

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame({"Close": [18.0]})

        fetcher = ExternalContextFetcher(cache_ttl_minutes=1)

        with (
            patch("yfinance.Ticker", return_value=mock_ticker),
            patch.dict("os.environ", {}, clear=True),
        ):
            fetcher.fetch()

        # Manually expire the cache
        fetcher._cached_at = datetime.now(tz=timezone.utc) - timedelta(minutes=5)

        # Now it should re-fetch (and fail → unavailable)
        with (
            patch("yfinance.Ticker", side_effect=Exception("fail")),
            patch.dict("os.environ", {}, clear=True),
        ):
            ctx = fetcher.fetch()
            assert ctx.source_quality == "unavailable"

    def test_invalidate_clears_cache(self) -> None:
        import pandas as pd

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame({"Close": [18.0]})

        fetcher = ExternalContextFetcher(cache_ttl_minutes=60)

        with (
            patch("yfinance.Ticker", return_value=mock_ticker),
            patch.dict("os.environ", {}, clear=True),
        ):
            fetcher.fetch()

        fetcher.invalidate()
        assert fetcher._cached is None

    def test_cot_and_cb_are_none(self) -> None:
        """COT and central bank are not yet implemented — always None."""
        import pandas as pd

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame({"Close": [18.0]})

        fetcher = ExternalContextFetcher()
        with (
            patch("yfinance.Ticker", return_value=mock_ticker),
            patch.dict("os.environ", {}, clear=True),
        ):
            ctx = fetcher.fetch()

        assert ctx.cot_net_spec is None
        assert ctx.central_bank_stance is None

    def test_fetched_at_is_recent(self) -> None:
        import pandas as pd

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame({"Close": [18.0]})

        before = datetime.now(tz=timezone.utc)
        fetcher = ExternalContextFetcher()
        with (
            patch("yfinance.Ticker", return_value=mock_ticker),
            patch.dict("os.environ", {}, clear=True),
        ):
            ctx = fetcher.fetch()
        after = datetime.now(tz=timezone.utc)

        assert before <= ctx.fetched_at <= after
