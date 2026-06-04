"""
Fetcher testleri — ağ bağlantısı gerektiren testler mock kullanır.
Gerçek veri çeken testler işaretlenir: @pytest.mark.network
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

from app.data.fetcher import (
    fetch_symbol_data,
    fetch_multiple_symbols,
    FetchResult,
    FetchError,
    EmptyDataError,
    InsufficientDataError,
    _normalize_symbol,
    _clean_ohlcv,
)


# ── Yardımcı ──────────────────────────────────────────────────────────────────

def _make_raw_df(n: int = 60, start: str = "2023-01-01") -> pd.DataFrame:
    """yfinance'in döndürdüğü formatta ham DataFrame."""
    idx = pd.date_range(start, periods=n, freq="B", tz="America/New_York")
    np.random.seed(0)
    close = 100 + np.cumsum(np.random.randn(n))
    return pd.DataFrame(
        {
            "Open": close * 0.99,
            "High": close * 1.01,
            "Low": close * 0.98,
            "Close": close,
            "Volume": np.random.randint(500_000, 5_000_000, n).astype(float),
            "Dividends": 0.0,
            "Stock Splits": 0.0,
        },
        index=idx,
    )


def _patch_ticker(df: pd.DataFrame):
    """yf.Ticker(...).history(...) çağrısını verilen df ile sahte yapar."""
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = df
    return patch("app.data.fetcher.yf.Ticker", return_value=mock_ticker)


# ── normalize_symbol ──────────────────────────────────────────────────────────

class TestNormalizeSymbol:
    def test_adds_is(self):
        assert _normalize_symbol("thyao") == "THYAO.IS"

    def test_no_double_is(self):
        assert _normalize_symbol("GARAN.IS") == "GARAN.IS"

    def test_strips_whitespace(self):
        assert _normalize_symbol("  akbnk  ") == "AKBNK.IS"

    def test_uppercase(self):
        assert _normalize_symbol("eregl.is") == "EREGL.IS"


# ── _clean_ohlcv ──────────────────────────────────────────────────────────────

class TestCleanOHLCV:
    def test_renames_columns(self):
        df = _make_raw_df(50)
        cleaned = _clean_ohlcv(df, "TEST.IS")
        assert list(cleaned.columns) == ["open", "high", "low", "close", "volume"]

    def test_removes_timezone(self):
        df = _make_raw_df(50)
        cleaned = _clean_ohlcv(df, "TEST.IS")
        assert cleaned.index.tz is None

    def test_drops_nan_rows(self):
        df = _make_raw_df(50)
        df.iloc[5, df.columns.get_loc("Close")] = float("nan")
        cleaned = _clean_ohlcv(df, "TEST.IS")
        assert len(cleaned) == 49

    def test_drops_zero_close(self):
        df = _make_raw_df(50)
        df.iloc[3, df.columns.get_loc("Close")] = 0.0
        cleaned = _clean_ohlcv(df, "TEST.IS")
        assert len(cleaned) == 49

    def test_drops_inverted_hl(self):
        df = _make_raw_df(50)
        df.iloc[10, df.columns.get_loc("High")] = 1.0
        df.iloc[10, df.columns.get_loc("Low")] = 999.0
        cleaned = _clean_ohlcv(df, "TEST.IS")
        assert len(cleaned) == 49

    def test_raises_on_missing_column(self):
        df = _make_raw_df(50).drop(columns=["Volume"])
        with pytest.raises(FetchError):
            _clean_ohlcv(df, "TEST.IS")


# ── fetch_symbol_data ─────────────────────────────────────────────────────────

class TestFetchSymbolData:
    def test_returns_fetch_result(self):
        with _patch_ticker(_make_raw_df(60)):
            result = fetch_symbol_data("THYAO", period="2y")
        assert isinstance(result, FetchResult)
        assert result.symbol == "THYAO.IS"
        assert result.row_count == 60
        assert not result.from_cache

    def test_columns_are_lowercase(self):
        with _patch_ticker(_make_raw_df(60)):
            result = fetch_symbol_data("THYAO")
        assert list(result.df.columns) == ["open", "high", "low", "close", "volume"]

    def test_index_is_datetime_no_tz(self):
        with _patch_ticker(_make_raw_df(60)):
            result = fetch_symbol_data("THYAO")
        assert isinstance(result.df.index, pd.DatetimeIndex)
        assert result.df.index.tz is None

    def test_empty_data_raises(self):
        with _patch_ticker(pd.DataFrame()):
            with pytest.raises(EmptyDataError):
                fetch_symbol_data("THYAO")

    def test_insufficient_data_raises(self):
        with _patch_ticker(_make_raw_df(10)):
            with pytest.raises(InsufficientDataError):
                fetch_symbol_data("THYAO", min_rows=30)

    def test_custom_min_rows(self):
        with _patch_ticker(_make_raw_df(15)):
            result = fetch_symbol_data("THYAO", min_rows=10)
        assert result.row_count == 15

    def test_yfinance_exception_raises_fetch_error(self):
        mock_ticker = MagicMock()
        mock_ticker.history.side_effect = RuntimeError("connection error")
        with patch("app.data.fetcher.yf.Ticker", return_value=mock_ticker):
            with pytest.raises(FetchError):
                fetch_symbol_data("THYAO")

    def test_start_end_uses_history_params(self):
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = _make_raw_df(60)
        with patch("app.data.fetcher.yf.Ticker", return_value=mock_ticker):
            fetch_symbol_data("THYAO", start="2023-01-01", end="2024-01-01")
        call_kwargs = mock_ticker.history.call_args.kwargs
        assert call_kwargs.get("start") == "2023-01-01"
        assert call_kwargs.get("end") == "2024-01-01"

    def test_period_uses_history_params(self):
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = _make_raw_df(60)
        with patch("app.data.fetcher.yf.Ticker", return_value=mock_ticker):
            fetch_symbol_data("THYAO", period="6mo")
        call_kwargs = mock_ticker.history.call_args.kwargs
        assert call_kwargs.get("period") == "6mo"


# ── FetchResult ───────────────────────────────────────────────────────────────

class TestFetchResult:
    def setup_method(self):
        with _patch_ticker(_make_raw_df(60)):
            self.result = fetch_symbol_data("GARAN", period="2y")

    def test_start_end_date(self):
        assert isinstance(self.result.start_date, datetime)
        assert isinstance(self.result.end_date, datetime)
        assert self.result.start_date < self.result.end_date

    def test_to_cache_records(self):
        records = self.result.to_cache_records()
        assert len(records) == 60
        first = records[0]
        assert set(first.keys()) == {"symbol", "interval", "date", "open", "high", "low", "close", "volume", "fetched_at"}
        assert first["symbol"] == "GARAN.IS"
        assert first["interval"] == "1d"

    def test_cache_records_have_naive_datetime(self):
        records = self.result.to_cache_records()
        for r in records:
            assert r["date"].tzinfo is None
            assert r["fetched_at"].tzinfo is None

    def test_from_cache_records_roundtrip(self):
        records = self.result.to_cache_records()
        restored = FetchResult.from_cache_records("GARAN.IS", records, interval="1d")
        assert restored.from_cache is True
        assert restored.row_count == 60
        assert list(restored.df.columns) == ["open", "high", "low", "close", "volume"]

    def test_repr_contains_symbol(self):
        assert "GARAN.IS" in repr(self.result)


# ── fetch_multiple_symbols ────────────────────────────────────────────────────

class TestFetchMultipleSymbols:
    def test_returns_dict_of_results(self):
        with _patch_ticker(_make_raw_df(60)):
            results = fetch_multiple_symbols(["THYAO", "GARAN"])
        assert len(results) == 2
        assert "THYAO.IS" in results
        assert "GARAN.IS" in results

    def test_skip_errors_true(self):
        good_df = _make_raw_df(60)

        def side_effect(symbol):
            mock = MagicMock()
            if "THYAO" in symbol:
                mock.history.return_value = good_df
            else:
                mock.history.return_value = pd.DataFrame()  # boş → EmptyDataError
            return mock

        with patch("app.data.fetcher.yf.Ticker", side_effect=side_effect):
            results = fetch_multiple_symbols(["THYAO", "BADONE"], skip_errors=True)
        assert "THYAO.IS" in results
        assert "BADONE.IS" not in results

    def test_skip_errors_false_raises(self):
        with _patch_ticker(pd.DataFrame()):  # boş → EmptyDataError
            with pytest.raises(FetchError):
                fetch_multiple_symbols(["THYAO"], skip_errors=False)

    def test_empty_list_returns_empty(self):
        results = fetch_multiple_symbols([])
        assert results == {}
