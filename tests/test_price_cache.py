"""
SQLite fiyat önbelleği ve cache'li çekim testleri.
"""

from unittest.mock import patch

import pandas as pd
import pytest

from app.config import settings
from app.data import price_cache
from app.data.fetcher import FetchResult, fetch_symbol_data_cached
from conftest import make_ohlcv


@pytest.fixture
def temp_cache(tmp_path, monkeypatch):
    """Önbelleği geçici bir DB dosyasına yönlendirir ve etkinleştirir."""
    db_file = tmp_path / "cache_test.db"
    monkeypatch.setattr(settings, "database_url", f"sqlite+aiosqlite:///{db_file}")
    monkeypatch.setattr(settings, "price_cache_enabled", True)
    yield


def _fetch_result(symbol="THYAO.IS", n=250):
    return FetchResult(symbol=symbol, df=make_ohlcv(n=n), period="1y", interval="1d")


# ── price_cache modülü ────────────────────────────────────────────────────────

class TestPriceCacheRoundTrip:
    def test_replace_then_load(self, temp_cache):
        df = make_ohlcv(n=60)
        assert price_cache.replace("THYAO.IS", "1d", df) is True

        cached = price_cache.load("THYAO.IS", "1d")
        assert cached is not None
        assert len(cached.df) == 60
        assert list(cached.df.columns) == ["open", "high", "low", "close", "volume"]

    def test_load_missing_returns_none(self, temp_cache):
        assert price_cache.load("YOKXX.IS", "1d") is None

    def test_upsert_updates_existing_bar(self, temp_cache):
        df = make_ohlcv(n=30)
        price_cache.replace("ASELS.IS", "1d", df)

        # Son barın kapanışını değiştir, upsert et
        patched = df.copy()
        patched.iloc[-1, patched.columns.get_loc("close")] = 999.0
        price_cache.upsert("ASELS.IS", "1d", patched.tail(1))

        cached = price_cache.load("ASELS.IS", "1d")
        assert cached.df["close"].iloc[-1] == 999.0
        assert len(cached.df) == 30  # satır sayısı artmaz

    def test_fail_open_on_bad_path(self, monkeypatch):
        # Geçersiz yol → fail-open, istisna fırlatmaz
        monkeypatch.setattr(settings, "database_url", "sqlite+aiosqlite:////nonexistent_dir/x.db")
        assert price_cache.load("X.IS", "1d") is None
        assert price_cache.replace("X.IS", "1d", make_ohlcv(n=5)) is False


# ── fetch_symbol_data_cached ──────────────────────────────────────────────────

class TestFetchCached:
    def test_disabled_bypasses_cache(self, monkeypatch):
        monkeypatch.setattr(settings, "price_cache_enabled", False)
        with patch("app.data.fetcher.fetch_symbol_data", return_value=_fetch_result()) as m:
            res = fetch_symbol_data_cached("THYAO", period="1y")
        assert m.call_count == 1
        assert res.from_cache is False

    def test_first_call_full_fetch_and_caches(self, temp_cache):
        with patch("app.data.fetcher.fetch_symbol_data", return_value=_fetch_result()) as m:
            res = fetch_symbol_data_cached("THYAO", period="1y")
        assert m.call_count == 1
        assert res.from_cache is False
        # Önbelleğe yazıldı
        assert price_cache.load("THYAO.IS", "1d") is not None

    def test_second_same_day_call_incremental(self, temp_cache):
        with patch("app.data.fetcher.fetch_symbol_data", return_value=_fetch_result()) as m:
            fetch_symbol_data_cached("THYAO", period="1y")          # tam indirme
            res2 = fetch_symbol_data_cached("THYAO", period="1y")   # artımlı

        # İkinci çağrı da fetch_symbol_data çağırır ama period="5d" ile
        assert m.call_count == 2
        assert m.call_args.kwargs.get("period") == "5d"
        assert res2.from_cache is True
        assert res2.source == "sqlite"
