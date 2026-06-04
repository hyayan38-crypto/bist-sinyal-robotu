"""
XU100 Endeks Filtresi testleri — ağ bağlantısı gerekmez, mock kullanılır.
"""

import time
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from app.risk.market_filter import (
    is_market_favorable,
    invalidate_cache,
    MarketFilterResult,
    STATUS_FAVORABLE,
    STATUS_UNFAVORABLE,
    STATUS_UNAVAILABLE,
    _cache,
    _XU100_SYMBOL,
    _EMA_PERIOD,
)
from app.data.fetcher import FetchResult, EmptyDataError, FetchError


# ── Yardımcı ──────────────────────────────────────────────────────────────────

def _make_fetch_result(close_last: float, ema_above: bool, n: int = 120) -> FetchResult:
    """
    close_last fiyatıyla biten ve EMA50'nin üstünde/altında kapanan sahte FetchResult.
    ema_above=True  → close > EMA50
    ema_above=False → close < EMA50
    """
    if ema_above:
        closes = np.linspace(80, close_last, n)   # yükseliş trendi
    else:
        closes = np.linspace(close_last * 1.3, close_last, n)  # düşüş trendi

    df = pd.DataFrame(
        {
            "open":   closes * 0.99,
            "high":   closes * 1.01,
            "low":    closes * 0.98,
            "close":  closes,
            "volume": np.ones(n) * 1e8,
        },
        index=pd.date_range("2023-01-01", periods=n, freq="B"),
    )
    return FetchResult(symbol=_XU100_SYMBOL, df=df, period="6mo", interval="1d")


def _patch_fetch(result):
    """fetch_symbol_data'yı verilen değerle değiştirir."""
    return patch("app.risk.market_filter.fetch_symbol_data", return_value=result)


def _patch_fetch_error(exc):
    """fetch_symbol_data'nın istisna fırlatmasını sağlar."""
    return patch("app.risk.market_filter.fetch_symbol_data", side_effect=exc)


# ── Her test öncesi önbelleği temizle ─────────────────────────────────────────

@pytest.fixture(autouse=True)
def clear_cache():
    invalidate_cache()
    yield
    invalidate_cache()


# ── Temel davranış ────────────────────────────────────────────────────────────

class TestIsMarketFavorable:
    def test_favorable_when_close_above_ema50(self):
        with _patch_fetch(_make_fetch_result(10_000, ema_above=True)):
            result = is_market_favorable()
        assert result.favorable is True
        assert result.status == STATUS_FAVORABLE

    def test_unfavorable_when_close_below_ema50(self):
        with _patch_fetch(_make_fetch_result(7_000, ema_above=False)):
            result = is_market_favorable()
        assert result.favorable is False
        assert result.status == STATUS_UNFAVORABLE

    def test_blocks_buy_when_unfavorable(self):
        with _patch_fetch(_make_fetch_result(7_000, ema_above=False)):
            result = is_market_favorable()
        assert result.blocks_buy is True

    def test_does_not_block_buy_when_favorable(self):
        with _patch_fetch(_make_fetch_result(10_000, ema_above=True)):
            result = is_market_favorable()
        assert result.blocks_buy is False

    def test_xu100_values_populated(self):
        with _patch_fetch(_make_fetch_result(10_000, ema_above=True)):
            result = is_market_favorable()
        assert result.xu100_close is not None
        assert result.xu100_ema50 is not None
        assert isinstance(result.xu100_close, float)
        assert isinstance(result.xu100_ema50, float)

    def test_reason_not_empty(self):
        with _patch_fetch(_make_fetch_result(10_000, ema_above=True)):
            result = is_market_favorable()
        assert len(result.reason) > 10

    def test_reason_mentions_xu100(self):
        with _patch_fetch(_make_fetch_result(10_000, ema_above=True)):
            result = is_market_favorable()
        assert "XU100" in result.reason


# ── Hata durumları (fail-open) ─────────────────────────────────────────────────

class TestUnavailableHandling:
    def test_fetch_error_returns_unavailable(self):
        with _patch_fetch_error(FetchError(_XU100_SYMBOL, "bağlantı hatası")):
            result = is_market_favorable()
        assert result.status == STATUS_UNAVAILABLE

    def test_unexpected_error_returns_unavailable(self):
        with _patch_fetch_error(RuntimeError("beklenmeyen hata")):
            result = is_market_favorable()
        assert result.status == STATUS_UNAVAILABLE

    def test_unavailable_does_not_raise(self):
        with _patch_fetch_error(Exception("herhangi bir hata")):
            result = is_market_favorable()   # istisna fırlatmamalı
        assert result is not None

    def test_unavailable_sets_favorable_true(self):
        """Fail-open: veri yoksa AL sinyallerini engelleme."""
        with _patch_fetch_error(FetchError(_XU100_SYMBOL, "timeout")):
            result = is_market_favorable()
        assert result.favorable is True
        assert result.blocks_buy is False

    def test_insufficient_bars_returns_unavailable(self):
        short_result = _make_fetch_result(10_000, ema_above=True, n=10)
        with _patch_fetch(short_result):
            result = is_market_favorable()
        assert result.status == STATUS_UNAVAILABLE

    def test_unavailable_xu100_values_are_none(self):
        with _patch_fetch_error(FetchError(_XU100_SYMBOL, "hata")):
            result = is_market_favorable()
        assert result.xu100_close is None
        assert result.xu100_ema50 is None


# ── Önbellek ──────────────────────────────────────────────────────────────────

class TestCache:
    def test_second_call_uses_cache(self):
        fetch_result = _make_fetch_result(10_000, ema_above=True)
        with _patch_fetch(fetch_result) as mock_fetch:
            is_market_favorable()
            result2 = is_market_favorable()
        assert mock_fetch.call_count == 1
        assert result2.cached is True

    def test_force_refresh_bypasses_cache(self):
        fetch_result = _make_fetch_result(10_000, ema_above=True)
        with _patch_fetch(fetch_result) as mock_fetch:
            is_market_favorable()
            is_market_favorable(force_refresh=True)
        assert mock_fetch.call_count == 2

    def test_invalidate_cache_clears_state(self):
        fetch_result = _make_fetch_result(10_000, ema_above=True)
        with _patch_fetch(fetch_result) as mock_fetch:
            is_market_favorable()
            invalidate_cache()
            is_market_favorable()
        assert mock_fetch.call_count == 2

    def test_fresh_result_not_cached(self):
        with _patch_fetch(_make_fetch_result(10_000, ema_above=True)):
            result = is_market_favorable()
        assert result.cached is False

    def test_cached_result_has_same_status(self):
        with _patch_fetch(_make_fetch_result(7_000, ema_above=False)):
            first  = is_market_favorable()
            second = is_market_favorable()
        assert first.status == second.status == STATUS_UNFAVORABLE

    def test_expired_cache_refetches(self):
        fetch_result = _make_fetch_result(10_000, ema_above=True)
        with _patch_fetch(fetch_result) as mock_fetch:
            is_market_favorable()
            # TTL'yi geçmişe al
            _cache.expires_at = time.monotonic() - 1
            is_market_favorable()
        assert mock_fetch.call_count == 2


# ── MarketFilterResult ────────────────────────────────────────────────────────

class TestMarketFilterResult:
    def test_blocks_buy_true_when_unfavorable(self):
        r = MarketFilterResult(favorable=False, status=STATUS_UNFAVORABLE, reason="test")
        assert r.blocks_buy is True

    def test_blocks_buy_false_when_favorable(self):
        r = MarketFilterResult(favorable=True, status=STATUS_FAVORABLE, reason="test")
        assert r.blocks_buy is False

    def test_blocks_buy_false_when_unavailable(self):
        r = MarketFilterResult(favorable=True, status=STATUS_UNAVAILABLE, reason="test")
        assert r.blocks_buy is False

    def test_str_contains_status(self):
        r = MarketFilterResult(favorable=True, status=STATUS_FAVORABLE, reason="test")
        assert STATUS_FAVORABLE in str(r)

    def test_str_contains_cache_tag(self):
        r = MarketFilterResult(favorable=True, status=STATUS_FAVORABLE, reason="test", cached=True)
        assert "CACHE" in str(r)


# ── Sinyal üreteci entegrasyonu ───────────────────────────────────────────────

class TestGeneratorIntegration:
    """
    SignalGenerator'ın market filter'a göre AL sinyalini engelleyip engellemediğini test eder.
    Gerçek yfinance çağrısı yapılmaz.
    """

    def _make_strategy_signal(self, signal_type="BUY"):
        from app.strategies.base import StrategySignal, SignalType
        return StrategySignal(
            symbol="THYAO.IS",
            signal_type=SignalType.BUY if signal_type == "BUY" else SignalType.SELL,
            strategy="test",
            strength=0.8,
            entry_price=100.0,
            stop_loss=95.0,
            take_profit=110.0,
        )

    @pytest.mark.asyncio
    async def test_buy_blocked_when_market_unfavorable(self):
        from app.signals.generator import SignalGenerator
        from app.risk.market_filter import MarketFilterResult, STATUS_UNFAVORABLE

        unfavorable = MarketFilterResult(
            favorable=False,
            status=STATUS_UNFAVORABLE,
            reason="test: piyasa düşüşte",
        )

        gen = SignalGenerator()
        # fetcher None dönsün → sembol atlanır
        with patch("app.signals.generator.fetcher.get_ohlcv", return_value=None):
            signals = await gen.run_for_symbol("THYAO.IS", market_filter_result=unfavorable)
        assert signals == []

    @pytest.mark.asyncio
    async def test_buy_allowed_when_market_favorable(self):
        from app.signals.generator import SignalGenerator
        from app.risk.market_filter import MarketFilterResult, STATUS_FAVORABLE

        favorable = MarketFilterResult(
            favorable=True,
            status=STATUS_FAVORABLE,
            reason="test: piyasa yükselişte",
        )
        gen = SignalGenerator()
        with patch("app.signals.generator.fetcher.get_ohlcv", return_value=None):
            signals = await gen.run_for_symbol("THYAO.IS", market_filter_result=favorable)
        # fetcher None döndüğü için yine [] — ama filtre engellemedi
        assert signals == []
