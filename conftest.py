"""
Merkezi pytest yapılandırması ve paylaşılan fixture'lar.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from app.data.fetcher import FetchResult
from app.risk.market_filter import MarketFilterResult, STATUS_FAVORABLE


# ── Scheduler event-loop izolasyonu ──────────────────────────────────────────

@pytest.fixture(autouse=True, scope="session")
def _mock_scheduler_global():
    """
    Tüm test session boyunca app.main.scheduler'ı mock'lar.

    - TestClient(app) lifespan'ı defalarca çalıştırıldığında AsyncIOScheduler
      event loop'u kapatıp diğer testleri bozmasın diye.
    - test_scheduler.py'deki TestBISTScheduler / TestRunDailyScan testleri
      app.main.scheduler'ı kullanmaz — kendi BISTScheduler() instance'larını
      oluşturur, bu mock'tan etkilenmezler.
    - TestSchedulerAPI testleri mock'un döndürdüğü değerlerle çalışır.
    """
    with patch("app.main.scheduler") as mock_sched:
        mock_sched.start  = MagicMock()
        mock_sched.stop   = MagicMock()
        mock_sched.status = MagicMock(return_value={
            "running": True,
            "schedule": "Her gün 18:10 TR saati",
            "next_run": "17.05.2026 18:10 +03",
            "timezone": "Europe/Istanbul",
            "job_id": "daily_bist_scan",
            "symbol_count": 15,
        })
        mock_sched.next_run_time   = MagicMock(return_value="17.05.2026 18:10 +03")
        mock_sched.update_schedule = MagicMock()
        yield


# ── Ortak DataFrame fabrikaları ───────────────────────────────────────────────

def make_ohlcv(n: int = 250, trend: str = "up", seed: int = 42) -> pd.DataFrame:
    """Teknik indikatörler ve strateji testleri için standart OHLCV DataFrame."""
    np.random.seed(seed)
    if trend == "up":
        close = np.linspace(80, 130, n) + np.random.randn(n) * 1.0
    elif trend == "down":
        close = np.linspace(130, 80, n) + np.random.randn(n) * 1.0
    else:
        close = 100 + np.cumsum(np.random.randn(n) * 1.0)

    high = close + np.abs(np.random.randn(n)) * 1.2
    low  = close - np.abs(np.random.randn(n)) * 1.2
    vol  = np.random.randint(500_000, 3_000_000, n).astype(float)

    return pd.DataFrame(
        {"open": close * 0.998, "high": high, "low": low, "close": close, "volume": vol},
        index=pd.date_range("2023-01-01", periods=n, freq="B"),
    )


def make_fetch_result(symbol: str = "THYAO.IS", **kwargs) -> FetchResult:
    return FetchResult(
        symbol=symbol,
        df=make_ohlcv(**kwargs),
        period="1y",
        interval="1d",
    )


def make_favorable() -> MarketFilterResult:
    return MarketFilterResult(favorable=True, status=STATUS_FAVORABLE, reason="test")


# ── Paylaşılan fixture'lar ────────────────────────────────────────────────────

@pytest.fixture
def ohlcv_df():
    """250 günlük yükseliş trendli OHLCV DataFrame."""
    return make_ohlcv()


@pytest.fixture
def ohlcv_with_indicators():
    """add_indicators() uygulanmış DataFrame."""
    from app.indicators.technical import add_indicators
    return add_indicators(make_ohlcv())


@pytest.fixture
def fetch_result():
    return make_fetch_result()


@pytest.fixture
def favorable_market():
    return make_favorable()


@pytest.fixture
def mock_fetch(fetch_result):
    """fetch_symbol_data'yı mocklayan patch."""
    with patch("app.data.fetcher.fetch_symbol_data", return_value=fetch_result):
        yield fetch_result
