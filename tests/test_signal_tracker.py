"""
Sinyal takibi: kayıt, TP/SL/süre değerlendirmesi ve isabet özeti testleri.
"""

from datetime import datetime, timedelta

import pandas as pd
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from app.database.models import Base, SignalStatus
from app.database.crud import create_signal, get_active_signals
from app.signals import tracker
from app.data.fetcher import FetchResult


# ── Geçici async DB ───────────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def db(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path/'t.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    async with Session() as session:
        yield session
    await engine.dispose()


def _report(signal="BUY", symbol="THYAO.IS", price=100.0, sl=97.0, tp=106.0):
    return {
        "results": [{
            "symbol": symbol, "signal": signal, "price": price,
            "stop_loss": sl, "take_profit": tp, "strategy": "trend_breakout",
            "strength": 0.7, "reason": "test",
        }]
    }


def _future_bars(high, low, days_ahead=1):
    """Bugünden sonraki tek bir bar içeren OHLCV DataFrame."""
    d = (datetime.utcnow() + timedelta(days=days_ahead)).date()
    idx = pd.to_datetime([d])
    df = pd.DataFrame(
        {"open": [low], "high": [high], "low": [low], "close": [high], "volume": [1e6]},
        index=idx,
    )
    return FetchResult(symbol="THYAO.IS", df=df, period="3mo", interval="1d")


# ── Kayıt ────────────────────────────────────────────────────────────────────

class TestPersist:
    @pytest.mark.asyncio
    async def test_buy_kaydedilir(self, db):
        created = await tracker.persist_scan_signals(db, _report("BUY"))
        assert created == 1
        assert len(await get_active_signals(db)) == 1

    @pytest.mark.asyncio
    async def test_early_watch_kaydedilmez(self, db):
        created = await tracker.persist_scan_signals(db, _report("EARLY_WATCH"))
        assert created == 0

    @pytest.mark.asyncio
    async def test_acik_sinyal_tekrar_kaydedilmez(self, db):
        await tracker.persist_scan_signals(db, _report("BUY"))
        created2 = await tracker.persist_scan_signals(db, _report("BUY"))
        assert created2 == 0
        assert len(await get_active_signals(db)) == 1


# ── Değerlendirme ──────────────────────────────────────────────────────────────

class TestEvaluate:
    @pytest.mark.asyncio
    async def test_take_profit_hit(self, db, monkeypatch):
        await tracker.persist_scan_signals(db, _report(sl=97, tp=106))
        monkeypatch.setattr(tracker, "fetch_symbol_data_cached",
                            lambda *a, **k: _future_bars(high=110, low=99))
        counts = await tracker.evaluate_open_signals(db)
        assert counts["hit_tp"] == 1
        assert len(await get_active_signals(db)) == 0

    @pytest.mark.asyncio
    async def test_stop_loss_hit(self, db, monkeypatch):
        await tracker.persist_scan_signals(db, _report(sl=97, tp=106))
        monkeypatch.setattr(tracker, "fetch_symbol_data_cached",
                            lambda *a, **k: _future_bars(high=101, low=95))
        counts = await tracker.evaluate_open_signals(db)
        assert counts["hit_sl"] == 1

    @pytest.mark.asyncio
    async def test_henuz_acik_kalir(self, db, monkeypatch):
        await tracker.persist_scan_signals(db, _report(sl=97, tp=106))
        monkeypatch.setattr(tracker, "fetch_symbol_data_cached",
                            lambda *a, **k: _future_bars(high=103, low=99))
        counts = await tracker.evaluate_open_signals(db)
        assert counts["still_open"] == 1
        assert len(await get_active_signals(db)) == 1


# ── Özet ─────────────────────────────────────────────────────────────────────

class TestSummary:
    @pytest.mark.asyncio
    async def test_isabet_orani(self, db, monkeypatch):
        # 1 TP + 1 SL → isabet %50
        await tracker.persist_scan_signals(db, _report(symbol="AAA.IS"))
        await tracker.persist_scan_signals(db, _report(symbol="BBB.IS"))

        def fake(symbol, *a, **k):
            return _future_bars(high=110, low=99) if symbol == "AAA.IS" \
                else _future_bars(high=101, low=95)
        monkeypatch.setattr(tracker, "fetch_symbol_data_cached", fake)

        await tracker.evaluate_open_signals(db)
        summary = await tracker.build_performance_summary(db, days=30)
        assert summary["hit_tp"] == 1
        assert summary["hit_sl"] == 1
        assert summary["win_rate"] == 50.0

    @pytest.mark.asyncio
    async def test_kapanan_yoksa_win_rate_none(self, db):
        await tracker.persist_scan_signals(db, _report())
        summary = await tracker.build_performance_summary(db, days=30)
        assert summary["win_rate"] is None
        assert summary["active"] == 1
