"""
Zamanlayıcı testleri — gerçek APScheduler ve Telegram çağrısı yapılmaz.
"""

import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient

from app.scheduler import BISTScheduler, run_daily_scan, _SCAN_TIMES
from app.main import app


# ── Yardımcı ──────────────────────────────────────────────────────────────────

def _make_scheduler() -> BISTScheduler:
    """Başlatılmamış temiz BISTScheduler."""
    return BISTScheduler()


def _mock_results(buy: int = 2, watch: int = 1) -> list[dict]:
    results = []
    for i in range(buy):
        results.append({
            "signal": "BUY", "symbol": f"SYM{i}.IS", "price": 100.0 + i,
            "reason": "test", "risk_level": "LOW", "strength": 0.7,
            "strength_score": 70, "stop_loss": 97.0, "take_profit": 106.0,
            "market_filter": "favorable", "conditions_met": 5, "distance_to_res_pct": 0.0,
        })
    for i in range(watch):
        results.append({
            "signal": "WATCH", "symbol": f"WCH{i}.IS", "price": 80.0 + i,
            "reason": "test", "risk_level": "LOW", "strength": 0.4,
            "strength_score": 40, "stop_loss": None, "take_profit": None,
            "market_filter": "favorable", "conditions_met": 4, "distance_to_res_pct": 1.5,
        })
    return results


def _mock_report(buy: int = 2, watch: int = 1) -> dict:
    """scan_bist100'ün döndürdüğü rapor dict'inin test taklidi."""
    results = _mock_results(buy, watch)
    return {
        "label": "BIST100",
        "results": results,
        "scanned": len(results),
        "buy_count": buy,
        "watch_count": watch,
        "setup_count": 0,
        "early_watch_count": 0,
        "late_breakout_count": 0,
        "error_count": 0,
        "error_symbols": [],
        "elapsed_seconds": 1.2,
        "market_filter": "favorable",
    }


# ── BISTScheduler ─────────────────────────────────────────────────────────────

class TestBISTScheduler:
    def test_baslangicta_calismiyor(self):
        s = _make_scheduler()
        assert not s._scheduler.running

    def test_start_baslatir(self):
        s = _make_scheduler()
        try:
            s.start()
            assert s._scheduler.running
        finally:
            s.stop()

    def test_stop_hata_vermez(self):
        # AsyncIOScheduler, event-loop dışında shutdown sonrası running=False
        # olmayabilir; sadece exception fırlatmadığını doğruluyoruz
        s = _make_scheduler()
        s.start()
        s.stop()   # hata vermemeli

    def test_stop_calismiyor_hata_vermez(self):
        s = _make_scheduler()
        s.stop()   # başlatmadan durdurmak hata vermemeli

    def test_tum_taramalar_eklendi(self):
        s = _make_scheduler()
        try:
            s.start()
            jobs = s._scheduler.get_jobs()
            assert len(jobs) == len(_SCAN_TIMES)
            job_ids = {j.id for j in jobs}
            for _h, _m, label in _SCAN_TIMES:
                assert f"bist100_scan_{label}" in job_ids
        finally:
            s.stop()

    def test_dort_tarama_saati(self):
        # 10:30 · 12:30 · 15:30 · 18:10
        assert len(_SCAN_TIMES) == 4
        saatler = {(h, m) for h, m, _ in _SCAN_TIMES}
        assert (10, 30) in saatler
        assert (18, 10) in saatler

    def test_status_dict_doner(self):
        s = _make_scheduler()
        try:
            s.start()
            status = s.status()
            assert isinstance(status, dict)
        finally:
            s.stop()

    def test_status_alanlari(self):
        s = _make_scheduler()
        try:
            s.start()
            status = s.status()
            for key in ("running", "schedule", "next_run", "timezone",
                        "jobs", "scan_universe"):
                assert key in status, f"'{key}' eksik"
            assert len(status["jobs"]) == len(_SCAN_TIMES)
        finally:
            s.stop()

    def test_status_running_true(self):
        s = _make_scheduler()
        try:
            s.start()
            assert s.status()["running"] is True
        finally:
            s.stop()

    def test_timezone_istanbul(self):
        s = _make_scheduler()
        try:
            s.start()
            assert "Istanbul" in s.status()["timezone"] or "Europe" in s.status()["timezone"]
        finally:
            s.stop()

    def test_next_run_time_mevcut(self):
        s = _make_scheduler()
        try:
            s.start()
            nrt = s.next_run_time()
            assert nrt is not None
            # En yakın çalışma zamanı 4 tarama saatinden biri olmalı
            saat_str = [f"{h:02d}:{m:02d}" for h, m, _ in _SCAN_TIMES]
            assert any(t in nrt for t in saat_str)
        finally:
            s.stop()


# ── run_daily_scan ────────────────────────────────────────────────────────────

class TestRunDailyScan:
    @pytest.mark.asyncio
    async def test_sinyal_varsa_mesaj_gonderilir(self):
        with patch("app.scheduler.scan_bist100", return_value=_mock_report(buy=2, watch=1)):
            with patch("app.scheduler.send_telegram_message", new_callable=AsyncMock) as mock_send:
                await run_daily_scan()
        # Tek tam rapor mesajı gönderilir
        assert mock_send.call_count == 1

    @pytest.mark.asyncio
    async def test_sinyal_yoksa_da_rapor_gonderilir(self):
        with patch("app.scheduler.scan_bist100", return_value=_mock_report(buy=0, watch=0)):
            with patch("app.scheduler.send_telegram_message", new_callable=AsyncMock) as mock_send:
                await run_daily_scan()
        # Sinyal olmasa da tarama raporu gönderilir
        assert mock_send.call_count == 1

    @pytest.mark.asyncio
    async def test_rapor_dict_doner(self):
        with patch("app.scheduler.scan_bist100", return_value=_mock_report(1, 0)):
            with patch("app.scheduler.send_telegram_message", new_callable=AsyncMock):
                report = await run_daily_scan()
        assert isinstance(report, dict)
        assert "buy_count" in report
        assert "scanned" in report

    @pytest.mark.asyncio
    async def test_buy_sayisi_dogru(self):
        with patch("app.scheduler.scan_bist100", return_value=_mock_report(buy=3, watch=0)):
            with patch("app.scheduler.send_telegram_message", new_callable=AsyncMock):
                report = await run_daily_scan()
        assert report["buy_count"] == 3

    @pytest.mark.asyncio
    async def test_scan_bist100_cagrilir(self):
        with patch("app.scheduler.scan_bist100", return_value=_mock_report(0, 0)) as mock_scan:
            with patch("app.scheduler.send_telegram_message", new_callable=AsyncMock):
                await run_daily_scan()
        assert mock_scan.called

    @pytest.mark.asyncio
    async def test_force_market_refresh_true(self):
        with patch("app.scheduler.scan_bist100", return_value=_mock_report(0, 0)) as mock_scan:
            with patch("app.scheduler.send_telegram_message", new_callable=AsyncMock):
                await run_daily_scan()
        call_kwargs = mock_scan.call_args.kwargs
        assert call_kwargs.get("force_market_refresh") is True

    @pytest.mark.asyncio
    async def test_telegram_hatasi_exception_firlatmaz(self):
        with patch("app.scheduler.scan_bist100", return_value=_mock_report(1, 0)):
            with patch("app.scheduler.send_telegram_message",
                       new_callable=AsyncMock, side_effect=Exception("bağlantı hatası")):
                try:
                    await run_daily_scan()
                except Exception:
                    pytest.fail("run_daily_scan Telegram hatasında exception fırlatmamalı")


# ── API endpoint'leri ─────────────────────────────────────────────────────────

class TestSchedulerAPI:
    @pytest.fixture(scope="class")
    def client(self):
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c

    def test_status_200_doner(self, client):
        r = client.get("/scheduler/status")
        assert r.status_code == 200

    def test_status_alanlari(self, client):
        body = client.get("/scheduler/status").json()
        for key in ("running", "schedule", "next_run", "timezone", "scan_universe"):
            assert key in body

    def test_status_running_true(self, client):
        # TestClient lifespan'ı çalıştırdığı için scheduler başlamış olmalı
        body = client.get("/scheduler/status").json()
        assert body["running"] is True

    def test_status_saat_icerir(self, client):
        body = client.get("/scheduler/status").json()
        assert "18:10" in body["schedule"]

    def test_run_now_202_veya_200(self, client):
        with patch("app.scheduler.run_daily_scan", new_callable=AsyncMock, return_value={}):
            r = client.post("/scheduler/run-now")
        assert r.status_code in (200, 202)

    def test_run_now_mesaj_doner(self, client):
        with patch("app.scheduler.run_daily_scan", new_callable=AsyncMock, return_value={}):
            body = client.post("/scheduler/run-now").json()
        assert "message" in body
        assert "symbols" in body
