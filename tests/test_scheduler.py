"""
Zamanlayıcı testleri — gerçek APScheduler ve Telegram çağrısı yapılmaz.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from app.scheduler import BISTScheduler, run_daily_scan
from app.main import app


# ── Yardımcı ──────────────────────────────────────────────────────────────────

def _make_scheduler() -> BISTScheduler:
    """Başlatılmamış temiz BISTScheduler."""
    return BISTScheduler()


def _mock_scan_results(buy: int = 2, watch: int = 1) -> list[dict]:
    results = []
    for i in range(buy):
        results.append({
            "signal": "BUY", "symbol": f"SYM{i}.IS", "price": 100.0 + i,
            "reason": "test", "risk_level": "LOW", "strength": 0.7,
            "stop_loss": 97.0, "take_profit": 106.0,
            "market_filter": "favorable", "conditions_met": 5, "distance_to_res_pct": 0.0,
        })
    for i in range(watch):
        results.append({
            "signal": "WATCH", "symbol": f"WCH{i}.IS", "price": 80.0 + i,
            "reason": "test", "risk_level": "LOW", "strength": 0.4,
            "stop_loss": None, "take_profit": None,
            "market_filter": "favorable", "conditions_met": 4, "distance_to_res_pct": 1.5,
        })
    return results


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

    def test_is_eklendi(self):
        s = _make_scheduler()
        try:
            s.start()
            job = s._scheduler.get_job(s._job_id)
            assert job is not None
            assert job.id == s._job_id
        finally:
            s.stop()

    def test_varsayilan_saat_18_10(self):
        s = _make_scheduler()
        assert s._hour == 18
        assert s._minute == 10

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
                        "job_id", "symbol_count"):
                assert key in status, f"'{key}' eksik"
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
            assert "18:10" in nrt
        finally:
            s.stop()

    def test_update_schedule_saati_gunceller(self):
        s = _make_scheduler()
        try:
            s.start()
            s.update_schedule(9, 30)
            assert s._hour == 9
            assert s._minute == 30
            nrt = s.next_run_time()
            assert "09:30" in nrt
        finally:
            s.stop()


# ── run_daily_scan ────────────────────────────────────────────────────────────

class TestRunDailyScan:
    @pytest.mark.asyncio
    async def test_buy_sinyali_varsa_mesaj_gonderilir(self):
        results = _mock_scan_results(buy=2, watch=1)
        with patch("app.scheduler.scan_market", return_value=results):
            with patch("app.scheduler.send_telegram_message", new_callable=AsyncMock) as mock_send:
                outcome = await run_daily_scan()
        assert mock_send.call_count >= 1   # en az özet mesajı

    @pytest.mark.asyncio
    async def test_sinyal_yoksa_bilgi_mesaji(self):
        with patch("app.scheduler.scan_market", return_value=[]):
            with patch("app.scheduler.send_telegram_message", new_callable=AsyncMock) as mock_send:
                outcome = await run_daily_scan()
        assert mock_send.call_count == 1
        msg = mock_send.call_args.args[0]
        assert "bulunamadı" in msg.lower()

    @pytest.mark.asyncio
    async def test_sonuc_dict_doner(self):
        with patch("app.scheduler.scan_market", return_value=_mock_scan_results(1, 0)):
            with patch("app.scheduler.send_telegram_message", new_callable=AsyncMock):
                outcome = await run_daily_scan()
        assert isinstance(outcome, dict)
        assert "buy" in outcome
        assert "watch" in outcome
        assert "scanned" in outcome

    @pytest.mark.asyncio
    async def test_buy_sayisi_dogru(self):
        with patch("app.scheduler.scan_market", return_value=_mock_scan_results(buy=3, watch=0)):
            with patch("app.scheduler.send_telegram_message", new_callable=AsyncMock):
                outcome = await run_daily_scan()
        assert outcome["buy"] == 3

    @pytest.mark.asyncio
    async def test_scan_market_cagrilir(self):
        with patch("app.scheduler.scan_market", return_value=[]) as mock_scan:
            with patch("app.scheduler.send_telegram_message", new_callable=AsyncMock):
                await run_daily_scan()
        assert mock_scan.called

    @pytest.mark.asyncio
    async def test_force_market_refresh_true(self):
        with patch("app.scheduler.scan_market", return_value=[]) as mock_scan:
            with patch("app.scheduler.send_telegram_message", new_callable=AsyncMock):
                await run_daily_scan()
        call_kwargs = mock_scan.call_args.kwargs
        assert call_kwargs.get("force_market_refresh") is True

    @pytest.mark.asyncio
    async def test_her_buy_icin_ayri_mesaj(self):
        results = _mock_scan_results(buy=3, watch=0)
        with patch("app.scheduler.scan_market", return_value=results):
            with patch("app.scheduler.send_telegram_message", new_callable=AsyncMock) as mock_send:
                await run_daily_scan()
        # 3 BUY mesajı + 1 özet = 4 çağrı
        assert mock_send.call_count == 4

    @pytest.mark.asyncio
    async def test_telegram_hatasi_exception_firlatmaz(self):
        with patch("app.scheduler.scan_market", return_value=_mock_scan_results(1, 0)):
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
        for key in ("running", "schedule", "next_run", "timezone", "symbol_count"):
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

    def test_update_schedule_200(self, client):
        r = client.post("/scheduler/update?hour=9&minute=30")
        assert r.status_code == 200

    def test_update_schedule_yanit(self, client):
        body = client.post("/scheduler/update?hour=10&minute=0").json()
        assert "next_run" in body
        # Saati geri al
        client.post("/scheduler/update?hour=18&minute=10")

    def test_update_gecersiz_saat_422(self, client):
        r = client.post("/scheduler/update?hour=25&minute=0")
        assert r.status_code == 422

    def test_update_gecersiz_dakika_422(self, client):
        r = client.post("/scheduler/update?hour=18&minute=60")
        assert r.status_code == 422
