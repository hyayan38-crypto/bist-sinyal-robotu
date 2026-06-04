"""
FastAPI endpoint testleri — ağ bağlantısı gerekmez, mock kullanılır.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient

from app.main import app
from app.data.fetcher import FetchResult, FetchError
from app.risk.market_filter import MarketFilterResult, STATUS_FAVORABLE, STATUS_UNFAVORABLE
from app.backtest.runner import BacktestResult


# ── Test istemcisi ────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


# ── Yardımcı ──────────────────────────────────────────────────────────────────

def _make_ohlcv(n: int = 250) -> pd.DataFrame:
    np.random.seed(7)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame(
        {
            "open":   close * 0.99,
            "high":   close * 1.01,
            "low":    close * 0.98,
            "close":  close,
            "volume": np.random.randint(500_000, 2_000_000, n).astype(float),
        },
        index=pd.date_range("2023-01-01", periods=n, freq="B"),
    )


def _fetch_result(symbol="THYAO.IS") -> FetchResult:
    return FetchResult(symbol=symbol, df=_make_ohlcv(), period="1y", interval="1d")


def _favorable() -> MarketFilterResult:
    return MarketFilterResult(favorable=True, status=STATUS_FAVORABLE, reason="test")


def _bt_result(symbol="THYAO.IS") -> BacktestResult:
    return BacktestResult(
        symbol=symbol, total_return_pct=12.5, total_trades=6,
        win_rate_pct=66.7, max_drawdown_pct=-5.2,
        best_trade_pct=5.8, worst_trade_pct=-2.1, avg_trade_pct=2.1,
        sharpe_ratio=1.23, profit_factor=2.1,
        start_date="2023-01-01", end_date="2024-12-31",
    )


# ─────────────────────────────────────────────────────────────────────────────
# GET /health
# ─────────────────────────────────────────────────────────────────────────────

class TestHealth:
    def test_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_has_status_ok(self, client):
        r = client.get("/health")
        assert r.json()["status"] == "ok"

    def test_has_version(self, client):
        r = client.get("/health")
        assert "version" in r.json()

    def test_has_symbol_count(self, client):
        r = client.get("/health")
        assert "symbol_count" in r.json()

    def test_has_telegram_flag(self, client):
        r = client.get("/health")
        assert "telegram_configured" in r.json()


# ─────────────────────────────────────────────────────────────────────────────
# GET /symbols
# ─────────────────────────────────────────────────────────────────────────────

class TestSymbols:
    def test_returns_200(self, client):
        assert client.get("/symbols").status_code == 200

    def test_has_count_and_list(self, client):
        r = client.get("/symbols").json()
        assert "count" in r
        assert "symbols" in r
        assert r["count"] == len(r["symbols"])

    def test_symbols_end_with_is(self, client):
        syms = client.get("/symbols").json()["symbols"]
        assert all(s.endswith(".IS") for s in syms)

    def test_add_symbol(self, client):
        r = client.post("/symbols/TESTX")
        assert r.status_code in (200, 409)
        if r.status_code == 200:
            assert "TESTX.IS" in r.json()["symbols"]
            # temizle
            client.delete("/symbols/TESTX")

    def test_add_duplicate_returns_409(self, client):
        client.post("/symbols/DUPTEST")
        r = client.post("/symbols/DUPTEST")
        assert r.status_code == 409
        client.delete("/symbols/DUPTEST")

    def test_remove_missing_returns_404(self, client):
        r = client.delete("/symbols/NONEXISTENT99")
        assert r.status_code == 404

    def test_reset_returns_default_list(self, client):
        client.post("/symbols/EXTRAX")
        r = client.post("/symbols/reset")
        assert r.status_code == 200
        syms = r.json()["symbols"]
        assert "EXTRAX.IS" not in syms


# ─────────────────────────────────────────────────────────────────────────────
# GET /scan
# ─────────────────────────────────────────────────────────────────────────────

class TestScan:
    def test_returns_200(self, client):
        with patch("app.main.scan_market", return_value=[]):
            r = client.get("/scan")
        assert r.status_code == 200

    def test_response_structure(self, client):
        with patch("app.main.scan_market", return_value=[]):
            r = client.get("/scan").json()
        assert "scanned" in r
        assert "buy_count" in r
        assert "watch_count" in r
        assert "results" in r

    def test_buy_count_correct(self, client):
        fake = [
            {"signal": "BUY",   "symbol": "A.IS", "price": 100.0,
             "reason": "", "risk_level": "LOW", "strength": 0.7},
            {"signal": "WATCH", "symbol": "B.IS", "price": 50.0,
             "reason": "", "risk_level": "LOW", "strength": 0.4},
        ]
        with patch("app.main.scan_market", return_value=fake):
            r = client.get("/scan").json()
        assert r["buy_count"] == 1
        assert r["watch_count"] == 1

    def test_include_watch_false_param_forwarded(self, client):
        with patch("app.main.scan_market", return_value=[]) as mock_scan:
            client.get("/scan?include_watch=false")
        call_kwargs = mock_scan.call_args.kwargs
        assert call_kwargs.get("include_watch") is False

    def test_custom_symbols_param(self, client):
        with patch("app.main.scan_market", return_value=[]) as mock_scan:
            client.get("/scan?symbols=THYAO.IS&symbols=GARAN.IS")
        call_args = mock_scan.call_args
        sym_list = call_args.args[0] if call_args.args else call_args.kwargs.get("symbols", [])
        # En az semboller iletildi
        assert isinstance(sym_list, list)


# ─────────────────────────────────────────────────────────────────────────────
# GET /signal/{symbol}
# ─────────────────────────────────────────────────────────────────────────────

class TestSignalEndpoint:
    def _patch_all(self, symbol="THYAO.IS", tb_signal="HOLD"):
        """fetch + indicator + tb_signal + market_filter mock'larını döner."""
        tb_out = {
            "signal": tb_signal, "price": 100.0,
            "reason": "test", "risk_level": "LOW", "strength": 0.0,
            "stop_loss": None, "take_profit": None,
            "details": {
                "c1_above_ema20": True, "c2_ema_uptrend": True,
                "c3_breakout": False, "c4_volume_surge": False, "c5_rsi_range": True,
                "close": 100.0, "ema_20": 98.0, "ema_50": 95.0,
                "rsi_14": 58.0, "atr_14": 1.5, "volume_ratio": 1.2,
                "prev_resistance": 102.0,
            },
        }
        if tb_signal == "BUY":
            tb_out["stop_loss"] = 97.0
            tb_out["take_profit"] = 106.0
            tb_out["strength"] = 0.72

        ctx = [
            patch("app.main.fetch_symbol_data", return_value=_fetch_result(symbol)),
            patch("app.main.add_indicators", return_value=_make_ohlcv()),
            patch("app.main.tb_signal", return_value=tb_out),
            patch("app.main.is_market_favorable", return_value=_favorable()),
        ]
        return ctx

    def test_returns_200_on_valid_symbol(self, client):
        ctxs = self._patch_all()
        with ctxs[0], ctxs[1], ctxs[2], ctxs[3]:
            r = client.get("/signal/THYAO")
        assert r.status_code == 200

    def test_response_has_required_keys(self, client):
        ctxs = self._patch_all()
        with ctxs[0], ctxs[1], ctxs[2], ctxs[3]:
            body = client.get("/signal/THYAO").json()
        for key in ("symbol", "signal", "price", "reason", "risk_level",
                    "strength", "stop_loss", "take_profit",
                    "conditions", "market_filter", "indicators", "data_range"):
            assert key in body, f"'{key}' eksik"

    def test_returns_404_on_fetch_error(self, client):
        with patch("app.main.fetch_symbol_data", side_effect=FetchError("X.IS", "hata")):
            r = client.get("/signal/BADONE")
        assert r.status_code == 404

    def test_buy_signal_has_stop_take(self, client):
        ctxs = self._patch_all(tb_signal="BUY")
        with ctxs[0], ctxs[1], ctxs[2], ctxs[3]:
            body = client.get("/signal/THYAO").json()
        assert body["signal"] == "BUY"
        assert body["stop_loss"] is not None
        assert body["take_profit"] is not None

    def test_buy_blocked_when_unfavorable(self, client):
        tb_out = {
            "signal": "BUY", "price": 100.0, "reason": "test",
            "risk_level": "LOW", "strength": 0.8,
            "stop_loss": 97.0, "take_profit": 106.0, "details": {},
        }
        unfav = MarketFilterResult(
            favorable=False, status=STATUS_UNFAVORABLE, reason="XU100 düşüşte"
        )
        with patch("app.main.fetch_symbol_data", return_value=_fetch_result()):
            with patch("app.main.add_indicators", return_value=_make_ohlcv()):
                with patch("app.main.tb_signal", return_value=tb_out):
                    with patch("app.main.is_market_favorable", return_value=unfav):
                        body = client.get("/signal/THYAO").json()
        assert body["signal"] == "HOLD"
        assert body["stop_loss"] is None
        assert "Endeks" in body["reason"]

    def test_market_filter_not_applied_when_disabled(self, client):
        ctxs = self._patch_all(tb_signal="BUY")
        with ctxs[0], ctxs[1], ctxs[2], ctxs[3]:
            body = client.get("/signal/THYAO?apply_filter=false").json()
        assert body["market_filter"]["status"] == "not_applied"

    def test_data_range_present(self, client):
        ctxs = self._patch_all()
        with ctxs[0], ctxs[1], ctxs[2], ctxs[3]:
            body = client.get("/signal/THYAO").json()
        assert "bars" in body["data_range"]
        assert body["data_range"]["bars"] > 0


# ─────────────────────────────────────────────────────────────────────────────
# GET /backtest/{symbol}
# ─────────────────────────────────────────────────────────────────────────────

class TestBacktestEndpoint:
    def test_returns_200_on_success(self, client):
        with patch("app.main.bt_run_single", return_value=_bt_result()):
            r = client.get("/backtest/THYAO")
        assert r.status_code == 200

    def test_response_has_all_metrics(self, client):
        with patch("app.main.bt_run_single", return_value=_bt_result()):
            body = client.get("/backtest/THYAO").json()
        for key in ("total_return_pct", "total_trades", "win_rate_pct",
                    "max_drawdown_pct", "best_trade_pct", "worst_trade_pct",
                    "avg_trade_pct", "sharpe_ratio", "profit_factor"):
            assert key in body, f"'{key}' eksik"

    def test_returns_400_on_error(self, client):
        err_result = BacktestResult(symbol="X.IS", error="veri yok")
        with patch("app.main.bt_run_single", return_value=err_result):
            r = client.get("/backtest/BADONE")
        assert r.status_code == 400

    def test_period_param_forwarded(self, client):
        with patch("app.main.bt_run_single", return_value=_bt_result()) as mock_bt:
            client.get("/backtest/THYAO?period=5y")
        assert mock_bt.call_args.kwargs.get("period") == "5y"

    def test_cash_param_forwarded(self, client):
        with patch("app.main.bt_run_single", return_value=_bt_result()) as mock_bt:
            client.get("/backtest/THYAO?cash=50000")
        assert mock_bt.call_args.kwargs.get("cash") == 50_000.0


# ─────────────────────────────────────────────────────────────────────────────
# GET /market-filter
# ─────────────────────────────────────────────────────────────────────────────

class TestMarketFilter:
    def test_returns_200(self, client):
        with patch("app.main.is_market_favorable", return_value=_favorable()):
            r = client.get("/market-filter")
        assert r.status_code == 200

    def test_has_required_fields(self, client):
        with patch("app.main.is_market_favorable", return_value=_favorable()):
            body = client.get("/market-filter").json()
        for key in ("status", "favorable", "blocks_buy", "reason"):
            assert key in body

    def test_invalidate_returns_200(self, client):
        with patch("app.main.invalidate_cache"):
            r = client.post("/market-filter/invalidate")
        assert r.status_code == 200


# ─────────────────────────────────────────────────────────────────────────────
# GET /backtest  (çoklu)
# ─────────────────────────────────────────────────────────────────────────────

class TestBacktestAll:
    def test_returns_200(self, client):
        from app.backtest.runner import MultiBacktestResult
        multi = MultiBacktestResult(results=[_bt_result()], period="2y", initial_cash=100_000)
        with patch("app.main.bt_run_multiple", return_value=multi):
            r = client.get("/backtest")
        assert r.status_code == 200

    def test_has_summary(self, client):
        from app.backtest.runner import MultiBacktestResult
        multi = MultiBacktestResult(results=[_bt_result()], period="2y", initial_cash=100_000)
        with patch("app.main.bt_run_multiple", return_value=multi):
            body = client.get("/backtest").json()
        assert "summary" in body


# ─────────────────────────────────────────────────────────────────────────────
# GET /test-telegram
# ─────────────────────────────────────────────────────────────────────────────

class TestTelegramEndpoint:
    """GET /test-telegram — üç senaryo: başarılı, yapılandırılmamış, gönderim hatası."""

    def test_yapilandirilmamis_400_doner(self, client):
        with patch("app.main.is_configured", return_value=False):
            r = client.get("/test-telegram")
        assert r.status_code == 400
        assert "TELEGRAM_BOT_TOKEN" in r.json()["detail"]

    def test_basarili_gonderim_200_doner(self, client):
        with patch("app.main.is_configured", return_value=True):
            with patch("app.main.send_telegram_message", new_callable=AsyncMock, return_value=True):
                r = client.get("/test-telegram")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert "✅" in body["message"]

    def test_gonderim_hatasi_502_doner(self, client):
        with patch("app.main.is_configured", return_value=True):
            with patch("app.main.send_telegram_message", new_callable=AsyncMock, return_value=False):
                r = client.get("/test-telegram")
        assert r.status_code == 502

    def test_basarili_yanit_chat_id_icerir(self, client):
        with patch("app.main.is_configured", return_value=True):
            with patch("app.main.send_telegram_message", new_callable=AsyncMock, return_value=True):
                body = client.get("/test-telegram").json()
        assert "chat_id" in body

    def test_dogru_mesaj_gonderilir(self, client):
        with patch("app.main.is_configured", return_value=True):
            with patch("app.main.send_telegram_message", new_callable=AsyncMock, return_value=True) as mock_send:
                client.get("/test-telegram")
        call_args = mock_send.call_args
        assert "✅ Telegram bağlantısı başarılı" in call_args.args[0]
