"""
Entegrasyon testleri — projenin beş temel işlevini doğrular.

1. Veri çekme        → FetchResult yapısı, kolon standardı, temizleme
2. İndikatör hesaplama → add_indicators() çıktısı, değer aralıkları
3. Sinyal üretme      → trend_breakout ve strateji motoru
4. Backtest JSON      → BacktestResult.to_dict() eksiksizliği
5. API /health        → FastAPI endpoint erişilebilirliği

Tüm testler ağ bağlantısı gerektirmez — yfinance mock'lanır.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# ── Proje import'ları ─────────────────────────────────────────────────────────
from app.data.fetcher import (
    FetchResult,
    FetchError,
    EmptyDataError,
    InsufficientDataError,
    fetch_symbol_data,
    fetch_multiple_symbols,
    _normalize_symbol,
    _clean_ohlcv,
)
from app.indicators.technical import (
    add_indicators,
    add_ema,
    add_rsi,
    add_macd,
    add_atr,
    add_bollinger,
    add_resistance,
    add_volatility_squeeze,
    get_latest,
    _INDICATOR_COLS,
)
from app.strategies.trend_breakout import (
    generate_signal,
    TrendBreakoutStrategy,
    _STOP_LOSS_PCT,
    _TAKE_PROFIT_PCT,
)
from app.strategies.base import SignalType
from app.backtest.runner import (
    BacktestResult,
    MultiBacktestResult,
    run_single,
    run_multiple,
    _INITIAL_CASH,
    _COMMISSION,
)
from app.main import app


# ── Yardımcı ──────────────────────────────────────────────────────────────────

def _raw_yf_df(n: int = 120, trend: str = "up") -> pd.DataFrame:
    """yfinance'in döndürdüğü format: büyük harf sütunlar + timezone'lu index."""
    np.random.seed(1)
    if trend == "up":
        close = np.linspace(80, 130, n) + np.random.randn(n) * 0.8
    else:
        close = np.linspace(130, 80, n) + np.random.randn(n) * 0.8

    idx = pd.date_range("2023-01-01", periods=n, freq="B", tz="America/New_York")
    return pd.DataFrame(
        {
            "Open":       close * 0.99,
            "High":       close * 1.01,
            "Low":        close * 0.98,
            "Close":      close,
            "Volume":     np.random.randint(500_000, 3_000_000, n).astype(float),
            "Dividends":  0.0,
            "Stock Splits": 0.0,
        },
        index=idx,
    )


def _mock_ticker(df: pd.DataFrame):
    m = MagicMock()
    m.history.return_value = df
    return m


def _ohlcv(n: int = 250, trend: str = "up") -> pd.DataFrame:
    """Temiz, küçük harf, tz-naive OHLCV (fetch sonrası format)."""
    np.random.seed(7)
    if trend == "up":
        close = np.linspace(80, 130, n) + np.random.randn(n) * 0.8
    else:
        close = np.linspace(130, 80, n) + np.random.randn(n) * 0.8
    high = close + np.abs(np.random.randn(n)) * 1.0
    low  = close - np.abs(np.random.randn(n)) * 1.0
    vol  = np.random.randint(500_000, 3_000_000, n).astype(float)
    return pd.DataFrame(
        {"open": close * 0.998, "high": high, "low": low, "close": close, "volume": vol},
        index=pd.date_range("2023-01-01", periods=n, freq="B"),
    )


def _fetch_result(symbol: str = "THYAO.IS") -> FetchResult:
    return FetchResult(symbol=symbol, df=_ohlcv(), period="1y", interval="1d")


# ═════════════════════════════════════════════════════════════════════════════
# 1. VERİ ÇEKME TESTLERİ
# ═════════════════════════════════════════════════════════════════════════════

class TestVeriCekme:
    """fetch_symbol_data ve _clean_ohlcv pipeline'ını doğrular."""

    # ── _normalize_symbol ─────────────────────────────────────────────────────

    def test_normalize_buyuk_harf(self):
        assert _normalize_symbol("thyao") == "THYAO.IS"

    def test_normalize_is_uzantisi_eklenir(self):
        assert _normalize_symbol("GARAN") == "GARAN.IS"

    def test_normalize_is_tekrar_eklenmez(self):
        assert _normalize_symbol("AKBNK.IS") == "AKBNK.IS"

    def test_normalize_bosluk_temizlenir(self):
        assert _normalize_symbol("  sise  ") == "SISE.IS"

    # ── _clean_ohlcv ──────────────────────────────────────────────────────────

    def test_clean_kolonlari_kucuk_harfe_cevirir(self):
        df = _raw_yf_df()
        clean = _clean_ohlcv(df, "TEST.IS")
        assert list(clean.columns) == ["open", "high", "low", "close", "volume"]

    def test_clean_timezone_kaldirilir(self):
        df = _raw_yf_df()
        clean = _clean_ohlcv(df, "TEST.IS")
        assert clean.index.tz is None

    def test_clean_nan_satirlar_atilir(self):
        df = _raw_yf_df(60)
        df.iloc[5, df.columns.get_loc("Close")] = float("nan")
        clean = _clean_ohlcv(df, "TEST.IS")
        assert len(clean) == 59

    def test_clean_sifir_close_atilir(self):
        df = _raw_yf_df(60)
        df.iloc[3, df.columns.get_loc("Close")] = 0.0
        clean = _clean_ohlcv(df, "TEST.IS")
        assert len(clean) == 59

    def test_clean_ters_hl_atilir(self):
        df = _raw_yf_df(60)
        df.iloc[10, df.columns.get_loc("High")] = 0.5
        df.iloc[10, df.columns.get_loc("Low")] = 999.0
        clean = _clean_ohlcv(df, "TEST.IS")
        assert len(clean) == 59

    def test_clean_eksik_kolon_hata_firlatir(self):
        df = _raw_yf_df().drop(columns=["Volume"])
        with pytest.raises(FetchError):
            _clean_ohlcv(df, "TEST.IS")

    # ── fetch_symbol_data (mock) ───────────────────────────────────────────────

    def test_bos_veri_empty_data_error(self):
        with patch("app.data.fetcher.yf.Ticker", return_value=_mock_ticker(pd.DataFrame())):
            with pytest.raises(EmptyDataError):
                fetch_symbol_data("THYAO")

    def test_yetersiz_veri_insufficient_error(self):
        with patch("app.data.fetcher.yf.Ticker", return_value=_mock_ticker(_raw_yf_df(10))):
            with pytest.raises(InsufficientDataError):
                fetch_symbol_data("THYAO", min_rows=30)

    def test_basarili_cekme_fetch_result_doner(self):
        with patch("app.data.fetcher.yf.Ticker", return_value=_mock_ticker(_raw_yf_df(120))):
            result = fetch_symbol_data("THYAO")
        assert isinstance(result, FetchResult)
        assert result.symbol == "THYAO.IS"

    def test_fetch_result_kolon_standardi(self):
        with patch("app.data.fetcher.yf.Ticker", return_value=_mock_ticker(_raw_yf_df(120))):
            result = fetch_symbol_data("THYAO")
        assert list(result.df.columns) == ["open", "high", "low", "close", "volume"]

    def test_fetch_result_tz_naive(self):
        with patch("app.data.fetcher.yf.Ticker", return_value=_mock_ticker(_raw_yf_df(120))):
            result = fetch_symbol_data("THYAO")
        assert result.df.index.tz is None

    def test_fetch_result_row_count(self):
        with patch("app.data.fetcher.yf.Ticker", return_value=_mock_ticker(_raw_yf_df(120))):
            result = fetch_symbol_data("THYAO")
        assert result.row_count == 120

    def test_yfinance_hatasi_fetch_error_firlatir(self):
        m = MagicMock()
        m.history.side_effect = RuntimeError("bağlantı hatası")
        with patch("app.data.fetcher.yf.Ticker", return_value=m):
            with pytest.raises(FetchError):
                fetch_symbol_data("THYAO")

    def test_cache_kayitlari_donusturme(self):
        with patch("app.data.fetcher.yf.Ticker", return_value=_mock_ticker(_raw_yf_df(120))):
            result = fetch_symbol_data("THYAO")
        records = result.to_cache_records()
        assert len(records) == 120
        assert "symbol" in records[0]
        assert "date" in records[0]
        assert records[0]["date"].tzinfo is None

    # ── fetch_multiple_symbols ────────────────────────────────────────────────

    def test_coklu_hata_atlama(self):
        good = _mock_ticker(_raw_yf_df(120))
        bad  = MagicMock()
        bad.history.return_value = pd.DataFrame()

        def side_effect(sym):
            return good if "THYAO" in sym else bad

        with patch("app.data.fetcher.yf.Ticker", side_effect=side_effect):
            results = fetch_multiple_symbols(["THYAO", "BADONE"], skip_errors=True)
        assert "THYAO.IS" in results
        assert "BADONE.IS" not in results

    def test_bos_liste_bos_dict(self):
        assert fetch_multiple_symbols([]) == {}


# ═════════════════════════════════════════════════════════════════════════════
# 2. İNDİKATÖR HESAPLAMA TESTLERİ
# ═════════════════════════════════════════════════════════════════════════════

class TestIndikatorHesaplama:
    """add_indicators() pipeline'ının doğruluğunu ve değer aralıklarını test eder."""

    @pytest.fixture(autouse=True)
    def df_with_indicators(self):
        self.df = add_indicators(_ohlcv(250))

    # ── Temel yapı ────────────────────────────────────────────────────────────

    def test_tum_kolon_isimleri_mevcut(self):
        for col in _INDICATOR_COLS:
            assert col in self.df.columns, f"Eksik kolon: {col}"

    def test_orijinal_kolonlar_korunur(self):
        for col in ("open", "high", "low", "close", "volume"):
            assert col in self.df.columns

    def test_satir_sayisi_degismez(self):
        assert len(self.df) == 250

    def test_kopya_doner_orijinal_degismez(self):
        orig = _ohlcv(100)
        orig_cols = set(orig.columns)
        add_indicators(orig)
        assert set(orig.columns) == orig_cols

    def test_eksik_kolon_hata_firlatir(self):
        df = _ohlcv(100).drop(columns=["volume"])
        with pytest.raises(ValueError):
            add_indicators(df)

    # ── EMA değer aralığı ─────────────────────────────────────────────────────

    def test_ema20_son_satir_nan_degil(self):
        assert pd.notna(self.df["ema_20"].iloc[-1])

    def test_ema50_son_satir_nan_degil(self):
        assert pd.notna(self.df["ema_50"].iloc[-1])

    def test_ema200_son_satir_nan_degil(self):
        assert pd.notna(self.df["ema_200"].iloc[-1])

    def test_ema_yukselen_trend_uyumu(self):
        last = self.df.iloc[-1]
        # Yükseliş trendinde EMA20 > EMA50 beklenir
        assert last["ema_20"] > last["ema_50"]

    # ── RSI14 aralığı ─────────────────────────────────────────────────────────

    def test_rsi_0_100_araliginda(self):
        valid = self.df["rsi_14"].dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_rsi_yukselen_trend_50_ustunde(self):
        assert self.df["rsi_14"].dropna().iloc[-1] > 50

    # ── MACD ──────────────────────────────────────────────────────────────────

    def test_macd_hist_cizgi_minus_sinyal(self):
        valid = self.df.dropna(subset=["macd", "macd_signal", "macd_hist"])
        diff = (valid["macd"] - valid["macd_signal"] - valid["macd_hist"]).abs()
        assert (diff < 1e-6).all()

    # ── ATR14 ─────────────────────────────────────────────────────────────────

    def test_atr_pozitif(self):
        assert (self.df["atr_14"].dropna() > 0).all()

    # ── Bollinger Bands ───────────────────────────────────────────────────────

    def test_bb_ust_alt_ustu(self):
        valid = self.df.dropna(subset=["bb_upper", "bb_lower"])
        assert (valid["bb_upper"] > valid["bb_lower"]).all()

    def test_bb_width_pozitif(self):
        assert (self.df["bb_width"].dropna() > 0).all()

    # ── Direnç (resistance_20) ────────────────────────────────────────────────

    def test_direnc_high_dan_buyuk_veya_esit(self):
        valid = self.df.dropna(subset=["resistance_20"])
        assert (valid["resistance_20"] >= valid["high"]).all()

    def test_direnc_ilk_19_satir_nan(self):
        assert self.df["resistance_20"].iloc[:19].isna().all()

    # ── Volatilite daralması ──────────────────────────────────────────────────

    def test_squeeze_bool_tipi(self):
        valid = self.df["volatility_squeeze"].dropna()
        assert set(valid.unique()).issubset({True, False})

    # ── get_latest ────────────────────────────────────────────────────────────

    def test_get_latest_dict_doner(self):
        assert isinstance(get_latest(self.df), dict)

    def test_get_latest_nan_icermez(self):
        result = get_latest(self.df)
        for k, v in result.items():
            if isinstance(v, float):
                assert not math.isnan(v), f"{k} NaN"

    def test_get_latest_ana_indikatorler_mevcut(self):
        result = get_latest(self.df)
        for key in ("ema_20", "ema_50", "rsi_14", "atr_14", "macd"):
            assert key in result


# ═════════════════════════════════════════════════════════════════════════════
# 3. SİNYAL ÜRETME TESTLERİ
# ═════════════════════════════════════════════════════════════════════════════

class TestSinyalUretme:
    """trend_breakout ve strateji entegrasyonunu test eder."""

    def _al_df(self) -> pd.DataFrame:
        """Tüm 5 AL koşulunu sağlayan DataFrame (sentetik)."""
        n = 60
        idx  = pd.date_range("2023-01-01", periods=n, freq="B")
        close = np.linspace(95, 110, n)
        ema20 = close - 2
        ema50 = ema20 - 3
        prev_res  = close - 1
        res_20    = np.roll(prev_res, 1)
        res_20[0] = prev_res[0]

        return pd.DataFrame({
            "close":         close,
            "open":          close * 0.99,
            "high":          close + 0.5,
            "low":           close - 0.5,
            "ema_20":        ema20,
            "ema_50":        ema50,
            "rsi_14":        np.full(n, 60.0),
            "atr_14":        np.full(n, 1.5),
            "volume":        np.full(n, 2_000_000.0),
            "volume_ma20":   np.full(n, 1_000_000.0),
            "resistance_20": res_20,
        }, index=idx)

    # ── generate_signal (standalone) ──────────────────────────────────────────

    def test_al_sinyali_uretir(self):
        result = generate_signal(self._al_df())
        assert result["signal"] == "BUY"

    def test_al_stop_loss_ayarlanir(self):
        result = generate_signal(self._al_df())
        price  = result["price"]
        assert result["stop_loss"] == round(price * (1 - _STOP_LOSS_PCT), 2)

    def test_al_take_profit_ayarlanir(self):
        result = generate_signal(self._al_df())
        price  = result["price"]
        assert result["take_profit"] == round(price * (1 + _TAKE_PROFIT_PCT), 2)

    def test_al_guc_skoru_0_1_arasi(self):
        result = generate_signal(self._al_df())
        assert 0.0 <= result["strength"] <= 1.0

    def test_al_risk_seviyesi_gecerli(self):
        result = generate_signal(self._al_df())
        assert result["risk_level"] in ("LOW", "MEDIUM", "HIGH")

    def test_al_detaylar_mevcut(self):
        result = generate_signal(self._al_df())
        for key in ("c1_above_ema20", "c2_ema_uptrend", "c3_breakout",
                    "c4_volume_surge", "c5_rsi_range"):
            assert key in result["details"]

    def test_hold_rsi_disinda(self):
        df = self._al_df()
        df["rsi_14"] = 80.0
        assert generate_signal(df)["signal"] == "HOLD"

    def test_hold_hacim_yetersiz(self):
        df = self._al_df()
        df["volume"] = df["volume_ma20"] * 1.5
        assert generate_signal(df)["signal"] == "HOLD"

    def test_hold_eksik_kolon(self):
        df = self._al_df().drop(columns=["resistance_20"])
        assert generate_signal(df)["signal"] == "HOLD"

    def test_satis_close_ema20_altinda(self):
        df = self._al_df()
        df["close"] = df["ema_20"] - 3
        assert generate_signal(df)["signal"] == "SELL"

    # ── TrendBreakoutStrategy (BaseStrategy entegrasyonu) ─────────────────────

    def test_strategy_adi(self):
        assert TrendBreakoutStrategy().name == "trend_breakout"

    def test_strategy_signal_nesnesi_doner(self):
        from app.strategies.base import StrategySignal
        strategy = TrendBreakoutStrategy()
        result   = strategy.generate_signal(self._al_df(), "THYAO.IS")
        if result is not None:
            assert isinstance(result, StrategySignal)
            assert result.signal_type == SignalType.BUY

    def test_yetersiz_veri_none_doner(self):
        df = self._al_df().iloc[:30]
        assert TrendBreakoutStrategy().generate_signal(df, "X.IS") is None

    def test_hold_none_doner(self):
        df = self._al_df()
        df["rsi_14"] = 45.0
        assert TrendBreakoutStrategy().generate_signal(df, "X.IS") is None

    # ── Strateji listesi ──────────────────────────────────────────────────────

    def test_trend_breakout_stratejiler_listesinde(self):
        from app.strategies import STRATEGIES
        names = [s.name for s in STRATEGIES]
        assert "trend_breakout" in names

    def test_ema_crossover_stratejiler_listesinde(self):
        from app.strategies import STRATEGIES
        names = [s.name for s in STRATEGIES]
        assert "ema_crossover" in names

    def test_rsi_bollinger_stratejiler_listesinde(self):
        from app.strategies import STRATEGIES
        names = [s.name for s in STRATEGIES]
        assert "rsi_bollinger" in names


# ═════════════════════════════════════════════════════════════════════════════
# 4. BACKTEST SONUCU JSON TESTLERİ
# ═════════════════════════════════════════════════════════════════════════════

class TestBacktestJSON:
    """BacktestResult / MultiBacktestResult JSON serileştirmesini doğrular."""

    def _sample(self, **kw) -> BacktestResult:
        base = dict(
            symbol="THYAO.IS",
            total_return_pct=14.5,
            buy_hold_return_pct=42.1,
            total_trades=8,
            win_rate_pct=62.5,
            max_drawdown_pct=-7.3,
            best_trade_pct=5.8,
            worst_trade_pct=-2.9,
            avg_trade_pct=1.8,
            sharpe_ratio=1.21,
            profit_factor=2.35,
            start_date="2023-01-02",
            end_date="2024-12-27",
        )
        return BacktestResult(**{**base, **kw})

    # ── to_dict() ─────────────────────────────────────────────────────────────

    def test_to_dict_dict_doner(self):
        assert isinstance(self._sample().to_dict(), dict)

    def test_to_dict_zorunlu_metrikler(self):
        d = self._sample().to_dict()
        for key in (
            "symbol", "strategy", "total_return_pct", "total_trades",
            "win_rate_pct", "max_drawdown_pct", "best_trade_pct",
            "worst_trade_pct", "avg_trade_pct", "sharpe_ratio", "profit_factor",
            "initial_cash", "commission_pct", "risk_per_trade_pct",
        ):
            assert key in d, f"'{key}' eksik"

    def test_to_dict_sayi_tipleri(self):
        d = self._sample().to_dict()
        assert isinstance(d["total_return_pct"], float)
        assert isinstance(d["total_trades"], int)
        assert isinstance(d["sharpe_ratio"], float)

    # ── to_json() ─────────────────────────────────────────────────────────────

    def test_to_json_gecerli_json(self):
        parsed = json.loads(self._sample().to_json())
        assert parsed["symbol"] == "THYAO.IS"

    def test_to_json_tum_metrikler(self):
        d = json.loads(self._sample().to_json())
        for key in ("total_return_pct", "win_rate_pct", "max_drawdown_pct",
                    "best_trade_pct", "worst_trade_pct", "avg_trade_pct"):
            assert key in d

    def test_to_json_nan_icermez(self):
        text = self._sample().to_json()
        assert "NaN" not in text
        assert "Infinity" not in text

    # ── profitable ────────────────────────────────────────────────────────────

    def test_profitable_pozitif_getiri(self):
        assert self._sample(total_return_pct=5.0).profitable is True

    def test_profitable_negatif_getiri(self):
        assert self._sample(total_return_pct=-2.0).profitable is False

    def test_profitable_hata_varsa_false(self):
        assert self._sample(total_return_pct=5.0, error="veri yok").profitable is False

    # ── MultiBacktestResult ───────────────────────────────────────────────────

    def test_multi_to_dict_yapi(self):
        multi = MultiBacktestResult(
            results=[self._sample(), self._sample(symbol="GARAN.IS", total_return_pct=-3.0)],
            period="2y",
            initial_cash=_INITIAL_CASH,
        )
        d = multi.to_dict()
        assert "summary" in d
        assert "results" in d
        assert d["summary"]["total_symbols"] == 2

    def test_multi_to_json_gecerli(self):
        multi = MultiBacktestResult(
            results=[self._sample()],
            period="2y",
            initial_cash=_INITIAL_CASH,
        )
        parsed = json.loads(multi.to_json())
        assert "summary" in parsed
        assert "results" in parsed

    def test_multi_ozet_hesaplama(self):
        r1 = self._sample(symbol="THYAO.IS", total_return_pct=20.0)
        r2 = self._sample(symbol="GARAN.IS", total_return_pct=10.0)
        multi = MultiBacktestResult(results=[r1, r2], period="2y", initial_cash=100_000)
        assert multi.summary["avg_return_pct"] == pytest.approx(15.0, abs=0.01)
        assert multi.summary["best_symbol"] == "THYAO.IS"
        assert multi.summary["worst_symbol"] == "GARAN.IS"

    def test_multi_hata_olan_dahil_edilmez_ozete(self):
        ok  = self._sample(symbol="THYAO.IS", total_return_pct=10.0)
        err = BacktestResult(symbol="BADONE.IS", error="veri yok")
        multi = MultiBacktestResult(results=[ok, err], period="2y", initial_cash=100_000)
        assert multi.summary["successful"] == 1
        assert multi.summary["failed"] == 1

    # ── run_single (mock) ─────────────────────────────────────────────────────

    def test_run_single_basarili(self):
        with patch("app.backtest.runner.fetch_symbol_data",
                   return_value=FetchResult("THYAO.IS", _ohlcv(300), "2y", "1d")):
            result = run_single("THYAO")
        assert isinstance(result, BacktestResult)
        assert result.error is None

    def test_run_single_hata_durumu(self):
        with patch("app.backtest.runner.fetch_symbol_data",
                   side_effect=FetchError("THYAO.IS", "hata")):
            result = run_single("THYAO")
        assert result.error is not None

    def test_run_single_json_olusturulabilir(self):
        with patch("app.backtest.runner.fetch_symbol_data",
                   return_value=FetchResult("THYAO.IS", _ohlcv(300), "2y", "1d")):
            result = run_single("THYAO")
        text = result.to_json()
        parsed = json.loads(text)
        assert parsed["symbol"] == "THYAO.IS"


# ═════════════════════════════════════════════════════════════════════════════
# 5. API /health TESTİ
# ═════════════════════════════════════════════════════════════════════════════

class TestAPIHealth:
    """FastAPI endpoint'lerinin temel erişilebilirliğini doğrular."""

    @pytest.fixture(scope="class")
    def client(self):
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c

    # ── /health ───────────────────────────────────────────────────────────────

    def test_health_200_doner(self, client):
        assert client.get("/health").status_code == 200

    def test_health_status_ok(self, client):
        assert client.get("/health").json()["status"] == "ok"

    def test_health_versiyon_mevcut(self, client):
        assert "version" in client.get("/health").json()

    def test_health_telegram_bayragi(self, client):
        assert "telegram_configured" in client.get("/health").json()

    def test_health_sembol_sayisi(self, client):
        r = client.get("/health").json()
        assert "symbol_count" in r
        assert r["symbol_count"] > 0

    # ── /symbols ──────────────────────────────────────────────────────────────

    def test_symbols_200_doner(self, client):
        assert client.get("/symbols").status_code == 200

    def test_symbols_is_uzantisi_ile(self, client):
        syms = client.get("/symbols").json()["symbols"]
        assert all(s.endswith(".IS") for s in syms)

    def test_symbols_count_liste_boyutu_esit(self, client):
        data = client.get("/symbols").json()
        assert data["count"] == len(data["symbols"])

    # ── /scan (mock) ──────────────────────────────────────────────────────────

    def test_scan_200_doner(self, client):
        with patch("app.main.scan_market", return_value=[]):
            assert client.get("/scan").status_code == 200

    def test_scan_yapi(self, client):
        with patch("app.main.scan_market", return_value=[]):
            r = client.get("/scan").json()
        assert "scanned" in r
        assert "buy_count" in r
        assert "watch_count" in r
        assert "results" in r

    # ── /signal/{symbol} (mock) ───────────────────────────────────────────────

    def test_signal_404_veri_yok(self, client):
        with patch("app.main.fetch_symbol_data",
                   side_effect=FetchError("X.IS", "hata")):
            r = client.get("/signal/BADONE")
        assert r.status_code == 404

    def test_signal_200_gecerli_sembol(self, client):
        tb_out = {
            "signal": "HOLD", "price": 100.0, "reason": "test",
            "risk_level": "LOW", "strength": 0.0,
            "stop_loss": None, "take_profit": None,
            "details": {"c1_above_ema20": True, "c2_ema_uptrend": False,
                        "close": 100.0, "ema_20": 98.0, "ema_50": 99.0},
        }
        from app.risk.market_filter import MarketFilterResult, STATUS_FAVORABLE
        mf = MarketFilterResult(favorable=True, status=STATUS_FAVORABLE, reason="ok")
        with patch("app.main.fetch_symbol_data",
                   return_value=FetchResult("THYAO.IS", _ohlcv(), "1y", "1d")):
            with patch("app.main.add_indicators", return_value=_ohlcv()):
                with patch("app.main.tb_signal", return_value=tb_out):
                    with patch("app.main.is_market_favorable", return_value=mf):
                        r = client.get("/signal/THYAO")
        assert r.status_code == 200

    def test_signal_gerekli_alanlar(self, client):
        tb_out = {
            "signal": "HOLD", "price": 100.0, "reason": "test",
            "risk_level": "LOW", "strength": 0.0,
            "stop_loss": None, "take_profit": None, "details": {},
        }
        from app.risk.market_filter import MarketFilterResult, STATUS_FAVORABLE
        mf = MarketFilterResult(favorable=True, status=STATUS_FAVORABLE, reason="ok")
        with patch("app.main.fetch_symbol_data",
                   return_value=FetchResult("THYAO.IS", _ohlcv(), "1y", "1d")):
            with patch("app.main.add_indicators", return_value=_ohlcv()):
                with patch("app.main.tb_signal", return_value=tb_out):
                    with patch("app.main.is_market_favorable", return_value=mf):
                        body = client.get("/signal/THYAO").json()
        for key in ("symbol", "signal", "price", "reason", "risk_level",
                    "strength", "market_filter", "data_range"):
            assert key in body, f"'{key}' eksik"

    # ── /backtest/{symbol} (mock) ─────────────────────────────────────────────

    def test_backtest_200_basarili(self, client):
        from app.backtest.runner import BacktestResult
        ok = BacktestResult(symbol="THYAO.IS", total_return_pct=10.0)
        with patch("app.main.bt_run_single", return_value=ok):
            assert client.get("/backtest/THYAO").status_code == 200

    def test_backtest_400_hata(self, client):
        err = BacktestResult(symbol="X.IS", error="veri yok")
        with patch("app.main.bt_run_single", return_value=err):
            assert client.get("/backtest/BADONE").status_code == 400

    def test_backtest_metrikler_mevcut(self, client):
        ok = BacktestResult(
            symbol="THYAO.IS", total_return_pct=10.0,
            total_trades=5, win_rate_pct=60.0, max_drawdown_pct=-5.0,
            best_trade_pct=4.0, worst_trade_pct=-2.0, avg_trade_pct=1.5,
        )
        with patch("app.main.bt_run_single", return_value=ok):
            body = client.get("/backtest/THYAO").json()
        for key in ("total_return_pct", "total_trades", "win_rate_pct",
                    "max_drawdown_pct", "best_trade_pct", "worst_trade_pct",
                    "avg_trade_pct"):
            assert key in body

    # ── /market-filter ────────────────────────────────────────────────────────

    def test_market_filter_200(self, client):
        from app.risk.market_filter import MarketFilterResult, STATUS_FAVORABLE
        mf = MarketFilterResult(favorable=True, status=STATUS_FAVORABLE, reason="ok")
        with patch("app.main.is_market_favorable", return_value=mf):
            assert client.get("/market-filter").status_code == 200

    # ── Docs erişilebilirliği ─────────────────────────────────────────────────

    def test_swagger_docs_erisim(self, client):
        assert client.get("/docs").status_code == 200

    def test_openapi_json_erisim(self, client):
        r = client.get("/openapi.json")
        assert r.status_code == 200
        assert "paths" in r.json()
