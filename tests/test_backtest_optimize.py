"""
Parametre optimizasyonu testleri: run_single(params), grid_search, walk_forward.
Ağ gerektirmez — fetch_symbol_data mock'lanır.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from app.backtest.runner import (
    run_single,
    grid_search,
    walk_forward,
    _DEFAULT_PARAM_GRID,
)
from app.data.fetcher import FetchResult, FetchError


def _make_ohlcv(n: int = 800, seed: int = 7) -> pd.DataFrame:
    np.random.seed(seed)
    close = np.linspace(80, 140, n) + np.random.randn(n) * 1.5
    high = close + np.abs(np.random.randn(n)) * 1.5
    low = close - np.abs(np.random.randn(n)) * 1.5
    vol = np.random.randint(500_000, 3_000_000, n).astype(float)
    vol[::15] *= 2.5
    return pd.DataFrame(
        {"open": close * 0.998, "high": high, "low": low, "close": close, "volume": vol},
        index=pd.date_range("2020-01-01", periods=n, freq="B"),
    )


def _fetch(symbol="THYAO.IS", n=800):
    return FetchResult(symbol=symbol, df=_make_ohlcv(n), period="5y", interval="1d")


def _patch(result=None):
    return patch("app.backtest.runner.fetch_symbol_data", return_value=result or _fetch())


# Hızlı testler için küçük ızgara
_SMALL_GRID = {"volume_mult": [1.5, 2.0], "atr_sl_mult": [1.0, 1.5]}


# ── run_single(params) ────────────────────────────────────────────────────────

class TestRunSingleParams:
    def test_params_kabul_edilir(self):
        with _patch():
            result = run_single("THYAO", params={"volume_mult": 2.5, "rsi_low": 55})
        assert result.error is None

    def test_gecersiz_param_hata_dondurmez_sonuc_dondurur(self):
        # backtesting.py bilinmeyen attribute'u AttributeError yapar → _error_result
        with _patch():
            result = run_single("THYAO", params={"saçma_param": 1})
        assert result.error is not None


# ── grid_search ───────────────────────────────────────────────────────────────

class TestGridSearch:
    def test_temel_yapı(self):
        with _patch():
            out = grid_search("THYAO", param_grid=_SMALL_GRID, metric="total_return_pct", min_trades=1)
        assert out["symbol"] == "THYAO.IS"
        assert out["combinations"] == 4
        assert "results" in out and isinstance(out["results"], list)

    def test_metrige_gore_sirali(self):
        with _patch():
            out = grid_search("THYAO", param_grid=_SMALL_GRID, metric="total_return_pct", min_trades=1)
        vals = [r["total_return_pct"] for r in out["results"]]
        assert vals == sorted(vals, reverse=True)

    def test_gecersiz_metrik(self):
        with _patch():
            with pytest.raises(ValueError):
                grid_search("THYAO", param_grid=_SMALL_GRID, metric="olmayan")

    def test_gecersiz_param_grid(self):
        with _patch():
            with pytest.raises(ValueError):
                grid_search("THYAO", param_grid={"olmayan_param": [1, 2]})

    def test_fetch_hatasi(self):
        with patch("app.backtest.runner.fetch_symbol_data", side_effect=FetchError("X.IS", "yok")):
            out = grid_search("X", param_grid=_SMALL_GRID)
        assert out["error"]
        assert out["results"] == []

    def test_default_grid_optimizable(self):
        # Varsayılan ızgaranın tüm anahtarları geçerli olmalı
        with _patch():
            out = grid_search("THYAO", param_grid=_DEFAULT_PARAM_GRID, min_trades=1, top_n=3)
        assert out["combinations"] == 81
        assert len(out["results"]) <= 3


# ── walk_forward ──────────────────────────────────────────────────────────────

class TestWalkForward:
    def test_temel_yapı(self):
        with _patch():
            out = walk_forward("THYAO", param_grid=_SMALL_GRID, n_splits=3,
                               metric="total_return_pct", min_trades=1)
        assert out["symbol"] == "THYAO.IS"
        assert out["n_splits"] == 3
        assert "summary" in out
        assert len(out["folds"]) == 3

    def test_oos_metrikleri_var(self):
        with _patch():
            out = walk_forward("THYAO", param_grid=_SMALL_GRID, n_splits=3,
                               metric="total_return_pct", min_trades=1)
        s = out["summary"]
        assert "avg_oos_return_pct" in s
        assert "oos_positive_rate" in s
        evaluated = [f for f in out["folds"] if "out_of_sample" in f]
        for f in evaluated:
            assert "best_params" in f
            assert "out_of_sample" in f

    def test_yetersiz_veri(self):
        with _patch(_fetch(n=200)):  # 200 / (4+1) = 40 bar/dilim < 80
            out = walk_forward("THYAO", param_grid=_SMALL_GRID, n_splits=4)
        assert out["error"]
        assert out["folds"] == []
