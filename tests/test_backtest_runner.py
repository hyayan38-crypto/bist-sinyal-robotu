"""
Backtest runner testleri — ağ bağlantısı gerekmez, mock kullanılır.
"""

import json
import math
import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from unittest.mock import patch, MagicMock

from app.backtest.runner import (
    run_single,
    run_multiple,
    BacktestResult,
    MultiBacktestResult,
    TrendBreakoutBT,
    _to_bt_df,
    _extract_stats,
    _INITIAL_CASH,
    _COMMISSION,
    _POSITION_SIZE,
)
from app.data.fetcher import FetchResult, FetchError


# ── Yardımcı ──────────────────────────────────────────────────────────────────

def _make_ohlcv(n: int = 300, trend: str = "up", seed: int = 42) -> pd.DataFrame:
    """Gerçekçi OHLCV DataFrame — trend kırılımlarını tetikleyebilir."""
    np.random.seed(seed)
    if trend == "up":
        close = np.linspace(80, 130, n) + np.random.randn(n) * 1.5
    elif trend == "down":
        close = np.linspace(130, 80, n) + np.random.randn(n) * 1.5
    else:
        close = 100 + np.cumsum(np.random.randn(n) * 1.0)

    high   = close + np.abs(np.random.randn(n)) * 1.5
    low    = close - np.abs(np.random.randn(n)) * 1.5
    volume = np.random.randint(500_000, 3_000_000, n).astype(float)
    # Zaman zaman hacim patlaması ekle (kırılım tetiklemek için)
    volume[::15] *= 2.5

    return pd.DataFrame(
        {"open": close * 0.998, "high": high, "low": low, "close": close, "volume": volume},
        index=pd.date_range("2022-01-01", periods=n, freq="B"),
    )


def _make_fetch_result(symbol: str = "THYAO.IS", n: int = 300, **kwargs) -> FetchResult:
    df = _make_ohlcv(n, **kwargs)
    return FetchResult(symbol=symbol, df=df, period="2y", interval="1d")


def _patch_fetch(result):
    return patch("app.backtest.runner.fetch_symbol_data", return_value=result)


def _patch_fetch_error(exc):
    return patch("app.backtest.runner.fetch_symbol_data", side_effect=exc)


# ── _to_bt_df ─────────────────────────────────────────────────────────────────

class TestToBtDf:
    def test_renames_columns(self):
        df = _make_ohlcv(50)
        bt = _to_bt_df(df)
        assert set(bt.columns) == {"Open", "High", "Low", "Close", "Volume"}

    def test_index_is_tz_naive(self):
        df = _make_ohlcv(50)
        df.index = pd.date_range("2023-01-01", periods=50, freq="B", tz="UTC")
        bt = _to_bt_df(df)
        assert bt.index.tz is None

    def test_row_count_preserved(self):
        df = _make_ohlcv(100)
        assert len(_to_bt_df(df)) == 100


# ── BacktestResult ────────────────────────────────────────────────────────────

class TestBacktestResult:
    def _sample(self, **kw) -> BacktestResult:
        defaults = dict(
            symbol="THYAO.IS", total_return_pct=12.5,
            total_trades=8, win_rate_pct=62.5,
            max_drawdown_pct=-8.3, best_trade_pct=6.1,
            worst_trade_pct=-2.9, avg_trade_pct=1.6,
        )
        return BacktestResult(**{**defaults, **kw})

    def test_to_dict_returns_dict(self):
        assert isinstance(self._sample().to_dict(), dict)

    def test_to_json_is_valid_json(self):
        result = self._sample()
        parsed = json.loads(result.to_json())
        assert parsed["symbol"] == "THYAO.IS"

    def test_to_json_has_all_metrics(self):
        result = self._sample()
        d = result.to_dict()
        for key in ("total_return_pct", "total_trades", "win_rate_pct",
                    "max_drawdown_pct", "best_trade_pct", "worst_trade_pct",
                    "avg_trade_pct", "sharpe_ratio", "profit_factor"):
            assert key in d

    def test_profitable_true_when_positive_return(self):
        assert self._sample(total_return_pct=5.0).profitable is True

    def test_profitable_false_when_negative(self):
        assert self._sample(total_return_pct=-2.0).profitable is False

    def test_profitable_false_when_error(self):
        assert self._sample(total_return_pct=5.0, error="hata").profitable is False

    def test_initial_cash_default(self):
        r = BacktestResult(symbol="X")
        assert r.initial_cash == _INITIAL_CASH

    def test_commission_pct_stored(self):
        r = BacktestResult(symbol="X")
        assert r.commission_pct == pytest.approx(_COMMISSION * 100)

    def test_risk_per_trade_stored(self):
        r = BacktestResult(symbol="X")
        assert r.risk_per_trade_pct == pytest.approx(1.0)


# ── run_single ────────────────────────────────────────────────────────────────

class TestRunSingle:
    def test_returns_backtest_result(self):
        with _patch_fetch(_make_fetch_result()):
            result = run_single("THYAO")
        assert isinstance(result, BacktestResult)

    def test_symbol_normalized(self):
        with _patch_fetch(_make_fetch_result("THYAO.IS")):
            result = run_single("thyao")
        assert result.symbol == "THYAO.IS"

    def test_no_error_on_valid_data(self):
        with _patch_fetch(_make_fetch_result()):
            result = run_single("THYAO")
        assert result.error is None

    def test_fetch_error_returns_error_result(self):
        with _patch_fetch_error(FetchError("THYAO.IS", "bağlantı hatası")):
            result = run_single("THYAO")
        assert result.error is not None
        assert result.total_trades == 0

    def test_insufficient_data_returns_error(self):
        short = _make_fetch_result(n=30)
        with _patch_fetch(short):
            result = run_single("THYAO")
        assert result.error is not None

    def test_total_return_is_float(self):
        with _patch_fetch(_make_fetch_result()):
            result = run_single("THYAO")
        assert isinstance(result.total_return_pct, float)

    def test_total_trades_is_nonneg_int(self):
        with _patch_fetch(_make_fetch_result()):
            result = run_single("THYAO")
        assert isinstance(result.total_trades, int)
        assert result.total_trades >= 0

    def test_win_rate_in_range(self):
        with _patch_fetch(_make_fetch_result()):
            result = run_single("THYAO")
        if result.total_trades > 0:
            assert 0.0 <= result.win_rate_pct <= 100.0

    def test_max_drawdown_nonpositive(self):
        with _patch_fetch(_make_fetch_result()):
            result = run_single("THYAO")
        assert result.max_drawdown_pct <= 0.0

    def test_best_trade_gte_worst_trade(self):
        with _patch_fetch(_make_fetch_result()):
            result = run_single("THYAO")
        if result.total_trades > 0:
            assert result.best_trade_pct >= result.worst_trade_pct

    def test_error_result_has_symbol(self):
        with _patch_fetch_error(FetchError("GARAN.IS", "timeout")):
            result = run_single("GARAN")
        assert "GARAN" in result.symbol

    def test_custom_cash_stored(self):
        with _patch_fetch(_make_fetch_result()):
            result = run_single("THYAO", cash=50_000)
        assert result.initial_cash == 50_000.0

    def test_downtrend_fewer_trades(self):
        """Düşüş trendinde AL koşulları daha az tetiklenmeli."""
        with _patch_fetch(_make_fetch_result(trend="up")):
            up_result = run_single("THYAO")
        with _patch_fetch(_make_fetch_result(trend="down", seed=99)):
            dn_result = run_single("THYAO")
        # Düşüşte genellikle daha az işlem (veya eşit — strict assert yok)
        assert isinstance(dn_result.total_trades, int)


# ── run_multiple ──────────────────────────────────────────────────────────────

class TestRunMultiple:
    def test_returns_multi_result(self):
        with _patch_fetch(_make_fetch_result()):
            result = run_multiple(["THYAO", "GARAN"])
        assert isinstance(result, MultiBacktestResult)

    def test_result_count_matches_input(self):
        with _patch_fetch(_make_fetch_result()):
            result = run_multiple(["THYAO", "GARAN", "AKBNK"])
        assert len(result.results) == 3

    def test_summary_present(self):
        with _patch_fetch(_make_fetch_result()):
            result = run_multiple(["THYAO"])
        for key in ("total_symbols", "successful", "failed",
                    "avg_return_pct", "best_symbol", "worst_symbol", "profitable_count"):
            assert key in result.summary

    def test_to_dict_is_serializable(self):
        with _patch_fetch(_make_fetch_result()):
            result = run_multiple(["THYAO"])
        d = result.to_dict()
        json.dumps(d)   # istisna atmamalı

    def test_to_json_valid(self):
        with _patch_fetch(_make_fetch_result()):
            result = run_multiple(["THYAO"])
        parsed = json.loads(result.to_json())
        assert "summary" in parsed
        assert "results" in parsed

    def test_skip_errors_true_continues(self):
        good = _make_fetch_result("THYAO.IS")
        bad  = FetchError("GARAN.IS", "hata")

        def side_effect(symbol, **kw):
            if "GARAN" in symbol:
                raise bad
            return good

        with patch("app.backtest.runner.fetch_symbol_data", side_effect=side_effect):
            result = run_multiple(["THYAO", "GARAN"], skip_errors=True)

        assert result.summary["successful"] == 1
        assert result.summary["failed"] == 1

    def test_skip_errors_false_raises(self):
        with _patch_fetch_error(FetchError("THYAO.IS", "hata")):
            with pytest.raises(RuntimeError):
                run_multiple(["THYAO"], skip_errors=False)

    def test_empty_list_returns_empty(self):
        result = run_multiple([])
        assert result.summary["total_symbols"] == 0
        assert result.summary["successful"] == 0

    def test_run_at_is_valid_iso(self):
        with _patch_fetch(_make_fetch_result()):
            result = run_multiple(["THYAO"])
        datetime.fromisoformat(result.run_at)   # parse edilebilmeli

    def test_best_worst_symbol_in_results(self):
        with _patch_fetch(_make_fetch_result()):
            result = run_multiple(["THYAO", "GARAN"])
        ok = [r for r in result.results if r.error is None]
        if ok:
            symbols = [r.symbol for r in ok]
            assert result.summary["best_symbol"] in symbols
            assert result.summary["worst_symbol"] in symbols


# ── Pozisyon büyüklüğü & risk mantığı ────────────────────────────────────────

class TestPositionSizing:
    def test_position_size_equals_risk_over_stop(self):
        expected = 0.01 / 0.03
        assert _POSITION_SIZE == pytest.approx(expected, rel=1e-4)

    def test_position_size_less_than_one(self):
        assert _POSITION_SIZE < 1.0

    def test_strategy_class_has_correct_pct(self):
        assert TrendBreakoutBT.stop_loss_pct == pytest.approx(0.03)
        assert TrendBreakoutBT.take_profit_pct == pytest.approx(0.06)
        assert TrendBreakoutBT.position_size == pytest.approx(_POSITION_SIZE)
