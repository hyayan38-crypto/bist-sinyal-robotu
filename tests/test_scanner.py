"""
Piyasa tarayıcı testleri — ağ bağlantısı gerekmez.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from app.signals.scanner import (
    scan_market,
    ScanResult,
    _assess_watch,
    _watch_reason,
    _scan_symbol,
    _WATCH_MIN_CONDITIONS,
    _WATCH_RESISTANCE_PCT,
)
from app.data.fetcher import FetchResult, FetchError
from app.risk.market_filter import (
    MarketFilterResult,
    STATUS_FAVORABLE,
    STATUS_UNFAVORABLE,
    STATUS_UNAVAILABLE,
)


# ── Yardımcı ──────────────────────────────────────────────────────────────────

def _make_ohlcv(n: int = 250, trend: str = "up", seed: int = 0) -> pd.DataFrame:
    np.random.seed(seed)
    if trend == "up":
        close = np.linspace(80, 120, n) + np.random.randn(n) * 0.5
    else:
        close = np.linspace(120, 80, n) + np.random.randn(n) * 0.5

    high = close + np.abs(np.random.randn(n)) * 0.8
    low  = close - np.abs(np.random.randn(n)) * 0.8
    vol  = np.random.randint(500_000, 2_000_000, n).astype(float)
    return pd.DataFrame(
        {"open": close * 0.99, "high": high, "low": low, "close": close, "volume": vol},
        index=pd.date_range("2023-01-01", periods=n, freq="B"),
    )


def _make_fetch_result(symbol="THYAO.IS", **kw) -> FetchResult:
    return FetchResult(symbol=symbol, df=_make_ohlcv(**kw), period="1y", interval="1d")


def _favorable() -> MarketFilterResult:
    return MarketFilterResult(favorable=True, status=STATUS_FAVORABLE, reason="test")


def _unfavorable() -> MarketFilterResult:
    return MarketFilterResult(favorable=False, status=STATUS_UNFAVORABLE, reason="test")


def _unavailable() -> MarketFilterResult:
    return MarketFilterResult(favorable=True, status=STATUS_UNAVAILABLE, reason="test")


def _patch_fetch(result):
    return patch("app.signals.scanner.fetch_symbol_data", return_value=result)


def _patch_fetch_error(exc):
    return patch("app.signals.scanner.fetch_symbol_data", side_effect=exc)


def _patch_mf(result: MarketFilterResult):
    return patch("app.signals.scanner.is_market_favorable", return_value=result)


# ── _assess_watch ─────────────────────────────────────────────────────────────

class TestAssessWatch:
    def _details(self, **overrides) -> dict:
        base = {
            "c1_above_ema20": True,
            "c2_ema_uptrend": True,
            "c3_breakout": False,
            "c4_volume_surge": True,
            "c5_rsi_range": True,
            "close": 100.0,
            "prev_resistance": 102.0,   # %2 uzakta
            "rsi_14": 58.0,
            "volume_ratio": 2.1,
        }
        return {**base, **overrides}

    def test_watch_when_trend_ok_and_enough_extras(self):
        is_w, n, _ = _assess_watch(self._details())
        assert is_w is True

    def test_not_watch_when_c1_false(self):
        is_w, _, _ = _assess_watch(self._details(c1_above_ema20=False))
        assert is_w is False

    def test_not_watch_when_c2_false(self):
        is_w, _, _ = _assess_watch(self._details(c2_ema_uptrend=False))
        assert is_w is False

    def test_conditions_count_correct(self):
        # c1+c2 True, c4+c5 True → 4 total
        _, n, _ = _assess_watch(self._details(c3_breakout=False))
        assert n == 4

    def test_distance_calculated(self):
        details = self._details(close=100.0, prev_resistance=102.0)
        _, _, dist = _assess_watch(details)
        assert dist == pytest.approx(2.0, abs=0.1)

    def test_distance_none_when_zero_close(self):
        _, _, dist = _assess_watch(self._details(close=0.0))
        assert dist is None

    def test_approaching_resistance_triggers_watch_with_one_extra(self):
        # Sadece c4 True, ama dirence %1.5 uzakta → watch
        details = self._details(
            c3_breakout=False, c5_rsi_range=False,
            close=100.0, prev_resistance=101.5,
        )
        is_w, _, _ = _assess_watch(details)
        assert is_w is True

    def test_no_watch_when_all_extras_false(self):
        details = self._details(c3_breakout=False, c4_volume_surge=False, c5_rsi_range=False)
        is_w, _, _ = _assess_watch(details)
        assert is_w is False


# ── ScanResult ────────────────────────────────────────────────────────────────

class TestScanResult:
    def _sample(self) -> ScanResult:
        return ScanResult(
            symbol="THYAO.IS", signal="BUY", price=123.45,
            reason="test reason", risk_level="MEDIUM", strength=0.72,
            strategy="trend_breakout", stop_loss=119.75, take_profit=130.85,
            market_filter=STATUS_FAVORABLE, conditions_met=5, distance_to_res_pct=0.0,
        )

    def test_to_dict_returns_dict(self):
        assert isinstance(self._sample().to_dict(), dict)

    def test_to_dict_has_required_keys(self):
        d = self._sample().to_dict()
        for key in ("symbol", "signal", "price", "reason", "risk_level"):
            assert key in d

    def test_to_dict_signal_value(self):
        assert self._sample().to_dict()["signal"] == "BUY"

    def test_scanned_at_is_iso_string(self):
        from datetime import datetime
        r = self._sample()
        datetime.fromisoformat(r.scanned_at)


# ── _scan_symbol ──────────────────────────────────────────────────────────────

class TestScanSymbol:
    def test_returns_none_on_fetch_error(self):
        with _patch_fetch_error(FetchError("THYAO.IS", "hata")):
            result = _scan_symbol("THYAO", _favorable())
        assert result is None

    def test_returns_scan_result_or_none(self):
        with _patch_fetch(_make_fetch_result()):
            result = _scan_symbol("THYAO", _favorable())
        assert result is None or isinstance(result, ScanResult)

    def test_buy_blocked_when_market_unfavorable(self):
        # Uptrend verisiyle BUY olabilir ama market unfavorable
        from unittest.mock import patch as _patch
        import app.signals.scanner as scanner_mod

        buy_signal = {
            "signal": "BUY", "price": 100.0, "reason": "test",
            "risk_level": "LOW", "strength": 0.8,
            "stop_loss": 97.0, "take_profit": 106.0,
            "details": {"c1_above_ema20": True, "c2_ema_uptrend": True},
        }
        with _patch_fetch(_make_fetch_result()):
            with _patch.object(scanner_mod, "tb_generate", return_value=buy_signal):
                result = _scan_symbol("THYAO", _unfavorable())
        assert result is None

    def test_buy_allowed_when_market_favorable(self):
        from unittest.mock import patch as _patch
        import app.signals.scanner as scanner_mod

        buy_signal = {
            "signal": "BUY", "price": 100.0, "reason": "test",
            "risk_level": "LOW", "strength": 0.8,
            "stop_loss": 97.0, "take_profit": 106.0,
            "details": {},
        }
        with _patch_fetch(_make_fetch_result()):
            with _patch.object(scanner_mod, "tb_generate", return_value=buy_signal):
                result = _scan_symbol("THYAO", _favorable())
        assert result is not None
        assert result.signal == "BUY"

    def test_watch_returned_when_conditions_met(self):
        from unittest.mock import patch as _patch
        import app.signals.scanner as scanner_mod

        hold_signal = {
            "signal": "HOLD", "price": 100.0, "reason": "test",
            "risk_level": "LOW", "strength": 0.0,
            "stop_loss": None, "take_profit": None,
            "details": {
                "c1_above_ema20": True, "c2_ema_uptrend": True,
                "c3_breakout": False, "c4_volume_surge": True, "c5_rsi_range": True,
                "close": 100.0, "prev_resistance": 101.5, "rsi_14": 60.0, "volume_ratio": 2.0,
            },
        }
        with _patch_fetch(_make_fetch_result()):
            with _patch.object(scanner_mod, "tb_generate", return_value=hold_signal):
                result = _scan_symbol("THYAO", _favorable())
        assert result is not None
        assert result.signal == "WATCH"

    def test_sell_returns_none(self):
        from unittest.mock import patch as _patch
        import app.signals.scanner as scanner_mod

        sell_signal = {
            "signal": "SELL", "price": 100.0, "reason": "exit",
            "risk_level": "LOW", "strength": 0.0,
            "stop_loss": None, "take_profit": None, "details": {},
        }
        with _patch_fetch(_make_fetch_result()):
            with _patch.object(scanner_mod, "tb_generate", return_value=sell_signal):
                result = _scan_symbol("THYAO", _favorable())
        assert result is None


# ── scan_market ───────────────────────────────────────────────────────────────

class TestScanMarket:
    def test_returns_list(self):
        with _patch_fetch(_make_fetch_result()):
            with _patch_mf(_favorable()):
                result = scan_market(["THYAO"])
        assert isinstance(result, list)

    def test_empty_symbols_returns_empty(self):
        result = scan_market([])
        assert result == []

    def test_each_item_is_dict(self):
        with _patch_fetch(_make_fetch_result()):
            with _patch_mf(_favorable()):
                results = scan_market(["THYAO", "GARAN"])
        for r in results:
            assert isinstance(r, dict)

    def test_required_keys_in_result(self):
        from unittest.mock import patch as _patch
        import app.signals.scanner as scanner_mod
        buy = {
            "signal": "BUY", "price": 100.0, "reason": "test",
            "risk_level": "MEDIUM", "strength": 0.75,
            "stop_loss": 97.0, "take_profit": 106.0, "details": {},
        }
        with _patch_fetch(_make_fetch_result()):
            with _patch_mf(_favorable()):
                with _patch.object(scanner_mod, "tb_generate", return_value=buy):
                    results = scan_market(["THYAO"])
        if results:
            for key in ("symbol", "signal", "price", "reason", "risk_level"):
                assert key in results[0]

    def test_buy_comes_before_watch(self):
        from unittest.mock import patch as _patch
        import app.signals.scanner as scanner_mod

        signals = {
            "THYAO.IS": {"signal": "WATCH", "price": 100.0, "reason": "w", "risk_level": "LOW",
                          "strength": 0.0, "stop_loss": None, "take_profit": None,
                          "details": {"c1_above_ema20": True, "c2_ema_uptrend": True,
                                      "c4_volume_surge": True, "c5_rsi_range": True,
                                      "close": 100.0, "prev_resistance": 101.0, "rsi_14": 60.0}},
            "GARAN.IS": {"signal": "BUY",   "price": 200.0, "reason": "b", "risk_level": "LOW",
                          "strength": 0.8, "stop_loss": 194.0, "take_profit": 212.0, "details": {}},
        }

        def fake_tb(df):
            close = float(df["close"].iloc[-1])
            sym = "THYAO.IS" if close < 150 else "GARAN.IS"
            # Return based on symbol guess from price range
            return signals["THYAO.IS"] if close < 150 else signals["GARAN.IS"]

        with _patch_fetch(_make_fetch_result()):
            with _patch_mf(_favorable()):
                with _patch.object(scanner_mod, "tb_generate", side_effect=fake_tb):
                    results = scan_market(["THYAO", "GARAN"])

        buy_indices   = [i for i, r in enumerate(results) if r.get("signal") == "BUY"]
        watch_indices = [i for i, r in enumerate(results) if r.get("signal") == "WATCH"]
        if buy_indices and watch_indices:
            assert min(buy_indices) < min(watch_indices)

    def test_fetch_error_skipped(self):
        def side_effect(symbol, **kw):
            if "THYAO" in symbol:
                raise FetchError(symbol, "hata")
            return _make_fetch_result("GARAN.IS")

        with patch("app.signals.scanner.fetch_symbol_data", side_effect=side_effect):
            with _patch_mf(_favorable()):
                results = scan_market(["THYAO", "GARAN"])
        # THYAO atlanmalı, hata fırlatmamalı
        assert isinstance(results, list)

    def test_include_watch_false_no_watch_results(self):
        from unittest.mock import patch as _patch
        import app.signals.scanner as scanner_mod

        hold = {
            "signal": "HOLD", "price": 100.0, "reason": "h", "risk_level": "LOW",
            "strength": 0.0, "stop_loss": None, "take_profit": None,
            "details": {"c1_above_ema20": True, "c2_ema_uptrend": True,
                         "c4_volume_surge": True, "c5_rsi_range": True,
                         "close": 100.0, "prev_resistance": 101.0, "rsi_14": 60.0},
        }
        with _patch_fetch(_make_fetch_result()):
            with _patch_mf(_favorable()):
                with _patch.object(scanner_mod, "tb_generate", return_value=hold):
                    results = scan_market(["THYAO"], include_watch=False)
        assert all(r["signal"] != "WATCH" for r in results)

    def test_apply_market_filter_false_bypasses_filter(self):
        from unittest.mock import patch as _patch
        import app.signals.scanner as scanner_mod

        buy = {
            "signal": "BUY", "price": 100.0, "reason": "test",
            "risk_level": "LOW", "strength": 0.8,
            "stop_loss": 97.0, "take_profit": 106.0, "details": {},
        }
        with _patch_fetch(_make_fetch_result()):
            with _patch.object(scanner_mod, "tb_generate", return_value=buy):
                results = scan_market(["THYAO"], apply_market_filter=False)
        # Endeks filtresi bypass → BUY geçmeli
        assert any(r["signal"] == "BUY" for r in results)

    def test_market_filter_status_in_result(self):
        from unittest.mock import patch as _patch
        import app.signals.scanner as scanner_mod

        buy = {
            "signal": "BUY", "price": 100.0, "reason": "test",
            "risk_level": "LOW", "strength": 0.8,
            "stop_loss": 97.0, "take_profit": 106.0, "details": {},
        }
        with _patch_fetch(_make_fetch_result()):
            with _patch_mf(_favorable()):
                with _patch.object(scanner_mod, "tb_generate", return_value=buy):
                    results = scan_market(["THYAO"])
        if results:
            assert "market_filter" in results[0]

    def test_results_sorted_by_strength_within_group(self):
        from unittest.mock import patch as _patch
        import app.signals.scanner as scanner_mod

        call_n = [0]
        def fake_buy(df):
            call_n[0] += 1
            strength = 0.9 if call_n[0] == 1 else 0.6
            return {
                "signal": "BUY", "price": 100.0, "reason": "test",
                "risk_level": "LOW", "strength": strength,
                "stop_loss": 97.0, "take_profit": 106.0, "details": {},
            }

        with _patch_fetch(_make_fetch_result()):
            with _patch_mf(_favorable()):
                with _patch.object(scanner_mod, "tb_generate", side_effect=fake_buy):
                    results = scan_market(["THYAO", "GARAN"])

        buy_results = [r for r in results if r["signal"] == "BUY"]
        if len(buy_results) >= 2:
            strengths = [r["strength"] for r in buy_results]
            assert strengths == sorted(strengths, reverse=True)
