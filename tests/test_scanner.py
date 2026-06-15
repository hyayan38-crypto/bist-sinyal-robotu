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
    _scan_symbol,
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

    def test_hold_returns_none_when_no_early_watch(self):
        # WATCH/SETUP kaldırıldı: pbs EARLY_WATCH üretmezse HOLD → None döner.
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
        pbs_hold = {"signal": "HOLD", "details": {}}
        with _patch_fetch(_make_fetch_result()):
            with _patch.object(scanner_mod, "tb_generate", return_value=hold_signal):
                with _patch.object(scanner_mod, "pbs_generate", return_value=pbs_hold):
                    result = _scan_symbol("THYAO", _favorable())
        assert result is None

    def test_early_watch_returned_from_pbs(self):
        # pbs EARLY_WATCH üretirse scanner bunu EARLY_WATCH olarak döndürür.
        from unittest.mock import patch as _patch
        import app.signals.scanner as scanner_mod

        hold_signal = {
            "signal": "HOLD", "price": 100.0, "reason": "test",
            "risk_level": "LOW", "strength": 0.0,
            "stop_loss": None, "take_profit": None, "details": {},
        }
        pbs_ew = {
            "signal": "EARLY_WATCH", "price": 100.0, "reason": "yaklaşıyor",
            "risk_level": "MEDIUM", "strength": 0.7,
            "stop_loss": 97.0, "take_profit": 105.0,
            "details": {"distance_to_res_pct": 2.0},
        }
        with _patch_fetch(_make_fetch_result()):
            with _patch.object(scanner_mod, "tb_generate", return_value=hold_signal):
                with _patch.object(scanner_mod, "pbs_generate", return_value=pbs_ew):
                    result = _scan_symbol("THYAO", _favorable())
        assert result is not None
        assert result.signal == "EARLY_WATCH"

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

    def test_buy_comes_before_late_breakout(self):
        # Görüntüleme sırası: EARLY_WATCH → BUY → LATE_BREAKOUT
        from unittest.mock import patch as _patch
        import app.signals.scanner as scanner_mod

        def fake_tb(df):
            close = float(df["close"].iloc[-1])
            if close < 150:
                return {"signal": "LATE_BREAKOUT", "price": 100.0, "reason": "geç",
                        "risk_level": "HIGH", "strength": 0.5,
                        "stop_loss": 97.0, "take_profit": 106.0, "details": {}}
            return {"signal": "BUY", "price": 200.0, "reason": "b", "risk_level": "LOW",
                    "strength": 0.8, "stop_loss": 194.0, "take_profit": 212.0, "details": {}}

        with _patch_fetch(_make_fetch_result()):
            with _patch_mf(_favorable()):
                with _patch.object(scanner_mod, "tb_generate", side_effect=fake_tb):
                    results = scan_market(["THYAO", "GARAN"])

        buy_indices  = [i for i, r in enumerate(results) if r.get("signal") == "BUY"]
        late_indices = [i for i, r in enumerate(results) if r.get("signal") == "LATE_BREAKOUT"]
        if buy_indices and late_indices:
            assert min(buy_indices) < min(late_indices)

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

    def test_include_watch_false_no_early_watch_results(self):
        from unittest.mock import patch as _patch
        import app.signals.scanner as scanner_mod

        hold = {
            "signal": "HOLD", "price": 100.0, "reason": "h", "risk_level": "LOW",
            "strength": 0.0, "stop_loss": None, "take_profit": None, "details": {},
        }
        pbs_ew = {
            "signal": "EARLY_WATCH", "price": 100.0, "reason": "yaklaşıyor",
            "risk_level": "MEDIUM", "strength": 0.7,
            "stop_loss": 97.0, "take_profit": 105.0,
            "details": {"distance_to_res_pct": 2.0},
        }
        with _patch_fetch(_make_fetch_result()):
            with _patch_mf(_favorable()):
                with _patch.object(scanner_mod, "tb_generate", return_value=hold):
                    with _patch.object(scanner_mod, "pbs_generate", return_value=pbs_ew):
                        results = scan_market(["THYAO"], include_watch=False)
        assert all(r["signal"] not in ("EARLY_WATCH", "LATE_BREAKOUT") for r in results)

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
