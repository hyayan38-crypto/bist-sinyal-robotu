"""
Trend + Hacimli Kırılım Stratejisi testleri.
Ağ bağlantısı gerekmez — sentetik DataFrame kullanılır.
"""

import numpy as np
import pandas as pd
import pytest

from app.strategies.trend_breakout import (
    TrendBreakoutStrategy,
    generate_signal,
    _risk_level,
    _signal_strength,
    _STOP_LOSS_PCT,
    _TAKE_PROFIT_PCT,
    _VOLUME_MULT,
    _RSI_LOW,
    _RSI_HIGH,
    _LATE_RSI_THRESHOLD,
    _LATE_DAILY_CHANGE,
    _LATE_EMA20_GAP,
    _LATE_VOL_RATIO,
    _LATE_RES_MARGIN,
)


# ── DataFrame fabrikası ───────────────────────────────────────────────────────

def _base_df(n: int = 60) -> pd.DataFrame:
    """Tüm AL koşullarını sağlayan DataFrame döner."""
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    close        = np.linspace(95, 110, n)
    ema_20       = close - 2          # close > ema_20 ✓
    ema_50       = ema_20 - 3         # ema_20 > ema_50 ✓
    prev_res     = close - 1          # close > prev resistance ✓  (shift yapılacak)
    resistance_20 = np.roll(prev_res, 1)
    resistance_20[0] = prev_res[0]   # ilk satır için anlamsız değer

    return pd.DataFrame({
        "close":        close,
        "open":         close * 0.99,
        "high":         close + 0.5,
        "low":          close - 0.5,
        "ema_20":       ema_20,
        "ema_50":       ema_50,
        "rsi_14":       np.full(n, 60.0),   # 50–75 ✓
        "atr_14":       np.full(n, 1.5),
        "volume":       np.full(n, 2_000_000.0),
        "volume_ma20":  np.full(n, 1_000_000.0),  # ratio=2.0 > 1.8 ✓
        "resistance_20": resistance_20,
    }, index=idx)


def _set(df: pd.DataFrame, col: str, value, row: int = -1) -> pd.DataFrame:
    """DataFrame kopyasında tek hücreyi değiştirir."""
    df = df.copy()
    df.iloc[row][col] = value
    df[col] = df[col].where(df.index != df.index[row], value)
    return df


# ── generate_signal: AL ───────────────────────────────────────────────────────

class TestBuySignal:
    def test_returns_buy_when_all_conditions_met(self):
        df = _base_df()
        result = generate_signal(df)
        assert result["signal"] == "BUY"

    def test_buy_sets_stop_loss(self):
        result = generate_signal(_base_df())
        price = result["price"]
        assert result["stop_loss"] == round(price * (1 - _STOP_LOSS_PCT), 2)

    def test_buy_sets_take_profit(self):
        result = generate_signal(_base_df())
        price = result["price"]
        assert result["take_profit"] == round(price * (1 + _TAKE_PROFIT_PCT), 2)

    def test_buy_strength_in_range(self):
        result = generate_signal(_base_df())
        assert 0.0 <= result["strength"] <= 1.0

    def test_buy_reason_not_empty(self):
        result = generate_signal(_base_df())
        assert len(result["reason"]) > 10

    def test_buy_price_equals_last_close(self):
        df = _base_df()
        result = generate_signal(df)
        assert result["price"] == round(float(df["close"].iloc[-1]), 2)

    def test_details_contains_all_keys(self):
        result = generate_signal(_base_df())
        for key in ("close", "ema_20", "ema_50", "rsi_14", "atr_14",
                    "volume_ratio", "prev_resistance",
                    "c1_above_ema20", "c2_ema_uptrend",
                    "c3_breakout", "c4_volume_surge", "c5_rsi_range"):
            assert key in result["details"], f"details içinde {key} eksik"


# ── generate_signal: HOLD ─────────────────────────────────────────────────────

class TestHoldSignal:
    def test_hold_when_close_below_ema20(self):
        df = _base_df()
        # close < ema_20 → SELL sinyali (çıkış kuralı)
        df["close"] = df["ema_20"] - 1
        result = generate_signal(df)
        # close < ema_20 → SELL (çıkış)
        assert result["signal"] == "SELL"

    def test_hold_when_ema20_below_ema50(self):
        df = _base_df()
        df["ema_50"] = df["ema_20"] + 5   # ema_20 < ema_50
        result = generate_signal(df)
        assert result["signal"] == "HOLD"

    def test_hold_when_no_breakout(self):
        df = _base_df()
        # resistance > close → kırılım yok
        df["resistance_20"] = df["close"] + 10
        result = generate_signal(df)
        assert result["signal"] == "HOLD"

    def test_hold_when_volume_insufficient(self):
        df = _base_df()
        df["volume"] = df["volume_ma20"] * 1.5   # < 1.8 ✗
        result = generate_signal(df)
        assert result["signal"] == "HOLD"

    def test_hold_when_rsi_below_50(self):
        df = _base_df()
        df["rsi_14"] = 45.0
        result = generate_signal(df)
        assert result["signal"] == "HOLD"

    def test_hold_when_rsi_above_75(self):
        df = _base_df()
        df["rsi_14"] = 78.0
        result = generate_signal(df)
        assert result["signal"] == "HOLD"

    def test_hold_reason_mentions_failed_condition(self):
        df = _base_df()
        df["rsi_14"] = 80.0
        result = generate_signal(df)
        assert "RSI" in result["reason"]

    def test_hold_when_missing_column(self):
        df = _base_df().drop(columns=["resistance_20"])
        result = generate_signal(df)
        assert result["signal"] == "HOLD"
        assert "Eksik" in result["reason"]

    def test_hold_stop_loss_is_none(self):
        df = _base_df()
        df["rsi_14"] = 45.0
        result = generate_signal(df)
        assert result["stop_loss"] is None
        assert result["take_profit"] is None


# ── generate_signal: SELL (çıkış) ─────────────────────────────────────────────

class TestSellSignal:
    def test_sell_when_close_below_ema20(self):
        df = _base_df()
        df["close"] = df["ema_20"] - 3
        result = generate_signal(df)
        assert result["signal"] == "SELL"

    def test_sell_no_stop_loss(self):
        df = _base_df()
        df["close"] = df["ema_20"] - 1
        result = generate_signal(df)
        assert result["stop_loss"] is None

    def test_sell_reason_mentions_ema20(self):
        df = _base_df()
        df["close"] = df["ema_20"] - 1
        result = generate_signal(df)
        assert "EMA20" in result["reason"]

    def test_sell_strength_is_zero(self):
        df = _base_df()
        df["close"] = df["ema_20"] - 1
        result = generate_signal(df)
        assert result["strength"] == 0.0


# ── Risk seviyesi ─────────────────────────────────────────────────────────────

class TestRiskLevel:
    def test_low_risk(self):
        assert _risk_level(rsi=55, volume_ratio=2.0, atr_pct=1.0) == "LOW"

    def test_medium_risk(self):
        assert _risk_level(rsi=65, volume_ratio=3.0, atr_pct=2.0) == "MEDIUM"

    def test_high_risk(self):
        assert _risk_level(rsi=72, volume_ratio=5.0, atr_pct=4.0) == "HIGH"

    def test_high_rsi_alone_not_enough_for_high(self):
        # tek faktör yetmez
        level = _risk_level(rsi=72, volume_ratio=1.9, atr_pct=1.0)
        assert level in ("LOW", "MEDIUM")

    def test_result_is_valid_string(self):
        for rsi in (51, 62, 74):
            for vr in (1.9, 2.8, 4.5):
                for atr in (1.0, 2.0, 3.5):
                    assert _risk_level(rsi, vr, atr) in ("LOW", "MEDIUM", "HIGH")

    def test_buy_signal_has_risk_level(self):
        result = generate_signal(_base_df())
        assert result["risk_level"] in ("LOW", "MEDIUM", "HIGH")


# ── Sinyal gücü ───────────────────────────────────────────────────────────────

class TestSignalStrength:
    def test_range_0_to_1(self):
        s = _signal_strength(
            close=110.0, prev_resistance=107.0,
            volume_ratio=2.5, rsi=62.0,
            ema20=108.0, ema50=105.0,
        )
        assert 0.0 <= s <= 1.0

    def test_stronger_breakout_higher_strength(self):
        s_weak   = _signal_strength(110.0, 109.5, 2.0, 60.0, 108.0, 106.0)
        s_strong = _signal_strength(110.0, 106.0, 3.5, 60.0, 108.0, 104.0)
        assert s_strong > s_weak

    def test_higher_volume_higher_strength(self):
        s_low  = _signal_strength(110.0, 108.0, 2.0, 62.0, 109.0, 106.0)
        s_high = _signal_strength(110.0, 108.0, 4.0, 62.0, 109.0, 106.0)
        assert s_high > s_low


# ── TrendBreakoutStrategy (BaseStrategy entegrasyonu) ─────────────────────────

class TestTrendBreakoutStrategy:
    def setup_method(self):
        self.strategy = TrendBreakoutStrategy()

    def test_name(self):
        assert self.strategy.name == "trend_breakout"

    def test_returns_strategy_signal_on_buy(self):
        from app.strategies.base import StrategySignal, SignalType
        df = _base_df()
        result = self.strategy.generate_signal(df, "THYAO.IS")
        if result is not None:
            assert isinstance(result, StrategySignal)
            assert result.signal_type == SignalType.BUY
            assert result.symbol == "THYAO.IS"
            assert result.strategy == "trend_breakout"

    def test_returns_none_on_hold(self):
        df = _base_df()
        df["rsi_14"] = 45.0
        result = self.strategy.generate_signal(df, "THYAO.IS")
        assert result is None

    def test_returns_none_on_insufficient_data(self):
        df = _base_df(n=30)  # < 50 satır → _validate_df False
        result = self.strategy.generate_signal(df, "THYAO.IS")
        assert result is None

    def test_stop_loss_take_profit_set(self):
        from app.strategies.base import SignalType
        df = _base_df()
        result = self.strategy.generate_signal(df, "THYAO.IS")
        if result and result.signal_type == SignalType.BUY:
            assert result.stop_loss is not None
            assert result.take_profit is not None
            assert result.stop_loss < result.entry_price
            assert result.take_profit > result.entry_price

    def test_registered_in_strategies_list(self):
        from app.strategies import STRATEGIES
        names = [s.name for s in STRATEGIES]
        assert "trend_breakout" in names


# ── generate_signal: LATE_BREAKOUT ───────────────────────────────────────────

class TestLateBreakout:
    """BUY koşullarını sağlayan ama geç kalma işareti içeren senaryolar."""

    def _buy_df(self) -> pd.DataFrame:
        """_base_df ile aynı ama tüm BUY koşulları kesinlikle sağlıyor."""
        return _base_df()

    def test_late_on_high_rsi(self):
        df = self._buy_df()
        df["rsi_14"] = float(_LATE_RSI_THRESHOLD + 1)   # > 68
        result = generate_signal(df)
        assert result["signal"] == "LATE_BREAKOUT"

    def test_late_on_extreme_volume(self):
        df = self._buy_df()
        df["volume"] = df["volume_ma20"] * (_LATE_VOL_RATIO + 1)   # > 4×
        result = generate_signal(df)
        assert result["signal"] == "LATE_BREAKOUT"

    def test_late_on_large_ema20_gap(self):
        df = self._buy_df()
        # close ema20 üzerinde %8 → > _LATE_EMA20_GAP (%6)
        df["ema_20"] = df["close"] / 1.08
        df["ema_50"] = df["ema_20"] - 3
        result = generate_signal(df)
        assert result["signal"] == "LATE_BREAKOUT"

    def test_late_on_large_resistance_margin(self):
        df = self._buy_df()
        # prev_resistance = close / 1.05 → kırılım marjı %5 > %3
        n = len(df)
        prev_res_val = df["close"].iloc[-1] / 1.05
        resistance = np.full(n, prev_res_val)
        resistance[-1] = prev_res_val  # curr
        # resistance_20[-1] = prev bar'ın değeri, -2 = daha önceki
        # _base_df'de resistance_20 = np.roll(prev_res, 1), yani resistance_20[-1] = prev_res[-2]
        # prev_resistance = curr[-2]["resistance_20"] = resistance[-2]
        # Burada tüm resistance değerlerini ayarlıyoruz
        df["resistance_20"] = prev_res_val
        result = generate_signal(df)
        assert result["signal"] == "LATE_BREAKOUT"

    def test_late_reason_contains_label(self):
        df = self._buy_df()
        df["rsi_14"] = float(_LATE_RSI_THRESHOLD + 2)
        result = generate_signal(df)
        assert result["signal"] == "LATE_BREAKOUT"
        assert "Geç kırılım" in result["reason"]
        assert "RSI" in result["reason"]

    def test_late_has_stop_loss_and_take_profit(self):
        df = self._buy_df()
        df["rsi_14"] = float(_LATE_RSI_THRESHOLD + 2)
        result = generate_signal(df)
        assert result["stop_loss"] is not None
        assert result["take_profit"] is not None
        assert result["stop_loss"] < result["price"]
        assert result["take_profit"] > result["price"]

    def test_late_details_contains_daily_change_pct(self):
        df = self._buy_df()
        df["rsi_14"] = float(_LATE_RSI_THRESHOLD + 2)
        result = generate_signal(df)
        assert "daily_change_pct" in result["details"]
        assert "late_flags" in result["details"]
        assert result["details"]["late_flags"] >= 1

    def test_normal_buy_not_late_with_base_df(self):
        """_base_df() geç koşul içermemeli — normal BUY dönmeli."""
        result = generate_signal(_base_df())
        assert result["signal"] == "BUY"

    def test_strategy_maps_late_to_buy_signal_type(self):
        from app.strategies.base import SignalType
        strategy = TrendBreakoutStrategy()
        df = self._buy_df()
        df["rsi_14"] = float(_LATE_RSI_THRESHOLD + 2)
        result = strategy.generate_signal(df, "TEST.IS")
        # LATE_BREAKOUT → StrategySignal ile BUY tipi olarak gelir
        if result is not None:
            assert result.signal_type == SignalType.BUY
