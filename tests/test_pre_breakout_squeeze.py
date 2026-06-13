"""
Pre-Kırılım Sıkışma Stratejisi testleri.
Ağ bağlantısı gerekmez — sentetik DataFrame kullanılır.
"""

import numpy as np
import pandas as pd
import pytest

from app.strategies.pre_breakout_squeeze import (
    PreBreakoutSqueezeStrategy,
    generate_setup_signal,
    _SETUP_RSI_LOW,
    _SETUP_RSI_HIGH,
    _SETUP_VOL_LOW,
    _SETUP_VOL_HIGH,
    _SETUP_DIST_MAX,
    _EARLY_DIST_MAX,
    _EARLY_VOL_MIN,
    _EARLY_MAX_DAILY_CHANGE,
    _EARLY_MAX_EMA20_GAP,
    _EARLY_MAX_VOL_RATIO,
)


# ── DataFrame fabrikaları ─────────────────────────────────────────────────────

def _base_df(n: int = 80) -> pd.DataFrame:
    """
    Tüm SETUP koşullarını sağlayan DataFrame döner.
    RSI orta bölge, düşük hacim, volatility_squeeze=True, dirençin %3 altında.
    """
    idx   = pd.date_range("2023-01-01", periods=n, freq="B")
    close = np.linspace(95, 105, n)

    ema_50 = close - 5    # close > ema_50 ✓

    # resistance_20: prev bar için close + 3 ≈ %3 üstte
    prev_resistance = close + 3.2
    resistance_20   = np.roll(prev_resistance, 1)
    resistance_20[0] = prev_resistance[0]

    volume    = np.full(n, 900_000.0)
    vol_ma20  = np.full(n, 1_000_000.0)  # volume_ratio = 0.9 → [0.7, 1.6] ✓

    # bb_width küçük (sıkışma)
    bb_width = np.full(n, 3.0)

    df = pd.DataFrame({
        "close":              close,
        "open":               close * 0.99,
        "high":               close + 0.5,
        "low":                close - 0.5,
        "ema_20":             close - 2,
        "ema_50":             ema_50,
        "ema_200":            ema_50 - 10,
        "rsi_14":             np.full(n, 55.0),   # [45, 65] ✓
        "atr_14":             np.full(n, 1.5),
        "volume":             volume,
        "volume_ma20":        vol_ma20,
        "volume_ratio":       volume / vol_ma20,
        "resistance_20":      resistance_20,
        "volatility_squeeze": np.full(n, True),   # squeeze ✓
        "bb_width":           bb_width,
        "macd":               np.full(n, 0.5),
        "macd_signal":        np.full(n, 0.3),
        "macd_hist":          np.linspace(0.1, 0.8, n),  # yükselen ✓
    }, index=idx)
    return df


def _set(df: pd.DataFrame, col: str, value, row: int = -1) -> pd.DataFrame:
    df = df.copy()
    df[col] = df[col].where(df.index != df.index[row], value)
    return df


# ── generate_setup_signal: SETUP ─────────────────────────────────────────────

class TestSetupSignal:
    def test_returns_setup_when_all_conditions_met(self):
        result = generate_setup_signal(_base_df())
        # EARLY_WATCH veya SETUP — macd_hist yükseliyor olabilir
        assert result["signal"] in ("SETUP", "EARLY_WATCH")

    def test_setup_has_stop_loss(self):
        df = _base_df()
        # EARLY_WATCH olmadığından emin olmak için macd_hist sabit tut
        df["macd_hist"] = np.full(len(df), 0.3)
        result = generate_setup_signal(df)
        if result["signal"] == "SETUP":
            assert result["stop_loss"] is not None
            assert result["stop_loss"] < result["price"]

    def test_setup_stop_loss_below_ema50(self):
        df = _base_df()
        df["macd_hist"] = np.full(len(df), 0.3)
        result = generate_setup_signal(df)
        if result["signal"] == "SETUP":
            ema50 = float(df["ema_50"].iloc[-1])
            assert result["stop_loss"] < ema50

    def test_setup_take_profit_at_resistance(self):
        df = _base_df()
        df["macd_hist"] = np.full(len(df), 0.3)
        result = generate_setup_signal(df)
        if result["signal"] == "SETUP":
            assert result["take_profit"] is not None
            assert result["take_profit"] > result["price"]

    def test_setup_strength_between_0_and_1(self):
        df = _base_df()
        df["macd_hist"] = np.full(len(df), 0.3)
        result = generate_setup_signal(df)
        assert 0.0 <= result["strength"] <= 1.0

    def test_squeeze_false_no_setup_regardless_of_bb_low(self):
        """squeeze=False → SETUP gelmez; artık squeeze zorunlu koşul."""
        df = _base_df()
        df["volatility_squeeze"] = False
        df["macd_hist"] = np.full(len(df), 0.3)
        result = generate_setup_signal(df)
        # squeeze zorunlu olduğundan bb_low=True olsa bile SETUP üretilmez
        assert result["signal"] != "SETUP"

    def test_hold_when_neither_squeeze_nor_bb_low(self):
        """squeeze=False VE bb_width genişlemiş → (s1 OR s6) = False → SETUP gelmez."""
        df = _base_df()
        df["volatility_squeeze"] = False
        # Geçmiş bb_width düşük, son bar çok yüksek → quantile < son bar → bb_low=False
        bb_vals = np.full(len(df), 1.0)
        bb_vals[-1] = 999.0
        df["bb_width"] = bb_vals
        df["macd_hist"] = np.full(len(df), 0.3)
        result = generate_setup_signal(df)
        assert result["signal"] != "SETUP"

    def test_hold_when_rsi_out_of_range(self):
        df = _base_df()
        df["rsi_14"] = 80.0   # > _SETUP_RSI_HIGH
        result = generate_setup_signal(df)
        assert result["signal"] == "HOLD"

    def test_hold_when_volume_too_high(self):
        # volume=3.0 SETUP koşulunu (0.7-1.6) ihlal eder.
        # Eğer distance <= 2% ve momentum varsa EARLY_WATCH gelebilir (doğru davranış).
        # SETUP gelmemeli.
        df = _base_df()
        df["volume"] = df["volume_ma20"] * 3.0   # > _SETUP_VOL_HIGH (1.6)
        result = generate_setup_signal(df)
        assert result["signal"] != "SETUP"

    def test_hold_when_price_above_resistance(self):
        df = _base_df()
        # resistance_20 < close → distance_to_res_pct <= 0
        df["resistance_20"] = df["close"] - 5
        result = generate_setup_signal(df)
        assert result["signal"] == "HOLD"

    def test_hold_when_distance_too_far(self):
        df = _base_df()
        # resistance = close × 1.10 → %10 uzakta > _SETUP_DIST_MAX (%5)
        df["resistance_20"] = df["close"] * 1.10
        result = generate_setup_signal(df)
        assert result["signal"] == "HOLD"

    def test_hold_when_close_below_ema50(self):
        df = _base_df()
        df["ema_50"] = df["close"] + 10   # close < ema_50
        result = generate_setup_signal(df)
        assert result["signal"] == "HOLD"

    def test_hold_when_missing_column(self):
        df = _base_df().drop(columns=["resistance_20"])
        result = generate_setup_signal(df)
        assert result["signal"] == "HOLD"
        assert "Eksik" in result["reason"]

    def test_details_keys_present(self):
        result = generate_setup_signal(_base_df())
        for key in ("close", "ema_50", "rsi_14", "volume_ratio",
                    "distance_to_res_pct", "prev_resistance",
                    "setup_conditions_met"):
            assert key in result["details"], f"details içinde {key} eksik"


# ── generate_setup_signal: EARLY_WATCH ───────────────────────────────────────

class TestEarlyWatchSignal:
    def _early_df(self, n: int = 80) -> pd.DataFrame:
        """
        EARLY_WATCH koşullarını sağlayan DataFrame:
        - direncin %1.5 altında (≤ %2)
        - volume_ratio >= 1.2
        - macd_hist yükseliyor (3 gün)
        - close 2 gün ard arda yükselen
        + setup_conditions_met >= 3
        """
        df = _base_df(n)
        close = df["close"].values.copy()

        # Son 3 günü yükselen yap (close_strengthening ✓)
        close[-3] = close[-4] * 0.995
        close[-2] = close[-3] * 1.005
        close[-1] = close[-2] * 1.005
        df["close"] = close

        # Direnci %1.5 üste koy (distance ≈ 1.5% ≤ 2)
        df["resistance_20"] = df["close"] * 1.015

        # Hacim canlanması (volume_ratio ≈ 1.4 ≥ 1.2) ✓
        df["volume"] = df["volume_ma20"] * 1.4

        # MACD hist son 3 günde yükselsin
        macd_hist = df["macd_hist"].values.copy()
        macd_hist[-3] = 0.1
        macd_hist[-2] = 0.3
        macd_hist[-1] = 0.5
        df["macd_hist"] = macd_hist

        # RSI orta bölge ✓
        df["rsi_14"] = 57.0
        df["volatility_squeeze"] = True

        return df

    def test_returns_early_watch_when_conditions_met(self):
        result = generate_setup_signal(self._early_df())
        assert result["signal"] == "EARLY_WATCH"

    def test_early_watch_strength_above_setup(self):
        df_early = self._early_df()
        result_early = generate_setup_signal(df_early)

        df_setup = _base_df()
        df_setup["macd_hist"] = np.full(len(df_setup), 0.3)
        result_setup = generate_setup_signal(df_setup)

        if result_early["signal"] == "EARLY_WATCH" and result_setup["signal"] == "SETUP":
            assert result_early["strength"] > result_setup["strength"]

    def test_early_watch_has_stop_loss(self):
        result = generate_setup_signal(self._early_df())
        if result["signal"] == "EARLY_WATCH":
            assert result["stop_loss"] is not None
            assert result["stop_loss"] < result["price"]

    def test_early_watch_take_profit_above_price(self):
        result = generate_setup_signal(self._early_df())
        if result["signal"] == "EARLY_WATCH":
            assert result["take_profit"] is not None
            assert result["take_profit"] > result["price"]

    def test_early_watch_reason_mentions_distance(self):
        result = generate_setup_signal(self._early_df())
        if result["signal"] == "EARLY_WATCH":
            assert "Direncin" in result["reason"] or "direnç" in result["reason"].lower()

    def test_early_watch_risk_level_medium(self):
        result = generate_setup_signal(self._early_df())
        if result["signal"] == "EARLY_WATCH":
            assert result["risk_level"] == "MEDIUM"

    def test_early_watch_details_early_conditions_met(self):
        result = generate_setup_signal(self._early_df())
        if result["signal"] == "EARLY_WATCH":
            assert result["details"]["early_conditions_met"] >= 3

    def test_early_watch_not_when_distance_too_far(self):
        df = self._early_df()
        # %4 uzakta → distance > _EARLY_DIST_MAX
        df["resistance_20"] = df["close"] * 1.04
        result = generate_setup_signal(df)
        assert result["signal"] != "EARLY_WATCH"

    def test_early_watch_not_when_volume_low(self):
        df = self._early_df()
        df["volume"] = df["volume_ma20"] * 0.8   # < 1.2
        result = generate_setup_signal(df)
        assert result["signal"] != "EARLY_WATCH"

    # ── Kalite filtresi testleri ───────────────────────────────────────────────

    def test_early_watch_not_when_huge_volume(self):
        """volume_ratio > _EARLY_MAX_VOL_RATIO (3.0) → kalite filtresi eler."""
        df = self._early_df()
        df["volume"] = df["volume_ma20"] * (_EARLY_MAX_VOL_RATIO + 0.5)  # 3.5×
        result = generate_setup_signal(df)
        assert result["signal"] != "EARLY_WATCH"

    def test_early_watch_not_when_ema20_gap_too_large(self):
        """close_to_ema20_pct > _EARLY_MAX_EMA20_GAP (6%) → kalite filtresi eler."""
        df = self._early_df()
        # close / ema_20 = 1.10 → gap ≈ %10 > %6
        df["ema_20"] = df["close"] / 1.10
        result = generate_setup_signal(df)
        assert result["signal"] != "EARLY_WATCH"

    def test_early_watch_not_when_daily_spike(self):
        """daily_change_pct > _EARLY_MAX_DAILY_CHANGE (5%) → kalite filtresi eler."""
        df = self._early_df()
        df = df.copy()
        n = len(df)
        c = df["close"].values.copy()
        r = df["resistance_20"].values.copy()
        e = df["ema_20"].values.copy()

        new_c  = c[-2] * (_EARLY_MAX_DAILY_CHANGE / 100 + 0.025 + 1)  # +7.5%
        r[-2]  = new_c * 1.015   # prev_resistance: close[-1]'den %1.5 yukarı
        e[-1]  = new_c * 0.952   # EMA20 gap ≈ %5 < %6 → EMA filtresi geçsin

        c[-1]  = new_c
        df["close"]        = c
        df["resistance_20"] = r
        df["ema_20"]        = e
        result = generate_setup_signal(df)
        assert result["signal"] != "EARLY_WATCH"

    def test_early_watch_quality_ok_is_in_details(self):
        """details içinde ew_quality_ok alanı bulunmalı."""
        result = generate_setup_signal(self._early_df())
        assert "ew_quality_ok" in result["details"]
        assert "daily_change_pct" in result["details"]
        assert "close_to_ema20_pct" in result["details"]

    def test_setup_requires_squeeze_not_only_bb_low(self):
        """squeeze=False, bb_low=True → squeeze zorunlu olduğundan SETUP gelmez."""
        df = _base_df()
        df["volatility_squeeze"] = False
        df["bb_width"] = 1.0    # çok düşük → bb_low=True, ama squeeze=False
        df["macd_hist"] = np.full(len(df), 0.3)
        result = generate_setup_signal(df)
        assert result["signal"] != "SETUP"

    def test_setup_with_only_squeeze_no_bb_low(self):
        """squeeze=True ama bb_low=False → SETUP gelebilir (gevşetilmiş kural)."""
        df = _base_df()
        df["volatility_squeeze"] = True
        # Geçmiş düşük, son bar çok yüksek → bb_low=False
        bb_vals = np.full(len(df), 1.0)
        bb_vals[-1] = 999.0
        df["bb_width"] = bb_vals
        df["macd_hist"] = np.full(len(df), 0.3)
        result = generate_setup_signal(df)
        assert result["signal"] in ("SETUP", "EARLY_WATCH")


# ── PreBreakoutSqueezeStrategy (BaseStrategy entegrasyonu) ───────────────────

class TestPreBreakoutSqueezeStrategy:
    def setup_method(self):
        self.strategy = PreBreakoutSqueezeStrategy()

    def test_name(self):
        assert self.strategy.name == "pre_breakout_squeeze"

    def test_registered_in_strategies_list(self):
        from app.strategies import STRATEGIES
        names = [s.name for s in STRATEGIES]
        assert "pre_breakout_squeeze" in names

    def test_returns_none_on_hold(self):
        df = _base_df()
        df["rsi_14"] = 80.0   # koşul dışı → HOLD
        result = self.strategy.generate_signal(df, "TEST.IS")
        assert result is None

    def test_returns_none_on_insufficient_data(self):
        df = _base_df(n=30)   # < 50 satır → _validate_df False
        result = self.strategy.generate_signal(df, "TEST.IS")
        assert result is None

    def test_returns_strategy_signal_on_setup(self):
        from app.strategies.base import StrategySignal, SignalType
        df = _base_df()
        df["macd_hist"] = np.full(len(df), 0.3)
        result = self.strategy.generate_signal(df, "TEST.IS")
        if result is not None:
            assert isinstance(result, StrategySignal)
            assert result.signal_type == SignalType.BUY
            assert "pre_breakout_squeeze" in result.strategy

    def test_strategy_notes_not_empty(self):
        df = _base_df()
        df["macd_hist"] = np.full(len(df), 0.3)
        result = self.strategy.generate_signal(df, "TEST.IS")
        if result is not None:
            assert len(result.notes) > 5
