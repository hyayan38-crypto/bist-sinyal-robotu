"""
Yapı analizi (swing + destek/direnç bazlı stop/hedef) testleri.
"""

import numpy as np
import pandas as pd

from app.indicators.structure import (
    find_swing_points,
    structure_sl_tp,
    add_swing_columns,
)


def _df_from_close(close: list[float]) -> pd.DataFrame:
    close_arr = np.array(close, dtype="float64")
    idx = pd.date_range("2024-01-01", periods=len(close_arr), freq="D")
    return pd.DataFrame(
        {
            "open":  close_arr,
            "high":  close_arr + 0.3,
            "low":   close_arr - 0.3,
            "close": close_arr,
            "volume": 1_000_000.0,
        },
        index=idx,
    )


class TestFindSwingPoints:
    def test_detects_clear_v_bottom(self):
        # Belirgin V dibi: ...50 48 46 44 46 48 50... (window=3 ile 44 swing dibi)
        close = [50, 49, 48, 47, 46, 45, 44, 45, 46, 47, 48, 49, 50, 51, 52]
        df = _df_from_close(close)
        lows, highs = find_swing_points(df, window=3)
        # 44 civarı bir swing dibi yakalanmalı (low = 44-0.3 = 43.7)
        assert any(abs(lo - 43.7) < 0.01 for lo in lows)

    def test_too_short_returns_empty(self):
        df = _df_from_close([10, 11, 12])
        lows, highs = find_swing_points(df, window=3)
        assert lows == [] and highs == []

    def test_swing_columns_shape(self):
        close = [50, 48, 46, 44, 46, 48, 50, 48, 46, 48, 50, 52, 54]
        df = add_swing_columns(_df_from_close(close), window=3)
        assert "swing_low" in df.columns and "swing_high" in df.columns
        assert len(df) == len(close)


class TestStructureSLTP:
    def test_stop_below_target_above(self):
        # Yukarı trend + ara dipler; son fiyat tepeye yakın.
        close = [40, 42, 41, 44, 43, 46, 45, 48, 47, 50, 49, 52, 51, 54, 55]
        df = _df_from_close(close)
        df["resistance_20"] = df["high"].rolling(10, min_periods=1).max()
        price = float(df["close"].iloc[-1])
        sl, tp = structure_sl_tp(df, price, atr=0.5)
        if sl is not None:
            assert sl < price
        if tp is not None:
            assert tp > price

    def test_degenerate_returns_none(self):
        df = _df_from_close([10, 11, 12])
        sl, tp = structure_sl_tp(df, 11.5, atr=0.2)
        assert sl is None and tp is None

    def test_zero_close_returns_none(self):
        df = _df_from_close([10, 12, 11, 13, 12, 14, 13])
        assert structure_sl_tp(df, 0.0, atr=0.2) == (None, None)

    def test_stop_uses_buffer_below_swing_low(self):
        # Net swing dibi 44 (low 43.7); stop bunun altında olmalı.
        close = [50, 48, 46, 44, 46, 48, 50, 52, 53]
        df = _df_from_close(close)
        df["resistance_20"] = df["high"].rolling(5, min_periods=1).max()
        price = float(df["close"].iloc[-1])  # 53
        sl, _ = structure_sl_tp(df, price, atr=0.5)
        if sl is not None:
            assert sl < 43.7  # swing low - tampon
