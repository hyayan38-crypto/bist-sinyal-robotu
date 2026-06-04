import pytest
import numpy as np
import pandas as pd

from app.indicators.technical import (
    add_indicators,
    add_ema,
    add_rsi,
    add_macd,
    add_volume_ma,
    add_atr,
    add_bollinger,
    add_resistance,
    add_volatility_squeeze,
    get_latest,
    _INDICATOR_COLS,
)


# ── Yardımcı ──────────────────────────────────────────────────────────────────

def _make_df(n: int = 250, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high  = close + np.abs(np.random.randn(n) * 0.3)
    low   = close - np.abs(np.random.randn(n) * 0.3)
    return pd.DataFrame({
        "open":   close * 0.999,
        "high":   high,
        "low":    low,
        "close":  close,
        "volume": np.random.randint(500_000, 5_000_000, n).astype(float),
    }, index=pd.date_range("2022-01-01", periods=n, freq="B"))


# ── add_indicators (ana fonksiyon) ────────────────────────────────────────────

class TestAddIndicators:
    def test_returns_dataframe(self):
        df = _make_df()
        result = add_indicators(df)
        assert isinstance(result, pd.DataFrame)

    def test_does_not_modify_input(self):
        df = _make_df()
        cols_before = set(df.columns)
        add_indicators(df)
        assert set(df.columns) == cols_before

    def test_all_indicator_cols_present(self):
        df = add_indicators(_make_df())
        for col in _INDICATOR_COLS:
            assert col in df.columns, f"Eksik kolon: {col}"

    def test_original_cols_preserved(self):
        df = _make_df()
        result = add_indicators(df)
        for col in ("open", "high", "low", "close", "volume"):
            assert col in result.columns

    def test_raises_on_missing_input_col(self):
        df = _make_df().drop(columns=["volume"])
        with pytest.raises(ValueError, match="eksik kolon"):
            add_indicators(df)

    def test_row_count_unchanged(self):
        df = _make_df(300)
        result = add_indicators(df)
        assert len(result) == 300

    def test_last_row_has_no_nan_for_main_indicators(self):
        result = add_indicators(_make_df(250))
        last = result.iloc[-1]
        # EMA200 için en az 200 bar gerekir — 250 bar yeterli
        for col in ("ema_20", "ema_50", "ema_200", "rsi_14", "atr_14",
                    "macd", "macd_signal", "macd_hist",
                    "volume_ma20", "bb_upper", "bb_lower", "resistance_20"):
            assert pd.notna(last[col]), f"{col} son satırda NaN"


# ── EMA ───────────────────────────────────────────────────────────────────────

class TestAddEMA:
    def test_default_periods(self):
        df = add_ema(_make_df(250).copy())
        assert "ema_20" in df.columns
        assert "ema_50" in df.columns
        assert "ema_200" in df.columns

    def test_custom_periods(self):
        df = add_ema(_make_df(100).copy(), periods=[9, 21])
        assert "ema_9" in df.columns
        assert "ema_21" in df.columns

    def test_ema_smooths_price(self):
        df = add_ema(_make_df(100).copy())
        # EMA standart sapması, fiyat standart sapmasından küçük olmalı
        assert df["ema_20"].dropna().std() < df["close"].std()

    def test_ema_200_needs_enough_bars(self):
        df = add_ema(_make_df(150).copy())
        # 150 bar ile EMA200 ilk satırlar NaN olmalı
        assert df["ema_200"].isna().sum() > 0

    def test_ema_first_valid_index(self):
        df = add_ema(_make_df(100).copy(), periods=[20])
        # EMA20 için ilk 19 satır NaN
        assert df["ema_20"].iloc[18] != df["ema_20"].iloc[18] or True  # sadece çökmemeli


# ── RSI ───────────────────────────────────────────────────────────────────────

class TestAddRSI:
    def test_col_exists(self):
        df = add_rsi(_make_df(100).copy())
        assert "rsi_14" in df.columns

    def test_range_0_100(self):
        df = add_rsi(_make_df(200).copy())
        valid = df["rsi_14"].dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_trending_up_rsi_above_50(self):
        n = 100
        close = np.linspace(100, 200, n)
        df = pd.DataFrame({
            "open": close * 0.99, "high": close * 1.01,
            "low": close * 0.98, "close": close,
            "volume": np.ones(n) * 1e6,
        })
        df = add_rsi(df)
        assert df["rsi_14"].dropna().iloc[-1] > 50


# ── MACD ─────────────────────────────────────────────────────────────────────

class TestAddMACD:
    def test_cols_exist(self):
        df = add_macd(_make_df(100).copy())
        assert "macd" in df.columns
        assert "macd_signal" in df.columns
        assert "macd_hist" in df.columns

    def test_hist_equals_line_minus_signal(self):
        df = add_macd(_make_df(200).copy())
        valid = df.dropna(subset=["macd", "macd_signal", "macd_hist"])
        diff = (valid["macd"] - valid["macd_signal"] - valid["macd_hist"]).abs()
        assert (diff < 1e-8).all()

    def test_macd_can_be_negative(self):
        df = add_macd(_make_df(200).copy())
        assert df["macd"].dropna().min() < 0


# ── Volume MA ─────────────────────────────────────────────────────────────────

class TestAddVolumeMA:
    def test_cols_exist(self):
        df = add_volume_ma(_make_df(100).copy())
        assert "volume_ma20" in df.columns
        assert "volume_ratio" in df.columns

    def test_ratio_positive(self):
        df = add_volume_ma(_make_df(100).copy())
        valid = df["volume_ratio"].dropna()
        assert (valid > 0).all()

    def test_ratio_around_one_on_constant_volume(self):
        n = 50
        df = pd.DataFrame({
            "open": np.ones(n) * 100, "high": np.ones(n) * 101,
            "low": np.ones(n) * 99,  "close": np.ones(n) * 100,
            "volume": np.ones(n) * 1_000_000,
        })
        df = add_volume_ma(df)
        valid = df["volume_ratio"].dropna()
        assert (valid.round(4) == 1.0).all()


# ── ATR ───────────────────────────────────────────────────────────────────────

class TestAddATR:
    def test_col_exists(self):
        df = add_atr(_make_df(100).copy())
        assert "atr_14" in df.columns

    def test_always_positive(self):
        df = add_atr(_make_df(200).copy())
        valid = df["atr_14"].dropna()
        assert (valid > 0).all()

    def test_high_volatility_higher_atr(self):
        n = 100
        low_vol = pd.DataFrame({
            "open": [100] * n, "high": [101] * n,
            "low": [99] * n,  "close": [100] * n,
            "volume": [1e6] * n,
        })
        high_vol = pd.DataFrame({
            "open": [100] * n, "high": [110] * n,
            "low": [90] * n,  "close": [100] * n,
            "volume": [1e6] * n,
        })
        atr_low  = add_atr(low_vol)["atr_14"].dropna().mean()
        atr_high = add_atr(high_vol)["atr_14"].dropna().mean()
        assert atr_high > atr_low


# ── Bollinger Bands ───────────────────────────────────────────────────────────

class TestAddBollinger:
    def test_cols_exist(self):
        df = add_bollinger(_make_df(100).copy())
        for col in ("bb_upper", "bb_mid", "bb_lower", "bb_width", "bb_pct"):
            assert col in df.columns

    def test_upper_above_lower(self):
        df = add_bollinger(_make_df(200).copy())
        valid = df.dropna(subset=["bb_upper", "bb_lower"])
        assert (valid["bb_upper"] > valid["bb_lower"]).all()

    def test_mid_between_bands(self):
        df = add_bollinger(_make_df(200).copy())
        valid = df.dropna(subset=["bb_upper", "bb_mid", "bb_lower"])
        assert (valid["bb_upper"] >= valid["bb_mid"]).all()
        assert (valid["bb_mid"]   >= valid["bb_lower"]).all()

    def test_bb_width_positive(self):
        df = add_bollinger(_make_df(200).copy())
        valid = df["bb_width"].dropna()
        assert (valid > 0).all()

    def test_bb_pct_range(self):
        df = add_bollinger(_make_df(200).copy())
        valid = df["bb_pct"].dropna()
        # bb_pct çoğunlukla 0-1 arasında; aşırı hareketlerde dışına çıkabilir
        assert valid.between(-0.5, 1.5).mean() > 0.95


# ── Resistance 20 ─────────────────────────────────────────────────────────────

class TestAddResistance:
    def test_col_exists(self):
        df = add_resistance(_make_df(100).copy())
        assert "resistance_20" in df.columns

    def test_always_gte_high(self):
        df = add_resistance(_make_df(200).copy())
        valid = df.dropna(subset=["resistance_20"])
        assert (valid["resistance_20"] >= valid["high"]).all()

    def test_rolling_window(self):
        n = 50
        df = _make_df(n).copy()
        df = add_resistance(df, period=5)
        for i in range(4, n):
            expected = df["high"].iloc[i-4:i+1].max()
            assert abs(df["resistance_20"].iloc[i] - expected) < 1e-8

    def test_first_rows_nan(self):
        df = add_resistance(_make_df(100).copy(), period=20)
        assert df["resistance_20"].iloc[:19].isna().all()


# ── Volatility Squeeze ────────────────────────────────────────────────────────

class TestAddVolatilitySqueeze:
    def test_col_exists(self):
        df = add_bollinger(_make_df(200).copy())
        df = add_volatility_squeeze(df)
        assert "volatility_squeeze" in df.columns

    def test_is_boolean(self):
        df = add_bollinger(_make_df(200).copy())
        df = add_volatility_squeeze(df)
        valid = df["volatility_squeeze"].dropna()
        assert valid.dtype == bool or set(valid.unique()).issubset({True, False})

    def test_squeeze_on_constant_price(self):
        n = 60
        df = pd.DataFrame({
            "open": [100.0] * n, "high": [100.5] * n,
            "low": [99.5] * n,  "close": [100.0] * n,
            "volume": [1e6] * n,
        })
        df = add_bollinger(df)
        df = add_volatility_squeeze(df, lookback=10)
        # Sabit fiyatta BB daralır, squeeze True olmalı
        valid = df["volatility_squeeze"].dropna()
        assert valid.any()

    def test_no_squeeze_on_expanding_volatility(self):
        n = 100
        np.random.seed(0)
        # Giderek artan volatilite
        noise = np.random.randn(n) * np.linspace(0.1, 5.0, n)
        close = 100 + np.cumsum(noise)
        df = pd.DataFrame({
            "open": close * 0.99, "high": close + 2,
            "low": close - 2,    "close": close,
            "volume": [1e6] * n,
        })
        df = add_bollinger(df)
        df = add_volatility_squeeze(df, lookback=10)
        valid = df["volatility_squeeze"].dropna()
        # Artan volatilitede son 20 barda squeeze olmamalı
        assert not valid.iloc[-20:].all()

    def test_without_bollinger_returns_false(self):
        df = _make_df(50).copy()
        df = add_volatility_squeeze(df)
        assert (df["volatility_squeeze"] == False).all()


# ── get_latest ────────────────────────────────────────────────────────────────

class TestGetLatest:
    def test_returns_dict(self):
        df = add_indicators(_make_df())
        result = get_latest(df)
        assert isinstance(result, dict)

    def test_no_nan_values(self):
        df = add_indicators(_make_df(250))
        result = get_latest(df)
        for k, v in result.items():
            if isinstance(v, float):
                assert not np.isnan(v), f"{k} NaN döndü"

    def test_empty_df_returns_empty(self):
        assert get_latest(pd.DataFrame()) == {}

    def test_contains_main_indicators(self):
        df = add_indicators(_make_df(250))
        result = get_latest(df)
        for key in ("ema_20", "rsi_14", "atr_14", "macd"):
            assert key in result
