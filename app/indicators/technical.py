"""
Teknik indikatör modülü.

Üretilen kolon adları
─────────────────────
Trend    : ema_20, ema_50, ema_200
Momentum : rsi_14, macd, macd_signal, macd_hist
Volatilite: atr_14, bb_upper, bb_mid, bb_lower, bb_width, bb_pct
Hacim    : volume_ma20, volume_ratio
Seviye   : resistance_20
Sıkışma  : volatility_squeeze (bool)

Giriş DataFrame: open, high, low, close, volume kolonları (küçük harf)
Çıkış           : Aynı DataFrame + yukarıdaki kolonlar (inplace DEĞİL, kopya)
"""

from __future__ import annotations

import pandas as pd
import pandas_ta_classic as ta
from loguru import logger


# ── Tek indikatör fonksiyonları ───────────────────────────────────────────────

def add_ema(df: pd.DataFrame, periods: list[int] = (20, 50, 200)) -> pd.DataFrame:
    """EMA kolonlarını ekler: ema_20, ema_50, ema_200."""
    for p in periods:
        df[f"ema_{p}"] = ta.ema(df["close"], length=p)
    return df


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """RSI kolonunu ekler: rsi_14."""
    df["rsi_14"] = ta.rsi(df["close"], length=period)
    return df


def add_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """MACD kolonlarını ekler: macd, macd_signal, macd_hist."""
    result = ta.macd(df["close"], fast=fast, slow=slow, signal=signal)
    if result is None or result.empty:
        df["macd"] = float("nan")
        df["macd_signal"] = float("nan")
        df["macd_hist"] = float("nan")
        return df

    col_line   = f"MACD_{fast}_{slow}_{signal}"
    col_signal = f"MACDs_{fast}_{slow}_{signal}"
    col_hist   = f"MACDh_{fast}_{slow}_{signal}"

    df["macd"]        = result.get(col_line,   float("nan"))
    df["macd_signal"] = result.get(col_signal, float("nan"))
    df["macd_hist"]   = result.get(col_hist,   float("nan"))
    return df


def add_volume_ma(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """
    Hacim hareketli ortalaması ekler: volume_ma20, volume_ratio.
    volume_ratio = güncel hacim / 20 günlük ortalama hacim.
    """
    df["volume_ma20"] = ta.sma(df["volume"], length=period)
    df["volume_ratio"] = df["volume"] / df["volume_ma20"]
    return df


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """ATR kolonunu ekler: atr_14."""
    df["atr_14"] = ta.atr(df["high"], df["low"], df["close"], length=period)
    return df


def add_bollinger(df: pd.DataFrame, period: int = 20, std: float = 2.0) -> pd.DataFrame:
    """
    Bollinger Bands ekler: bb_upper, bb_mid, bb_lower, bb_width, bb_pct.
    bb_width = (üst - alt) / orta * 100   →  genişlik yüzdesi
    bb_pct   = (kapanış - alt) / (üst - alt)  →  0–1 arası konum
    """
    result = ta.bbands(df["close"], length=period, std=std)
    if result is None or result.empty:
        for col in ("bb_upper", "bb_mid", "bb_lower", "bb_width", "bb_pct"):
            df[col] = float("nan")
        return df

    suffix = f"{period}_{std}"
    df["bb_upper"] = result.get(f"BBU_{suffix}", float("nan"))
    df["bb_mid"]   = result.get(f"BBM_{suffix}", float("nan"))
    df["bb_lower"] = result.get(f"BBL_{suffix}", float("nan"))
    df["bb_pct"]   = result.get(f"BBP_{suffix}", float("nan"))

    df["bb_width"] = (
        (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"] * 100
    ).where(df["bb_mid"] != 0)
    return df


def add_resistance(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """
    Son 20 günün en yüksek direncini ekler: resistance_20.
    Rolling maksimum high — mevcut bar dahil.
    """
    df["resistance_20"] = df["high"].rolling(window=period, min_periods=period).max()
    return df


def add_volatility_squeeze(df: pd.DataFrame, lookback: int = 10) -> pd.DataFrame:
    """
    Son 10 gün volatilite daralmasını tespit eder: volatility_squeeze (bool).

    Kural:
      bb_width, son `lookback` barın minimum bb_width değerinin %5 içindeyse
      sıkışma (squeeze) var demektir.

    Gereksinim: add_bollinger daha önce çalışmış olmalı (bb_width kolonu lazım).
    """
    if "bb_width" not in df.columns:
        logger.warning("add_bollinger önce çalıştırılmamış, volatility_squeeze atlandı")
        df["volatility_squeeze"] = False
        return df

    rolling_min = df["bb_width"].rolling(window=lookback, min_periods=lookback).min()
    df["volatility_squeeze"] = df["bb_width"] <= rolling_min * 1.05
    return df


# ── Ana fonksiyon ─────────────────────────────────────────────────────────────

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tüm indikatörleri DataFrame'e ekler ve kopya olarak döner.

    Hesaplama sırası önemlidir:
      1. EMA, RSI, MACD, Volume, ATR  — bağımsız
      2. Bollinger Bands               — bb_width için önce çalışmalı
      3. Resistance                    — bağımsız
      4. Volatility squeeze            — bb_width'e bağımlı
    """
    _validate_input(df)
    out = df.copy()

    out = add_ema(out)
    out = add_rsi(out)
    out = add_macd(out)
    out = add_volume_ma(out)
    out = add_atr(out)
    out = add_bollinger(out)
    out = add_resistance(out)
    out = add_volatility_squeeze(out)

    _log_coverage(out)
    return out


# ── Yardımcı ─────────────────────────────────────────────────────────────────

_REQUIRED_INPUT = {"open", "high", "low", "close", "volume"}

_INDICATOR_COLS = (
    "ema_20", "ema_50", "ema_200",
    "rsi_14",
    "macd", "macd_signal", "macd_hist",
    "volume_ma20", "volume_ratio",
    "atr_14",
    "bb_upper", "bb_mid", "bb_lower", "bb_width", "bb_pct",
    "resistance_20",
    "volatility_squeeze",
)


def _validate_input(df: pd.DataFrame) -> None:
    missing = _REQUIRED_INPUT - set(df.columns)
    if missing:
        raise ValueError(f"add_indicators: eksik kolon(lar): {sorted(missing)}")
    if len(df) < 30:
        logger.warning(f"DataFrame yalnızca {len(df)} satır içeriyor; bazı indikatörler NaN olabilir")


def _log_coverage(df: pd.DataFrame) -> None:
    last = df.iloc[-1]
    nan_cols = [c for c in _INDICATOR_COLS if c in df.columns and pd.isna(last[c])]
    if nan_cols:
        logger.debug(f"Son satırda NaN olan indikatörler: {nan_cols}")
    else:
        logger.debug(f"Tüm {len(_INDICATOR_COLS)} indikatör hesaplandı ({len(df)} bar)")


def get_latest(df: pd.DataFrame) -> dict:
    """Son satırın indikatör değerlerini {kolon: değer} olarak döner (NaN hariç)."""
    if df.empty:
        return {}
    last = df.iloc[-1]
    result = {}
    for col in _INDICATOR_COLS:
        if col not in df.columns:
            continue
        val = last[col]
        if isinstance(val, bool) or not pd.isna(val):
            result[col] = bool(val) if isinstance(val, (bool, pd.BooleanDtype)) else round(float(val), 4)
    return result


# ── Geriye dönük uyumluluk ────────────────────────────────────────────────────

class TechnicalIndicators:
    """Eski sınıf arayüzü — mevcut kod değişmeden çalışır."""

    def add_all(self, df: pd.DataFrame) -> pd.DataFrame:
        return add_indicators(df)

    def get_current_values(self, df: pd.DataFrame) -> dict:
        return get_latest(df)


indicators = TechnicalIndicators()
