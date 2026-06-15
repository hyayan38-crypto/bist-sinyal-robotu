"""
Grafik Yapı Analizi — Swing Dip/Tepe + Destek/Direnç Bazlı Stop/Hedef
=====================================================================
ATR seviyelerinin yanında, grafiğin *yapısına* dayalı stop-loss / take-profit
üretir. Saf pandas/numpy — yeni bağımlılık yok.

Kullanım (scanner içinde, indikatörler eklendikten sonra):

    sl, tp = structure_sl_tp(df, close, atr)   # None → ATR'ye düşülür

Kolon standardı: küçük harf (`open, high, low, close, volume`). `resistance_20`
varsa hedef hesabında kullanılır; yoksa yalnız swing tepelerine bakılır.
"""

from __future__ import annotations

import pandas as pd
from typing import Optional

# Bir bar, solundaki ve sağındaki `window` bardan daha düşük/yüksekse swing sayılır.
_SWING_WINDOW = 3
# Stop, swing dibinin biraz altına konur: min(0.25×ATR, dip×0.5%) tampon.
_BUFFER_ATR_MULT = 0.25
_BUFFER_PCT      = 0.005


def find_swing_points(
    df: pd.DataFrame, window: int = _SWING_WINDOW
) -> tuple[list[float], list[float]]:
    """
    Yerel ekstrem swing dip ve tepelerini döner: (swing_lows, swing_highs).

    Bir `low[i]`, [i-window, i+window] penceresindeki tüm diğer low'lardan küçük
    veya eşitse swing dibi; `high[i]` benzer şekilde büyük/eşitse swing tepesidir.
    Pencere kenarındaki (ilk/son `window`) barlar atlanır — iki yanı dolu değil.

    Fiyat sırasına göre (eski → yeni) döner.
    """
    n = len(df)
    if n < 2 * window + 1:
        return [], []

    lows  = df["low"].to_numpy(dtype="float64")
    highs = df["high"].to_numpy(dtype="float64")

    swing_lows:  list[float] = []
    swing_highs: list[float] = []

    for i in range(window, n - window):
        lo = lows[i]
        hi = highs[i]
        if lo != lo or hi != hi:  # NaN kontrolü
            continue
        left_low   = lows[i - window:i]
        right_low  = lows[i + 1:i + window + 1]
        left_high  = highs[i - window:i]
        right_high = highs[i + 1:i + window + 1]

        if (lo <= left_low).all() and (lo <= right_low).all():
            swing_lows.append(round(float(lo), 2))
        if (hi >= left_high).all() and (hi >= right_high).all():
            swing_highs.append(round(float(hi), 2))

    return swing_lows, swing_highs


def add_swing_columns(df: pd.DataFrame, window: int = _SWING_WINDOW) -> pd.DataFrame:
    """
    `swing_low` / `swing_high` kolonlarını ekler (swing noktasında fiyat, aksi NaN).
    Yalnızca grafik işaretlemesi için — ana indikatör akışından bağımsız çağrılır.
    Kopya döner, orijinali bozmaz.
    """
    out = df.copy()
    n = len(out)
    out["swing_low"]  = float("nan")
    out["swing_high"] = float("nan")
    if n < 2 * window + 1:
        return out

    lows  = out["low"].to_numpy(dtype="float64")
    highs = out["high"].to_numpy(dtype="float64")
    lo_col = out.columns.get_loc("swing_low")
    hi_col = out.columns.get_loc("swing_high")

    for i in range(window, n - window):
        lo, hi = lows[i], highs[i]
        if lo == lo and (lo <= lows[i - window:i]).all() and (lo <= lows[i + 1:i + window + 1]).all():
            out.iat[i, lo_col] = lo
        if hi == hi and (hi >= highs[i - window:i]).all() and (hi >= highs[i + 1:i + window + 1]).all():
            out.iat[i, hi_col] = hi
    return out


def structure_sl_tp(
    df: pd.DataFrame, close: float, atr: Optional[float] = None
) -> tuple[Optional[float], Optional[float]]:
    """
    Grafik yapısına dayalı (stop_loss, take_profit) döner.

    Stop  : girişin altındaki en yakın swing dibi − küçük tampon.
    Hedef : girişin üstündeki en yakın swing tepesi; yoksa `resistance_20` üstü.

    Yapı bulunamayan taraf için ilgili değer None döner — çağıran tarafta
    ATR seviyesine düşülür. Mantıksız sonuçlar (stop ≥ close, hedef ≤ close)
    None'a indirgenir.
    """
    if not close or close <= 0:
        return None, None

    swing_lows, swing_highs = find_swing_points(df)

    # ── Stop: girişin altındaki en yakın swing dibi ──────────────────────────
    sl: Optional[float] = None
    below = [lo for lo in swing_lows if lo < close]
    if below:
        nearest_low = max(below)  # girişe en yakın (en yüksek) dip
        if atr and atr > 0 and atr == atr:  # NaN değil
            buffer = min(_BUFFER_ATR_MULT * atr, nearest_low * _BUFFER_PCT)
        else:
            buffer = nearest_low * _BUFFER_PCT
        sl = round(nearest_low - buffer, 2)
        if sl >= close:
            sl = None

    # ── Hedef: girişin üstündeki en yakın swing tepesi / direnç ──────────────
    tp: Optional[float] = None
    above = [hi for hi in swing_highs if hi > close]
    resistance = df["resistance_20"].iloc[-1] if "resistance_20" in df.columns else None
    if resistance is not None and resistance == resistance and resistance > close:
        above.append(round(float(resistance), 2))
    if above:
        tp = round(min(above), 2)  # girişe en yakın (en düşük) tepe/direnç
        if tp <= close:
            tp = None

    return sl, tp
