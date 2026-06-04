"""
Pre-Kırılım Sıkışma Stratejisi
================================
Volatilite sıkışması (squeeze) ve dirençe yaklaşma kombinasyonunu tespit eder.

SETUP       → Squeeze aktif + fiyat direncin ≤%5 altında + düşük hacim + RSI orta bölge
EARLY_WATCH → Dirençe ≤%2 + hacim canlanıyor + MACD hist yükseliyor + fiyat güçleniyor
HOLD        → Koşullar henüz oluşmamış
"""

from __future__ import annotations

import pandas as pd
from typing import Optional, TypedDict

from app.strategies.base import BaseStrategy, StrategySignal, SignalType

# ── Gerekli kolonlar ──────────────────────────────────────────────────────────

_REQUIRED = [
    "close", "ema_50", "rsi_14",
    "volume", "volume_ma20",
    "resistance_20",
]

# ── SETUP eşikleri ────────────────────────────────────────────────────────────

_SETUP_RSI_LOW    = 45
_SETUP_RSI_HIGH   = 65
_SETUP_VOL_LOW    = 0.7
_SETUP_VOL_HIGH   = 1.6
_SETUP_DIST_MAX   = 5.0    # % direnç altında maksimum uzaklık
_SETUP_BB_PCTILE  = 0.40   # bb_width son 60 günün alt %40'ında

# ── EARLY_WATCH eşikleri ──────────────────────────────────────────────────────

_EARLY_DIST_MAX  = 2.0
_EARLY_VOL_MIN   = 1.2
_EARLY_MIN_FLAGS = 3       # 4 spesifik koşuldan en az kaçı sağlanmalı

# EARLY_WATCH kalite filtreleri — aşılırsa EARLY_WATCH üretilmez
_EARLY_MAX_DAILY_CHANGE = 5.0    # % kapanış bazlı günlük artış üst sınırı
_EARLY_MAX_EMA20_GAP    = 6.0    # % EMA20 üzerinde maksimum mesafe
_EARLY_MAX_VOL_RATIO    = 3.0    # hacim çarpanı üst sınırı (üstünde hareket başlamıştır)

# ── SL/TP çarpanları ──────────────────────────────────────────────────────────

_SETUP_SL_MULT  = 0.98    # EMA50 × 0.98
_SETUP_TP_MULT  = 1.015   # prev_resistance × 1.015
_EARLY_SL_MULT  = 0.97    # EMA50 × 0.97
_EARLY_TP_MULT  = 1.02    # prev_resistance × 1.02


# ── Çıktı tipi ────────────────────────────────────────────────────────────────

class SetupSignal(TypedDict):
    signal:      str            # "SETUP" | "EARLY_WATCH" | "HOLD"
    reason:      str
    price:       float
    risk_level:  str
    stop_loss:   Optional[float]
    take_profit: Optional[float]
    strength:    float          # 0.0 – 1.0
    details:     dict


# ── Ana fonksiyon ─────────────────────────────────────────────────────────────

def generate_setup_signal(df: pd.DataFrame) -> SetupSignal:
    """
    Pre-kırılım sıkışma sinyali üretir.

    Gerekli kolonlar: close, ema_50, rsi_14, volume, volume_ma20, resistance_20
    İsteğe bağlı:     volatility_squeeze, bb_width, macd_hist
    """
    missing = [c for c in _REQUIRED if c not in df.columns]
    if missing:
        price = float(df["close"].iloc[-1]) if "close" in df.columns else 0.0
        return _hold(f"Eksik kolon: {missing}", price)

    clean = df.dropna(subset=_REQUIRED)
    if len(clean) < 3:
        return _hold("Yetersiz geçerli satır", float(df["close"].iloc[-1]))

    curr = clean.iloc[-1]
    prev = clean.iloc[-2]

    close        = float(curr["close"])
    ema_50       = float(curr["ema_50"])
    rsi          = float(curr["rsi_14"])
    volume       = float(curr["volume"])
    volume_ma20  = float(curr["volume_ma20"])
    volume_ratio = volume / volume_ma20 if volume_ma20 > 0 else 0.0

    prev_resistance = float(prev["resistance_20"])
    distance_to_res_pct = (
        (prev_resistance - close) / close * 100
        if close > 0 and prev_resistance > close
        else 0.0
    )

    prev_close_val      = float(prev["close"])
    daily_change_pct    = (close - prev_close_val) / prev_close_val * 100 if prev_close_val > 0 else 0.0

    close_to_ema20_pct = 0.0
    if "ema_20" in df.columns:
        _v = curr["ema_20"]
        if not pd.isna(_v) and float(_v) > 0:
            close_to_ema20_pct = (close / float(_v) - 1) * 100

    # ── İsteğe bağlı indikatörler ──────────────────────────────────────────────

    volatility_squeeze = False
    if "volatility_squeeze" in df.columns:
        volatility_squeeze = bool(curr.get("volatility_squeeze", False))

    bb_width_in_low = False
    if "bb_width" in df.columns:
        bb_series = df["bb_width"].dropna()
        if len(bb_series) >= 20:
            bb_thresh = bb_series.tail(60).quantile(_SETUP_BB_PCTILE)
            curr_bb   = curr.get("bb_width", float("nan"))
            if not pd.isna(curr_bb):
                bb_width_in_low = float(curr_bb) <= float(bb_thresh)

    macd_hist_rising = False
    if "macd_hist" in df.columns and len(clean) >= 4:
        mh = clean["macd_hist"].dropna()
        if len(mh) >= 3:
            macd_hist_rising = (
                float(mh.iloc[-1]) > float(mh.iloc[-2]) > float(mh.iloc[-3])
            )

    close_strengthening = (
        len(clean) >= 3
        and float(clean.iloc[-1]["close"]) > float(clean.iloc[-2]["close"])
        and float(clean.iloc[-2]["close"]) > float(clean.iloc[-3]["close"])
    )

    # ── SETUP koşulları (6 koşul) ──────────────────────────────────────────────

    s1_squeeze    = volatility_squeeze
    s2_above_ema50 = close > ema_50
    s3_rsi        = _SETUP_RSI_LOW <= rsi <= _SETUP_RSI_HIGH
    s4_volume     = _SETUP_VOL_LOW <= volume_ratio <= _SETUP_VOL_HIGH
    s5_distance   = 0 < distance_to_res_pct <= _SETUP_DIST_MAX
    s6_bb_low     = bb_width_in_low

    setup_met = sum([s1_squeeze, s2_above_ema50, s3_rsi, s4_volume, s5_distance, s6_bb_low])

    # ── EARLY_WATCH spesifik koşulları (4 koşul) ───────────────────────────────

    ew1_distance = 0 < distance_to_res_pct <= _EARLY_DIST_MAX
    ew2_volume   = volume_ratio >= _EARLY_VOL_MIN
    ew3_macd     = macd_hist_rising
    ew4_close    = close_strengthening

    early_met = sum([ew1_distance, ew2_volume, ew3_macd, ew4_close])

    # ── EARLY_WATCH kalite filtreleri ────────────────────────────────────────────
    ew_quality_ok = (
        daily_change_pct    <= _EARLY_MAX_DAILY_CHANGE   # günlük artış çok büyük değil
        and close_to_ema20_pct <= _EARLY_MAX_EMA20_GAP   # EMA20'den çok uzaklaşmamış
        and volume_ratio       <= _EARLY_MAX_VOL_RATIO    # hacim patlaması yok
    )

    details = {
        "close":                round(close, 2),
        "ema_50":               round(ema_50, 2),
        "rsi_14":               round(rsi, 2),
        "volume_ratio":         round(volume_ratio, 2),
        "distance_to_res_pct":  round(distance_to_res_pct, 2),
        "prev_resistance":      round(prev_resistance, 2),
        "daily_change_pct":     round(daily_change_pct, 2),
        "close_to_ema20_pct":   round(close_to_ema20_pct, 2),
        "ew_quality_ok":        ew_quality_ok,
        "volatility_squeeze":   s1_squeeze,
        "bb_width_in_low":      s6_bb_low,
        "macd_hist_rising":     macd_hist_rising,
        "close_strengthening":  close_strengthening,
        "setup_conditions_met": setup_met,
        "early_conditions_met": early_met,
    }

    # ── EARLY_WATCH: zorunlu + momentum + kalite filtreleri ───────────────────
    is_early_watch = (
        s2_above_ema50
        and (_SETUP_RSI_LOW - 5) <= rsi <= _SETUP_RSI_HIGH
        and ew1_distance        # zorunlu: direncin ≤%2 altında
        and ew2_volume          # zorunlu: hacim canlanıyor
        and (ew3_macd or ew4_close)  # en az 1 momentum işareti
        and setup_met >= 3      # setup koşullarının çoğu
        and ew_quality_ok       # kalite filtreleri
    )

    # ── SETUP: zorunlu koşullar + squeeze veya BB daralmasından en az biri ────
    is_setup = (
        s2_above_ema50
        and s3_rsi
        and s4_volume
        and s5_distance
        and (s1_squeeze or s6_bb_low)   # sıkışma veya BB daralması — en az biri
    )

    if is_early_watch:
        sl       = round(ema_50 * _EARLY_SL_MULT, 2)
        tp_raw   = round(prev_resistance * _EARLY_TP_MULT, 2)
        tp       = tp_raw if tp_raw > close else None
        strength = round(min(1.0, 0.65 + (early_met / 4) * 0.25 + (setup_met / 6) * 0.10), 3)
        return SetupSignal(
            signal="EARLY_WATCH",
            reason=_early_watch_reason(distance_to_res_pct, volume_ratio, macd_hist_rising, close_strengthening),
            price=round(close, 2),
            risk_level="MEDIUM",
            stop_loss=sl,
            take_profit=tp,
            strength=strength,
            details=details,
        )

    if is_setup:
        sl       = round(ema_50 * _SETUP_SL_MULT, 2)
        tp_raw   = round(prev_resistance * _SETUP_TP_MULT, 2)
        tp       = tp_raw if tp_raw > close else None
        strength = 0.85  # tüm 6 koşul karşılandı; EARLY_WATCH'tan düşük
        return SetupSignal(
            signal="SETUP",
            reason=_setup_reason(s1_squeeze, s6_bb_low, rsi, volume_ratio, distance_to_res_pct),
            price=round(close, 2),
            risk_level="LOW",
            stop_loss=sl,
            take_profit=tp,
            strength=strength,
            details=details,
        )

    return _hold("Ön kırılım koşulları oluşmadı", close, details)


# ── Açıklama oluşturucular ────────────────────────────────────────────────────

def _setup_reason(
    squeeze: bool,
    bb_low: bool,
    rsi: float,
    vol_ratio: float,
    dist_pct: float,
) -> str:
    parts: list[str] = []
    if squeeze:
        parts.append("Sıkışma aktif")
    if bb_low:
        parts.append("BB daralma bölgesinde")
    parts.append(f"RSI {rsi:.0f}")
    parts.append(f"Hacim x{vol_ratio:.1f}")
    parts.append(f"Direncin %{dist_pct:.1f} altında")
    return " | ".join(parts)


def _early_watch_reason(
    dist_pct: float,
    vol_ratio: float,
    macd_rising: bool,
    close_str: bool,
) -> str:
    parts = [f"Direncin %{dist_pct:.1f} altında"]
    if vol_ratio >= _EARLY_VOL_MIN:
        parts.append(f"Hacim canlanıyor (x{vol_ratio:.1f})")
    if macd_rising:
        parts.append("MACD hist yükseliyor")
    if close_str:
        parts.append("Fiyat güçleniyor")
    return " | ".join(parts)


# ── Yardımcı ─────────────────────────────────────────────────────────────────

def _hold(reason: str, price: float, details: dict | None = None) -> SetupSignal:
    return SetupSignal(
        signal="HOLD",
        reason=reason,
        price=round(price, 2),
        risk_level="LOW",
        stop_loss=None,
        take_profit=None,
        strength=0.0,
        details=details or {},
    )


# ── BaseStrategy entegrasyonu ─────────────────────────────────────────────────

class PreBreakoutSqueezeStrategy(BaseStrategy):
    """Pre-kırılım sıkışma stratejisi — sinyal motoruyla entegre."""

    name = "pre_breakout_squeeze"

    def generate_signal(self, df: pd.DataFrame, symbol: str) -> Optional[StrategySignal]:
        if not self._validate_df(df, _REQUIRED):
            return None

        result = generate_setup_signal(df)
        if result["signal"] == "HOLD":
            return None

        sig_type = SignalType.BUY  # SETUP ve EARLY_WATCH → potansiyel alım fırsatı

        return StrategySignal(
            symbol=symbol,
            signal_type=sig_type,
            strategy=f"{self.name}:{result['signal']}",
            strength=result["strength"],
            entry_price=result["price"],
            stop_loss=result["stop_loss"],
            take_profit=result["take_profit"],
            notes=result["reason"],
        )
