"""
Trend + Hacimli Kırılım Stratejisi
====================================
AL  → fiyat, EMA trendi üzerindeyken 20 günlük direnci hacim onayıyla kırarsa
ÇIKIŞ → fiyat EMA20 altına düşerse  (stop loss / take profit risk yöneticisine delege edilir)
HOLD → şartlar sağlanmıyorsa
"""

from __future__ import annotations

import pandas as pd
from typing import Optional, TypedDict

from app.strategies.base import BaseStrategy, StrategySignal, SignalType

# ── Sabitler ──────────────────────────────────────────────────────────────────

_REQUIRED = [
    "close", "ema_20", "ema_50",
    "rsi_14", "atr_14",
    "volume", "volume_ma20",
    "resistance_20",
]

_STOP_LOSS_PCT  = 0.03   # %3 aşağı — ATR hesaplanamazsa yedek
_TAKE_PROFIT_PCT = 0.06  # %6 yukarı — ATR hesaplanamazsa yedek
_ATR_SL_MULT    = 1.5    # stop  = close − 1.5 × ATR  (volatiliteye uyarlı)
_ATR_TP_MULT    = 3.0    # hedef = close + 3.0 × ATR  (R/R = 2.0)
_VOLUME_MULT    = 1.8    # hacim çarpanı
_RSI_LOW        = 50
_RSI_HIGH       = 75

# Geç kırılım tespiti — herhangi biri tetiklenirse LATE_BREAKOUT döner
_LATE_RSI_THRESHOLD = 68      # RSI aşırı alıma yakın
_LATE_DAILY_CHANGE  = 5.0     # % kapanış bazlı günlük artış
_LATE_EMA20_GAP     = 0.06    # EMA20 üzerinde %6
_LATE_VOL_RATIO     = 4.0     # aşırı hacim çarpanı
_LATE_RES_MARGIN    = 0.03    # direnç üzerinde %3

# ── Çıktı tipi ────────────────────────────────────────────────────────────────

class BreakoutSignal(TypedDict):
    signal: str          # "BUY" | "SELL" | "HOLD"
    reason: str
    price: float
    risk_level: str      # "LOW" | "MEDIUM" | "HIGH"
    stop_loss: Optional[float]
    take_profit: Optional[float]
    strength: float      # 0.0 – 1.0
    details: dict        # her koşulun ayrı değerleri


# ── Yardımcı hesaplamalar ─────────────────────────────────────────────────────

def _risk_level(rsi: float, volume_ratio: float, atr_pct: float) -> str:
    """
    Üç faktörü puanlar:
      • RSI 50–60 → düşük baskı   | 60–70 → orta | 70–75 → yüksek
      • volume_ratio 1.8–2.5 → normal | 2.5–4 → güçlü | 4+ → aşırı
      • ATR/close % < 1.5 → dar | 1.5–3 → normal | 3+ → geniş
    Üçten ikisi "yüksek" ise HIGH, ikisi "orta-üstü" ise MEDIUM, geri kalanı LOW.
    """
    score = 0
    score += 2 if rsi >= 70 else (1 if rsi >= 60 else 0)
    score += 2 if volume_ratio >= 4 else (1 if volume_ratio >= 2.5 else 0)
    score += 2 if atr_pct >= 3 else (1 if atr_pct >= 1.5 else 0)

    if score >= 4:
        return "HIGH"
    if score >= 2:
        return "MEDIUM"
    return "LOW"


def _signal_strength(
    close: float,
    prev_resistance: float,
    volume_ratio: float,
    rsi: float,
    ema20: float,
    ema50: float,
) -> float:
    """
    0–1 arası güç skoru:
      • kırılım marjı   (close/resistance – 1) → %0–3 bandına normalize
      • ek hacim oranı  (volume_ratio – 1.8) / 2.2 → 0–1
      • RSI 50–75 içindeki konum → ortaya yakın daha yüksek puan
      • EMA ayrışması   (ema20/ema50 – 1) → %0–2
    """
    breakout_score = min((close / prev_resistance - 1) / 0.03, 1.0)
    volume_score   = min((volume_ratio - _VOLUME_MULT) / 2.2, 1.0)
    rsi_score      = 1 - abs(rsi - 62.5) / 12.5           # 62.5 merkez → en yüksek puan
    ema_score      = min((ema20 / ema50 - 1) / 0.02, 1.0)

    raw = 0.35 * breakout_score + 0.30 * volume_score + 0.20 * rsi_score + 0.15 * ema_score
    return round(max(0.0, min(1.0, raw)), 3)


# ── Ana fonksiyon ─────────────────────────────────────────────────────────────

def generate_signal(df: pd.DataFrame) -> BreakoutSignal:
    """
    İndikatörlü OHLCV DataFrame'i alır, kırılım sinyali üretir.

    Gerekli kolonlar: close, ema_20, ema_50, rsi_14, atr_14,
                      volume, volume_ma20, resistance_20

    Returns:
        BreakoutSignal dict — signal, reason, price, risk_level,
                              stop_loss, take_profit, strength, details
    """
    # ── Veri doğrulama ────────────────────────────────────────────────────────
    missing = [c for c in _REQUIRED if c not in df.columns]
    if missing:
        return _hold(f"Eksik kolon: {missing}", df["close"].iloc[-1] if "close" in df.columns else 0.0)

    clean = df.dropna(subset=_REQUIRED)
    if len(clean) < 2:
        return _hold("Yetersiz geçerli satır sayısı", df["close"].iloc[-1])

    curr = clean.iloc[-1]
    prev = clean.iloc[-2]

    close        = float(curr["close"])
    ema20        = float(curr["ema_20"])
    ema50        = float(curr["ema_50"])
    rsi          = float(curr["rsi_14"])
    atr          = float(curr["atr_14"])
    volume       = float(curr["volume"])
    volume_ma20  = float(curr["volume_ma20"])
    volume_ratio = volume / volume_ma20 if volume_ma20 > 0 else 0.0
    atr_pct      = atr / close * 100

    # Kırılım için bir önceki barın 20 günlük direncini kullan
    prev_resistance = float(prev["resistance_20"])

    # ── AL koşulları ──────────────────────────────────────────────────────────
    c1_trend_above_ema20 = close > ema20
    c2_ema_uptrend       = ema20 > ema50
    c3_breakout          = close > prev_resistance
    c4_volume_surge      = volume_ratio >= _VOLUME_MULT
    c5_rsi_range         = _RSI_LOW <= rsi <= _RSI_HIGH

    details = {
        "close":           round(close, 2),
        "ema_20":          round(ema20, 2),
        "ema_50":          round(ema50, 2),
        "rsi_14":          round(rsi, 2),
        "atr_14":          round(atr, 2),
        "atr_pct":         round(atr_pct, 2),
        "volume_ratio":    round(volume_ratio, 2),
        "prev_resistance": round(prev_resistance, 2),
        "c1_above_ema20":  c1_trend_above_ema20,
        "c2_ema_uptrend":  c2_ema_uptrend,
        "c3_breakout":     c3_breakout,
        "c4_volume_surge": c4_volume_surge,
        "c5_rsi_range":    c5_rsi_range,
    }

    # ── ÇIKIŞ sinyali (AL pozisyonundan çıkış) ───────────────────────────────
    if not c1_trend_above_ema20:
        return BreakoutSignal(
            signal="SELL",
            reason=f"Çıkış: Close ({close:.2f}) EMA20 ({ema20:.2f}) altına düştü",
            price=close,
            risk_level=_risk_level(rsi, volume_ratio, atr_pct),
            stop_loss=None,
            take_profit=None,
            strength=0.0,
            details=details,
        )

    # ── AL sinyali ────────────────────────────────────────────────────────────
    if all([c1_trend_above_ema20, c2_ema_uptrend, c3_breakout, c4_volume_surge, c5_rsi_range]):
        prev_close       = float(prev["close"])
        daily_change_pct = (close - prev_close) / prev_close * 100 if prev_close > 0 else 0.0
        res_margin       = close / prev_resistance - 1 if prev_resistance > 0 else 0.0

        late_checks: list[tuple[bool, str]] = [
            (daily_change_pct > _LATE_DAILY_CHANGE,   f"Günlük +%{daily_change_pct:.1f}"),
            (close / ema20 - 1 > _LATE_EMA20_GAP,     f"EMA20 +%{(close/ema20 - 1)*100:.1f}"),
            (rsi > _LATE_RSI_THRESHOLD,                f"RSI {rsi:.0f}"),
            (volume_ratio > _LATE_VOL_RATIO,           f"Hacim x{volume_ratio:.1f}"),
            (res_margin > _LATE_RES_MARGIN,            f"Direnç +%{res_margin*100:.1f}"),
        ]

        details["daily_change_pct"] = round(daily_change_pct, 2)
        details["late_flags"]       = sum(triggered for triggered, _ in late_checks)

        sl, tp   = _atr_sl_tp(close, atr)
        strength = _signal_strength(close, prev_resistance, volume_ratio, rsi, ema20, ema50)

        if any(triggered for triggered, _ in late_checks):
            late_reason = "Geç kırılım: " + " | ".join(
                label for triggered, label in late_checks if triggered
            )
            return BreakoutSignal(
                signal="LATE_BREAKOUT",
                reason=late_reason,
                price=close,
                risk_level=_risk_level(rsi, volume_ratio, atr_pct),
                stop_loss=sl,
                take_profit=tp,
                strength=strength,
                details=details,
            )

        reason = (
            f"Kırılım: {close:.2f} > direnç {prev_resistance:.2f} | "
            f"EMA20/50 trend yukarı | "
            f"Hacim x{volume_ratio:.1f} | "
            f"RSI14: {rsi:.1f}"
        )
        return BreakoutSignal(
            signal="BUY",
            reason=reason,
            price=close,
            risk_level=_risk_level(rsi, volume_ratio, atr_pct),
            stop_loss=sl,
            take_profit=tp,
            strength=strength,
            details=details,
        )

    # ── HOLD — hangi koşul sağlanmıyor ───────────────────────────────────────
    failed = []
    if not c2_ema_uptrend:
        failed.append(f"EMA20 ({ema20:.2f}) ≤ EMA50 ({ema50:.2f})")
    if not c3_breakout:
        failed.append(f"Fiyat ({close:.2f}) ≤ direnç ({prev_resistance:.2f})")
    if not c4_volume_surge:
        failed.append(f"Hacim oranı {volume_ratio:.2f} < {_VOLUME_MULT}")
    if not c5_rsi_range:
        failed.append(f"RSI14 {rsi:.1f} ∉ [{_RSI_LOW}–{_RSI_HIGH}]")

    return _hold(" | ".join(failed) if failed else "Koşullar sağlanmıyor", close, details)


def _atr_sl_tp(close: float, atr: float) -> tuple[float, float]:
    """
    ATR bazlı stop-loss / take-profit.
    ATR geçersizse (NaN/0) sabit yüzdeye düşer.
    Aşırı volatil hisselerde stop %8'den, sakin hisselerde %1.5'tan
    uzağa taşınmaz — yüzde sınırlarıyla kelepçelenir.
    """
    if atr and atr > 0 and not pd.isna(atr):
        sl_dist = min(max(_ATR_SL_MULT * atr, close * 0.015), close * 0.08)
        tp_dist = sl_dist * (_ATR_TP_MULT / _ATR_SL_MULT)
        return round(close - sl_dist, 2), round(close + tp_dist, 2)
    return round(close * (1 - _STOP_LOSS_PCT), 2), round(close * (1 + _TAKE_PROFIT_PCT), 2)


# ── Yardımcı inşa fonksiyonları ───────────────────────────────────────────────

def _hold(reason: str, price: float, details: dict | None = None) -> BreakoutSignal:
    return BreakoutSignal(
        signal="HOLD",
        reason=reason,
        price=round(price, 2),
        risk_level="LOW",
        stop_loss=None,
        take_profit=None,
        strength=0.0,
        details=details or {},
    )


# ── BaseStrategy entegrasyonu (sinyal motoru için) ────────────────────────────

class TrendBreakoutStrategy(BaseStrategy):
    """
    Trend + hacimli kırılım stratejisi — sinyal motoruyla entegre.
    generate_signal(df) sonucunu StrategySignal'e çevirir.
    """

    name = "trend_breakout"

    def generate_signal(self, df: pd.DataFrame, symbol: str) -> Optional[StrategySignal]:
        if not self._validate_df(df, _REQUIRED):
            return None

        result = generate_signal(df)

        if result["signal"] == "HOLD":
            return None

        if result["signal"] in ("BUY", "LATE_BREAKOUT"):
            sig_type = SignalType.BUY
        else:
            sig_type = SignalType.SELL

        return StrategySignal(
            symbol=symbol,
            signal_type=sig_type,
            strategy=self.name,
            strength=result["strength"],
            entry_price=result["price"],
            stop_loss=result["stop_loss"],
            take_profit=result["take_profit"],
            notes=result["reason"],
        )
