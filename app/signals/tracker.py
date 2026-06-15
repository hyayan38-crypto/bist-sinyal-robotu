"""
Sinyal Takibi ve Sonuç Değerlendirme
======================================
Üretilen BUY / LATE_BREAKOUT sinyallerini veritabanına kaydeder, sonraki
taramalarda fiyat hareketine göre sonuçlandırır (TP / SL / süre dolumu) ve
isabet oranı özeti üretir. Böylece "stratejim canlıda gerçekten kazandırıyor
mu?" sorusu ölçülebilir hale gelir.

Akış (scheduler her tarama sonunda çağırır):
    evaluate_open_signals(db)   → açık sinyalleri güncelle
    persist_scan_signals(db, report) → yeni sinyalleri kaydet
    build_performance_summary(db)    → (gün sonu) Telegram özeti

Tüm DB işlemleri scheduler tarafında fail-open sarmalanır; bu modül
istisnaları yutmaz, çağıran katman yönetir.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.data.fetcher import fetch_symbol_data_cached, FetchError
from app.database.crud import (
    create_signal,
    get_active_signals,
    get_active_signal_for,
    get_signals_since,
    update_signal_status,
)
from app.database.models import SignalStatus

# Kaydedilecek sinyal tipleri (EARLY_WATCH yalnızca izleme — pozisyon değil)
_TRACKED_SIGNALS = ("BUY", "LATE_BREAKOUT")


# ── Kayıt ────────────────────────────────────────────────────────────────────

async def persist_scan_signals(db: AsyncSession, report: dict) -> int:
    """
    Tarama raporundaki BUY/LATE_BREAKOUT sinyallerini DB'ye kaydeder.
    Aynı sembol+strateji için açık sinyal varsa tekrar oluşturmaz.

    Returns: yeni kaydedilen sinyal sayısı.
    """
    results = report.get("results", [])
    created = 0
    expires_at = datetime.utcnow() + timedelta(days=settings.signal_expiry_days)

    for r in results:
        if r.get("signal") not in _TRACKED_SIGNALS:
            continue
        if r.get("stop_loss") is None or r.get("take_profit") is None:
            continue

        symbol   = r["symbol"]
        strategy = r.get("strategy", "trend_breakout")

        existing = await get_active_signal_for(db, symbol, strategy)
        if existing is not None:
            continue

        await create_signal(db, {
            "symbol":      symbol,
            "signal_type": r["signal"],
            "strategy":    strategy,
            "strength":    float(r.get("strength") or 0.0),
            "entry_price": float(r["price"]),
            "stop_loss":   float(r["stop_loss"]),
            "take_profit": float(r["take_profit"]),
            "status":      SignalStatus.ACTIVE.value,
            "notes":       r.get("reason", ""),
            "expires_at":  expires_at,
        })
        created += 1

    if created:
        logger.info(f"[TAKİP] {created} yeni sinyal kaydedildi")
    return created


# ── Değerlendirme ──────────────────────────────────────────────────────────────

def _resolve_outcome(signal, df) -> str | None:
    """
    Sinyalin oluşturulduğu günden SONRAKİ barlara bakarak sonucu belirler.
    Aynı bar içinde hem SL hem TP varsa, ihtiyatlı davranıp SL'yi öne alır.

    Returns: SignalStatus.value veya None (henüz açık).
    """
    created_date = signal.created_at.date()
    after = df[df.index.date > created_date]

    for _, bar in after.iterrows():
        if signal.stop_loss is not None and bar["low"] <= signal.stop_loss:
            return SignalStatus.HIT_SL.value
        if signal.take_profit is not None and bar["high"] >= signal.take_profit:
            return SignalStatus.HIT_TP.value

    if signal.expires_at and datetime.utcnow() >= signal.expires_at:
        return SignalStatus.EXPIRED.value
    return None


async def evaluate_open_signals(db: AsyncSession) -> dict:
    """
    Tüm açık sinyalleri güncel fiyat verisiyle değerlendirir ve durumlarını
    günceller (HIT_TP / HIT_SL / EXPIRED).

    Returns: {"evaluated": n, "hit_tp": .., "hit_sl": .., "expired": .., "still_open": ..}
    """
    active = await get_active_signals(db)
    counts = {"evaluated": len(active), "hit_tp": 0, "hit_sl": 0, "expired": 0, "still_open": 0}
    if not active:
        return counts

    loop = asyncio.get_running_loop()

    for sig in active:
        try:
            fetch = await loop.run_in_executor(
                None,
                lambda s=sig: fetch_symbol_data_cached(s.symbol, period="3mo", interval="1d"),
            )
        except FetchError as exc:
            logger.debug(f"[TAKİP] {sig.symbol} değerlendirme atlandı: {exc}")
            counts["still_open"] += 1
            continue

        outcome = _resolve_outcome(sig, fetch.df)
        if outcome is None:
            counts["still_open"] += 1
            continue

        await update_signal_status(db, sig.id, outcome)
        if outcome == SignalStatus.HIT_TP.value:
            counts["hit_tp"] += 1
        elif outcome == SignalStatus.HIT_SL.value:
            counts["hit_sl"] += 1
        else:
            counts["expired"] += 1

    logger.info(
        f"[TAKİP] Değerlendirme: {counts['hit_tp']} TP | {counts['hit_sl']} SL | "
        f"{counts['expired']} süre doldu | {counts['still_open']} açık"
    )
    return counts


# ── Özet ─────────────────────────────────────────────────────────────────────

async def build_performance_summary(db: AsyncSession, days: int | None = None) -> dict:
    """
    Son `days` günde oluşturulan sinyallerin sonuç dağılımını ve isabet oranını
    döner. Isabet = TP / (TP + SL).
    """
    days = days or settings.performance_window_days
    since = datetime.utcnow() - timedelta(days=days)
    signals = await get_signals_since(db, since)

    tally = {"HIT_TP": 0, "HIT_SL": 0, "EXPIRED": 0, "ACTIVE": 0}
    for s in signals:
        tally[s.status] = tally.get(s.status, 0) + 1

    closed = tally["HIT_TP"] + tally["HIT_SL"]
    win_rate = round(tally["HIT_TP"] / closed * 100, 1) if closed else None

    return {
        "window_days": days,
        "total":       len(signals),
        "hit_tp":      tally["HIT_TP"],
        "hit_sl":      tally["HIT_SL"],
        "expired":     tally["EXPIRED"],
        "active":      tally["ACTIVE"],
        "win_rate":    win_rate,
    }
