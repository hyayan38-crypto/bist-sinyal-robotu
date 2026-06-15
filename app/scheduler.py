"""
Zamanlanmış BIST100 Tarama Sistemi
=====================================
Her iş gününde 4 kez BIST100 hisselerini paralel olarak tarar,
BUY sinyali üretenleri Telegram'a gönderir.

Tarama saatleri (TR, Pazartesi–Cuma):
  10:30 — Açılış sonrası
  12:30 — Öğle arası
  15:30 — Kapanış öncesi
  18:10 — Kapanış sonrası (gün sonu değerlendirme)

Kullanım:
  scheduler.start()  → FastAPI lifespan'da çağrılır
  scheduler.stop()   → uygulama kapanırken çağrılır
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from functools import partial
from zoneinfo import ZoneInfo

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from loguru import logger

from app.notifications.telegram import (
    format_bist100_signals,
    format_bist100_early_signals,
    format_bist100_scan_report,
    format_bist100_full_report,
    format_performance_summary,
    send_telegram_message,
)
from app.signals.scanner import scan_bist100

_TZ = ZoneInfo("Europe/Istanbul")

_SCAN_TIMES = [
    (10, 30, "acilis"),
    (12, 30, "ogle"),
    (15, 30, "kapanis_oncesi"),
    (18, 10, "gun_sonu"),
]

_TOP_SIGNALS = 5   # Telegram'a gönderilecek maksimum sinyal sayısı

# Tarama sırasında veri alınamayan sembol oranı bu eşiği aşarsa Telegram'a
# sağlık uyarısı gönderilir (robot sessizce "ölü" kalmasın).
_ERROR_ALERT_RATIO = 0.30


# ── Tarama görevi ─────────────────────────────────────────────────────────────

async def run_bist100_scan(force_market_refresh: bool = False) -> dict:
    """
    BIST100 tam tarama:
    1. Tüm BIST100 hisselerini paralel olarak tara
    2. En güçlü BUY sinyallerini Telegram'a gönder
    3. Tarama raporunu gönder

    Returns:
        Tarama raporu dict'i (buy_count, early_watch_count, scanned, error_count vb.)
    """
    started_at = datetime.now(_TZ)
    logger.info(f"BIST100 taraması başladı — {started_at.strftime('%d.%m.%Y %H:%M')} (TR)")

    loop = asyncio.get_running_loop()
    report: dict = await loop.run_in_executor(
        None,
        lambda: scan_bist100(
            period="1y",
            include_watch=True,
            force_market_refresh=force_market_refresh,
        ),
    )

    results         = report.get("results", [])
    early_watch_list = [r for r in results if r["signal"] == "EARLY_WATCH"]
    buy_list         = [r for r in results if r["signal"] == "BUY"]
    late_list        = [r for r in results if r["signal"] == "LATE_BREAKOUT"]

    logger.info(
        f"BIST100 tarama bitti: {report['scanned']} sembol | "
        f"{len(buy_list)} BUY | {len(early_watch_list)} EARLY_WATCH | "
        f"{len(late_list)} LATE | "
        f"{report['error_count']} hata | {report['elapsed_seconds']:.1f}s"
    )

    async def _send(text: str):
        try:
            await send_telegram_message(text)
        except Exception as exc:
            logger.error(f"Telegram gönderim hatası: {exc}")

    # Tüm sinyal tiplerini tek mesajda gönder
    await _send(format_bist100_full_report(report))

    # ── Sağlık uyarısı: veri alınamayan sembol oranı yüksekse haber ver ───────
    scanned     = report.get("scanned", 0)
    error_count = report.get("error_count", 0)
    if scanned > 0 and error_count / scanned >= _ERROR_ALERT_RATIO:
        pct = error_count / scanned * 100
        logger.warning(f"Yüksek tarama hata oranı: {error_count}/{scanned} (%{pct:.0f})")
        await _send(
            f"⚠️ *Tarama Sağlık Uyarısı*\n"
            f"{'─' * 22}\n"
            f"Veri alınamayan sembol: `{error_count}/{scanned}` (%{pct:.0f})\n"
            f"Sinyaller eksik olabilir — veri kaynağı (yfinance) sorunlu olabilir."
        )

    return report


# Geriye dönük uyumluluk için — scheduler.py dışarıdan import edilebiliyor
async def run_daily_scan() -> dict:
    """run_bist100_scan'ın geriye dönük uyumlu takma adı."""
    return await run_bist100_scan(force_market_refresh=True)


# ── Sinyal takibi entegrasyonu (DB) ──────────────────────────────────────────

async def _track_signals(report: dict, send_summary: bool) -> None:
    """
    Tarama sonrası DB işlemleri — tamamen fail-open:
      1. Açık sinyalleri güncel fiyatla değerlendir (TP/SL/süre)
      2. Bu taramadaki yeni BUY/LATE sinyallerini kaydet
      3. (gün sonu) isabet özetini Telegram'a gönder

    Herhangi bir DB/ağ hatası taramayı veya bildirimi bozmaz.
    """
    try:
        from app.database import AsyncSessionLocal
        from app.signals.tracker import (
            evaluate_open_signals,
            persist_scan_signals,
            build_performance_summary,
        )

        async with AsyncSessionLocal() as db:
            await evaluate_open_signals(db)
            await persist_scan_signals(db, report)

            if send_summary:
                summary = await build_performance_summary(db)
                try:
                    await send_telegram_message(format_performance_summary(summary))
                except Exception as exc:  # noqa: BLE001
                    logger.error(f"Performans özeti gönderim hatası: {exc}")
    except Exception as exc:  # noqa: BLE001 — takip katmanı taramayı asla bozmamalı
        logger.error(f"Sinyal takibi atlandı (fail-open): {exc}")


async def scheduled_scan_job(label: str = "manuel", send_summary: bool = False) -> dict:
    """
    Zamanlanmış iş sarmalayıcısı: tam taramayı çalıştırır, ardından sinyal
    takibini (kayıt + değerlendirme + özet) tetikler. Cron işleri bunu çağırır;
    run_bist100_scan / run_daily_scan çekirdeği değişmeden kalır.
    """
    report = await run_bist100_scan(force_market_refresh=True)
    await _track_signals(report, send_summary=send_summary)
    return report


# ── Zamanlayıcı ───────────────────────────────────────────────────────────────

class BISTScheduler:
    """APScheduler AsyncIOScheduler sarmalayıcı — 4 günlük tarama zamanı."""

    def __init__(self):
        self._scheduler = AsyncIOScheduler(timezone=str(_TZ))

    # ── Yaşam döngüsü ─────────────────────────────────────────────────────────

    def start(self):
        """Zamanlayıcıyı başlatır ve tüm günlük tarama işlerini ekler."""
        for hour, minute, label in _SCAN_TIMES:
            job_id = f"bist100_scan_{label}"
            # İsabet özeti yalnızca gün sonu taramasında gönderilir
            send_summary = label == "gun_sonu"
            self._scheduler.add_job(
                partial(scheduled_scan_job, label, send_summary),
                trigger=CronTrigger(
                    day_of_week="mon-fri",
                    hour=hour,
                    minute=minute,
                    timezone=_TZ,
                ),
                id=job_id,
                name=f"BIST100 Tarama {hour:02d}:{minute:02d}",
                replace_existing=True,
                misfire_grace_time=3600,  # 1 saat — sunucu geç başlasa da job çalışır
            )

        self._scheduler.start()
        times_str = " | ".join(f"{h:02d}:{m:02d}" for h, m, _ in _SCAN_TIMES)
        next_run  = self._next_run_all()
        logger.info(
            f"Zamanlayıcı başlatıldı — Pzt-Cum [{times_str}] TR "
            f"| Sonraki: {next_run}"
        )

    def stop(self):
        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)
            logger.info("Zamanlayıcı durduruldu")

    # ── Durum & kontrol ───────────────────────────────────────────────────────

    def status(self) -> dict:
        jobs = self._scheduler.get_jobs()
        job_list = []
        for job in jobs:
            job_list.append({
                "id":       job.id,
                "name":     job.name,
                "next_run": job.next_run_time.strftime("%d.%m.%Y %H:%M %Z") if job.next_run_time else None,
            })

        times_str = ", ".join(f"{h:02d}:{m:02d}" for h, m, _ in _SCAN_TIMES)
        return {
            "running":       self._scheduler.running,
            "schedule":      f"Pazartesi-Cuma {times_str} TR saati",
            "timezone":      str(_TZ),
            "jobs":          job_list,
            "next_run":      self._next_run_all(),
            "scan_universe": "BIST100 (100 hisse)",
        }

    def next_run_time(self) -> str | None:
        """İlk yaklaşan tarama zamanını döner."""
        return self._next_run_all()

    def _next_run_all(self) -> str | None:
        """Tüm işler arasından en yakın çalışma zamanını döner."""
        jobs = self._scheduler.get_jobs()
        run_times = [j.next_run_time for j in jobs if j.next_run_time]
        if not run_times:
            return None
        earliest = min(run_times)
        return earliest.strftime("%d.%m.%Y %H:%M %Z")

    def trigger_now(self) -> None:
        """Manuel tetikleme — zamanlayıcıyı beklemeden hemen tarama başlatır."""
        job_id = f"bist100_scan_{_SCAN_TIMES[0][2]}"
        self._scheduler.modify_job(job_id, next_run_time=datetime.now(_TZ))


# ── Singleton ─────────────────────────────────────────────────────────────────

scheduler = BISTScheduler()
