"""
Telegram Bildirim Modülü
========================
Yapılandırma (.env):
  TELEGRAM_BOT_TOKEN=<BotFather'dan alınan token>
  TELEGRAM_CHAT_ID=<hedef chat/kanal ID'si>

Bot yoksa veya yapılandırılmamışsa hiçbir fonksiyon istisna fırlatmaz;
yalnızca uyarı loglanır ve False/None döner.
"""

from __future__ import annotations

import asyncio
import io
from datetime import datetime
from typing import Optional

from loguru import logger
from telegram import Bot
from telegram.constants import ParseMode
from telegram.error import TelegramError

from app.config import settings


# ── Disclaimer sabiti ─────────────────────────────────────────────────────────

_DISCLAIMER = "⚠️ Bu yatırım tavsiyesi değildir."


# ── Bot yönetimi ──────────────────────────────────────────────────────────────

_bot: Optional[Bot] = None


def _get_bot() -> Optional[Bot]:
    global _bot
    if not settings.telegram_bot_token:
        return None
    if _bot is None:
        _bot = Bot(token=settings.telegram_bot_token)
    return _bot


def is_configured() -> bool:
    """Bot token ve chat ID her ikisi de dolu ise True döner."""
    return bool(settings.telegram_bot_token and settings.telegram_chat_id)


# ── Düşük seviye gönderici ────────────────────────────────────────────────────

async def send_telegram_message(
    message: str,
    parse_mode: str = ParseMode.MARKDOWN,
    retries: int = 2,
) -> bool:
    """
    Ham metin mesajı gönderir.

    Args:
        message:    Gönderilecek metin (Markdown destekli).
        parse_mode: ParseMode.MARKDOWN (varsayılan) veya ParseMode.HTML.
        retries:    Hata durumunda tekrar deneme sayısı.

    Returns:
        True → mesaj iletildi, False → yapılandırma eksik veya hata.
    """
    bot = _get_bot()
    if not bot:
        logger.warning("Telegram bot token ayarlanmamış — mesaj atlandı")
        return False
    if not settings.telegram_chat_id:
        logger.warning("Telegram chat ID ayarlanmamış — mesaj atlandı")
        return False

    for attempt in range(1, retries + 2):
        try:
            await bot.send_message(
                chat_id=settings.telegram_chat_id,
                text=message,
                parse_mode=parse_mode,
            )
            logger.debug(f"Telegram mesajı gönderildi (deneme {attempt})")
            return True
        except TelegramError as exc:
            if attempt <= retries:
                logger.warning(f"Telegram hata (deneme {attempt}/{retries+1}): {exc} — yeniden deneniyor")
                await asyncio.sleep(1)
            else:
                logger.error(f"Telegram gönderme başarısız: {exc}")
    return False


async def send_telegram_photo(
    photo: bytes,
    caption: str = "",
    parse_mode: str = ParseMode.MARKDOWN,
    retries: int = 2,
) -> bool:
    """
    PNG/JPEG byte içeriğini foto olarak gönderir (işaretli sinyal grafiği).
    `send_telegram_message` ile aynı fail-open desen: yapılandırma eksikse veya
    içerik boşsa sessizce False döner, istisna fırlatmaz.
    """
    if not photo:
        return False
    bot = _get_bot()
    if not bot:
        logger.warning("Telegram bot token ayarlanmamış — foto atlandı")
        return False
    if not settings.telegram_chat_id:
        logger.warning("Telegram chat ID ayarlanmamış — foto atlandı")
        return False

    for attempt in range(1, retries + 2):
        try:
            await bot.send_photo(
                chat_id=settings.telegram_chat_id,
                photo=io.BytesIO(photo),
                caption=caption or None,
                parse_mode=parse_mode if caption else None,
            )
            logger.debug(f"Telegram fotoğrafı gönderildi (deneme {attempt})")
            return True
        except TelegramError as exc:
            if attempt <= retries:
                logger.warning(f"Telegram foto hata (deneme {attempt}/{retries+1}): {exc} — yeniden deneniyor")
                await asyncio.sleep(1)
            else:
                logger.error(f"Telegram foto gönderme başarısız: {exc}")
    return False


# ── Mesaj formatlayıcılar ─────────────────────────────────────────────────────

def _strip_is(symbol: str) -> str:
    """'ASELS.IS' → 'ASELS'"""
    return symbol.removesuffix(".IS")


def _levels_block(result: dict) -> str:
    """
    BUY/LATE için tek temiz stop/hedef satırı üretir.

    Öncelik AI seviyeleri; yoksa sessizce yapı, o da yoksa ATR seviyesine düşer
    (kullanıcı yalnız AI sonucunu görmek istiyor, ama AI çalışmazsa sinyal
    stop/hedefsiz kalmasın diye yedek var). AI teyidi varsa 🤖 ile işaretlenir.
    """
    ai_sl   = result.get("ai_stop_loss")
    ai_tp   = result.get("ai_take_profit")
    ai_note = result.get("ai_rationale")

    sl = ai_sl or result.get("struct_stop_loss")   or result.get("stop_loss")
    tp = ai_tp or result.get("struct_take_profit") or result.get("take_profit")
    is_ai = bool(ai_sl and ai_tp)
    tag = "🤖 " if is_ai else ""

    lines = ""
    if sl:
        lines += f"🛑 {tag}Stop: `{sl:.2f} TL`\n"
    if tp:
        lines += f"🎯 {tag}Hedef: `{tp:.2f} TL`\n"
    if is_ai and ai_note:
        lines += f"💬 _{ai_note}_\n"
    return lines


def format_signal_message(result: dict) -> str:
    """
    Tek bir scan sonucunu Telegram mesajına çevirir.

    EARLY_WATCH  → 🟠 BIST YAKLAŞIYOR
    BUY          → 🟢 BIST SİNYALİ
    LATE_BREAKOUT → 🔴 BIST GEÇ KIRILI
    """
    signal   = result.get("signal", "").upper()
    symbol   = _strip_is(result.get("symbol", ""))
    price    = result.get("price", 0.0)
    reason   = result.get("reason", "")
    risk     = result.get("risk_level", "")
    strength = result.get("strength", 0.0)
    n_met    = result.get("conditions_met")
    dist     = result.get("distance_to_res_pct")

    ts = datetime.now().strftime("%d.%m.%Y %H:%M")

    if signal == "EARLY_WATCH":
        dist_line = f"Dirence: `%{dist:.1f}`\n" if dist and dist > 0 else ""
        levels = _levels_block(result)
        return (
            f"🟠 *BIST YAKLAŞIYOR*\n"
            f"{'─' * 22}\n"
            f"Hisse: *{symbol}*\n"
            f"Sinyal: *EARLY WATCH* 🟠\n"
            f"Fiyat: `{price:.2f} TL`\n"
            f"Güç: `{strength:.0%}`\n"
            f"Sebep: {reason}\n"
            f"{dist_line}"
            f"{levels}"
            f"🕐 `{ts}`\n"
            f"{'─' * 22}\n"
            f"{_DISCLAIMER}"
        )

    if signal == "BUY":
        levels = _levels_block(result)
        return (
            f"🚨 *BIST SİNYALİ*\n"
            f"{'─' * 22}\n"
            f"Hisse: *{symbol}*\n"
            f"Sinyal: *BUY* 🟢\n"
            f"Fiyat: `{price:.2f} TL`\n"
            f"Güç: `{strength:.0%}`\n"
            f"Sebep: {reason}\n"
            f"Risk: *{risk}*\n"
            f"{levels}"
            f"🕐 `{ts}`\n"
            f"{'─' * 22}\n"
            f"{_DISCLAIMER}"
        )

    if signal == "LATE_BREAKOUT":
        levels = _levels_block(result)
        return (
            f"🔴 *BIST GEÇ KIRILI*\n"
            f"{'─' * 22}\n"
            f"Hisse: *{symbol}*\n"
            f"Sinyal: *LATE BREAKOUT* 🔴\n"
            f"Fiyat: `{price:.2f} TL`\n"
            f"Güç: `{strength:.0%}`\n"
            f"Sebep: {reason}\n"
            f"Risk: *{risk}*\n"
            f"{levels}"
            f"⚠️ Hareket kaçmış olabilir — dikkatli olun\n"
            f"🕐 `{ts}`\n"
            f"{'─' * 22}\n"
            f"{_DISCLAIMER}"
        )

    # Bilinmeyen sinyal tipi — ham mesaj
    return (
        f"📊 *BIST — {signal}*\n"
        f"Hisse: *{symbol}* | Fiyat: `{price:.2f} TL`\n"
        f"{reason}\n{_DISCLAIMER}"
    )


def format_scan_summary(results: list[dict]) -> str:
    """scan_market() çıktısının tamamı için özet mesaj formatlar."""
    by_type = {
        sig: [r for r in results if r.get("signal") == sig]
        for sig in ("EARLY_WATCH", "BUY", "LATE_BREAKOUT")
    }
    ts = datetime.now().strftime("%d.%m.%Y %H:%M")

    ew  = len(by_type["EARLY_WATCH"])
    bu  = len(by_type["BUY"])
    lat = len(by_type["LATE_BREAKOUT"])

    lines = [
        f"📊 *BIST Tarama Özeti*",
        f"🕐 `{ts}`",
        f"{'─' * 22}",
        f"🟠 EARLY WATCH: `{ew}` | 🟢 BUY: `{bu}` | 🔴 GEÇ: `{lat}`",
    ]

    if by_type["EARLY_WATCH"]:
        lines.append("\n*🟠 Yaklaşan Kırılımlar:*")
        for r in by_type["EARLY_WATCH"]:
            sym  = _strip_is(r["symbol"])
            dist = r.get("distance_to_res_pct")
            dist_str = f" (%{dist:.1f} uzakta)" if dist and dist > 0 else ""
            lines.append(f"  • *{sym}* — `{r['price']:.2f} TL`{dist_str}")

    if by_type["BUY"]:
        lines.append("\n*🟢 AL Sinyalleri:*")
        for r in by_type["BUY"]:
            sym = _strip_is(r["symbol"])
            lines.append(f"  • *{sym}* — `{r['price']:.2f} TL` güç `{r['strength']:.0%}`")

    if by_type["LATE_BREAKOUT"]:
        lines.append("\n*🔴 Geç Kırılım (dikkat):*")
        for r in by_type["LATE_BREAKOUT"]:
            sym = _strip_is(r["symbol"])
            lines.append(f"  • *{sym}* — `{r['price']:.2f} TL` ⚠️")

    if not any(by_type.values()):
        lines.append("Aktif sinyal bulunamadı.")

    lines.append(f"\n{_DISCLAIMER}")
    return "\n".join(lines)


def format_bist100_signals(buy_results: list[dict], top_n: int = 5) -> str:
    """
    BIST100 taramasının en güçlü BUY sinyallerini formatlar.
    strength_score'a göre sıralı gelmesi beklenir.
    """
    if not buy_results:
        return "🔍 BIST100 taramasında aktif BUY sinyali bulunamadı."

    top = buy_results[:top_n]
    number_emojis = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣"]
    ts = datetime.now().strftime("%d.%m.%Y %H:%M")

    lines = [
        f"🚨 *BIST100 SİNYALLERİ*",
        f"🕐 `{ts}`",
        f"{'─' * 22}",
    ]

    for i, r in enumerate(top):
        sym   = _strip_is(r.get("symbol", ""))
        score = r.get("strength_score", 0)
        price = r.get("price", 0.0)
        reasons: list[str] = r.get("score_reasons") or []

        emoji = number_emojis[i] if i < len(number_emojis) else f"{i + 1}\\."
        lines.append(f"\n{emoji} *{sym}*")
        lines.append(f"Skor: `{score}`")
        lines.append(f"Fiyat: `{price:.2f} TL`")
        if reasons:
            lines.append("Sebep:")
            for rr in reasons:
                lines.append(f"  - {rr}")

    lines.append(f"\n{'─' * 22}")
    lines.append(_DISCLAIMER)
    return "\n".join(lines)


def format_bist100_early_signals(results: list[dict], top_n: int = 3) -> str:
    """
    BIST100 taramasının EARLY_WATCH sinyallerini formatlar.
    Kırılım öncesi fırsatlar listesi.
    """
    early_watch = [r for r in results if r.get("signal") == "EARLY_WATCH"]

    if not early_watch:
        return ""

    ts = datetime.now().strftime("%d.%m.%Y %H:%M")
    lines = [
        f"🔮 *BIST100 — Yaklaşan Fırsatlar*",
        f"🕐 `{ts}`",
        f"{'─' * 22}",
    ]

    if early_watch:
        lines.append(f"\n*🟠 Kırılıma Yakın ({len(early_watch)} hisse):*")
        for r in early_watch[:top_n]:
            sym  = _strip_is(r.get("symbol", ""))
            price = r.get("price", 0.0)
            dist  = r.get("distance_to_res_pct")
            dist_str = f" — direncin %{dist:.1f} altında" if dist and dist > 0 else ""
            reason = r.get("reason", "")
            lines.append(f"  🟠 *{sym}* `{price:.2f} TL`{dist_str}")
            lines.append(f"     _{reason}_")

    lines.append(f"\n{'─' * 22}")
    lines.append(_DISCLAIMER)
    return "\n".join(lines)


def format_bist100_scan_report(report: dict) -> str:
    """
    scan_bist100() çıktısından tarama özeti mesajı formatlar.
    Tüm sinyal tipleri için sayım içerir.
    """
    label        = report.get("label", "BIST100")
    scanned      = report.get("scanned", 0)
    buy_count    = report.get("buy_count", 0)
    ew_count     = report.get("early_watch_count", 0)
    late_count   = report.get("late_breakout_count", 0)
    error_count  = report.get("error_count", 0)
    elapsed      = report.get("elapsed_seconds", 0.0)
    mf_status    = report.get("market_filter", "")
    ts           = datetime.now().strftime("%d.%m.%Y %H:%M")

    mf_emoji = "🟢" if mf_status == "favorable" else ("🔴" if mf_status == "unfavorable" else "🟡")

    lines = [
        f"📊 *{label} Tarama Raporu*",
        f"🕐 `{ts}`",
        f"{'─' * 22}",
        f"🔍 Taranan: `{scanned}` hisse",
        f"🟠 EARLY: `{ew_count}` | 🟢 BUY: `{buy_count}` | 🔴 GEÇ: `{late_count}`",
        f"❌ Hata: `{error_count}` | ⏱ Süre: `{elapsed:.1f}s`",
        f"{mf_emoji} Endeks: `{mf_status}`",
    ]

    results = report.get("results", [])

    ew_list = [r for r in results if r.get("signal") == "EARLY_WATCH"]
    if ew_list:
        lines.append(f"\n*🟠 Yaklaşan Kırılımlar:*")
        for r in ew_list[:3]:
            sym  = _strip_is(r["symbol"])
            dist = r.get("distance_to_res_pct")
            dist_str = f" (%{dist:.1f})" if dist and dist > 0 else ""
            lines.append(f"  • *{sym}* — `{r['price']:.2f} TL`{dist_str}")

    buy_list = [r for r in results if r.get("signal") == "BUY"]
    if buy_list:
        lines.append(f"\n*🟢 En Güçlü BUY Sinyalleri:*")
        for r in buy_list[:5]:
            sym   = _strip_is(r["symbol"])
            score = r.get("strength_score", 0)
            price = r.get("price", 0.0)
            lines.append(f"  • *{sym}* — `{price:.2f} TL` skor `{score}`")

    lines.append(f"\n{_DISCLAIMER}")
    return "\n".join(lines)


def _format_symbol_row(idx: int, r: dict) -> str:
    """Bir sinyali kısa tek satır formatına çevirir."""
    sym   = _strip_is(r.get("symbol", ""))
    score = r.get("strength_score") or 0
    price = r.get("price", 0.0)
    rsi   = r.get("rsi_14")
    vol   = r.get("volume_ratio")
    dist  = r.get("distance_to_res_pct")
    day   = r.get("daily_change_pct")
    ema20 = r.get("close_to_ema20_pct")

    rsi_s  = f"{rsi:.0f}"     if rsi  is not None else "–"
    vol_s  = f"{vol:.2f}x"    if vol  is not None else "–"
    dist_s = f"%{dist:.1f}"   if (dist is not None and dist > 0) else "–"
    day_s  = f"%{day:+.1f}"   if day  is not None else "–"
    ema_s  = f"%{ema20:+.1f}" if ema20 is not None else "–"

    row = (
        f"{idx}) *{sym}* | Skor: `{score}` | Fiyat: `{price:.2f}` | "
        f"RSI: `{rsi_s}` | Hacim: `{vol_s}` | "
        f"Direnç: `{dist_s}` | Günlük: `{day_s}` | EMA20: `{ema_s}`"
    )

    # BUY/LATE için kompakt stop/hedef alt satırı — tek değer (AI > yapı > ATR)
    if r.get("signal") in ("BUY", "LATE_BREAKOUT"):
        is_ai = bool(r.get("ai_stop_loss") and r.get("ai_take_profit"))
        stop = _best_level(r, "stop_loss", "struct_stop_loss", "ai_stop_loss")
        targ = _best_level(r, "take_profit", "struct_take_profit", "ai_take_profit")
        if stop is not None or targ is not None:
            tag = "🤖 " if is_ai else ""
            s = f"`{stop:.2f}`" if stop is not None else "–"
            t = f"`{targ:.2f}`" if targ is not None else "–"
            row += f"\n    {tag}🛑 {s} | 🎯 {t}"
    return row


def _best_level(r: dict, atr_key: str, struct_key: str, ai_key: str):
    """Tek tercih edilen seviye: AI > yapı > ATR (ilk dolu olan)."""
    return r.get(ai_key) or r.get(struct_key) or r.get(atr_key)


def format_bist100_full_report(report: dict, top_n: int = 5) -> str:
    """
    Tüm sinyal tiplerini tek Telegram mesajında birleştirir.
    Sıra: EARLY_WATCH → BUY → LATE_BREAKOUT
    Her bölümde en fazla top_n hisse; boş bölümler gösterilmez.
    """
    ts      = datetime.now().strftime("%d.%m.%Y %H:%M")
    label   = report.get("label", "BIST100")
    results = report.get("results", [])

    sig_names = ("EARLY_WATCH", "BUY", "LATE_BREAKOUT")
    by_type: dict[str, list[dict]] = {
        sig: sorted(
            [r for r in results if r.get("signal") == sig],
            key=lambda r: -(r.get("strength_score") or 0),
        )
        for sig in sig_names
    }

    ew_n = len(by_type["EARLY_WATCH"])
    bu_n = len(by_type["BUY"])
    la_n = len(by_type["LATE_BREAKOUT"])

    scanned = report.get("scanned", 0)
    errors  = report.get("error_count", 0)
    elapsed = report.get("elapsed_seconds", 0.0)
    mf      = report.get("market_filter", "")
    mf_emoji = "🟢" if mf == "favorable" else ("🔴" if mf == "unfavorable" else "🟡")

    lines = [
        f"📊 *{label} Taraması* | `{ts}`",
        f"{'─' * 22}",
        f"🟠 EARLY: `{ew_n}` | 🟢 BUY: `{bu_n}` | 🔴 LATE: `{la_n}`",
        f"{'─' * 22}",
    ]

    _SECTIONS = [
        ("EARLY_WATCH",   "🟠 EARLY WATCH"),
        ("BUY",           "🟢 BUY"),
        ("LATE_BREAKOUT", "🔴 LATE BREAKOUT"),
    ]

    any_signal = False
    for sig, header in _SECTIONS:
        items = by_type[sig][:top_n]
        if not items:
            continue
        any_signal = True
        lines.append(f"\n*{header}*")
        for i, r in enumerate(items, 1):
            lines.append(_format_symbol_row(i, r))

    if not any_signal:
        lines.append("\nAktif sinyal bulunamadı.")

    lines.append(f"\n{'─' * 22}")
    lines.append(
        f"🔍 `{scanned}` hisse | ❌ `{errors}` hata | "
        f"⏱ `{elapsed:.1f}s` | {mf_emoji} `{mf}`"
    )
    lines.append(_DISCLAIMER)
    return "\n".join(lines)


def format_daily_summary(signals: list[dict]) -> str:
    """Günlük sinyal özetini formatlar."""
    buy  = sum(1 for s in signals if s.get("signal_type") == "BUY")
    sell = sum(1 for s in signals if s.get("signal_type") == "SELL")
    ts   = datetime.now().strftime("%d.%m.%Y")
    return (
        f"📈 *Günlük Özet — {ts}*\n"
        f"{'─' * 22}\n"
        f"Toplam sinyal: `{len(signals)}`\n"
        f"🟢 Alış: `{buy}` | 🔴 Satış: `{sell}`"
    )


def format_performance_summary(summary: dict) -> str:
    """Sinyal isabet raporunu (TP/SL/süre dolumu) Telegram için formatlar."""
    win = summary.get("win_rate")
    win_line = f"🎯 İsabet: `%{win}`" if win is not None else "🎯 İsabet: `—` (kapanan sinyal yok)"
    return (
        f"📊 *Sinyal Performansı — son {summary['window_days']} gün*\n"
        f"{'─' * 22}\n"
        f"Toplam üretilen: `{summary['total']}`\n"
        f"✅ Hedefe ulaşan (TP): `{summary['hit_tp']}`\n"
        f"🛑 Stop olan (SL): `{summary['hit_sl']}`\n"
        f"⏳ Süre dolan: `{summary['expired']}`\n"
        f"🔵 Hâlâ açık: `{summary['active']}`\n"
        f"{'─' * 22}\n"
        f"{win_line}"
    )


# ── Üst düzey yardımcı gönderici ─────────────────────────────────────────────

async def notify_signal(result: dict) -> bool:
    """Tek bir scan sonucunu formatlar ve gönderir."""
    message = format_signal_message(result)
    sent = await send_telegram_message(message)
    if sent:
        logger.info(f"Telegram sinyal bildirimi: {result.get('symbol')} {result.get('signal')}")
    return sent


async def notify_scan_summary(results: list[dict]) -> bool:
    """Tarama özetini gönderir."""
    if not results:
        return False
    return await send_telegram_message(format_scan_summary(results))


# ── TelegramNotifier — geriye dönük uyumluluk ────────────────────────────────

class TelegramNotifier:
    """
    Eski arayüz — generator.py ve diğer modüller bu sınıfı kullanır.
    İç implementasyon modül düzeyindeki fonksiyonlara delege eder.
    """

    async def send_signal(self, signal, assessment) -> bool:
        """StrategySignal + RiskAssessment → Telegram mesajı."""
        from app.strategies.base import SignalType

        sig_type  = signal.signal_type
        direction = "ALIŞ" if sig_type == SignalType.BUY else "SATIŞ"
        emoji     = "🟢" if sig_type == SignalType.BUY else "🔴"
        sl        = assessment.adjusted_stop_loss
        tp        = assessment.adjusted_take_profit

        sl_line = f"🛑 Stop Loss: `{sl:.2f} TL`\n" if sl else ""
        tp_line = f"🎯 Take Profit: `{tp:.2f} TL`\n" if tp else ""
        ts      = datetime.now().strftime("%d.%m.%Y %H:%M")

        message = (
            f"{emoji} *{direction} SİNYALİ*\n"
            f"{'─' * 22}\n"
            f"Hisse: *{_strip_is(signal.symbol)}*\n"
            f"Strateji: `{signal.strategy}`\n"
            f"Güç: `{signal.strength:.0%}`\n"
            f"Fiyat: `{signal.entry_price:.2f} TL`\n"
            f"{sl_line}"
            f"{tp_line}"
            f"⚖️ R/R: `{assessment.risk_reward_ratio}`\n"
            f"📝 {signal.notes}\n"
            f"🕐 `{ts}`\n"
            f"{'─' * 22}\n"
            f"{_DISCLAIMER}"
        )
        return await send_telegram_message(message)

    async def send_text(self, text: str) -> bool:
        return await send_telegram_message(text)

    async def send_daily_summary(self, signals: list) -> bool:
        return await send_telegram_message(format_daily_summary(signals))


notifier = TelegramNotifier()
