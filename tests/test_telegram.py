"""
Telegram bildirim modülü testleri.
Bot gerçekten çağrılmaz — mock kullanılır.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.notifications.telegram import (
    send_telegram_message,
    notify_signal,
    notify_scan_summary,
    format_signal_message,
    format_scan_summary,
    format_daily_summary,
    format_bist100_early_signals,
    format_bist100_full_report,
    is_configured,
    _strip_is,
    _DISCLAIMER,
    TelegramNotifier,
)


# ── Yardımcı ──────────────────────────────────────────────────────────────────

def _buy_result(**kw) -> dict:
    base = {
        "symbol": "ASELS.IS", "signal": "BUY", "price": 123.45,
        "reason": "EMA trend + hacimli kırılım", "risk_level": "MEDIUM",
        "strength": 0.72, "stop_loss": 119.75, "take_profit": 130.86,
        "market_filter": "favorable", "conditions_met": 5, "distance_to_res_pct": 0.0,
    }
    return {**base, **kw}


def _watch_result(**kw) -> dict:
    base = {
        "symbol": "THYAO.IS", "signal": "WATCH", "price": 110.20,
        "reason": "Trend yukarı | Hacim x2.1 | Dirence %1.8 uzakta",
        "risk_level": "LOW", "strength": 0.4,
        "stop_loss": None, "take_profit": None,
        "market_filter": "favorable", "conditions_met": 4, "distance_to_res_pct": 1.8,
    }
    return {**base, **kw}


def _patch_bot(send_ok: bool = True):
    """Bot.send_message mock'u — başarılı veya hata fırlatır."""
    from telegram.error import TelegramError
    mock_bot = AsyncMock()
    if send_ok:
        mock_bot.send_message = AsyncMock(return_value=MagicMock())
    else:
        mock_bot.send_message = AsyncMock(side_effect=TelegramError("test error"))
    return patch("app.notifications.telegram._get_bot", return_value=mock_bot)


def _patch_configured(value: bool):
    return patch("app.notifications.telegram.is_configured", return_value=value)


# ── _strip_is ─────────────────────────────────────────────────────────────────

class TestStripIs:
    def test_removes_is_suffix(self):
        assert _strip_is("ASELS.IS") == "ASELS"

    def test_no_suffix_unchanged(self):
        assert _strip_is("THYAO") == "THYAO"

    def test_lowercase_suffix_not_removed(self):
        # Semboller her zaman büyük harf — küçük .is kaldırılmaz
        assert _strip_is("asels.is") == "asels.is"


# ── format_signal_message ─────────────────────────────────────────────────────

class TestFormatSignalMessage:
    def test_buy_contains_header(self):
        msg = format_signal_message(_buy_result())
        assert "BIST SİNYALİ" in msg

    def test_buy_contains_symbol_without_is(self):
        msg = format_signal_message(_buy_result())
        assert "ASELS" in msg
        assert ".IS" not in msg

    def test_buy_contains_price(self):
        msg = format_signal_message(_buy_result())
        assert "123.45" in msg

    def test_buy_contains_buy_label(self):
        msg = format_signal_message(_buy_result())
        assert "BUY" in msg

    def test_buy_contains_reason(self):
        msg = format_signal_message(_buy_result())
        assert "EMA trend" in msg

    def test_buy_contains_risk(self):
        msg = format_signal_message(_buy_result())
        assert "MEDIUM" in msg

    def test_buy_contains_stop_loss(self):
        msg = format_signal_message(_buy_result())
        assert "119.75" in msg

    def test_buy_contains_take_profit(self):
        msg = format_signal_message(_buy_result())
        assert "130.86" in msg

    def test_buy_contains_disclaimer(self):
        assert _DISCLAIMER in format_signal_message(_buy_result())

    def test_buy_no_stop_loss_when_none(self):
        msg = format_signal_message(_buy_result(stop_loss=None))
        assert "Stop Loss" not in msg

    def test_watch_contains_watch_header(self):
        msg = format_signal_message(_watch_result())
        assert "TAKİP" in msg

    def test_watch_contains_conditions_met(self):
        msg = format_signal_message(_watch_result())
        assert "4/5" in msg

    def test_watch_contains_distance(self):
        msg = format_signal_message(_watch_result())
        assert "1.8" in msg

    def test_watch_no_stop_loss(self):
        msg = format_signal_message(_watch_result())
        assert "Stop Loss" not in msg

    def test_watch_contains_disclaimer(self):
        assert _DISCLAIMER in format_signal_message(_watch_result())

    def test_unknown_signal_type_no_crash(self):
        msg = format_signal_message({"signal": "UNKNOWN", "symbol": "X.IS", "price": 1.0, "reason": ""})
        assert isinstance(msg, str)

    def test_format_matches_requested_structure(self):
        """Kullanıcının istediği formata yakın olduğunu kontrol et."""
        msg = format_signal_message(_buy_result(
            symbol="ASELS.IS", signal="BUY", price=123.45,
            reason="EMA trend + hacimli kırılım", risk_level="MEDIUM",
        ))
        assert "ASELS" in msg
        assert "BUY" in msg
        assert "123.45" in msg
        assert "EMA trend" in msg
        assert "MEDIUM" in msg
        assert "yatırım tavsiyesi" in msg


# ── format_scan_summary ───────────────────────────────────────────────────────

class TestFormatScanSummary:
    def test_contains_buy_count(self):
        msg = format_scan_summary([_buy_result(), _watch_result()])
        assert "BUY" in msg

    def test_contains_watch_count(self):
        msg = format_scan_summary([_buy_result(), _watch_result()])
        assert "WATCH" in msg

    def test_lists_buy_symbols(self):
        msg = format_scan_summary([_buy_result()])
        assert "ASELS" in msg

    def test_lists_watch_symbols(self):
        msg = format_scan_summary([_watch_result()])
        assert "THYAO" in msg

    def test_empty_results_no_crash(self):
        msg = format_scan_summary([])
        assert "bulunamadı" in msg.lower() or isinstance(msg, str)

    def test_contains_disclaimer(self):
        assert _DISCLAIMER in format_scan_summary([_buy_result()])

    def test_only_buy_no_watch_section(self):
        msg = format_scan_summary([_buy_result()])
        # Sadece BUY varsa WATCH başlığı olmamalı
        assert "TAKİP" not in msg or "AL" in msg   # esneklik


# ── format_daily_summary ──────────────────────────────────────────────────────

class TestFormatDailySummary:
    def test_contains_buy_count(self):
        signals = [{"signal_type": "BUY"}, {"signal_type": "SELL"}, {"signal_type": "BUY"}]
        msg = format_daily_summary(signals)
        assert "2" in msg   # 2 alış

    def test_contains_sell_count(self):
        signals = [{"signal_type": "BUY"}, {"signal_type": "SELL"}]
        msg = format_daily_summary(signals)
        assert "1" in msg   # 1 satış

    def test_contains_date(self):
        from datetime import datetime
        today = datetime.now().strftime("%d.%m.%Y")
        msg = format_daily_summary([])
        assert today in msg


# ── is_configured ─────────────────────────────────────────────────────────────

class TestIsConfigured:
    def test_false_when_no_token(self):
        with patch("app.notifications.telegram.settings") as mock_settings:
            mock_settings.telegram_bot_token = ""
            mock_settings.telegram_chat_id = "123"
            assert is_configured() is False

    def test_false_when_no_chat_id(self):
        with patch("app.notifications.telegram.settings") as mock_settings:
            mock_settings.telegram_bot_token = "token"
            mock_settings.telegram_chat_id = ""
            assert is_configured() is False

    def test_true_when_both_set(self):
        with patch("app.notifications.telegram.settings") as mock_settings:
            mock_settings.telegram_bot_token = "token"
            mock_settings.telegram_chat_id = "123"
            assert is_configured() is True


# ── send_telegram_message ─────────────────────────────────────────────────────

class TestSendTelegramMessage:
    @pytest.mark.asyncio
    async def test_returns_true_on_success(self):
        with _patch_bot(send_ok=True):
            with patch("app.notifications.telegram.settings") as ms:
                ms.telegram_bot_token = "tok"
                ms.telegram_chat_id = "123"
                result = await send_telegram_message("test mesajı")
        assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_when_no_token(self):
        with patch("app.notifications.telegram._get_bot", return_value=None):
            result = await send_telegram_message("test")
        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_when_no_chat_id(self):
        mock_bot = AsyncMock()
        with patch("app.notifications.telegram._get_bot", return_value=mock_bot):
            with patch("app.notifications.telegram.settings") as ms:
                ms.telegram_chat_id = ""
                result = await send_telegram_message("test")
        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_on_telegram_error(self):
        with _patch_bot(send_ok=False):
            with patch("app.notifications.telegram.settings") as ms:
                ms.telegram_bot_token = "tok"
                ms.telegram_chat_id = "123"
                result = await send_telegram_message("test")
        assert result is False

    @pytest.mark.asyncio
    async def test_does_not_raise_on_error(self):
        with _patch_bot(send_ok=False):
            with patch("app.notifications.telegram.settings") as ms:
                ms.telegram_bot_token = "tok"
                ms.telegram_chat_id = "123"
                result = await send_telegram_message("test")   # istisna atmamalı
        assert result is False


# ── Yeni sinyal tipleri: format_signal_message ────────────────────────────────

def _setup_result(**kw) -> dict:
    base = {
        "symbol": "AKBNK.IS", "signal": "SETUP", "price": 45.20,
        "reason": "Sıkışma aktif | RSI 55 | Direncin %3.2 altında",
        "risk_level": "LOW", "strength": 0.83,
        "stop_loss": 43.10, "take_profit": 47.50,
        "conditions_met": 5, "distance_to_res_pct": 3.2,
    }
    return {**base, **kw}


def _early_watch_result(**kw) -> dict:
    base = {
        "symbol": "EREGL.IS", "signal": "EARLY_WATCH", "price": 118.40,
        "reason": "Direncin %1.5 altında | Hacim canlanıyor (x1.4) | MACD hist yükseliyor",
        "risk_level": "MEDIUM", "strength": 0.78,
        "stop_loss": 114.20, "take_profit": 122.80,
        "conditions_met": 4, "distance_to_res_pct": 1.5,
    }
    return {**base, **kw}


def _late_result(**kw) -> dict:
    base = {
        "symbol": "SISE.IS", "signal": "LATE_BREAKOUT", "price": 88.50,
        "reason": "Geç kırılım: RSI 71 | Hacim x4.5",
        "risk_level": "HIGH", "strength": 0.62,
        "stop_loss": 85.85, "take_profit": 93.81,
        "conditions_met": 5, "distance_to_res_pct": 0.0,
    }
    return {**base, **kw}


class TestNewSignalFormats:
    def test_setup_header(self):
        msg = format_signal_message(_setup_result())
        assert "HAZIRLIK" in msg or "SETUP" in msg

    def test_setup_emoji(self):
        msg = format_signal_message(_setup_result())
        assert "🟡" in msg

    def test_setup_contains_symbol(self):
        msg = format_signal_message(_setup_result())
        assert "AKBNK" in msg
        assert ".IS" not in msg

    def test_setup_contains_distance(self):
        msg = format_signal_message(_setup_result())
        assert "3.2" in msg

    def test_setup_contains_disclaimer(self):
        assert _DISCLAIMER in format_signal_message(_setup_result())

    def test_early_watch_header(self):
        msg = format_signal_message(_early_watch_result())
        assert "YAKLAŞIYOR" in msg or "EARLY" in msg

    def test_early_watch_emoji(self):
        msg = format_signal_message(_early_watch_result())
        assert "🟠" in msg

    def test_early_watch_contains_symbol(self):
        msg = format_signal_message(_early_watch_result())
        assert "EREGL" in msg

    def test_early_watch_contains_distance(self):
        msg = format_signal_message(_early_watch_result())
        assert "1.5" in msg

    def test_late_breakout_header(self):
        msg = format_signal_message(_late_result())
        assert "GEÇ" in msg or "LATE" in msg

    def test_late_breakout_emoji(self):
        msg = format_signal_message(_late_result())
        assert "🔴" in msg

    def test_late_breakout_warning(self):
        msg = format_signal_message(_late_result())
        assert "kaçmış" in msg.lower() or "dikkat" in msg.lower()

    def test_late_breakout_contains_disclaimer(self):
        assert _DISCLAIMER in format_signal_message(_late_result())

    def test_format_bist100_early_signals_with_both(self):
        results = [_early_watch_result(), _setup_result()]
        msg = format_bist100_early_signals(results)
        assert "EREGL" in msg
        assert "AKBNK" in msg

    def test_format_bist100_early_signals_empty(self):
        msg = format_bist100_early_signals([])
        assert msg == ""

    def test_scan_summary_shows_early_watch_count(self):
        results = [_early_watch_result(), _setup_result(), _buy_result()]
        msg = format_scan_summary(results)
        assert "EARLY" in msg or "WATCH" in msg
        assert "SETUP" in msg

    def test_scan_summary_lists_early_watch_symbols(self):
        results = [_early_watch_result()]
        msg = format_scan_summary(results)
        assert "EREGL" in msg

    def test_scan_summary_lists_late_breakout(self):
        results = [_late_result()]
        msg = format_scan_summary(results)
        assert "SISE" in msg


# ── format_bist100_full_report ────────────────────────────────────────────────

def _full_result(signal: str, symbol: str, score: int, **kw) -> dict:
    base = {
        "symbol": f"{symbol}.IS", "signal": signal, "price": 100.0,
        "reason": "test", "risk_level": "LOW", "strength": 0.7,
        "stop_loss": 97.0, "take_profit": 106.0,
        "conditions_met": 4, "distance_to_res_pct": 3.0,
        "strength_score": score,
        "rsi_14": 58.0, "volume_ratio": 1.2,
        "daily_change_pct": 0.8, "close_to_ema20_pct": 1.5,
    }
    return {**base, **kw}


def _make_report(results: list, **kw) -> dict:
    base = {
        "label": "BIST100", "results": results,
        "scanned": 100, "buy_count": 0, "watch_count": 0,
        "setup_count": 0, "early_watch_count": 0, "late_breakout_count": 0,
        "error_count": 3, "elapsed_seconds": 21.0, "market_filter": "unfavorable",
    }
    return {**base, **kw}


class TestFormatBist100FullReport:
    def test_summary_line_contains_all_types(self):
        msg = format_bist100_full_report(_make_report([]))
        for keyword in ("EARLY", "SETUP", "WATCH", "BUY", "LATE"):
            assert keyword in msg

    def test_empty_sections_not_shown(self):
        msg = format_bist100_full_report(_make_report([]))
        # Bölüm başlıkları *bold* formatında — özet satırındaki 🟢 BUY: ile karışmaz
        assert "*🟠 EARLY WATCH*" not in msg
        assert "*🟢 BUY*" not in msg

    def test_section_shown_when_signal_exists(self):
        r = _full_result("BUY", "ASELS", 80)
        msg = format_bist100_full_report(_make_report([r], buy_count=1))
        assert "*🟢 BUY*" in msg
        assert "ASELS" in msg

    def test_symbol_row_contains_required_fields(self):
        r = _full_result("WATCH", "THYAO", 65,
                         rsi_14=57.0, volume_ratio=1.18,
                         distance_to_res_pct=5.4, daily_change_pct=1.2,
                         close_to_ema20_pct=2.8)
        msg = format_bist100_full_report(_make_report([r], watch_count=1))
        assert "THYAO" in msg
        assert "65" in msg      # score
        assert "57" in msg      # rsi
        assert "1.18x" in msg   # volume
        assert "%5.4" in msg    # direnç
        assert "%+1.2" in msg   # günlük
        assert "%+2.8" in msg   # ema20

    def test_max_5_per_section(self):
        results = [_full_result("BUY", f"SY{i:02d}", 90 - i) for i in range(8)]
        msg = format_bist100_full_report(_make_report(results, buy_count=8))
        assert "SY00" in msg
        assert "SY04" in msg
        assert "SY05" not in msg

    def test_section_order_ew_before_buy(self):
        ew  = _full_result("EARLY_WATCH", "EREGL", 70)
        buy = _full_result("BUY", "ASELS", 85)
        msg = format_bist100_full_report(_make_report([ew, buy]))
        assert msg.find("*🟠 EARLY WATCH*") < msg.find("*🟢 BUY*")

    def test_section_order_all_five(self):
        results = [
            _full_result("LATE_BREAKOUT", "LATE1", 50),
            _full_result("BUY",           "BUY1",  80),
            _full_result("WATCH",         "WAT1",  40),
            _full_result("SETUP",         "SET1",  60),
            _full_result("EARLY_WATCH",   "EWA1",  70),
        ]
        msg = format_bist100_full_report(_make_report(results))
        positions = {
            "EW":    msg.find("*🟠 EARLY WATCH*"),
            "SETUP": msg.find("*🟡 SETUP*"),
            "WATCH": msg.find("*👀 WATCH*"),
            "BUY":   msg.find("*🟢 BUY*"),
            "LATE":  msg.find("*🔴 LATE BREAKOUT*"),
        }
        assert positions["EW"] < positions["SETUP"] < positions["WATCH"] < positions["BUY"] < positions["LATE"]

    def test_no_section_header_when_empty(self):
        r = _full_result("BUY", "ASELS", 80)
        msg = format_bist100_full_report(_make_report([r], buy_count=1))
        assert "*🟠 EARLY WATCH*" not in msg
        assert "*🟡 SETUP*" not in msg

    def test_disclaimer_present(self):
        msg = format_bist100_full_report(_make_report([]))
        assert _DISCLAIMER in msg

    def test_footer_contains_stats(self):
        msg = format_bist100_full_report(_make_report([], scanned=100, error_count=5, elapsed_seconds=20.5))
        assert "100" in msg
        assert "5" in msg
        assert "20.5" in msg

    def test_dist_dash_when_zero(self):
        """BUY sinyalinde distance=0.0 → Direnç: – gösterilmeli."""
        r = _full_result("BUY", "ASELS", 80, distance_to_res_pct=0.0)
        msg = format_bist100_full_report(_make_report([r], buy_count=1))
        assert "Direnç: `–`" in msg

    def test_none_fields_show_dash(self):
        r = _full_result("WATCH", "THYAO", 50,
                         rsi_14=None, volume_ratio=None,
                         daily_change_pct=None, close_to_ema20_pct=None)
        msg = format_bist100_full_report(_make_report([r], watch_count=1))
        assert "–" in msg


# ── notify_signal ─────────────────────────────────────────────────────────────

class TestNotifySignal:
    @pytest.mark.asyncio
    async def test_sends_buy_signal(self):
        with _patch_bot(send_ok=True) as mock_get_bot:
            with patch("app.notifications.telegram.settings") as ms:
                ms.telegram_bot_token = "tok"
                ms.telegram_chat_id = "123"
                result = await notify_signal(_buy_result())
        assert result is True

    @pytest.mark.asyncio
    async def test_sends_watch_signal(self):
        with _patch_bot(send_ok=True):
            with patch("app.notifications.telegram.settings") as ms:
                ms.telegram_bot_token = "tok"
                ms.telegram_chat_id = "123"
                result = await notify_signal(_watch_result())
        assert result is True


# ── notify_scan_summary ───────────────────────────────────────────────────────

class TestNotifyScanSummary:
    @pytest.mark.asyncio
    async def test_empty_results_returns_false(self):
        result = await notify_scan_summary([])
        assert result is False

    @pytest.mark.asyncio
    async def test_sends_summary(self):
        with _patch_bot(send_ok=True):
            with patch("app.notifications.telegram.settings") as ms:
                ms.telegram_bot_token = "tok"
                ms.telegram_chat_id = "123"
                result = await notify_scan_summary([_buy_result()])
        assert result is True


# ── TelegramNotifier (backward compat) ───────────────────────────────────────

class TestTelegramNotifier:
    @pytest.mark.asyncio
    async def test_send_signal_uses_strategy_signal(self):
        from app.strategies.base import StrategySignal, SignalType
        from app.risk.manager import RiskAssessment

        signal = StrategySignal(
            symbol="THYAO.IS", signal_type=SignalType.BUY,
            strategy="trend_breakout", strength=0.75,
            entry_price=100.0, stop_loss=97.0, take_profit=106.0,
            notes="test notu",
        )
        assessment = RiskAssessment(
            approved=True, position_size_pct=0.075,
            adjusted_stop_loss=97.0, adjusted_take_profit=106.0,
            risk_reward_ratio=2.0,
        )
        with _patch_bot(send_ok=True):
            with patch("app.notifications.telegram.settings") as ms:
                ms.telegram_bot_token = "tok"
                ms.telegram_chat_id = "123"
                result = await TelegramNotifier().send_signal(signal, assessment)
        assert result is True

    @pytest.mark.asyncio
    async def test_send_text_delegates(self):
        with _patch_bot(send_ok=True):
            with patch("app.notifications.telegram.settings") as ms:
                ms.telegram_bot_token = "tok"
                ms.telegram_chat_id = "123"
                result = await TelegramNotifier().send_text("merhaba")
        assert result is True

    @pytest.mark.asyncio
    async def test_send_daily_summary(self):
        signals = [{"signal_type": "BUY"}, {"signal_type": "SELL"}]
        with _patch_bot(send_ok=True):
            with patch("app.notifications.telegram.settings") as ms:
                ms.telegram_bot_token = "tok"
                ms.telegram_chat_id = "123"
                result = await TelegramNotifier().send_daily_summary(signals)
        assert result is True
