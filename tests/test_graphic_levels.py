"""
Grafik-bazlı seviyelerin scanner entegrasyonu ve Telegram gösterimi testleri.
"""

import numpy as np
import pandas as pd

from app.config import settings
from app.notifications.telegram import _levels_block, _best_level, format_signal_message
from app.signals import scanner as scn
from app.ai.vision_levels import AILevels


def _indicator_df() -> pd.DataFrame:
    close = np.linspace(40, 55, 80) + np.sin(np.linspace(0, 12, 80)) * 1.5
    idx = pd.date_range("2024-01-01", periods=80, freq="D")
    df = pd.DataFrame(
        {"open": close - 0.2, "high": close + 0.5, "low": close - 0.5,
         "close": close, "volume": 1e6},
        index=idx,
    )
    df["atr_14"] = 0.6
    df["resistance_20"] = df["high"].rolling(20, min_periods=1).max()
    return df


# ── Telegram seviye bloğu ─────────────────────────────────────────────────────

class TestLevelsBlock:
    def test_ai_preferred_and_tagged(self):
        # AI varsa yalnız AI seviyeleri 🤖 ile gösterilir; ATR/yapı görünmez.
        block = _levels_block({
            "stop_loss": 43.0, "take_profit": 49.0,
            "struct_stop_loss": 42.5, "struct_take_profit": 48.5,
            "ai_stop_loss": 41.80, "ai_take_profit": 48.00,
            "ai_rationale": "20 günlük dirence kadar alan açık",
        })
        assert "🤖 " in block
        assert "`41.80 TL`" in block and "`48.00 TL`" in block
        assert "20 günlük dirence kadar alan açık" in block
        # ATR/yapı değerleri görünmemeli
        assert "43.00" not in block and "42.50" not in block

    def test_falls_back_to_struct_when_no_ai(self):
        block = _levels_block({
            "stop_loss": 43.0, "take_profit": 49.0,
            "struct_stop_loss": 42.5, "struct_take_profit": 48.5,
        })
        assert "`42.50 TL`" in block and "`48.50 TL`" in block
        assert "🤖" not in block
        assert "43.00" not in block  # ATR'ye düşmedi

    def test_falls_back_to_atr_when_nothing_else(self):
        block = _levels_block({"stop_loss": 43.0, "take_profit": 49.0})
        assert "`43.00 TL`" in block and "`49.00 TL`" in block
        assert "🤖" not in block

    def test_empty_when_no_levels(self):
        assert _levels_block({}) == ""

    def test_best_level_priority(self):
        # AI > yapı > ATR
        assert _best_level({"stop_loss": 1, "struct_stop_loss": 2, "ai_stop_loss": 3},
                           "stop_loss", "struct_stop_loss", "ai_stop_loss") == 3
        assert _best_level({"stop_loss": 1, "struct_stop_loss": 2},
                           "stop_loss", "struct_stop_loss", "ai_stop_loss") == 2
        assert _best_level({"stop_loss": 1},
                           "stop_loss", "struct_stop_loss", "ai_stop_loss") == 1


class TestBuyMessageWithLevels:
    def test_buy_message_shows_only_ai(self):
        result = {
            "signal": "BUY", "symbol": "ASELS.IS", "price": 45.0,
            "reason": "Kırılım", "risk_level": "MEDIUM", "strength": 0.8,
            "stop_loss": 43.0, "take_profit": 49.0,
            "struct_stop_loss": 42.5, "struct_take_profit": 48.5,
            "ai_stop_loss": 42.8, "ai_take_profit": 48.9, "ai_rationale": "test",
        }
        msg = format_signal_message(result)
        assert "🤖 Stop: `42.80 TL`" in msg
        assert "🤖 Hedef: `48.90 TL`" in msg
        # ATR/yapı seviyeleri gösterilmemeli
        assert "43.00" not in msg and "42.50" not in msg


# ── Scanner _graphic_levels ───────────────────────────────────────────────────

class TestGraphicLevels:
    def test_structure_only_when_ai_disabled(self, monkeypatch):
        monkeypatch.setattr(settings, "enable_ai_levels", False)
        monkeypatch.setattr(settings, "enable_signal_charts", False)
        df = _indicator_df()
        price = float(df["close"].iloc[-1])
        out = scn._graphic_levels("TEST.IS", df, price, 43.0, 49.0)
        # Yapı seviyeleri hesaplanır; AI alanları None.
        assert out["ai_stop_loss"] is None and out["ai_take_profit"] is None
        assert out["chart_b64"] is None
        assert "struct_stop_loss" in out

    def test_ai_levels_filled_when_enabled(self, monkeypatch):
        monkeypatch.setattr(settings, "enable_ai_levels", True)
        monkeypatch.setattr(settings, "enable_signal_charts", False)
        # Grafik ve vision çağrılarını mock'la (ağ/anahtar gerekmez).
        monkeypatch.setattr(
            "app.charts.render.render_signal_chart",
            lambda *a, **k: b"PNGBYTES",
        )
        monkeypatch.setattr(
            "app.ai.vision_levels.confirm_levels",
            lambda *a, **k: AILevels(stop_loss=42.8, take_profit=48.9,
                                     rationale="yapay zeka teyidi", confidence="high"),
        )
        df = _indicator_df()
        price = float(df["close"].iloc[-1])
        out = scn._graphic_levels("TEST.IS", df, price, 43.0, 49.0)
        assert out["ai_stop_loss"] == 42.8
        assert out["ai_take_profit"] == 48.9
        assert out["ai_rationale"] == "yapay zeka teyidi"

    def test_chart_b64_set_when_charts_enabled(self, monkeypatch):
        monkeypatch.setattr(settings, "enable_ai_levels", False)
        monkeypatch.setattr(settings, "enable_signal_charts", True)
        monkeypatch.setattr(
            "app.charts.render.render_signal_chart",
            lambda *a, **k: b"PNGBYTES",
        )
        df = _indicator_df()
        price = float(df["close"].iloc[-1])
        out = scn._graphic_levels("TEST.IS", df, price, 43.0, 49.0)
        assert out["chart_b64"] is not None  # base64 PNGBYTES

    def test_failopen_when_render_raises(self, monkeypatch):
        monkeypatch.setattr(settings, "enable_ai_levels", True)
        monkeypatch.setattr(settings, "enable_signal_charts", False)

        def _boom(*a, **k):
            raise RuntimeError("render patladı")

        monkeypatch.setattr("app.charts.render.render_signal_chart", _boom)
        df = _indicator_df()
        price = float(df["close"].iloc[-1])
        out = scn._graphic_levels("TEST.IS", df, price, 43.0, 49.0)
        # Hata yutulur; yapı seviyeleri yine döner, AI None.
        assert out["ai_stop_loss"] is None
