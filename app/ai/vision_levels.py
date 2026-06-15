"""
Claude Vision ile Stop/Hedef Teyidi
===================================
Algoritmik (ATR + yapı) seviyeler işaretlenmiş mum grafiğini Claude'a gönderip
seviyeleri grafik üzerinden teyit/düzeltmesini ister. Tamamen opsiyonel ve
fail-open: anahtar yoksa, SDK kurulu değilse veya çağrı hata/refusal verirse
None döner ve pipeline algoritmik seviyelerle devam eder.

Model: claude-opus-4-8 (vision + structured outputs + adaptive thinking).
Yalnızca kesinleşmiş BUY/LATE sinyalleri için, scanner tarafından çağrılır.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Optional

from loguru import logger

from app.config import settings

try:
    import anthropic
    _SDK_AVAILABLE = True
except Exception:  # ImportError
    _SDK_AVAILABLE = False

_MODEL = "claude-opus-4-8"
_MAX_TOKENS = 2048

_SYSTEM_PROMPT = (
    "Sen bir teknik analiz asistanısın. Sana bir BIST hissesinin günlük mum "
    "grafiği veriliyor; grafikte algoritmik olarak hesaplanmış stop-loss "
    "(kırmızı) ve take-profit (yeşil) çizgileri işaretli. Görevin: grafikteki "
    "gerçek destek/direnç ve swing dip/tepe yapısına bakarak bu seviyeleri "
    "teyit etmek veya en yakın anlamlı yapı noktasına göre düzeltmek. "
    "Stop her zaman giriş fiyatının altında, hedef üstünde olmalı. "
    "Spekülasyon yapma, yalnızca grafikte görünen yapıya dayan. "
    "Gerekçeyi tek cümleyle, Türkçe ve kısa yaz."
)

_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "stop_loss":   {"type": "number"},
        "take_profit": {"type": "number"},
        "rationale":   {"type": "string"},
        "confidence":  {"type": "string", "enum": ["low", "medium", "high"]},
    },
    "required": ["stop_loss", "take_profit", "rationale", "confidence"],
    "additionalProperties": False,
}


@dataclass
class AILevels:
    stop_loss:   float
    take_profit: float
    rationale:   str
    confidence:  str


def _build_user_text(
    symbol: str, close: float,
    atr_sl: Optional[float], atr_tp: Optional[float],
    struct_sl: Optional[float], struct_tp: Optional[float],
) -> str:
    def fmt(v: Optional[float]) -> str:
        return f"{v:.2f}" if v is not None else "yok"
    return (
        f"Hisse: {symbol.removesuffix('.IS')}\n"
        f"Güncel fiyat (giriş): {close:.2f} TL\n"
        f"Algoritmik seviyeler — referans:\n"
        f"  • ATR stop: {fmt(atr_sl)} | ATR hedef: {fmt(atr_tp)}\n"
        f"  • Yapı stop: {fmt(struct_sl)} | Yapı hedef: {fmt(struct_tp)}\n"
        f"Grafiği incele ve stop_loss / take_profit seviyelerini ver."
    )


def confirm_levels(
    symbol: str,
    chart_png: bytes,
    close: float,
    atr_sl: Optional[float] = None,
    atr_tp: Optional[float] = None,
    struct_sl: Optional[float] = None,
    struct_tp: Optional[float] = None,
) -> Optional[AILevels]:
    """
    Claude'dan grafik bazlı stop/hedef teyidi alır. Üretilemezse None (fail-open).

    Çağrı koşulları (hiçbiri sağlanmazsa sessizce None):
      • enable_ai_levels açık, anthropic_api_key dolu, SDK kurulu, grafik mevcut.
    """
    if not settings.enable_ai_levels:
        return None
    if not _SDK_AVAILABLE:
        logger.warning("anthropic SDK kurulu değil — AI seviye teyidi atlandı")
        return None
    if not settings.anthropic_api_key:
        logger.debug("ANTHROPIC_API_KEY boş — AI seviye teyidi atlandı")
        return None
    if not chart_png:
        return None

    try:
        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        b64 = base64.standard_b64encode(chart_png).decode("utf-8")
        response = client.messages.create(
            model=_MODEL,
            max_tokens=_MAX_TOKENS,
            thinking={"type": "adaptive"},
            system=_SYSTEM_PROMPT,
            output_config={"format": {"type": "json_schema", "schema": _OUTPUT_SCHEMA}},
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": _build_user_text(
                            symbol, close, atr_sl, atr_tp, struct_sl, struct_tp
                        ),
                    },
                ],
            }],
        )

        if response.stop_reason == "refusal":
            logger.warning(f"{symbol} AI seviye teyidi reddedildi (refusal)")
            return None

        text = next((b.text for b in response.content if b.type == "text"), None)
        if not text:
            return None
        data = json.loads(text)

        sl = float(data["stop_loss"])
        tp = float(data["take_profit"])
        # Tutarlılık kapısı — mantıksız seviyeyi kabul etme.
        if not (sl < close < tp):
            logger.warning(
                f"{symbol} AI seviyesi tutarsız (sl={sl}, close={close}, tp={tp}) — atlandı"
            )
            return None

        return AILevels(
            stop_loss=round(sl, 2),
            take_profit=round(tp, 2),
            rationale=str(data.get("rationale", "")).strip(),
            confidence=str(data.get("confidence", "")).strip(),
        )
    except Exception as exc:
        logger.warning(f"{symbol} AI seviye teyidi başarısız: {exc}")
        return None
