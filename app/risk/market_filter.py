"""
XU100 Endeks Filtresi
=====================
Borsa İstanbul genelinde trend kontrolü yapar.
Close > EMA50 → piyasa uygun (AL sinyaline izin ver)
Close < EMA50 → piyasa elverişsiz (AL sinyalini engelle)

Veri çekilemezse filtre devredışı bırakılır ("market_filter_unavailable");
sinyal üretimini durdurmaz, yalnızca uyarı kaydeder.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import pandas_ta_classic as ta
from loguru import logger

from app.data.fetcher import fetch_symbol_data, FetchError

# ── Sabitler ──────────────────────────────────────────────────────────────────

_XU100_SYMBOL  = "XU100.IS"   # yfinance sembolü
_EMA_PERIOD    = 50
_CACHE_TTL_SEC = 3600          # 1 saat — gün içi taramalarda tekrar çekmez

# ── Durum sabitleri ───────────────────────────────────────────────────────────

STATUS_FAVORABLE   = "favorable"
STATUS_UNFAVORABLE = "unfavorable"
STATUS_UNAVAILABLE = "market_filter_unavailable"

# ── Sonuç nesnesi ─────────────────────────────────────────────────────────────

@dataclass
class MarketFilterResult:
    favorable: bool                  # True → AL sinyaline izin ver
    status: str                      # STATUS_* sabitlerinden biri
    reason: str                      # insan okunabilir açıklama
    xu100_close: Optional[float] = None
    xu100_ema50: Optional[float] = None
    cached: bool = False             # True → önbellekten geldi

    @property
    def blocks_buy(self) -> bool:
        """AL sinyalini engelliyor mu?"""
        return self.status == STATUS_UNFAVORABLE

    def __str__(self) -> str:
        tag = "[CACHE]" if self.cached else ""
        return f"MarketFilter({self.status}{tag}): {self.reason}"


# ── Önbellek (bellek içi, TTL'li) ─────────────────────────────────────────────

@dataclass
class _Cache:
    result: Optional[MarketFilterResult] = None
    expires_at: float = 0.0

    def get(self) -> Optional[MarketFilterResult]:
        if self.result is None or time.monotonic() > self.expires_at:
            return None
        cached = MarketFilterResult(
            favorable=self.result.favorable,
            status=self.result.status,
            reason=self.result.reason,
            xu100_close=self.result.xu100_close,
            xu100_ema50=self.result.xu100_ema50,
            cached=True,
        )
        return cached

    def set(self, result: MarketFilterResult, ttl: float = _CACHE_TTL_SEC):
        self.result = result
        self.expires_at = time.monotonic() + ttl

    def invalidate(self):
        self.result = None
        self.expires_at = 0.0


_cache = _Cache()

# ── Yardımcı ─────────────────────────────────────────────────────────────────

def _unavailable(reason: str) -> MarketFilterResult:
    logger.warning(f"XU100 endeks filtresi devre dışı: {reason}")
    return MarketFilterResult(
        favorable=True,
        status=STATUS_UNAVAILABLE,
        reason=reason,
    )


# ── Ana fonksiyon ─────────────────────────────────────────────────────────────

def is_market_favorable(
    force_refresh: bool = False,
    period: str = "6mo",
) -> MarketFilterResult:
    """
    XU100 endeksine göre piyasa koşulunu değerlendirir.

    Args:
        force_refresh: True → önbelleği görmezden gel, yeniden çek.
        period:        yfinance veri periyodu (varsayılan "6mo").

    Returns:
        MarketFilterResult — favorable, status, reason, fiyat değerleri.

    Hiçbir zaman istisna fırlatmaz; hata durumunda STATUS_UNAVAILABLE döner.
    """
    # ── Önbellek kontrolü ─────────────────────────────────────────────────────
    if not force_refresh:
        cached = _cache.get()
        if cached is not None:
            logger.debug(f"XU100 filtresi önbellekten: {cached.status}")
            return cached

    # ── Veri çekme ────────────────────────────────────────────────────────────
    try:
        result = fetch_symbol_data(_XU100_SYMBOL, period=period, interval="1d")
        df = result.df
    except FetchError as exc:
        return _unavailable(f"Veri çekme hatası: {exc}")
    except Exception as exc:
        return _unavailable(f"Beklenmeyen hata: {exc}")

    # ── EMA50 hesapla ─────────────────────────────────────────────────────────
    if len(df) < _EMA_PERIOD:
        return _unavailable(
            f"Yetersiz veri: {len(df)} bar < EMA{_EMA_PERIOD} için gereken {_EMA_PERIOD}"
        )

    ema_series = ta.ema(df["close"], length=_EMA_PERIOD)
    if ema_series is None or ema_series.dropna().empty:
        return _unavailable("EMA hesaplanamadı")

    xu100_close = float(df["close"].iloc[-1])
    xu100_ema50 = float(ema_series.iloc[-1])

    # ── Karar ─────────────────────────────────────────────────────────────────
    pct_diff = (xu100_close - xu100_ema50) / xu100_ema50 * 100

    if xu100_close > xu100_ema50:
        filter_result = MarketFilterResult(
            favorable=True,
            status=STATUS_FAVORABLE,
            reason=(
                f"XU100 {xu100_close:.0f} > EMA50 {xu100_ema50:.0f} "
                f"(+{pct_diff:.1f}%) — piyasa yükseliş trendinde"
            ),
            xu100_close=round(xu100_close, 2),
            xu100_ema50=round(xu100_ema50, 2),
        )
    else:
        filter_result = MarketFilterResult(
            favorable=False,
            status=STATUS_UNFAVORABLE,
            reason=(
                f"XU100 {xu100_close:.0f} < EMA50 {xu100_ema50:.0f} "
                f"({pct_diff:.1f}%) — piyasa düşüş trendinde, AL sinyalleri engellendi"
            ),
            xu100_close=round(xu100_close, 2),
            xu100_ema50=round(xu100_ema50, 2),
        )

    _cache.set(filter_result)
    logger.info(str(filter_result))
    return filter_result


def invalidate_cache():
    """Önbelleği temizler — test veya zorla yenileme için."""
    _cache.invalidate()
