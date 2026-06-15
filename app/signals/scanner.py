"""
Piyasa Tarayıcı
===============
Sembolleri tarar; aşağıdaki 3 sinyal tipini döner:

  EARLY_WATCH  → Kırılıma yakın, hacim canlanıyor, MACD ve fiyat yükseliyor
  BUY          → Trend kırılımı — tüm koşullar sağlandı, geç değil
  LATE_BREAKOUT → Tüm BUY koşulları sağlandı ancak geç kalınma işaretleri var

Tarama/görüntüleme sırası: EARLY_WATCH → BUY → LATE_BREAKOUT

Not: pre_breakout_squeeze içindeki SETUP koşulları EARLY_WATCH için
ara basamak olarak hesaplanır ama ayrı bir sinyal olarak yayınlanmaz.

BIST100 Tarama:
  scan_bist100() / scan_bist50() / scan_bist30()
  - ThreadPoolExecutor ile paralel veri çekimi
  - Likidite filtresi: ortalama günlük TL hacmi ≥ 50M TL
  - strength_score (0-100): EMA+Hacim+RSI+Breakout+MACD puanlaması
  - 1 saatlik önbellek
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from threading import Lock
from typing import Optional


class _EWDiag:
    """Tarama boyunca EARLY_WATCH koşul başarısızlıklarını sayar."""

    def __init__(self):
        self._lock   = Lock()
        self._counts: dict[str, int] = {}

    def record(self, ew_flags: dict):
        with self._lock:
            for flag, met in ew_flags.items():
                if not met:
                    self._counts[flag] = self._counts.get(flag, 0) + 1

    def top_failures(self, n: int = 5) -> list[tuple[str, int]]:
        return sorted(self._counts.items(), key=lambda x: -x[1])[:n]

    def has_data(self) -> bool:
        return bool(self._counts)

import time as _time

import pandas as pd
from loguru import logger

from app.config import settings
from app.data.fetcher import fetch_symbol_data, fetch_symbol_data_cached, FetchError
from app.indicators.technical import add_indicators
from app.risk.market_filter import (
    is_market_favorable,
    MarketFilterResult,
    STATUS_UNAVAILABLE,
    STATUS_UNFAVORABLE,
)
from app.strategies.trend_breakout import generate_signal as tb_generate
from app.strategies.pre_breakout_squeeze import generate_setup_signal as pbs_generate

# ── Eşikler ───────────────────────────────────────────────────────────────────

_BUY_MIN_STRENGTH       = 0.0

_LIQUIDITY_MIN_TL       = 50_000_000
_SCORE_VOLUME_MULT      = 2.0
_SCORE_RSI_LOW          = 55
_SCORE_RSI_HIGH         = 70
_SCORE_EMA_GAP          = 0.01
_SCORE_BREAKOUT_MARGIN  = 0.01

_CACHE_TTL              = timedelta(hours=1)
_PARALLEL_MAX_WORKERS   = 3
_PARALLEL_BATCH_DELAY   = 0.3

# Görüntüleme sırası: küçük sayı = önce
_SIGNAL_ORDER = {
    "EARLY_WATCH":   0,
    "BUY":           1,
    "LATE_BREAKOUT": 2,
}


# ── Önbellek (thread-safe) ───────────────────────────────────────────────────

@dataclass
class _CacheEntry:
    result: Optional["ScanResult"]
    cached_at: datetime

_cache: dict[str, _CacheEntry] = {}
_cache_lock = Lock()


def _cache_get(symbol: str) -> tuple[bool, Optional["ScanResult"]]:
    with _cache_lock:
        entry = _cache.get(symbol)
    if entry is None:
        return False, None
    if datetime.now() - entry.cached_at > _CACHE_TTL:
        return False, None
    return True, entry.result


def _cache_set(symbol: str, result: Optional["ScanResult"]):
    with _cache_lock:
        _cache[symbol] = _CacheEntry(result=result, cached_at=datetime.now())


def clear_scan_cache():
    """Tüm önbelleği temizler (test veya zorla yenileme için)."""
    with _cache_lock:
        _cache.clear()


# ── Sonuç nesnesi ─────────────────────────────────────────────────────────────

@dataclass
class ScanResult:
    symbol:               str
    signal:               str    # "EARLY_WATCH"|"BUY"|"LATE_BREAKOUT"
    price:                float
    reason:               str
    risk_level:           str
    strength:             float
    strategy:             str
    stop_loss:            Optional[float]
    take_profit:          Optional[float]
    market_filter:        str
    conditions_met:       int
    distance_to_res_pct:  Optional[float]
    strength_score:       int = 0
    score_reasons:        list = field(default_factory=list)
    rsi_14:               Optional[float] = None
    volume_ratio:         Optional[float] = None
    daily_change_pct:     Optional[float] = None
    close_to_ema20_pct:   Optional[float] = None
    scanned_at:           str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def to_dict(self) -> dict:
        return asdict(self)


# ── Strength Score (0-100) ────────────────────────────────────────────────────
#
# TEK KURAL: strength_score = round(strategy.strength × 100)
# Stratejinin sürekli güç skoru (trend_breakout._signal_strength / pbs.strength)
# tek sıralama kaynağıdır. Aşağıdaki fonksiyonlar yalnızca insan-okunur
# GEREKÇE üretir — sayıyı belirlemezler. Böylece "iki ayrı puanlama" çakışması
# ve sıralama tutarsızlığı ortadan kalkar.

def _score_from_strength(strength: float) -> int:
    return max(0, min(100, int(round((strength or 0.0) * 100))))


def _breakout_reasons(details: dict, df: pd.DataFrame) -> list[str]:
    """trend_breakout BUY/LATE sinyali için açıklayıcı gerekçe etiketleri."""
    reasons: list[str] = []

    ema20 = details.get("ema_20", 0.0)
    ema50 = details.get("ema_50", 0.0)
    if ema50 > 0 and ema20 >= ema50 * (1 + _SCORE_EMA_GAP):
        reasons.append("EMA trend güçlü")

    vol_ratio = details.get("volume_ratio", 0.0)
    if vol_ratio >= _SCORE_VOLUME_MULT:
        reasons.append(f"Hacim patlaması (x{vol_ratio:.1f})")

    rsi = details.get("rsi_14", 0.0)
    if _SCORE_RSI_LOW <= rsi <= _SCORE_RSI_HIGH:
        reasons.append(f"RSI ideal ({rsi:.1f})")

    close    = details.get("close", 0.0)
    prev_res = details.get("prev_resistance", 0.0)
    if prev_res > 0 and close >= prev_res * (1 + _SCORE_BREAKOUT_MARGIN):
        reasons.append(f"Güçlü breakout (%{(close / prev_res - 1) * 100:.1f})")

    if not df.empty and "macd" in df.columns and "macd_signal" in df.columns:
        last = df.iloc[-1]
        macd     = last["macd"]
        macd_sig = last["macd_signal"]
        if pd.notna(macd) and pd.notna(macd_sig) and float(macd) > 0 and float(macd) > float(macd_sig):
            reasons.append("MACD pozitif")

    return reasons


def _compute_strength_score(strength: float, details: dict, df: pd.DataFrame) -> tuple[int, list[str]]:
    """Kanonik skor (strength×100) + trend_breakout gerekçeleri."""
    return _score_from_strength(strength), _breakout_reasons(details, df)


def _pbs_reasons(pbs_details: dict) -> list[str]:
    """Pre-breakout EARLY_WATCH sinyali için açıklayıcı gerekçe etiketleri."""
    reasons: list[str] = []
    if pbs_details.get("volatility_squeeze"):
        reasons.append("Squeeze aktif")
    if pbs_details.get("bb_width_in_low"):
        reasons.append("BB daralma bölgesinde")
    if pbs_details.get("macd_hist_rising"):
        reasons.append("MACD hist yükseliyor")
    if pbs_details.get("close_strengthening"):
        reasons.append("Fiyat güçleniyor")

    dist = pbs_details.get("distance_to_res_pct", 0.0)
    if dist and dist > 0:
        reasons.append(f"Direncin %{dist:.1f} altında")
    return reasons


def _pbs_strength_score(strength: float, pbs_details: dict) -> tuple[int, list[str]]:
    """Kanonik skor (strength×100) + pre-breakout gerekçeleri."""
    return _score_from_strength(strength), _pbs_reasons(pbs_details)


# ── Likidite filtresi ─────────────────────────────────────────────────────────

def _liquidity_ok(df: pd.DataFrame, min_tl: float = _LIQUIDITY_MIN_TL) -> bool:
    if df.empty or len(df) < 5:
        return True
    last20 = df.tail(20)
    avg_tl = (last20["close"] * last20["volume"]).mean()
    return avg_tl >= min_tl


# ── Sinyal işleme çekirdeği ───────────────────────────────────────────────────

def _ema20_pct(details: dict) -> Optional[float]:
    close = details.get("close", 0.0)
    ema20 = details.get("ema_20", 0.0)
    if close and ema20:
        return round((close / ema20 - 1) * 100, 2)
    return None


def _risk_reward(price: float, stop_loss: Optional[float], take_profit: Optional[float]) -> Optional[float]:
    """Long sinyal için R/R = (hedef − giriş) / (giriş − stop). Hesaplanamazsa None."""
    if not price or stop_loss is None or take_profit is None:
        return None
    risk = price - stop_loss
    reward = take_profit - price
    if risk <= 0:
        return None
    return round(reward / risk, 2)


def _rr_gate_ok(symbol: str, signal: str, price: float,
                stop_loss: Optional[float], take_profit: Optional[float]) -> bool:
    """
    Risk/ödül kalite kapısı. Zamanlanmış tarama risk/manager.py'den geçmediği
    için min R/R kuralı burada uygulanır (CLAUDE.md risk katmanı ile tutarlı).
    R/R hesaplanamazsa fail-open (engellemez).
    """
    rr = _risk_reward(price, stop_loss, take_profit)
    if rr is not None and rr < settings.min_risk_reward:
        logger.info(f"[BLOCKED] {symbol} {signal} — R/R {rr} < {settings.min_risk_reward}")
        return False
    return True


def _process_df(
    symbol: str,
    df: pd.DataFrame,
    mf: MarketFilterResult,
    ew_diag: Optional[_EWDiag] = None,
) -> Optional[ScanResult]:
    """
    İndikatörlü DataFrame'den sinyal üretir ve ScanResult döner.
    Sinyal tiplerini (EARLY_WATCH, BUY, LATE_BREAKOUT) destekler.
    """
    tb          = tb_generate(df)
    signal_type = tb["signal"]
    details     = tb["details"]
    price       = tb["price"]
    risk_level  = tb["risk_level"]
    strength    = tb["strength"]
    score, reasons = _compute_strength_score(strength, details, df)

    # ── BUY ──────────────────────────────────────────────────────────────────
    if signal_type == "BUY":
        if mf.blocks_buy:
            logger.info(f"[BLOCKED] {symbol} BUY — endeks filtresi: {mf.status}")
            return None
        if not _rr_gate_ok(symbol, "BUY", price, tb["stop_loss"], tb["take_profit"]):
            return None
        return ScanResult(
            symbol=symbol, signal="BUY", price=price, reason=tb["reason"],
            risk_level=risk_level, strength=strength, strategy="trend_breakout",
            stop_loss=tb["stop_loss"], take_profit=tb["take_profit"],
            market_filter=mf.status, conditions_met=5, distance_to_res_pct=0.0,
            strength_score=score, score_reasons=reasons,
            rsi_14=details.get("rsi_14"),
            volume_ratio=details.get("volume_ratio"),
            daily_change_pct=details.get("daily_change_pct"),
            close_to_ema20_pct=_ema20_pct(details),
        )

    # ── LATE_BREAKOUT ─────────────────────────────────────────────────────────
    if signal_type == "LATE_BREAKOUT":
        if mf.blocks_buy:
            logger.info(f"[BLOCKED] {symbol} LATE_BREAKOUT — endeks filtresi: {mf.status}")
            return None
        if not _rr_gate_ok(symbol, "LATE_BREAKOUT", price, tb["stop_loss"], tb["take_profit"]):
            return None
        return ScanResult(
            symbol=symbol, signal="LATE_BREAKOUT", price=price, reason=tb["reason"],
            risk_level=risk_level, strength=strength, strategy="trend_breakout",
            stop_loss=tb["stop_loss"], take_profit=tb["take_profit"],
            market_filter=mf.status, conditions_met=5, distance_to_res_pct=0.0,
            strength_score=score, score_reasons=reasons,
            rsi_14=details.get("rsi_14"),
            volume_ratio=details.get("volume_ratio"),
            daily_change_pct=details.get("daily_change_pct"),
            close_to_ema20_pct=_ema20_pct(details),
        )

    # ── HOLD → önce pre_breakout_squeeze dene ────────────────────────────────
    if signal_type == "HOLD":
        pbs         = pbs_generate(df)
        pbs_signal  = pbs["signal"]
        pbs_details = pbs.get("details", {})

        if pbs_signal == "EARLY_WATCH":
            pbs_score, pbs_reasons = _pbs_strength_score(pbs["strength"], pbs_details)
            dist_pct = pbs_details.get("distance_to_res_pct", None)
            mf_note = ""
            if mf.status == STATUS_UNFAVORABLE:
                mf_note = " [Endeks Olumsuz]"
            elif mf.status == STATUS_UNAVAILABLE:
                mf_note = " [Endeks Bilinmiyor]"
            return ScanResult(
                symbol=symbol,
                signal=pbs_signal,
                price=pbs["price"],
                reason=pbs["reason"] + mf_note,
                risk_level=pbs["risk_level"],
                strength=pbs["strength"],
                strategy="pre_breakout_squeeze",
                stop_loss=pbs["stop_loss"],
                take_profit=pbs["take_profit"],
                market_filter=mf.status,
                conditions_met=pbs_details.get("setup_conditions_met", 0),
                distance_to_res_pct=dist_pct,
                strength_score=pbs_score,
                score_reasons=pbs_reasons,
                rsi_14=pbs_details.get("rsi_14"),
                volume_ratio=pbs_details.get("volume_ratio"),
                daily_change_pct=pbs_details.get("daily_change_pct"),
                close_to_ema20_pct=pbs_details.get("close_to_ema20_pct"),
            )

        # pbs EARLY_WATCH üretmedi — EW koşul başarısızlıklarını tanı için kaydet
        if ew_diag is not None:
            ew_flags = pbs_details.get("ew_flags", {})
            if ew_flags:
                ew_diag.record(ew_flags)

        return None

    return None  # SELL → ilgi yok


# ── Tek sembol tarama ─────────────────────────────────────────────────────────

def _scan_symbol(
    symbol: str,
    mf: MarketFilterResult,
    period: str = "1y",
    min_tl_volume: float = 0.0,
) -> Optional[ScanResult]:
    try:
        fetch_result = fetch_symbol_data(symbol, period=period, interval="1d")
    except FetchError as exc:
        logger.warning(f"[SKIP] {symbol}: {exc}")
        return None

    df = add_indicators(fetch_result.df)

    if min_tl_volume > 0 and not _liquidity_ok(df, min_tl_volume):
        logger.debug(f"[LİKİDİTE] {symbol}: ort. günlük TL hacmi < {min_tl_volume / 1e6:.0f}M TL — atlandı")
        return None

    return _process_df(symbol, df, mf)


# ── Önbellekli sembol tarama (paralel path için) ──────────────────────────────

def _scan_symbol_cached(
    symbol: str,
    mf: MarketFilterResult,
    period: str,
    min_tl_volume: float,
    ew_diag: Optional[_EWDiag] = None,
) -> tuple[Optional[ScanResult], bool]:
    hit, cached = _cache_get(symbol)
    if hit:
        return cached, False

    try:
        _time.sleep(_PARALLEL_BATCH_DELAY)
        fetch_result = fetch_symbol_data_cached(symbol, period=period, interval="1d")
    except FetchError as exc:
        logger.warning(f"[SKIP] {symbol}: {exc}")
        _cache_set(symbol, None)
        return None, True

    df = add_indicators(fetch_result.df)

    if min_tl_volume > 0 and not _liquidity_ok(df, min_tl_volume):
        logger.debug(f"[LİKİDİTE] {symbol}: ort. günlük TL hacmi < {min_tl_volume / 1e6:.0f}M TL")
        _cache_set(symbol, None)
        return None, False

    result = _process_df(symbol, df, mf, ew_diag)
    _cache_set(symbol, result)
    return result, False


# ── Paralel tarama motoru ─────────────────────────────────────────────────────

def _scan_parallel(
    symbols: list[str],
    label: str = "ÖZEL",
    period: str = "1y",
    apply_market_filter: bool = True,
    force_market_refresh: bool = False,
    include_watch: bool = True,
    min_tl_volume: float = _LIQUIDITY_MIN_TL,
    max_workers: int = _PARALLEL_MAX_WORKERS,
) -> dict:
    """
    Sembolleri paralel olarak tarar.

    Sıralama: EARLY_WATCH → BUY → LATE_BREAKOUT
    Her grup içinde strength_score'a göre azalan.

    Returns:
        {
          label, results, scanned, buy_count,
          early_watch_count, late_breakout_count,
          error_count, elapsed_seconds, market_filter
        }
    """
    if not symbols:
        logger.warning(f"_scan_parallel [{label}]: boş sembol listesi")
        return _empty_report(label)

    t0 = time.perf_counter()

    if apply_market_filter:
        mf = is_market_favorable(force_refresh=force_market_refresh)
        if mf.blocks_buy:
            logger.warning(f"[{label}] Endeks filtresi — BUY sinyalleri kapalı: {mf.reason}")
        elif mf.status == STATUS_UNAVAILABLE:
            logger.warning(f"[{label}] Endeks filtresi mevcut değil, fail-open ile devam")
    else:
        from app.risk.market_filter import MarketFilterResult, STATUS_FAVORABLE
        mf = MarketFilterResult(favorable=True, status=STATUS_FAVORABLE, reason="filtre devre dışı")

    all_results:   list[ScanResult] = []
    error_symbols: list[str]        = []
    ew_diag = _EWDiag()

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_scan_symbol_cached, sym, mf, period, min_tl_volume, ew_diag): sym
            for sym in symbols
        }
        for future in as_completed(futures):
            sym = futures[future]
            try:
                result, had_error = future.result()
            except Exception as exc:
                logger.error(f"[HATA] {sym}: {exc}")
                error_symbols.append(sym)
                continue

            if had_error:
                error_symbols.append(sym)
                continue

            if result is None:
                continue

            if not include_watch and result.signal in ("EARLY_WATCH", "LATE_BREAKOUT"):
                continue

            all_results.append(result)

    # Tarama sırası: EARLY_WATCH → BUY → LATE_BREAKOUT
    # Her grup içinde strength_score azalan
    all_results.sort(
        key=lambda r: (_SIGNAL_ORDER.get(r.signal, 99), -r.strength_score)
    )

    signal_counts: dict[str, int] = {}
    for r in all_results:
        signal_counts[r.signal] = signal_counts.get(r.signal, 0) + 1

    elapsed = time.perf_counter() - t0
    logger.info(
        f"[{label}] Tarama tamamlandı | {len(symbols)} sembol | "
        f"{signal_counts.get('BUY', 0)} BUY | "
        f"{signal_counts.get('EARLY_WATCH', 0)} EARLY_WATCH | "
        f"{signal_counts.get('LATE_BREAKOUT', 0)} LATE | "
        f"{len(error_symbols)} hata | {elapsed:.1f}s | Endeks: {mf.status}"
    )
    if error_symbols:
        logger.warning(f"[{label}] Veri alınamayan semboller ({len(error_symbols)}): {error_symbols}")

    # ── İlk 15 sinyal tablosu ──────────────────────────────────────────────────
    if all_results:
        _log_top15(label, all_results)

    # ── EARLY_WATCH = 0 tanısı ────────────────────────────────────────────────
    if signal_counts.get("EARLY_WATCH", 0) == 0 and ew_diag.has_data():
        top = ew_diag.top_failures(7)
        fail_summary = " | ".join(f"{flag}={cnt}" for flag, cnt in top)
        logger.warning(
            f"[{label}] EARLY_WATCH=0 — en çok elenen koşullar: {fail_summary}"
        )

    return {
        "label":               label,
        "results":             [r.to_dict() for r in all_results],
        "scanned":             len(symbols),
        "buy_count":           signal_counts.get("BUY", 0),
        "early_watch_count":   signal_counts.get("EARLY_WATCH", 0),
        "late_breakout_count": signal_counts.get("LATE_BREAKOUT", 0),
        "error_count":         len(error_symbols),
        "error_symbols":       error_symbols,
        "elapsed_seconds":     round(elapsed, 1),
        "market_filter":       mf.status,
    }


def _log_top15(label: str, results: list[ScanResult]) -> None:
    header = (
        f"{'Sembol':<10} {'Sinyal':<12} {'Skor':>4} {'Fiyat':>9} "
        f"{'RSI':>5} {'HacimR':>6} {'Mes%':>6} {'Gün%':>6} {'EMA20%':>7}"
    )
    rows = []
    for r in results[:15]:
        rows.append(
            f"{r.symbol:<10} {r.signal:<12} {r.strength_score:>4} "
            f"{r.price:>9.2f} "
            f"{r.rsi_14 or 0:>5.1f} {r.volume_ratio or 0:>6.2f} "
            f"{r.distance_to_res_pct or 0:>6.1f} "
            f"{r.daily_change_pct or 0:>6.1f} "
            f"{r.close_to_ema20_pct or 0:>7.1f}"
        )
    logger.info(
        f"[{label}] İlk {min(15, len(results))} sinyal "
        f"(toplam {len(results)}):\n{header}\n" + "\n".join(rows)
    )


def _empty_report(label: str) -> dict:
    return {
        "label": label, "results": [], "scanned": 0,
        "buy_count": 0, "early_watch_count": 0, "late_breakout_count": 0,
        "error_count": 0, "error_symbols": [],
        "elapsed_seconds": 0.0, "market_filter": "unknown",
    }


# ── Kamuya açık BIST tarama fonksiyonları ────────────────────────────────────

def scan_bist30(
    period: str = "1y",
    apply_market_filter: bool = True,
    force_market_refresh: bool = False,
    include_watch: bool = True,
    min_tl_volume: float = _LIQUIDITY_MIN_TL,
) -> dict:
    """BIST30 (XU030) hisselerini paralel tarar."""
    from app.data.bist100 import BIST30_SYMBOLS
    return _scan_parallel(
        symbols=BIST30_SYMBOLS, label="BIST30", period=period,
        apply_market_filter=apply_market_filter,
        force_market_refresh=force_market_refresh,
        include_watch=include_watch, min_tl_volume=min_tl_volume,
    )


def scan_bist50(
    period: str = "1y",
    apply_market_filter: bool = True,
    force_market_refresh: bool = False,
    include_watch: bool = True,
    min_tl_volume: float = _LIQUIDITY_MIN_TL,
) -> dict:
    """BIST50 (XU050) hisselerini paralel tarar."""
    from app.data.bist100 import BIST50_SYMBOLS
    return _scan_parallel(
        symbols=BIST50_SYMBOLS, label="BIST50", period=period,
        apply_market_filter=apply_market_filter,
        force_market_refresh=force_market_refresh,
        include_watch=include_watch, min_tl_volume=min_tl_volume,
    )


def scan_bist100(
    period: str = "1y",
    apply_market_filter: bool = True,
    force_market_refresh: bool = False,
    include_watch: bool = True,
    min_tl_volume: float = _LIQUIDITY_MIN_TL,
    max_workers: int = _PARALLEL_MAX_WORKERS,
) -> dict:
    """BIST100 (XU100) hisselerini paralel tarar."""
    from app.data.bist100 import BIST100_SYMBOLS
    return _scan_parallel(
        symbols=BIST100_SYMBOLS, label="BIST100", period=period,
        apply_market_filter=apply_market_filter,
        force_market_refresh=force_market_refresh,
        include_watch=include_watch, min_tl_volume=min_tl_volume,
        max_workers=max_workers,
    )


# ── Mevcut sequential tarama (geriye dönük uyumlu) ───────────────────────────

def scan_market(
    symbols: list[str],
    period: str = "1y",
    apply_market_filter: bool = True,
    force_market_refresh: bool = False,
    include_watch: bool = True,
) -> list[dict]:
    """
    Tüm sembolleri sırayla tarar; tüm sinyal tiplerini döner.
    Mevcut API ve testlerle geriye dönük uyumludur.

    Sıralama: EARLY_WATCH → BUY → LATE_BREAKOUT
    """
    if not symbols:
        logger.warning("scan_market: boş sembol listesi")
        return []

    t0 = time.perf_counter()

    if apply_market_filter:
        mf = is_market_favorable(force_refresh=force_market_refresh)
        if mf.blocks_buy:
            logger.warning(f"Endeks filtresi — BUY sinyalleri kapalı: {mf.reason}")
        elif mf.status == STATUS_UNAVAILABLE:
            logger.warning("Endeks filtresi mevcut değil, fail-open ile devam ediliyor")
    else:
        from app.risk.market_filter import MarketFilterResult, STATUS_FAVORABLE
        mf = MarketFilterResult(favorable=True, status=STATUS_FAVORABLE, reason="filtre devre dışı")

    all_results: list[ScanResult] = []

    for symbol in symbols:
        result = _scan_symbol(symbol, mf, period=period)
        if result is None:
            continue
        if not include_watch and result.signal in ("EARLY_WATCH", "LATE_BREAKOUT"):
            continue
        all_results.append(result)

    # Sıralama _scan_parallel ile aynı kanonik anahtar: strength_score (0-100)
    all_results.sort(
        key=lambda r: (_SIGNAL_ORDER.get(r.signal, 99), -r.strength_score)
    )

    elapsed = time.perf_counter() - t0
    counts = {s: sum(1 for r in all_results if r.signal == s)
              for s in _SIGNAL_ORDER}
    logger.info(
        f"Tarama tamamlandı | {len(symbols)} sembol | "
        f"{counts.get('BUY',0)} BUY | {counts.get('EARLY_WATCH',0)} EARLY_WATCH | "
        f"{counts.get('LATE_BREAKOUT',0)} LATE | "
        f"{elapsed:.1f}s | Endeks: {mf.status}"
    )

    return [r.to_dict() for r in all_results]
