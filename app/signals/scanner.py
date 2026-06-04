"""
Piyasa Tarayıcı
===============
Sembolleri tarar; aşağıdaki sinyal tiplerini döner:

  EARLY_WATCH  → Kırılıma ≤%2, hacim canlanıyor, MACD ve fiyat yükseliyor
  SETUP        → Squeeze aktif, dirençe ≤%5, düşük hacim, RSI orta bölge
  BUY          → Trend kırılımı — tüm koşullar sağlandı, geç değil
  WATCH        → Trend doğru ama kırılım henüz yok (eski mantık, geriye dönük uyumlu)
  LATE_BREAKOUT → Tüm BUY koşulları sağlandı ancak geç kalınma işaretleri var

Tarama sırası: EARLY_WATCH → SETUP → BUY → WATCH → LATE_BREAKOUT

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

import time as _time

import pandas as pd
from loguru import logger

from app.data.fetcher import fetch_symbol_data, FetchError
from app.indicators.technical import add_indicators
from app.risk.market_filter import (
    is_market_favorable,
    MarketFilterResult,
    STATUS_UNAVAILABLE,
    STATUS_UNFAVORABLE,
)
from app.strategies.trend_breakout import (
    generate_signal as tb_generate,
    _VOLUME_MULT,
    _RSI_LOW,
    _RSI_HIGH,
)
from app.strategies.pre_breakout_squeeze import generate_setup_signal as pbs_generate

# ── Eşikler ───────────────────────────────────────────────────────────────────

_WATCH_MIN_CONDITIONS   = 2
_WATCH_RESISTANCE_PCT   = 0.03
_BUY_MIN_STRENGTH       = 0.0
_WATCH_RSI_RANGE        = (_RSI_LOW - 10, _RSI_HIGH)

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
    "SETUP":         1,
    "BUY":           2,
    "WATCH":         3,
    "LATE_BREAKOUT": 4,
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
    signal:               str    # "EARLY_WATCH"|"SETUP"|"BUY"|"WATCH"|"LATE_BREAKOUT"
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
    scanned_at:           str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def to_dict(self) -> dict:
        return asdict(self)


# ── Strength Score (0-100) ────────────────────────────────────────────────────

def _compute_strength_score(details: dict, df: pd.DataFrame) -> tuple[int, list[str]]:
    score = 0
    reasons: list[str] = []

    ema20 = details.get("ema_20", 0.0)
    ema50 = details.get("ema_50", 0.0)
    if ema50 > 0 and ema20 >= ema50 * (1 + _SCORE_EMA_GAP):
        score += 20
        reasons.append("EMA trend güçlü")

    vol_ratio = details.get("volume_ratio", 0.0)
    if vol_ratio >= _SCORE_VOLUME_MULT:
        score += 25
        reasons.append(f"Hacim patlaması (x{vol_ratio:.1f})")

    rsi = details.get("rsi_14", 0.0)
    if _SCORE_RSI_LOW <= rsi <= _SCORE_RSI_HIGH:
        score += 15
        reasons.append(f"RSI ideal ({rsi:.1f})")

    close    = details.get("close", 0.0)
    prev_res = details.get("prev_resistance", 0.0)
    if prev_res > 0 and close >= prev_res * (1 + _SCORE_BREAKOUT_MARGIN):
        score += 25
        reasons.append(f"Güçlü breakout (%{(close / prev_res - 1) * 100:.1f})")

    if not df.empty and "macd" in df.columns and "macd_signal" in df.columns:
        last = df.iloc[-1]
        macd     = last["macd"]
        macd_sig = last["macd_signal"]
        if pd.notna(macd) and pd.notna(macd_sig) and float(macd) > 0 and float(macd) > float(macd_sig):
            score += 15
            reasons.append("MACD pozitif")

    return min(score, 100), reasons


def _pbs_strength_score(pbs_details: dict) -> tuple[int, list[str]]:
    """Pre-breakout sinyal gücü → 0-100 tamsayıya çevirir."""
    setup_met = pbs_details.get("setup_conditions_met", 0)
    early_met = pbs_details.get("early_conditions_met", 0)
    reasons: list[str] = []

    score = int(round(
        (setup_met / 6) * 50 + (early_met / 4) * 50
    ))

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

    return min(score, 100), reasons


# ── Likidite filtresi ─────────────────────────────────────────────────────────

def _liquidity_ok(df: pd.DataFrame, min_tl: float = _LIQUIDITY_MIN_TL) -> bool:
    if df.empty or len(df) < 5:
        return True
    last20 = df.tail(20)
    avg_tl = (last20["close"] * last20["volume"]).mean()
    return avg_tl >= min_tl


# ── WATCH koşulu değerlendirme (geriye dönük uyumlu) ─────────────────────────

def _assess_watch(details: dict) -> tuple[bool, int, Optional[float]]:
    c1 = bool(details.get("c1_above_ema20", False))
    c2 = bool(details.get("c2_ema_uptrend", False))
    c3 = bool(details.get("c3_breakout", False))
    c4 = bool(details.get("c4_volume_surge", False))
    c5 = bool(details.get("c5_rsi_range", False))

    if not (c1 and c2):
        return False, sum([c1, c2, c3, c4, c5]), None

    extra_met  = sum([c3, c4, c5])
    total_met  = 2 + extra_met

    close    = details.get("close", 0.0)
    prev_res = details.get("prev_resistance", 0.0)
    dist_pct = round((prev_res - close) / close * 100, 2) if close > 0 and prev_res > 0 else None

    approaching = dist_pct is not None and 0 < dist_pct <= _WATCH_RESISTANCE_PCT * 100

    rsi    = details.get("rsi_14", 0.0)
    rsi_ok = _WATCH_RSI_RANGE[0] <= rsi <= _WATCH_RSI_RANGE[1]

    is_watch = extra_met >= _WATCH_MIN_CONDITIONS or (extra_met >= 1 and approaching and rsi_ok)
    return is_watch, total_met, dist_pct


def _watch_reason(details: dict, dist_pct: Optional[float]) -> str:
    parts = ["Trend yukarı (EMA20>EMA50)"]
    if details.get("c4_volume_surge"):
        parts.append(f"Hacim x{details.get('volume_ratio', 0):.1f}")
    if details.get("c5_rsi_range"):
        parts.append(f"RSI14: {details.get('rsi_14', 0):.1f}")
    if dist_pct is not None and dist_pct > 0:
        parts.append(f"Dirence %{dist_pct:.1f} uzakta")
    return " | ".join(parts)


# ── Sinyal işleme çekirdeği ───────────────────────────────────────────────────

def _process_df(
    symbol: str,
    df: pd.DataFrame,
    mf: MarketFilterResult,
) -> Optional[ScanResult]:
    """
    İndikatörlü DataFrame'den sinyal üretir ve ScanResult döner.
    Tüm sinyal tiplerini (EARLY_WATCH, SETUP, BUY, WATCH, LATE_BREAKOUT) destekler.
    """
    tb          = tb_generate(df)
    signal_type = tb["signal"]
    details     = tb["details"]
    price       = tb["price"]
    risk_level  = tb["risk_level"]
    strength    = tb["strength"]
    score, reasons = _compute_strength_score(details, df)

    # ── BUY ──────────────────────────────────────────────────────────────────
    if signal_type == "BUY":
        if mf.blocks_buy:
            logger.info(f"[BLOCKED] {symbol} BUY — endeks filtresi: {mf.status}")
            return None
        return ScanResult(
            symbol=symbol, signal="BUY", price=price, reason=tb["reason"],
            risk_level=risk_level, strength=strength, strategy="trend_breakout",
            stop_loss=tb["stop_loss"], take_profit=tb["take_profit"],
            market_filter=mf.status, conditions_met=5, distance_to_res_pct=0.0,
            strength_score=score, score_reasons=reasons,
        )

    # ── LATE_BREAKOUT ─────────────────────────────────────────────────────────
    if signal_type == "LATE_BREAKOUT":
        if mf.blocks_buy:
            logger.info(f"[BLOCKED] {symbol} LATE_BREAKOUT — endeks filtresi: {mf.status}")
            return None
        return ScanResult(
            symbol=symbol, signal="LATE_BREAKOUT", price=price, reason=tb["reason"],
            risk_level=risk_level, strength=strength, strategy="trend_breakout",
            stop_loss=tb["stop_loss"], take_profit=tb["take_profit"],
            market_filter=mf.status, conditions_met=5, distance_to_res_pct=0.0,
            strength_score=score, score_reasons=reasons,
        )

    # ── HOLD → önce pre_breakout_squeeze dene ────────────────────────────────
    if signal_type == "HOLD":
        pbs = pbs_generate(df)
        pbs_signal = pbs["signal"]

        if pbs_signal in ("EARLY_WATCH", "SETUP"):
            pbs_details = pbs.get("details", {})
            pbs_score, pbs_reasons = _pbs_strength_score(pbs_details)
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
            )

        # Geriye dönük WATCH mantığı
        is_watch, n_met, dist_pct = _assess_watch(details)
        if not is_watch:
            return None

        mf_note = ""
        if mf.status == STATUS_UNFAVORABLE:
            mf_note = " [Endeks Olumsuz]"
        elif mf.status == STATUS_UNAVAILABLE:
            mf_note = " [Endeks Bilinmiyor]"

        return ScanResult(
            symbol=symbol, signal="WATCH", price=price,
            reason=_watch_reason(details, dist_pct) + mf_note,
            risk_level=risk_level, strength=round(n_met / 5, 2),
            strategy="trend_breakout", stop_loss=None, take_profit=None,
            market_filter=mf.status, conditions_met=n_met,
            distance_to_res_pct=dist_pct, strength_score=score, score_reasons=reasons,
        )

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
) -> tuple[Optional[ScanResult], bool]:
    hit, cached = _cache_get(symbol)
    if hit:
        return cached, False

    try:
        _time.sleep(_PARALLEL_BATCH_DELAY)
        fetch_result = fetch_symbol_data(symbol, period=period, interval="1d")
    except FetchError as exc:
        logger.warning(f"[SKIP] {symbol}: {exc}")
        _cache_set(symbol, None)
        return None, True

    df = add_indicators(fetch_result.df)

    if min_tl_volume > 0 and not _liquidity_ok(df, min_tl_volume):
        logger.debug(f"[LİKİDİTE] {symbol}: ort. günlük TL hacmi < {min_tl_volume / 1e6:.0f}M TL")
        _cache_set(symbol, None)
        return None, False

    result = _process_df(symbol, df, mf)
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

    Sıralama: EARLY_WATCH → SETUP → BUY → WATCH → LATE_BREAKOUT
    Her grup içinde strength_score'a göre azalan.

    Returns:
        {
          label, results, scanned, buy_count, watch_count,
          setup_count, early_watch_count, late_breakout_count,
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

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_scan_symbol_cached, sym, mf, period, min_tl_volume): sym
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

            if not include_watch and result.signal in ("WATCH", "SETUP", "EARLY_WATCH", "LATE_BREAKOUT"):
                continue

            all_results.append(result)

    # Tarama sırası: EARLY_WATCH → SETUP → BUY → WATCH → LATE_BREAKOUT
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
        f"{signal_counts.get('SETUP', 0)} SETUP | "
        f"{signal_counts.get('LATE_BREAKOUT', 0)} LATE | "
        f"{signal_counts.get('WATCH', 0)} WATCH | "
        f"{len(error_symbols)} hata | {elapsed:.1f}s | Endeks: {mf.status}"
    )
    if error_symbols:
        logger.warning(f"[{label}] Veri alınamayan semboller ({len(error_symbols)}): {error_symbols}")

    return {
        "label":               label,
        "results":             [r.to_dict() for r in all_results],
        "scanned":             len(symbols),
        "buy_count":           signal_counts.get("BUY", 0),
        "watch_count":         signal_counts.get("WATCH", 0),
        "setup_count":         signal_counts.get("SETUP", 0),
        "early_watch_count":   signal_counts.get("EARLY_WATCH", 0),
        "late_breakout_count": signal_counts.get("LATE_BREAKOUT", 0),
        "error_count":         len(error_symbols),
        "error_symbols":       error_symbols,
        "elapsed_seconds":     round(elapsed, 1),
        "market_filter":       mf.status,
    }


def _empty_report(label: str) -> dict:
    return {
        "label": label, "results": [], "scanned": 0,
        "buy_count": 0, "watch_count": 0,
        "setup_count": 0, "early_watch_count": 0, "late_breakout_count": 0,
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

    Sıralama: EARLY_WATCH → SETUP → BUY → WATCH → LATE_BREAKOUT
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
        if not include_watch and result.signal in ("WATCH", "SETUP", "EARLY_WATCH", "LATE_BREAKOUT"):
            continue
        all_results.append(result)

    all_results.sort(
        key=lambda r: (_SIGNAL_ORDER.get(r.signal, 99), -r.strength)
    )

    elapsed = time.perf_counter() - t0
    counts = {s: sum(1 for r in all_results if r.signal == s)
              for s in _SIGNAL_ORDER}
    logger.info(
        f"Tarama tamamlandı | {len(symbols)} sembol | "
        f"{counts.get('BUY',0)} BUY | {counts.get('EARLY_WATCH',0)} EARLY_WATCH | "
        f"{counts.get('SETUP',0)} SETUP | {counts.get('LATE_BREAKOUT',0)} LATE | "
        f"{elapsed:.1f}s | Endeks: {mf.status}"
    )

    return [r.to_dict() for r in all_results]
