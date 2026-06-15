"""
BIST Sinyal Robotu — FastAPI Uygulaması
========================================
Çalıştırma:
  uvicorn app.main:app --reload --port 8000
  python -m app.main
"""

from __future__ import annotations

import asyncio
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    HTTPException,
    Query,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import init_db, get_db
from app.database.crud import (
    create_signal,
    get_active_signals,
    get_backtest_results,
    get_signals_by_symbol,
    save_backtest_result,
)
from app.data.fetcher import fetch_symbol_data, FetchError
from app.data.symbols import registry
from app.indicators.technical import add_indicators
from app.risk.market_filter import invalidate_cache, is_market_favorable
from app.signals.generator import signal_generator
from app.signals.scanner import scan_market, _scan_symbol, scan_bist30, scan_bist50, scan_bist100
from app.strategies.trend_breakout import generate_signal as tb_signal
from app.backtest.runner import run_single as bt_run_single, run_multiple as bt_run_multiple
from app.notifications.telegram import send_telegram_message, is_configured
from app.scheduler import scheduler, run_daily_scan
from app.utils.helpers import setup_logger

_VERSION = "0.1.0"


# ── Uygulama yaşam döngüsü ────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logger(settings.log_level)
    await init_db()
    scheduler.start()
    logger.info(f"BIST Sinyal Robotu v{_VERSION} başlatıldı")
    yield
    scheduler.stop()
    logger.info("BIST Sinyal Robotu durduruldu")


# ── Uygulama ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title="BIST Sinyal Robotu",
    description=(
        "Borsa İstanbul teknik analiz ve sinyal üretim sistemi.\n\n"
        "**Temel endpoint'ler:** `/health` · `/symbols` · `/scan` · `/signal/{symbol}` · `/backtest/{symbol}`"
    ),
    version=_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# GET /health
# ─────────────────────────────────────────────────────────────────────────────

@app.get(
    "/health",
    summary="Sağlık kontrolü",
    tags=["Sistem"],
)
async def health():
    """API'nin çalışıp çalışmadığını kontrol eder."""
    return {
        "status": "ok",
        "version": _VERSION,
        "telegram_configured": is_configured(),
        "symbol_count": len(registry),
    }


# ─────────────────────────────────────────────────────────────────────────────
# GET /test-telegram
# ─────────────────────────────────────────────────────────────────────────────

@app.get(
    "/test-telegram",
    summary="Telegram bağlantısını test et",
    tags=["Sistem"],
)
async def test_telegram():
    """
    Telegram'a test mesajı gönderir.

    - Token veya Chat ID eksikse yapılandırma hatası döner (400).
    - Mesaj iletilemezse Telegram hatası döner (502).
    - Başarılıysa gönderim onayı döner (200).
    """
    if not is_configured():
        raise HTTPException(
            status_code=400,
            detail=(
                "Telegram yapılandırılmamış. "
                ".env dosyasına TELEGRAM_BOT_TOKEN ve TELEGRAM_CHAT_ID ekleyin."
            ),
        )

    sent = await send_telegram_message("✅ Telegram bağlantısı başarılı")

    if not sent:
        raise HTTPException(
            status_code=502,
            detail="Mesaj gönderilemedi. Token veya Chat ID hatalı olabilir.",
        )

    return {
        "status": "ok",
        "message": "✅ Telegram bağlantısı başarılı",
        "chat_id": settings.telegram_chat_id,
    }


# ─────────────────────────────────────────────────────────────────────────────
# GET /symbols  (+ yönetim endpoint'leri)
# ─────────────────────────────────────────────────────────────────────────────

@app.get(
    "/symbols",
    summary="İzleme listesi",
    tags=["Semboller"],
)
async def get_symbols():
    """Aktif sembol listesini döner."""
    return {"count": len(registry), "symbols": registry.symbols}


# ⚠️ Literal route'lar ({symbol} parametresinden ÖNCE tanımlanmalı
@app.post(
    "/symbols/reset",
    summary="Varsayılan listeye dön",
    tags=["Semboller"],
)
async def reset_symbols():
    registry.reset()
    return {"message": "Varsayılan listeye dönüldü", "symbols": registry.symbols}


@app.post(
    "/symbols/{symbol}",
    summary="Sembol ekle",
    tags=["Semboller"],
)
async def add_symbol(symbol: str):
    added = registry.add(symbol)
    if not added:
        raise HTTPException(status_code=409, detail=f"{symbol.upper()} zaten listede")
    return {"message": f"{symbol.upper()} eklendi", "symbols": registry.symbols}


@app.delete(
    "/symbols/{symbol}",
    summary="Sembol çıkar",
    tags=["Semboller"],
)
async def remove_symbol(symbol: str):
    removed = registry.remove(symbol)
    if not removed:
        raise HTTPException(status_code=404, detail=f"{symbol.upper()} listede bulunamadı")
    return {"message": f"{symbol.upper()} çıkarıldı", "symbols": registry.symbols}


@app.post(
    "/symbols/upload/csv",
    summary="CSV'den sembol yükle",
    tags=["Semboller"],
)
async def upload_symbols_csv(file: UploadFile = File(...), replace: bool = False):
    """
    CSV dosyası yükleyerek sembol listesini günceller.
    `?replace=true` ile mevcut listeyi tamamen değiştirir.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Sadece .csv dosyası kabul edilir")
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)
    try:
        if replace:
            syms = registry.replace_from_csv(tmp_path)
            return {"message": "Liste değiştirildi", "count": len(syms), "symbols": syms}
        added = registry.load_csv(tmp_path)
        return {"message": "Semboller eklendi", "added": added, "symbols": registry.symbols}
    finally:
        tmp_path.unlink(missing_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# GET /scan
# ─────────────────────────────────────────────────────────────────────────────

@app.get(
    "/scan",
    summary="Tüm piyasayı tara",
    tags=["Sinyaller"],
)
async def market_scan(
    period: str = Query("1y", description="Veri periyodu: 6mo · 1y · 2y"),
    include_watch: bool = Query(True, description="Kırılım dışı (EARLY_WATCH/LATE) sinyalleri dahil et"),
    refresh_market: bool = Query(False, description="XU100 önbelleğini yenile"),
    symbols: Optional[List[str]] = Query(None, description="Taranacak semboller (boşsa kayıtlı liste)"),
):
    """
    Tüm sembolleri tarar; **EARLY_WATCH**, **BUY** ve **LATE_BREAKOUT** sinyallerini döner.

    - EARLY_WATCH → Kırılıma yakın, hacim/momentum canlanıyor 🟠
    - BUY  → Tüm kırılım koşulları + endeks filtresi ✅
    - LATE_BREAKOUT → Kırılım oldu ama geç kalınma işaretleri var 🔴
    """
    sym_list = symbols or registry.symbols
    results = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: scan_market(
            sym_list,
            period=period,
            include_watch=include_watch,
            force_market_refresh=refresh_market,
        ),
    )
    return {
        "scanned":             len(sym_list),
        "buy_count":           sum(1 for r in results if r["signal"] == "BUY"),
        "early_watch_count":   sum(1 for r in results if r["signal"] == "EARLY_WATCH"),
        "late_breakout_count": sum(1 for r in results if r["signal"] == "LATE_BREAKOUT"),
        "results":             results,
    }


# ─────────────────────────────────────────────────────────────────────────────
# GET /scan/bist100  |  /scan/bist50  |  /scan/bist30
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/scan/bist100", summary="BIST100 tam tarama", tags=["Sinyaller"])
async def scan_bist100_endpoint(
    period: str = Query("1y", description="Veri periyodu: 6mo · 1y · 2y"),
    include_watch: bool = Query(True, description="Kırılım dışı (EARLY_WATCH/LATE) sinyalleri dahil et"),
    refresh_market: bool = Query(False, description="XU100 önbelleğini yenile"),
    min_tl_volume: float = Query(50_000_000, description="Minimum günlük TL hacmi"),
):
    """
    BIST100 endeksindeki 100 hisseyi **paralel** olarak tarar.
    - `strength_score` (0–100) ile sıralı BUY sinyalleri döner
    - Likidite filtresi: `min_tl_volume` altındaki hisseler atlanır
    - 1 saatlik önbellek — kısa aralıklı çağrılarda tekrar veri çekilmez
    """
    return await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: scan_bist100(
            period=period,
            include_watch=include_watch,
            force_market_refresh=refresh_market,
            min_tl_volume=min_tl_volume,
        ),
    )


@app.get("/scan/bist50", summary="BIST50 tarama", tags=["Sinyaller"])
async def scan_bist50_endpoint(
    period: str = Query("1y"),
    include_watch: bool = Query(True),
    refresh_market: bool = Query(False),
    min_tl_volume: float = Query(50_000_000),
):
    """BIST50 endeksindeki 50 hisseyi paralel olarak tarar."""
    return await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: scan_bist50(
            period=period,
            include_watch=include_watch,
            force_market_refresh=refresh_market,
            min_tl_volume=min_tl_volume,
        ),
    )


@app.get("/scan/bist30", summary="BIST30 tarama", tags=["Sinyaller"])
async def scan_bist30_endpoint(
    period: str = Query("1y"),
    include_watch: bool = Query(True),
    refresh_market: bool = Query(False),
):
    """BIST30 endeksindeki 30 hisseyi paralel olarak tarar (likidite filtresi uygulanmaz)."""
    return await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: scan_bist30(
            period=period,
            include_watch=include_watch,
            force_market_refresh=refresh_market,
            min_tl_volume=0.0,
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# GET /signal/{symbol}
# ─────────────────────────────────────────────────────────────────────────────

@app.get(
    "/signal/{symbol}",
    summary="Tek hisse anlık sinyali",
    tags=["Sinyaller"],
)
async def get_signal(
    symbol: str,
    period: str = Query("1y", description="Veri periyodu"),
    apply_filter: bool = Query(True, description="XU100 endeks filtresini uygula"),
):
    """
    Tek hisse için anlık sinyal üretir.

    Döner:
    - **signal**: BUY · LATE_BREAKOUT · HOLD · SELL
    - **price**, **reason**, **risk_level**, **strength**
    - **stop_loss**, **take_profit** (BUY ise)
    - **conditions**: her koşulun ayrı sonucu
    - **market_filter**: XU100 filtresi durumu
    - **indicators**: son barın indikatör değerleri
    """
    # Veri çek
    try:
        fetch_result = await asyncio.get_event_loop().run_in_executor(
            None, lambda: fetch_symbol_data(symbol, period=period, interval="1d")
        )
    except FetchError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    # İndikatörler
    df = await asyncio.get_event_loop().run_in_executor(
        None, lambda: add_indicators(fetch_result.df)
    )

    # Trend-breakout sinyali (tüm detaylarıyla)
    tb = await asyncio.get_event_loop().run_in_executor(
        None, lambda: tb_signal(df)
    )

    # Son barın indikatör değerleri
    from app.indicators.technical import get_latest
    indicator_snapshot = get_latest(df)

    # Endeks filtresi
    mf = None
    if apply_filter:
        mf = await asyncio.get_event_loop().run_in_executor(
            None, is_market_favorable
        )

    # BUY ise endeks filtresi engelli mi?
    blocked_by_filter = (
        tb["signal"] == "BUY"
        and mf is not None
        and mf.blocks_buy
    )
    effective_signal = "HOLD" if blocked_by_filter else tb["signal"]

    return {
        "symbol":        fetch_result.symbol,
        "signal":        effective_signal,
        "price":         tb["price"],
        "reason":        tb["reason"] if not blocked_by_filter else f"Endeks filtresi engelledi: {mf.reason}",
        "risk_level":    tb["risk_level"],
        "strength":      tb["strength"],
        "stop_loss":     tb["stop_loss"] if not blocked_by_filter else None,
        "take_profit":   tb["take_profit"] if not blocked_by_filter else None,
        "conditions":    tb["details"],
        "market_filter": {
            "status":    mf.status if mf else "not_applied",
            "favorable": mf.favorable if mf else True,
            "reason":    mf.reason if mf else "",
        },
        "indicators":    indicator_snapshot,
        "data_range": {
            "start": str(fetch_result.start_date.date()) if fetch_result.start_date else None,
            "end":   str(fetch_result.end_date.date())   if fetch_result.end_date   else None,
            "bars":  fetch_result.row_count,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# GET /backtest/{symbol}
# ─────────────────────────────────────────────────────────────────────────────

@app.get(
    "/backtest/{symbol}",
    summary="Tek hisse backtest",
    tags=["Backtest"],
)
async def backtest_symbol(
    symbol: str,
    period: str = Query("2y", description="Veri periyodu: 1y · 2y · 5y · max"),
    cash: float = Query(100_000, description="Başlangıç sermayesi (TL)"),
    commission: float = Query(0.001, description="Komisyon oranı (0.001 = %0.1)"),
):
    """
    **Trend + Hacimli Kırılım** stratejisi için geçmiş performans analizi.

    Döner:
    - **total_return_pct**: Toplam getiri %
    - **total_trades**: İşlem sayısı
    - **win_rate_pct**: Kazanma oranı %
    - **max_drawdown_pct**: Maksimum düşüş %
    - **best_trade_pct / worst_trade_pct / avg_trade_pct**
    - **sharpe_ratio**, **profit_factor**
    - **buy_hold_return_pct**: Al-tut karşılaştırması
    """
    result = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: bt_run_single(symbol, period=period, cash=cash, commission=commission),
    )
    if result.error:
        raise HTTPException(status_code=400, detail=result.error)
    return result.to_dict()


@app.get(
    "/backtest",
    summary="Tüm sembollere backtest",
    tags=["Backtest"],
)
async def backtest_all(
    period: str = Query("2y"),
    cash: float = Query(100_000),
    symbols: Optional[List[str]] = Query(None),
):
    """Birden fazla sembol için toplu backtest çalıştırır."""
    sym_list = symbols or registry.symbols
    multi = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: bt_run_multiple(sym_list, period=period, cash=cash),
    )
    return multi.to_dict()


# ─────────────────────────────────────────────────────────────────────────────
# Yardımcı endpoint'ler
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Zamanlayıcı endpoint'leri
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/scheduler/status", summary="Zamanlayıcı durumu", tags=["Zamanlayıcı"])
async def scheduler_status():
    """Günlük tarama zamanlayıcısının durumunu ve bir sonraki çalışma zamanını döner."""
    return scheduler.status()


@app.post("/scheduler/run-now", summary="Taramayı hemen çalıştır", tags=["Zamanlayıcı"])
async def scheduler_run_now(background_tasks: BackgroundTasks):
    """
    Günlük taramayı hemen arka planda başlatır.
    Sonuç Telegram'a gönderilir.
    """
    background_tasks.add_task(run_daily_scan)
    return {
        "message": "Tarama başlatıldı — sonuç Telegram'a gönderilecek",
        "symbols": registry.symbols,
    }


@app.get("/market-filter", summary="XU100 endeks filtresi", tags=["Sistem"])
async def market_filter_status(refresh: bool = False):
    """XU100 Close > EMA50 kontrolü. `?refresh=true` önbelleği sıfırlar."""
    mf = is_market_favorable(force_refresh=refresh)
    return {
        "status":      mf.status,
        "favorable":   mf.favorable,
        "blocks_buy":  mf.blocks_buy,
        "reason":      mf.reason,
        "xu100_close": mf.xu100_close,
        "xu100_ema50": mf.xu100_ema50,
        "cached":      mf.cached,
    }


@app.post("/market-filter/invalidate", summary="XU100 önbelleğini temizle", tags=["Sistem"])
async def invalidate_market_filter():
    invalidate_cache()
    return {"message": "Endeks filtresi önbelleği temizlendi"}


@app.get("/symbols/{symbol}/ohlcv", summary="OHLCV fiyat verisi", tags=["Semboller"])
async def get_ohlcv(symbol: str, period: str = "3mo", interval: str = "1d"):
    """Ham OHLCV verisi döner (grafik için)."""
    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None, lambda: fetch_symbol_data(symbol, period=period, interval=interval)
        )
    except FetchError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    df = result.df.reset_index()
    df.columns = ["date"] + list(df.columns[1:])
    df["date"] = df["date"].astype(str)
    return {"symbol": result.symbol, "bars": result.row_count, "data": df.to_dict(orient="records")}


@app.post("/signals/trigger", summary="Arka planda sinyal tarama", tags=["Sinyaller"])
async def trigger_signal_scan(
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    symbols: Optional[List[str]] = None,
):
    """Sinyal taramasını arka planda başlatır ve DB'ye kaydeder."""
    active = await get_active_signals(db)

    async def _run():
        sym_list = symbols or registry.symbols
        for sym in sym_list:
            sigs = await signal_generator.run_for_symbol(sym, len(active))
            for s in sigs:
                await create_signal(db, s)

    background_tasks.add_task(_run)
    return {"message": "Tarama başlatıldı", "watching": symbols or registry.symbols}


@app.get("/signals/history", summary="Kaydedilen sinyaller", tags=["Sinyaller"])
async def signal_history(
    symbol: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    """DB'deki aktif sinyalleri döner. `?symbol=THYAO.IS` ile filtreler."""
    if symbol:
        sigs = await get_signals_by_symbol(db, symbol.upper())
    else:
        sigs = await get_active_signals(db)
    return {
        "count":   len(sigs),
        "signals": [
            {k: v for k, v in s.__dict__.items() if not k.startswith("_")}
            for s in sigs
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Genel hata yakalayıcı
# ─────────────────────────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"İşlenmeyen hata [{request.url}]: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Sunucu hatası — loglara bakın"},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Doğrudan çalıştırma
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=settings.log_level.lower(),
    )
