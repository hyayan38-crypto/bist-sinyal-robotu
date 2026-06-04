"""
BIST veri çekici — yfinance tabanlı, SQLite cache için hazır.

Kolon standardı (tüm kod tabanında küçük harf):
  open, high, low, close, volume
  index: DatetimeIndex (timezone-naive, UTC normalised)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import yfinance as yf
from loguru import logger

from app.config import settings


# ── İstisnalar ────────────────────────────────────────────────────────────────

class FetchError(Exception):
    """Veri çekme işlemi başarısız olduğunda fırlatılır."""
    def __init__(self, symbol: str, reason: str):
        self.symbol = symbol
        self.reason = reason
        super().__init__(f"[{symbol}] {reason}")


class EmptyDataError(FetchError):
    """yfinance'den boş DataFrame döndüğünde fırlatılır."""


class InsufficientDataError(FetchError):
    """Satır sayısı minimum eşiğin altında kaldığında fırlatılır."""


# ── Sonuç nesnesi ─────────────────────────────────────────────────────────────

@dataclass
class FetchResult:
    symbol: str
    df: pd.DataFrame
    period: str
    interval: str
    fetched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    from_cache: bool = False
    source: str = "yfinance"

    @property
    def row_count(self) -> int:
        return len(self.df)

    @property
    def start_date(self) -> Optional[datetime]:
        return self.df.index[0].to_pydatetime() if not self.df.empty else None

    @property
    def end_date(self) -> Optional[datetime]:
        return self.df.index[-1].to_pydatetime() if not self.df.empty else None

    def to_cache_records(self) -> list[dict]:
        """PriceCache modeli ile uyumlu kayıt listesi üretir."""
        records = []
        for ts, row in self.df.iterrows():
            records.append({
                "symbol": self.symbol,
                "interval": self.interval,
                "date": ts.to_pydatetime().replace(tzinfo=None),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
                "fetched_at": self.fetched_at.replace(tzinfo=None),
            })
        return records

    @staticmethod
    def from_cache_records(symbol: str, records: list[dict], interval: str) -> "FetchResult":
        """Cache kayıtlarından FetchResult oluşturur."""
        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")[["open", "high", "low", "close", "volume"]]
        df.index.name = None
        return FetchResult(
            symbol=symbol,
            df=df,
            period="cache",
            interval=interval,
            from_cache=True,
            source="sqlite",
        )

    def __repr__(self) -> str:
        return (
            f"FetchResult({self.symbol} | {self.row_count} bar | "
            f"{self.start_date:%Y-%m-%d} → {self.end_date:%Y-%m-%d} | "
            f"{'cache' if self.from_cache else 'live'})"
        )


# ── Yardımcı fonksiyonlar ─────────────────────────────────────────────────────

_MIN_ROWS = 30  # indikatör hesabı için minimum bar sayısı


def _normalize_symbol(symbol: str) -> str:
    s = symbol.strip().upper()
    return s if s.endswith(".IS") else f"{s}.IS"


def _clean_ohlcv(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Standart 5 kolonu seçer, timezone bilgisini siler,
    negatif/sıfır fiyat ve NaN satırları temizler.
    """
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise FetchError(symbol, f"Eksik kolonlar: {missing}")

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["open", "high", "low", "close", "volume"]

    # Timezone kaldır (SQLite tz-naive ister, backtesting.py de)
    if hasattr(df.index, "tz") and df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)

    df.index = pd.to_datetime(df.index)
    df.index.name = None

    before = len(df)
    df.dropna(inplace=True)
    df = df[(df["close"] > 0) & (df["high"] >= df["low"])]

    removed = before - len(df)
    if removed:
        logger.debug(f"{symbol}: {removed} geçersiz satır temizlendi")

    return df


def _log_summary(result: FetchResult):
    logger.info(
        f"{result.symbol} | {result.row_count} bar "
        f"({result.start_date:%Y-%m-%d} → {result.end_date:%Y-%m-%d}) "
        f"| {result.interval} | {'cache' if result.from_cache else 'yfinance'}"
    )


# ── Ana fonksiyonlar ──────────────────────────────────────────────────────────

def fetch_symbol_data(
    symbol: str,
    period: str = "2y",
    interval: str = "1d",
    start: Optional[str] = None,
    end: Optional[str] = None,
    min_rows: int = _MIN_ROWS,
    auto_adjust: bool = True,
) -> FetchResult:
    """
    Tek sembol için OHLCV verisi çeker.

    Args:
        symbol:      Hisse kodu ('THYAO' veya 'THYAO.IS')
        period:      yfinance periyot ('1mo','3mo','6mo','1y','2y','5y','max')
        interval:    Bar aralığı ('1d','1wk','1mo')
        start:       Başlangıç tarihi — period yerine geçer ('2022-01-01')
        end:         Bitiş tarihi ('2024-01-01')
        min_rows:    Yetersiz veri için eşik (varsayılan 30)
        auto_adjust: Temettü/bölünme düzeltmesi (varsayılan True)

    Returns:
        FetchResult — df, metadata ve cache yardımcılarını içerir.

    Raises:
        EmptyDataError:        yfinance boş DataFrame döndürdüğünde.
        InsufficientDataError: Satır sayısı min_rows altında kaldığında.
        FetchError:            Diğer çekme hatalarında.
    """
    ticker_symbol = _normalize_symbol(symbol)
    t0 = time.perf_counter()

    try:
        ticker = yf.Ticker(ticker_symbol)
        raw = (
            ticker.history(start=start, end=end, interval=interval, auto_adjust=auto_adjust)
            if start
            else ticker.history(period=period, interval=interval, auto_adjust=auto_adjust)
        )
    except Exception as exc:
        raise FetchError(ticker_symbol, f"yfinance hatası: {exc}") from exc

    if raw is None or raw.empty:
        raise EmptyDataError(ticker_symbol, "yfinance boş veri döndürdü")

    df = _clean_ohlcv(raw, ticker_symbol)

    if len(df) < min_rows:
        raise InsufficientDataError(
            ticker_symbol,
            f"Yetersiz veri: {len(df)} satır < minimum {min_rows}",
        )

    elapsed = time.perf_counter() - t0
    result = FetchResult(symbol=ticker_symbol, df=df, period=period, interval=interval)
    _log_summary(result)
    logger.debug(f"{ticker_symbol} çekme süresi: {elapsed:.2f}s")
    return result


def fetch_multiple_symbols(
    symbols: list[str],
    period: str = "2y",
    interval: str = "1d",
    skip_errors: bool = True,
) -> dict[str, FetchResult]:
    """
    Birden fazla sembol için OHLCV verisi çeker.

    Args:
        symbols:     Hisse kodu listesi
        period:      yfinance periyot
        interval:    Bar aralığı
        skip_errors: True → hatalı sembolleri atla ve devam et
                     False → ilk hatada FetchError fırlat

    Returns:
        {normalised_symbol: FetchResult} — yalnızca başarılı sonuçlar.
    """
    results: dict[str, FetchResult] = {}
    errors: list[str] = []

    for symbol in symbols:
        try:
            result = fetch_symbol_data(symbol, period=period, interval=interval)
            results[result.symbol] = result
        except FetchError as exc:
            if not skip_errors:
                raise
            errors.append(exc.symbol)
            logger.warning(f"Atlandı — {exc}")

    logger.info(
        f"fetch_multiple: {len(results)}/{len(symbols)} başarılı"
        + (f" | {len(errors)} hata: {errors}" if errors else "")
    )
    return results


# ── Geriye dönük uyumluluk ────────────────────────────────────────────────────

class BISTDataFetcher:
    """
    Eski arayüz — main.py ve backtest engine bu sınıfı kullanır.
    İç implementasyon fetch_symbol_data'ya delege eder.
    """

    def get_ohlcv(
        self,
        symbol: str,
        period: Optional[str] = None,
        interval: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        try:
            result = fetch_symbol_data(
                symbol,
                period=period or settings.data_period,
                interval=interval or settings.data_interval,
                start=start,
                end=end,
            )
            return result.df
        except FetchError as exc:
            logger.error(str(exc))
            return None

    def get_multiple(self, symbols: list, **kwargs) -> dict[str, pd.DataFrame]:
        results = fetch_multiple_symbols(symbols, **kwargs)
        return {sym: r.df for sym, r in results.items()}

    def get_ticker_info(self, symbol: str) -> dict:
        ticker_symbol = _normalize_symbol(symbol)
        try:
            info = yf.Ticker(ticker_symbol).info
            return {
                "symbol": ticker_symbol,
                "name": info.get("longName", ""),
                "sector": info.get("sector", ""),
                "market_cap": info.get("marketCap"),
                "current_price": info.get("currentPrice"),
                "52w_high": info.get("fiftyTwoWeekHigh"),
                "52w_low": info.get("fiftyTwoWeekLow"),
            }
        except Exception as exc:
            logger.error(f"{ticker_symbol} bilgi çekme hatası: {exc}")
            return {}


fetcher = BISTDataFetcher()
