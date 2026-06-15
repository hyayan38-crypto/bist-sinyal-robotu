"""
SQLite Fiyat Önbelleği (senkron)
=================================
Paralel tarama ThreadPoolExecutor içinde **senkron** çalışır; asyncio tabanlı
veritabanı katmanı (aiosqlite) bu bağlamda kullanılamaz. Bu modül stdlib
`sqlite3` ile aynı `bist_robot.db` dosyasındaki `price_cache` tablosuna yazar.

Tasarım kararları
─────────────────
- Tablo şeması `app/database/models.py::PriceCache` ile uyumludur; async
  `init_db()` çalışmamış olsa bile burada `CREATE TABLE IF NOT EXISTS` ile
  kendi kendine oluşturulur.
- Her işlem yeni bir bağlantı açar (sqlite3 bağlantıları thread'ler arası
  paylaşılamaz). WAL + busy_timeout eşzamanlı yazımları tolere eder.
- **Fail-open**: herhangi bir önbellek hatası yutulur (debug log) ve çağıran
  taraf doğrudan yfinance'e düşer. Önbellek asla taramayı bozmaz.

Kullanım: `app/data/fetcher.py::fetch_symbol_data_cached` üzerinden dolaylı.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional

import pandas as pd
from loguru import logger

from app.config import settings

_COLS = ("open", "high", "low", "close", "volume")


@dataclass
class CachedSeries:
    df: pd.DataFrame             # index=date, kolonlar: open,high,low,close,volume
    last_fetch_date: date        # MAX(fetched_at) — en son indirme günü


# ── Bağlantı / şema ────────────────────────────────────────────────────────────

def _db_path() -> str:
    """settings.database_url'den dosya yolunu çıkarır (async engine ile aynı dosya)."""
    url = settings.database_url
    for prefix in ("sqlite+aiosqlite:///", "sqlite:///"):
        if url.startswith(prefix):
            return url[len(prefix):]
    return url


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_db_path(), timeout=5.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS price_cache (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol     VARCHAR(20)  NOT NULL,
            interval   VARCHAR(5)   NOT NULL DEFAULT '1d',
            date       DATETIME     NOT NULL,
            open       FLOAT,
            high       FLOAT,
            low        FLOAT,
            close      FLOAT,
            volume     FLOAT,
            fetched_at DATETIME,
            UNIQUE (symbol, interval, date)
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS ix_price_cache_symbol ON price_cache (symbol)")
    return conn


# ── Okuma ────────────────────────────────────────────────────────────────────

def load(symbol: str, interval: str = "1d") -> Optional[CachedSeries]:
    """Önbellekteki tüm barları DataFrame olarak döner. Yoksa/boşsa None."""
    try:
        conn = _connect()
        try:
            rows = conn.execute(
                "SELECT date, open, high, low, close, volume, fetched_at "
                "FROM price_cache WHERE symbol=? AND interval=? ORDER BY date",
                (symbol, interval),
            ).fetchall()
        finally:
            conn.close()
    except Exception as exc:  # noqa: BLE001 — fail-open
        logger.debug(f"[CACHE] {symbol} okuma hatası: {exc}")
        return None

    if not rows:
        return None

    df = pd.DataFrame(rows, columns=["date", *_COLS, "fetched_at"])
    fetched = pd.to_datetime(df.pop("fetched_at"), errors="coerce")
    last_fetch = fetched.max()
    last_fetch_date = last_fetch.date() if pd.notna(last_fetch) else date.min

    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")[list(_COLS)]
    df.index.name = None
    return CachedSeries(df=df, last_fetch_date=last_fetch_date)


# ── Yazma ────────────────────────────────────────────────────────────────────

def _records(symbol: str, interval: str, df: pd.DataFrame, fetched_at: datetime) -> list[tuple]:
    fa = fetched_at.replace(tzinfo=None).isoformat(sep=" ", timespec="seconds")
    out = []
    for ts, row in df.iterrows():
        out.append((
            symbol, interval,
            pd.Timestamp(ts).to_pydatetime().replace(tzinfo=None).isoformat(sep=" "),
            float(row["open"]), float(row["high"]), float(row["low"]),
            float(row["close"]), float(row["volume"]), fa,
        ))
    return out


def replace(symbol: str, interval: str, df: pd.DataFrame,
            fetched_at: Optional[datetime] = None) -> bool:
    """Sembolün tüm barlarını siler ve yeniden yazar (tam indirme sonrası)."""
    fetched_at = fetched_at or datetime.now()
    try:
        conn = _connect()
        try:
            conn.execute("DELETE FROM price_cache WHERE symbol=? AND interval=?", (symbol, interval))
            conn.executemany(
                "INSERT OR REPLACE INTO price_cache "
                "(symbol, interval, date, open, high, low, close, volume, fetched_at) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                _records(symbol, interval, df, fetched_at),
            )
            conn.commit()
        finally:
            conn.close()
        return True
    except Exception as exc:  # noqa: BLE001 — fail-open
        logger.debug(f"[CACHE] {symbol} replace hatası: {exc}")
        return False


def upsert(symbol: str, interval: str, df: pd.DataFrame,
           fetched_at: Optional[datetime] = None) -> bool:
    """Son barları günceller/ekler (gün içi artımlı yenileme)."""
    fetched_at = fetched_at or datetime.now()
    try:
        conn = _connect()
        try:
            conn.executemany(
                "INSERT OR REPLACE INTO price_cache "
                "(symbol, interval, date, open, high, low, close, volume, fetched_at) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                _records(symbol, interval, df, fetched_at),
            )
            conn.commit()
        finally:
            conn.close()
        return True
    except Exception as exc:  # noqa: BLE001 — fail-open
        logger.debug(f"[CACHE] {symbol} upsert hatası: {exc}")
        return False
