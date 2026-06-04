"""
Trend + Hacimli Kırılım Stratejisi — Backtest Koşucusu
=======================================================
backtesting.py üzerinde çalışır.
Tek veya çoklu hisse için JSON uyumlu sonuç döner.

Pozisyon büyüklüğü kuralı:
  İşlem başına maksimum risk = sermaye × %1
  Stop-loss mesafesi         = giriş fiyatı × %3
  → Pozisyon = risk / (fiyat × stop_pct) = sermayenin ~%33'ü
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import pandas_ta_classic as ta
from backtesting import Backtest, Strategy
from loguru import logger

from app.data.fetcher import fetch_symbol_data, FetchError
from app.indicators.technical import add_indicators
from app.strategies.trend_breakout import (
    generate_signal as tb_generate,
    _STOP_LOSS_PCT,
    _TAKE_PROFIT_PCT,
    _VOLUME_MULT,
    _RSI_LOW,
    _RSI_HIGH,
)
from app.strategies.pre_breakout_squeeze import generate_setup_signal as pbs_generate

# ── Sabitler ──────────────────────────────────────────────────────────────────

_INITIAL_CASH  = 100_000.0
_COMMISSION    = 0.001       # %0.1
_RISK_PER_TRADE = 0.01       # sermayenin %1'i
_POSITION_SIZE = _RISK_PER_TRADE / _STOP_LOSS_PCT  # ≈ 0.333


# ── pandas_ta → numpy sarmalayıcılar (backtesting.py için) ───────────────────

def _ema(arr: np.ndarray, length: int) -> np.ndarray:
    s = ta.ema(pd.Series(arr), length=length)
    return s.ffill().bfill().to_numpy(dtype=float)


def _rsi(arr: np.ndarray, length: int = 14) -> np.ndarray:
    s = ta.rsi(pd.Series(arr), length=length)
    return s.fillna(50.0).to_numpy(dtype=float)


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, length: int = 14) -> np.ndarray:
    s = ta.atr(pd.Series(high), pd.Series(low), pd.Series(close), length=length)
    return s.ffill().bfill().to_numpy(dtype=float)


def _sma(arr: np.ndarray, length: int) -> np.ndarray:
    s = pd.Series(arr).rolling(length).mean()
    return s.ffill().bfill().to_numpy(dtype=float)


def _rolling_max(arr: np.ndarray, period: int) -> np.ndarray:
    s = pd.Series(arr).rolling(period, min_periods=1).max()
    return s.to_numpy(dtype=float)


# ── backtesting.py Strateji Sınıfı ───────────────────────────────────────────

class TrendBreakoutBT(Strategy):
    """
    Trend + Hacimli Kırılım — backtesting.py uyumlu versiyon.

    Tüm şartlar trend_breakout.py ile birebir aynı;
    sadece backtesting.py'nin `self.I()` arayüzüne uyarlandı.
    """

    # Ayarlanabilir parametreler (optimize için)
    ema_fast          = 20
    ema_slow          = 50
    rsi_period        = 14
    atr_period        = 14
    volume_period     = 20
    resistance_period = 20
    rsi_low           = _RSI_LOW
    rsi_high          = _RSI_HIGH
    volume_mult       = _VOLUME_MULT
    stop_loss_pct     = _STOP_LOSS_PCT
    take_profit_pct   = _TAKE_PROFIT_PCT
    position_size     = _POSITION_SIZE

    def init(self):
        c = self.data.Close
        h = self.data.High
        l = self.data.Low
        v = self.data.Volume

        self.ema20        = self.I(_ema, c, self.ema_fast,    name="EMA20")
        self.ema50        = self.I(_ema, c, self.ema_slow,    name="EMA50")
        self.rsi          = self.I(_rsi, c, self.rsi_period,  name="RSI14")
        self.atr          = self.I(_atr, h, l, c, self.atr_period, name="ATR14")
        self.vol_ma       = self.I(_sma, v, self.volume_period,    name="VolMA20")
        self.resistance   = self.I(_rolling_max, h, self.resistance_period, name="Res20")

    def next(self):
        close  = self.data.Close[-1]
        ema20  = self.ema20[-1]
        ema50  = self.ema50[-1]
        rsi    = self.rsi[-1]
        atr    = self.atr[-1]
        vol    = self.data.Volume[-1]
        vol_ma = self.vol_ma[-1]

        # Bir önceki barın direnci (shift(1))
        prev_res = self.resistance[-2] if len(self.resistance) > 1 else self.resistance[-1]

        vol_ratio = vol / vol_ma if vol_ma > 0 else 0.0

        # ── Mevcut pozisyondan çıkış ──────────────────────────────────────────
        if self.position:
            if close < ema20:
                self.position.close()
            return

        # ── AL koşulları ──────────────────────────────────────────────────────
        c1 = close > ema20
        c2 = ema20 > ema50
        c3 = close > prev_res
        c4 = vol_ratio >= self.volume_mult
        c5 = self.rsi_low <= rsi <= self.rsi_high

        if not all([c1, c2, c3, c4, c5]):
            return

        sl = close * (1 - self.stop_loss_pct)
        tp = close * (1 + self.take_profit_pct)

        # Pozisyon büyüklüğü: risk/sermaye = %1, stop = %3 → ~%33
        size = min(self.position_size, 0.95)   # maksimum %95 sermaye
        self.buy(sl=sl, tp=tp, size=size)


# ── Sonuç nesnesi ─────────────────────────────────────────────────────────────

@dataclass
class BacktestResult:
    symbol:            str
    strategy:          str   = "trend_breakout"
    period:            str   = "2y"
    start_date:        str   = ""
    end_date:          str   = ""
    initial_cash:      float = _INITIAL_CASH
    final_equity:      float = 0.0
    total_return_pct:  float = 0.0
    buy_hold_return_pct: float = 0.0
    total_trades:      int   = 0
    win_rate_pct:      float = 0.0
    max_drawdown_pct:  float = 0.0
    best_trade_pct:    float = 0.0
    worst_trade_pct:   float = 0.0
    avg_trade_pct:     float = 0.0
    sharpe_ratio:      float = 0.0
    profit_factor:     float = 0.0
    commission_pct:    float = _COMMISSION * 100
    risk_per_trade_pct: float = _RISK_PER_TRADE * 100
    error:             Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @property
    def profitable(self) -> bool:
        return self.total_return_pct > 0 and self.error is None


def _error_result(symbol: str, reason: str) -> BacktestResult:
    return BacktestResult(symbol=symbol, error=reason)


# ── DataFrame dönüşümü ────────────────────────────────────────────────────────

def _to_bt_df(df: pd.DataFrame) -> pd.DataFrame:
    """Küçük harf OHLCV → backtesting.py büyük harf format, tz-naive."""
    bt = df.rename(columns={
        "open": "Open", "high": "High",
        "low": "Low",  "close": "Close", "volume": "Volume",
    })[["Open", "High", "Low", "Close", "Volume"]].copy()

    if hasattr(bt.index, "tz") and bt.index.tz is not None:
        bt.index = bt.index.tz_convert("UTC").tz_localize(None)
    bt.index = pd.to_datetime(bt.index)
    return bt


# ── İstatistik çıkarımı ───────────────────────────────────────────────────────

def _extract_stats(stats, symbol: str, period: str, cash: float = _INITIAL_CASH) -> BacktestResult:
    def _safe(key: str, default=0.0):
        val = stats.get(key, default)
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return default
        return val

    trades_df: pd.DataFrame = stats.get("_trades", pd.DataFrame())

    best_trade  = float(trades_df["ReturnPct"].max() * 100) if not trades_df.empty else 0.0
    worst_trade = float(trades_df["ReturnPct"].min() * 100) if not trades_df.empty else 0.0
    avg_trade   = float(trades_df["ReturnPct"].mean() * 100) if not trades_df.empty else 0.0

    return BacktestResult(
        symbol=symbol,
        period=period,
        initial_cash=cash,
        start_date=str(_safe("Start", "")),
        end_date=str(_safe("End", "")),
        final_equity=round(float(_safe("Equity Final [$]", _INITIAL_CASH)), 2),
        total_return_pct=round(float(_safe("Return [%]")), 2),
        buy_hold_return_pct=round(float(_safe("Buy & Hold Return [%]")), 2),
        total_trades=int(_safe("# Trades", 0)),
        win_rate_pct=round(float(_safe("Win Rate [%]")), 2),
        max_drawdown_pct=round(float(_safe("Max. Drawdown [%]")), 2),
        best_trade_pct=round(best_trade, 2),
        worst_trade_pct=round(worst_trade, 2),
        avg_trade_pct=round(avg_trade, 2),
        sharpe_ratio=round(float(_safe("Sharpe Ratio")), 3),
        profit_factor=round(float(_safe("Profit Factor")), 3),
    )


# ── Tek hisse backtest ────────────────────────────────────────────────────────

def run_single(
    symbol: str,
    period: str = "2y",
    start: Optional[str] = None,
    end: Optional[str] = None,
    cash: float = _INITIAL_CASH,
    commission: float = _COMMISSION,
) -> BacktestResult:
    """
    Tek sembol için trend_breakout backtesti çalıştırır.

    Args:
        symbol:     Hisse kodu ('THYAO' veya 'THYAO.IS')
        period:     yfinance periyodu ('1y', '2y', '5y', 'max')
        start/end:  Tarih aralığı — period yerine geçer ('2022-01-01')
        cash:       Başlangıç sermayesi (TL)
        commission: Komisyon oranı (0.001 = %0.1)

    Returns:
        BacktestResult — to_dict() / to_json() ile JSON alınabilir.
    """
    try:
        fetch_result = fetch_symbol_data(
            symbol, period=period, interval="1d", start=start, end=end
        )
    except FetchError as exc:
        logger.error(str(exc))
        return _error_result(symbol, str(exc))

    bt_df = _to_bt_df(fetch_result.df)

    if len(bt_df) < 60:
        msg = f"Yetersiz veri: {len(bt_df)} bar (min 60)"
        logger.warning(f"{symbol}: {msg}")
        return _error_result(symbol, msg)

    try:
        bt = Backtest(
            bt_df,
            TrendBreakoutBT,
            cash=cash,
            commission=commission,
            exclusive_orders=True,
            trade_on_close=False,
        )
        stats = bt.run()
    except Exception as exc:
        logger.error(f"{symbol} backtest hatası: {exc}")
        return _error_result(symbol, str(exc))

    result = _extract_stats(stats, fetch_result.symbol, period, cash=cash)
    logger.info(
        f"Backtest {result.symbol} | "
        f"Getiri: %{result.total_return_pct:+.1f} | "
        f"Trades: {result.total_trades} | "
        f"Kazanma: %{result.win_rate_pct:.0f} | "
        f"MaxDD: %{result.max_drawdown_pct:.1f}"
    )
    return result


# ── Çoklu hisse backtest ──────────────────────────────────────────────────────

@dataclass
class MultiBacktestResult:
    results:       list[BacktestResult]
    period:        str
    initial_cash:  float
    run_at:        str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    summary:       dict = field(default_factory=dict)

    def __post_init__(self):
        ok = [r for r in self.results if r.error is None]
        self.summary = {
            "total_symbols":    len(self.results),
            "successful":       len(ok),
            "failed":           len(self.results) - len(ok),
            "avg_return_pct":   round(sum(r.total_return_pct for r in ok) / len(ok), 2) if ok else 0.0,
            "best_symbol":      max(ok, key=lambda r: r.total_return_pct).symbol if ok else None,
            "worst_symbol":     min(ok, key=lambda r: r.total_return_pct).symbol if ok else None,
            "profitable_count": sum(1 for r in ok if r.profitable),
        }

    def to_dict(self) -> dict:
        return {
            "run_at":       self.run_at,
            "period":       self.period,
            "initial_cash": self.initial_cash,
            "summary":      self.summary,
            "results":      [r.to_dict() for r in self.results],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


# ── İleriye dönük getiri analizi ─────────────────────────────────────────────

_FORWARD_DAYS = [1, 3, 5, 10]


@dataclass
class ForwardReturnStats:
    """Bir sinyal tipinin ileriye dönük getiri istatistikleri."""
    signal_type:       str
    count:             int
    avg_max_1d:        float   # Ortalama 1 günlük maksimum getiri %
    avg_max_3d:        float
    avg_max_5d:        float
    avg_max_10d:       float
    median_max_5d:     float
    positive_5d_rate:  float   # 5. günde pozitif olan sinyallerin oranı

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ForwardReturnAnalysis:
    symbol:  str
    period:  str
    records: list[dict]            # her sinyal için ham kayıt
    stats:   list[ForwardReturnStats]   # sinyal tipine göre özet
    run_at:  str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def to_dict(self) -> dict:
        return {
            "symbol":  self.symbol,
            "period":  self.period,
            "run_at":  self.run_at,
            "stats":   [s.to_dict() for s in self.stats],
            "records": self.records,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


def compute_signal_forward_returns(
    symbol: str,
    period: str = "2y",
) -> ForwardReturnAnalysis:
    """
    Geçmiş veriler üzerinde her sinyal tipi için ileriye dönük maksimum getiri hesaplar.

    SETUP / EARLY_WATCH → Squeeze bölgesinden breakout gelirse ne kadar kazanılabilirdi?
    BUY                 → Kırılım anındaki ortalama ileriye dönük getiri
    LATE_BREAKOUT       → Geç giriş sonrası geri çekilme profili

    Her sinyal için: 1, 3, 5, 10 günlük maksimum high bazlı getiri (%)
    """
    try:
        fetch_result = fetch_symbol_data(symbol, period=period, interval="1d")
    except FetchError as exc:
        logger.error(f"ForwardReturn {symbol}: {exc}")
        return ForwardReturnAnalysis(symbol=symbol, period=period, records=[], stats=[])

    df_raw = fetch_result.df
    df     = add_indicators(df_raw)

    if len(df) < 70:
        logger.warning(f"ForwardReturn {symbol}: yetersiz veri ({len(df)} bar)")
        return ForwardReturnAnalysis(symbol=symbol, period=period, records=[], stats=[])

    records: list[dict] = []
    min_bars = 60

    for i in range(min_bars, len(df)):
        sub_df = df.iloc[:i + 1]

        tb = tb_generate(sub_df)
        signal_type = tb["signal"]

        if signal_type == "HOLD":
            pbs = pbs_generate(sub_df)
            signal_type = pbs["signal"]

        if signal_type in ("HOLD", "SELL"):
            continue

        entry_price = float(df.iloc[i]["close"])
        date_str    = str(df.index[i].date())

        forward: dict[str, Optional[float]] = {}
        for days in _FORWARD_DAYS:
            future_end = i + days
            if future_end < len(df):
                future_slice = df.iloc[i + 1: future_end + 1]
                if not future_slice.empty:
                    future_max = float(future_slice["high"].max())
                    forward[f"max_{days}d"] = round((future_max - entry_price) / entry_price * 100, 2)
                else:
                    forward[f"max_{days}d"] = None
            else:
                forward[f"max_{days}d"] = None

        records.append({
            "date":   date_str,
            "signal": signal_type,
            "price":  round(entry_price, 2),
            **forward,
        })

    # Sinyal tipine göre istatistik hesapla
    stats: list[ForwardReturnStats] = []
    for sig_type in ("EARLY_WATCH", "SETUP", "BUY", "LATE_BREAKOUT"):
        group = [r for r in records if r["signal"] == sig_type]
        if not group:
            continue

        def _avg(key: str) -> float:
            vals = [r[key] for r in group if r.get(key) is not None]
            return round(sum(vals) / len(vals), 2) if vals else 0.0

        def _median(key: str) -> float:
            vals = sorted(r[key] for r in group if r.get(key) is not None)
            if not vals:
                return 0.0
            n = len(vals)
            return round((vals[n // 2] + vals[(n - 1) // 2]) / 2, 2)

        pos_5d = [r for r in group if r.get("max_5d") is not None and r["max_5d"] > 0]

        stats.append(ForwardReturnStats(
            signal_type=sig_type,
            count=len(group),
            avg_max_1d=_avg("max_1d"),
            avg_max_3d=_avg("max_3d"),
            avg_max_5d=_avg("max_5d"),
            avg_max_10d=_avg("max_10d"),
            median_max_5d=_median("max_5d"),
            positive_5d_rate=round(len(pos_5d) / len(group) * 100, 1) if group else 0.0,
        ))

    logger.info(
        f"ForwardReturn {fetch_result.symbol} | {len(records)} sinyal kaydı | "
        f"{', '.join(s.signal_type + f'({s.count})' for s in stats)}"
    )
    return ForwardReturnAnalysis(
        symbol=fetch_result.symbol,
        period=period,
        records=records,
        stats=stats,
    )


def run_forward_returns(
    symbols: list[str],
    period: str = "2y",
    skip_errors: bool = True,
) -> dict:
    """
    Birden fazla sembol için forward return analizi çalıştırır.

    Returns:
        {
          "period": str,
          "run_at": str,
          "per_symbol": list[dict],      # her sembolün ForwardReturnAnalysis.to_dict()
          "aggregate": list[dict],       # tüm sembollerde sinyal tipine göre ortalama
        }
    """
    per_symbol: list[ForwardReturnAnalysis] = []

    for symbol in symbols:
        try:
            result = compute_signal_forward_returns(symbol, period=period)
            per_symbol.append(result)
        except Exception as exc:
            logger.error(f"run_forward_returns {symbol}: {exc}")
            if not skip_errors:
                raise

    # Aggregate: sinyal tipine göre tüm sembollerde ortalama
    agg: dict[str, list] = {}
    for analysis in per_symbol:
        for stat in analysis.stats:
            agg.setdefault(stat.signal_type, []).append(stat)

    aggregate: list[dict] = []
    for sig_type, stat_list in agg.items():
        def _mean(attr: str) -> float:
            vals = [getattr(s, attr) for s in stat_list]
            return round(sum(vals) / len(vals), 2) if vals else 0.0

        total_count = sum(s.count for s in stat_list)
        aggregate.append({
            "signal_type":      sig_type,
            "symbol_count":     len(stat_list),
            "total_signals":    total_count,
            "avg_max_1d":       _mean("avg_max_1d"),
            "avg_max_3d":       _mean("avg_max_3d"),
            "avg_max_5d":       _mean("avg_max_5d"),
            "avg_max_10d":      _mean("avg_max_10d"),
            "avg_positive_5d_rate": _mean("positive_5d_rate"),
        })

    aggregate.sort(key=lambda x: _SIGNAL_PRIORITY.get(x["signal_type"], 99))

    return {
        "period":     period,
        "run_at":     datetime.now().isoformat(timespec="seconds"),
        "per_symbol": [a.to_dict() for a in per_symbol],
        "aggregate":  aggregate,
    }


_SIGNAL_PRIORITY = {"EARLY_WATCH": 0, "SETUP": 1, "BUY": 2, "LATE_BREAKOUT": 3}


def run_multiple(
    symbols: list[str],
    period: str = "2y",
    cash: float = _INITIAL_CASH,
    commission: float = _COMMISSION,
    skip_errors: bool = True,
) -> MultiBacktestResult:
    """
    Birden fazla sembol için backtest çalıştırır.

    Args:
        symbols:     Hisse kodu listesi
        period:      yfinance periyodu
        cash:        Her hisse için ayrı başlangıç sermayesi
        commission:  Komisyon oranı
        skip_errors: True → hatalı sembolleri kayıt altına al ve devam et

    Returns:
        MultiBacktestResult — summary + tüm bireysel sonuçlar.
    """
    results: list[BacktestResult] = []

    for symbol in symbols:
        result = run_single(symbol, period=period, cash=cash, commission=commission)
        if result.error and not skip_errors:
            raise RuntimeError(f"{symbol} backtest başarısız: {result.error}")
        results.append(result)

    multi = MultiBacktestResult(results=results, period=period, initial_cash=cash)
    logger.info(
        f"Çoklu backtest tamamlandı: {multi.summary['successful']}/{multi.summary['total_symbols']} başarılı | "
        f"Ort. getiri: %{multi.summary['avg_return_pct']:+.1f} | "
        f"En iyi: {multi.summary['best_symbol']}"
    )
    return multi
