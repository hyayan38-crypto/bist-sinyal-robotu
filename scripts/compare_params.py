"""
Parametre karşılaştırma scripti
================================
Mevcut strateji (sabit %3 SL / %6 TP) ile ATR bazlı SL/TP ve farklı
hacim eşiklerini aynı veri üzerinde karşılaştırır.

Kullanım:
    source venv/bin/activate
    python scripts/compare_params.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from backtesting import Backtest

from app.backtest.runner import TrendBreakoutBT, _to_bt_df, _INITIAL_CASH, _COMMISSION
from app.data.fetcher import fetch_symbol_data, FetchError

SYMBOLS = [
    "AKBNK.IS", "ARCLK.IS", "ASELS.IS", "BIMAS.IS", "EREGL.IS",
    "FROTO.IS", "GARAN.IS", "ISCTR.IS", "KCHOL.IS", "KRDMD.IS",
    "PGSUS.IS", "SAHOL.IS", "SASA.IS", "SISE.IS", "TAVHL.IS",
    "TCELL.IS", "THYAO.IS", "TOASO.IS", "TUPRS.IS", "YKBNK.IS",
    "AKSEN.IS", "BRSAN.IS", "MGROS.IS", "DOAS.IS", "ULKER.IS",
]
PERIOD = "2y"


class TrendBreakoutATR(TrendBreakoutBT):
    """ATR bazlı SL/TP varyantı — risk pariteli pozisyon büyüklüğü."""

    atr_sl_mult = 1.5
    atr_tp_mult = 3.0
    risk_per_trade = 0.01

    def next(self):
        close  = self.data.Close[-1]
        ema20  = self.ema20[-1]
        ema50  = self.ema50[-1]
        rsi    = self.rsi[-1]
        atr    = self.atr[-1]
        vol    = self.data.Volume[-1]
        vol_ma = self.vol_ma[-1]

        prev_res  = self.resistance[-2] if len(self.resistance) > 1 else self.resistance[-1]
        vol_ratio = vol / vol_ma if vol_ma > 0 else 0.0

        if self.position:
            if close < ema20:
                self.position.close()
            return

        c1 = close > ema20
        c2 = ema20 > ema50
        c3 = close > prev_res
        c4 = vol_ratio >= self.volume_mult
        c5 = self.rsi_low <= rsi <= self.rsi_high

        if not all([c1, c2, c3, c4, c5]):
            return
        if atr <= 0 or np.isnan(atr):
            return

        sl = close - self.atr_sl_mult * atr
        tp = close + self.atr_tp_mult * atr
        if sl <= 0 or sl >= close:
            return

        # Risk paritesi: işlem başına sermayenin %1'i riske edilir
        stop_dist_pct = (close - sl) / close
        size = min(self.risk_per_trade / stop_dist_pct, 0.95)
        if size <= 0:
            return
        self.buy(sl=sl, tp=tp, size=size)


CONFIGS = [
    ("A_mevcut (SL%3/TP%6, vol1.8)",      TrendBreakoutBT,  {}),
    ("B_ATR (1.5/3.0, vol1.8)",           TrendBreakoutATR, {}),
    ("C_ATR (1.5/3.0, vol1.5)",           TrendBreakoutATR, {"volume_mult": 1.5}),
    ("D_ATR (2.0/4.0, vol1.8)",           TrendBreakoutATR, {"atr_sl_mult": 2.0, "atr_tp_mult": 4.0}),
    ("E_mevcut vol1.5",                   TrendBreakoutBT,  {"volume_mult": 1.5}),
]


def main():
    data: dict[str, pd.DataFrame] = {}
    for sym in SYMBOLS:
        try:
            fr = fetch_symbol_data(sym, period=PERIOD, interval="1d")
            bt_df = _to_bt_df(fr.df)
            if len(bt_df) >= 200:
                data[sym] = bt_df
            else:
                print(f"SKIP {sym}: {len(bt_df)} bar", flush=True)
        except FetchError as exc:
            print(f"SKIP {sym}: {exc}", flush=True)

    print(f"\n{len(data)} sembol yüklendi, {len(CONFIGS)} konfigürasyon test ediliyor...\n", flush=True)

    summary_rows = []
    for name, cls, params in CONFIGS:
        rets, wins, trades, pfs, dds, sharpes = [], [], [], [], [], []
        for sym, bt_df in data.items():
            try:
                bt = Backtest(bt_df, cls, cash=_INITIAL_CASH,
                              commission=_COMMISSION, exclusive_orders=True)
                stats = bt.run(**params)
            except Exception as exc:
                print(f"  HATA {name} {sym}: {exc}", flush=True)
                continue
            n = int(stats["# Trades"])
            trades.append(n)
            rets.append(float(stats["Return [%]"]))
            dds.append(float(stats["Max. Drawdown [%]"]))
            if n > 0:
                wr = stats["Win Rate [%]"]
                pf = stats["Profit Factor"]
                sr = stats["Sharpe Ratio"]
                if not pd.isna(wr): wins.append(float(wr))
                if not pd.isna(pf) and np.isfinite(pf): pfs.append(float(pf))
                if not pd.isna(sr): sharpes.append(float(sr))

        row = {
            "config":        name,
            "avg_return":    round(np.mean(rets), 2) if rets else 0,
            "median_return": round(np.median(rets), 2) if rets else 0,
            "pos_count":     sum(1 for r in rets if r > 0),
            "total_trades":  sum(trades),
            "avg_winrate":   round(np.mean(wins), 1) if wins else 0,
            "avg_pf":        round(np.mean(pfs), 2) if pfs else 0,
            "avg_maxdd":     round(np.mean(dds), 1) if dds else 0,
        }
        summary_rows.append(row)
        print(f"{name:<32} ortGetiri %{row['avg_return']:>6} | medyan %{row['median_return']:>6} | "
              f"pozitif {row['pos_count']}/{len(rets)} | işlem {row['total_trades']:>3} | "
              f"kazanma %{row['avg_winrate']} | PF {row['avg_pf']} | maxDD %{row['avg_maxdd']}", flush=True)

    print("\n=== ÖZET TABLO ===", flush=True)
    df = pd.DataFrame(summary_rows)
    print(df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
