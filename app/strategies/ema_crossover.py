import pandas as pd
from typing import Optional
from app.strategies.base import BaseStrategy, StrategySignal, SignalType


class EMACrossoverStrategy(BaseStrategy):
    """EMA 9/21 kesişimi + RSI filtresi ile sinyal üretir."""

    name = "ema_crossover"

    def __init__(self, rsi_oversold: float = 40, rsi_overbought: float = 60):
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought

    def generate_signal(self, df: pd.DataFrame, symbol: str) -> Optional[StrategySignal]:
        required = ["ema_20", "ema_50", "rsi_14", "atr_14", "close"]
        if not self._validate_df(df, required):
            return None

        df = df.dropna(subset=required)
        if len(df) < 3:
            return None

        curr = df.iloc[-1]
        prev = df.iloc[-2]

        close = curr["close"]
        atr   = curr["atr_14"]
        rsi   = curr["rsi_14"]

        golden_cross = prev["ema_20"] <= prev["ema_50"] and curr["ema_20"] > curr["ema_50"]
        death_cross  = prev["ema_20"] >= prev["ema_50"] and curr["ema_20"] < curr["ema_50"]

        if golden_cross and rsi < self.rsi_overbought:
            strength = self._calc_strength(rsi, "BUY", curr["ema_20"], curr["ema_50"])
            return StrategySignal(
                symbol=symbol,
                signal_type=SignalType.BUY,
                strategy=self.name,
                strength=strength,
                entry_price=close,
                stop_loss=round(close - 1.5 * atr, 2),
                take_profit=round(close + 2.5 * atr, 2),
                notes=f"EMA20/50 golden cross | RSI14: {rsi:.1f}",
            )

        if death_cross and rsi > self.rsi_oversold:
            strength = self._calc_strength(rsi, "SELL", curr["ema_20"], curr["ema_50"])
            return StrategySignal(
                symbol=symbol,
                signal_type=SignalType.SELL,
                strategy=self.name,
                strength=strength,
                entry_price=close,
                stop_loss=round(close + 1.5 * atr, 2),
                take_profit=round(close - 2.5 * atr, 2),
                notes=f"EMA20/50 death cross | RSI14: {rsi:.1f}",
            )

        return None

    def _calc_strength(
        self, rsi: float, direction: str, ema_fast: float, ema_slow: float
    ) -> float:
        divergence = abs(ema_fast - ema_slow) / ema_slow
        if direction == "BUY":
            rsi_score = max(0, (70 - rsi) / 70)
        else:
            rsi_score = max(0, (rsi - 30) / 70)
        return round(min(1.0, 0.5 * rsi_score + 0.5 * min(divergence * 100, 1.0)), 3)
