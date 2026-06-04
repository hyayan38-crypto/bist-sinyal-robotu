import pandas as pd
from typing import Optional
from app.strategies.base import BaseStrategy, StrategySignal, SignalType


class RSIBollingerStrategy(BaseStrategy):
    """RSI aşırı bölge + Bollinger Band dokunuşu stratejisi."""

    name = "rsi_bollinger"

    def __init__(self, rsi_low: float = 30, rsi_high: float = 70):
        self.rsi_low = rsi_low
        self.rsi_high = rsi_high

    def generate_signal(self, df: pd.DataFrame, symbol: str) -> Optional[StrategySignal]:
        required = ["rsi_14", "bb_upper", "bb_lower", "bb_pct", "atr_14", "close"]
        if not self._validate_df(df, required):
            return None

        df = df.dropna(subset=required)
        if len(df) < 3:
            return None

        curr = df.iloc[-1]
        prev = df.iloc[-2]

        close  = curr["close"]
        atr    = curr["atr_14"]
        rsi    = curr["rsi_14"]
        bb_pct = curr["bb_pct"]

        oversold      = rsi < self.rsi_low  and bb_pct < 0.1
        rsi_recovering = prev["rsi_14"] < self.rsi_low  and curr["rsi_14"] > prev["rsi_14"]

        overbought  = rsi > self.rsi_high and bb_pct > 0.9
        rsi_falling = prev["rsi_14"] > self.rsi_high and curr["rsi_14"] < prev["rsi_14"]

        if oversold and rsi_recovering:
            strength = round((self.rsi_low - rsi) / self.rsi_low * 0.7 + (0.1 - bb_pct) * 3, 3)
            strength = min(1.0, max(0.0, strength))
            return StrategySignal(
                symbol=symbol,
                signal_type=SignalType.BUY,
                strategy=self.name,
                strength=strength,
                entry_price=close,
                stop_loss=round(curr["bb_lower"] - atr * 0.5, 2),
                take_profit=round(curr["bb_upper"], 2),
                notes=f"RSI14 aşırı satım + BB alt band | RSI: {rsi:.1f} BB%: {bb_pct:.2f}",
            )

        if overbought and rsi_falling:
            strength = round((rsi - self.rsi_high) / (100 - self.rsi_high) * 0.7 + (bb_pct - 0.9) * 3, 3)
            strength = min(1.0, max(0.0, strength))
            return StrategySignal(
                symbol=symbol,
                signal_type=SignalType.SELL,
                strategy=self.name,
                strength=strength,
                entry_price=close,
                stop_loss=round(curr["bb_upper"] + atr * 0.5, 2),
                take_profit=round(curr["bb_lower"], 2),
                notes=f"RSI14 aşırı alım + BB üst band | RSI: {rsi:.1f} BB%: {bb_pct:.2f}",
            )

        return None
