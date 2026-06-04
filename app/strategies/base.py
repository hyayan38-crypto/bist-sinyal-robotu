from abc import ABC, abstractmethod
import pandas as pd
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class SignalType(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class StrategySignal:
    symbol: str
    signal_type: SignalType
    strategy: str
    strength: float          # 0.0 - 1.0
    entry_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    notes: str = ""


class BaseStrategy(ABC):
    name: str = "base"

    @abstractmethod
    def generate_signal(self, df: pd.DataFrame, symbol: str) -> Optional[StrategySignal]:
        """İndikatörlü DataFrame alır, sinyal üretir."""
        ...

    def _validate_df(self, df: pd.DataFrame, required_cols: list) -> bool:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            return False
        if len(df) < 50:
            return False
        return True
