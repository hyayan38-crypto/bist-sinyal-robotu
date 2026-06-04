from app.strategies.ema_crossover import EMACrossoverStrategy
from app.strategies.rsi_bb import RSIBollingerStrategy
from app.strategies.trend_breakout import TrendBreakoutStrategy
from app.strategies.pre_breakout_squeeze import PreBreakoutSqueezeStrategy

STRATEGIES = [
    EMACrossoverStrategy(),
    RSIBollingerStrategy(),
    TrendBreakoutStrategy(),
    PreBreakoutSqueezeStrategy(),
]

__all__ = [
    "STRATEGIES",
    "EMACrossoverStrategy",
    "RSIBollingerStrategy",
    "TrendBreakoutStrategy",
    "PreBreakoutSqueezeStrategy",
]
