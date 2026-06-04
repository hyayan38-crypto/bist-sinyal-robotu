import pytest
import pandas as pd
import numpy as np
from app.strategies.ema_crossover import EMACrossoverStrategy
from app.strategies.rsi_bb import RSIBollingerStrategy
from app.risk.manager import RiskManager
from app.strategies.base import SignalType


def make_df(n=100, trend="up") -> pd.DataFrame:
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(n) * 0.5 + (0.1 if trend == "up" else -0.1))
    df = pd.DataFrame({
        "open": prices * 0.999,
        "high": prices * 1.005,
        "low": prices * 0.995,
        "close": prices,
        "volume": np.random.randint(100_000, 1_000_000, n).astype(float),
    })
    return df


def add_indicators(df):
    from app.indicators.technical import add_indicators as _add
    return _add(df)


class TestEMACrossover:
    def test_returns_signal_or_none(self):
        df = add_indicators(make_df(200, "up"))
        strategy = EMACrossoverStrategy()
        result = strategy.generate_signal(df, "THYAO.IS")
        assert result is None or result.signal_type in (SignalType.BUY, SignalType.SELL)

    def test_insufficient_data_returns_none(self):
        df = add_indicators(make_df(20))
        strategy = EMACrossoverStrategy()
        assert strategy.generate_signal(df, "THYAO.IS") is None

    def test_signal_strength_in_range(self):
        df = add_indicators(make_df(200, "up"))
        strategy = EMACrossoverStrategy()
        result = strategy.generate_signal(df, "THYAO.IS")
        if result:
            assert 0.0 <= result.strength <= 1.0


class TestRiskManager:
    def setup_method(self):
        self.rm = RiskManager()
        self.rm.min_signal_strength = 0.5
        self.rm.max_open_signals = 5

    def _make_signal(self, strength=0.7, sl=95.0, tp=110.0):
        from app.strategies.base import StrategySignal
        return StrategySignal(
            symbol="TEST.IS",
            signal_type=SignalType.BUY,
            strategy="test",
            strength=strength,
            entry_price=100.0,
            stop_loss=sl,
            take_profit=tp,
        )

    def test_approves_good_signal(self):
        signal = self._make_signal(strength=0.75, sl=92.0, tp=115.0)
        assessment = self.rm.assess(signal, open_signal_count=0)
        assert assessment.approved

    def test_rejects_weak_signal(self):
        signal = self._make_signal(strength=0.2)
        assessment = self.rm.assess(signal, open_signal_count=0)
        assert not assessment.approved

    def test_rejects_too_many_signals(self):
        signal = self._make_signal(strength=0.9, sl=85.0, tp=120.0)
        assessment = self.rm.assess(signal, open_signal_count=10)
        assert not assessment.approved

    def test_rejects_bad_rr(self):
        signal = self._make_signal(strength=0.8, sl=99.0, tp=101.0)
        assessment = self.rm.assess(signal, open_signal_count=0)
        assert not assessment.approved

    def test_position_size_scaled_by_strength(self):
        s1 = self._make_signal(strength=0.6, sl=90.0, tp=115.0)
        s2 = self._make_signal(strength=0.9, sl=90.0, tp=115.0)
        a1 = self.rm.assess(s1)
        a2 = self.rm.assess(s2)
        if a1.approved and a2.approved:
            assert a2.position_size_pct > a1.position_size_pct
