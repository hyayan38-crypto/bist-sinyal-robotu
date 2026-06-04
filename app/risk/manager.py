from dataclasses import dataclass
from typing import Optional
from app.config import settings
from app.strategies.base import StrategySignal, SignalType
from loguru import logger


@dataclass
class RiskAssessment:
    approved: bool
    position_size_pct: float
    adjusted_stop_loss: Optional[float]
    adjusted_take_profit: Optional[float]
    rejection_reason: Optional[str] = None
    risk_reward_ratio: Optional[float] = None


class RiskManager:
    def __init__(self):
        self.max_position_size = settings.max_position_size
        self.stop_loss_pct = settings.stop_loss_pct
        self.take_profit_pct = settings.take_profit_pct
        self.max_open_signals = settings.max_open_signals
        self.min_signal_strength = settings.min_signal_strength

    def assess(self, signal: StrategySignal, open_signal_count: int = 0) -> RiskAssessment:
        # Sinyal gücü kontrolü
        if signal.strength < self.min_signal_strength:
            return RiskAssessment(
                approved=False,
                position_size_pct=0,
                adjusted_stop_loss=None,
                adjusted_take_profit=None,
                rejection_reason=f"Sinyal gücü yetersiz: {signal.strength:.2f} < {self.min_signal_strength}",
            )

        # Açık sinyal limiti
        if open_signal_count >= self.max_open_signals:
            return RiskAssessment(
                approved=False,
                position_size_pct=0,
                adjusted_stop_loss=None,
                adjusted_take_profit=None,
                rejection_reason=f"Maksimum açık sinyal sayısına ulaşıldı: {open_signal_count}",
            )

        # Stop loss ve take profit hesapla / düzelt
        entry = signal.entry_price
        if signal.signal_type == SignalType.BUY:
            sl = signal.stop_loss or round(entry * (1 - self.stop_loss_pct), 2)
            tp = signal.take_profit or round(entry * (1 + self.take_profit_pct), 2)
        else:
            sl = signal.stop_loss or round(entry * (1 + self.stop_loss_pct), 2)
            tp = signal.take_profit or round(entry * (1 - self.take_profit_pct), 2)

        # Risk/Reward hesapla
        risk = abs(entry - sl)
        reward = abs(tp - entry)
        rr_ratio = round(reward / risk, 2) if risk > 0 else 0

        if rr_ratio < 1.5:
            return RiskAssessment(
                approved=False,
                position_size_pct=0,
                adjusted_stop_loss=sl,
                adjusted_take_profit=tp,
                rejection_reason=f"Risk/Reward oranı yetersiz: {rr_ratio} < 1.5",
                risk_reward_ratio=rr_ratio,
            )

        # Pozisyon büyüklüğü — sinyal gücüne göre ölçekle
        position_size = round(self.max_position_size * signal.strength, 4)

        logger.info(
            f"{signal.symbol} | {signal.signal_type} | RR: {rr_ratio} | Pozisyon: %{position_size*100:.1f}"
        )

        return RiskAssessment(
            approved=True,
            position_size_pct=position_size,
            adjusted_stop_loss=sl,
            adjusted_take_profit=tp,
            risk_reward_ratio=rr_ratio,
        )


risk_manager = RiskManager()
