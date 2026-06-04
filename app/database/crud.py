from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from app.database.models import Signal, BacktestResult, SignalStatus
from datetime import datetime
from typing import List, Optional


async def create_signal(db: AsyncSession, signal_data: dict) -> Signal:
    signal = Signal(**signal_data)
    db.add(signal)
    await db.commit()
    await db.refresh(signal)
    return signal


async def get_active_signals(db: AsyncSession) -> List[Signal]:
    result = await db.execute(
        select(Signal).where(Signal.status == SignalStatus.ACTIVE)
    )
    return result.scalars().all()


async def get_signals_by_symbol(db: AsyncSession, symbol: str) -> List[Signal]:
    result = await db.execute(
        select(Signal).where(Signal.symbol == symbol).order_by(Signal.created_at.desc())
    )
    return result.scalars().all()


async def update_signal_status(db: AsyncSession, signal_id: int, status: str):
    await db.execute(
        update(Signal).where(Signal.id == signal_id).values(
            status=status, updated_at=datetime.utcnow()
        )
    )
    await db.commit()


async def save_backtest_result(db: AsyncSession, result_data: dict) -> BacktestResult:
    result = BacktestResult(**result_data)
    db.add(result)
    await db.commit()
    await db.refresh(result)
    return result


async def get_backtest_results(
    db: AsyncSession, strategy: Optional[str] = None, symbol: Optional[str] = None
) -> List[BacktestResult]:
    query = select(BacktestResult)
    if strategy:
        query = query.where(BacktestResult.strategy == strategy)
    if symbol:
        query = query.where(BacktestResult.symbol == symbol)
    result = await db.execute(query.order_by(BacktestResult.created_at.desc()))
    return result.scalars().all()
