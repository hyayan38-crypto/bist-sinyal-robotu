from loguru import logger
from typing import List
from app.data.fetcher import fetcher
from app.data.symbols import registry
from app.indicators.technical import indicators
from app.strategies import STRATEGIES
from app.strategies.base import SignalType
from app.risk.manager import risk_manager
from app.risk.market_filter import is_market_favorable, STATUS_UNAVAILABLE
from app.notifications.telegram import notifier


class SignalGenerator:
    async def run_for_symbol(
        self,
        symbol: str,
        open_signal_count: int = 0,
        market_filter_result=None,
    ) -> List[dict]:
        approved_signals = []

        df = fetcher.get_ohlcv(symbol)
        if df is None:
            return []

        df = indicators.add_all(df)

        for strategy in STRATEGIES:
            try:
                signal = strategy.generate_signal(df, symbol)
                if signal is None:
                    continue

                # AL sinyali → endeks filtresi uygula
                if signal.signal_type == SignalType.BUY:
                    mf = market_filter_result or is_market_favorable()
                    if mf.blocks_buy:
                        logger.info(
                            f"{symbol} | {strategy.name} AL sinyali endeks filtresiyle engellendi "
                            f"[{mf.status}]: {mf.reason}"
                        )
                        continue
                    if mf.status == STATUS_UNAVAILABLE:
                        logger.warning(
                            f"{symbol} | {strategy.name}: endeks filtresi mevcut değil, "
                            "sinyal onaylandı (fail-open)"
                        )

                assessment = risk_manager.assess(signal, open_signal_count)
                if not assessment.approved:
                    logger.info(
                        f"{symbol} | {strategy.name} reddedildi: {assessment.rejection_reason}"
                    )
                    continue

                signal_dict = {
                    "symbol": signal.symbol,
                    "signal_type": signal.signal_type.value,
                    "strategy": signal.strategy,
                    "strength": signal.strength,
                    "entry_price": signal.entry_price,
                    "stop_loss": assessment.adjusted_stop_loss,
                    "take_profit": assessment.adjusted_take_profit,
                    "notes": signal.notes,
                }
                approved_signals.append(signal_dict)

                await notifier.send_signal(signal, assessment)

            except Exception as e:
                logger.error(f"{symbol} | {strategy.name} sinyal hatası: {e}")

        return approved_signals

    async def run_all(self, open_signal_count: int = 0) -> List[dict]:
        # Tüm semboller için tek seferde piyasa kontrolü yap
        mf = is_market_favorable()
        if mf.blocks_buy:
            logger.warning(f"Endeks filtresi: AL sinyalleri kapalı — {mf.reason}")
        elif mf.status == STATUS_UNAVAILABLE:
            logger.warning("Endeks filtresi devre dışı, tüm sinyaller işleniyor")

        all_signals = []
        for symbol in registry.symbols:
            signals = await self.run_for_symbol(symbol, open_signal_count, mf)
            all_signals.extend(signals)

        logger.info(f"Toplam {len(all_signals)} sinyal üretildi ({len(registry)} sembol)")
        return all_signals


signal_generator = SignalGenerator()
