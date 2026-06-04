import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import backtesting.lib as lib
from loguru import logger
from typing import Optional
from app.data.fetcher import fetcher


class EMACrossoverBT(Strategy):
    ema_fast = 9
    ema_slow = 21
    rsi_period = 14

    def init(self):
        close = self.data.Close
        self.ema_f = self.I(lib.EMA, close, self.ema_fast)
        self.ema_s = self.I(lib.EMA, close, self.ema_slow)

    def next(self):
        if crossover(self.ema_f, self.ema_s):
            self.buy()
        elif crossover(self.ema_s, self.ema_f):
            self.sell()


class BacktestEngine:
    def run(
        self,
        symbol: str,
        strategy_class=EMACrossoverBT,
        cash: float = 100_000,
        commission: float = 0.001,
        period: Optional[str] = None,
        **strategy_params,
    ) -> Optional[dict]:
        df = fetcher.get_ohlcv(symbol, period=period or "2y")
        if df is None:
            return None

        # backtesting.py büyük harf sütun adı ister
        bt_df = df.rename(columns={
            "open": "Open", "high": "High",
            "low": "Low", "close": "Close", "volume": "Volume"
        })
        bt_df.index = pd.to_datetime(bt_df.index).tz_localize(None)

        try:
            bt = Backtest(
                bt_df,
                strategy_class,
                cash=cash,
                commission=commission,
                exclusive_orders=True,
            )
            if strategy_params:
                stats = bt.run(**strategy_params)
            else:
                stats = bt.run()

            result = {
                "symbol": symbol,
                "strategy": strategy_class.__name__,
                "total_return": round(float(stats["Return [%]"]), 2),
                "buy_hold_return": round(float(stats["Buy & Hold Return [%]"]), 2),
                "sharpe_ratio": round(float(stats["Sharpe Ratio"]), 3),
                "max_drawdown": round(float(stats["Max. Drawdown [%]"]), 2),
                "win_rate": round(float(stats["Win Rate [%]"]), 2),
                "total_trades": int(stats["# Trades"]),
                "profit_factor": round(float(stats.get("Profit Factor", 0) or 0), 3),
                "start_date": str(stats["Start"]),
                "end_date": str(stats["End"]),
            }

            logger.info(
                f"Backtest {symbol} | Getiri: %{result['total_return']} | "
                f"Sharpe: {result['sharpe_ratio']} | Kazanma: %{result['win_rate']}"
            )
            return result

        except Exception as e:
            logger.error(f"Backtest hatası {symbol}: {e}")
            return None

    def optimize(self, symbol: str, strategy_class=EMACrossoverBT, **param_ranges) -> Optional[dict]:
        df = fetcher.get_ohlcv(symbol)
        if df is None:
            return None

        bt_df = df.rename(columns={
            "open": "Open", "high": "High",
            "low": "Low", "close": "Close", "volume": "Volume"
        })
        bt_df.index = pd.to_datetime(bt_df.index).tz_localize(None)

        bt = Backtest(bt_df, strategy_class, cash=100_000, commission=0.001, exclusive_orders=True)
        stats, heatmap = bt.optimize(
            **param_ranges,
            maximize="Sharpe Ratio",
            return_heatmap=True,
        )
        return {"best_params": stats._strategy.__dict__, "sharpe": float(stats["Sharpe Ratio"])}


backtest_engine = BacktestEngine()
