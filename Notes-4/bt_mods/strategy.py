import backtrader as bt
from .trade_logger import TradeLogger
from .logging_config import setup_logger

logger = setup_logger()

class CrossSignalStrategy(bt.Strategy):
    params = (
        ("size", 1),
        ("valid_signal_values", {1}),
    )

    def __init__(self):
        self.cross_signal = self.datas[0].Crossover
        self.sl = self.datas[0].SL
        self.tp = self.datas[0].TP
        self.trade_logger = TradeLogger()
        self.last_entry = {}

    def next(self):
        if self.position:
            return

        signal = self.cross_signal[0]
        if signal not in self.params.valid_signal_values:
            return

        entry_price = self.data.close[0]
        stop_loss = self.sl[0]
        take_profit = self.tp[0]

        if not self._validate_trade(entry_price, stop_loss, take_profit):
            logger.warning(f"Invalid trade params: entry={entry_price}, SL={stop_loss}, TP={take_profit}")
            return

        # Store entry details
        self.last_entry = {
            "price": entry_price,
            "sl": stop_loss,
            "tp": take_profit,
            "datetime": self.data.datetime.datetime(0),
            "index": len(self) - 1,
            "exit_status": "Other"
        }

        self.buy_bracket(size=self.params.size, price=entry_price, stopprice=stop_loss, limitprice=take_profit)
        logger.info(f"Placed order: price={entry_price}, SL={stop_loss}, TP={take_profit}")

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.exectype == bt.Order.Stop:
                self.last_entry["exit_status"] = "SL"
            elif order.exectype == bt.Order.Limit:
                self.last_entry["exit_status"] = "TP"
            else:
                self.last_entry["exit_status"] = "Other"

    def notify_trade(self, trade):
        if trade.isclosed:
            self.trade_logger.log_trade(
                trade_id=len(self.trade_logger.trades) + 1,
                symbol=self.data._name,
                strategy_name=type(self).__name__,
                entry_time=self.last_entry.get("datetime"),
                exit_time=self.data.datetime.datetime(0),
                entry_price=self.last_entry.get("price"),
                exit_price=trade.price,
                position_size=trade.size,
                direction="Long",
                pnl=trade.pnl,
                entry_index=self.last_entry.get("index"),
                exit_index=len(self) - 1,
                stop_loss=self.last_entry.get("sl"),
                take_profit=self.last_entry.get("tp"),
                commission=self.broker.getcommissioninfo(self.data).getcommission(trade.size, trade.price),
                spread=self.data.high[0] - self.data.low[0]
            )
            self.trade_logger.trades[-1]["Exit Status"] = self.last_entry.get("exit_status", "Other")

    def stop(self):
        df = self.trade_logger.to_dataframe()
        if df.empty:
            logger.info("No trades executed.")
        else:
            logger.info(f"Total trades executed: {len(df)}")

    def _validate_trade(self, price, sl, tp):
        return all(isinstance(x, (int, float)) for x in [price, sl, tp]) and price > sl and price < tp
