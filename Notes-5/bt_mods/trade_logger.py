import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime

class TradeLogger:
    """Centralized trade logging to DataFrame."""
    def __init__(self):
        self.trades: List[Dict[str, Any]] = []

    def log_trade(
        self,
        trade_id: int,
        symbol: str,
        strategy_name: str,
        entry_time: datetime,
        exit_time: datetime,
        entry_price: float,
        exit_price: float,
        position_size: int,
        direction: str,
        pnl: float,
        entry_index: int,
        exit_index: int,
        stop_loss: Optional[float],
        take_profit: Optional[float],
        commission: float,
        spread: float
    ) -> None:
        self.trades.append({
            "Trade ID": trade_id,
            "Symbol": symbol,
            "Strategy Name": strategy_name,
            "Entry Time": entry_time,
            "Exit Time": exit_time,
            "Entry Price": entry_price,
            "Exit Price": exit_price,
            "Position Size": position_size,
            "Direction": direction,
            "PnL": pnl,
            "PnL %": (pnl / entry_price * 100) if entry_price else None,
            "Candle Index Entry": entry_index,
            "Candle Index Exit": exit_index,
            "SL": stop_loss,
            "TP": take_profit,
            "Risk-Reward Ratio": ((take_profit - entry_price) / (entry_price - stop_loss)
                                  if stop_loss and take_profit and entry_price > stop_loss else None),
            "Commissions": commission,
            "Spread": spread
        })

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.trades)
