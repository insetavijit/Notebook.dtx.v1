import backtrader as bt
import pandas as pd
from .strategy import CrossSignalStrategy
from .trade_logger import TradeLogger
from .data_loader import CSVData

def run_backtest(df: pd.DataFrame,
                 strategy=CrossSignalStrategy,
                 instrumentName="tst",
                 cash=100000,
                 size=1,
                 valid_signals={1}):
    """
    Core backtest function — expects a DataFrame with
    columns [Open, High, Low, Close, Volume, Crossover, SL, TP]
    """
    data = CSVData(dataname=df, name=instrumentName)

    cerebro = bt.Cerebro()
    cerebro.adddata(data)
    cerebro.addstrategy(strategy, size=size, valid_signal_values=valid_signals)
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission=0.001)

    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analyzer')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')

    results = cerebro.run()
    strat = results[0]

    trades_df = strat.trade_logger.to_dataframe()

    # Stats
    trade_analyzer = strat.analyzers.trade_analyzer.get_analysis()
    total_trades = getattr(trade_analyzer.total, "closed", 0)
    wins = getattr(trade_analyzer.won, "total", 0)
    losses = getattr(trade_analyzer.lost, "total", 0)
    win_rate = (wins / total_trades * 100) if total_trades else 0

    total_pnl = trades_df["PnL"].sum() if not trades_df.empty else 0
    avg_rr = trades_df["Risk-Reward Ratio"].mean() if "Risk-Reward Ratio" in trades_df else None
    drawdown = strat.analyzers.drawdown.get_analysis()
    sharpe = strat.analyzers.sharpe.get_analysis().get("sharperatio")

    summary = {
        "Starting Cash": cash,
        "Ending Cash": cerebro.broker.getvalue(),
        "Total Return %": (cerebro.broker.getvalue() - cash) / cash * 100,
        "Total Trades": total_trades,
        "Wins": wins,
        "Losses": losses,
        "Win Rate %": win_rate,
        "Total PnL": total_pnl,
        "Average Risk-Reward Ratio": avg_rr,
        "Max Drawdown %": drawdown.max.drawdown if "max" in drawdown else 0,
        "Sharpe Ratio": sharpe
    }
    return trades_df, summary


def bktst(data, **kwargs):
    """
    Wrapper: accepts either DataFrame or CSV path.
    - If DataFrame → use directly
    - If str → try to load CSV into DataFrame
    """
    if isinstance(data, str):
        df = pd.read_csv(data, parse_dates=["DateTime"]).set_index("DateTime")
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise ValueError("bktst expects DataFrame or CSV path as input")

    # Ensure extra columns exist
    for col in ["Crossover", "SL", "TP"]:
        if col not in df.columns:
            df[col] = 0

    return run_backtest(df, **kwargs)
