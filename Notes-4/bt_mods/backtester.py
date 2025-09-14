import backtrader as bt
from .data_loader import load_csv_data, CSVData
from .strategy import CrossSignalStrategy

def run_backtest(df, strategy=CrossSignalStrategy, instrumentName="tst", cash=100000, size=1, valid_signals={1}):
    data = CSVData(dataname=df, name=instrumentName)

    cerebro = bt.Cerebro()
    cerebro.adddata(data)
    cerebro.addstrategy(strategy, size=size, valid_signal_values=valid_signals)
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission=0.001)

    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trade_analyzer")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")

    results = cerebro.run()
    strat = results[0]

    trades_df = strat.trade_logger.to_dataframe()

    trade_analyzer = strat.analyzers.trade_analyzer.get_analysis()
    total_trades = trade_analyzer.total.closed if hasattr(trade_analyzer.total, "closed") else 0
    wins = trade_analyzer.won.total if hasattr(trade_analyzer.won, "total") else 0
    losses = trade_analyzer.lost.total if hasattr(trade_analyzer.lost, "total") else 0
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0

    summary = {
        "Starting Cash": cash,
        "Ending Cash": cerebro.broker.getvalue(),
        "Total Trades": total_trades,
        "Wins": wins,
        "Losses": losses,
        "Win Rate %": win_rate,
        "Total PnL": trades_df["PnL"].sum() if not trades_df.empty else 0,
    }
    return trades_df, df, cerebro, summary


def bktst(data, **kwargs):
    """Wrapper: accepts CSV path or DataFrame."""
    import pandas as pd
    if isinstance(data, str):
        df = load_csv_data(data)
    elif isinstance(data, pd.DataFrame):
        df = data
    else:
        raise ValueError("bktst expects CSV path or DataFrame")
    return run_backtest(df, **kwargs)

