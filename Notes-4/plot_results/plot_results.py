# plot_results.py
from .plotter import CandlestickPlotter

def plot_results(ohlc_df, trades_df=None, start_row=0, end_row=None, title=None):
    """
    Wrapper function for plotting candlestick chart with trades.
    
    Args:
        ohlc_df (pd.DataFrame): OHLC data
        trades_df (pd.DataFrame, optional): Trade data
        start_row (int): Start index of slice
        end_row (int, optional): End index of slice
        title (str, optional): Chart title
    """
    plotter = CandlestickPlotter()
    plotter.plot_candles_with_rects(
        ohlc_df=ohlc_df,
        trades_df=trades_df,
        start_row=start_row,
        end_row=end_row,
        title=title
    )
