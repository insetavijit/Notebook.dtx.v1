import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
from typing import Optional
from .config import DEFAULT_PLOT_CONFIG
from .utils import add_rectangle


class CandlestickPlotter:
    """Class to create candlestick charts with trade rectangles."""

    def __init__(self, config: Optional[dict] = None):
        self.config = config or DEFAULT_PLOT_CONFIG

    def plot_candles_with_rects(self, ohlc_df: pd.DataFrame,
                                trades_df: Optional[pd.DataFrame] = None,
                                start_row: int = 0, end_row: Optional[int] = None,
                                title: Optional[str] = None) -> None:
        """Plot OHLC candlestick chart with optional trade rectangles."""
        view = ohlc_df.iloc[start_row:end_row].copy()
        rects = []

        if trades_df is not None:
            for _, trade in trades_df.iterrows():
                try:
                    entry_idx = view.index.get_loc(trade['Entry Time'])
                    exit_idx = view.index.get_loc(trade['Exit Time'])
                    width = exit_idx - entry_idx + 1

                    # Main trade rectangle
                    add_rectangle(rects, entry_idx, trade['Entry Price'], trade['Exit Price'],
                                  width=width, **self.config['main_rect'])

                    # Stop loss rectangle
                    if pd.notna(trade.get('SL')):
                        add_rectangle(rects, entry_idx, trade['Entry Price'], trade['SL'],
                                      width=width, **self.config['sl_rect'])

                    # Take profit rectangle
                    if pd.notna(trade.get('TP')):
                        add_rectangle(rects, entry_idx, trade['Entry Price'], trade['TP'],
                                      width=width, **self.config['tp_rect'])

                    # Golden outline for winning trades
                    if trade.get('Exit Status', '').upper() == 'TP':
                        main_low = min(trade['Entry Price'], trade['Exit Price'])
                        main_high = max(trade['Entry Price'], trade['Exit Price'])
                        add_rectangle(rects, entry_idx, main_low, main_high, width=width,
                                      fill_color='none', **self.config['golden_outline'])

                except KeyError as e:
                    continue
                except Exception as e:
                    continue

        # Plot chart
        fig, axlist = mpf.plot(
            view,
            type='candle',
            style=self.config['style'],
            returnfig=True,
            figsize=self.config['figsize'],
            title=title or self.config['title']
        )
        ax = axlist[0]

        # Add rectangles
        for rect in rects:
            ax.add_patch(rect)

        plt.show()
