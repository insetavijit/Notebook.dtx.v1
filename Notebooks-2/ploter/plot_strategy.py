# ploter/plot_strategy.py

import pandas as pd
import mplfinance as mpf

class CandlePlotter:
    def __init__(self, data: pd.DataFrame, chart_info: dict | None = None):
        """
        data: DataFrame with OHLC columns (Open, High, Low, Close, Volume)
        chart_info: dict with optional config like {"title": "My Chart"}
        """
        self.data = data.copy()
        self.chart_info = chart_info or {}

    def plot(self):
        """Just print the candlestick chart."""
        mpf.plot(
            self.data,
            type="candle",
            volume=True,
            title=self.chart_info.get("title", "Candle Chart"),
            style="yahoo"
        )
