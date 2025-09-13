# ploter/plot_strategy.py

import pandas as pd
import matplotlib.pyplot as plt

class CandlePlotter:
    def __init__(self, data: pd.DataFrame, chart_info: dict | None = None):
        """
        data: DataFrame with OHLC columns
        chart_info: dict with config like {"title": "My Chart"}
        """
        self.data = data.copy()
        self.chart_info = chart_info or {}

    def add_rsi(self, period: int = 14):
        """Placeholder for RSI indicator (to implement later)."""
        print(f"Adding RSI with period={period} (not implemented yet)")

    def plot(self):
        """Barebone plot using matplotlib (to replace with mplfinance later)."""
        plt.figure(figsize=(10, 5))
        plt.plot(self.data.index, self.data["Close"], label="Close Price")
        plt.title(self.chart_info.get("title", "Candle Plot"))
        plt.legend()
        plt.show()
