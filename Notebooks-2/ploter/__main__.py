# ploter/__main__.py

import argparse
from ploter.utils import load_csv
from ploter.plot_strategy import CandlePlotter

def main():
    parser = argparse.ArgumentParser(description="Plot trading strategies")
    parser.add_argument("--file", required=True, help="Path to CSV file")
    parser.add_argument(
        "--with", nargs="*", default=[], help="Indicators (e.g., rsi ema macd)"
    )

    args = parser.parse_args()

    # Load data
    df = load_csv(args.file)

    # Create plotter
    plotter = CandlePlotter(df)

    # Dynamically add indicators
    for ind in args.with:
        method = getattr(plotter, f"add_{ind}", None)
        if callable(method):
            method()
        else:
            print(f"[WARN] Indicator '{ind}' not found.")

    # Plot
    plotter.plot()

if __name__ == "__main__":
    main()
