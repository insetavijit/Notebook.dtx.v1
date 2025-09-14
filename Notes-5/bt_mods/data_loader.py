import pandas as pd
import backtrader as bt

def load_csv_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path, parse_dates=["DateTime"])
    df.set_index("DateTime", inplace=True)

    # Ensure required columns
    for col in ["Crossover", "SL", "TP"]:
        if col not in df.columns:
            df[col] = 0
    return df


class CSVData(bt.feeds.PandasData):
    """Custom PandasData with extra lines."""
    lines = ('Crossover', 'SL', 'TP')
    params = (
        ('datetime', None),
        ('open', 'Open'),
        ('high', 'High'),
        ('low', 'Low'),
        ('close', 'Close'),
        ('volume', -1),
        ('openinterest', -1),
        ('Crossover', -1),
        ('SL', -1),
        ('TP', -1),
    )
