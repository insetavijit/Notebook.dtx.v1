# %%
# %%
import numpy as np
import pandas as pd
import mplfinance as mpf

# Load CSV
df = pd.read_csv('auto-indig.csv')

# Parse action list from a multi-line string
Query = """
Close above EMA_21
"""
arr = np.array([line.split() for line in Query.splitlines() if line.strip()])
actlist = arr[:, arr.shape[1] // 2].tolist()  # middle word of each line

actlist

# %%
def above(df: pd.DataFrame, x: str, y: str, new_col: str = None) -> pd.DataFrame:
    new_col = new_col or f"{x}_above_{y}"
    df[new_col] = (df[x] > df[y]).astype(int)
    return df

def below(df: pd.DataFrame, x: str, y: str, new_col: str = None) -> pd.DataFrame:
    new_col = new_col or f"{x}_below_{y}"
    df[new_col] = (df[x] < df[y]).astype(int)
    return df

def crossed_up(df: pd.DataFrame, x: str, y: str, new_col: str = None) -> pd.DataFrame:
    new_col = new_col or f"{x}_cross_up_{y}"
    diff = df[x] - df[y]
    df[new_col] = ((diff > 0) & (diff.shift(1) <= 0)).astype(int)
    return df

def crossed_dn(df: pd.DataFrame, x: str, y: str, new_col: str = None) -> pd.DataFrame:
    new_col = new_col or f"{x}_cross_down_{y}"
    diff = df[x] - df[y]
    df[new_col] = ((diff < 0) & (diff.shift(1) >= 0)).astype(int)
    return df

# %%
def cabr(df: pd.DataFrame, action: str, clm1: str, clm2: str):
    return {
        "above": above,
        "below": below,
        "crossed_up": crossed_up,
        "crossed_dn": crossed_dn
    }.get(action, lambda df, x, y: "err")(df, clm1, clm2)

# %%
pltdf = df.iloc[50:110].copy()
pltdf['DateTime'] = pd.to_datetime(pltdf['DateTime'])
pltdf.set_index('DateTime', inplace=True)

pltdf.T

# %%
# Suppose you have columns in your df like 'Close', 'EMA21', 'Rsi_14', etc.
# And actlist currently is something like ['above', 'above_lvl'] or similar
# But we need to know what columns to compare. You can parse from your Query too.

# Let's parse column pairs from the Query
pairs = []
for line in Query.splitlines():
    if line.strip():
        words = line.split()
        col1 = words[0]  # first column
        action = words[1]  # action: above / crossed_up / etc.
        col2 = words[-1]  # second column / level
        pairs.append((action, col1, col2))

# Now run cabr automatically for all
for action, col1, col2 in pairs:
    # If col2 is a number (like '30'), convert to float
    try:
        col2_val = float(col2)
        pltdf[col2] = col2_val  # add as a constant column for comparison
    except:
        col2_val = col2
    pltdf = cabr(pltdf, action, col1, col2_val)


pltdf.T



