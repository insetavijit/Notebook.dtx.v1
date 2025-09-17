import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import trendln

# 1. Download last 1 month of 1-min gold data
symbol = "GC=F"   # Gold Futures on Yahoo (XAUUSD alternative)
df = yf.download(symbol, interval="1m", period="1mo")

# Ensure we have data
print(df.tail())

# 2. Extract highs and lows
highs = df['High']
lows = df['Low']

# 3. Calculate support & resistance trendlines (channels)
min_idx, min_trendlines = trendln.calc_support_resistance(lows, accuracy=5)
max_idx, max_trendlines = trendln.calc_support_resistance(highs, accuracy=5)

# 4. Plot chart with detected channel
plt.figure(figsize=(14,7))
plt.plot(df['Close'], label="Close", color="blue")

# Plot support lines (green)
for line in min_trendlines:
    plt.plot(df.index[line[0]], line[1], 'g--', lw=1)

# Plot resistance lines (red)
for line in max_trendlines:
    plt.plot(df.index[line[0]], line[1], 'r--', lw=1)

plt.title("Gold Futures (1m) with Trendln Support/Resistance Channels")
plt.xlabel("Time")
plt.ylabel("Price (USD)")
plt.legend()
plt.show()

