import yfinance as yf
import matplotlib.pyplot as plt
import trendln

symbol = "GC=F"   # Gold Futures on Yahoo
df = yf.download(symbol, interval="1m", period="7d")  # use 7d instead of 1mo

if df.empty:
    raise ValueError("No data fetched. Check symbol/interval/period combo.")

highs = df['High']
lows = df['Low']

min_idx, min_trendlines = trendln.calc_support_resistance(lows, accuracy=5)
max_idx, max_trendlines = trendln.calc_support_resistance(highs, accuracy=5)

plt.figure(figsize=(14,7))
plt.plot(df['Close'], label="Close", color="blue")

for line in min_trendlines:
    plt.plot(df.index[line[0]], line[1], 'g--', lw=1)
for line in max_trendlines:
    plt.plot(df.index[line[0]], line[1], 'r--', lw=1)

plt.title("Gold Futures (1m, 7d) with Support/Resistance Channels")
plt.legend()
plt.show()

