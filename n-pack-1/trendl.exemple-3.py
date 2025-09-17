import yfinance as yf
import mplfinance as mpf

# -------------------------------
# Parameters
# -------------------------------
symbol = "AAPL"          # Reliable symbol
interval = "1d"          # Daily interval
period = "1mo"           # Last 1 month
mav = (5, 10)            # Moving averages
output_file = "candlestick_final.png"

# -------------------------------
# Download data
# -------------------------------
df = yf.download(symbol, interval=interval, period=period)

if df.empty:
    raise ValueError("No data fetched. Check symbol/interval/period.")

# -------------------------------
# Plot candlestick chart
# -------------------------------
mpf.plot(
    df,
    type='candle',
    style='charles',
    title=f"{symbol} - Daily Candlestick ({period})",
    volume=True,
    mav=mav,
    figratio=(16, 9),
    figscale=1.2,
    savefig=output_file
)

print(f"Candlestick chart saved as '{output_file}'")

