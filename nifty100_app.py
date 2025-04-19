import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Step 1: Define the date range
start_date = "2025-04-01"
end_date = datetime.today().strftime('%Y-%m-%d')

# Step 2: List of Nifty 100 companies (example subset, you can expand to full 100)
nifty_100_tickers = [
    'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS',
    'HINDUNILVR.NS', 'LT.NS', 'SBIN.NS', 'AXISBANK.NS', 'ITC.NS'
    # Add all Nifty 100 tickers here
]

# Step 3: Download stock data
data = yf.download(nifty_100_tickers, start=start_date, end=end_date)['Adj Close']

# Step 4: Calculate percentage change from April 1 to today
returns = (data.iloc[-1] - data.iloc[0]) / data.iloc[0] * 100
returns = returns.sort_values(ascending=False)

# Step 5: Plotting
plt.figure(figsize=(15, 8))
bars = plt.bar(returns.index, returns.values, color='grey')

# Highlight top 10
for i in range(10):
    bars[i].set_color('green')

# Labels and title
plt.xticks(rotation=90)
plt.ylabel('Return % since 1 April 2025')
plt.title('Nifty 100 Performance (Top 10 Highlighted)')

# Annotate top 10
for i in range(10):
    plt.text(i, returns.values[i]+0.5, f'{returns.values[i]:.1f}%', ha='center', fontsize=9)

plt.tight_layout()
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.show()
