
import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Config
st.set_page_config(page_title="Nifty 100 Daily Tracker", layout="wide")
st.title("📊 Nifty 100 Daily Performance Dashboard")

# Date Range
start_date = "2025-04-01"
end_date = datetime.today().strftime('%Y-%m-%d')

# Nifty 100 Tickers (shortened list for demo, use full list in prod)
nifty_100_tickers = [
    'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS',
    'HINDUNILVR.NS', 'LT.NS', 'SBIN.NS', 'AXISBANK.NS', 'ITC.NS'
    # Add all Nifty 100 tickers here
]

# Load data
with st.spinner('📈 Fetching stock data...'):
    raw_data = yf.download(nifty_100_tickers, start=start_date, end=end_date)
    if 'Adj Close' in raw_data.columns:
        data = raw_data['Adj Close']
    else:
        st.error("Could not fetch adjusted close prices for any tickers. Please try again later.")
        st.stop()

# Drop incomplete stocks
data.dropna(axis=1, inplace=True)

# Calculate returns
returns = ((data.iloc[-1] - data.iloc[0]) / data.iloc[0]) * 100
returns = returns.sort_values(ascending=False)

# Download NIFTY Index and compare
nifty_index = yf.download('^NSEI', start=start_date, end=end_date)['Adj Close']
nifty_return = (nifty_index.iloc[-1] - nifty_index.iloc[0]) / nifty_index.iloc[0] * 100

# Plot returns
fig, ax = plt.subplots(figsize=(15, 8))
bars = ax.bar(returns.index, returns.values, color='gray')

# Highlight top 10
for i in range(min(10, len(bars))):
    bars[i].set_color('green')
    ax.text(i, returns.values[i]+0.5, f"{returns.values[i]:.1f}%", ha='center', fontsize=8)

# Add NIFTY index line
ax.axhline(y=nifty_return, color='red', linestyle='--', label=f'NIFTY Index Return: {nifty_return:.2f}%')

# Label formatting
ax.set_ylabel("Return %")
ax.set_title(f"Performance of Nifty 100 Stocks from {start_date} to {end_date}")
ax.set_xticks(range(len(returns)))
ax.set_xticklabels(returns.index, rotation=90)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.5)

# Show chart
st.pyplot(fig)

# Optional CSV download
csv = returns.reset_index()
csv.columns = ['Ticker', 'Return %']
st.download_button(
    label="📥 Download Returns as CSV",
    data=csv.to_csv(index=False),
    file_name='nifty100_returns.csv',
    mime='text/csv'
)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | Data from Yahoo Finance")
