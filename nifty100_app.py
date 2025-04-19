import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.cluster import KMeans
from matplotlib import cm
import numpy as np

# Config
st.set_page_config(page_title="Nifty 100 Daily Tracker", layout="wide")
st.title("📈 Nifty 100 Daily % Change Cluster Tracker")

# Date Range
start_date = "2025-04-01"
end_date = datetime.today().strftime('%Y-%m-%d')

# Nifty 100 Tickers List (full list)
nifty_100_tickers = [
    'ADANIENT.NS', 'ADANIGREEN.NS', 'ADANIPORTS.NS', 'ADANITRANS.NS', 'AMBUJACEM.NS',
    'APOLLOHOSP.NS', 'ASIANPAINT.NS', 'AUROPHARMA.NS', 'AXISBANK.NS', 'BAJAJ-AUTO.NS',
    'BAJAJFINSV.NS', 'BAJFINANCE.NS', 'BANDHANBNK.NS', 'BANKBARODA.NS', 'BEL.NS',
    'BERGEPAINT.NS', 'BHARATFORG.NS', 'BHARTIARTL.NS', 'BIOCON.NS', 'BOSCHLTD.NS',
    'BPCL.NS', 'BRITANNIA.NS', 'CANBK.NS', 'CHOLAFIN.NS', 'CIPLA.NS',
    'COALINDIA.NS', 'COLPAL.NS', 'DABUR.NS', 'DIVISLAB.NS', 'DLF.NS',
    'DRREDDY.NS', 'EICHERMOT.NS', 'ESCORTS.NS', 'GAIL.NS', 'GODREJCP.NS',
    'GRASIM.NS', 'HAVELLS.NS', 'HCLTECH.NS', 'HDFC.NS', 'HDFCBANK.NS',
    'HDFCLIFE.NS', 'HEROMOTOCO.NS', 'HINDALCO.NS', 'HINDPETRO.NS', 'HINDUNILVR.NS',
    'ICICIBANK.NS', 'ICICIPRULI.NS', 'IDEA.NS', 'IDFCFIRSTB.NS', 'IGL.NS',
    'INDHOTEL.NS', 'INDIGO.NS', 'INDUSINDBK.NS', 'INFY.NS', 'IOC.NS',
    'ITC.NS', 'JINDALSTEL.NS', 'JSWSTEEL.NS', 'KOTAKBANK.NS', 'L&TFH.NS',
    'LT.NS', 'LTI.NS', 'LTTS.NS', 'M&M.NS', 'M&MFIN.NS',
    'MARICO.NS', 'MARUTI.NS', 'MUTHOOTFIN.NS', 'NAUKRI.NS', 'NESTLEIND.NS',
    'NTPC.NS', 'ONGC.NS', 'PAGEIND.NS', 'PEL.NS', 'PETRONET.NS',
    'PFC.NS', 'PIDILITIND.NS', 'PIIND.NS', 'POWERGRID.NS', 'RECLTD.NS',
    'RELIANCE.NS', 'SAIL.NS', 'SBILIFE.NS', 'SBIN.NS', 'SHREECEM.NS',
    'SIEMENS.NS', 'SRF.NS', 'SUNPHARMA.NS', 'TATACHEM.NS', 'TATACONSUM.NS',
    'TATAMOTORS.NS', 'TATASTEEL.NS', 'TCS.NS', 'TECHM.NS', 'TITAN.NS',
    'TORNTPHARM.NS', 'TRENT.NS', 'TVSMOTOR.NS', 'UBL.NS', 'ULTRACEMCO.NS',
    'UPL.NS', 'VEDL.NS', 'VOLTAS.NS', 'WIPRO.NS', 'ZEEL.NS'
]

# Load data
with st.spinner('📈 Fetching stock data...'):
    raw_data = yf.download(nifty_100_tickers, start=start_date, end=end_date, group_by='ticker', auto_adjust=False)
    if raw_data.empty:
        st.error("Could not fetch data for any tickers. Please try again later.")
        st.stop()

    # Build price table using Close or Adj Close
    close_prices = pd.DataFrame()
    for ticker in nifty_100_tickers:
        try:
            if 'Adj Close' in raw_data[ticker]:
                close_prices[ticker] = raw_data[ticker]['Adj Close']
            elif 'Close' in raw_data[ticker]:
                close_prices[ticker] = raw_data[ticker]['Close']
        except KeyError:
            continue

    if close_prices.empty:
        st.error("Closing prices are unavailable for all tickers.")
        st.stop()
    else:
        data = close_prices

# Drop incomplete stocks
data.dropna(axis=1, inplace=True)

# Normalize returns relative to April 1 (as percentage change)
data_pct_change = ((data - data.iloc[0]) / data.iloc[0]) * 100

# Transpose so rows = stocks, columns = daily % returns
stock_returns = data_pct_change.T

# Apply clustering (KMeans with 10 clusters)
kmeans = KMeans(n_clusters=10, random_state=42, n_init='auto')
clusters = kmeans.fit_predict(stock_returns)

# Assign cluster labels
stock_returns['Cluster'] = clusters
cluster_colors = cm.get_cmap('Spectral', 10)
colors = [cluster_colors(c) for c in clusters]

# Plot line chart with cluster coloring
fig, ax = plt.subplots(figsize=(18, 10))
for i, stock in enumerate(stock_returns.index):
    ax.plot(data_pct_change.index, data_pct_change[stock], color=colors[i], linewidth=0.9, label=stock)

ax.set_title("% Price Change of Nifty 100 Stocks Since April 1, 2025 (Clustered)")
ax.set_ylabel("% Change")
ax.set_xlabel("Date")
ax.grid(True, linestyle='--', alpha=0.5)

# Optional legend
# ax.legend(loc='upper left', fontsize='xx-small', ncol=2, frameon=False)

# Show chart
st.pyplot(fig)

# Optional CSV download
csv = data_pct_change.reset_index()
st.download_button(
    label="📥 Download % Change Data as CSV",
    data=csv.to_csv(index=False),
    file_name='nifty100_percent_change.csv',
    mime='text/csv'
)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | Data from Yahoo Finance")
