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
st.title("📈 Nifty 100 Cluster-wise Performance Charts")

# Date Range
start_date = "2025-04-01"
end_date = datetime.today().strftime('%Y-%m-%d')

# Nifty 100 Tickers List (full list)
nifty_100_tickers = [...]

# Optional sector mapping (example, expand as needed)
sector_map = {
    'RELIANCE.NS': 'Energy', 'TCS.NS': 'IT', 'HDFCBANK.NS': 'Banking',
    'INFY.NS': 'IT', 'ICICIBANK.NS': 'Banking', 'HINDUNILVR.NS': 'FMCG',
    'ITC.NS': 'FMCG', 'SBIN.NS': 'Banking', 'AXISBANK.NS': 'Banking', 'LT.NS': 'Infrastructure'
    # Extend this to all stocks as needed
}

# Load data
with st.spinner('📈 Fetching stock data...'):
    raw_data = yf.download(nifty_100_tickers, start=start_date, end=end_date, group_by='ticker', auto_adjust=False)
    if raw_data.empty:
        st.error("Could not fetch data for any tickers. Please try again later.")
        st.stop()

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

# Transpose for clustering (rows = stocks, columns = daily returns)
stock_returns = data_pct_change.T

# Apply clustering (KMeans with 5 clusters)
kmeans = KMeans(n_clusters=5, random_state=42, n_init='auto')
clusters = kmeans.fit_predict(stock_returns)
stock_returns['Cluster'] = clusters

# Compute average and std return of each cluster to sort clusters
avg_returns = stock_returns.drop('Cluster', axis=1).mean(axis=1)
std_devs = stock_returns.drop('Cluster', axis=1).std(axis=1)
cluster_avg = avg_returns.groupby(clusters).mean()
cluster_order = cluster_avg.sort_values(ascending=False).index.tolist()
cluster_colors = cm.get_cmap('viridis', 5)

# Sector filter
selected_sector = st.selectbox("Filter by Sector (optional):", options=['All'] + sorted(set(sector_map.values())))

# Chart 1: All stocks + green dots for daily top 5
fig1, ax1 = plt.subplots(figsize=(18, 10))
for stock in data_pct_change.columns:
    if selected_sector == 'All' or sector_map.get(stock, '') == selected_sector:
        ax1.plot(data_pct_change.index, data_pct_change[stock], linewidth=0.8)

for date in data_pct_change.index:
    top5 = data_pct_change.loc[date].sort_values(ascending=False).head(5)
    for ticker in top5.index:
        if selected_sector == 'All' or sector_map.get(ticker, '') == selected_sector:
            ax1.plot(date, top5[ticker], 'go', markersize=4)
            ax1.text(date, top5[ticker]+1, ticker.replace('.NS',''), fontsize=6, color='green', ha='center')

ax1.set_title("All Stocks with Green Dots for Top 5 on Each Day")
ax1.set_ylabel("% Change")
ax1.set_xlabel("Date")
ax1.grid(True, linestyle='--', alpha=0.5)
st.pyplot(fig1)

# Dropdown to select cluster (optional)
selected_cluster = st.selectbox("Select Cluster to View Details:", options=cluster_order, format_func=lambda x: f"Cluster {x+1}")

# Charts 2–6: One for each cluster (ordered by avg return)
for i, cluster_id in enumerate(cluster_order):
    fig, ax = plt.subplots(figsize=(18, 10))
    members = stock_returns[stock_returns['Cluster'] == cluster_id].index
    for stock in members:
        if selected_sector == 'All' or sector_map.get(stock, '') == selected_sector:
            ax.plot(data_pct_change.index, data_pct_change[stock], color=cluster_colors(cluster_id), linewidth=1.2, label=stock)
    avg_return = data_pct_change[members].iloc[-1].mean()
    std_return = data_pct_change[members].iloc[-1].std()
    ax.set_title(f"Cluster {cluster_id+1} - {len(members)} Stocks | Avg: {avg_return:.2f}% | Std: {std_return:.2f}%")
    ax.set_ylabel("% Change")
    ax.set_xlabel("Date")
    ax.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig)

# CSV Download
csv = data_pct_change.reset_index()
st.download_button(
    label="📥 Download % Change Data as CSV",
    data=csv.to_csv(index=False),
    file_name='nifty100_percent_change.csv',
    mime='text/csv'
)

# Download cluster label mapping
cluster_map = pd.DataFrame({'Stock': stock_returns.index, 'Cluster': stock_returns['Cluster']})
st.download_button(
    label="📥 Download Cluster Labels",
    data=cluster_map.to_csv(index=False),
    file_name='nifty100_clusters.csv',
    mime='text/csv'
)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | Data from Yahoo Finance")
