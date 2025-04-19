import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.cluster import KMeans
from matplotlib import cm
import numpy as np

st.set_page_config(page_title="Nifty 100 Cluster Dashboard", layout="wide")
st.title("📊 Enhanced Nifty 100 Clustering Dashboard")

# Tickers (shortened for testing)
nifty_100_tickers = [
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ITC.NS'
]

start_date = "2025-04-01"
end_date = datetime.today().strftime('%Y-%m-%d')

<<<<<<< HEAD
# Download data
with st.spinner("📥 Downloading data..."):
    df = yf.download(nifty_100_tickers, start=start_date, end=end_date)['Adj Close']
    df.dropna(axis=1, inplace=True)
=======
# Nifty 100 Tickers List (full list)
nifty_100_tickers = [...]
>>>>>>> parent of 8da5b44 (Update nifty100_app.py)

returns = ((df - df.iloc[0]) / df.iloc[0]) * 100

# Select number of clusters
num_clusters = st.slider("Select number of clusters:", min_value=2, max_value=5, value=3)

# Select date range for clustering
selected_range = st.date_input("Select range for clustering:", [pd.to_datetime(returns.index[0]), pd.to_datetime(returns.index[-1])])
start_sel, end_sel = pd.to_datetime(selected_range[0]), pd.to_datetime(selected_range[1])
returns_filtered = returns.loc[start_sel:end_sel]

# KMeans Clustering
X = returns_filtered.T
kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init='auto').fit(X)
labels = kmeans.labels_

# Plot cluster-wise
fig, ax = plt.subplots(figsize=(14, 7))
colors = cm.Set1(labels / (num_clusters + 1))
for i, stock in enumerate(X.index):
    ax.plot(returns_filtered.index, returns_filtered.loc[stock], label=stock, color=colors[i])
    ax.text(returns_filtered.index[-1], returns_filtered.loc[stock].iloc[-1], stock.replace('.NS',''), fontsize=8)

ax.set_title(f"Clustering Stock Returns from {start_sel.date()} to {end_sel.date()}")
ax.set_ylabel("% Return")
ax.set_xlabel("Date")
ax.grid(True, linestyle='--', alpha=0.5)
st.pyplot(fig)

# Display cluster info with average return
final_day_returns = returns_filtered.iloc[-1]
result = pd.DataFrame({"Stock": X.index, "Cluster": labels, "% Return": final_day_returns.values})
st.markdown("### 📋 Cluster Details")
st.dataframe(result.sort_values("Cluster"))

# Cluster Summary
cluster_summary = result.groupby("Cluster")["% Return"].agg(['mean', 'std', 'count']).rename(columns={'mean': 'Avg Return', 'std': 'Volatility', 'count': 'Num Stocks'})
st.markdown("### 📊 Cluster Summary")
st.dataframe(cluster_summary.round(2))
