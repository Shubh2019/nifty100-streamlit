import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.cluster import KMeans
import numpy as np
from matplotlib import cm

st.set_page_config(page_title="Nifty 100 Daily Returns", layout="wide")
st.title("📈 Nifty 100 Performance Analysis (Clusters + Top Performers)")

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

start_date = "2025-04-01"
end_date = datetime.today().strftime('%Y-%m-%d')

with st.spinner("📥 Fetching Nifty 100 data..."):
    try:
        df = yf.download(nifty_100_tickers, start=start_date, end=end_date)['Close']
    except Exception as e:
        st.error(f"Download failed: {e}")
        st.stop()

    df.dropna(axis=1, inplace=True)

    if df.empty:
        st.error("Downloaded data is empty. Try again later.")
        st.stop()

returns = ((df - df.iloc[0]) / df.iloc[0]) * 100
X = returns.T

st.markdown("### 📊 Nifty 100 Stock Returns with Top 5 Daily Markers")
fig, ax = plt.subplots(figsize=(14, 7))

for ticker in df.columns:
    ax.plot(returns.index, returns[ticker], color='gray', alpha=0.3, linewidth=0.8)

final_returns = returns.iloc[-1].sort_values(ascending=False)
top10 = final_returns.head(10)
for ticker in top10.index:
    ax.plot(returns.index, returns[ticker], linewidth=1.5, label=ticker)

# ✅ Mark top 5 performers of each day with green dots and labels
for date in returns.index:
    top5 = returns.loc[date].sort_values(ascending=False).head(5)
    for ticker in top5.index:
        ax.plot(date, returns.loc[date, ticker], 'go', markersize=4)
        ax.text(date, returns.loc[date, ticker], ticker.replace(".NS", ""), fontsize=5, ha='right', color='green')

ax.set_title("Nifty 100 Performance Since April 1, 2025")
ax.set_ylabel("% Return")
ax.set_xlabel("Date")
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(loc='upper left', fontsize='small')
st.pyplot(fig)

st.success("✅ Main chart loaded successfully. Now rendering clusters...")

st.markdown("### 🧠 Clustering Nifty 100 Stocks (5 Groups)")
kmeans = KMeans(n_clusters=5, random_state=42, n_init='auto')
cluster_labels = kmeans.fit_predict(X)
cluster_df = pd.DataFrame({'Ticker': X.index, 'Cluster': cluster_labels})

cluster_means = X.groupby(cluster_labels).apply(lambda x: x.iloc[:, -1].mean())
cluster_order = cluster_means.sort_values(ascending=False).index
colors = cm.get_cmap('tab10', 5)

with st.expander("📂 View Cluster Charts", expanded=True):
    for rank, cluster_id in enumerate(cluster_order, 1):
        st.markdown(f"#### Cluster {cluster_id + 1} — Rank #{rank} by Avg Return on Last Day")
        fig_cluster, ax_cluster = plt.subplots(figsize=(14, 6))
        cluster_members = cluster_df[cluster_df['Cluster'] == cluster_id]['Ticker']
        cluster_color = colors(cluster_id)

        for ticker in cluster_members:
            ax_cluster.plot(returns.index, returns[ticker], label=ticker.replace(".NS", ""), linewidth=1.2, color=cluster_color)

        avg_line = returns[cluster_members].mean(axis=1)
        ax_cluster.plot(returns.index, avg_line, color=cluster_color, linestyle='--', linewidth=3, label='Cluster Avg')

        ax_cluster.set_title(f"Cluster {cluster_id + 1} — Stocks in Rank #{rank} Cluster")
        ax_cluster.set_ylabel("% Return")
        ax_cluster.set_xlabel("Date")
        ax_cluster.grid(True, linestyle='--', alpha=0.5)
        ax_cluster.legend(fontsize='x-small', ncol=4)
        st.pyplot(fig_cluster)
