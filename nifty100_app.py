import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.cluster import KMeans
import numpy as np

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

# --- Top 5 markers per day ---
day_top5 = returns.apply(lambda row: row.sort_values(ascending=False).head(5), axis=1)

# --- Plot All Stocks ---
st.markdown("### 📊 Nifty 100 Stock Returns with Top 5 Daily Markers")
fig, ax = plt.subplots(figsize=(14, 7))

# Plot all with lighter lines
for ticker in df.columns:
    ax.plot(returns.index, returns[ticker], color='gray', alpha=0.3, linewidth=0.8)

# Plot top 10 with thinner colored lines
final_returns = returns.iloc[-1].sort_values(ascending=False)
top10 = final_returns.head(10)
for ticker in top10.index:
    ax.plot(returns.index, returns[ticker], linewidth=1.5, label=ticker)

# Plot green dots for daily top 5
for date in day_top5.index:
    for ticker in day_top5.loc[date].index:
        ax.plot(date, returns.loc[date, ticker], 'go', markersize=4)
        ax.text(date, returns.loc[date, ticker], ticker.replace(".NS", ""), fontsize=5, color='green')

ax.set_title("Nifty 100 Performance Since April 1, 2025")
ax.set_ylabel("% Return")
ax.set_xlabel("Date")
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(loc='upper left', fontsize='small')
st.pyplot(fig)

# --- Cluster Analysis ---
st.markdown("### 🧠 Clustering Nifty 100 Stocks (5 Groups)")
kmeans = KMeans(n_clusters=5, random_state=42, n_init='auto')
cluster_labels = kmeans.fit_predict(returns.T)
cluster_df = pd.DataFrame({'Ticker': returns.columns, 'Cluster': cluster_labels})

# Plot per-cluster returns
for c in sorted(cluster_df['Cluster'].unique()):
    st.markdown(f"#### Cluster {c + 1}")
    fig_cluster, ax_cluster = plt.subplots(figsize=(12, 5))
    cluster_members = cluster_df[cluster_df['Cluster'] == c]['Ticker']
    for ticker in cluster_members:
        ax_cluster.plot(returns.index, returns[ticker], label=ticker.replace(".NS", ""), linewidth=1.5)
    ax_cluster.set_title(f"Stock Returns - Cluster {c + 1}")
    ax_cluster.set_ylabel("% Return")
    ax_cluster.set_xlabel("Date")
    ax_cluster.grid(True, linestyle='--', alpha=0.5)
    ax_cluster.legend(fontsize='x-small', ncol=3)
    st.pyplot(fig_cluster)
