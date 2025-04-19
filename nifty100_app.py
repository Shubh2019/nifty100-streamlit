import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.cluster import KMeans
from matplotlib import cm
import numpy as np
import seaborn as sns

st.set_page_config(page_title="Nifty 100 Cluster Dashboard", layout="wide")
st.title("📊 Nifty 100 Cluster Evolution Dashboard")

# Tickers (shortened for testing)
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

with st.spinner("📥 Downloading data..."):
    try:
        df = yf.download(nifty_100_tickers, start=start_date, end=end_date)['Adj Close']
    except Exception as e:
        st.error(f"Download failed: {e}")
        st.stop()

    df.dropna(axis=1, inplace=True)

    if df.empty:
        st.error("Downloaded data is empty. Try again later or with fewer tickers.")
        st.stop()

returns = ((df - df.iloc[0]) / df.iloc[0]) * 100

# 🎞️ Animate cluster evolution
if len(returns.index) > 6:
    st.markdown("### 🎞️ Animate Cluster Evolution")
    slider_min = pd.to_datetime(returns.index[min(5, len(returns.index)-2)])
    slider_max = pd.to_datetime(returns.index[-1])
    date_slider = st.slider(
        "Select Date for Clustering",
        min_value=slider_min,
        max_value=slider_max,
        value=slider_max,
        format="%Y-%m-%d"
    )

    selected_data = returns.loc[:str(date_slider)]
    selected_returns = selected_data.T
    kmeans_dynamic = KMeans(n_clusters=3, random_state=42, n_init='auto')
    dynamic_clusters = kmeans_dynamic.fit_predict(selected_returns)

    fig_dyn, ax_dyn = plt.subplots(figsize=(14, 6))
    colors_dyn = cm.tab10(dynamic_clusters)
    for i, stock in enumerate(selected_returns.index):
        ax_dyn.plot(selected_data.index, selected_returns.loc[stock], label=stock, color=colors_dyn[i], linewidth=1)
        ax_dyn.text(selected_data.index[-1], selected_returns.loc[stock].iloc[-1], stock.replace('.NS',''), fontsize=6, color=colors_dyn[i])
    ax_dyn.set_title(f"Cluster Assignment on {date_slider.date()} (Animated View)")
    ax_dyn.set_xlabel("Date")
    ax_dyn.set_ylabel("% Return Since April 1")
    ax_dyn.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig_dyn)

# TRACKER: Build cluster history DataFrame for each date
date_range = returns.index[5:]
cluster_history = pd.DataFrame(index=nifty_100_tickers, columns=date_range)

for date in date_range:
    try:
        rolling_returns = returns.loc[:str(date)].T
        if len(rolling_returns.columns) > 4:
            kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
            cluster_labels = kmeans.fit_predict(rolling_returns)
            cluster_history.loc[:, date] = cluster_labels
    except Exception:
        continue

cluster_history = cluster_history.apply(pd.to_numeric, errors='coerce')
diff_counts = cluster_history.T.diff().ne(0).sum()
st.markdown("### 🔄 Stock Cluster Change Tracker")
st.dataframe(diff_counts.sort_values(ascending=False).rename("# Cluster Changes").to_frame())

st.markdown("### 🔥 Cluster Transition Heatmap")
fig_heatmap, ax_heatmap = plt.subplots(figsize=(18, 8))
sns.heatmap(cluster_history.T, cmap="tab20", cbar_kws={'label': 'Cluster ID'}, ax=ax_heatmap)
ax_heatmap.set_title("Daily Cluster Assignments per Stock")
ax_heatmap.set_xlabel("Stock")
ax_heatmap.set_ylabel("Date")
plt.xticks(rotation=90)
st.pyplot(fig_heatmap)
