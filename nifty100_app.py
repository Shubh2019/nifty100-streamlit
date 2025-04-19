import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.cluster import KMeans
from matplotlib import cm
import numpy as np
import seaborn as sns

# Config
st.set_page_config(page_title="Nifty 100 Daily Tracker", layout="wide")
st.title("📈 Nifty 100 Cluster-wise Performance Charts")

# Date Range
start_date = "2025-04-01"
end_date = datetime.today().strftime('%Y-%m-%d')

# Tickers
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

# Optional sector map
sector_map = {
    'RELIANCE.NS': 'Energy', 'TCS.NS': 'IT', 'HDFCBANK.NS': 'Banking',
    'INFY.NS': 'IT', 'ICICIBANK.NS': 'Banking', 'HINDUNILVR.NS': 'FMCG',
    'ITC.NS': 'FMCG', 'SBIN.NS': 'Banking', 'AXISBANK.NS': 'Banking', 'LT.NS': 'Infrastructure'
}

# Load data
with st.spinner('📈 Fetching stock data...'):
    raw_data = yf.download(nifty_100_tickers, start=start_date, end=end_date, group_by='ticker', auto_adjust=False)
    close_prices = pd.DataFrame()
    for ticker in nifty_100_tickers:
        try:
            if 'Adj Close' in raw_data[ticker]:
                close_prices[ticker] = raw_data[ticker]['Adj Close']
            elif 'Close' in raw_data[ticker]:
                close_prices[ticker] = raw_data[ticker]['Close']
        except KeyError:
            continue

data = close_prices.dropna(axis=1)
data_pct_change = ((data - data.iloc[0]) / data.iloc[0]) * 100

# 🎞️ Animate cluster evolution
st.markdown("### 🎞️ Animate Cluster Evolution")
date_slider = st.slider("Select Date for Clustering", min_value=data_pct_change.index[1], max_value=data_pct_change.index[-1], value=data_pct_change.index[-1])
selected_data = data_pct_change.loc[:date_slider]
selected_returns = selected_data.T
kmeans_dynamic = KMeans(n_clusters=10, random_state=42, n_init='auto')
dynamic_clusters = kmeans_dynamic.fit_predict(selected_returns)

fig_dyn, ax_dyn = plt.subplots(figsize=(18, 8))
colors_dyn = cm.tab10(dynamic_clusters)
for i, stock in enumerate(selected_returns.index):
    ax_dyn.plot(selected_data.columns, selected_returns.loc[stock], label=stock, color=colors_dyn[i], linewidth=1)
    ax_dyn.text(selected_data.columns[-1], selected_returns.loc[stock].iloc[-1], stock.replace('.NS',''), fontsize=6, color=colors_dyn[i])
ax_dyn.set_title(f"Cluster Assignment on {date_slider.date()} (Animated View)")
ax_dyn.set_xlabel("Date")
ax_dyn.set_ylabel("% Return Since April 1")
ax_dyn.grid(True, linestyle='--', alpha=0.5)
st.pyplot(fig_dyn)

# 🔁 Cluster tracker & 🔥 heatmap
date_range = data_pct_change.index[5:]
cluster_history = pd.DataFrame(index=nifty_100_tickers, columns=date_range)
for date in date_range:
    try:
        rolling_returns = data_pct_change.loc[:date].T
        if len(rolling_returns.columns) > 4:
            kmeans = KMeans(n_clusters=10, random_state=42, n_init='auto')
            cluster_labels = kmeans.fit_predict(rolling_returns)
            cluster_history.loc[:, date] = cluster_labels
    except:
        continue

cluster_history = cluster_history.apply(pd.to_numeric, errors='coerce')
diff_counts = cluster_history.T.diff().ne(0).sum()
st.markdown("### 🔄 Stock Cluster Change Tracker")
st.dataframe(diff_counts.sort_values(ascending=False).rename("# Cluster Changes").to_frame())

st.markdown("### 🔥 Cluster Transition Heatmap")
fig_heatmap, ax_heatmap = plt.subplots(figsize=(20, 10))
sns.heatmap(cluster_history.T, cmap="tab20", cbar_kws={'label': 'Cluster ID'}, ax=ax_heatmap)
ax_heatmap.set_title("Daily Cluster Assignments per Stock")
ax_heatmap.set_xlabel("Stock")
ax_heatmap.set_ylabel("Date")
plt.xticks(rotation=90)
st.pyplot(fig_heatmap)

# Pie/bar toggle
st.markdown("### 📎 Cluster Composition")
view_type = st.radio("Choose Chart Type", ["Pie Chart", "Bar Chart"], horizontal=True)
final_clusters = cluster_history.iloc[:, -1].dropna()
cluster_counts = final_clusters.value_counts().sort_index()

if view_type == "Pie Chart":
    fig_pie, ax_pie = plt.subplots()
    ax_pie.pie(cluster_counts, labels=[f"C{i+1}" for i in cluster_counts.index], autopct='%1.1f%%', startangle=140)
    ax_pie.axis('equal')
    st.pyplot(fig_pie)
else:
    fig_bar, ax_bar = plt.subplots()
    ax_bar.bar([f"C{i+1}" for i in cluster_counts.index], cluster_counts, color=cm.viridis(np.linspace(0, 1, len(cluster_counts))))
    ax_bar.set_title("Number of Stocks per Cluster")
    ax_bar.set_ylabel("Stock Count")
    st.pyplot(fig_bar)

# 📥 Download section
st.download_button("📥 Download Final Cluster Labels", cluster_history.iloc[:, -1].reset_index().rename(columns={cluster_history.columns[-1]: "Cluster"}).to_csv(index=False), file_name="final_clusters.csv")
