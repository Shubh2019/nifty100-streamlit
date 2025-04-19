import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

st.set_page_config(page_title="Nifty 100 Daily Returns", layout="wide")
st.title("📈 Nifty 100: Top 10 Performers Since April 1, 2025")

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
final_returns = returns.iloc[-1].sort_values(ascending=False)
top10 = final_returns.head(10)

st.markdown("### 🏆 Top 10 Performing Nifty 100 Stocks Since April 1")
fig, ax = plt.subplots(figsize=(12, 6))
for ticker in df.columns:
    if ticker in top10.index:
        ax.plot(returns.index, returns[ticker], linewidth=2.5, label=ticker)
        ax.text(returns.index[-1], returns[ticker].iloc[-1], ticker.replace(".NS", ""), fontsize=8, color='green')
    else:
        ax.plot(returns.index, returns[ticker], color='gray', alpha=0.2, linewidth=0.5)

ax.set_title("Nifty 100 Stock Returns (Apr 1 to Today)")
ax.set_ylabel("% Return")
ax.set_xlabel("Date")
ax.legend(loc='upper left', fontsize='small')
ax.grid(True, linestyle='--', alpha=0.5)
st.pyplot(fig)

st.dataframe(top10.rename("% Return"))
