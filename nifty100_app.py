import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Step 1: Define the date range
start_date = "2025-04-01"
end_date = datetime.today().strftime('%Y-%m-%d')

# Step 2: List of Nifty 100 companies (example subset, you can expand to full 100)
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
