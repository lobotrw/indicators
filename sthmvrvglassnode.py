import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# Placeholder function to simulate fetching STH MVRV data from Glassnode
def fetch_sth_mvrv_data():
    # Replace with actual API call to Glassnode
    dates = pd.date_range(start='2020-01-01', end=pd.Timestamp.today(), freq='D')
    sth_mvrv = np.random.uniform(0.8, 2.0, size=len(dates))
    return pd.DataFrame({'Date': dates, 'STH MVRV': sth_mvrv})

# Fetch historical Bitcoin price data from Yahoo Finance
btc_data = yf.download('BTC-USD', start='2020-01-01', end=pd.Timestamp.today())
btc_data.reset_index(inplace=True)

# Fetch historical STH MVRV data
sth_mvrv_data = fetch_sth_mvrv_data()

# Calculate 365-day simple moving average (365D-SMA)
sth_mvrv_data['365D-SMA'] = sth_mvrv_data['STH MVRV'].rolling(window=365).mean()

# Calculate STH MVRV momentum
sth_mvrv_data['STH MVRV Momentum'] = sth_mvrv_data['STH MVRV'] - sth_mvrv_data['365D-SMA']

# Smooth out the STH MVRV Momentum using a 30-day moving average
sth_mvrv_data['STH MVRV Momentum Smoothed'] = sth_mvrv_data['STH MVRV Momentum'].rolling(window=30).mean()

# Plot the data
fig, ax1 = plt.subplots(figsize=(14, 7))

# Plot smoothed STH MVRV momentum
ax1.plot(sth_mvrv_data['Date'], sth_mvrv_data['STH MVRV Momentum Smoothed'], label='STH MVRV Momentum (Smoothed)', color='orange', alpha=0.7)

# Plot standard deviation lines
std_devs = [0.5, 1, 2, 3, -0.5, -1, -2, -3]
colors = ['green', 'yellow', 'red', 'purple', 'green', 'yellow', 'red', 'purple']
labels = ['0.5 Std Dev', '1 Std Dev', '2 Std Dev', '3 Std Dev', '-0.5 Std Dev', '-1 Std Dev', '-2 Std Dev', '-3 Std Dev']

for std_dev, color, label in zip(std_devs, colors, labels):
    ax1.axhline(y=std_dev * sth_mvrv_data['STH MVRV Momentum Smoothed'].std(), color=color, linestyle='--', label=label)

# Add a secondary y-axis for Bitcoin price
ax2 = ax1.twinx()
ax2.plot(btc_data['Date'], btc_data['Adj Close'], label='Bitcoin Price', color='blue', alpha=0.6)
ax2.set_ylabel('Bitcoin Price (USD)')

# Labeling and titles
ax1.set_ylabel('STH MVRV Momentum (Smoothed)')
ax1.set_xlabel('Date')
ax1.set_title('Bitcoin STH MVRV Momentum with Standard Deviations and Bitcoin Price')

# Add legends
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.show()