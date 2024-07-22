import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# Fetch Bitcoin price data from Yahoo Finance
btc_data = yf.download('BTC-USD', start='2020-01-01', end=pd.Timestamp.today())
btc_data = btc_data[['Close']]
btc_data = btc_data.rename(columns={'Close': 'BTC Price'})

# Define function to fetch on-chain active address data
def fetch_active_addresses_data():
    dates = pd.date_range(start='2020-01-01', end=pd.Timestamp.today(), freq='D')
    active_addresses = np.random.randint(50000, 2000000, size=len(dates))
    return pd.DataFrame({'Date': dates, 'Active Addresses': active_addresses})

# Fetch historical Bitcoin active address data
active_addresses_data = fetch_active_addresses_data()

# Merge the price and active address data
data = pd.merge(btc_data, active_addresses_data, left_index=True, right_on='Date')
data.set_index('Date', inplace=True)

# Calculate 28-day percentage changes
data['28D Price Change'] = data['BTC Price'].pct_change(periods=28) * 100

# Calculate standard deviation bands for the 28-day price changes
data['Price Change Mean'] = data['28D Price Change'].rolling(window=28).mean()
data['Price Change Std'] = data['28D Price Change'].rolling(window=28).std()
data['0.5 Std'] = data['Price Change Mean'] + 0.5 * data['Price Change Std']
data['1 Std'] = data['Price Change Mean'] + data['Price Change Std']
data['2 Std'] = data['Price Change Mean'] + 2 * data['Price Change Std']
data['3 Std'] = data['Price Change Mean'] + 3 * data['Price Change Std']
data['-0.5 Std'] = data['Price Change Mean'] - 0.5 * data['Price Change Std']
data['-1 Std'] = data['Price Change Mean'] - data['Price Change Std']
data['-2 Std'] = data['Price Change Mean'] - 2 * data['Price Change Std']
data['-3 Std'] = data['Price Change Mean'] - 3 * data['Price Change Std']

# Plot the data
fig, ax1 = plt.subplots(figsize=(14, 7))

# Plot BTC price
ax1.plot(data.index, data['BTC Price'], label='BTC Price', color='blue', alpha=0.5)
ax1.set_ylabel('BTC Price', color='blue')

# Create another y-axis for the changes
ax2 = ax1.twinx()

# Plot 28-day percentage change in price
ax2.plot(data.index, data['28D Price Change'], label='28D Price Change', color='orange', alpha=0.7)

# Plot standard deviation lines
ax2.axhline(y=0.5 * data['Price Change Std'].mean(), color='green', linestyle='--', label='0.5 Std Dev')
ax2.axhline(y=data['Price Change Std'].mean(), color='yellow', linestyle='--', label='1 Std Dev')
ax2.axhline(y=2 * data['Price Change Std'].mean(), color='red', linestyle='--', label='2 Std Dev')
ax2.axhline(y=3 * data['Price Change Std'].mean(), color='purple', linestyle='--', label='3 Std Dev')
ax2.axhline(y=-0.5 * data['Price Change Std'].mean(), color='green', linestyle='--', label='-0.5 Std Dev')
ax2.axhline(y=-data['Price Change Std'].mean(), color='yellow', linestyle='--', label='-1 Std Dev')
ax2.axhline(y=-2 * data['Price Change Std'].mean(), color='red', linestyle='--', label='-2 Std Dev')
ax2.axhline(y=-3 * data['Price Change Std'].mean(), color='purple', linestyle='--', label='-3 Std Dev')

# Labeling and titles
ax2.set_ylabel('28D Percentage Change')
ax1.set_xlabel('Date')
ax1.set_title('Active Address Sentiment Indicator (AASI) with Standard Deviations')

# Add legends
fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))

plt.show()
