import yfinance as yf
import pandas as pd

def get_btc_data(start_date="2018-01-01"):
    """
    Fetches BTC-USD daily historical data from the specified start date to the present.

    Args:
        start_date (str): The start date for fetching data in 'YYYY-MM-DD' format.

    Returns:
        pandas.DataFrame: A DataFrame containing the OHLCV data for BTC-USD,
                          or None if data fetching fails.
    """
    try:
        btc_ticker = yf.Ticker("BTC-USD")
        btc_data = btc_ticker.history(start=start_date, interval="1d")
        if btc_data.empty:
            print("No data found for BTC-USD. Check the ticker or date range.")
            return None
        print(f"Successfully fetched BTC-USD data from {start_date} to present.")
        print("Sample data:")
        print(btc_data.head())
        return btc_data
    except Exception as e:
        print(f"Error fetching BTC-USD data: {e}")
        return None

if __name__ == "__main__":
    data = get_btc_data()
    if data is not None:
        print("\nData summary:")
        print(data.info())
        print("\nLast 5 days:")
        print(data.tail())
