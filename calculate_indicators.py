import pandas as pd
import pandas_ta as ta
from fetch_btc_data import get_btc_data

def calculate_and_display_indicators(df):
    """
    Calculates and displays various technical indicators.

    Args:
        df (pd.DataFrame): DataFrame with OHLCV data.
                           Must include 'High', 'Low', 'Close', 'Volume'.
                           'Open' is also typically present.
    """
    if df is None or df.empty:
        print("Dataframe is empty. Cannot calculate indicators.")
        return

    # print(f"Available pandas_ta methods: {dir(df.ta)}") # Temporary debug line

    print("\n--- Calculating Technical Indicators ---")

    # Ensure correct column names (case-insensitive check, then standardize)
    # yfinance typically returns title-cased columns: Open, High, Low, Close, Volume
    column_map = {col.lower(): col for col in df.columns}
    required_cols_lower = ['high', 'low', 'close', 'volume', 'open']

    for col_l in required_cols_lower:
        if col_l not in column_map:
            print(f"Missing required column: {col_l.capitalize()}")
            return

    # Standardize column names for pandas_ta (lowercase)
    df.rename(columns={v: k for k, v in column_map.items()}, inplace=True)

    # List to store successfully calculated indicators
    calculated_indicators = []

    # 1. 24-hour Volume (already in data)
    if 'volume' in df.columns:
        print("\n1. 24-hour Volume:")
        print(df['volume'].tail())
        calculated_indicators.append("24-hour Volume")

    # 2. Accumulation/Distribution Line (ADL is 'ad' in pandas_ta 0.3.14b0)
    try:
        df.ta.ad(append=True) # Appends 'AD' column
        print("\n2. Accumulation/Distribution Line (AD):")
        print(df['AD'].tail())
        calculated_indicators.append("Accumulation/Distribution Line (AD)")
    except Exception as e:
        print(f"Error calculating ADL (AD): {e}")

    # 3. Advance/Decline Line - Skipped (requires market breadth data, not just BTC)
    # 4. Advance/Decline Ratio - Skipped (requires market breadth data)
    # 5. Advance/Decline Ratio (Bars) - Skipped (requires market breadth data)
    # 6. Analyst price forecast - Skipped (proprietary/not feasible)

    # 7. Arnaud Legoux Moving Average (ALMA)
    try:
        # Default parameters for ALMA in pandas_ta: length=9, sigma=6, offset_factor=0.85
        df.ta.alma(length=9, append=True) # Appends 'ALMA_9_6.0_0.85'
        print("\n7. Arnaud Legoux Moving Average (ALMA_9_6.0_0.85):")
        print(df['ALMA_9_6.0_0.85'].tail())
        calculated_indicators.append("Arnaud Legoux Moving Average (ALMA)")
    except Exception as e:
        print(f"Error calculating ALMA: {e}")

    # 8. Aroon
    try:
        # Default length=14
        aroon = df.ta.aroon(length=14) # Returns DataFrame with AROOND_14, AROONU_14, AROONOSC_14
        df = pd.concat([df, aroon], axis=1)
        print("\n8. Aroon (14):")
        print(df[['AROOND_14', 'AROONU_14', 'AROONOSC_14']].tail())
        calculated_indicators.append("Aroon")
    except Exception as e:
        print(f"Error calculating Aroon: {e}")

    # 9. Auto Fib Extension - Skipped (visual/not standard calculation)
    # 10. Auto Fib Retracement - Skipped (visual/not standard calculation)
    # 11. Auto Pitchfork - Skipped (visual/not standard calculation)
    # 12. Auto Trendlines - Skipped (visual/not standard calculation)

    # 13. Average Day Range (ADR) - pandas_ta doesn't have ADR directly. ATR is more common.
    #     ADR is typically (High - Low) average over N days. We can calculate manually if needed,
    #     but ATR is generally preferred and available. For now, skipping direct ADR.

    # 14. Average Directional Index (ADX)
    try:
        # Default length=14
        adx = df.ta.adx(length=14) # Returns ADX_14, DMP_14, DMN_14
        df = pd.concat([df, adx], axis=1)
        print("\n14. Average Directional Index (ADX_14):")
        print(df[['ADX_14', 'DMP_14', 'DMN_14']].tail())
        calculated_indicators.append("Average Directional Index (ADX)")
    except Exception as e:
        print(f"Error calculating ADX: {e}")

    # 15. Average True Range (ATR)
    try:
        # Default length=14
        df.ta.atr(length=14, append=True) # Appends 'ATRr_14'
        print("\n15. Average True Range (ATR_14):")
        print(df['ATRr_14'].tail()) # pandas_ta uses ATRr_ for RMA-based ATR
        calculated_indicators.append("Average True Range (ATR)")
    except Exception as e:
        print(f"Error calculating ATR: {e}")

    # 16. Awesome Oscillator (AO)
    try:
        # Default fast=5, slow=34
        df.ta.ao(append=True) # Appends 'AO_5_34'
        print("\n16. Awesome Oscillator (AO_5_34):")
        print(df['AO_5_34'].tail())
        calculated_indicators.append("Awesome Oscillator (AO)")
    except Exception as e:
        print(f"Error calculating AO: {e}")

    # 17. Balance of Power (BOP)
    try:
        df.ta.bop(append=True) # Appends 'BOP'
        print("\n17. Balance of Power (BOP):")
        print(df['BOP'].tail())
        calculated_indicators.append("Balance of Power (BOP)")
    except Exception as e:
        print(f"Error calculating BOP: {e}")

    # 18. BBTrend - This seems to be a custom/specific TradingView indicator.
    #     pandas_ta has bbands. Let's use standard Bollinger Bands.

    # 19. Bollinger Bands (BB)
    try:
        # Default length=20, std=2
        bbands = df.ta.bbands(length=20, std=2) # BBL_20_2.0, BBM_20_2.0, BBU_20_2.0, BBB_20_2.0, BBP_20_2.0
        df = pd.concat([df, bbands], axis=1)
        print("\n19. Bollinger Bands (20, 2):")
        print(df[['BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0']].tail())
        calculated_indicators.append("Bollinger Bands (BB)")
    except Exception as e:
        print(f"Error calculating Bollinger Bands: {e}")

    # 20. Bollinger Bands %B (%B)
    # This is typically part of the bbands calculation in pandas_ta (BBP_20_2.0)
    if 'BBP_20_2.0' in df.columns:
        print("\n20. Bollinger Bands %B (BBP_20_2.0):")
        print(df['BBP_20_2.0'].tail())
        calculated_indicators.append("Bollinger Bands %B")

    # 21. Bollinger BandWidth (BBW)
    # This is also part of the bbands calculation in pandas_ta (BBB_20_2.0)
    if 'BBB_20_2.0' in df.columns:
        print("\n21. Bollinger BandWidth (BBB_20_2.0):")
        print(df['BBB_20_2.0'].tail())
        calculated_indicators.append("Bollinger BandWidth (BBW)")

    # 22. Bollinger Bars - Skipped (visual interpretation/not a standard series)

    # 23. Bull Bear Power
    # Elder's Bull Power and Bear Power. pandas_ta has `ebsw` (Elder Bull Bear Stop Width)
    # Let's try to find direct Bull Bear Power or implement it.
    # BullPower = High - EMA(close, length)
    # BearPower = Low - EMA(close, length)
    # TradingView default for BullBear is EMA 13.
    try:
        ema_13 = df.ta.ema(length=13)
        df['BULLP_13'] = df['high'] - ema_13
        df['BEARP_13'] = df['low'] - ema_13
        print("\n23. Bull Bear Power (EMA 13):")
        print(df[['BULLP_13', 'BEARP_13']].tail())
        calculated_indicators.append("Bull Bear Power")
    except Exception as e:
        print(f"Error calculating Bull Bear Power: {e}")

    # 24. Chaikin Money Flow (CMF)
    try:
        # Default length=20
        df.ta.cmf(length=20, append=True) # Appends 'CMF_20'
        print("\n24. Chaikin Money Flow (CMF_20):")
        print(df['CMF_20'].tail())
        calculated_indicators.append("Chaikin Money Flow (CMF)")
    except Exception as e:
        print(f"Error calculating CMF: {e}")

    # 25. Chaikin Oscillator
    # Uses ADL (now 'AD'). Default fast=3, slow=10 for EMA of ADL.
    # pandas_ta 0.3.14b0 has 'adosc' for Chaikin Oscillator.
    try:
        # Ensure AD is calculated first if not already
        if 'AD' not in df.columns:
             df.ta.ad(append=True) # Use 'ad'

        df.ta.adosc(fast=3, slow=10, append=True) # Appends ADOSC_3_10
        print("\n25. Chaikin Oscillator (ADOSC_3_10):")
        print(df['ADOSC_3_10'].tail())
        calculated_indicators.append("Chaikin Oscillator (ADOSC)")
    except Exception as e:
        print(f"Error calculating Chaikin Oscillator (ADOSC): {e}")

    # 26. Chande Kroll Stop - This is more of a stop-loss system based on ATR.
    # pandas_ta does not have it directly. Might be complex to implement quickly. Skipping for now.

    # 27. Chande Momentum Oscillator (CMO)
    try:
        # Default length=9
        df.ta.cmo(length=9, append=True) # Appends 'CMO_9'
        print("\n27. Chande Momentum Oscillator (CMO_9):")
        print(df['CMO_9'].tail())
        calculated_indicators.append("Chande Momentum Oscillator (CMO)")
    except Exception as e:
        print(f"Error calculating CMO: {e}")

    # 28. Chop Zone - Appears to be a TradingView specific indicator based on other standard ones.
    # Not directly in pandas_ta. Skipping for now.

    # 29. Choppiness Index (CHOP)
    try:
        # Default length=14
        df.ta.chop(length=14, append=True) # Appends 'CHOP_14_1_100' (length, atr_length, scalar)
        print("\n29. Choppiness Index (CHOP_14):")
        print(df['CHOP_14_1_100'].tail()) # Using default atr_length=1
        calculated_indicators.append("Choppiness Index (CHOP)")
    except Exception as e:
        print(f"Error calculating CHOP: {e}")

    # 30. Commodity Channel Index (CCI)
    try:
        # Default length=14
        df.ta.cci(length=14, append=True) # Appends 'CCI_14_0.015'
        print("\n30. Commodity Channel Index (CCI_14):")
        print(df['CCI_14_0.015'].tail())
        calculated_indicators.append("Commodity Channel Index (CCI)")
    except Exception as e:
        print(f"Error calculating CCI: {e}")

    # --- Continue with more indicators to reach 30 if some were skipped ---

    # 31. Detrended Price Oscillator (DPO)
    try:
        # Default length=20
        df.ta.dpo(length=20, append=True) # Appends 'DPO_20'
        print("\n31. Detrended Price Oscillator (DPO_20):")
        print(df['DPO_20'].tail())
        calculated_indicators.append("Detrended Price Oscillator (DPO)")
    except Exception as e:
        print(f"Error calculating DPO: {e}")

    # 32. Donchian Channels (DC)
    try:
        # Default lower_length=20, upper_length=20
        donchian = df.ta.donchian(lower_length=20, upper_length=20) #DCL_20_20, DCM_20_20, DCU_20_20
        df = pd.concat([df, donchian], axis=1)
        print("\n32. Donchian Channels (20):")
        print(df[['DCL_20_20', 'DCM_20_20', 'DCU_20_20']].tail())
        calculated_indicators.append("Donchian Channels (DC)")
    except Exception as e:
        print(f"Error calculating Donchian Channels: {e}")

    # 33. Double Exponential Moving Average (DEMA)
    try:
        # Default length=10
        df.ta.dema(length=10, append=True) # Appends 'DEMA_10'
        print("\n33. Double Exponential Moving Average (DEMA_10):")
        print(df['DEMA_10'].tail())
        calculated_indicators.append("Double Exponential Moving Average (DEMA)")
    except Exception as e:
        print(f"Error calculating DEMA: {e}")

    # 34. Ease of Movement (EOM)
    try:
        # Default length=14
        df.ta.eom(length=14, append=True) # Appends 'EOM_14_100000000'
        print("\n34. Ease of Movement (EOM_14):")
        print(df['EOM_14_100000000'].tail())
        calculated_indicators.append("Ease of Movement (EOM)")
    except Exception as e:
        print(f"Error calculating EOM: {e}")

    # 35. Elder's Force Index (EFI)
    try:
        # Default length=13
        df.ta.efi(length=13, append=True) # Appends 'EFI_13'
        print("\n35. Elder's Force Index (EFI_13):")
        print(df['EFI_13'].tail())
        calculated_indicators.append("Elder's Force Index (EFI)")
    except Exception as e:
        print(f"Error calculating EFI: {e}")

    # 36. Exponential Moving Average (EMA)
    try:
        # Default length=10 (pandas_ta default)
        # TradingView default for EMA itself is usually 9 or specified by user
        df.ta.ema(length=10, append=True) # Appends 'EMA_10'
        df.ta.ema(length=20, append=True) # Appends 'EMA_20'
        df.ta.ema(length=50, append=True) # Appends 'EMA_50'
        print("\n36. Exponential Moving Average (EMA_10, EMA_20, EMA_50):")
        print(df[['EMA_10', 'EMA_20', 'EMA_50']].tail())
        calculated_indicators.append("Exponential Moving Average (EMA)")
    except Exception as e:
        print(f"Error calculating EMA: {e}")

    # 37. Fisher Transform
    try:
        # Default length=9
        fisher = df.ta.fisher(length=9) # Returns FISHT_9_1, FISHERTsig_9_1
        df = pd.concat([df, fisher], axis=1)
        print("\n37. Fisher Transform (9):")
        print(df[['FISHERT_9_1', 'FISHERTs_9_1']].tail()) # pandas_ta uses FISHTs_ for signal
        calculated_indicators.append("Fisher Transform")
    except Exception as e:
        print(f"Error calculating Fisher Transform: {e}")

    # 38. Historical Volatility - 'hvol' not available in pandas_ta 0.3.14b0. Skipping.
    #    (ATR and STDEV are available alternatives for volatility)
    print("\n38. Historical Volatility (hvol) - Skipped (not available in this pandas_ta version)")

    # 39. Hull Moving Average (HMA)
    try:
        # Default length=9
        df.ta.hma(length=9, append=True) # Appends 'HMA_9'
        print("\n39. Hull Moving Average (HMA_9):")
        print(df['HMA_9'].tail())
        calculated_indicators.append("Hull Moving Average (HMA)")
    except Exception as e:
        print(f"Error calculating HMA: {e}")

    # 40. Ichimoku Cloud
    try:
        # Defaults: tenkan=9, kijun=26, senkou=52
        ichimoku_df = df.ta.ichimoku(tenkan=9, kijun=26, senkou=52)
        # ichimoku returns a tuple: (DataFrame with ISA_9, ISB_26, ITS_9, IKS_26, ICS_26, df with all columns)
        # We want the DataFrame part
        df = pd.concat([df, ichimoku_df[0]], axis=1)
        print("\n40. Ichimoku Cloud (9, 26, 52):")
        print(df[['ISA_9', 'ISB_26', 'ITS_9', 'IKS_26', 'ICS_26']].tail())
        calculated_indicators.append("Ichimoku Cloud")
    except Exception as e:
        print(f"Error calculating Ichimoku Cloud: {e}")

    # 41. Keltner Channels (KC)
    try:
        # Default length=20, atr_length=10, scalar=2 (for pandas_ta)
        # TradingView Keltner Channels often use EMA as basis, ATR for bands.
        # pandas_ta kc uses EMA for center line (length), and ATR (atr_length) times scalar for bands.
        keltner_df = df.ta.kc(length=20, scalar=2, atr_length=10)
        # For pandas_ta 0.3.14b0, kc returns: KCe_20_2, KCea_20_2, KCel_20_2 if mamode="ema" (default)
        # Or KCs_20_2 etc for sma. Let's check the exact names.
        # Default mamode for kc in 0.3.14b is SMA. So, KCs_20_2.0, KCa_20_2.0, KCc_20_2.0
        # The parameters are length, scalar, atr_length.
        # Column names are usually like: KCL_length_scalar, KCM_length_scalar, KCU_length_scalar (for SMA based)
        # or KCLe_length_scalar etc for EMA based.
        # The default for 0.3.14b0's kc is: mamode="sma", length=20, scalar=2, atr_length=10
        # So columns should be KCL_20_2.0, KCM_20_2.0, KCU_20_2.0
        # Let's try with explicit ema, as TV default is usually EMA based for KC.
        keltner_df = df.ta.kc(mamode="ema", length=20, scalar=2, atr_length=10)
        # Expected columns: KCLe_20_2.0, KCUe_20_2.0, KCMNe_20_2.0 (center)
        df = pd.concat([df, keltner_df], axis=1)
        # print(f"Keltner columns: {keltner_df.columns}") # Debug line
        kc_cols = [col for col in keltner_df.columns if 'KC' in col]
        print("\n41. Keltner Channels (EMA_20, ATR_10, Scalar_2):")
        print(df[kc_cols].tail())
        calculated_indicators.append("Keltner Channels (KC)")
    except Exception as e:
        print(f"Error calculating Keltner Channels: {e}")

    # 42. Know Sure Thing (KST)
    try:
        # Default params in pandas_ta for 0.3.14b0 might differ slightly in naming from newer versions.
        # It should create columns like KST_roc1_roc2.., KSTs_signal_period
        kst_df = df.ta.kst()
        df = pd.concat([df, kst_df], axis=1)
        kst_cols = [col for col in kst_df.columns if "KST" in col]
        print("\n42. Know Sure Thing (KST):")
        print(df[kst_cols].tail())
        calculated_indicators.append("Know Sure Thing (KST)")
    except Exception as e:
        print(f"Error calculating KST: {e}")

    # 43. MACD (Moving Average Convergence/Divergence)
    try:
        # Default fast=12, slow=26, signal=9
        macd = df.ta.macd(fast=12, slow=26, signal=9) # MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
        df = pd.concat([df, macd], axis=1)
        print("\n43. MACD (12, 26, 9):")
        print(df[['MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9']].tail())
        calculated_indicators.append("MACD")
    except Exception as e:
        print(f"Error calculating MACD: {e}")

    # 44. Relative Strength Index (RSI)
    try:
        # Default length=14
        df.ta.rsi(length=14, append=True) # Appends 'RSI_14'
        print("\n44. Relative Strength Index (RSI_14):")
        print(df['RSI_14'].tail())
        calculated_indicators.append("Relative Strength Index (RSI)")
    except Exception as e:
        print(f"Error calculating RSI: {e}")

    # 45. Stochastic Oscillator (STOCH)
    try:
        # Default k=14, d=3, smooth_k=3
        stoch_df = df.ta.stoch(k=14, d=3, smooth_k=3) # STOCHk_14_3_3, STOCHd_14_3_3
        df = pd.concat([df, stoch_df], axis=1)
        stoch_cols = [col for col in stoch_df.columns if "STOCH" in col]
        print("\n45. Stochastic Oscillator (14,3,3):")
        print(df[stoch_cols].tail())
        calculated_indicators.append("Stochastic Oscillator (STOCH)")
    except Exception as e:
        print(f"Error calculating Stochastic Oscillator: {e}")

    # 46. Simple Moving Average (SMA)
    try:
        df.ta.sma(length=10, append=True) # Appends 'SMA_10'
        df.ta.sma(length=20, append=True) # Appends 'SMA_20'
        df.ta.sma(length=50, append=True) # Appends 'SMA_50'
        print("\n46. Simple Moving Average (SMA_10, SMA_20, SMA_50):")
        print(df[['SMA_10', 'SMA_20', 'SMA_50']].tail())
        calculated_indicators.append("Simple Moving Average (SMA)")
    except Exception as e:
        print(f"Error calculating SMA: {e}")

    # 47. Volume Weighted Average Price (VWAP)
    # VWAP is typically calculated with intraday data (tick or minute level).
    # pandas_ta can calculate it on OHLCV data, often reset daily.
    # For daily bars, it might just be (High+Low+Close)/3 * Volume / Volume, effectively (H+L+C)/3 if not anchored.
    # The default in pandas_ta for longer periods might run a cumulative/anchored VWAP from the start of the dataset.
    try:
        df.ta.vwap(append=True) # Appends VWAP_D (D for Daily default anchor)
        print("\n47. Volume Weighted Average Price (VWAP_D):")
        print(df['VWAP_D'].tail())
        calculated_indicators.append("Volume Weighted Average Price (VWAP)")
    except Exception as e:
        print(f"Error calculating VWAP: {e}")

    print(f"\n--- Successfully calculated {len(calculated_indicators)} indicators ---")
    # print("Final DataFrame columns:", df.columns)
    return df # Return the dataframe with all indicators for potential plotting

def plot_sma_example(df):
    """
    Generates and saves a plot of Close price and SMAs.
    """
    if df is None or not all(col in df.columns for col in ['close', 'SMA_20', 'SMA_50']):
        print("DataFrame is missing necessary columns for plotting (close, SMA_20, SMA_50).")
        return

    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(14, 7))
        plt.plot(df.index, df['close'], label='Close Price', alpha=0.7)
        plt.plot(df.index, df['SMA_20'], label='SMA 20', alpha=0.7)
        plt.plot(df.index, df['SMA_50'], label='SMA 50', alpha=0.7)
        plt.title('BTC/USD Close Price with 20-day and 50-day SMAs')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.grid(True)
        plot_filename = "btc_sma_plot.png"
        plt.savefig(plot_filename)
        print(f"\n--- Plot saved to {plot_filename} ---")
    except ImportError:
        print("Matplotlib not installed. Skipping plot generation.")
    except Exception as e:
        print(f"Error generating plot: {e}")


if __name__ == "__main__":
    btc_ohlcv_data = get_btc_data()
    if btc_ohlcv_data is not None:
        # Create a copy to avoid SettingWithCopyWarning if get_btc_data returns a slice
        data_with_indicators = btc_ohlcv_data.copy()
        data_with_indicators = calculate_and_display_indicators(data_with_indicators)

        if data_with_indicators is not None:
            plot_sma_example(data_with_indicators.tail(365)) # Plot last year of data for clarity
