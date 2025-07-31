"""
Enhanced Trading Strategies with Multiple Indicators
Supports various TradingView-style indicators for backtesting
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
from backtesting_engine import BacktestEngine, ProfessionalScoring

class EnhancedTradingStrategies:
    """Collection of advanced trading strategies using multiple indicators"""
    
    def __init__(self):
        self.available_strategies = {
            'ema_sma_rsi': 'EMA/SMA/RSI Crossover (Current)',
            'macd_bollinger': 'MACD + Bollinger Bands',
            'adx_sar_stoch': 'ADX + Parabolic SAR + Stochastic',
            'supertrend_rsi': 'Supertrend + RSI',
            'triple_ma_volume': 'Triple MA + Volume Analysis',
            'momentum_combo': 'Multi-Momentum (RSI+MACD+CCI)',
            'volatility_breakout': 'ATR + Bollinger + Keltner',
            'trend_strength': 'ADX + Aroon + Vortex'
        }
    
    def get_strategy_list(self):
        """Return list of available strategies"""
        return self.available_strategies
    
    def ema_sma_rsi_strategy(self, data, ema_period=21, sma_period=50, rsi_period=14, 
                           rsi_oversold=30, rsi_overbought=70):
        """PERPETUAL EMA/SMA/RSI strategy - Always in position (Long/Short)"""
        df = data.copy()
        
        # Calculate indicators
        df['EMA'] = ta.ema(df['Close'], length=ema_period)
        df['SMA'] = ta.sma(df['Close'], length=sma_period)
        df['RSI'] = ta.rsi(df['Close'], length=rsi_period)
        
        # Generate PERPETUAL signals (always Long or Short)
        df['Signal'] = 0
        
        # PERPETUAL LOGIC: Always in position
        # Long: EMA > SMA AND RSI confirms trend or neutral
        long_condition = (
            (df['EMA'] > df['SMA']) &  # Uptrend
            (df['RSI'] >= rsi_oversold)  # Not severely oversold
        )
        
        # Short: EMA < SMA AND RSI confirms trend or neutral  
        short_condition = (
            (df['EMA'] < df['SMA']) &  # Downtrend
            (df['RSI'] <= rsi_overbought)  # Not severely overbought
        )
        
        # Apply signals with forward fill to ensure always in position
        df.loc[long_condition, 'Signal'] = 1
        df.loc[short_condition, 'Signal'] = -1
        
        # Forward fill to maintain position when no clear signal
        df['Signal'] = df['Signal'].replace(0, np.nan).fillna(method='ffill').fillna(1)
        
        return df
    
    def macd_bollinger_strategy(self, data, macd_fast=12, macd_slow=26, macd_signal=9, 
                              bb_period=20, bb_std=2):
        """PERPETUAL MACD + Bollinger Bands Strategy - Always in position"""
        df = data.copy()
        
        # Calculate MACD
        macd_data = ta.macd(df['Close'], fast=macd_fast, slow=macd_slow, signal=macd_signal)
        df['MACD'] = macd_data[f'MACD_{macd_fast}_{macd_slow}_{macd_signal}']
        df['MACD_Signal'] = macd_data[f'MACDs_{macd_fast}_{macd_slow}_{macd_signal}']
        df['MACD_Hist'] = macd_data[f'MACDh_{macd_fast}_{macd_slow}_{macd_signal}']
        
        # Calculate Bollinger Bands
        bb_data = ta.bbands(df['Close'], length=bb_period, std=bb_std)
        df['BB_Lower'] = bb_data[f'BBL_{bb_period}_{bb_std}.0']
        df['BB_Middle'] = bb_data[f'BBM_{bb_period}_{bb_std}.0']
        df['BB_Upper'] = bb_data[f'BBU_{bb_period}_{bb_std}.0']
        
        # Calculate BB position
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # Generate PERPETUAL signals
        df['Signal'] = 0
        
        # Long: MACD bullish AND price position favorable
        long_condition = (
            (df['MACD'] > df['MACD_Signal']) &  # MACD bullish
            (df['BB_Position'] < 0.8)  # Not too close to upper band
        )
        
        # Short: MACD bearish AND price position favorable
        short_condition = (
            (df['MACD'] < df['MACD_Signal']) &  # MACD bearish
            (df['BB_Position'] > 0.2)  # Not too close to lower band
        )
        
        df.loc[long_condition, 'Signal'] = 1
        df.loc[short_condition, 'Signal'] = -1
        
        # Forward fill to maintain position
        df['Signal'] = df['Signal'].replace(0, np.nan).fillna(method='ffill').fillna(1)
        
        return df
    
    def adx_sar_stoch_strategy(self, data, adx_period=14, adx_threshold=25, sar_af=0.02, sar_max=0.2,
                             stoch_k=14, stoch_d=3, stoch_oversold=20, stoch_overbought=80):
        """PERPETUAL ADX + Parabolic SAR + Stochastic Strategy - Always in position"""
        df = data.copy()
        
        # Calculate ADX
        adx_data = ta.adx(df['High'], df['Low'], df['Close'], length=adx_period)
        df['ADX'] = adx_data[f'ADX_{adx_period}']
        df['DI_Plus'] = adx_data[f'DMP_{adx_period}']
        df['DI_Minus'] = adx_data[f'DMN_{adx_period}']
        
        # Calculate Parabolic SAR
        sar_data = ta.psar(df['High'], df['Low'], df['Close'], af=sar_af, max_af=sar_max)
        df['SAR'] = sar_data['PSARl_0.02_0.2']
        
        # Calculate Stochastic
        stoch_data = ta.stoch(df['High'], df['Low'], df['Close'], k=stoch_k, d=stoch_d)
        df['Stoch_K'] = stoch_data[f'STOCHk_{stoch_k}_{stoch_d}_3']
        df['Stoch_D'] = stoch_data[f'STOCHd_{stoch_k}_{stoch_d}_3']
        
        # Generate PERPETUAL signals
        df['Signal'] = 0
        
        # Long: Bullish conditions
        long_condition = (
            (df['Close'] > df['SAR']) &  # Price above SAR (uptrend)
            (df['DI_Plus'] > df['DI_Minus'])  # Positive directional movement
        )
        
        # Short: Bearish conditions
        short_condition = (
            (df['Close'] < df['SAR']) |  # Price below SAR (downtrend)
            (df['DI_Plus'] < df['DI_Minus'])  # Negative directional movement
        )
        
        df.loc[long_condition, 'Signal'] = 1
        df.loc[short_condition, 'Signal'] = -1
        
        # Forward fill to maintain position
        df['Signal'] = df['Signal'].replace(0, np.nan).fillna(method='ffill').fillna(1)
        
        return df
    
    def supertrend_rsi_strategy(self, data, atr_period=10, atr_multiplier=3.0, 
                              rsi_period=14, rsi_oversold=30, rsi_overbought=70):
        """PERPETUAL Supertrend + RSI Strategy - Always in position"""
        df = data.copy()
        
        # Calculate Supertrend
        supertrend_data = ta.supertrend(df['High'], df['Low'], df['Close'], 
                                      length=atr_period, multiplier=atr_multiplier)
        df['Supertrend'] = supertrend_data[f'SUPERT_{atr_period}_{atr_multiplier}']
        df['Supertrend_Direction'] = supertrend_data[f'SUPERTd_{atr_period}_{atr_multiplier}']
        
        # Calculate RSI
        df['RSI'] = ta.rsi(df['Close'], length=rsi_period)
        
        # Generate PERPETUAL signals
        df['Signal'] = 0
        
        # Long: Supertrend bullish AND RSI conditions favor long
        long_condition = (
            (df['Supertrend_Direction'] == 1) &  # Supertrend bullish
            (df['RSI'] < rsi_overbought)  # RSI not overbought
        )
        
        # Short: Supertrend bearish AND RSI conditions favor short
        short_condition = (
            (df['Supertrend_Direction'] == -1) &  # Supertrend bearish
            (df['RSI'] > rsi_oversold)  # RSI not oversold
        )
        
        df.loc[long_condition, 'Signal'] = 1
        df.loc[short_condition, 'Signal'] = -1
        
        # Forward fill to maintain position
        df['Signal'] = df['Signal'].replace(0, np.nan).fillna(method='ffill').fillna(1)
        
        return df
    
    def momentum_combo_strategy(self, data, rsi_period=14, macd_fast=12, macd_slow=26, 
                              macd_signal=9, cci_period=20, cci_oversold=-100, cci_overbought=100):
        """PERPETUAL Multi-Momentum Strategy (RSI + MACD + CCI) - Always in position"""
        df = data.copy()
        
        # Calculate RSI
        df['RSI'] = ta.rsi(df['Close'], length=rsi_period)
        
        # Calculate MACD
        macd_data = ta.macd(df['Close'], fast=macd_fast, slow=macd_slow, signal=macd_signal)
        df['MACD'] = macd_data[f'MACD_{macd_fast}_{macd_slow}_{macd_signal}']
        df['MACD_Signal'] = macd_data[f'MACDs_{macd_fast}_{macd_slow}_{macd_signal}']
        
        # Calculate CCI
        df['CCI'] = ta.cci(df['High'], df['Low'], df['Close'], length=cci_period)
        
        # Create momentum score
        df['Momentum_Score'] = 0
        
        # Add points for bullish momentum indicators
        df.loc[df['RSI'] > 50, 'Momentum_Score'] += 1
        df.loc[df['MACD'] > df['MACD_Signal'], 'Momentum_Score'] += 1
        df.loc[df['CCI'] > 0, 'Momentum_Score'] += 1
        
        # Generate PERPETUAL signals based on momentum score
        df['Signal'] = 0
        
        # Long: Majority bullish momentum (score >= 2)
        long_condition = df['Momentum_Score'] >= 2
        
        # Short: Majority bearish momentum (score <= 1)
        short_condition = df['Momentum_Score'] <= 1
        
        df.loc[long_condition, 'Signal'] = 1
        df.loc[short_condition, 'Signal'] = -1
        
        # Forward fill to maintain position
        df['Signal'] = df['Signal'].replace(0, np.nan).fillna(method='ffill').fillna(1)
        
        return df
    
    def volatility_breakout_strategy(self, data, atr_period=14, bb_period=20, bb_std=2, 
                                   kc_period=20, kc_multiplier=2.0):
        """PERPETUAL ATR + Bollinger + Keltner Volatility Strategy - Always in position"""
        df = data.copy()
        
        # Calculate ATR
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=atr_period)
        df['ATR_MA'] = ta.sma(df['ATR'], length=10)  # ATR moving average
        
        # Calculate Bollinger Bands
        bb_data = ta.bbands(df['Close'], length=bb_period, std=bb_std)
        df['BB_Lower'] = bb_data[f'BBL_{bb_period}_{bb_std}.0']
        df['BB_Upper'] = bb_data[f'BBU_{bb_period}_{bb_std}.0']
        df['BB_Middle'] = bb_data[f'BBM_{bb_period}_{bb_std}.0']
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['Close'] * 100
        
        # Calculate Keltner Channels
        kc_data = ta.kc(df['High'], df['Low'], df['Close'], length=kc_period, scalar=kc_multiplier)
        df['KC_Lower'] = kc_data[f'KCLe_{kc_period}_{kc_multiplier}']
        df['KC_Upper'] = kc_data[f'KCUe_{kc_period}_{kc_multiplier}']
        df['KC_Middle'] = kc_data[f'KCBe_{kc_period}_{kc_multiplier}']
        
        # Generate PERPETUAL signals
        df['Signal'] = 0
        
        # Long: Price above middle lines (bullish bias)
        long_condition = (
            (df['Close'] > df['BB_Middle']) &  # Above BB middle
            (df['Close'] > df['KC_Middle'])    # Above KC middle
        )
        
        # Short: Price below middle lines (bearish bias)
        short_condition = (
            (df['Close'] < df['BB_Middle']) &  # Below BB middle
            (df['Close'] < df['KC_Middle'])    # Below KC middle
        )
        
        df.loc[long_condition, 'Signal'] = 1
        df.loc[short_condition, 'Signal'] = -1
        
        # Forward fill to maintain position
        df['Signal'] = df['Signal'].replace(0, np.nan).fillna(method='ffill').fillna(1)
        
        return df

class EnhancedParameterOptimizer:
    """Enhanced optimizer supporting multiple strategies"""
    
    def __init__(self, backtest_engine):
        self.engine = backtest_engine
        self.strategies = EnhancedTradingStrategies()
        self.results = []
    
    def optimize_strategy(self, data, strategy_name, parameter_ranges):
        """Optimize a specific strategy with given parameter ranges"""
        
        print(f"Optimizing strategy: {strategy_name}")
        print(f"Parameter ranges: {parameter_ranges}")
        
        # Get strategy function
        strategy_func = getattr(self.strategies, f"{strategy_name}_strategy")
        
        # Generate all parameter combinations
        import itertools
        
        param_names = list(parameter_ranges.keys())
        param_values = list(parameter_ranges.values())
        combinations = list(itertools.product(*param_values))
        
        print(f"Testing {len(combinations)} parameter combinations...")
        
        tested = 0
        strategy_results = []
        
        for combination in combinations:
            # Create parameter dict
            params = dict(zip(param_names, combination))
            
            try:
                # Generate signals with current parameters
                df_with_signals = strategy_func(data, **params)
                
                # Create a simple strategy object for backtesting
                class TempStrategy:
                    def generate_signals(self, data):
                        return strategy_func(data, **params)
                
                temp_strategy = TempStrategy()
                
                # Run backtest
                metrics = self.engine.run_backtest(data, temp_strategy)
                
                if metrics and metrics['total_trades'] > 0:
                    # Calculate professional score
                    score_metrics = {
                        'profit_factor': metrics['profit_factor'],
                        'sharpe_ratio': metrics['sharpe_ratio'],
                        'sortino_ratio': metrics['sortino_ratio'],
                        'win_rate': metrics['win_rate'],
                        'intra_trade_max_dd': metrics['intra_trade_max_dd'],
                        'omega_ratio': metrics['omega_ratio'],
                        'total_trades': metrics['total_trades']
                    }
                    
                    composite_score = ProfessionalScoring.calculate_composite_score(score_metrics)
                    
                    result = {
                        'strategy': strategy_name,
                        'composite_score': composite_score,
                        **params,
                        **metrics
                    }
                    
                    # Add zone colors
                    for metric_name, value in score_metrics.items():
                        result[f'{metric_name}_zone'] = ProfessionalScoring.get_zone_color(metric_name, value)
                    
                    strategy_results.append(result)
                
            except Exception as e:
                print(f"Error testing {strategy_name} with {params}: {e}")
            
            tested += 1
            if tested % 20 == 0:
                print(f"Progress: {tested}/{len(combinations)} combinations tested")
        
        # Sort by composite score
        strategy_results.sort(key=lambda x: x['composite_score'], reverse=True)
        
        print(f"Strategy optimization complete! {len(strategy_results)} valid results")
        
        return strategy_results

def main():
    """Example usage of enhanced strategies"""
    
    print("=== Enhanced Trading Strategies Backtesting ===\n")
    
    # Initialize
    engine = BacktestEngine(initial_capital=100000, commission=0.001)
    strategies = EnhancedTradingStrategies()
    
    # Fetch data
    print("Fetching BTC data...")
    btc_data = engine.fetch_btc_data(start_date='2020-01-01')  # Shorter period for demo
    
    print(f"Available strategies:")
    for key, desc in strategies.get_strategy_list().items():
        print(f"  {key}: {desc}")
    
    # Test MACD + Bollinger strategy
    print(f"\nTesting MACD + Bollinger Bands Strategy...")
    
    # Create strategy instance
    class MADCBollingerStrategy:
        def generate_signals(self, data):
            return strategies.macd_bollinger_strategy(data)
    
    strategy = MADCBollingerStrategy()
    
    # Run backtest
    metrics = engine.run_backtest(btc_data, strategy)
    
    if metrics:
        print(f"\nResults:")
        print(f"Total Return: {metrics['total_return']:.2f}%")
        print(f"Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Win Rate: {metrics['win_rate']:.1f}%")
        print(f"Total Trades: {metrics['total_trades']}")

if __name__ == "__main__":
    main()
