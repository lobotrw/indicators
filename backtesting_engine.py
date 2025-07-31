"""
Bitcoin Backtesting Engine with Professional Scoring Metrics
Designed for daily BTC data from 2018 to present day
"""

import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class ProfessionalScoring:
    """Professional scoring system based on key trading metrics"""
    
    @staticmethod
    def get_zone_color(metric_name, value):
        """Determine the performance zone (Green/Yellow/Red) for a given metric"""
        
        zones = {
            'intra_trade_max_dd': {'green': lambda x: x < 25, 'yellow': lambda x: 25 <= x <= 40, 'red': lambda x: x > 40},
            'sortino_ratio': {'green': lambda x: x > 2.90, 'yellow': lambda x: 2 <= x <= 2.90, 'red': lambda x: x < 2},
            'sharpe_ratio': {'green': lambda x: x > 2, 'yellow': lambda x: 1 <= x <= 2, 'red': lambda x: x < 1},
            'profit_factor': {'green': lambda x: x > 4, 'yellow': lambda x: 2 <= x <= 4, 'red': lambda x: x < 2},
            'win_rate': {'green': lambda x: x > 50, 'yellow': lambda x: 35 <= x <= 50, 'red': lambda x: x < 35},
            'total_trades': {'green': lambda x: 45 <= x <= 105, 'yellow': lambda x: 40 <= x <= 44, 'red': lambda x: x < 40 or x > 105},
            'omega_ratio': {'green': lambda x: x > 1.31, 'yellow': lambda x: 1.1 <= x <= 1.31, 'red': lambda x: x < 1.1}
        }
        
        if metric_name not in zones:
            return 'unknown'
            
        zone_functions = zones[metric_name]
        
        if zone_functions['green'](value):
            return 'green'
        elif zone_functions['yellow'](value):
            return 'yellow'
        else:
            return 'red'
    
    @staticmethod
    def calculate_composite_score(metrics):
        """Calculate a composite score based on all metrics"""
        scores = {'green': 3, 'yellow': 2, 'red': 1}
        
        metric_weights = {
            'profit_factor': 0.25,
            'sharpe_ratio': 0.20,
            'sortino_ratio': 0.20,
            'win_rate': 0.15,
            'intra_trade_max_dd': 0.10,
            'omega_ratio': 0.05,
            'total_trades': 0.05
        }
        
        total_score = 0
        for metric, weight in metric_weights.items():
            if metric in metrics:
                zone = ProfessionalScoring.get_zone_color(metric, metrics[metric])
                total_score += scores[zone] * weight
        
        return total_score

class TradingStrategy:
    """Base trading strategy using EMA, SMA, and RSI crossovers"""
    
    def __init__(self, ema_period=21, sma_period=50, rsi_period=14, rsi_oversold=30, rsi_overbought=70):
        self.ema_period = ema_period
        self.sma_period = sma_period
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
    
    def generate_signals(self, data):
        """Generate buy/sell signals based on EMA, SMA, and RSI"""
        df = data.copy()
        
        # Calculate indicators
        df['EMA'] = ta.ema(df['Close'], length=self.ema_period)
        df['SMA'] = ta.sma(df['Close'], length=self.sma_period)
        df['RSI'] = ta.rsi(df['Close'], length=self.rsi_period)
        
        # Generate signals
        df['Signal'] = 0
        
        # Buy signal: EMA > SMA and RSI < oversold (coming from oversold)
        buy_condition = (df['EMA'] > df['SMA']) & (df['RSI'] < self.rsi_oversold)
        
        # Sell signal: EMA < SMA or RSI > overbought
        sell_condition = (df['EMA'] < df['SMA']) | (df['RSI'] > self.rsi_overbought)
        
        df.loc[buy_condition, 'Signal'] = 1
        df.loc[sell_condition, 'Signal'] = -1
        
        return df

class BacktestEngine:
    """Comprehensive backtesting engine with professional metrics"""
    
    def __init__(self, initial_capital=100000, commission=0.001):
        self.initial_capital = initial_capital
        self.commission = commission
        self.trades = []
        self.equity_curve = []
    
    def fetch_btc_data(self, start_date='2018-01-01', end_date=None):
        """Fetch daily BTC data from Yahoo Finance"""
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        print(f"Fetching BTC data from {start_date} to {end_date}")
        
        ticker = yf.Ticker("BTC-USD")
        data = ticker.history(start=start_date, end=end_date, interval='1d')
        
        # Clean data
        data = data.dropna()
        data.index = pd.to_datetime(data.index)
        
        print(f"Downloaded {len(data)} daily bars")
        return data
    
    def run_backtest(self, data, strategy):
        """Run PERPETUAL backtest - always in position (Long/Short)"""
        df = strategy.generate_signals(data)
        
        # Initialize variables for perpetual trading
        position = 0  # 0 = no position, 1 = long, -1 = short
        entry_price = 0
        entry_date = None
        capital = self.initial_capital
        trades = []
        equity_curve = []
        current_value = self.initial_capital
        
        for i, row in df.iterrows():
            current_price = row['Close']
            signal = row['Signal']
            
            # Ensure signal is always 1 (long) or -1 (short) for perpetual strategy
            if signal == 0:
                signal = 1  # Default to long if no signal
            
            # Track equity for current position
            if position != 0:
                # Calculate current value based on position
                if position == 1:  # Long position
                    current_value = capital * (current_price / entry_price)
                else:  # Short position (position == -1)
                    current_value = capital * (2 - (current_price / entry_price))
            else:
                current_value = capital
            
            equity_curve.append({
                'Date': i,
                'Equity': current_value,
                'Price': current_price,
                'Position': position,
                'Signal': signal
            })
            
            # Handle position changes for PERPETUAL strategy
            if position == 0:  # First entry
                position = signal
                entry_price = current_price
                entry_date = i
                
            elif signal != position:  # Position flip required
                # Close current position and record trade
                if position == 1:  # Closing long position
                    exit_value = capital * (current_price / entry_price) * (1 - self.commission)
                else:  # Closing short position (position == -1)  
                    exit_value = capital * (2 - (current_price / entry_price)) * (1 - self.commission)
                
                # Record completed trade
                trade = {
                    'entry_date': entry_date,
                    'exit_date': i,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'position_type': 'Long' if position == 1 else 'Short',
                    'profit_loss': exit_value - capital,
                    'return_pct': (exit_value - capital) / capital * 100,
                    'days_held': (i - entry_date).days
                }
                trades.append(trade)
                
                # Open new position immediately (PERPETUAL)
                capital = exit_value * (1 - self.commission)  # Commission for new position
                position = signal
                entry_price = current_price
                entry_date = i
        
        # Close final position at end of data
        if position != 0:
            final_price = df.iloc[-1]['Close']
            
            if position == 1:  # Long position
                final_value = capital * (final_price / entry_price) * (1 - self.commission)
            else:  # Short position
                final_value = capital * (2 - (final_price / entry_price)) * (1 - self.commission)
            
            trade = {
                'entry_date': entry_date,
                'exit_date': df.index[-1],
                'entry_price': entry_price,
                'exit_price': final_price,
                'position_type': 'Long' if position == 1 else 'Short',
                'profit_loss': final_value - capital,
                'return_pct': (final_value - capital) / capital * 100,
                'days_held': (df.index[-1] - entry_date).days
            }
            trades.append(trade)
            capital = final_value
        
        self.trades = trades
        self.equity_curve = pd.DataFrame(equity_curve)
        
        return self.calculate_metrics()
    
    def calculate_metrics(self):
        """Calculate comprehensive performance metrics"""
        if not self.trades:
            return None
        
        trades_df = pd.DataFrame(self.trades)
        
        # Basic metrics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['profit_loss'] > 0])
        losing_trades = len(trades_df[trades_df['profit_loss'] < 0])
        
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        total_profit = trades_df['profit_loss'].sum()
        gross_profit = trades_df[trades_df['profit_loss'] > 0]['profit_loss'].sum()
        gross_loss = abs(trades_df[trades_df['profit_loss'] < 0]['profit_loss'].sum())
        
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
        
        # Returns analysis
        returns = trades_df['return_pct'] / 100
        avg_return = returns.mean()
        
        # Risk metrics
        if len(returns) > 1:
            std_returns = returns.std()
            sharpe_ratio = avg_return / std_returns * np.sqrt(252) if std_returns > 0 else 0
            
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0:
                downside_std = downside_returns.std()
                sortino_ratio = avg_return / downside_std * np.sqrt(252) if downside_std > 0 else 0
            else:
                sortino_ratio = np.inf
        else:
            sharpe_ratio = 0
            sortino_ratio = 0
        
        # Drawdown analysis
        if len(self.equity_curve) > 0:
            equity_values = self.equity_curve['Equity'].values
            rolling_max = np.maximum.accumulate(equity_values)
            drawdowns = (equity_values - rolling_max) / rolling_max * 100
            max_drawdown = abs(drawdowns.min()) if len(drawdowns) > 0 else 0
            
            # Intra-trade max drawdown (approximate)
            intra_trade_max_dd = max_drawdown  # Simplified calculation
        else:
            max_drawdown = 0
            intra_trade_max_dd = 0
        
        # Omega ratio calculation
        threshold = 0  # Risk-free rate
        excess_returns = returns - threshold
        positive_returns = excess_returns[excess_returns > 0].sum()
        negative_returns = abs(excess_returns[excess_returns < 0].sum())
        omega_ratio = positive_returns / negative_returns if negative_returns > 0 else np.inf
        
        # Average trade metrics
        avg_trade = trades_df['profit_loss'].mean()
        avg_winning_trade = trades_df[trades_df['profit_loss'] > 0]['profit_loss'].mean() if winning_trades > 0 else 0
        avg_losing_trade = trades_df[trades_df['profit_loss'] < 0]['profit_loss'].mean() if losing_trades > 0 else 0
        
        # Final portfolio value
        final_value = self.initial_capital + total_profit
        total_return = (final_value - self.initial_capital) / self.initial_capital * 100
        
        metrics = {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_return': total_return,
            'total_profit': total_profit,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'avg_trade': avg_trade,
            'avg_winning_trade': avg_winning_trade,
            'avg_losing_trade': avg_losing_trade,
            'max_drawdown': max_drawdown,
            'intra_trade_max_dd': intra_trade_max_dd,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'omega_ratio': omega_ratio,
            'final_value': final_value,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades
        }
        
        return metrics

class ParameterOptimizer:
    """Optimize strategy parameters using professional scoring"""
    
    def __init__(self, backtest_engine):
        self.engine = backtest_engine
        self.results = []
    
    def optimize_parameters(self, data, ema_range, sma_range, rsi_range):
        """Test all parameter combinations and rank by professional score"""
        
        print(f"Starting optimization...")
        print(f"EMA range: {ema_range}")
        print(f"SMA range: {sma_range}")
        print(f"RSI range: {rsi_range}")
        
        total_combinations = len(ema_range) * len(sma_range) * len(rsi_range)
        print(f"Total combinations to test: {total_combinations}")
        
        tested = 0
        
        for ema in ema_range:
            for sma in sma_range:
                for rsi in rsi_range:
                    # Skip invalid combinations
                    if ema >= sma:  # EMA should be faster than SMA
                        continue
                    
                    strategy = TradingStrategy(
                        ema_period=ema,
                        sma_period=sma,
                        rsi_period=rsi
                    )
                    
                    try:
                        metrics = self.engine.run_backtest(data, strategy)
                        
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
                                'ema_period': ema,
                                'sma_period': sma,
                                'rsi_period': rsi,
                                'composite_score': composite_score,
                                **metrics
                            }
                            
                            # Add zone colors for each metric
                            for metric_name, value in score_metrics.items():
                                result[f'{metric_name}_zone'] = ProfessionalScoring.get_zone_color(metric_name, value)
                            
                            self.results.append(result)
                    
                    except Exception as e:
                        print(f"Error testing EMA:{ema}, SMA:{sma}, RSI:{rsi} - {e}")
                    
                    tested += 1
                    if tested % 10 == 0:
                        print(f"Progress: {tested}/{total_combinations} combinations tested")
        
        # Sort results by composite score
        self.results.sort(key=lambda x: x['composite_score'], reverse=True)
        
        print(f"Optimization complete! Tested {len(self.results)} valid combinations")
        
        return pd.DataFrame(self.results)

def main():
    """Main execution function"""
    print("=== Bitcoin Backtesting Engine with Professional Scoring ===\n")
    
    # Initialize backtesting engine
    engine = BacktestEngine(initial_capital=100000, commission=0.001)
    
    # Fetch BTC data
    print("Step 1: Fetching Bitcoin data...")
    btc_data = engine.fetch_btc_data(start_date='2018-01-01')
    print(f"Data loaded: {len(btc_data)} days from {btc_data.index[0].date()} to {btc_data.index[-1].date()}\n")
    
    # Define parameter ranges for optimization
    ema_range = [5, 10, 15, 20, 25]
    sma_range = [20, 30, 40, 50, 60]
    rsi_range = [10, 14, 18, 22, 26]
    
    # Run optimization
    print("Step 2: Running parameter optimization...")
    optimizer = ParameterOptimizer(engine)
    results_df = optimizer.optimize_parameters(btc_data, ema_range, sma_range, rsi_range)
    
    # Display top results
    print("\nStep 3: Top 10 performing parameter combinations:")
    print("=" * 120)
    
    display_columns = [
        'ema_period', 'sma_period', 'rsi_period', 'composite_score',
        'profit_factor', 'profit_factor_zone',
        'sharpe_ratio', 'sharpe_ratio_zone',
        'win_rate', 'win_rate_zone',
        'total_trades', 'total_return'
    ]
    
    top_results = results_df.head(10)[display_columns]
    print(top_results.to_string(index=False))
    
    # Save results
    results_filename = f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    results_df.to_csv(results_filename, index=False)
    print(f"\nAll results saved to: {results_filename}")
    
    # Show best strategy details
    if len(results_df) > 0:
        best = results_df.iloc[0]
        print(f"\n=== Best Strategy Details ===")
        print(f"Parameters: EMA={best['ema_period']}, SMA={best['sma_period']}, RSI={best['rsi_period']}")
        print(f"Composite Score: {best['composite_score']:.2f}")
        print(f"Total Return: {best['total_return']:.2f}%")
        print(f"Profit Factor: {best['profit_factor']:.2f} ({best['profit_factor_zone'].upper()})")
        print(f"Sharpe Ratio: {best['sharpe_ratio']:.2f} ({best['sharpe_ratio_zone'].upper()})")
        print(f"Win Rate: {best['win_rate']:.1f}% ({best['win_rate_zone'].upper()})")
        print(f"Total Trades: {best['total_trades']} ({best['total_trades_zone'].upper()})")

if __name__ == "__main__":
    main()
