"""
Multi-Strategy Backtesting System with Indicator Range Testing
Supports testing parameter ranges for various TradingView-style strategies
"""

import pandas as pd
import numpy as np
from datetime import datetime
from enhanced_strategies import EnhancedTradingStrategies, EnhancedParameterOptimizer
from backtesting_engine import BacktestEngine
from visualization import BacktestVisualizer

class MultiStrategyTester:
    """Test multiple strategies with different parameter ranges"""
    
    def __init__(self, initial_capital=100000, commission=0.001):
        self.engine = BacktestEngine(initial_capital, commission)
        self.strategies = EnhancedTradingStrategies()
        self.optimizer = EnhancedParameterOptimizer(self.engine)
        self.visualizer = BacktestVisualizer()
        
        # Define parameter ranges for each strategy
        self.strategy_ranges = {
            'ema_sma_rsi': {
                'ema_period': [10, 15, 20, 25],
                'sma_period': [30, 40, 50, 60],
                'rsi_period': [12, 14, 16, 18],
                'rsi_oversold': [25, 30, 35],
                'rsi_overbought': [65, 70, 75]
            },
            'macd_bollinger': {
                'macd_fast': [8, 12, 16],
                'macd_slow': [21, 26, 31],
                'macd_signal': [6, 9, 12],
                'bb_period': [15, 20, 25],
                'bb_std': [1.5, 2.0, 2.5]
            },
            'adx_sar_stoch': {
                'adx_period': [12, 14, 16],
                'adx_threshold': [20, 25, 30],
                'sar_af': [0.015, 0.02, 0.025],
                'stoch_k': [12, 14, 16],
                'stoch_oversold': [15, 20, 25],
                'stoch_overbought': [75, 80, 85]
            },
            'supertrend_rsi': {
                'atr_period': [8, 10, 12],
                'atr_multiplier': [2.5, 3.0, 3.5],
                'rsi_period': [12, 14, 16],
                'rsi_oversold': [25, 30, 35],
                'rsi_overbought': [65, 70, 75]
            },
            'momentum_combo': {
                'rsi_period': [12, 14, 16],
                'macd_fast': [8, 12, 16],
                'macd_slow': [21, 26, 31],
                'cci_period': [18, 20, 22],
                'cci_oversold': [-120, -100, -80],
                'cci_overbought': [80, 100, 120]
            },
            'volatility_breakout': {
                'atr_period': [12, 14, 16],
                'bb_period': [18, 20, 22],
                'bb_std': [1.8, 2.0, 2.2],
                'kc_period': [18, 20, 22],
                'kc_multiplier': [1.8, 2.0, 2.2]
            }
        }
    
    def test_single_strategy(self, data, strategy_name, custom_ranges=None):
        """Test a single strategy with parameter optimization"""
        
        if strategy_name not in self.strategy_ranges:
            print(f"Strategy '{strategy_name}' not found. Available strategies:")
            for name, desc in self.strategies.get_strategy_list().items():
                print(f"  {name}: {desc}")
            return None
        
        # Use custom ranges if provided, otherwise use defaults
        ranges = custom_ranges if custom_ranges else self.strategy_ranges[strategy_name]
        
        print(f"\n=== Testing Strategy: {strategy_name.upper()} ===")
        print(f"Strategy: {self.strategies.get_strategy_list()[strategy_name]}")
        
        # Run optimization
        results = self.optimizer.optimize_strategy(data, strategy_name, ranges)
        
        if results:
            results_df = pd.DataFrame(results)
            
            # Show top results
            print(f"\nTop 5 parameter combinations:")
            print("=" * 120)
            
            # Create display columns based on strategy parameters
            param_cols = list(ranges.keys())
            display_cols = ['strategy'] + param_cols + [
                'composite_score', 'total_return', 'profit_factor', 
                'sharpe_ratio', 'win_rate', 'total_trades'
            ]
            
            top_5 = results_df.head(5)[display_cols].copy()
            
            # Format for display
            for col in ['total_return', 'profit_factor', 'sharpe_ratio']:
                if col in top_5.columns:
                    top_5[col] = top_5[col].map('{:.2f}'.format)
            
            if 'win_rate' in top_5.columns:
                top_5['win_rate'] = top_5['win_rate'].map('{:.1f}%'.format)
            
            print(top_5.to_string(index=False))
            
            # Save results
            filename = f"{strategy_name}_optimization_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
            results_df.to_csv(filename, index=False)
            print(f"\nResults saved to: {filename}")
            
            return results_df
        
        return None
    
    def test_all_strategies(self, data, max_combinations_per_strategy=50):
        """Test all strategies with limited parameter ranges for speed"""
        
        print("=== Testing All Strategies ===\n")
        
        all_results = []
        strategy_summaries = []
        
        for strategy_name in self.strategy_ranges.keys():
            print(f"Testing {strategy_name}...")
            
            # Limit parameter ranges for faster testing
            limited_ranges = {}
            for param, values in self.strategy_ranges[strategy_name].items():
                # Take every other value to reduce combinations
                limited_ranges[param] = values[::2] if len(values) > 2 else values
            
            try:
                results = self.optimizer.optimize_strategy(data, strategy_name, limited_ranges)
                
                if results:
                    results_df = pd.DataFrame(results)
                    all_results.extend(results)
                    
                    # Strategy summary
                    best = results_df.iloc[0]
                    summary = {
                        'strategy': strategy_name,
                        'best_score': best['composite_score'],
                        'best_return': best['total_return'],
                        'best_profit_factor': best['profit_factor'],
                        'best_sharpe': best['sharpe_ratio'],
                        'best_win_rate': best['win_rate'],
                        'total_tested': len(results_df)
                    }
                    strategy_summaries.append(summary)
                    
                    print(f"  ✅ {strategy_name}: Best score {best['composite_score']:.2f}, Return {best['total_return']:.1f}%")
                else:
                    print(f"  ❌ {strategy_name}: No valid results")
                    
            except Exception as e:
                print(f"  ❌ {strategy_name}: Error - {e}")
        
        # Create comparison summary
        if strategy_summaries:
            summary_df = pd.DataFrame(strategy_summaries)
            summary_df = summary_df.sort_values('best_score', ascending=False)
            
            print(f"\n=== Strategy Comparison Summary ===")
            print("=" * 100)
            
            display_summary = summary_df.copy()
            display_summary['best_return'] = display_summary['best_return'].map('{:.1f}%'.format)
            display_summary['best_profit_factor'] = display_summary['best_profit_factor'].map('{:.2f}'.format)
            display_summary['best_sharpe'] = display_summary['best_sharpe'].map('{:.2f}'.format)
            display_summary['best_win_rate'] = display_summary['best_win_rate'].map('{:.1f}%'.format)
            display_summary['best_score'] = display_summary['best_score'].map('{:.2f}'.format)
            
            print(display_summary.to_string(index=False))
            
            # Save comparison
            filename = f"strategy_comparison_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
            summary_df.to_csv(filename, index=False)
            print(f"\nStrategy comparison saved to: {filename}")
            
            # Save all results
            if all_results:
                all_results_df = pd.DataFrame(all_results)
                all_filename = f"all_strategies_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
                all_results_df.to_csv(all_filename, index=False)
                print(f"All results saved to: {all_filename}")
                
                return all_results_df, summary_df
        
        return None, None
    
    def quick_strategy_demo(self, data):
        """Quick demo of different strategies"""
        
        print("=== Quick Strategy Demonstration ===\n")
        
        demo_strategies = [
            ('ema_sma_rsi', {'ema_period': 21, 'sma_period': 50, 'rsi_period': 14}),
            ('macd_bollinger', {'macd_fast': 12, 'macd_slow': 26, 'bb_period': 20}),
            ('supertrend_rsi', {'atr_period': 10, 'atr_multiplier': 3.0, 'rsi_period': 14}),
            ('momentum_combo', {'rsi_period': 14, 'macd_fast': 12, 'cci_period': 20})
        ]
        
        demo_results = []
        
        for strategy_name, params in demo_strategies:
            print(f"Testing {strategy_name} with default parameters...")
            
            try:
                # Get strategy function
                strategy_func = getattr(self.strategies, f"{strategy_name}_strategy")
                
                # Create strategy object
                class TempStrategy:
                    def generate_signals(self, data):
                        return strategy_func(data, **params)
                
                temp_strategy = TempStrategy()
                
                # Run backtest
                metrics = self.engine.run_backtest(data, temp_strategy)
                
                if metrics:
                    result = {
                        'strategy': strategy_name,
                        'parameters': str(params),
                        **metrics
                    }
                    demo_results.append(result)
                    
                    print(f"  Return: {metrics['total_return']:.1f}%, "
                          f"PF: {metrics['profit_factor']:.2f}, "
                          f"Sharpe: {metrics['sharpe_ratio']:.2f}, "
                          f"Trades: {metrics['total_trades']}")
                else:
                    print(f"  No valid trades generated")
                    
            except Exception as e:
                print(f"  Error: {e}")
        
        if demo_results:
            demo_df = pd.DataFrame(demo_results)
            print(f"\n=== Demo Results Summary ===")
            
            # Sort by total return
            demo_df = demo_df.sort_values('total_return', ascending=False)
            
            summary_cols = ['strategy', 'total_return', 'profit_factor', 'sharpe_ratio', 'win_rate', 'total_trades']
            print(demo_df[summary_cols].to_string(index=False))
            
            return demo_df
        
        return None

def main():
    """Interactive menu for multi-strategy testing"""
    
    print("Multi-Strategy Backtesting System")
    print("=" * 50)
    
    # Initialize tester
    tester = MultiStrategyTester(initial_capital=100000, commission=0.001)
    
    # Load data
    print("Loading BTC data...")
    btc_data = tester.engine.fetch_btc_data(start_date='2020-01-01')  # Use recent data for speed
    print(f"Data loaded: {len(btc_data)} days\n")
    
    while True:
        print("\nOptions:")
        print("1. Quick Strategy Demo")
        print("2. Test Single Strategy (with optimization)")
        print("3. Test All Strategies (comparison)")
        print("4. List Available Strategies")
        print("5. Custom Parameter Ranges")
        print("0. Exit")
        
        choice = input("\nSelect option (0-5): ").strip()
        
        if choice == '1':
            print("\nRunning quick demo...")
            tester.quick_strategy_demo(btc_data)
            
        elif choice == '2':
            print("\nAvailable strategies:")
            strategies_list = tester.strategies.get_strategy_list()
            for i, (key, desc) in enumerate(strategies_list.items(), 1):
                print(f"  {i}. {key}: {desc}")
            
            try:
                strategy_choice = int(input(f"Select strategy (1-{len(strategies_list)}): ")) - 1
                strategy_name = list(strategies_list.keys())[strategy_choice]
                
                print(f"\nOptimizing {strategy_name}...")
                results = tester.test_single_strategy(btc_data, strategy_name)
                
                if results is not None and len(results) > 0:
                    # Create visualization
                    tester.visualizer.plot_performance_zones(results)
                
            except (ValueError, IndexError):
                print("Invalid selection.")
                
        elif choice == '3':
            print("\nTesting all strategies (this may take a few minutes)...")
            all_results, summary = tester.test_all_strategies(btc_data)
            
            if summary is not None:
                print("\nWould you like to create visualizations? (y/n)")
                if input().lower() == 'y':
                    tester.visualizer.plot_performance_zones(all_results)
                    
        elif choice == '4':
            print("\nAvailable Strategies:")
            for key, desc in tester.strategies.get_strategy_list().items():
                print(f"  {key}: {desc}")
                
        elif choice == '5':
            print("\nCustom Parameter Ranges - Feature coming soon!")
            print("You can modify the ranges in the MultiStrategyTester class.")
            
        elif choice == '0':
            print("Goodbye!")
            break
            
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
