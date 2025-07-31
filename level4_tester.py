"""
LEVEL 4 PERPETUAL Strategy Tester
Compliant with COBRA trading guidelines:
- Always in position (Long/Short)
- No stop losses or take profits
- No exit conditions
- Robustness testing ready
"""

import pandas as pd
import numpy as np
from datetime import datetime
from enhanced_strategies import EnhancedTradingStrategies
from backtesting_engine import BacktestEngine, ProfessionalScoring
from visualization import BacktestVisualizer

class Level4PerpetualTester:
    """LEVEL 4 compliant perpetual strategy tester"""
    
    def __init__(self, initial_capital=100000, commission=0.001):
        self.engine = BacktestEngine(initial_capital, commission)
        self.strategies = EnhancedTradingStrategies()
        self.visualizer = BacktestVisualizer()
        
        # LEVEL 4 compliant parameter ranges for robustness testing
        self.robustness_ranges = {
            'ema_sma_rsi': {
                'ema_period': list(range(15, 26, 2)),  # 15, 17, 19, 21, 23, 25
                'sma_period': list(range(40, 61, 5)),  # 40, 45, 50, 55, 60
                'rsi_period': list(range(10, 21, 2)),  # 10, 12, 14, 16, 18, 20
                'rsi_oversold': [20, 25, 30, 35],
                'rsi_overbought': [65, 70, 75, 80]
            },
            'supertrend_rsi': {
                'atr_period': list(range(8, 16, 2)),   # 8, 10, 12, 14
                'atr_multiplier': [2.0, 2.5, 3.0, 3.5, 4.0],
                'rsi_period': list(range(10, 21, 2)),  # 10, 12, 14, 16, 18, 20
                'rsi_oversold': [20, 25, 30, 35],
                'rsi_overbought': [65, 70, 75, 80]
            },
            'macd_bollinger': {
                'macd_fast': list(range(8, 17, 2)),    # 8, 10, 12, 14, 16
                'macd_slow': list(range(20, 31, 2)),   # 20, 22, 24, 26, 28, 30
                'macd_signal': [6, 8, 9, 10, 12],
                'bb_period': list(range(15, 26, 2)),   # 15, 17, 19, 21, 23, 25
                'bb_std': [1.5, 1.8, 2.0, 2.2, 2.5]
            },
            'momentum_combo': {
                'rsi_period': list(range(10, 21, 2)),
                'macd_fast': list(range(8, 17, 2)),
                'macd_slow': list(range(20, 31, 2)),
                'cci_period': list(range(16, 25, 2)),  # 16, 18, 20, 22, 24
                'cci_oversold': [-120, -100, -80],
                'cci_overbought': [80, 100, 120]
            }
        }
    
    def validate_cobra_metrics(self, metrics):
        """Validate strategy meets COBRA requirements"""
        
        if not metrics:
            return False, "No metrics available"
        
        # Check for any RED zone metrics
        red_metrics = []
        green_metrics = []
        
        score_metrics = {
            'profit_factor': metrics['profit_factor'],
            'sharpe_ratio': metrics['sharpe_ratio'],
            'sortino_ratio': metrics['sortino_ratio'],
            'win_rate': metrics['win_rate'],
            'intra_trade_max_dd': metrics['intra_trade_max_dd'],
            'omega_ratio': metrics['omega_ratio'],
            'total_trades': metrics['total_trades']
        }
        
        for metric_name, value in score_metrics.items():
            zone = ProfessionalScoring.get_zone_color(metric_name, value)
            if zone == 'red':
                red_metrics.append(metric_name)
            elif zone == 'green':
                green_metrics.append(metric_name)
        
        # COBRA requirements
        has_red_metrics = len(red_metrics) > 0
        has_enough_green = len(green_metrics) >= 5
        
        status = "PASS" if not has_red_metrics and has_enough_green else "FAIL"
        
        message = f"GREEN: {len(green_metrics)}/7, RED: {len(red_metrics)}/7"
        if red_metrics:
            message += f" - RED METRICS: {', '.join(red_metrics)}"
        
        return status == "PASS", message
    
    def robustness_test_single_strategy(self, data, strategy_name, test_name="Parameter"):
        """Run robustness test on single strategy"""
        
        print(f"\n=== LEVEL 4 ROBUSTNESS TEST: {strategy_name.upper()} ===")
        print(f"Test Type: {test_name}")
        print(f"Data Period: {data.index[0].date()} to {data.index[-1].date()}")
        print(f"Total Days: {len(data)}")
        
        if strategy_name not in self.robustness_ranges:
            print(f"Strategy {strategy_name} not configured for robustness testing")
            return None
        
        ranges = self.robustness_ranges[strategy_name]
        
        # Calculate total combinations
        import itertools
        param_names = list(ranges.keys())
        param_values = list(ranges.values())
        combinations = list(itertools.product(*param_values))
        
        print(f"Parameter Ranges:")
        for param, values in ranges.items():
            print(f"  {param}: {values}")
        print(f"Total Combinations: {len(combinations)}")
        
        # Test all combinations
        results = []
        cobra_passes = 0
        
        for i, combination in enumerate(combinations):
            params = dict(zip(param_names, combination))
            
            try:
                # Get strategy function
                strategy_func = getattr(self.strategies, f"{strategy_name}_strategy")
                
                # Create temporary strategy
                class TempStrategy:
                    def generate_signals(self, data):
                        return strategy_func(data, **params)
                
                temp_strategy = TempStrategy()
                
                # Run backtest
                metrics = self.engine.run_backtest(data, temp_strategy)
                
                if metrics and metrics['total_trades'] > 0:
                    # Check COBRA compliance
                    cobra_pass, cobra_message = self.validate_cobra_metrics(metrics)
                    if cobra_pass:
                        cobra_passes += 1
                    
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
                        'test_type': test_name,
                        'strategy': strategy_name,
                        'combination_id': i + 1,
                        'cobra_pass': cobra_pass,
                        'cobra_message': cobra_message,
                        'composite_score': composite_score,
                        **params,
                        **metrics
                    }
                    
                    # Add zone colors
                    for metric_name, value in score_metrics.items():
                        result[f'{metric_name}_zone'] = ProfessionalScoring.get_zone_color(metric_name, value)
                    
                    results.append(result)
                
            except Exception as e:
                print(f"Error testing combination {i+1}: {e}")
            
            # Progress update
            if (i + 1) % 50 == 0:
                print(f"Progress: {i+1}/{len(combinations)} combinations tested")
        
        if results:
            results_df = pd.DataFrame(results)
            
            # Sort by COBRA compliance and then by composite score
            results_df = results_df.sort_values(['cobra_pass', 'composite_score'], ascending=[False, False])
            
            # Summary statistics
            total_tested = len(results_df)
            cobra_pass_rate = (cobra_passes / total_tested) * 100 if total_tested > 0 else 0
            
            print(f"\n=== ROBUSTNESS TEST RESULTS ===")
            print(f"Total Valid Combinations: {total_tested}")
            print(f"COBRA Compliant: {cobra_passes} ({cobra_pass_rate:.1f}%)")
            
            # Show top COBRA compliant results
            cobra_compliant = results_df[results_df['cobra_pass'] == True]
            
            if len(cobra_compliant) > 0:
                print(f"\nTop 5 COBRA Compliant Results:")
                print("-" * 80)
                
                display_cols = list(params.keys()) + [
                    'composite_score', 'total_return', 'profit_factor', 
                    'sharpe_ratio', 'win_rate', 'total_trades'
                ]
                
                top_5 = cobra_compliant.head(5)[display_cols].copy()
                
                # Format for display
                numeric_cols = ['composite_score', 'total_return', 'profit_factor', 'sharpe_ratio']
                for col in numeric_cols:
                    if col in top_5.columns:
                        top_5[col] = top_5[col].map('{:.2f}'.format)
                
                if 'win_rate' in top_5.columns:
                    top_5['win_rate'] = top_5['win_rate'].map('{:.1f}%'.format)
                
                print(top_5.to_string(index=False))
                
                # Best result details
                best = cobra_compliant.iloc[0]
                print(f"\n🏆 BEST COBRA COMPLIANT RESULT:")
                print(f"Parameters: {dict((k, v) for k, v in best.items() if k in params.keys())}")
                print(f"Composite Score: {best['composite_score']:.2f}/3.00")
                print(f"Total Return: {best['total_return']:.1f}%")
                print(f"COBRA Status: {best['cobra_message']}")
                
            else:
                print("\n❌ NO COBRA COMPLIANT RESULTS FOUND")
                print("Strategy needs optimization to meet LEVEL 4 requirements")
                
                # Show best non-compliant result
                best_overall = results_df.iloc[0]
                print(f"\nBest Overall Result (Non-Compliant):")
                print(f"COBRA Status: {best_overall['cobra_message']}")
                print(f"Composite Score: {best_overall['composite_score']:.2f}")
            
            # Save results
            filename = f"{strategy_name}_robustness_{test_name.lower()}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
            results_df.to_csv(filename, index=False)
            print(f"\nResults saved to: {filename}")
            
            return results_df
        
        else:
            print("\n❌ No valid results generated")
            return None
    
    def quick_cobra_check(self, data, strategy_name, params=None):
        """Quick COBRA compliance check for a strategy"""
        
        print(f"\n=== QUICK COBRA COMPLIANCE CHECK ===")
        print(f"Strategy: {strategy_name}")
        
        # Use default params if none provided
        if params is None:
            default_params = {
                'ema_sma_rsi': {'ema_period': 21, 'sma_period': 50, 'rsi_period': 14},
                'supertrend_rsi': {'atr_period': 10, 'atr_multiplier': 3.0, 'rsi_period': 14},
                'macd_bollinger': {'macd_fast': 12, 'macd_slow': 26, 'bb_period': 20},
                'momentum_combo': {'rsi_period': 14, 'macd_fast': 12, 'cci_period': 20}
            }
            params = default_params.get(strategy_name, {})
        
        print(f"Parameters: {params}")
        
        try:
            # Get strategy function
            strategy_func = getattr(self.strategies, f"{strategy_name}_strategy")
            
            # Create strategy
            class TempStrategy:
                def generate_signals(self, data):
                    return strategy_func(data, **params)
            
            temp_strategy = TempStrategy()
            
            # Run backtest
            metrics = self.engine.run_backtest(data, temp_strategy)
            
            if metrics:
                # Check COBRA compliance
                cobra_pass, cobra_message = self.validate_cobra_metrics(metrics)
                
                print(f"\n📊 RESULTS:")
                print(f"Total Return: {metrics['total_return']:.1f}%")
                print(f"Profit Factor: {metrics['profit_factor']:.2f}")
                print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
                print(f"Sortino Ratio: {metrics['sortino_ratio']:.2f}")
                print(f"Win Rate: {metrics['win_rate']:.1f}%")
                print(f"Max Drawdown: {metrics['max_drawdown']:.1f}%")
                print(f"Total Trades: {metrics['total_trades']}")
                
                print(f"\n🏅 COBRA COMPLIANCE:")
                print(f"Status: {'✅ PASS' if cobra_pass else '❌ FAIL'}")
                print(f"Details: {cobra_message}")
                
                # Show zone breakdown
                score_metrics = {
                    'profit_factor': metrics['profit_factor'],
                    'sharpe_ratio': metrics['sharpe_ratio'],
                    'sortino_ratio': metrics['sortino_ratio'],
                    'win_rate': metrics['win_rate'],
                    'intra_trade_max_dd': metrics['intra_trade_max_dd'],
                    'omega_ratio': metrics['omega_ratio'],
                    'total_trades': metrics['total_trades']
                }
                
                print(f"\nZone Breakdown:")
                for metric_name, value in score_metrics.items():
                    zone = ProfessionalScoring.get_zone_color(metric_name, value)
                    emoji = "🟢" if zone == "green" else "🟡" if zone == "yellow" else "🔴"
                    print(f"  {emoji} {metric_name.replace('_', ' ').title()}: {zone.upper()}")
                
                return metrics, cobra_pass
            
        except Exception as e:
            print(f"❌ Error: {e}")
            
        return None, False

def main():
    """LEVEL 4 Testing Menu"""
    
    print("🎯 LEVEL 4 PERPETUAL STRATEGY TESTER")
    print("Compliant with COBRA Trading Guidelines")
    print("=" * 60)
    
    tester = Level4PerpetualTester()
    
    # Load data from 2018 (COBRA requirement)
    print("📊 Loading BTC data from 2018-01-01 (COBRA requirement)...")
    btc_data = tester.engine.fetch_btc_data(start_date='2018-01-01')
    print(f"✅ Data loaded: {len(btc_data)} days")
    
    while True:
        print(f"\n🎯 LEVEL 4 TESTING OPTIONS:")
        print("1. Quick COBRA Compliance Check")
        print("2. Full Robustness Test (Single Strategy)")
        print("3. Compare All Strategies (COBRA Check)")
        print("4. View COBRA Requirements")
        print("0. Exit")
        
        choice = input("\nSelect option (0-4): ").strip()
        
        if choice == '1':
            print("\nAvailable Strategies:")
            strategies = ['ema_sma_rsi', 'supertrend_rsi', 'macd_bollinger', 'momentum_combo']
            for i, strategy in enumerate(strategies, 1):
                print(f"  {i}. {strategy}")
            
            try:
                strategy_choice = int(input(f"Select strategy (1-{len(strategies)}): ")) - 1
                strategy_name = strategies[strategy_choice]
                tester.quick_cobra_check(btc_data, strategy_name)
            except (ValueError, IndexError):
                print("Invalid selection.")
                
        elif choice == '2':
            print("\nAvailable Strategies for Robustness Testing:")
            available = list(tester.robustness_ranges.keys())
            for i, strategy in enumerate(available, 1):
                print(f"  {i}. {strategy}")
            
            try:
                strategy_choice = int(input(f"Select strategy (1-{len(available)}): ")) - 1
                strategy_name = available[strategy_choice]
                
                print(f"\n⚠️  WARNING: This will test many parameter combinations.")
                print(f"Estimated time: 5-15 minutes depending on strategy.")
                confirm = input("Continue? (y/n): ").lower()
                
                if confirm == 'y':
                    tester.robustness_test_single_strategy(btc_data, strategy_name, "Parameter")
            except (ValueError, IndexError):
                print("Invalid selection.")
                
        elif choice == '3':
            print("\n🔍 Quick COBRA check for all strategies...")
            strategies = ['ema_sma_rsi', 'supertrend_rsi', 'macd_bollinger', 'momentum_combo']
            
            for strategy in strategies:
                print(f"\n--- {strategy.upper()} ---")
                metrics, cobra_pass = tester.quick_cobra_check(btc_data, strategy)
                
        elif choice == '4':
            print(f"\n📋 COBRA REQUIREMENTS (LEVEL 4):")
            print("✅ NO metrics in RED zone")
            print("✅ At least 5 of 7 metrics in GREEN zone")
            print("✅ Always in position (Long/Short)")
            print("✅ No stop losses or take profits")
            print("✅ No exit conditions")
            print("✅ Data starts from 2018-01-01")
            print("✅ Robustness testing required")
            print("✅ No repainting indicators")
            
        elif choice == '0':
            print("Goodbye!")
            break
            
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    main()
