"""
Quick Start Runner for Bitcoin Backtesting Engine
Simple script to get started with backtesting
"""

from backtesting_engine import BacktestEngine, TradingStrategy, ParameterOptimizer
from visualization import BacktestVisualizer
import pandas as pd
from datetime import datetime

def quick_test():
    """Run a quick test with a single strategy"""
    
    print("=== Quick Strategy Test ===\n")
    
    # Initialize engine
    engine = BacktestEngine(initial_capital=100000, commission=0.001)
    
    # Fetch data
    print("Fetching BTC data...")
    btc_data = engine.fetch_btc_data(start_date='2018-01-01')
    
    # Test a single strategy
    strategy = TradingStrategy(ema_period=21, sma_period=50, rsi_period=14)
    
    print("Running backtest...")
    metrics = engine.run_backtest(btc_data, strategy)
    
    if metrics:
        print("\nResults:")
        print(f"Total Return: {metrics['total_return']:.2f}%")
        print(f"Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Win Rate: {metrics['win_rate']:.1f}%")
        print(f"Total Trades: {metrics['total_trades']}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
        
        # Create equity curve visualization
        if len(engine.equity_curve) > 0:
            visualizer = BacktestVisualizer()
            visualizer.plot_equity_curve(engine.equity_curve, "Quick Test - EMA(21)/SMA(50)/RSI(14)")
    
    return metrics

def limited_optimization():
    """Run optimization with a smaller parameter set for faster testing"""
    
    print("=== Limited Parameter Optimization ===\n")
    
    # Initialize
    engine = BacktestEngine(initial_capital=100000, commission=0.001)
    
    # Fetch data
    print("Fetching BTC data...")
    btc_data = engine.fetch_btc_data(start_date='2020-01-01')  # Shorter period for faster testing
    
    # Limited parameter ranges
    ema_range = [10, 15, 20]
    sma_range = [30, 40, 50]
    rsi_range = [12, 14, 16]
    
    print("Running limited optimization...")
    optimizer = ParameterOptimizer(engine)
    results_df = optimizer.optimize_parameters(btc_data, ema_range, sma_range, rsi_range)
    
    if len(results_df) > 0:
        print(f"\nTested {len(results_df)} combinations")
        print("\nTop 5 results:")
        
        top_5 = results_df.head(5)[['ema_period', 'sma_period', 'rsi_period', 
                                    'composite_score', 'total_return', 'profit_factor', 
                                    'win_rate', 'total_trades']]
        print(top_5.to_string(index=False))
        
        # Save results
        filename = f"limited_optimization_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        results_df.to_csv(filename, index=False)
        print(f"\nResults saved to: {filename}")
        
        # Create visualizations
        visualizer = BacktestVisualizer()
        visualizer.plot_performance_zones(results_df)
    
    return results_df

def full_optimization():
    """Run the full optimization (this will take longer)"""
    
    print("=== Full Parameter Optimization ===\n")
    print("Warning: This will test many combinations and may take several minutes...\n")
    
    # Import and run the main function from backtesting_engine
    from backtesting_engine import main
    main()

def main_menu():
    """Interactive menu for running different tests"""
    
    print("Bitcoin Backtesting Engine - Quick Start")
    print("=" * 50)
    print("1. Quick Test (Single Strategy)")
    print("2. Limited Optimization (Fast)")
    print("3. Full Optimization (Comprehensive)")
    print("4. Load and Visualize Previous Results")
    print("0. Exit")
    print()
    
    choice = input("Select option (0-4): ").strip()
    
    if choice == '1':
        quick_test()
    elif choice == '2':
        limited_optimization()
    elif choice == '3':
        full_optimization()
    elif choice == '4':
        load_and_visualize()
    elif choice == '0':
        print("Goodbye!")
        return False
    else:
        print("Invalid choice. Please try again.")
    
    return True

def load_and_visualize():
    """Load previous results and create visualizations"""
    
    try:
        import glob
        
        # Find all result files
        result_files = glob.glob("*optimization*.csv") + glob.glob("backtest_results_*.csv")
        
        if not result_files:
            print("No result files found.")
            return
        
        print("Available result files:")
        for i, file in enumerate(result_files, 1):
            print(f"{i}. {file}")
        
        choice = input(f"Select file (1-{len(result_files)}): ").strip()
        
        try:
            file_idx = int(choice) - 1
            if 0 <= file_idx < len(result_files):
                filename = result_files[file_idx]
                print(f"Loading {filename}...")
                
                results_df = pd.read_csv(filename)
                
                # Create visualizations
                visualizer = BacktestVisualizer()
                visualizer.create_performance_report(results_df)
                visualizer.plot_performance_zones(results_df)
                
            else:
                print("Invalid selection.")
        except ValueError:
            print("Please enter a valid number.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    
    print("Bitcoin Backtesting Engine - Professional Scoring System")
    print("Testing EMA/SMA/RSI strategies on daily BTC data (2018-present)")
    print()
    
    # Check if this is first run
    try:
        import glob
        if not glob.glob("*optimization*.csv") and not glob.glob("backtest_results_*.csv"):
            print("No previous results found. Running quick test to get started...")
            quick_test()
            print("\nFor more comprehensive testing, run the menu options.")
    except:
        pass
    
    # Interactive menu
    while True:
        print("\n" + "="*50)
        if not main_menu():
            break
