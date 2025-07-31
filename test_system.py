"""
Simple test script to verify the backtesting engine works
"""

# Test import
try:
    import pandas as pd
    import numpy as np
    import yfinance as yf
    import pandas_ta as ta
    print("✅ All required packages imported successfully!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    exit(1)

# Test basic functionality
def test_basic_functionality():
    print("\n=== Testing Basic Functionality ===")
    
    # Test data fetching
    try:
        print("Testing data fetch...")
        ticker = yf.Ticker("BTC-USD")
        # Fetch just a small amount of data for testing
        data = ticker.history(start="2024-01-01", end="2024-01-10", interval='1d')
        print(f"✅ Data fetch successful: {len(data)} rows")
        
        # Test indicator calculation
        print("Testing indicator calculation...")
        close_prices = data['Close']
        ema = ta.ema(close_prices, length=10)
        sma = ta.sma(close_prices, length=20)
        rsi = ta.rsi(close_prices, length=14)
        
        print(f"✅ EMA calculated: {len(ema.dropna())} values")
        print(f"✅ SMA calculated: {len(sma.dropna())} values") 
        print(f"✅ RSI calculated: {len(rsi.dropna())} values")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in basic functionality test: {e}")
        return False

# Test our backtesting engine import
def test_engine_import():
    print("\n=== Testing Engine Import ===")
    
    try:
        from backtesting_engine import BacktestEngine, TradingStrategy, ProfessionalScoring
        print("✅ Backtesting engine imported successfully!")
        
        # Test creating instances
        engine = BacktestEngine()
        strategy = TradingStrategy()
        print("✅ Engine and strategy instances created!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error importing backtesting engine: {e}")
        return False

if __name__ == "__main__":
    print("Bitcoin Backtesting Engine - System Test")
    print("=" * 50)
    
    # Run tests
    test1 = test_basic_functionality()
    test2 = test_engine_import()
    
    if test1 and test2:
        print("\n🎉 All tests passed! The system is ready to use.")
        print("\nNext steps:")
        print("1. Run 'python quick_start.py' for interactive menu")
        print("2. Run 'python backtesting_engine.py' for full optimization")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
