# Bitcoin Backtesting Engine with Professional Scoring

A comprehensive backtesting system for Bitcoin trading strategies using technical indicators with professional performance scoring based on industry standards.

## Features

- **Professional Scoring System**: Uses Green/Yellow/Red zones for performance evaluation
- **Multiple Indicators**: EMA, SMA, RSI with configurable parameters
- **Daily BTC Data**: Tests on historical data from 2018 to present
- **Advanced Metrics**: Sharpe, Sortino, Omega ratios, drawdown analysis
- **Interactive Visualizations**: Heatmaps, scatter plots, equity curves
- **Parameter Optimization**: Systematic testing of indicator combinations

## Performance Zones

### Green Zone (Excellent):
- Profit Factor > 4
- Sharpe Ratio > 2
- Sortino Ratio > 2.90
- Win Rate > 50%
- Max Drawdown < 25%
- Total Trades: 45-105

### Yellow Zone (Acceptable):
- Profit Factor: 2-4
- Sharpe Ratio: 1-2
- Sortino Ratio: 2-2.9
- Win Rate: 35%-50%
- Max Drawdown: 25%-40%
- Total Trades: 40-44

### Red Zone (Poor):
- Profit Factor < 2
- Sharpe Ratio < 1
- Sortino Ratio < 2
- Win Rate < 35%
- Max Drawdown > 40%
- Total Trades: < 40 or > 105

## Quick Start

1. **Setup Virtual Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Run Quick Test**:
   ```bash
   python quick_start.py
   ```
   Select option 1 for a quick single strategy test.

3. **Run Full Optimization**:
   ```bash
   python backtesting_engine.py
   ```
   This will test multiple parameter combinations and rank by professional score.

## File Structure

- `backtesting_engine.py`: Main backtesting engine with scoring system
- `visualization.py`: Advanced plotting and visualization tools
- `quick_start.py`: Interactive menu for easy testing
- `test_system.py`: System verification script
- `requirements.txt`: Required Python packages

## Strategy Logic

The trading strategy uses:
- **EMA/SMA Crossover**: Entry when EMA > SMA
- **RSI Filter**: Buy when RSI < 30 (oversold), sell when RSI > 70 (overbought)
- **Exit Conditions**: EMA < SMA or RSI overbought

## Example Usage

```python
from backtesting_engine import BacktestEngine, TradingStrategy

# Initialize engine
engine = BacktestEngine(initial_capital=100000, commission=0.001)

# Fetch BTC data
btc_data = engine.fetch_btc_data(start_date='2018-01-01')

# Create strategy
strategy = TradingStrategy(ema_period=21, sma_period=50, rsi_period=14)

# Run backtest
metrics = engine.run_backtest(btc_data, strategy)

print(f"Total Return: {metrics['total_return']:.2f}%")
print(f"Profit Factor: {metrics['profit_factor']:.2f}")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
```

## Parameter Optimization

The system automatically tests combinations of:
- **EMA periods**: 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55
- **SMA periods**: 20, 25, 30, 35, 40, 45, 50, 55, 60
- **RSI periods**: 10, 12, 14, 16, 18, 20, 22, 24, 26

Results are ranked by composite score considering all professional metrics.

## Visualizations

- **Performance Zones Scatter Plot**: Shows metric relationships with color coding
- **Parameter Heatmaps**: Visualize optimal parameter combinations  
- **Equity Curves**: Track portfolio performance over time
- **Drawdown Analysis**: Identify risk periods

## Output Files

- `backtest_results_YYYYMMDD_HHMM.csv`: Complete optimization results
- `optimization_heatmap_*.png`: Parameter optimization visualizations
- `performance_zones_*.png`: Professional scoring analysis
- `equity_curve_*.png`: Strategy performance charts

## Notes

- Uses Yahoo Finance for BTC-USD daily data
- Commission set to 0.1% per trade (configurable)
- All metrics calculated on daily returns
- Professional scoring based on institutional trading standards

## Requirements

- Python 3.8+
- pandas, numpy, yfinance, pandas_ta
- matplotlib, seaborn, plotly (for visualizations)

## Disclaimer

This is for educational and research purposes only. Past performance does not guarantee future results. Always do your own research before making investment decisions.
