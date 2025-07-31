# Bitcoin Indicators Backtesting Project

## Project Overview
This project implements a comprehensive backtesting system for Bitcoin trading strategies using technical indicators. The main focus is on testing combinations of EMA, SMA, and RSI indicators with configurable parameters on **daily BTC data** from January 1, 2018 to present day.

## Current Architecture
- `fetch_btc_data.py`: Data fetching from Yahoo Finance
- `calculate_indicators.py`: Technical indicator calculations using pandas_ta
- `requirements.txt`: Project dependencies

## Data Requirements
- **Timeframe**: Daily BTC data
- **Date Range**: January 1, 2018 to present day
- **Source**: Yahoo Finance (BTC-USD)
- **Data Points**: OHLCV (Open, High, Low, Close, Volume)

## Analysis of Existing Backtesting Results
Based on the provided CSV data (BTCST003_BTCUSD_1 hour_2025-07-31.csv), the strategy tests:
- EMA periods: 5, 15, 25, 35, 45, 55
- SMA periods: 5, 15, 25, 35, 45, 55
- RSI periods: 5, 15, 25, 35, 45, 55

Best performing combinations show:
- Profit factors up to 5.5+
- Win rates around 40-60%
- Net profits exceeding 2000%

## Professional Scoring Metrics (from Majorstats.jpg)
Key performance metrics with optimal ranges:

### Green Zone (Excellent Performance):
- Intra-Trade Max DD: < 25%
- Sortino Ratio: > 2.90
- Sharpe Ratio: > 2
- Profit Factor: > 4
- % Profitable: > 50%
- # of Trades: 45-105
- Omega Ratio: > 1.31

### Yellow Zone (Acceptable Performance):
- Intra-Trade Max DD: 25%-40%
- Sortino Ratio: 2-2.9
- Sharpe Ratio: 1-2
- Profit Factor: 2-4
- % Profitable: 35%-50%
- # of Trades: 40-44
- Omega Ratio: 1.1-1.31

### Red Zone (Poor Performance):
- Intra-Trade Max DD: > 40%
- Sortino Ratio: < 2
- Sharpe Ratio: < 1
- Profit Factor: < 2
- % Profitable: < 35%
- # of Trades: < 40 & > 105
- Omega Ratio: < 1.1

## Next Steps
1. ✅ Create comprehensive backtesting framework with professional scoring
2. ✅ Implement parameter optimization using the scoring system
3. ✅ Add visualization and reporting with color-coded performance zones
4. ✅ Support multiple timeframes
5. ✅ Generate detailed performance reports with all key metrics

## Implementation Complete - LEVEL 4 COBRA COMPLIANT
Created a comprehensive Bitcoin backtesting system with LEVEL 4 PERPETUAL strategies:

### Core Files:
- `backtesting_engine.py`: **UPDATED** - PERPETUAL backtesting (always in position)
- `enhanced_strategies.py`: **UPDATED** - 6 LEVEL 4 compliant strategies  
- `level4_tester.py`: **NEW** - COBRA compliance checker and robustness tester
- `multi_strategy_tester.py`: Multi-strategy testing and comparison
- `visualization.py`: Advanced visualizations with zone-based performance analysis
- `quick_start.py`: Easy-to-use runner script with interactive menu

### LEVEL 4 PERPETUAL STRATEGIES (COBRA Compliant):
**✅ All strategies modified for LEVEL 4:**
1. **EMA/SMA/RSI PERPETUAL**: Always Long/Short based on trend
2. **MACD + Bollinger PERPETUAL**: Momentum + volatility, always positioned
3. **ADX + SAR + Stochastic PERPETUAL**: Trend strength, always positioned
4. **Supertrend + RSI PERPETUAL**: Advanced trend following, no exits
5. **Multi-Momentum PERPETUAL**: RSI + MACD + CCI scoring system
6. **Volatility Breakout PERPETUAL**: ATR + BB + KC, always positioned

### COBRA COMPLIANCE FEATURES:
- ✅ **NO STOP LOSSES OR TAKE PROFITS**: Strategy logic is the stop loss
- ✅ **ALWAYS IN POSITION**: Long or Short, never cash
- ✅ **NO EXIT CONDITIONS**: Only position flips (Long ↔ Short)
- ✅ **PERPETUAL TRADING**: Continuous market exposure
- ✅ **ROBUSTNESS TESTING**: Parameter, Exchange, Timeframe testing ready
- ✅ **PROFESSIONAL SCORING**: Green/Yellow/Red zones
- ✅ **NO REPAINTING**: Only forward-looking indicators
- ✅ **DATA FROM 2018**: Meets COBRA historical requirements

### COBRA METRICS VALIDATION:
**Required Standards:**
- 🚫 **NONE in RED zone** (automatic fail if any red)
- ✅ **At least 5 of 7 in GREEN zone** (minimum pass)
- 📊 **All 7 metrics tracked**: PF, Sharpe, Sortino, Win Rate, Drawdown, Omega, Trades

### Robustness Testing Ready:
- **Parameter Testing**: Multiple ranges for each indicator
- **Exchange Testing**: Ready for 5+ exchanges (BTC-USDT/USDC/USD)
- **Timeframe Testing**: Configurable for different timeframes
- **Historical Data**: 2018-01-01 start date compliance

### Usage for LEVEL 4:
1. **COBRA Check**: `python level4_tester.py` → Option 1
2. **Robustness Test**: `python level4_tester.py` → Option 2  
3. **Strategy Comparison**: `python level4_tester.py` → Option 3
4. **Requirements**: `python level4_tester.py` → Option 4

### Key LEVEL 4 Changes:
- **Signal Logic**: Forward-fill signals to ensure always in position
- **Backtesting Engine**: Modified for perpetual Long/Short trading
- **No Exit Conditions**: Removed all stop losses and take profits
- **Position Management**: Immediate position flips (Long ↔ Short)
- **Commission Handling**: Applied on position changes only
