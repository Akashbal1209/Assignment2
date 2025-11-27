# Stock Correlation Analysis

A comprehensive Python-based tool for analyzing correlations between stock prices and returns, with visualizations and detailed reporting.

## Overview

This project analyzes the correlation between two stocks (Mahindra Scooters and Bajaj Holdings) over a 4-year period, providing insights into:
- Price movements and correlations
- Daily returns correlations
- Rolling correlation trends
- Risk metrics and performance statistics

## Features

- **Data Fetching**: Automatically downloads historical stock data using Yahoo Finance API
- **Multiple Correlation Types**: Analyzes price correlation, returns correlation, and log returns correlation
- **Comprehensive Visualizations**: Generates a multi-panel chart showing:
  - Normalized price trends
  - Returns scatter plot with regression line
  - 30-day rolling correlation
- **Excel Export**: Saves all data and analysis to a multi-sheet Excel workbook
- **Text Report**: Generates a detailed summary report with all key statistics

## Requirements

```
yfinance
pandas
numpy
matplotlib
seaborn
openpyxl
```

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
```

2. Activate the virtual environment:
- Windows: `venv\Scripts\activate`
- macOS/Linux: `source venv/bin/activate`

3. Install required packages:
```bash
pip install yfinance pandas numpy matplotlib seaborn openpyxl
```

## Usage

Simply run the main script:
```bash
python main.py
```

## Output Files

The script generates three output files:

1. **correlation_analysis.png** - Comprehensive visualization with 4 panels:
   - Normalized stock prices (base = 100)
   - Returns correlation scatter plot
   - 30-day rolling correlation

2. **stock_analysis_results.xlsx** - Excel workbook with 11 sheets:
   - Prices (raw historical data)
   - Daily Returns
   - Log Returns
   - Price Statistics
   - Returns Statistics
   - Correlation Summary
   - Price Correlation Matrix
   - Returns Correlation Matrix
   - Key Metrics (Sharpe Ratio, Max Drawdown, etc.)
   - Rolling Correlation
   - Cumulative Returns

3. **analysis_report.txt** - Detailed text report with:
   - Complete price and returns statistics
   - All correlation matrices
   - Key performance metrics
   - Risk indicators

## Key Findings (Example)

Based on the analysis period (2021-11-24 to 2025-11-24):

- **Price Correlation**: 0.9459 (very high - stocks move together)
- **Returns Correlation**: 0.3292 (moderate - daily returns less correlated)
- **Trading Days Analyzed**: 989

This demonstrates an important concept: while stock prices may trend together over time, their daily returns show more independence, suggesting different short-term volatility patterns.

## Customization

To analyze different stocks, modify the `tickers` list in the `main()` function:

```python
tickers = ["YOUR_STOCK1.NS", "YOUR_STOCK2.NS"]
```

To change the analysis period, modify the date range in `fetch_data()`:

```python
prices = fetch_data(tickers, start="YYYY-MM-DD", end="YYYY-MM-DD")
```

## Understanding the Metrics

- **Price Correlation**: Measures how the stock prices move together over time
- **Returns Correlation**: Measures how daily percentage changes are related
- **Sharpe Ratio**: Risk-adjusted return (higher is better)
- **Max Drawdown**: Largest peak-to-trough decline (indicates risk)
- **Volatility**: Standard deviation of returns (indicates price stability)

## Notes

- The script uses adjusted closing prices for accuracy
- All returns are calculated as percentage changes
- Sharpe ratios assume a risk-free rate of 0
- Rolling correlations use a 30-day window