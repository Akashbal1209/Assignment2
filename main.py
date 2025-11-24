import yfinance as yf
import pandas as pd
import numpy as np

def fetch_data(tickers, start="2021-11-24", end="2025-11-24"):
    """Fetch historical closing prices for given tickers between start and end dates."""
    return yf.download(tickers, start=start, end=end)['Close'].dropna()

def compute_returns(price_df):
    """Compute daily percentage returns and log returns."""
    returns = price_df.pct_change().dropna()
    log_returns = np.log(price_df / price_df.shift(1)).dropna()
    return returns, log_returns

def main():
    tickers = ["MAHSCOOTER.NS", "BAJAJHLDNG.NS"]
    
    # Fetch prices
    prices = fetch_data(tickers)
    print("=== Prices (head) ===\n", prices.head())

    # Correlation of prices
    print("\n=== Correlation of Prices ===\n", prices.corr())

    # Compute returns
    returns, log_returns = compute_returns(prices)
    print("\n=== Returns (head) ===\n", returns.head())

    # Correlation of returns
    print("\n=== Correlation of Daily Returns ===\n", returns.corr())
    print("\n=== Correlation of Log Returns ===\n", log_returns.corr())

    # Summary
    print(f"\nSummary: price corr = {prices.corr().iloc[0,1]:.4f}, returns corr = {returns.corr().iloc[0,1]:.4f}")

if __name__ == "__main__":
    main()
