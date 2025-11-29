import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

def fetch_data(tickers, start="2021-11-24", end="2025-11-24"):
    """Fetch historical closing prices for given tickers between start and end dates."""
    return yf.download(tickers, start=start, end=end)['Close'].dropna()

def compute_returns(price_df):
    """Compute daily percentage returns and log returns."""
    returns = price_df.pct_change().dropna()
    log_returns = np.log(price_df / price_df.shift(1)).dropna()
    return returns, log_returns

def plot_correlation_analysis(prices, returns, log_returns, save_path='correlation_analysis.png'):
    """Create a single comprehensive figure with 4 subplots showing correlation analysis."""
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # 1. Normalized Price Chart
    ax1 = fig.add_subplot(gs[0, :])
    normalized = (prices / prices.iloc[0]) * 100
    for col in normalized.columns:
        ax1.plot(normalized.index, normalized[col], label=col, linewidth=2)
    ax1.set_title('Normalized Stock Prices (Base = 100)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date', fontsize=11)
    ax1.set_ylabel('Normalized Price', fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. Scatter Plot with Regression
    ax2 = fig.add_subplot(gs[1, 0])
    x = returns.iloc[:, 0]
    y = returns.iloc[:, 1]
    ax2.scatter(x, y, alpha=0.5, s=15, color='blue', edgecolors='black', linewidth=0.3)
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    ax2.plot(x.sort_values(), p(x.sort_values()), "r--", linewidth=2, 
             label=f'y={z[0]:.2f}x+{z[1]:.4f}')
    corr = returns.corr().iloc[0, 1]
    ax2.set_title(f'Returns Correlation Scatter\nCorrelation = {corr:.4f}', 
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel(f'{returns.columns[0]} Returns', fontsize=10)
    ax2.set_ylabel(f'{returns.columns[1]} Returns', fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 3. Rolling Correlation
    ax3 = fig.add_subplot(gs[1, 1])
    window = 30
    rolling_corr = returns.iloc[:, 0].rolling(window=window).corr(returns.iloc[:, 1])
    ax3.plot(rolling_corr.index, rolling_corr, linewidth=2, color='purple')
    ax3.axhline(returns.corr().iloc[0, 1], color='red', linestyle='--', linewidth=2,
                label=f'Avg: {returns.corr().iloc[0, 1]:.4f}')
    ax3.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    ax3.set_title(f'{window}-Day Rolling Correlation', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Date', fontsize=10)
    ax3.set_ylabel('Correlation', fontsize=10)
    ax3.set_ylim(-1, 1)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()

def save_to_excel(prices, returns, log_returns, save_path='stock_analysis_results.xlsx'):
    """Save all data and analysis to Excel with multiple sheets."""
    with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
        # Sheet 1: Raw Prices
        prices.to_excel(writer, sheet_name='Prices')
        
        # Sheet 2: Daily Returns
        returns.to_excel(writer, sheet_name='Daily Returns')
        
        # Sheet 3: Log Returns
        log_returns.to_excel(writer, sheet_name='Log Returns')
        
        # Sheet 4: Price Statistics
        price_stats = prices.describe()
        price_stats.to_excel(writer, sheet_name='Price Statistics')
        
        # Sheet 5: Returns Statistics
        returns_stats = returns.describe()
        returns_stats.to_excel(writer, sheet_name='Returns Statistics')
        
        # Sheet 6: Correlation Analysis
        corr_df = pd.DataFrame({
            'Metric': ['Price Correlation', 'Returns Correlation', 'Log Returns Correlation'],
            'Correlation Coefficient': [
                prices.corr().iloc[0, 1],
                returns.corr().iloc[0, 1],
                log_returns.corr().iloc[0, 1]
            ]
        })
        corr_df.to_excel(writer, sheet_name='Correlation Summary', index=False)
        
        # Sheet 7: Price Correlation Matrix
        prices.corr().to_excel(writer, sheet_name='Price Correlation Matrix')
        
        # Sheet 8: Returns Correlation Matrix
        returns.corr().to_excel(writer, sheet_name='Returns Correlation Matrix')
        
        # Sheet 9: Key Metrics
        metrics_data = []
        for col in returns.columns:
            total_return = ((prices[col].iloc[-1] / prices[col].iloc[0]) - 1) * 100
            annualized_return = ((prices[col].iloc[-1] / prices[col].iloc[0]) ** (252 / len(prices)) - 1) * 100
            volatility = returns[col].std() * np.sqrt(252) * 100
            sharpe_ratio = (returns[col].mean() / returns[col].std()) * np.sqrt(252)
            
            metrics_data.append({
                'Ticker': col,
                'Start Price': prices[col].iloc[0],
                'End Price': prices[col].iloc[-1],
                'Total Return (%)': total_return,
                'Annualized Return (%)': annualized_return,
                'Mean Daily Return (%)': returns[col].mean() * 100,
                'Volatility (Annual %)': volatility,
                'Sharpe Ratio': sharpe_ratio,
                'Max Drawdown (%)': ((prices[col] / prices[col].cummax()) - 1).min() * 100
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        metrics_df.to_excel(writer, sheet_name='Key Metrics', index=False)
        
        # Sheet 10: Rolling Correlation (30-day)
        rolling_corr = returns.iloc[:, 0].rolling(window=30).corr(returns.iloc[:, 1])
        rolling_corr_df = pd.DataFrame({
            'Date': rolling_corr.index,
            '30-Day Rolling Correlation': rolling_corr.values
        })
        rolling_corr_df.to_excel(writer, sheet_name='Rolling Correlation', index=False)
        
        # Sheet 11: Cumulative Returns
        cumulative = (1 + returns).cumprod()
        cumulative.to_excel(writer, sheet_name='Cumulative Returns')
        
    print(f"✓ Saved: {save_path}")
    print(f"  - Contains 11 sheets with comprehensive analysis")

def save_summary_report(prices, returns, log_returns, save_path='analysis_report.txt'):
    """Save detailed summary statistics to a text file."""
    with open(save_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("STOCK CORRELATION ANALYSIS REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("TICKERS ANALYZED:\n")
        f.write(f"  {', '.join(prices.columns)}\n\n")
        
        f.write("ANALYSIS PERIOD:\n")
        f.write(f"  Start: {prices.index[0].strftime('%Y-%m-%d')}\n")
        f.write(f"  End: {prices.index[-1].strftime('%Y-%m-%d')}\n")
        f.write(f"  Trading Days: {len(prices)}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("PRICE STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(prices.describe().to_string())
        f.write("\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("CORRELATION MATRICES\n")
        f.write("-" * 80 + "\n")
        f.write("\nPrice Correlation:\n")
        f.write(prices.corr().to_string())
        f.write("\n\nDaily Returns Correlation:\n")
        f.write(returns.corr().to_string())
        f.write("\n\nLog Returns Correlation:\n")
        f.write(log_returns.corr().to_string())
        f.write("\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("RETURNS STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(returns.describe().to_string())
        f.write("\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("KEY METRICS\n")
        f.write("-" * 80 + "\n")
        for col in returns.columns:
            f.write(f"\n{col}:\n")
            f.write(f"  Mean Daily Return: {returns[col].mean():.6f} ({returns[col].mean()*100:.4f}%)\n")
            f.write(f"  Volatility (Std): {returns[col].std():.6f} ({returns[col].std()*100:.4f}%)\n")
            f.write(f"  Sharpe Ratio (annualized, rf=0): {(returns[col].mean()/returns[col].std())*np.sqrt(252):.4f}\n")
            f.write(f"  Total Return: {((prices[col].iloc[-1]/prices[col].iloc[0])-1)*100:.2f}%\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")
    
    print(f"✓ Saved: {save_path}")

def main():
    print("\n" + "="*80)
    print("STOCK CORRELATION ANALYSIS WITH VISUALIZATIONS")
    print("="*80 + "\n")
    
    tickers = ["MAHSCOOTER.NS", "BAJAJHLDNG.NS"]
    
    # Fetch prices
    print("Fetching data...")
    prices = fetch_data(tickers)
    print(f"✓ Data fetched: {len(prices)} trading days\n")
    
    print("=== Prices (head) ===")
    print(prices.head())
    
    # Correlation of prices
    print("\n=== Correlation of Prices ===")
    print(prices.corr())
    
    # Compute returns
    returns, log_returns = compute_returns(prices)
    print("\n=== Returns (head) ===")
    print(returns.head())
    
    # Correlation of returns
    print("\n=== Correlation of Daily Returns ===")
    print(returns.corr())
    print("\n=== Correlation of Log Returns ===")
    print(log_returns.corr())
    
    # Summary
    price_corr = prices.corr().iloc[0, 1]
    returns_corr = returns.corr().iloc[0, 1]
    print(f"\n{'='*80}")
    print(f"SUMMARY: Price Correlation = {price_corr:.4f} | Returns Correlation = {returns_corr:.4f}")
    print(f"{'='*80}\n")
    
    # Generate single comprehensive visualization
    print("Generating correlation analysis visualization...\n")
    plot_correlation_analysis(prices, returns, log_returns)
    
    # Save to Excel
    print("\nSaving data to Excel...\n")
    save_to_excel(prices, returns, log_returns)
    
    # Save summary report
    print("\nGenerating summary report...\n")
    save_summary_report(prices, returns, log_returns)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE! All outputs saved successfully.")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()