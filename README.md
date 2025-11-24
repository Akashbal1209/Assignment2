# Stock Correlation Matrix – MAH Scooter & Bajaj Holdings

## Description
This project analyzes the relationship between **Mahindra Scooter (MAHSCOOTER.NS)** and **Bajaj Holdings (BAJAJHLDNG.NS)** over the past **5 years**.  
It fetches historical stock prices, calculates the **correlation coefficient**, and presents a **correlation matrix** to show how closely the two stocks move together.

## Features
- Fetch historical stock prices using Yahoo Finance (`yfinance`)
- Compute correlation on **prices** and **daily returns**
- Generate **heatmaps** to visualize correlations
- Plot **stock price trends** and **scatter matrix of returns**
- Display **rolling 90-day correlation** to see changes over time

## Installation
Make sure you have Python installed (>=3.8) and install the required packages:

```bash
pip install yfinance pandas matplotlib seaborn
```
## Run the Project

After installing the required packages, run the script using:

```bash
python main.py
```

## Outputs:

Correlation matrix of prices

Correlation matrix of daily returns

Heatmaps and plots for visualization

Rolling correlation plot

## Example Output

Correlation of prices: ~0.94 → indicates strong positive correlation

Daily returns correlation: slightly lower, showing realistic movement

Visual plots to understand the relationship between the two stocks

