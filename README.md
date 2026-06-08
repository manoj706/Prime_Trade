# Bitcoin Market Sentiment × Trader Performance Analysis

## Overview

This project investigates the relationship between **Bitcoin market sentiment** and **trader performance** using historical trading data from Hyperliquid and the Bitcoin Fear & Greed Index.

The goal is to identify whether market sentiment influences profitability, trading behavior, win rates, and directional bias, and to uncover actionable trading insights.

---

## Objective

Analyze how trader performance changes across different market sentiment regimes:

* Extreme Fear
* Fear
* Neutral
* Greed
* Extreme Greed

Key questions explored:

* Do traders perform better during Fear or Greed markets?
* Does market sentiment influence win rates?
* Is there a directional bias toward long or short positions?
* Can contrarian strategies outperform the average trader?
* How do top-performing traders behave under different sentiment conditions?

---

## Dataset Information

### 1. Bitcoin Fear & Greed Index

Features:

* Date
* Fear & Greed Score
* Sentiment Classification

### 2. Hyperliquid Historical Trader Data

Features include:

* Account
* Symbol
* Side (BUY / SELL)
* Execution Price
* Position Size
* Closed PnL
* Leverage
* Timestamp

---

## Data Summary

| Metric                 | Value               |
| ---------------------- | ------------------- |
| Total Trade Records    | 211,224             |
| Closed Trades Analyzed | 104,402             |
| Unique Traders         | 32                  |
| Unique Symbols         | 246                 |
| Analysis Period        | May 2023 – May 2025 |
| Total Generated PnL    | $10.25M             |

---

## Methodology

### Data Preparation

* Cleaned and standardized timestamps
* Filtered closed trades
* Merged trading data with daily Fear & Greed Index
* Engineered additional features:

  * Win/Loss indicators
  * Long/Short classification
  * Daily performance metrics

### Analytical Techniques

* Sentiment-based performance analysis
* Win rate analysis
* Directional bias analysis
* Contrarian strategy evaluation
* Statistical significance testing
* Time-series analysis
* Top trader segmentation

---

## Key Findings

### 1. Extreme Greed Generates Highest Profitability

| Sentiment     | Avg PnL | Win Rate |
| ------------- | ------- | -------- |
| Extreme Fear  | $71.03  | 76.2%    |
| Fear          | $112.63 | 87.3%    |
| Neutral       | $71.20  | 82.4%    |
| Greed         | $85.40  | 76.9%    |
| Extreme Greed | $130.21 | 89.2%    |

**Insight:** Extreme Greed produced the strongest trading performance with the highest average PnL and win rate.

---

### 2. Contrarian Strategies Outperform

| Strategy           | Avg PnL | Win Rate |
| ------------------ | ------- | -------- |
| Long During Fear   | $186.85 | 82.3%    |
| Short During Greed | $144.13 | 87.1%    |
| All Other Trades   | $61.48  | 81.7%    |

**Insight:** Contrarian positioning significantly outperformed the baseline strategy.

* Longing during Fear generated approximately 3× higher average PnL.
* Shorting during Greed generated approximately 2.3× higher average PnL.

---

### 3. Persistent Net-Short Bias

| Sentiment     | BUY % |
| ------------- | ----- |
| Extreme Fear  | 30.0% |
| Fear          | 31.0% |
| Neutral       | 32.3% |
| Greed         | 45.1% |
| Extreme Greed | 31.2% |

**Insight:** Traders maintained a structural preference toward short positions regardless of market sentiment.

---

### 4. Sentiment Significantly Impacts Performance

Statistical Testing:

* T-Test (Extreme Greed vs Extreme Fear)

  * t = 3.862
  * p = 0.000113

Result:

* Difference in profitability is statistically significant (p < 0.001)

---

## Visualizations

### Dashboard Analysis

* Average PnL by Sentiment
* Win Rate by Sentiment
* Long Bias by Sentiment
* Buy vs Sell Profitability Heatmap
* Contrarian Strategy Performance
* Total PnL by Sentiment

### Temporal Analysis

* Cumulative PnL Growth
* Fear & Greed Index Overlay
* PnL Distribution by Sentiment

### Top Trader Analysis

* Top Trader Profitability by Sentiment
* Trader Win Rate Heatmaps

---

## Project Structure

```text
├── historical_data.csv
├── fear_greed_index.csv
├── primetrade_analysis.py
├── fig1_dashboard.png
├── fig2_timeseries.png
├── fig3_toptraders.png
├── analysis_summary.txt
└── README.md
```

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* SciPy

---

## Business Implications

The analysis suggests that:

* Market sentiment can be a useful contextual feature for trading decisions.
* Contrarian positioning during emotional market extremes may generate superior risk-adjusted returns.
* Extreme Greed periods offer the highest profit opportunities but may also introduce elevated risk.
* Monitoring sentiment can improve trade timing and portfolio allocation decisions.

---

## Limitations

* Sentiment data is available at daily granularity while trades occur intraday.
* Correlation does not imply causation.
* Trader-specific risk management strategies were not modeled.
* Results are based solely on Hyperliquid trading activity and may not generalize across all crypto markets.

---
