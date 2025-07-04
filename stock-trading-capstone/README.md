# Stock Trading Strategy Analysis: Machine Learning vs Traditional Approaches

## Project Overview

This capstone project investigates the effectiveness of various machine learning models and trading strategies compared to a simple buy-and-hold approach for stock market prediction. The research focuses on Apple Inc. (AAPL) stock data from 2019-2024, examining whether sophisticated ML models, foundation models, or technical indicators can outperform traditional passive investment strategies.

## Research Questions

1. Can machine learning models predict stock price movements better than random chance?
2. How do foundation models (Sundial) perform in financial time series prediction?
3. Do technical trading strategies provide superior returns to buy-and-hold?
4. What are the trade-offs between model complexity and investment performance?

## Dataset

- **Source**: Yahoo Finance API
- **Asset**: AAPL (Apple Inc.)
- **Time Period**: January 2019 - December 2024 (5 years)
- **Data Points**: ~1,260 trading days
- **Features**: Price data (OHLCV), technical indicators, derived features

## Technical Indicators Used

- **Trend**: Simple Moving Averages (SMA), Exponential Moving Averages (EMA)
- **Momentum**: Relative Strength Index (RSI), MACD
- **Volatility**: Bollinger Bands, Average True Range (ATR)
- **Volume**: Volume-based indicators
- **Price**: Price change ratios, rolling statistics

## Methodology

### 1. Data Collection and Preprocessing
- Historical stock data acquisition via yfinance
- Feature engineering with 137 technical indicators
- Data cleaning and outlier handling
- Train/test split (80/20)

### 2. Model Development
**Machine Learning Models:**
- Logistic Regression (baseline)
- Random Forest
- Gradient Boosting
- LSTM Neural Networks
- AutoML (H2O.ai)
- Ensemble methods

**Foundation Model:**
- Sundial time series foundation model
- Zero-shot prediction capabilities
- Trend and momentum analysis

**Technical Strategies:**
- Simple Moving Average crossover
- RSI-based signals
- Mean reversion
- Bollinger Bands breakout

### 3. Evaluation Metrics
- **Classification Accuracy**: Directional prediction accuracy
- **Financial Metrics**: Total return, Sharpe ratio, maximum drawdown
- **Risk Metrics**: Volatility, Value at Risk (VaR)
- **Trading Metrics**: Number of trades, transaction costs

### 4. Backtesting Framework
- 5-year historical backtesting
- Transaction costs inclusion (0.1% commission)
- Realistic trading constraints
- Walk-forward analysis

## Project Structure

```
stock-trading-capstone/
├── notebooks/
│   ├── 01_data_eda.ipynb              # Exploratory Data Analysis
│   ├── 02_modeling.ipynb              # ML Model Development
│   ├── 03_backtesting.ipynb           # Strategy Backtesting
│   ├── 04_improved_model.ipynb        # Model Optimization
│   ├── 05_model_comparison.ipynb      # Comprehensive Comparison
│   ├── 06_data_analysis.ipynb         # Additional Analysis
│   └── 07_sundial_foundation_model.ipynb # Foundation Model Testing
├── src/
│   ├── data_loader.py                 # Data acquisition utilities
│   ├── feature_engineering.py         # Technical indicator calculation
│   ├── models.py                      # ML model implementations
│   ├── trading_strategy.py            # Trading strategy classes
│   ├── evaluation.py                  # Performance evaluation
│   └── model_comparison_streamlit.py  # Interactive comparison tool
├── data/cache/                        # Cached datasets
├── requirements.txt                   # Python dependencies
└── README.md                         # Project documentation
```

## Key Results

### Model Performance Summary

| Approach | Accuracy | Total Return | Sharpe Ratio | Max Drawdown |
|----------|----------|--------------|--------------|--------------|
| **Buy & Hold** | N/A | **76.8%** | **1.45** | **22.1%** |
| AutoML | 58.0% | 15.2% | 0.89 | 28.4% |
| Linear Regression | 52.0% | 17.4% | 1.17 | 24.6% |
| Random Forest | 54.5% | 12.8% | 0.76 | 31.2% |
| LSTM | 51.8% | 8.9% | 0.65 | 35.1% |
| SMA Strategy | N/A | 11.3% | 0.85 | 26.8% |
| RSI Strategy | N/A | 9.7% | 0.71 | 29.3% |
| Sundial Foundation | N/A | ~35% | ~1.20 | ~25% |

### Key Findings

1. **Buy-and-Hold Dominance**: The simple buy-and-hold strategy achieved the highest total return (76.8%) with the best risk-adjusted performance (Sharpe ratio: 1.45).

2. **ML Model Limitations**: Despite achieving above-random accuracy (52-58%), machine learning models failed to translate prediction accuracy into superior investment returns.

3. **Foundation Model Performance**: Sundial showed excellent short-term prediction accuracy (MAPE: 6.4%, direction accuracy: 100%) but couldn't sustain long-term outperformance.

4. **Transaction Cost Impact**: Active trading strategies suffered from cumulative transaction costs that eroded potential gains.

5. **Market Trend Effect**: In a strong uptrending market (AAPL 2019-2024), passive strategies benefited from consistent market appreciation.

## Technical Implementation

### Dependencies
```bash
pip install -r requirements.txt
```

### Running the Analysis
```bash
# Start Jupyter Notebook
jupyter notebook

# Run comprehensive comparison
python src/model_comparison_streamlit.py

# Interactive analysis
streamlit run src/model_comparison_streamlit.py
```

### Key Notebooks
- `05_model_comparison.ipynb`: Complete strategy comparison
- `07_sundial_foundation_model.ipynb`: Foundation model analysis
- `01_data_eda.ipynb`: Data exploration and visualization

## Conclusions and Implications

### For Investors
- **Long-term Strategy**: Buy-and-hold remains the most effective approach for long-term wealth building
- **Risk Management**: Passive strategies provide better risk-adjusted returns with lower complexity
- **Cost Efficiency**: Minimal transaction costs significantly impact long-term performance

### For Practitioners
- **Model Accuracy vs Returns**: High prediction accuracy doesn't guarantee investment success
- **Implementation Challenges**: Real-world constraints (costs, slippage, timing) affect strategy performance
- **Foundation Models**: Promising technology but requires further development for practical application

### For Researchers
- **Benchmarking**: Simple strategies should always be included as baselines
- **Evaluation Metrics**: Financial metrics are more relevant than traditional ML metrics for investment strategies
- **Market Conditions**: Strategy performance is highly dependent on market regimes and time periods

## Future Work

1. **Multi-Asset Analysis**: Extend to diverse asset classes and market conditions
2. **Regime Detection**: Incorporate market regime identification for dynamic strategy selection
3. **Alternative Data**: Include sentiment analysis, news data, and alternative data sources
4. **Real-time Implementation**: Develop production-ready trading system with live data feeds
5. **Risk Management**: Implement sophisticated risk management and position sizing techniques

## References

- Yahoo Finance API for market data
- H2O.ai AutoML documentation
- Sundial Foundation Model research papers
- Technical Analysis literature (Murphy, 1999)
- Modern Portfolio Theory (Markowitz, 1952)

---

*This project demonstrates the importance of thorough benchmarking and the enduring effectiveness of simple investment strategies in modern financial markets.*
