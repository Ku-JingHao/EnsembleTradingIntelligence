# Stock Price Prediction and Zero-Lag MACD Enhancement Using Ensemble Methods

## Overview
This project investigates how ensemble machine learning models using simple averaging can improve the accuracy and generalizability of stock price predictions. These enhanced predictions are then integrated into trading strategies based on the Zero-Lag MACD technical indicator. 
The ultimate goal is to filter trading signals by forecasting one-step-ahead stock prices, enabling more effective buy and sell decisions in real-world trading scenarios.

By combining robust predictive models with technical analysis, this system aims to:
- To propose an ensemble machine learning approach for stock price prediction through the enhancement of technical indicators in a trading strategy.
- To compare the performance of individual machine learning models against ensemble machine learning approaches for stock price prediction.
- To assess the real-world viability and benchmark conventional statistical methods with the proposed ensemble machine learning approach through backtesting and real-time trading simulations.
  
---

## 📌 Project Motivation
Traditional technical trading strategies, such as those based on MACD or Zero-Lag MACD, often generate false signals due to market noise and lagging indicators.

By integrating machine learning predictions—especially ensemble models that combine the strengths of multiple algorithms, we can filter out less reliable signals and improve overall trading performance.

---

## ✨ Key Features
- **Comprehensive Model Library**: Includes Linear Regression, Random Forest, XGBoost, LSTM, GRU, and multiple ensemble combinations  
- **Ensemble Averaging**: Benchmarks various ensemble combinations to identify the best-performing model  
- **Zero-Lag MACD Enhancement**: Uses predicted prices to confirm/filter trading signals  
- **Parameter Optimization**: Auto-tuning of MACD/Zero-Lag MACD for each stock and strategy to maximize the ROI  
- **Backtesting Framework**: With detailed metrics and visualizations  
- **Modular & Extensible**: Easily add new models, indicators, or data sources  

---

## 📂 Project Structure
```text
├── app.py                      # Backtesting for classic MACD/Zero-Lag MACD strategies (no AI)
├── parameter_optimization.py   # Parameter optimization for classic strategies
├── stock_analysis.py           # Data fetching, visualization, and analysis (classic)
│
├── ai_app.py                   # Backtesting for AI-enhanced MACD/Zero-Lag MACD strategies
├── ai_parameter_optimization.py# Parameter optimization for AI-enhanced strategies
├── ai_stock_analysis.py        # Data fetching, visualization, and analysis (AI-enhanced)
├── model_loader.py             # Model loading and prediction utilities for AI models
│
├── Daily_ModelConstruction.ipynb      # Jupyter notebook: model training & ensemble research (daily)
├── Hourly_ModelConstruction.ipynb     # Jupyter notebook: model training & ensemble research (hourly)
├── Daily_BacktestingFramework.ipynb   # Jupyter notebook: backtesting (daily)
├── Hourly_BacktestingFramework.ipynb  # Jupyter notebook: backtesting (hourly)
│
├── models/                     # Saved/trained models and configs
├── realtime_tradingbot/        # Various types of real-time trading bots using the innovative ML-enhanced Zero-Lag MACD trading strategy
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## ⚙️ How It Works

### 1. **Data Preparation**
- Fetches historical stock data (hourly/daily) using `yfinance`
- Preprocessing and feature engineering

### 2. **Model Training**
- Trains individual models:
  - Linear Regression
  - Random Forest
  - XGBoost
  - LSTM
  - GRU
- Builds ensemble combinations using **simple averaging**

### 3. **Model Selection**
- Evaluates models on:
  - RMSE
  - MSE
  - MAE
  - MAPE
  - R² Score
  - CV
- Selects best performing model/ensemble per stock and interval

### 4. **Prediction-Enhanced Trading**
- Integrates predictions with Zero-Lag MACD
- Only executes trades if predicted direction confirms MACD signal

### 5. **Parameter Optimization**
- Uses `Optuna` to tune MACD parameters per strategy

### 6. **Backtesting & Evaluation**
- Runs full backtests with:
  - Equity curve visualization
  - Trade logs
  - Performance metrics

### 7. **Real-Time Evaluation**
- Evaluates constructed trading strategies and model predictions on live or streaming market data.
- Monitors performance and adapts strategies in real time for practical deployment.
---

## 📈 Zero-Lag MACD Strategy Enhancement

### 🔹 Classic Approach
- Trades based solely on MACD/Zero-Lag MACD crossovers

### 🔹 AI-Enhanced Approach
- Executes trade **only if**:
  - Technical signal (MACD crossover) **and**
  - **Model prediction** confirms direction  
  *(e.g., only BUY if MACD crosses up **and** predicted price > current)*

### ✅ Result:
- Fewer false signals  
- Improved **risk-adjusted returns**

---

## 🚀 Usage Instructions

### 1. Classic Backtesting (No Machine Learning)

**Use:**
- `app.py`
- `parameter_optimization.py`
- `stock_analysis.py`

These scripts/notebooks allow you to run and optimize conventional MACD/Zero-Lag MACD strategies on any stock and timeframe for backtesting, enabling evaluation of the trading strategies' ROI. They also include parameter optimization features to maximize ROI.

---

### 2. Machine Learning-Enhanced Backtesting

**Use:**
- `ai_app.py`
- `ai_parameter_optimization.py`
- `ai_stock_analysis.py`
- `model_loader.py`

These scripts/notebooks let you:
- Use the deployed individual and ensemble machine learning models.
- Integrate predictions into conventional MACD/Zero-Lag MACD trading strategies to evaluate novel machine learning–enhanced Zero-Lag MACD strategies.
- Optimize parameters and backtest the enhanced strategies.
  
---

### 3. Jupyter Notebooks

Use the provided `.ipynb` notebooks for:
- Step-by-step experimentation  
- Visualization  
- Research documentation

**Included Notebooks:**
- `Daily_ModelConstruction.ipynb`: Model construction and ensemble research for daily data.
- `Hourly_ModelConstruction.ipynb`: Model construction and ensemble research for hourly data.
- `Daily_BacktestingFramework.ipynb`: Backtesting and evaluation of enhanced Zero-Lag MACD strategies (daily).
- `Hourly_BacktestingFramework.ipynb`: Backtesting and evaluation of enhanced Zero-Lag MACD strategies (hourly).

These notebooks provide for building models, constructing ensembles, and evaluating the enhanced Zero-Lag MACD trading strategy.

---

### 4. Model Deployment & Loading

- Trained models are saved in the `models/` directory.
- Use `model_loader.py` to:
  - Load models
  - Generate predictions for new data or live trading
    
---

### 5. Real-Time Trading Bots

- The `realtime_tradingbot/` directory contains implementations of real-time trading bots using the ML-enhanced Zero-Lag MACD strategy.
- These bots are designed for live trading scenarios, integrating model predictions and technical signals for automated decision-making.


## 🔁 Flowcharts

*Insert your flowcharts here to illustrate the overall system architecture and trading strategy construction process.*

---

## 📦 Dependencies

- Python 3.8+
- `shiny`
- `shinyswatch`
- `yfinance`
- `plotly`
- `pandas`
- `numpy`
- `Jinja2`
- `darts`
- `optuna`
- `scikit-learn`
- `torch`
- `pytorch-lightning`
- `matplotlib`
- `seaborn`
- `tabulate`
- `joblib`

### Install dependencies with:

pip install -r requirements.txt

---

## 📚 References

- **Zero-Lag MACD**  
  [https://www.tradingview.com/script/chlgDc8f-Zero-Lag-Multi-Timeframe-MACD/](https://www.tradingview.com/script/chlgDc8f-Zero-Lag-Multi-Timeframe-MACD/)

- **Darts Time Series Library**  
  [https://github.com/unit8co/darts](https://github.com/unit8co/darts)

- **Optuna Hyperparameter Optimization**  
  [https://optuna.org/](https://optuna.org/)

- **Yahoo Finance API**  
  [https://github.com/ranaroussi/yfinance](https://github.com/ranaroussi/yfinance)

- **Alpaca API**  
  [https://alpaca.markets/](https://alpaca.markets/)

---

## 📜 License

This project is for academic and research purposes only.  
See the LICENSE file for details.

---

## 🤝 Contributions & Contact

For questions, suggestions, or contributions:

- Open an issue  
- Or contact the project maintainer

