# Stock Price Prediction and Zero-Lag MACD Enhancement Using Ensemble Methods

# Stock Price Prediction & Zero-Lag MACD Trading Strategy Enhancement with Ensemble Machine Learning

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
.
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
├── requirements.txt            # Python dependencies
└── README.md                   # This file
