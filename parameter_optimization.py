import pandas as pd
import numpy as np
import warnings
from stock_analysis import calculate_macd, calculate_zero_lag_macd, generate_signals, backtest_strategy, crossover
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
from itertools import product

warnings.filterwarnings('ignore')

def evaluate_parameters(args):
    data, filtered_data, indicator_type, strategy_type, start_date, end_date, fast, slow, signal, transaction_fee = args
    try:
        if indicator_type == "MACD":
            indicators = calculate_macd(data, fast_length=fast, slow_length=slow, signal_length=signal)
            signals = pd.DataFrame(index=data.index)
            signals['Signal'] = 0
            
            if strategy_type == "Buy Above Sell Above":
                buy_condition = (crossover(indicators['MACD'], indicators['Signal']) & (indicators['MACD'] > 0))
                sell_condition = (crossover(indicators['Signal'], indicators['MACD']) & (indicators['MACD'] > 0))
            elif strategy_type == "Buy Below Sell Above":
                buy_condition = (crossover(indicators['MACD'], indicators['Signal']) & (indicators['MACD'] < 0))
                sell_condition = (crossover(indicators['Signal'], indicators['MACD']) & (indicators['MACD'] > 0))
            elif strategy_type == "Buy Above Sell Below":
                buy_condition = (crossover(indicators['MACD'], indicators['Signal']) & (indicators['MACD'] > 0))
                sell_condition = (crossover(indicators['Signal'], indicators['MACD']) & (indicators['MACD'] < 0))
            elif strategy_type == "Buy Below Sell Below":
                buy_condition = (crossover(indicators['MACD'], indicators['Signal']) & (indicators['MACD'] < 0))
                sell_condition = (crossover(indicators['Signal'], indicators['MACD']) & (indicators['MACD'] < 0))
            elif strategy_type == "Histogram Trend Reversal":
                histogram = indicators['Histogram']
                histogram_diff = histogram.diff()
                buy_condition = ((histogram.shift(1) < 0) & (histogram_diff.shift(1) < 0) & (histogram_diff > 0))
                sell_condition = ((histogram.shift(1) > 0) & (histogram_diff.shift(1) > 0) & (histogram_diff < 0))
            
            signals.loc[buy_condition, 'Signal'] = 1
            signals.loc[sell_condition, 'Signal'] = -1
            filtered_signals = signals.loc[start_date:end_date]
            backtest_results = backtest_strategy(filtered_data, filtered_signals, transaction_fee=transaction_fee)
            
            if backtest_results:
                return {
                    'Fast_Length': fast,
                    'Slow_Length': slow,
                    'Signal_Length': signal,
                    'ROI': backtest_results['roi'],
                    'Max_Profit': backtest_results['max_profit'],
                    'Max_Loss': backtest_results['max_loss']
                }
        elif indicator_type == "Zero Lag MACD":
            for macd_ema in range(1, 31):
                indicators = calculate_zero_lag_macd(data, fast_length=fast, slow_length=slow, signal_length=signal, macd_ema_length=macd_ema)
                signals = pd.DataFrame(index=data.index)
                signals['Signal'] = 0
                
                if strategy_type == "Buy Above Sell Above":
                    buy_condition = (crossover(indicators['ZL_MACD'], indicators['Signal']) & (indicators['ZL_MACD'] > 0))
                    sell_condition = (crossover(indicators['Signal'], indicators['ZL_MACD']) & (indicators['ZL_MACD'] > 0))
                elif strategy_type == "Buy Below Sell Above":
                    buy_condition = (crossover(indicators['ZL_MACD'], indicators['Signal']) & (indicators['ZL_MACD'] < 0))
                    sell_condition = (crossover(indicators['Signal'], indicators['ZL_MACD']) & (indicators['ZL_MACD'] > 0))
                elif strategy_type == "Buy Above Sell Below":
                    buy_condition = (crossover(indicators['ZL_MACD'], indicators['Signal']) & (indicators['ZL_MACD'] > 0))
                    sell_condition = (crossover(indicators['Signal'], indicators['ZL_MACD']) & (indicators['ZL_MACD'] < 0))
                elif strategy_type == "Buy Below Sell Below":
                    buy_condition = (crossover(indicators['ZL_MACD'], indicators['Signal']) & (indicators['ZL_MACD'] < 0))
                    sell_condition = (crossover(indicators['Signal'], indicators['ZL_MACD']) & (indicators['ZL_MACD'] < 0))
                elif strategy_type == "Histogram Trend Reversal":
                    histogram = indicators['Histogram']
                    histogram_diff = histogram.diff()
                    buy_condition = ((histogram.shift(1) < 0) & (histogram_diff.shift(1) < 0) & (histogram_diff > 0))
                    sell_condition = ((histogram.shift(1) > 0) & (histogram_diff.shift(1) > 0) & (histogram_diff < 0))
                
                signals.loc[buy_condition, 'Signal'] = 1
                signals.loc[sell_condition, 'Signal'] = -1
                filtered_signals = signals.loc[start_date:end_date]
                backtest_results = backtest_strategy(filtered_data, filtered_signals, transaction_fee=transaction_fee)
                
                if backtest_results:
                    return {
                        'Fast_Length': fast,
                        'Slow_Length': slow,
                        'Signal_Length': signal,
                        'MACD_EMA_Length': macd_ema,
                        'ROI': backtest_results['roi'],
                        'Max_Profit': backtest_results['max_profit'],
                        'Max_Loss': backtest_results['max_loss']
                    }
    except Exception as e:
        print(f"Error with parameters (fast={fast}, slow={slow}, signal={signal}): {str(e)}")
    return None

def optimize_parameters(data, indicator_type, strategy_type, start_date, end_date, param_range=(1, 30), transaction_fee=0.0):
    if not isinstance(start_date, pd.Timestamp):
        start_date = pd.to_datetime(start_date).tz_localize('America/New_York')
    if not isinstance(end_date, pd.Timestamp):
        end_date = pd.to_datetime(end_date).tz_localize('America/New_York')
    
    filtered_data = data.loc[start_date:end_date]
    
    # Create parameter grid with smarter step sizes
    step_size = max(1, (param_range[1] - param_range[0]) // 10)
    fast_range = range(param_range[0], param_range[1] + 1, step_size)
    slow_range = range(param_range[0], param_range[1] + 1, step_size)
    signal_range = range(param_range[0], param_range[1] + 1, step_size)
    
    # Generate valid parameter combinations
    param_combinations = [
        (data, filtered_data, indicator_type, strategy_type, start_date, end_date, fast, slow, signal, transaction_fee)
        for fast, slow, signal in product(fast_range, slow_range, signal_range)
        if slow > fast  # Ensure slow is always greater than fast
    ]
    
    results = []
    # Use parallel processing to evaluate parameters
    with ProcessPoolExecutor(max_workers=None) as executor:  # None uses all available CPU cores
        for result in executor.map(evaluate_parameters, param_combinations):
            if result is not None:
                results.append(result)
    
    if not results:
        return pd.DataFrame()  # Return empty DataFrame instead of None
    
    # Convert results to DataFrame and sort by ROI
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('ROI', ascending=False)
    
    # Format numeric columns
    for col in ['ROI', 'Max_Profit', 'Max_Loss']:
        results_df[col] = results_df[col].apply(lambda x: f"{x:.2f}")
    
    # Convert parameter columns to integers
    for col in ['Fast_Length', 'Slow_Length', 'Signal_Length']:
        if col in results_df.columns:
            results_df[col] = results_df[col].astype(int)
    
    if 'MACD_EMA_Length' in results_df.columns:
        results_df['MACD_EMA_Length'] = results_df['MACD_EMA_Length'].astype(int)
    
    return results_df.head(30)  # Return top 30 results