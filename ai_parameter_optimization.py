import pandas as pd
import numpy as np
import warnings
from ai_stock_analysis import calculate_macd, calculate_zero_lag_macd, generate_ai_enhanced_signals, backtest_strategy_with_ai_predictions, crossover
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
from itertools import product
import os
import sys

# Add the current directory to path for model imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from model_loader import ModelLoader
except ImportError:
    print("Warning: ModelLoader not found. ML features may not work.")
    ModelLoader = None

warnings.filterwarnings('ignore')

def get_fallback_predictions(data):
    """Generate fallback trend-based predictions when ML models fail"""
    predictions = []
    for i in range(1, len(data)):
        if i > 1:
            trend = data['Close'].iloc[i-1] - data['Close'].iloc[i-2]
            pred = data['Close'].iloc[i-1] + trend * 1.1
        else:
            pred = data['Close'].iloc[i]
        predictions.append(pred)
        
    return pd.DataFrame({
        'Close': predictions
    }, index=data.index[1:])

def optimize_ai_enhanced_parameters(data, indicator_type, strategy_type, ai_model, start_date, end_date, param_range=(1, 30), transaction_fee=0.0, interval="1d", stock_name=None):
    """
    🔥 FIXED: Optimize parameters with proper lag accommodation and dynamic model naming
    """
    print(f"🤖 Starting ML-Enhanced Parameter Optimization with {ai_model}")
    print(f"📊 Indicator: {indicator_type}")
    print(f"📈 Strategy: {strategy_type}")
    print(f"⏰ Interval: {interval}")
    print(f" Stock: {stock_name}")
    
    # Handle timezone conversion
    if not isinstance(start_date, pd.Timestamp):
        start_date = pd.to_datetime(start_date).tz_localize('America/New_York')
    elif start_date.tz is None:
        start_date = start_date.tz_localize('America/New_York')
    
    if not isinstance(end_date, pd.Timestamp):
        end_date = pd.to_datetime(end_date).tz_localize('America/New_York')
    elif end_date.tz is None:
        end_date = end_date.tz_localize('America/New_York')
    
    # Make sure data index is timezone-aware and consistent
    if data.index.tz is None:
        data.index = data.index.tz_localize('America/New_York')
    elif data.index.tz != start_date.tz:
        data.index = data.index.tz_convert('America/New_York')
        start_date = start_date.tz_convert('America/New_York')
        end_date = end_date.tz_convert('America/New_York')
    
    print(f"📅 User's desired range: {start_date.strftime('%Y-%m-%d %H:%M:%S %Z')} to {end_date.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"📊 Current dataset points: {len(data)}")
    print(f"📅 Dataset range: {data.index[0].strftime('%Y-%m-%d %H:%M:%S %Z')} to {data.index[-1].strftime('%Y-%m-%d %H:%M:%S %Z')}")
    
    # 🔥 STEP 1: Calculate maximum lag requirements for ALL parameter combinations
    max_slow = param_range[1]  # Maximum slow length to be tested
    max_signal = param_range[1]  # Maximum signal length to be tested
    
    # Calculate lag requirements using worst-case scenario (maximum parameters)
    from ai_stock_analysis import calculate_lag_requirements
    total_extension_needed = calculate_lag_requirements(
        ai_model, 
        param_range[0], 
        max_slow, 
        max_signal,
        interval=interval
    )
    
    # 🔥 STEP 2: IMPROVED date calculation with proper timezone handling
    data_start = data.index[0]
    user_start = start_date
    
    # 🔥 FIX: Ensure both dates are at the same precision (date only for day calculation)
    data_start_date = data_start.normalize()  # Strip time, keep date only
    user_start_date = user_start.normalize()  # Strip time, keep date only
    
    days_before_user_start = (user_start_date - data_start_date).days
    
    print(f"🔍 DETAILED DATE ANALYSIS:")
    print(f"   Data starts: {data_start} → normalized: {data_start_date}")
    print(f"   User starts: {user_start} → normalized: {user_start_date}")
    print(f"   📏 Calendar days difference: {days_before_user_start}")
    print(f"   🎯 Extension needed: {total_extension_needed} days")
    
    if days_before_user_start < total_extension_needed:
        print(f"⚠️ Insufficient historical data. Have {days_before_user_start} days, need {total_extension_needed} days")
        
        # 🔥 IMPROVED: More detailed analysis
        shortage = total_extension_needed - days_before_user_start
        print(f"📊 Shortage: {shortage} day(s)")
        
        # CHECK: If shortage is small (1-2 days), it might be acceptable
        if shortage <= 2:
            print(f"✅ Small shortage ({shortage} days) - continuing with available data")
            print(f"🔧 ML models will adapt to available historical context")
        else:
            print(f"❌ Significant shortage ({shortage} days) - this may affect model performance")
            print(f"💡 Consider increasing extension_days in get_extended_stock_data()")
        
        print(f"🔄 Fetching extended data to accommodate lags...")
        print(f"ℹ️ Will handle data extension in individual parameter evaluations")
    else:
        print(f"✅ Sufficient historical data available ({days_before_user_start} >= {total_extension_needed} days)")

    # Create parameter grid
    step_size = max(1, (param_range[1] - param_range[0]) // 15)
    fast_range = range(param_range[0], param_range[1] + 1, step_size)
    slow_range = range(param_range[0], param_range[1] + 1, step_size)
    signal_range = range(param_range[0], param_range[1] + 1, step_size)
    
    valid_combinations = [
        (fast, slow, signal, ai_model) 
        for fast, slow, signal in product(fast_range, slow_range, signal_range)
        if fast < slow and signal > 0
    ]
    
    if not valid_combinations:
        print("❌ No valid parameter combinations found.")
        return pd.DataFrame(), None
    
    print(f" Testing {len(valid_combinations)} parameter combinations...")
    
    # 🔥 STEP 3: Pass FULL data and extension info to evaluation function with stock_name and interval
    args_list = [
        (data, start_date, end_date, indicator_type, strategy_type, ai_model, fast, slow, signal, transaction_fee, total_extension_needed, stock_name, interval)
        for fast, slow, signal, ai_model in valid_combinations
    ]
    
    max_workers = min(4, len(valid_combinations))
    
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(evaluate_ai_enhanced_parameters_with_extension, args_list))
        
        # Process results...
        valid_results = []
        chart_data_dict = {}
        
        for i, result in enumerate(results):
            if result is not None:
                clean_result, chart_data = result
                valid_results.append(clean_result)
                
                result_key = f"{clean_result['fast']}-{clean_result['slow']}-{clean_result['signal']}-{clean_result['roi']:.6f}"
                chart_data_dict[result_key] = chart_data
        
        if not valid_results:
            print("❌ No valid optimization results obtained.")
            return pd.DataFrame(), None
        
        results_df = pd.DataFrame(valid_results)
        results_df = results_df.sort_values('roi', ascending=False)
        
        best_result = results_df.iloc[0]
        best_result_key = f"{best_result['fast']}-{best_result['slow']}-{best_result['signal']}-{best_result['roi']:.6f}"
        best_chart_data = chart_data_dict.get(best_result_key)
        
        print(f"✅ ML-Enhanced optimization complete!")
        print(f" Best ROI: {results_df['roi'].iloc[0]:.6f}%")
        print(f"🤖 Best Confirmation Rate: {results_df['confirmation_rate'].iloc[0]:.1f}%")
        
        return results_df, best_chart_data
        
    except Exception as e:
        print(f"❌ Error during ML-enhanced optimization: {e}")
        return pd.DataFrame(), None
    
    
def analyze_ai_model_performance(data, indicator_type, strategy_type, start_date, end_date, best_params, transaction_fee=0.0, stock_name=None, interval=None):
    """
    Analyze performance across different ML models using the best parameters
    """
    ai_models = ['Linear Regression', 'Random Forest', 'XGBoost', 'LSTM', 'GRU', 'Ensemble']
    
    print(f"🤖 Analyzing ML Model Performance Comparison")
    print(f"📊 Using best parameters: Fast={best_params['fast']}, Slow={best_params['slow']}, Signal={best_params['signal']}")
    
    model_results = []
    
    for ai_model in ai_models:
        print(f"\n🔄 Testing {ai_model}...")
        
        try:
            # Generate ML-enhanced signals
            signals, predicted_data = generate_ai_enhanced_signals(
                data.loc[start_date:end_date], 
                indicator_type, 
                strategy_type, 
                ai_model,
                fast_length=best_params['fast'],
                slow_length=best_params['slow'], 
                signal_length=best_params['signal'],
                stock_name=stock_name,
                interval=interval
            )
            
            # Backtest with ML validation
            result = backtest_strategy_with_ai_predictions(
                data.loc[start_date:end_date],
                signals,
                predicted_data,
                transaction_fee=transaction_fee
            )
            
            # Calculate ML-specific metrics - REMOVED ai_effectiveness and ai_score
            confirmation_rate = (result['confirmed_signals'] / max(result['total_signals'], 1)) * 100
            
            model_results.append({
                'ai_model': ai_model,
                'roi': result['roi'],
                'total_trades': result['total_trades'],
                'total_signals': result['total_signals'],
                'confirmed_signals': result['confirmed_signals'],
                'confirmation_rate': confirmation_rate,
                'final_value': result['final_value']
                # REMOVED: 'ai_effectiveness' and 'ai_score'
            })
            
            print(f"✅ {ai_model}: ROI={result['roi']:.2f}%, Confirmation={confirmation_rate:.1f}%")
            
        except Exception as e:
            print(f"❌ Error testing {ai_model}: {e}")
            model_results.append({
                'ai_model': ai_model,
                'roi': -100,
                'total_trades': 0,
                'total_signals': 0,
                'confirmed_signals': 0,
                'confirmation_rate': 0,
                'final_value': 10000
            })
    
    comparison_df = pd.DataFrame(model_results)
    
    # CHANGED: Sort by ROI only (removed ai_score)
    comparison_df = comparison_df.sort_values('roi', ascending=False)
    
    print(f"\n🏆 ML Model Performance Ranking:")
    for i, row in comparison_df.head(3).iterrows():
        print(f"{i+1}. {row['ai_model']}: ROI={row['roi']:.2f}%, Confirmation={row['confirmation_rate']:.1f}%")
    
    return comparison_df

def get_ai_enhanced_recommendations(optimization_results, model_comparison_results):
    """
    Generate ML-enhanced trading recommendations based on optimization results
    """
    if optimization_results.empty:
        return {
            'status': 'error',
            'message': 'No optimization results available'
        }
    
    best_result = optimization_results.iloc[0]
    
    # Find best ML model from comparison
    if not model_comparison_results.empty:
        best_ai_model = model_comparison_results.iloc[0]['ai_model']
        best_roi = model_comparison_results.iloc[0]['roi']
    else:
        best_ai_model = best_result['ai_model']
        best_roi = best_result['roi']
    
    recommendations = {
        'status': 'success',
        'best_parameters': {
            'fast_length': int(best_result['fast']),
            'slow_length': int(best_result['slow']),
            'signal_length': int(best_result['signal'])
        },
        'best_ai_model': best_ai_model,
        'performance_metrics': {
            'expected_roi': f"{best_result['roi']:.2f}%",
            'confirmation_rate': f"{best_result['confirmation_rate']:.1f}%",
            'total_signals': int(best_result['total_signals']),
            'confirmed_signals': int(best_result['confirmed_signals'])
        },
        'ai_insights': {
            'signal_quality': 'High' if best_result['confirmation_rate'] > 70 else 'Medium' if best_result['confirmation_rate'] > 50 else 'Low',
            'recommendation': 'RECOMMENDED' if best_roi > 5 and best_result['confirmation_rate'] > 60 else 'CAUTION'
        }
    }
    
    return recommendations

def format_ai_optimization_summary(optimization_results, model_comparison_results):
    """
    Format a comprehensive summary of ML-enhanced optimization results
    """
    if optimization_results.empty:
        return "❌ No optimization results available."
    
    summary = []
    summary.append("🤖 ML-ENHANCED PARAMETER OPTIMIZATION SUMMARY")
    summary.append("=" * 60)
    
    # Best overall result
    best = optimization_results.iloc[0]
    summary.append(f"\n🏆 OPTIMAL CONFIGURATION:")
    summary.append(f"   📊 Parameters: Fast={best['fast']}, Slow={best['slow']}, Signal={best['signal']}")
    summary.append(f"   🤖 ML Model: {best['ai_model']}")
    summary.append(f"   📈 Expected ROI: {best['roi']:.2f}%")
    summary.append(f"   ✅ Confirmation Rate: {best['confirmation_rate']:.1f}%")
    # REMOVED: ML Score line
    
    # Model comparison
    if not model_comparison_results.empty:
        summary.append(f"\n🤖 ML MODEL PERFORMANCE RANKING:")
        for i, row in model_comparison_results.head(3).iterrows():
            medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
            summary.append(f"   {medal} {row['ai_model']}: ROI={row['roi']:.2f}%, Confirmation={row['confirmation_rate']:.1f}%")
            # REMOVED: ML Score from display
    
    # Top parameter combinations
    summary.append(f"\n📊 TOP 5 PARAMETER COMBINATIONS:")
    for i, row in optimization_results.head(5).iterrows():
        summary.append(f"   {i+1}. [{row['fast']}-{row['slow']}-{row['signal']}] {row['ai_model']}: ROI={row['roi']:.2f}%")
        # REMOVED: ML Score from display
    
    # ML insights
    summary.append(f"\n🔍 ML INSIGHTS:")
    avg_confirmation = optimization_results['confirmation_rate'].mean()
    avg_roi = optimization_results['roi'].mean()
    
    summary.append(f"   📈 Average ROI: {avg_roi:.2f}%")
    summary.append(f"   🤖 Average Confirmation Rate: {avg_confirmation:.1f}%")
    summary.append(f"   📊 Total Combinations Tested: {len(optimization_results)}")
    
    signal_quality = "High" if avg_confirmation > 70 else "Medium" if avg_confirmation > 50 else "Low"
    summary.append(f"   🎯 Signal Quality: {signal_quality}")
    
    return "\n".join(summary)

def evaluate_ai_enhanced_parameters_with_extension(args):
    """
    🔥 NEW: Evaluate parameters with proper lag extension handling and dynamic model naming
    """
    full_data, start_date, end_date, indicator_type, strategy_type, ai_model, fast, slow, signal, transaction_fee, total_extension_needed, stock_name, interval = args
    
    try:
        print(f"Testing parameters: Fast={fast}, Slow={slow}, Signal={signal} with {ai_model}")
        print(f"Full dataset: {len(full_data)} points")
        
        # 🔥 STEP 1: Ensure we have enough data BEFORE user's start date
        data_start = full_data.index[0]
        days_before_user_start = (start_date - data_start).days
        
        if days_before_user_start < total_extension_needed:
            print(f"⚠️ Insufficient historical data for proper lags ({days_before_user_start} < {total_extension_needed})")
            print(f"🔄 Using available data with adjusted expectations")
        
        # 🔥 STEP 2: Calculate indicators on FULL dataset (with proper lag accommodation)
        if indicator_type == "MACD":
            full_indicators = calculate_macd(full_data, fast_length=fast, slow_length=slow, signal_length=signal)
            macd_col = 'MACD'
        else:
            full_indicators = calculate_zero_lag_macd(full_data, fast_length=fast, slow_length=slow, signal_length=signal)
            macd_col = 'ZL_MACD'
        
        if full_indicators.empty:
            print(f"❌ Failed to calculate indicators for {fast}-{slow}-{signal}")
            return None
        
        # 🔥 STEP 3: Generate signals on FULL dataset
        full_signals = pd.DataFrame(index=full_data.index)
        full_signals['Signal'] = 0
        
        # Apply strategy conditions on full data
        if strategy_type == "Buy Above Sell Above":
            buy_condition = (crossover(full_indicators[macd_col], full_indicators['Signal']) & (full_indicators[macd_col] > 0))
            sell_condition = (crossover(full_indicators['Signal'], full_indicators[macd_col]) & (full_indicators[macd_col] > 0))
        elif strategy_type == "Buy Below Sell Above":
            buy_condition = (crossover(full_indicators[macd_col], full_indicators['Signal']) & (full_indicators[macd_col] < 0))
            sell_condition = (crossover(full_indicators['Signal'], full_indicators[macd_col]) & (full_indicators[macd_col] > 0))
        elif strategy_type == "Buy Above Sell Below":
            buy_condition = (crossover(full_indicators[macd_col], full_indicators['Signal']) & (full_indicators[macd_col] > 0))
            sell_condition = (crossover(full_indicators['Signal'], full_indicators[macd_col]) & (full_indicators[macd_col] < 0))
        elif strategy_type == "Buy Below Sell Below":
            buy_condition = (crossover(full_indicators[macd_col], full_indicators['Signal']) & (full_indicators[macd_col] < 0))
            sell_condition = (crossover(full_indicators['Signal'], full_indicators[macd_col]) & (full_indicators[macd_col] < 0))
        elif strategy_type == "Histogram Trend Reversal":
            histogram = full_indicators['Histogram']
            histogram_diff = histogram.diff()
            buy_condition = ((histogram.shift(1) < 0) & (histogram_diff.shift(1) < 0) & (histogram_diff > 0))
            sell_condition = ((histogram.shift(1) > 0) & (histogram_diff.shift(1) > 0) & (histogram_diff < 0))
        
        full_signals.loc[buy_condition, 'Signal'] = 1
        full_signals.loc[sell_condition, 'Signal'] = -1
        
        # STEP 4: NOW filter to user's ACTUAL desired date range
        try:
            filtered_data = full_data.loc[start_date:end_date]
            filtered_signals = full_signals.loc[start_date:end_date]
            filtered_indicators = full_indicators.loc[start_date:end_date]
        except Exception as e:
            print(f"Date filtering error: {e}")
            # Fallback timezone handling
            full_data.index = full_data.index.tz_localize(None) if full_data.index.tz is not None else full_data.index
            start_date_naive = start_date.tz_localize(None) if start_date.tz is not None else start_date
            end_date_naive = end_date.tz_localize(None) if end_date.tz is not None else end_date
            filtered_data = full_data.loc[start_date_naive:end_date_naive]
            filtered_signals = full_signals.loc[start_date_naive:end_date_naive]
            filtered_indicators = full_indicators.loc[start_date_naive:end_date_naive]
        
        print(f" User's period data: {len(filtered_data)} points")
        print(f"📈 Signals in user's period: {len(filtered_signals[filtered_signals['Signal'] != 0])}")
        
        # 🔥 STEP 5: Get ML predictions for user's period WITH full historical context
        predicted_data = get_ai_predictions_for_optimization_with_context(
            full_data,        # Full dataset with historical context
            filtered_data,    # User's desired period
            ai_model,
            start_date,
            end_date,
            stock_name,       # Pass stock_name for dynamic model loading
            interval          # Pass interval for dynamic model loading
        )
        
        # 🔥 STEP 6: Backtest on user's filtered period (with proper historical context)
        result = backtest_strategy_with_ai_predictions(
            filtered_data, 
            filtered_signals, 
            predicted_data, 
            transaction_fee=transaction_fee
        )
        
        print(f"Backtest result: ROI={result['roi']:.6f}%, Trades={result['total_trades']}")
        
        # Calculate metrics
        confirmation_rate = (result['confirmed_signals'] / max(result['total_signals'], 1)) * 100
        
        clean_result = {
            'fast': fast,
            'slow': slow, 
            'signal': signal,
            'roi': result['roi'],
            'total_trades': result['total_trades'],
            'total_signals': result['total_signals'],
            'confirmed_signals': result['confirmed_signals'],
            'confirmation_rate': confirmation_rate,
            'max_profit': result['max_profit'],
            'max_loss': result['max_loss'],
            'final_value': result['final_value'],
            'ai_model': ai_model
        }
        
        # Chart data contains ONLY user's desired period (but calculated with full context)
        chart_data = {
            'filtered_data': filtered_data.copy(),
            'signals': filtered_signals.copy(),
            'indicators': filtered_indicators.copy(),
            'predicted_data': predicted_data.copy(),
            'backtest_results': result.copy()
        }
        
        return (clean_result, chart_data)
        
    except Exception as e:
        print(f"Error evaluating parameters {fast}-{slow}-{signal}: {e}")
        return None

def get_ai_predictions_for_optimization_with_context(full_data, filtered_data, ai_model, start_date, end_date, stock_name=None, interval=None):
    """
    🔥 FIXED: Generate ML predictions using ModelLoader's proven methods with dynamic naming
    """
    if ModelLoader is None:
        print("ModelLoader not available. Using trend-based predictions.")
        return get_fallback_predictions(filtered_data)
    
    try:
        loader = ModelLoader()
        
        # Convert model name to match deployed model names
        model_map = {
            'Linear Regression': 'linear_regression',
            'Random Forest': 'random_forest', 
            'XGBoost': 'xgboost',
            'LSTM': 'lstm',
            'GRU': 'gru',
            'Ensemble': 'ensemble'
        }
        
        model_key = model_map.get(ai_model, 'ensemble')
        
        # Get model requirements with dynamic naming
        model_info = loader.load_model(model_key, stock_name=stock_name, interval=interval)
        if model_info is None:
            print(f"Could not load model {model_key} for {stock_name}_{interval}, using fallback predictions")
            return get_fallback_predictions(filtered_data)
        
        config = model_info['config']
        
        # Get model's input requirements
        if config['type'] == 'deep_learning':
            input_sequence_length = config['params'].get('input_chunk_length', 7)
        elif config['type'] == 'traditional_ml':
            input_sequence_length = config['params'].get('lags', 7)
        else:
            input_sequence_length = 7
        
        print(f"🤖 {ai_model} needs {input_sequence_length} historical points per prediction")
        
        predictions = []
        prediction_dates = []
        
        # 🔥 GENERATE PREDICTIONS: Use proven ModelLoader methods
        for i, target_date in enumerate(filtered_data.index):
            try:
                # Find the target date in the full dataset
                try:
                    full_data_idx = full_data.index.get_loc(target_date)
                except KeyError:
                    # Target date not found in full data, skip
                    print(f"⚠️ Date {target_date} not found in full dataset")
                    if i > 0:
                        predictions.append(predictions[-1])  # Use last prediction
                    else:
                        predictions.append(filtered_data['Close'].iloc[i])
                    prediction_dates.append(target_date)
                    continue
                
                # Get historical data BEFORE target date from full dataset
                if full_data_idx >= input_sequence_length:
                    historical_start_idx = full_data_idx - input_sequence_length
                    historical_end_idx = full_data_idx
                    historical_data = full_data.iloc[historical_start_idx:historical_end_idx]
                    
                    # 🔥 USE PROVEN METHOD: Call ModelLoader's data preparation and prediction
                    pred = call_model_using_loader_methods(loader, model_info, historical_data, interval, stock_name)
                    
                    if pred is not None:
                        predictions.append(pred)
                    else:
                        # Fallback to trend for individual failures
                        if len(historical_data) >= 2:
                            trend = historical_data['Close'].iloc[-1] - historical_data['Close'].iloc[-2]
                            predictions.append(historical_data['Close'].iloc[-1] + trend * 1.1)
                        else:
                            predictions.append(filtered_data['Close'].iloc[i])
                else:
                    # Not enough historical data for this prediction point
                    if i > 0:
                        trend = filtered_data['Close'].iloc[i] - filtered_data['Close'].iloc[i-1]
                        predictions.append(filtered_data['Close'].iloc[i-1] + trend * 1.1)
                    else:
                        predictions.append(filtered_data['Close'].iloc[i])
                
                prediction_dates.append(target_date)
                
            except Exception as e:
                print(f"Error predicting for {target_date}: {e}")
                # Individual failure fallback
                if i > 0:
                    predictions.append(predictions[-1])  # Use last prediction
                else:
                    predictions.append(filtered_data['Close'].iloc[i])
                prediction_dates.append(target_date)
        
        print(f"✅ Generated {len(predictions)} ML predictions using proven ModelLoader methods")
        
        return pd.DataFrame({
            'Close': predictions
        }, index=prediction_dates)
        
    except Exception as e:
        print(f"❌ Error in context-aware ML predictions: {e}")
        return get_fallback_predictions(filtered_data)

def call_model_using_loader_methods(loader, model_info, historical_data, interval="1d", stock_name=None):
    """
    🔥 NEW: Use ModelLoader's proven generate_predictions method with historical data
    """
    try:
        # Create a temporary file-like structure to simulate the loader's normal flow
        # This bypasses the yfinance fetch and uses our historical data directly
        
        model = model_info['model']
        config = model_info['config']
        
        if config['type'] == 'deep_learning':
            # Use ModelLoader's proven method
            target_ts, scaler = loader.prepare_deep_learning_data(historical_data, interval=interval)
            if target_ts is None:
                return None
            
            input_chunk_length = config['params'].get('input_chunk_length', 7)
            if len(target_ts) < input_chunk_length:
                return None
            
            # Generate predictions using only target series (no past_covariates)
            predictions = model.predict(n=1, series=target_ts)
            
            # Inverse transform prediction
            pred_values = predictions.values().reshape(-1, 1)
            inverse_pred = scaler.inverse_transform(pred_values)
            return inverse_pred.flatten()[0]
            
        elif config['type'] == 'traditional_ml':
            # Use ModelLoader's proven method
            model_name = config['name'].lower().replace(' ', '_')
            target_ts, covariate_ts, scaler = loader.prepare_traditional_ml_data(
                historical_data, 
                model_name, 
                interval=interval,
                stock_name=stock_name
            )
            if target_ts is None:
                return None
            
            lags = config['params'].get('lags', 7)
            if len(target_ts) < lags:
                return None
            
            predictions = model.predict(n=1, series=target_ts, past_covariates=covariate_ts)
            
            # Inverse transform prediction
            pred_values = predictions.values().reshape(-1, 1)
            inverse_pred = scaler.inverse_transform(pred_values)
            return inverse_pred.flatten()[0]
            
        else:
            return None
                
    except Exception as e:
        print(f"❌ Error using loader methods: {e}")
        return None
