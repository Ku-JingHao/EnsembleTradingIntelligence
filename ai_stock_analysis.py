import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
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

def get_extended_stock_data(symbol, start_date, end_date, interval, extension_days=30):
    """
    Fetch stock data with extended start date to accommodate lags
    
    Parameters:
        symbol: Stock symbol
        start_date: User's desired start date
        end_date: User's desired end date
        interval: Data interval (1d, 1h)  # REMOVED 30m
        extension_days: Number of days to extend backwards for lags
    
    Returns:
        Extended DataFrame with data from (start_date - extension_days) to end_date
    """
    try:
        # Calculate extended start date
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        # Extend start date backwards by extension_days
        extended_start_date = start_date - pd.Timedelta(days=extension_days)
        
        print(f"📅 User requested: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print(f"🔄 Extended fetch: {extended_start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} (+{extension_days} days for lags)")
        
        # Fetch extended data using existing get_stock_data logic
        stock = yf.Ticker(symbol)
        
        # Use period calculation similar to get_stock_data
        if interval == "1d":
            period = "max"
        elif interval == "1h":
            period = "2y"
        # REMOVED: elif interval == "30m": period = "1mo"
        
        # Fetch with extended date range
        df = stock.history(period=period, interval=interval, start=extended_start_date, end=end_date + pd.Timedelta(days=1))
        
        if df.empty:
            print(f"❌ No extended data available for {symbol}")
            # Fallback to original date range
            return get_stock_data(symbol, interval)
        
        print(f"✅ Extended data fetched: {len(df)} points (covers lag requirements)")
        return df
        
    except Exception as e:
        print(f"❌ Error fetching extended data: {e}")
        print(f"🔄 Falling back to original get_stock_data")
        return get_stock_data(symbol, interval)

def calculate_lag_requirements(ai_model, fast_length, slow_length, signal_length, interval="1d"):
    """
    Calculate lag requirements with INTERVAL-AWARE extension calculation
    
    Parameters:
        ai_model: ML model name
        fast_length, slow_length, signal_length: MACD parameters
        interval: Data interval ("1d", "1h")  # REMOVED "30m", "15m"
    
    Returns:
        Dictionary with extension info for different intervals
    """
    try:
        # Get ML model lag requirements (in data points) - use default values based on model type
        # We don't need to load the specific stock model for lag calculation
        model_lag_map = {
            'Linear Regression': 7,
            'Random Forest': 7,
            'XGBoost': 7,
            'LSTM': 7,
            'GRU': 7,
            'Ensemble': 7
        }
        
        ai_lag = model_lag_map.get(ai_model, 7)  # Default to 7 if model not found
        
        # Calculate indicator lag requirements (in data points)
        indicator_lag = max(slow_length, signal_length)  # MACD needs slow_length periods
        
        # 🔥 NEW: Calculate base requirement in data points
        base_requirement_points = max(ai_lag, indicator_lag)
        
        # 🔥 NEW: Convert to appropriate time units based on interval
        if interval == "1d":
            # Daily: Each point = 1 day
            safety_buffer_days = 20  # 20 business days buffer
            total_extension_days = base_requirement_points + safety_buffer_days
            extension_info = {
                'points_needed': base_requirement_points,
                'extension_days': total_extension_days,
                'extension_type': 'days'
            }
            
        elif interval == "1h":
            # Hourly: Each point = 1 hour
            # Convert to days: hours / 24, but add buffer for weekends
            hours_needed = base_requirement_points
            safety_buffer_hours = 72  # 2 days buffer in hours
            total_hours = hours_needed + safety_buffer_hours
            
            # Convert to days (with weekend consideration)
            # Business hours: ~8-10 hours per day, so use 10 for safety
            business_hours_per_day = 24  # For 24h markets, use 24
            extension_days = max(int(total_hours / business_hours_per_day) + 1, 3)  # At least 3 days
            
            extension_info = {
                'points_needed': base_requirement_points,
                'extension_days': extension_days,
                'extension_type': 'days_for_hours',
                'hours_needed': hours_needed,
                'total_hours_with_buffer': total_hours
            }
            
        # REMOVED: elif interval == "30m": ... entire 30m block
        # REMOVED: elif interval == "15m": ... entire 15m block
        else:
            # Default fallback
            extension_info = {
                'points_needed': base_requirement_points,
                'extension_days': 30,
                'extension_type': 'default'
            }
        
        # 🔥 DETAILED LOGGING
        print(f"📊 Interval-Aware Lag Requirements ({interval}):")
        print(f"   🤖 ML Model ({ai_model}): {ai_lag} points")
        print(f"   📈 MACD Indicators: {indicator_lag} points")
        print(f"   📊 Base Requirement: {base_requirement_points} {interval} points")
        
        if interval == "1h":
            print(f"   ⏰ Hours needed: {extension_info['hours_needed']} hours")
            print(f"   ⏰ Total with buffer: {extension_info['total_hours_with_buffer']} hours")
            print(f"   📅 Extension days: {extension_info['extension_days']} days")
        # REMOVED: elif interval in ["30m", "15m"]: ... logging block
        else:
            print(f"   📅 Extension days: {extension_info['extension_days']} days")
        
        return extension_info['extension_days']
        
    except Exception as e:
        print(f"⚠️ Error calculating lag requirements: {e}")
        # Return interval-appropriate defaults
        if interval == "1d":
            return 30
        elif interval == "1h":
            return 3  # 3 days for hourly
        # REMOVED: elif interval in ["30m", "15m"]: return 2
        else:
            return 10  # Safe default
    
def get_stock_data(symbol, interval):
    """Fetch stock data from Yahoo Finance with error handling"""
    try:
        stock = yf.Ticker(symbol)

        if interval == "1d":
            period = "max"  
        elif interval == "1h":
            period = "2y"
        # REMOVED: elif interval == "30m": period = "1mo"

        # Fetch stock data with the chosen period and interval
        df = stock.history(period=period, interval=interval)  
        
        if df.empty:
            print(f"No data available for {symbol}")
            return pd.DataFrame()
            
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()

def calculate_macd(data, fast_length=12, slow_length=26, signal_length=9):
    """Calculate regular MACD indicator"""
    if data.empty:
        return pd.DataFrame()
        
    try:
        fast_ema = data['Close'].ewm(span=fast_length, adjust=False).mean()
        slow_ema = data['Close'].ewm(span=slow_length, adjust=False).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal_length, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return pd.DataFrame({
            'MACD': macd_line,
            'Signal': signal_line,
            'Histogram': histogram
        })
    except Exception as e:
        print(f"Error calculating MACD: {e}")
        return pd.DataFrame()

def calculate_zero_lag_macd(data, fast_length=12, slow_length=26, signal_length=9, macd_ema_length=9, use_ema=True, use_old_algo=False):
    """Calculate Zero Lag MACD indicator with options for EMA/SMA and legacy algorithm."""
    if data.empty:
        return pd.DataFrame()
    
    try:
        # Define zero lag calculation for EMA or SMA
        def zero_lag(series, length, use_ema):
            if use_ema:
                ma1 = series.ewm(span=length, adjust=False).mean()
                ma2 = ma1.ewm(span=length, adjust=False).mean()
            else:
                ma1 = series.rolling(window=length).mean()
                ma2 = ma1.rolling(window=length).mean()
            return 2 * ma1 - ma2
        
        # Calculate Zero Lag Fast and Slow MAs
        fast_zlema = zero_lag(data['Close'], fast_length, use_ema)
        slow_zlema = zero_lag(data['Close'], slow_length, use_ema)
        
        # MACD Line
        zl_macd = fast_zlema - slow_zlema
        
        # Signal Line Calculation
        if use_old_algo:
            signal_line = zl_macd.rolling(window=signal_length).mean()
        else:
            ema_sig1 = zl_macd.ewm(span=signal_length, adjust=False).mean()
            ema_sig2 = ema_sig1.ewm(span=signal_length, adjust=False).mean()
            signal_line = 2 * ema_sig1 - ema_sig2
        
        # Histogram
        histogram = zl_macd - signal_line
        
        # EMA on MACD Line (Optional)
        macd_ema = zl_macd.ewm(span=macd_ema_length, adjust=False).mean()
        
        return pd.DataFrame({
            'ZL_MACD': zl_macd,
            'Signal': signal_line,
            'Histogram': histogram,
            'MACD_EMA': macd_ema
        })
    except Exception as e:
        print(f"Error calculating Zero Lag MACD: {e}")
        return pd.DataFrame()

def get_ai_predictions(data, model_name, forecast_steps=None, stock_name=None, interval=None):
    """Get ML predictions for the given data with dynamic model naming"""
    if ModelLoader is None:
        print("ModelLoader not available. Returning dummy predictions.")
        # Return dummy predictions for fallback
        dummy_predictions = pd.DataFrame({
            'Close': data['Close'] * (1 + np.random.normal(0, 0.01, len(data)))
        }, index=data.index)
        return dummy_predictions
    
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
        
        model_key = model_map.get(model_name, 'ensemble')
        
        # Calculate how many predictions we need
        if forecast_steps is None:
            forecast_steps = len(data)
        
        predictions = []
        prediction_dates = []
        
        # Generate predictions for each time step
        for i in range(1, min(len(data), forecast_steps + 1)):
            # Use data up to current point for prediction
            current_data = data.iloc[:i+30] if i+30 < len(data) else data  # Use at least 30 points for context
            
            # Get start and end dates for the prediction
            if len(current_data) > 30:
                start_date = current_data.index[-31].strftime('%Y-%m-%d')
                end_date = current_data.index[-1].strftime('%Y-%m-%d')
                
                # Get ML prediction for next step with dynamic model naming
                result = loader.generate_predictions(
                    model_key,
                    ticker=stock_name,  # Use actual stock name
                    start_date=start_date,
                    end_date=end_date,
                    interval=interval,  # Use actual interval
                    forecast_horizon=1
                )
                
                if result and result['success'] and len(result['predictions']) > 0:
                    predictions.append(result['predictions'][0])
                    prediction_dates.append(data.index[i])
                else:
                    # Fallback: use simple trend
                    if i > 0:
                        trend = data['Close'].iloc[i] - data['Close'].iloc[i-1]
                        predictions.append(data['Close'].iloc[i-1] + trend * 1.1)
                    else:
                        predictions.append(data['Close'].iloc[i])
                    prediction_dates.append(data.index[i])
            else:
                # For early points, use simple trend
                if i > 0:
                    predictions.append(data['Close'].iloc[i])
                else:
                    predictions.append(data['Close'].iloc[0])
                prediction_dates.append(data.index[i])
        
        # Create predictions DataFrame
        predicted_data = pd.DataFrame({
            'Close': predictions
        }, index=prediction_dates)
        
        return predicted_data
        
    except Exception as e:
        print(f"Error getting ML predictions: {e}")
        # Return fallback predictions
        noise = np.random.normal(0, 0.01, len(data))
        dummy_predictions = pd.DataFrame({
            'Close': data['Close'] * (1 + noise)
        }, index=data.index)
        return dummy_predictions

def crossover(series1, series2):
    """Check for crossover between two series"""
    if series1.isnull().any() or series2.isnull().any():
        print("Warning: NaN values detected in series.")
    
    crossover_condition = (series1.shift(1) < series2.shift(1)) & (series1 > series2)
    return crossover_condition

def generate_ai_enhanced_signals(data, indicator_type, strategy_type, ai_model, fast_length=12, slow_length=26, signal_length=9, stock_name=None, interval=None):
    """Generate ML-enhanced buy/sell signals based on selected strategy with dynamic model naming"""
    if data.empty:
        print("Data is empty. Cannot generate signals.")
        return pd.DataFrame(), pd.DataFrame()
        
    try:
        # Calculate indicators
        if indicator_type == "MACD":
            indicators = calculate_macd(data, fast_length=fast_length, slow_length=slow_length, signal_length=signal_length)
            macd_col = 'MACD'
        else:
            indicators = calculate_zero_lag_macd(data, fast_length=fast_length, slow_length=slow_length, signal_length=signal_length)
            macd_col = 'ZL_MACD'
        
        # Get ML predictions with dynamic model naming
        print(f"Getting ML predictions using {ai_model}...")
        predicted_data = get_ai_predictions(data, ai_model, stock_name=stock_name, interval=interval)
        
        signals = pd.DataFrame(index=data.index)
        signals['Signal'] = 0
        
        # Generate traditional signals first
        if strategy_type == "Buy Above Sell Above":
            buy_condition = (crossover(indicators[macd_col], indicators['Signal']) & (indicators[macd_col] > 0))
            sell_condition = (crossover(indicators['Signal'], indicators[macd_col]) & (indicators[macd_col] > 0))
        elif strategy_type == "Buy Below Sell Above":
            buy_condition = (crossover(indicators[macd_col], indicators['Signal']) & (indicators[macd_col] < 0))
            sell_condition = (crossover(indicators['Signal'], indicators[macd_col]) & (indicators[macd_col] > 0))
        elif strategy_type == "Buy Above Sell Below":
            buy_condition = (crossover(indicators[macd_col], indicators['Signal']) & (indicators[macd_col] > 0))
            sell_condition = (crossover(indicators['Signal'], indicators[macd_col]) & (indicators[macd_col] < 0))
        elif strategy_type == "Buy Below Sell Below":
            buy_condition = (crossover(indicators[macd_col], indicators['Signal']) & (indicators[macd_col] < 0))
            sell_condition = (crossover(indicators['Signal'], indicators[macd_col]) & (indicators[macd_col] < 0))
        elif strategy_type == "Histogram Trend Reversal":
            histogram = indicators['Histogram']
            histogram_diff = histogram.diff()
            buy_condition = ((histogram.shift(1) < 0) & (histogram_diff.shift(1) < 0) & (histogram_diff > 0))
            sell_condition = ((histogram.shift(1) > 0) & (histogram_diff.shift(1) > 0) & (histogram_diff < 0))
        
        # Apply traditional signals first
        signals.loc[buy_condition, 'Signal'] = 1
        signals.loc[sell_condition, 'Signal'] = -1
        
        print(f"Generated {len(signals[signals['Signal'] != 0])} traditional signals")
        print(f"ML model {ai_model} will validate these signals during backtesting")
        
        return signals, predicted_data
        
    except Exception as e:
        print(f"Error generating ML-enhanced signals: {e}")
        return pd.DataFrame(), pd.DataFrame()

def backtest_strategy_with_ai_predictions(data, signals, predicted_data, initial_cash=10000, transaction_fee=0.01):
    """
    Backtest strategy using actual prices for signals but confirming trades with ML predicted prices
    
    Parameters:
        data: DataFrame with actual prices
        signals: DataFrame with trading signals
        predicted_data: DataFrame with ML predicted prices (for next-day validation)
        initial_cash: Initial investment amount
        transaction_fee: Transaction fee as percentage
    """
    if data.empty or signals.empty:
        return {
            'final_value': initial_cash,
            'total_trades': 0,
            'total_signals': 0,
            'confirmed_signals': 0,
            'roi': 0,
            'max_profit': 0,
            'max_loss': 0,
            'trade_log': []
        }
    
    cash = initial_cash
    shares = 0
    total_trades = 0
    last_sell_cash = initial_cash
    buy_prices = []
    sell_prices = []
    trade_log = []
    equity_curve = []
    total_signals = 0
    confirmed_signals = 0
    
    # Create a lookup for predicted prices
    predicted_lookup = predicted_data['Close'].to_dict() if not predicted_data.empty else {}
    
    for i in range(len(data) - 1):  # -1 because we need to look ahead
        current_date = data.index[i]
        next_date = data.index[i + 1]
        current_price = data['Close'].iloc[i]
        
        # Calculate current portfolio value
        current_value = cash + (shares * current_price)
        equity_curve.append({
            'date': current_date,
            'value': current_value
        })
        
        # Get ML prediction for next date (if available)
        next_predicted_price = predicted_lookup.get(next_date, None)
        
        # If no ML prediction available, use simple trend extrapolation
        if next_predicted_price is None:
            if i > 0:
                trend = current_price - data['Close'].iloc[i-1]
                next_predicted_price = current_price + trend
            else:
                next_predicted_price = current_price
        
        # Buy signal logic
        if signals['Signal'].iloc[i] == 1:  # Buy signal detected in actual data
            total_signals += 1
            
            # ML Validation: Only execute if predicted next price is higher than current price
            if next_predicted_price > current_price and cash > 0:
                confirmed_signals += 1
                max_investment = cash / (1 + transaction_fee)
                shares_to_buy = int(max_investment // current_price)
                
                if shares_to_buy > 0:
                    purchase_cost = shares_to_buy * current_price
                    fee_cost = purchase_cost * transaction_fee
                    total_cost = purchase_cost + fee_cost
                    
                    if total_cost <= cash:
                        cash -= total_cost
                        shares += shares_to_buy
                        buy_prices.append(current_price)
                        total_trades += 1
                        trade_log.append({
                            'type': 'BUY',
                            'date': str(current_date),
                            'price': current_price,
                            'predicted_next': next_predicted_price,
                            'shares': shares_to_buy,
                            'fee': fee_cost,
                            'cash_left': cash,
                            'validation': 'CONFIRMED'
                        })
            else:
                # Log the rejected buy signal for analysis
                trade_log.append({
                    'type': 'BUY',
                    'date': str(current_date),
                    'price': current_price,
                    'predicted_next': next_predicted_price,
                    'shares': 0,
                    'fee': 0,
                    'cash_left': cash,
                    'validation': 'REJECTED'
                })
                
        # Sell signal logic
        elif signals['Signal'].iloc[i] == -1:  # Sell signal detected in actual data
            total_signals += 1
            
            # ML Validation: Only execute if predicted next price is lower than current price
            if next_predicted_price < current_price and shares > 0:
                confirmed_signals += 1
                sell_value = shares * current_price
                fee_cost = sell_value * transaction_fee
                net_sell_value = sell_value - fee_cost
                
                cash += net_sell_value
                last_sell_cash = cash
                sell_prices.append(current_price)
                trade_log.append({
                    'type': 'SELL',
                    'date': str(current_date),
                    'price': current_price,
                    'predicted_next': next_predicted_price,
                    'shares': shares,
                    'fee': fee_cost,
                    'cash_left': cash,
                    'validation': 'CONFIRMED'
                })
                shares = 0
                total_trades += 1
            else:
                # Log the rejected sell signal for analysis
                trade_log.append({
                    'type': 'SELL',
                    'date': str(current_date),
                    'price': current_price, 
                    'predicted_next': next_predicted_price,
                    'shares': 0,
                    'fee': 0,
                    'cash_left': cash,
                    'validation': 'REJECTED'
                })
    
    # Add the final point to the equity curve
    if len(data) > 0:
        final_value = cash + (shares * data['Close'].iloc[-1])
        equity_curve.append({
            'date': data.index[-1],
            'value': final_value
        })
    
    # Calculate returns and metrics
    if shares > 0:
        final_value = last_sell_cash
    else:
        final_value = cash + (shares * data['Close'].iloc[-1] if len(data) > 0 else 0)
    
    roi = ((final_value - initial_cash) / initial_cash) * 100
    max_profit = max(sell_prices) - min(buy_prices) if buy_prices and sell_prices else 0
    max_loss = min(sell_prices) - max(buy_prices) if buy_prices and sell_prices else 0
    
    return {
        'final_value': final_value,
        'total_trades': total_trades,
        'total_signals': total_signals,
        'confirmed_signals': confirmed_signals,
        'roi': roi,
        'max_profit': max_profit,
        'max_loss': max_loss,
        'trade_log': trade_log,
        'equity_curve': pd.DataFrame(equity_curve).set_index('date') if equity_curve else pd.DataFrame()
    }

def create_ai_enhanced_chart(data, indicator_select, strategy_select, signals, macd_data, predicted_data, ai_model):
    """Create an ML-enhanced combined chart of stock price and technical indicators."""
    
    # SIMPLIFIED: Use data exactly as provided from optimization - NO timezone conversion
    # This ensures 100% consistency with optimization results
    
    # Create subplots with shared x-axis
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03)
    
    # Add candlestick chart using exact data
    candlestick_trace = go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Actual Price'
    )
    fig.add_trace(candlestick_trace, row=1, col=1)
    
    # Add ML predictions line using exact predicted_data
    if predicted_data is not None and not predicted_data.empty:
        fig.add_trace(go.Scatter(
            x=predicted_data.index,
            y=predicted_data['Close'],
            mode='lines',
            name=f'🤖 {ai_model} Predictions',
            line=dict(color='purple', width=2, dash='dash')
        ), row=1, col=1)

    # Add buy/sell signals using exact signals data
    if signals is not None and not signals.empty:
        buy_signals = data[signals['Signal'] == 1]
        sell_signals = data[signals['Signal'] == -1]
        
        # Add annotations for ML-enhanced signals
        for i in range(len(buy_signals)):
            fig.add_annotation(
                x=buy_signals.index[i],
                y=buy_signals['Close'].iloc[i],
                text="🤖 ML Buy",
                showarrow=True,
                arrowhead=2,
                ax=0,
                ay=-40,
                bgcolor='green',
                font=dict(color='white')
            )
   
        for i in range(len(sell_signals)):
            fig.add_annotation(
                x=sell_signals.index[i],
                y=sell_signals['Close'].iloc[i],
                text="🤖 ML Sell",
                showarrow=True,
                arrowhead=2,
                ax=0,
                ay=-40,
                bgcolor='red',
                font=dict(color='white')
            )
            
        # Plot buy signals
        if not buy_signals.empty:
            fig.add_trace(go.Scatter(
                x=buy_signals.index,
                y=buy_signals['Close'],
                mode='markers',
                marker=dict(color='green', size=10, symbol='triangle-up'),
                name='🤖 ML Buy Signal',
                visible='legendonly' 
            ))
        
        if not sell_signals.empty:
            fig.add_trace(go.Scatter(
                x=sell_signals.index,
                y=sell_signals['Close'],
                mode='markers',
                marker=dict(color='red', size=10, symbol='triangle-down'),
                name='🤖 ML Sell Signal',
                visible='legendonly' 
            ))
    
    # Add MACD indicators using exact macd_data
    if macd_data is not None and not macd_data.empty:
        if indicator_select == "MACD":
            macd_fig = plot_macd(data, macd_data)
            for trace in macd_fig.data:
                fig.add_trace(trace, row=2, col=1)
        else:
            zl_macd_fig = plot_zero_lag_macd(data, macd_data)
            for trace in zl_macd_fig.data:
                fig.add_trace(trace, row=2, col=1)

    # Update layout for ML enhancement
    fig.update_layout(
        hovermode="x unified",
        title=f'🤖 ML-Enhanced Stock Analysis with {ai_model}',
        yaxis_title='Price',
        height=800,
        template='plotly_white',
        xaxis=dict(
            rangeslider=dict(visible=False),
            rangebreaks=[
                dict(bounds=["sat", "mon"]),  # hide weekends
            ]
        ),
        xaxis2=dict(
            rangebreaks=[
                dict(bounds=["sat", "mon"]),  # hide weekends
            ]
        )
    )

    return fig

def plot_macd(data, macd_data):
    """Plot MACD indicator"""
    if data.empty:
        return go.Figure()
        
    try:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           vertical_spacing=0.03, row_heights=[0.13, 0.7])
        
        # Add MACD line
        fig.add_trace(go.Scatter(
            x=data.index, 
            y=macd_data['MACD'],
            name='MACD',
            line=dict(color='blue')
        ), row=2, col=1)
        
        # Add Signal line
        fig.add_trace(go.Scatter(
            x=data.index,
            y=macd_data['Signal'],
            name='Signal',
            line=dict(color='orange')
        ), row=2, col=1)
        
        # Add histogram
        colors = ['red' if val < 0 else 'green' for val in macd_data['Histogram']]
        fig.add_trace(go.Bar(
            x=data.index,
            y=macd_data['Histogram'],
            name='Histogram',
            marker_color=colors
        ), row=2, col=1)
        
        fig.update_layout(
            title='MACD Indicator',
            yaxis2_title='MACD',
            showlegend=True,
            template='plotly_white',
            height=400  
        )
        
        return fig
    except Exception as e:
        print(f"Error creating MACD plot: {e}")
        return go.Figure()

def plot_zero_lag_macd(data, zl_macd_data):
    """Plot Zero Lag MACD indicator"""
    if data.empty:
        return go.Figure()
        
    try:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           vertical_spacing=0.03, row_heights=[0.13, 0.7])
        
        # Add Zero Lag MACD line
        fig.add_trace(go.Scatter(
            x=data.index,
            y=zl_macd_data['ZL_MACD'],
            name='Zero Lag MACD',
            line=dict(color='blue')
        ), row=2, col=1)

        # Add EMA on MACD line
        fig.add_trace(go.Scatter(
            x=data.index,
            y=zl_macd_data['MACD_EMA'],
            name='EMA on MACD',
            line=dict(color='red')
        ), row=2, col=1)
        
        # Add Signal line
        fig.add_trace(go.Scatter(
            x=data.index,
            y=zl_macd_data['Signal'],
            name='Signal',
            line=dict(color='orange')
        ), row=2, col=1)
        
        # Add histogram
        colors = ['red' if val < 0 else 'green' for val in zl_macd_data['Histogram']]
        fig.add_trace(go.Bar(
            x=data.index,
            y=zl_macd_data['Histogram'],
            name='Histogram',
            marker_color=colors
        ), row=2, col=1)
        
        fig.update_layout(
            title='Zero Lag MACD Indicator',
            yaxis2_title='Zero Lag MACD',
            showlegend=True,
            template='plotly_white',
            height=400  # Set a fixed height
        )
        
        return fig
    except Exception as e:
        print(f"Error creating Zero Lag MACD plot: {e}")
        return go.Figure()

# Legacy function for backward compatibility
def backtest_strategy(data, signals, initial_cash=10000, transaction_fee=0.01):
    """Legacy backtest function - redirects to ML-enhanced version with dummy predictions"""
    dummy_predictions = pd.DataFrame({'Close': data['Close']}, index=data.index)
    return backtest_strategy_with_ai_predictions(data, signals, dummy_predictions, initial_cash, transaction_fee)

# Legacy function for backward compatibility  
def generate_signals(data, indicator_type, strategy_type, fast_length=12, slow_length=26, signal_length=9):
    """Legacy signal generation - redirects to AMLI-enhanced version with dummy model"""
    signals, _ = generate_ai_enhanced_signals(data, indicator_type, strategy_type, 'Ensemble', fast_length, slow_length, signal_length)
    return signals