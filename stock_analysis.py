import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def get_stock_data(symbol, interval):
    """Fetch stock data from Yahoo Finance with error handling"""
    try:
        stock = yf.Ticker(symbol)

        if interval == "1d":
            period = "max"  
        elif interval == "1h":
            period = "2y"
        elif interval == "30m":
            period = "1mo"  

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

def calculate_zero_lag_macd(data, fast_length=12, slow_length=26, signal_length=9, macd_ema_length=9, use_ema=True, use_old_algo=False
):
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

def create_combined_chart(data, indicator_select, strategy_select, signals, macd_data, zl_macd_data):
    """Create a combined chart of stock price and technical indicators."""
    
    # Remove non-trading periods by reindexing with only valid trading times
    valid_times = data.index[~data['Close'].isna()]
    data = data.loc[valid_times]
    if signals is not None:
        signals = signals.loc[valid_times]
    if macd_data is not None:
        macd_data = macd_data.loc[valid_times]
    if zl_macd_data is not None:
        zl_macd_data = zl_macd_data.loc[valid_times]
    
    # Create subplots with shared x-axis
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03)
    
    # Add candlestick chart
    candlestick_trace = go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Candlestick'
    )
    fig.add_trace(candlestick_trace, row=1, col=1)

    if signals is not None:
        buy_signals = data[signals['Signal'] == 1]
        for i in range(len(buy_signals)):
            fig.add_annotation(
                x=buy_signals.index[i],
                y=buy_signals['Close'].iloc[i],
                text="Buy",
                showarrow=True,
                arrowhead=2,
                ax=0,
                ay=-40,
                bgcolor='green',
                font=dict(color='white')
            )
   
        sell_signals = data[signals['Signal'] == -1]
        for i in range(len(sell_signals)):
            fig.add_annotation(
                x=sell_signals.index[i],
                y=sell_signals['Close'].iloc[i],
                text="Sell",
                showarrow=True,
                arrowhead=2,
                ax=0,
                ay=-40,
                bgcolor='red',
                font=dict(color='white')
            )
            
    # Plot buy signals
    if signals is not None:
        buy_signals = data[signals['Signal'] == 1]
        fig.add_trace(go.Scatter(
            x=buy_signals.index,
            y=buy_signals['Close'],
            mode='markers',
            marker=dict(color='green', size=10, symbol='triangle-up'),
            name='Buy Signal',
            visible='legendonly' 
        ))
   
        sell_signals = data[signals['Signal'] == -1]
        fig.add_trace(go.Scatter(
            x=sell_signals.index,
            y=sell_signals['Close'],
            mode='markers',
            marker=dict(color='red', size=10, symbol='triangle-down'),
            name='Sell Signal',
            visible='legendonly' 
        ))
    
    if indicator_select == "MACD":
        macd_fig = plot_macd(data, macd_data)
        # Add traces from the MACD plot
        for trace in macd_fig.data:
            fig.add_trace(trace, row=2, col=1)
    else:
        zl_macd_fig = plot_zero_lag_macd(data, zl_macd_data)
        # Add traces from the Zero Lag MACD plot
        for trace in zl_macd_fig.data:
            fig.add_trace(trace, row=2, col=1)


    # Update layout for hovermode and x-axis
    fig.update_layout(
        hovermode="x unified",
        title='Stock Price and Technical Indicator',
        yaxis_title='Price',
        height=800,
        template='plotly_white',
        xaxis=dict(
            rangeslider=dict(visible=False),
            rangebreaks=[
                dict(bounds=["sat", "mon"]),  # hide weekends
                dict(bounds=[16, 9.5], pattern="hour"),  # hide non-trading hours
            ]
        ),
        xaxis2=dict(
            rangebreaks=[
                dict(bounds=["sat", "mon"]),  # hide weekends
                dict(bounds=[16, 9.5], pattern="hour"),  # hide non-trading hours
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

def backtest_strategy(data, signals, initial_cash=10000, transaction_fee=0.01):
    """Backtest the trading strategy based on generated signals."""
    cash = initial_cash
    shares = 0
    total_trades = 0
    last_sell_cash = initial_cash  # Tracks the cash value after the last sell action
    buy_prices = []
    sell_prices = []
    trade_log = []  # Store trade logs

    for i in range(len(data)):
        if signals['Signal'].iloc[i] == 1:  # Buy signal
            if cash > 0:
                # Calculate maximum shares considering transaction fee
                max_investment = cash / (1 + transaction_fee)
                shares_to_buy = int(max_investment // data['Close'].iloc[i])
                
                if shares_to_buy > 0:
                    purchase_cost = shares_to_buy * data['Close'].iloc[i]
                    fee_cost = purchase_cost * transaction_fee
                    total_cost = purchase_cost + fee_cost
                    
                    if total_cost <= cash:
                        cash -= total_cost
                        shares += shares_to_buy
                        buy_prices.append(data['Close'].iloc[i])
                        total_trades += 1
                        trade_log.append({
                            'type': 'BUY',
                            'date': str(data.index[i]),
                            'price': data['Close'].iloc[i],
                            'shares': shares_to_buy,
                            'fee': fee_cost,
                            'cash_left': cash
                        })

        elif signals['Signal'].iloc[i] == -1:  # Sell signal
            if shares > 0:
                sell_value = shares * data['Close'].iloc[i]
                fee_cost = sell_value * transaction_fee
                net_sell_value = sell_value - fee_cost
                
                cash += net_sell_value
                last_sell_cash = cash  # Update the last sell cash value
                sell_prices.append(data['Close'].iloc[i])
                trade_log.append({
                    'type': 'SELL',
                    'date': str(data.index[i]),
                    'price': data['Close'].iloc[i],
                    'shares': shares,
                    'fee': fee_cost,
                    'cash_left': cash
                })
                shares = 0
                total_trades += 1

    # Determine final portfolio value
    if shares > 0:  # If last action was a "buy," use the last sell cash value
        final_value = last_sell_cash
    else:  # Otherwise, include the current cash and shares value
        final_value = cash + (shares * data['Close'].iloc[-1])
    
    roi = ((final_value - initial_cash) / initial_cash) * 100
    max_profit = max(sell_prices) - min(buy_prices) if buy_prices and sell_prices else 0
    max_loss = min(sell_prices) - max(buy_prices) if buy_prices and sell_prices else 0

    return {
        'final_value': final_value,
        'total_trades': total_trades,
        'roi': roi,
        'max_profit': max_profit,
        'max_loss': max_loss,
        'trade_log': trade_log  # Include trade log in the return
    }

def generate_signals(data, indicator_type, strategy_type, fast_length=12, slow_length=26, signal_length=9):
    """Generate buy/sell signals based on selected strategy"""
    if data.empty:
        print("Data is empty. Cannot generate signals.")
        return pd.DataFrame()  # Return empty DataFrame if no data is available
        
    try:
        if indicator_type == "MACD":
            indicators = calculate_macd(data, fast_length=fast_length, slow_length=slow_length, signal_length=signal_length)
            macd_col = 'MACD'
        else:
            indicators = calculate_zero_lag_macd(data, fast_length=fast_length, slow_length=slow_length, signal_length=signal_length)
            macd_col = 'ZL_MACD'
        
        signals = pd.DataFrame(index=data.index)
        signals['Signal'] = 0
        
        # Print values for debugging
        print("Date\t\tSignal\t\tMACD\t\tSignal Line\tHistogram")
        print("--------------------------------------------------------------")
        
        # Define strategy conditions
        if strategy_type == "Buy Above Sell Above":
            buy_condition = (
                crossover(indicators[macd_col], indicators['Signal']) & 
                (indicators[macd_col] > 0) 
            )
            sell_condition = (
                crossover(indicators['Signal'], indicators[macd_col]) & 
                (indicators[macd_col] > 0) 
            )
        
        elif strategy_type == "Buy Below Sell Above":
            buy_condition = (
                crossover(indicators[macd_col], indicators['Signal']) & 
                (indicators[macd_col] < 0) 
            )
            sell_condition = (
                crossover(indicators['Signal'], indicators[macd_col]) & 
                (indicators[macd_col] > 0) 
            )
        
        elif strategy_type == "Buy Above Sell Below":
            buy_condition = (
                crossover(indicators[macd_col], indicators['Signal']) & 
                (indicators[macd_col] > 0) 
            )
            sell_condition = (
                crossover(indicators['Signal'], indicators[macd_col]) & 
                (indicators[macd_col] < 0) 
            )
        
        elif strategy_type == "Buy Below Sell Below":
            buy_condition = (
                crossover(indicators[macd_col], indicators['Signal']) & 
                (indicators[macd_col] < 0) 
            )
            sell_condition = (
                crossover(indicators['Signal'], indicators[macd_col]) & 
                (indicators[macd_col] < 0) 
            )
        
        elif strategy_type == "Histogram Trend Reversal":
            # Calculate the rate of change of the histogram
            histogram = indicators['Histogram']
            histogram_diff = histogram.diff()

            # Buy Condition: After the histogram has been continuously decreasing and starts increasing
            buy_condition = (
                (histogram.shift(1) < 0) &  # Histogram is negative
                (histogram_diff.shift(1) < 0) &  # Previously, histogram was decreasing
                (histogram_diff > 0)  # Now, histogram starts increasing
            )

            # Sell Condition: After the histogram has been continuously increasing and starts decreasing
            sell_condition = (
                (histogram.shift(1) > 0) &  # Histogram is positive
                (histogram_diff.shift(1) > 0) &  # Previously, histogram was increasing
                (histogram_diff < 0)  # Now, histogram starts decreasing
            )

        
        signals.loc[buy_condition, 'Signal'] = 1
        signals.loc[sell_condition, 'Signal'] = -1

        # for i in range(len(data)):
        #     print(f"Index: {i} | Date: {data.index[i]} | Signal: {signals['Signal'].iloc[i]} | "
        #         f"MACD: {indicators[macd_col].iloc[i]:.2f} | "
        #         f"Signal Line: {indicators['Signal'].iloc[i]:.2f} | "
        #         f"Histogram: {indicators['Histogram'].iloc[i]:.2f}")
        
        return signals
    except Exception as e:
        print(f"Error generating signals: {e}")
        return pd.DataFrame()

def crossover(series1, series2):
    # Check for NaN values and handle them
    if series1.isnull().any() or series2.isnull().any():
        print("Warning: NaN values detected in series.")
    
    # Calculate crossover
    crossover_condition = (series1.shift(1) < series2.shift(1)) & (series1 > series2)
    
    # Debugging output
    for i in range(1, len(series1)):
        if crossover_condition.iloc[i]:
            print(f"Crossover detected at index {i}: series1={series1.iloc[i]}, series2={series2.iloc[i]}")
    
    return crossover_condition