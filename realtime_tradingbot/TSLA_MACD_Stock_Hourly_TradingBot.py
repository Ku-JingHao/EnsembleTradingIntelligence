import pandas as pd
import numpy as np
import yfinance as yf
import logging
import warnings
import time
import sys
import os
import traceback
import keyboard
import colorama
from datetime import datetime, timedelta
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from colorama import Fore, Back, Style

# Initialize colorama for colored terminal output
colorama.init(autoreset=True)
warnings.filterwarnings('ignore')

# Custom logging formatter with colors and better structure
class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors and better structure for log messages"""
    
    FORMATS = {
        logging.DEBUG: Fore.CYAN + "%(asctime)s | %(levelname)-8s | %(message)s",
        logging.INFO: Fore.GREEN + "%(asctime)s | %(levelname)-8s | %(message)s",
        logging.WARNING: Fore.YELLOW + "%(asctime)s | %(levelname)-8s | %(message)s",
        logging.ERROR: Fore.RED + "%(asctime)s | %(levelname)-8s | %(message)s",
        logging.CRITICAL: Fore.RED + Back.WHITE + "%(asctime)s | %(levelname)-8s | %(message)s" + Style.RESET_ALL
    }
    
    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)

# Setup logging with custom formatter
def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # File handler with standard formatting
    file_handler = logging.FileHandler("zero_lag_macd_trading.log", encoding='utf-8')
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    
    # Console handler with colored formatting
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ColoredFormatter())
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

trade_history = []
performance_metrics = {}

# Zero Lag MACD Parameters
FAST_LENGTH = 12
SLOW_LENGTH = 26
SIGNAL_LENGTH = 9
MACD_EMA_LENGTH = 9
USE_EMA = True
USE_OLD_ALGO = False

TICKER = "TSLA" 
YFINANCE_TICKER = "TSLA"
INTERVAL = "1h"

# Alpaca API credentials
ALPACA_API_KEY = "PKJNP12V0TGGUIMFIYAN"
ALPACA_SECRET_KEY = "T31sbuEEnNUmuoUaD5pYddBJalY7zDcKbyG2ETEV"

# Initialize Alpaca clients
trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)
data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

def get_historical_data():
    """Fetch historical data using Yahoo Finance (no subscription required)"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=10)  
        
        # Fetch data using yfinance (free and reliable)
        df = yf.download(YFINANCE_TICKER, start=start_date, end=end_date, interval="1h")
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
    
        if df.empty:
            logger.error("❌ No data fetched from Yahoo Finance")
            return pd.DataFrame()
        
        # Reset index to make Date a column
        df = df.reset_index()
        
        # Rename columns to match expected format
        df = df.rename(columns={
            'Datetime': 'Date', 
            'Open': 'Open',
            'High': 'High',
            'Low': 'Low',
            'Close': 'Close',
            'Volume': 'Volume'
        })
        
        # Handle missing values
        df = df.fillna(method='ffill').fillna(method='bfill')
        df = df.dropna()
        
        logger.info(f"✅ Fetched {len(df)} hour bars for {TICKER} from Yahoo Finance")
        return df
        
    except Exception as e:
        logger.error(f"❌ Error fetching data from Yahoo Finance: {e}")
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

def calculate_signals(pd_data):
    """Calculate Zero Lag MACD signals based on crossovers"""
    # Calculate Zero Lag MACD indicators
    zl_macd_data = calculate_zero_lag_macd(
        pd_data, 
        fast_length=FAST_LENGTH, 
        slow_length=SLOW_LENGTH, 
        signal_length=SIGNAL_LENGTH,
        macd_ema_length=MACD_EMA_LENGTH,
        use_ema=USE_EMA,
        use_old_algo=USE_OLD_ALGO
    )
    
    if zl_macd_data.empty:
        logger.error("Failed to calculate Zero Lag MACD indicators")
        return pd_data
    
    # Merge MACD data with price data
    for col in zl_macd_data.columns:
        pd_data[col] = zl_macd_data[col]
    
    # Drop rows with NaN values
    pd_data = pd_data.dropna()
    pd_data = pd_data.reset_index(drop=True)
    
    if len(pd_data) < 2:
        logger.error("Not enough data points for signal calculation")
        return pd_data
    
    # Calculate crossover signals
    # Buy Condition: When ZL_MACD crosses above Signal line
    pd_data["Above"] = (pd_data["ZL_MACD"].shift(1) <= pd_data["Signal"].shift(1)) & (pd_data["ZL_MACD"] > pd_data["Signal"])
    
    # Sell Condition: When ZL_MACD crosses below Signal line  
    pd_data["Below"] = (pd_data["ZL_MACD"].shift(1) >= pd_data["Signal"].shift(1)) & (pd_data["ZL_MACD"] < pd_data["Signal"])
    
    # Generate buy and sell signals
    pd_data["Buy"] = pd_data["Above"]
    pd_data["Sell"] = pd_data["Below"]
    
    return pd_data

def get_current_position():
    """Get the current position for the ticker"""
    try:
        positions = trading_client.get_all_positions()
        for position in positions:
            if position.symbol == TICKER: 
                return float(position.qty)
        return 0
    except Exception as e:
        logger.error(f"Error getting position: {e}")
        return 0

def execute_trade(side, price):
    """Execute a trade on Alpaca"""
    try:
        if side not in [OrderSide.BUY, OrderSide.SELL]:
            logger.error(f"Invalid order side: {side}")
            return False
        
        # Get the account information
        account = trading_client.get_account()
        cash_balance = float(account.cash)
        
        if side == OrderSide.BUY:
            trade_amount = 1000  
            quantity = trade_amount / price
            cash_left = cash_balance - (quantity * price)
        else:
            # Sell all the shares we have
            current_position = get_current_position()
            quantity = current_position
            cash_left = cash_balance + (quantity * price)  
        
        order_data = MarketOrderRequest(
            symbol="TSLA", 
            qty=quantity,
            side=side,
            time_in_force=TimeInForce.GTC
        )
        
        # Submit order
        order = trading_client.submit_order(order_data)
        
        # Record the trade in history
        trade_history.append({
            'timestamp': datetime.now(),
            'side': side,
            'price': price,
            'quantity': quantity,
            'cash_left': cash_left,
            'execution_reason': 'Zero Lag MACD Signal'
        })
        
        print(f"{'=' * 60}")
        logger.info(f"📊 TRADE EXECUTED - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Type           : {side.name}")
        logger.info(f"Symbol         : {TICKER}")
        logger.info(f"Shares         : {quantity:.6f}")
        logger.info(f"Price          : ${price:.2f}")
        logger.info(f"Total Value    : ${quantity * price:.2f}")
        logger.info(f"Cash Left      : ${cash_left:.2f}")
        logger.info(f"Order ID       : {order.id}")
        logger.info(f"Strategy       : Zero Lag MACD")
        print(f"{'=' * 60}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error executing trade: {e}")
        return False

def print_bot_header():
    """Print a nicely formatted header when the bot starts"""
    header = f"""
{'=' * 60}
{Fore.CYAN}
 📊 ZERO LAG MACD TRADING BOT 📊
 
 ███████╗███████╗██████╗  ██████╗     ██╗      █████╗  ██████╗ 
 ╚══███╔╝██╔════╝██╔══██╗██╔═══██╗    ██║     ██╔══██╗██╔════╝ 
   ███╔╝ █████╗  ██████╔╝██║   ██║    ██║     ███████║██║  ███╗
  ███╔╝  ██╔══╝  ██╔══██╗██║   ██║    ██║     ██╔══██║██║   ██║
 ███████╗███████╗██║  ██║╚██████╔╝    ███████╗██║  ██║╚██████╔╝
 ╚══════╝╚══════╝╚═╝  ╚═╝ ╚═════╝     ╚══════╝╚═╝  ╚═╝ ╚═════╝ 
{Style.RESET_ALL}
{'=' * 60}
"""
    print(header)
    
    logger.info(f"📊 STRATEGY       : Zero Lag MACD Trading Bot")
    logger.info(f"📊 SYMBOL         : {TICKER}")
    logger.info(f"⏰ TIMEFRAME      : {INTERVAL}")
    logger.info(f"⚡ FAST LENGTH    : {FAST_LENGTH}")
    logger.info(f"🐌 SLOW LENGTH    : {SLOW_LENGTH}")
    logger.info(f"📡 SIGNAL LENGTH  : {SIGNAL_LENGTH}")
    logger.info(f"📈 TRADING MODE   : Paper Trading (Alpaca)")
    logger.info(f"💰 ASSET TYPE     : Stock")

def print_detailed_signals(signals_df):
    """Print detailed information for each bar in the dataframe with all signals"""
    print(f"{'=' * 60}")
    logger.info(f"DETAILED SIGNAL ANALYSIS FOR ALL {len(signals_df)} BARS")
    print("\n")

    header = (
        f"{'Date/Time':<20} | "
        f"{'Open':>10} | "
        f"{'High':>10} | "
        f"{'Low':>10} | "
        f"{'Close':>10} | "
        f"{'ZL_MACD':>10} | "
        f"{'Signal':>10} | "
        f"{'Above':^7} | "
        f"{'Below':^7} | "
        f"{'Buy':^7} | "
        f"{'Sell':^7} | "
        f"{'Signal':<10}"
    )
    
    logger.info(header)
    logger.info(f"{'-' * 90}")
    
    # Print most recent rows with proper formatting
    for i, row in signals_df.tail(10).iterrows():
        if isinstance(row['Date'], pd.Timestamp):
            date_str = row['Date'].strftime('%Y-%m-%d %H:%M')
        else:
            date_str = str(row['Date'])[:16]

        if row['Buy']:
            signal_color = Fore.CYAN
            signal_text = "BUY 🔵"
        elif row['Sell']:
            signal_color = Fore.MAGENTA
            signal_text = "SELL 🔴"
        else:
            signal_color = Fore.YELLOW
            signal_text = "NONE ⚪"
        
        row_data = (
            f"{date_str:<20} | "
            f"${row['Open']:>9.4f} | "
            f"${row['High']:>9.4f} | "
            f"${row['Low']:>9.4f} | "
            f"${row['Close']:>9.4f} | "
            f"{row['ZL_MACD']:>9.4f} | "
            f"{row['Signal']:>9.4f} | "
            f"{str(row['Above']):^7} | "
            f"{str(row['Below']):^7} | "
            f"{str(row['Buy']):^7} | "
            f"{str(row['Sell']):^7} | "
            f"{signal_color}{signal_text}{Style.RESET_ALL}"
        )
        
        logger.info(row_data)
    
    print(f"{'=' * 60}")

def print_signal_update(latest_bar, current_price, current_position, signal_count=1):
    """Print a nicely formatted signal update"""
    print(f"{'=' * 60}")
    logger.info(f"📊 SIGNAL UPDATE #{signal_count} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Format datetime for hour data
    if isinstance(latest_bar['Date'], pd.Timestamp):
        date_str = latest_bar['Date'].strftime('%Y-%m-%d %H:%M:%S')
    else:
        date_str = str(latest_bar['Date'])
    
    logger.info(f"Date Time      : {date_str}")
    logger.info(f"Price          : ${current_price:.4f}")
    logger.info(f"ZL MACD        : {latest_bar['ZL_MACD']:.4f}")
    logger.info(f"Signal Line    : {latest_bar['Signal']:.4f}")
    logger.info(f"Histogram      : {latest_bar['Histogram']:.4f}")
    logger.info(f"Position       : {current_position} {TICKER}") 
    
    # Use different colored signals based on buy/sell
    if latest_bar['Buy']:
        logger.info(f"{Fore.CYAN}SIGNAL         : BUY 🔵{Style.RESET_ALL}")
    elif latest_bar['Sell']:
        logger.info(f"{Fore.MAGENTA}SIGNAL         : SELL 🔴{Style.RESET_ALL}")
    else:
        logger.info(f"{Fore.YELLOW}SIGNAL         : NO SIGNAL ⚪{Style.RESET_ALL}")

def print_waiting_message(next_check_time):
    """Print a clear waiting message to show bot is active and waiting for next check"""
    print(f"{'=' * 60}")
    logger.info(f"{Fore.BLUE}📊 BOT STATUS      : MONITORING {TICKER}{Style.RESET_ALL}")
    logger.info(f"Current Time   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Next Check     : {next_check_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Calculate time remaining
    time_remaining = next_check_time - datetime.now()
    hours, remainder = divmod(time_remaining.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    logger.info(f"Time Remaining : {hours:02d}:{minutes:02d}:{seconds:02d}")

    if time_remaining.days > 0:
        logger.info(f"Days Remaining : {time_remaining.days}")

    current_time = datetime.now()
    is_weekend = current_time.weekday() >= 5
    
    if is_weekend:
        logger.info(f"{Fore.YELLOW}Market Status  : Weekend - Limited Trading Activity{Style.RESET_ALL}")
    else:
        logger.info(f"{Fore.GREEN}Market Status  : Weekday - Normal Trading Hours{Style.RESET_ALL}")
    
    logger.info(f"📊 Strategy      : Pure Zero Lag MACD")
    print(f"{'=' * 60}")

def run_live_trading():
    """Run the Zero Lag MACD strategy continuously in live trading mode"""
    # Setup custom logging
    global logger
    logger = setup_logging()
    
    # Print the bot header
    print_bot_header()
    
    last_check_hour = None
    signal_count = 0  
    waiting_message_shown = False

    # Print instructions for stopping the bot
    logger.info(f"{Fore.YELLOW}To stop the bot and view performance summary, hold the 'q' key for a few seconds.{Style.RESET_ALL}")
        
    try:
        while True:
            try:
                # Check if user pressed 'q' to quit
                if keyboard.is_pressed('q'):
                    logger.info("Bot stopped by user. Displaying performance summary...")
                    calculate_roi_summary()
                    break
                
                # Get current hour
                current_datetime = datetime.now()
                current_hour = current_datetime.replace(minute=0, second=0, microsecond=0)

                # Only process once per hour for 1-hour timeframe
                if last_check_hour != current_hour:
                    # Reset waiting message flag when starting a new hour check
                    waiting_message_shown = False

                    df = get_historical_data() # Get historical data
                    signals_df = calculate_signals(df) # Calculate signals
                    latest_bar = signals_df.iloc[-1] # Check latest signal (most recent bar)
                    current_price = latest_bar['Close']
                    
                    current_position = get_current_position() # Get current position

                    signal_count += 1 # Increment signal counter
                    print_signal_update(latest_bar, current_price, current_position, signal_count)
                    
                    # Process buy signal
                    if latest_bar['Buy']:
                        if current_position > 0:
                            logger.info(f"Already holding {current_position} {TICKER}, no action taken.")
                        else:
                            logger.info(f"📊 EXECUTING BUY: Opening new long position at ${current_price:.4f}")
                            execute_trade(OrderSide.BUY, current_price)
                    
                    # Process sell signal
                    elif latest_bar['Sell']:
                        if current_position > 0:
                            logger.info(f"📊 EXECUTING SELL: Selling {current_position} {TICKER} at ${current_price:.4f}")
                            execute_trade(OrderSide.SELL, current_price)
                        else:
                            logger.info("No position to sell.")
                    
                    else:
                        logger.info("No new trading signals this hour")
                    
                    last_check_hour = current_hour
                            
                next_check_time = current_datetime + timedelta(hours=1)

                # Show waiting message only once after processing signals and if not already shown
                if last_check_hour == current_hour and not waiting_message_shown:
                    print_waiting_message(next_check_time)
                    waiting_message_shown = True  
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in live trading execution: {e}")
                logger.error(f"{'=' * 60}")
                time.sleep(60)  # Retry after 60 seconds

    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        logger.info("Bot stopped by user. Displaying performance summary...")
        calculate_roi_summary()

def run_signal_test():
    """Run a test to print all historical signals without executing trades"""
    # Setup custom logging
    global logger
    logger = setup_logging()
    
    # Print the bot header
    print_bot_header()
    logger.info("📊 RUNNING IN SIGNAL TEST MODE - NO TRADES WILL BE EXECUTED")
    
    df = get_historical_data() # Get historical data
    signals_df = calculate_signals(df) # Calculate signals
    print_detailed_signals(signals_df) # Print detailed signals for recent bars
    latest_bar = signals_df.iloc[-1] # Check latest signal (most recent bar)
    current_price = latest_bar['Close']

    current_position = get_current_position() # Get current position
    
    print_signal_update(latest_bar, current_price, current_position, signal_count=1)
    
    buy_signals = signals_df['Buy'].sum()
    sell_signals = signals_df['Sell'].sum()
    
    print(f"{'=' * 60}")
    logger.info(f"📊 SIGNAL SUMMARY")
    logger.info(f"Total bars     : {len(signals_df)}")
    logger.info(f"Buy signals    : {buy_signals} ({buy_signals/len(signals_df)*100:.2f}%)")
    logger.info(f"Sell signals   : {sell_signals} ({sell_signals/len(signals_df)*100:.2f}%)")
    logger.info(f"No signals     : {len(signals_df) - buy_signals - sell_signals} ({(len(signals_df) - buy_signals - sell_signals)/len(signals_df)*100:.2f}%)")
    print(f"{'=' * 60}")

def calculate_roi_summary():
    """Calculate and print ROI summary for each weekly interval"""
    if not trade_history:
        logger.info("No trades executed. ROI summary not available.")
        return
    
    # Group trades by weekly intervals
    for trade in trade_history:
        # Create a weekly interval key 
        interval_time = trade['timestamp'].replace(
            hour=0, minute=0, second=0, microsecond=0
        ) - timedelta(days=trade['timestamp'].weekday())
        interval_key = interval_time.strftime('%Y-%m-%d')
        
        if interval_key not in performance_metrics:
            performance_metrics[interval_key] = {
                'initial_cash': trade['cash_left'] + (trade['quantity'] * trade['price'] if trade['side'] == OrderSide.SELL else trade['cash_left']),
                'final_cash': trade['cash_left'],
                'trades': []
            }
        performance_metrics[interval_key]['trades'].append(trade)
        performance_metrics[interval_key]['final_cash'] = trade['cash_left']
    
    # Calculate ROI for each weekly interval
    interval_counter = 1  # Initialize interval counter
    print(f"{'=' * 60}")
    logger.info(f"{Fore.CYAN}PERFORMANCE SUMMARY BY WEEKLY INTERVALS{Style.RESET_ALL}")
    
    for interval, data in sorted(performance_metrics.items()):
        initial_cash = data['initial_cash']
        final_cash = data['final_cash']
        roi = ((final_cash - initial_cash) / initial_cash) * 100
        
        # Color code the ROI based on performance
        if roi > 0:
            roi_color = Fore.GREEN
        elif roi < 0:
            roi_color = Fore.RED
        else:
            roi_color = Fore.YELLOW
            
        logger.info(f"Week {interval_counter}: {interval}")
        logger.info(f"Initial Cash   : ${initial_cash:.2f}")
        logger.info(f"Final Cash     : ${final_cash:.2f}")
        logger.info(f"ROI            : {roi_color}{roi:+.2f}%{Style.RESET_ALL}")
        logger.info(f"Number of Trades: {len(data['trades'])}")
        logger.info(f"{'-' * 40}")
        interval_counter += 1
    
    # Calculate overall performance
    if trade_history:
        first_trade = trade_history[0]
        last_trade = trade_history[-1]
        
        # Get initial capital
        overall_initial = first_trade['cash_left'] + (first_trade['quantity'] * first_trade['price'] if first_trade['side'] == OrderSide.SELL else first_trade['cash_left'])
        
        # For final value, check if we have an open position
        current_position = get_current_position()
        
        if current_position > 0:
            # Get latest price to value the position
            try:
                df = get_historical_data()
                signals_df = calculate_signals(df)
                current_price = signals_df.iloc[-1]['Close']
                position_value = current_position * current_price
                overall_final = last_trade['cash_left'] + position_value
                logger.info(f"Open position  : {current_position} {TICKER} valued at ${position_value:.2f}")
            except Exception as e:
                logger.error(f"Error getting current position value: {e}")
                overall_final = last_trade['cash_left']
        else:
            overall_final = last_trade['cash_left']
            
        overall_roi = ((overall_final - overall_initial) / overall_initial) * 100
        
        logger.info(f"{Fore.CYAN}OVERALL PERFORMANCE{Style.RESET_ALL}")
        logger.info(f"Starting Capital: ${overall_initial:.2f}")
        logger.info(f"Current Capital : ${overall_final:.2f}")
        
        if overall_roi > 0:
            logger.info(f"Total ROI       : {Fore.GREEN}+{overall_roi:.2f}%{Style.RESET_ALL}")
        elif overall_roi < 0:
            logger.info(f"Total ROI       : {Fore.RED}{overall_roi:.2f}%{Style.RESET_ALL}")
        else:
            logger.info(f"Total ROI       : {Fore.YELLOW}{overall_roi:.2f}%{Style.RESET_ALL}")
            
        logger.info(f"Total Trades    : {len(trade_history)}")
    
    print(f"{'=' * 60}")

if __name__ == "__main__":
    #run_signal_test()  # Use this to test signals without trading
    run_live_trading()  # Use this for actual trading