import pandas as pd
import numpy as np
import yfinance as yf
import logging
import warnings
import time
import sys
import os
import pickle
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
from darts import TimeSeries
from darts.models import RNNModel
from sklearn.preprocessing import MinMaxScaler
from darts.utils.missing_values import fill_missing_values

# Initialize colorama for colored terminal output
colorama.init(autoreset=True)
warnings.filterwarnings('ignore')

# 🔥 NEW: ENSEMBLE CONFIGURATION
ENABLE_AI_VALIDATION = True  # Set to True to enable AI validation
MODELS_DIR = 'models'  # Directory containing deployed models

# Ensemble Models Configuration for HRMY
ENSEMBLE_CONFIG = {
    'linear_regression': {
        'display_name': 'Linear Regression',
        'model_file': 'linear_regression_HRMY_1h_model.pkl',
        'config_file': 'linear_regression_HRMY_1h_config.pkl',
        'type': 'traditional_ml',
        'weight': 0.33  # Equal weight for 3-model ensemble
    },
    'random_forest': {
        'display_name': 'Random Forest',
        'model_file': 'random_forest_HRMY_1h_model.pkl',
        'config_file': 'random_forest_HRMY_1h_config.pkl',
        'type': 'traditional_ml',
        'weight': 0.33  # Equal weight for 3-model ensemble
    },
    'lstm': {
        'display_name': 'LSTM',
        'model_file': 'lstm_HRMY_1h_model.pt',
        'config_file': None,  # LSTM model has built-in config
        'type': 'deep_learning',
        'weight': 0.34  # Slightly higher to sum to 1.0
    }
}

# Define MetricsCallback locally for pickle compatibility
class MetricsCallback:
    """Dummy MetricsCallback class for pickle compatibility"""
    def __init__(self):
        self.metrics = {
            'train_loss': [],
            'val_loss': []
        }

# Make it available globally for unpickling
globals()['MetricsCallback'] = MetricsCallback
sys.modules[__name__].MetricsCallback = MetricsCallback

# Custom pickle loader
class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'MetricsCallback':
            return MetricsCallback
        return super().find_class(module, name)

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
    file_handler = logging.FileHandler("ensemble_trading.log", encoding='utf-8')
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
ai_validation_stats = {
    'total_signals': 0,
    'confirmed_signals': 0,
    'rejected_signals': 0,
    'ai_failures': 0,
    'fallback_executions': 0,
    'ensemble_predictions': 0,
    'lr_predictions': 0,
    'rf_predictions': 0,  # 🔥 NEW: Random Forest tracking
    'lstm_predictions': 0
}

# Zero Lag MACD Parameters
FAST_LENGTH = 12
SLOW_LENGTH = 26
SIGNAL_LENGTH = 9
MACD_EMA_LENGTH = 9
USE_EMA = True
USE_OLD_ALGO = False

# Tesla Stock Configuration
TICKER = "HRMY"
YFINANCE_TICKER = "HRMY"
INTERVAL = "1h"

# Alpaca API credentials
ALPACA_API_KEY = "PKJNP12V0TGGUIMFIYAN"
ALPACA_SECRET_KEY = "T31sbuEEnNUmuoUaD5pYddBJalY7zDcKbyG2ETEV"

# Initialize Alpaca clients
trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)
data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

def load_ensemble_models():
    """Load Linear Regression, Random Forest, and LSTM models for ensemble prediction"""
    models = {}
    
    # Load Linear Regression Model
    try:
        lr_config = ENSEMBLE_CONFIG['linear_regression']
        model_path = f'{MODELS_DIR}/{lr_config["model_file"]}'
        config_path = f'{MODELS_DIR}/{lr_config["config_file"]}'
        
        if os.path.exists(model_path) and os.path.exists(config_path):
            with open(config_path, 'rb') as f:
                unpickler = CustomUnpickler(f)
                config = unpickler.load()
            
            with open(model_path, 'rb') as f:
                unpickler = CustomUnpickler(f)
                model = unpickler.load()
            
            models['linear_regression'] = {
                'model': model,
                'config': config,
                'type': 'traditional_ml',
                'display_name': lr_config['display_name'],
                'weight': lr_config['weight']
            }
            logger.info(f"✅ {lr_config['display_name']} model loaded successfully")
        else:
            logger.warning(f"❌ {lr_config['display_name']} model files not found")
    except Exception as e:
        logger.error(f"❌ Failed to load Linear Regression model: {e}")
    
    # 🔥 NEW: Load Random Forest Model
    try:
        rf_config = ENSEMBLE_CONFIG['random_forest']
        model_path = f'{MODELS_DIR}/{rf_config["model_file"]}'
        config_path = f'{MODELS_DIR}/{rf_config["config_file"]}'
        
        if os.path.exists(model_path) and os.path.exists(config_path):
            with open(config_path, 'rb') as f:
                unpickler = CustomUnpickler(f)
                config = unpickler.load()
            
            with open(model_path, 'rb') as f:
                unpickler = CustomUnpickler(f)
                model = unpickler.load()
            
            models['random_forest'] = {
                'model': model,
                'config': config,
                'type': 'traditional_ml',
                'display_name': rf_config['display_name'],
                'weight': rf_config['weight']
            }
            logger.info(f"✅ {rf_config['display_name']} model loaded successfully")
        else:
            logger.warning(f"❌ {rf_config['display_name']} model files not found")
    except Exception as e:
        logger.error(f"❌ Failed to load Random Forest model: {e}")
    
    # Load LSTM Model (unchanged)
    try:
        lstm_config = ENSEMBLE_CONFIG['lstm']
        model_path = f'{MODELS_DIR}/{lstm_config["model_file"]}'
        
        if os.path.exists(model_path):
            model = RNNModel.load(model_path)
            
            models['lstm'] = {
                'model': model,
                'config': {
                    'type': 'deep_learning',
                    'name': 'LSTM',
                    'input_chunk_length': model.input_chunk_length,
                    'output_chunk_length': model.output_chunk_length
                },
                'type': 'deep_learning',
                'display_name': lstm_config['display_name'],
                'weight': lstm_config['weight']
            }
            logger.info(f"✅ {lstm_config['display_name']} model loaded successfully")
        else:
            logger.warning(f"❌ {lstm_config['display_name']} model file not found: {model_path}")
    except Exception as e:
        logger.error(f"❌ Failed to load LSTM model: {e}")
    
    return models

def prepare_traditional_ml_data(data, interval="1h"):
    """Prepare data for Linear Regression model"""
    try:
        logger.info(f"📊 DEBUG - Preparing ML data: {len(data)} points")
        
        # Handle NaN values
        clean_data = data.copy()
        clean_data = clean_data.ffill().bfill()
        clean_data = clean_data.dropna()
        
        if clean_data.empty:
            logger.error("❌ DEBUG - No valid data after NaN handling")
            return None, None, None
        
        # Prepare target and covariates
        target_data = clean_data[['Close']].copy()
        covariate_data = clean_data[['Open', 'High', 'Low', 'Volume']].copy()
        
        # Check for constant values
        close_values = target_data['Close'].values
        if len(np.unique(close_values)) == 1:
            logger.warning(f"🚨 DEBUG - All Close values identical, adding noise")
            noise = np.random.normal(0, abs(close_values[0]) * 1e-8, len(close_values))
            target_data['Close'] = close_values + noise
        
        # Create scalers
        target_scaler = MinMaxScaler()
        covariate_scaler = MinMaxScaler()
        
        # Scale data
        try:
            scaled_target_values = target_scaler.fit_transform(target_data)
            scaled_covariate_values = covariate_scaler.fit_transform(covariate_data)
            
            # Handle NaN in scaling results
            if np.isnan(scaled_target_values).any():
                logger.warning("🚨 DEBUG - Target scaling produced NaN, using manual scaling")
                data_min = target_data['Close'].min()
                data_max = target_data['Close'].max()
                if data_max == data_min:
                    scaled_target_values = np.full_like(target_data['Close'].values, 0.5).reshape(-1, 1)
                else:
                    scaled_target_values = ((target_data['Close'].values - data_min) / (data_max - data_min)).reshape(-1, 1)
            
            if np.isnan(scaled_covariate_values).any():
                logger.warning("🚨 DEBUG - Covariate scaling produced NaN, using manual scaling")
                scaled_covariate_values = np.zeros_like(covariate_data.values)
                for i, col in enumerate(covariate_data.columns):
                    col_data = covariate_data[col].values
                    col_min, col_max = col_data.min(), col_data.max()
                    if col_max == col_min:
                        scaled_covariate_values[:, i] = 0.5
                    else:
                        scaled_covariate_values[:, i] = (col_data - col_min) / (col_max - col_min)
            
        except Exception as scaling_error:
            logger.error(f"❌ DEBUG - Scaling failed: {scaling_error}")
            return None, None, None
        
        # Create DataFrames
        scaled_target = pd.DataFrame(
            scaled_target_values,
            columns=['Close'],
            index=target_data.index
        )
        
        scaled_covariates = pd.DataFrame(
            scaled_covariate_values,
            columns=covariate_data.columns,
            index=covariate_data.index
        )
        
        # Handle timezone
        if scaled_target.index.tz is not None:
            logger.info("🔧 DEBUG - Removing timezone for Darts compatibility")
            scaled_target.index = scaled_target.index.tz_localize(None)
            scaled_covariates.index = scaled_covariates.index.tz_localize(None)
        
        # Convert to TimeSeries
        freq_map = {"1m": "T", "1h": "H", "1d": "B"}
        freq = freq_map.get(interval, "H")
        
        try:
            target_ts = TimeSeries.from_dataframe(
                scaled_target, 
                freq=freq,
                fill_missing_dates=True,
                fillna_value=None
            )
            
            covariate_ts = TimeSeries.from_dataframe(
                scaled_covariates, 
                freq=freq,
                fill_missing_dates=True,
                fillna_value=None
            )
            
            # Fill any remaining NaN values
            target_values = target_ts.values()
            covariate_values = covariate_ts.values()
            
            if np.isnan(target_values).any() or np.isnan(covariate_values).any():
                logger.info("🔧 DEBUG - Filling NaN values in TimeSeries...")
                target_ts = fill_missing_values(target_ts, fill='auto')
                covariate_ts = fill_missing_values(covariate_ts, fill='auto')
            
            # Final validation
            target_values = target_ts.values()
            covariate_values = covariate_ts.values()
            
            if np.isnan(target_values).any() or np.isinf(target_values).any():
                logger.error("🚨 DEBUG - Target TimeSeries still has invalid values")
                return None, None, None
            
            if np.isnan(covariate_values).any() or np.isinf(covariate_values).any():
                logger.error("🚨 DEBUG - Covariate TimeSeries still has invalid values")
                return None, None, None
            
            logger.info(f"✅ DEBUG - ML data prepared successfully")
            return target_ts, covariate_ts, target_scaler
            
        except Exception as ts_error:
            logger.error(f"❌ DEBUG - TimeSeries creation failed: {ts_error}")
            return None, None, None
            
    except Exception as e:
        logger.error(f"❌ DEBUG - Error preparing ML data: {e}")
        return None, None, None

def prepare_deep_learning_data(data, interval="1h"):
    """Prepare data for LSTM model"""
    try:
        # Create a new scaler for the target data
        scaler = MinMaxScaler()
        target_data = data[['Close']].copy()
        
        logger.info(f"📊 DEBUG - LSTM data: {len(target_data)} points")
        
        # Handle NaN values
        target_data = target_data.ffill().bfill()
        target_data = target_data.dropna()
        
        if target_data.empty:
            logger.error("❌ DEBUG - No valid data for LSTM")
            return None, None
        
        # Check for constant values
        close_values = target_data['Close'].values
        if len(np.unique(close_values)) == 1:
            logger.warning(f"🚨 DEBUG - All values identical for LSTM, adding noise")
            noise = np.random.normal(0, abs(close_values[0]) * 1e-8, len(close_values))
            target_data['Close'] = close_values + noise
            close_values = target_data['Close'].values
        
        # Safe scaling
        try:
            scaled_values = scaler.fit_transform(target_data[['Close']])
            
            if np.isnan(scaled_values).any() or np.isinf(scaled_values).any():
                logger.warning("🚨 DEBUG - LSTM scaler produced invalid values, using manual scaling")
                data_min = close_values.min()
                data_max = close_values.max()
                data_range = data_max - data_min
                
                if data_range == 0:
                    scaled_values = np.full_like(close_values, 0.5).reshape(-1, 1)
                else:
                    scaled_values = ((close_values - data_min) / data_range).reshape(-1, 1)
                
                # Create manual scaler for inverse transform
                class ManualScaler:
                    def __init__(self, data_min, data_max):
                        self.data_min_ = data_min
                        self.data_max_ = data_max
                    
                    def inverse_transform(self, X):
                        if self.data_max_ == self.data_min_:
                            return np.full_like(X, self.data_min_)
                        return X * (self.data_max_ - self.data_min_) + self.data_min_
                
                scaler = ManualScaler(data_min, data_max)
            
        except Exception as scaling_error:
            logger.error(f"❌ DEBUG - LSTM scaling failed: {scaling_error}")
            return None, None
        
        # Handle timezone
        clean_index = target_data.index
        if hasattr(clean_index, 'tz') and clean_index.tz is not None:
            logger.info(f"📊 DEBUG - Removing timezone for LSTM")
            clean_index = clean_index.tz_localize(None)
        
        # Create continuous time index
        start_time = clean_index.min()
        end_time = clean_index.max()
        continuous_index = pd.date_range(start=start_time, end=end_time, freq='H')
        
        # Create DataFrame with continuous index
        scaled_target = pd.DataFrame(
            scaled_values,
            columns=['Close'],
            index=clean_index
        )
        
        # Reindex to continuous timeline and forward fill gaps
        scaled_target_continuous = scaled_target.reindex(continuous_index, method='ffill')
        scaled_target_continuous = scaled_target_continuous.ffill().bfill()
        
        # Convert to TimeSeries
        try:
            target_ts = TimeSeries.from_dataframe(
                scaled_target_continuous, 
                time_col=None,
                value_cols=['Close'],
                fillna_value=0.0
            )
            
            # Handle any remaining NaN values
            ts_values = target_ts.values()
            if np.isnan(ts_values).any():
                logger.warning(f"🚨 DEBUG - LSTM TimeSeries has NaN, filling...")
                valid_values = ts_values[~np.isnan(ts_values)]
                if len(valid_values) > 0:
                    fill_value = np.median(valid_values)
                else:
                    fill_value = 0.5
                
                filled_values = np.where(np.isnan(ts_values), fill_value, ts_values)
                filled_df = pd.DataFrame(
                    filled_values,
                    columns=['Close'],
                    index=scaled_target_continuous.index
                )
                
                target_ts = TimeSeries.from_dataframe(filled_df)
            
            # Final validation
            ts_values = target_ts.values()
            if np.isnan(ts_values).any() or np.isinf(ts_values).any():
                logger.error("🚨 DEBUG - LSTM TimeSeries still has invalid values")
                return None, None
            
            logger.info(f"✅ DEBUG - LSTM data prepared successfully")
            return target_ts, scaler
            
        except Exception as ts_error:
            logger.error(f"❌ DEBUG - LSTM TimeSeries creation failed: {ts_error}")
            return None, None
            
    except Exception as e:
        logger.error(f"❌ DEBUG - Error preparing LSTM data: {e}")
        return None, None

def fetch_and_prepare_data_for_ai(ticker_symbol, days_back=10):
    """Fetch extended historical data for AI prediction"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Fetch HRMY data using yfinance
        df = yf.download(YFINANCE_TICKER, start=start_date, end=end_date, interval="1h")
        
        if df.empty:
            logger.error("❌ No data fetched from yfinance")
            return pd.DataFrame()
        
        # Handle multi-level columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        
        # Enhanced data cleaning
        logger.info(f"📊 Raw data: {len(df)} points, NaN count: {df.isnull().sum().sum()}")
        
        # Handle missing values
        df = df.fillna(method='ffill').fillna(method='bfill')
        df = df.dropna()
        
        # Validate price values
        for price_col in ['Open', 'High', 'Low', 'Close']:
            if (df[price_col] <= 0).any():
                logger.warning(f"⚠️ Found zero/negative prices in {price_col}, fixing...")
                df[price_col] = df[price_col].mask(df[price_col] <= 0).interpolate(method='linear')
                df[price_col] = df[price_col].fillna(method='ffill').fillna(method='bfill')
        
        # Handle infinite values
        if np.isinf(df.select_dtypes(include=[np.number])).any().any():
            logger.warning("⚠️ Found infinite values, replacing...")
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.fillna(method='ffill').fillna(method='bfill')
        
        logger.info(f"✅ Clean data: {len(df)} points, final NaN count: {df.isnull().sum().sum()}")
        
        return df
        
    except Exception as e:
        logger.error(f"❌ Error fetching data for AI: {e}")
        return pd.DataFrame()

def generate_ensemble_prediction(current_price, historical_data):
    """Generate ensemble prediction using Linear Regression, Random Forest, and LSTM models"""
    try:
        models = load_ensemble_models()
        if not models:
            logger.warning("⚠️ No models available for ensemble")
            return None, "No models available"
        
        predictions = {}
        weights = {}
        
        # Generate Linear Regression prediction
        if 'linear_regression' in models:
            try:
                lr_model = models['linear_regression']['model']
                lr_config = models['linear_regression']['config']
                lags = lr_config['params'].get('lags', 7)
                
                target_ts, covariate_ts, lr_scaler = prepare_traditional_ml_data(historical_data, interval="1h")
                if target_ts is not None and len(target_ts) >= lags:
                    lr_predictions = lr_model.predict(
                        n=1,
                        series=target_ts,
                        past_covariates=covariate_ts
                    )
                    
                    lr_pred_values = lr_predictions.values().reshape(-1, 1)
                    lr_inverse_pred = lr_scaler.inverse_transform(lr_pred_values)
                    lr_predicted_price = lr_inverse_pred.flatten()[0]
                    
                    predictions['linear_regression'] = lr_predicted_price
                    weights['linear_regression'] = models['linear_regression']['weight']
                    
                    logger.info(f"🤖 Linear Regression Prediction: ${lr_predicted_price:.4f}")
                    ai_validation_stats['lr_predictions'] += 1
                    
            except Exception as lr_error:
                logger.error(f"❌ Linear Regression prediction failed: {lr_error}")
        
        # 🔥 NEW: Generate Random Forest prediction
        if 'random_forest' in models:
            try:
                rf_model = models['random_forest']['model']
                rf_config = models['random_forest']['config']
                lags = rf_config['params'].get('lags', 7)
                
                target_ts, covariate_ts, rf_scaler = prepare_traditional_ml_data(historical_data, interval="1h")
                if target_ts is not None and len(target_ts) >= lags:
                    rf_predictions = rf_model.predict(
                        n=1,
                        series=target_ts,
                        past_covariates=covariate_ts
                    )
                    
                    rf_pred_values = rf_predictions.values().reshape(-1, 1)
                    rf_inverse_pred = rf_scaler.inverse_transform(rf_pred_values)
                    rf_predicted_price = rf_inverse_pred.flatten()[0]
                    
                    predictions['random_forest'] = rf_predicted_price
                    weights['random_forest'] = models['random_forest']['weight']
                    
                    logger.info(f"🤖 Random Forest Prediction: ${rf_predicted_price:.4f}")
                    ai_validation_stats['rf_predictions'] += 1
                    
            except Exception as rf_error:
                logger.error(f"❌ Random Forest prediction failed: {rf_error}")
        
        # Generate LSTM prediction (unchanged)
        if 'lstm' in models:
            try:
                lstm_model = models['lstm']['model']
                input_chunk_length = models['lstm']['config']['input_chunk_length']
                
                target_ts, lstm_scaler = prepare_deep_learning_data(historical_data, interval="1h")
                if target_ts is not None and len(target_ts) >= input_chunk_length:
                    lstm_predictions = lstm_model.predict(n=1, series=target_ts)
                    
                    lstm_pred_values = lstm_predictions.values().reshape(-1, 1)
                    lstm_inverse_pred = lstm_scaler.inverse_transform(lstm_pred_values)
                    lstm_predicted_price = lstm_inverse_pred.flatten()[0]
                    
                    predictions['lstm'] = lstm_predicted_price
                    weights['lstm'] = models['lstm']['weight']
                    
                    logger.info(f"🤖 LSTM Prediction: ${lstm_predicted_price:.4f}")
                    ai_validation_stats['lstm_predictions'] += 1
                    
            except Exception as lstm_error:
                logger.error(f"❌ LSTM prediction failed: {lstm_error}")
        
        # Calculate ensemble prediction using weighted average
        if predictions:
            if len(predictions) == 1:
                model_name = list(predictions.keys())[0]
                ensemble_price = predictions[model_name]
                logger.info(f"🔄 Using single model prediction: {model_name}")
                status = f"Single model: {model_name}"
            else:
                total_weight = sum(weights.values())
                weighted_sum = sum(predictions[model] * weights[model] for model in predictions)
                ensemble_price = weighted_sum / total_weight
                
                logger.info(f"🎯 THREE-MODEL ENSEMBLE PREDICTION: ${ensemble_price:.4f}")
                logger.info(f"   - Linear Regression: ${predictions.get('linear_regression', 'N/A')}")
                logger.info(f"   - Random Forest: ${predictions.get('random_forest', 'N/A')}")
                logger.info(f"   - LSTM: ${predictions.get('lstm', 'N/A')}")
                logger.info(f"   - Weighted Average: ${ensemble_price:.4f}")
                
                ai_validation_stats['ensemble_predictions'] += 1
                status = f"Ensemble ({len(predictions)} models)"
            
            return ensemble_price, status
        else:
            logger.warning("⚠️ All model predictions failed")
            return None, "All models failed"
            
    except Exception as e:
        logger.error(f"❌ Ensemble prediction failed: {e}")
        return None, str(e)

def validate_signal_with_ensemble(signal_type, current_price, historical_data):
    """
    Validate buy/sell signal using ensemble prediction from both LR and LSTM
    
    Returns:
        - True: Execute the trade
        - False: Reject the trade
        - execution_reason: Detailed reason for the decision
    """
    global ai_validation_stats
    
    ai_validation_stats['total_signals'] += 1
    
    # Log signal detection
    logger.info(f"🔍 ENSEMBLE VALIDATION: {signal_type.upper()} signal detected at ${current_price:.4f}")
    
    if not ENABLE_AI_VALIDATION:
        logger.info(f"⚙️ AI validation disabled, executing traditional signal")
        ai_validation_stats['fallback_executions'] += 1
        return True, "AI validation disabled - Traditional signal executed"
    
    # Generate ensemble prediction
    predicted_price, prediction_status = generate_ensemble_prediction(current_price, historical_data)
    
    if predicted_price is None:
        # Ensemble prediction failed - fallback to traditional signal
        logger.warning(f"⚠️ Ensemble prediction failed: {prediction_status}")
        logger.info(f"🔄 FALLBACK: Executing traditional {signal_type} signal")
        ai_validation_stats['ai_failures'] += 1
        ai_validation_stats['fallback_executions'] += 1
        return True, f"Ensemble failed ({prediction_status}) - Traditional signal executed"
    
    # Ensemble validation logic
    if signal_type.lower() == 'buy':
        # Buy signal: Execute if predicted price > current price
        if predicted_price > current_price:
            price_diff = predicted_price - current_price
            price_diff_pct = (price_diff / current_price) * 100
            
            logger.info(f"✅ ENSEMBLE CONFIRMED BUY: Predicted=${predicted_price:.4f} > Current=${current_price:.4f}")
            logger.info(f"📈 Expected gain: ${price_diff:.4f} (+{price_diff_pct:.2f}%)")
            
            ai_validation_stats['confirmed_signals'] += 1
            return True, f"Ensemble confirmed ({prediction_status}) - Expected gain: +{price_diff_pct:.2f}%"
        else:
            price_diff = current_price - predicted_price
            price_diff_pct = (price_diff / current_price) * 100
            
            logger.info(f"❌ ENSEMBLE REJECTED BUY: Predicted=${predicted_price:.4f} < Current=${current_price:.4f}")
            logger.info(f"📉 Expected loss: ${price_diff:.4f} (-{price_diff_pct:.2f}%)")
            
            ai_validation_stats['rejected_signals'] += 1
            return False, f"Ensemble rejected ({prediction_status}) - Expected loss: -{price_diff_pct:.2f}%"
    
    elif signal_type.lower() == 'sell':
        # Sell signal: Execute if predicted price < current price
        if predicted_price < current_price:
            price_diff = current_price - predicted_price
            price_diff_pct = (price_diff / current_price) * 100
            
            logger.info(f"✅ ENSEMBLE CONFIRMED SELL: Predicted=${predicted_price:.4f} < Current=${current_price:.4f}")
            logger.info(f"📉 Expected decline: ${price_diff:.4f} (-{price_diff_pct:.2f}%)")
            
            ai_validation_stats['confirmed_signals'] += 1
            return True, f"Ensemble confirmed ({prediction_status}) - Expected decline: -{price_diff_pct:.2f}%"
        else:
            price_diff = predicted_price - current_price
            price_diff_pct = (price_diff / current_price) * 100
            
            logger.info(f"❌ ENSEMBLE REJECTED SELL: Predicted=${predicted_price:.4f} > Current=${current_price:.4f}")
            logger.info(f"📈 Expected rise: ${price_diff:.4f} (+{price_diff_pct:.2f}%)")
            
            ai_validation_stats['rejected_signals'] += 1
            return False, f"Ensemble rejected ({prediction_status}) - Expected rise: +{price_diff_pct:.2f}%"
    
    # Should not reach here
    logger.error(f"❌ Unknown signal type: {signal_type}")
    ai_validation_stats['fallback_executions'] += 1
    return True, f"Unknown signal type - Traditional signal executed"

def get_historical_data():
    """Fetch historical data using Yahoo Finance"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=10)
        
        # Fetch HRMY data using yfinance
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
        
        logger.info(f"✅ Fetched {len(df)} hourly bars for {TICKER} from Yahoo Finance")
        return df
        
    except Exception as e:
        logger.error(f"❌ Error fetching data from Yahoo Finance: {e}")
        return pd.DataFrame()

def calculate_zero_lag_macd(data, fast_length=12, slow_length=26, signal_length=9, macd_ema_length=9, use_ema=True, use_old_algo=False):
    """Calculate Zero Lag MACD indicator"""
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
    pd_data["Above"] = (pd_data["ZL_MACD"].shift(1) <= pd_data["Signal"].shift(1)) & (pd_data["ZL_MACD"] > pd_data["Signal"])
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

def execute_trade(side, price, execution_reason="Traditional signal"):
    """Execute a trade on Alpaca with ensemble validation tracking"""
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
            symbol=TICKER,
            qty=quantity,
            side=side,
            time_in_force=TimeInForce.DAY
        )
        
        # Submit order
        order = trading_client.submit_order(order_data)
        
        # Record the trade in history with ensemble validation info
        trade_history.append({
            'timestamp': datetime.now(),
            'side': side,
            'price': price,
            'quantity': quantity,
            'cash_left': cash_left,
            'execution_reason': execution_reason,
            'ai_validation': ENABLE_AI_VALIDATION
        })
        
        print(f"{'=' * 60}")
        logger.info(f"🤖 ENSEMBLE TRADE EXECUTED - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Type           : {side.name}")
        logger.info(f"Symbol         : {TICKER}")
        logger.info(f"Shares         : {quantity:.6f}")
        logger.info(f"Price          : ${price:.2f}")
        logger.info(f"Total Value    : ${quantity * price:.2f}")
        logger.info(f"Cash Left      : ${cash_left:.2f}")
        logger.info(f"Order ID       : {order.id}")
        logger.info(f"Ensemble Info  : {execution_reason}")
        print(f"{'=' * 60}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error executing trade: {e}")
        return False

def print_bot_header():
    """Print a nicely formatted header when the bot starts"""
    header = f"""
{'=' * 70}
{Fore.CYAN}
 🤖 THREE-MODEL ENSEMBLE AI TRADING BOT 🤖
 
 ████████╗██╗  ██╗██████╗ ███████╗███████╗    ███╗   ███╗ ██████╗ ██████╗ ███████╗██╗     
 ╚══██╔══╝██║  ██║██╔══██╗██╔════╝██╔════╝    ████╗ ████║██╔═══██╗██╔══██╗██╔════╝██║     
    ██║   ███████║██████╔╝█████╗  █████╗      ██╔████╔██║██║   ██║██║  ██║█████╗  ██║     
    ██║   ██╔══██║██╔══██╗██╔══╝  ██╔══╝      ██║╚██╔╝██║██║   ██║██║  ██║██╔══╝  ██║     
    ██║   ██║  ██║██║  ██║███████╗███████╗    ██║ ╚═╝ ██║╚██████╔╝██████╔╝███████╗███████╗
    ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚══════╝    ╚═╝     ╚═╝ ╚═════╝ ╚═════╝ ╚══════╝╚══════╝
{Style.RESET_ALL}
{'=' * 70}
"""
    print(header)
    
    logger.info(f"🤖 STRATEGY       : Three-Model Ensemble AI-Enhanced Zero Lag MACD Trading Bot")
    logger.info(f"📊 SYMBOL         : {TICKER} (Tesla Inc.)")
    logger.info(f"⏰ TIMEFRAME      : {INTERVAL}")
    logger.info(f"⚡ FAST LENGTH    : {FAST_LENGTH}")
    logger.info(f"🐌 SLOW LENGTH    : {SLOW_LENGTH}")
    logger.info(f"📡 SIGNAL LENGTH  : {SIGNAL_LENGTH}")
    logger.info(f"🤖 AI MODELS      : Linear Regression + Random Forest + LSTM Ensemble")
    logger.info(f"✅ AI VALIDATION  : {'ENABLED' if ENABLE_AI_VALIDATION else 'DISABLED'}")
    logger.info(f"📈 TRADING MODE   : Paper Trading (Alpaca)")
    logger.info(f"💰 ASSET TYPE     : Stock")
    
    # Show ensemble configuration
    logger.info(f"🎯 THREE-MODEL ENSEMBLE CONFIG:")
    for model_key, config in ENSEMBLE_CONFIG.items():
        logger.info(f"   - {config['display_name']}: Weight {config['weight']}")

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
    logger.info(f"🤖 ENSEMBLE SIGNAL UPDATE #{signal_count} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Format datetime for hourly data
    if isinstance(latest_bar['Date'], pd.Timestamp):
        date_str = latest_bar['Date'].strftime('%Y-%m-%d %H:%M:%S')
    else:
        date_str = str(latest_bar['Date'])
    
    logger.info(f"Date Time      : {date_str}")
    logger.info(f"Price          : ${current_price:.2f}")
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

def print_ai_validation_stats():
    """Print three-model ensemble validation statistics"""
    stats = ai_validation_stats
    print(f"{'=' * 60}")
    logger.info(f"🤖 THREE-MODEL ENSEMBLE VALIDATION STATISTICS")
    logger.info(f"Total Signals         : {stats['total_signals']}")
    logger.info(f"Ensemble Confirmed    : {stats['confirmed_signals']}")
    logger.info(f"Ensemble Rejected     : {stats['rejected_signals']}")
    logger.info(f"AI Failures           : {stats['ai_failures']}")
    logger.info(f"Fallback Executions   : {stats['fallback_executions']}")
    logger.info(f"Ensemble Predictions  : {stats['ensemble_predictions']}")
    logger.info(f"LR Only Predictions   : {stats['lr_predictions']}")
    logger.info(f"RF Only Predictions   : {stats['rf_predictions']}")  # 🔥 NEW
    logger.info(f"LSTM Only Predictions : {stats['lstm_predictions']}")
    
    if stats['total_signals'] > 0:
        confirmation_rate = (stats['confirmed_signals'] / stats['total_signals']) * 100
        rejection_rate = (stats['rejected_signals'] / stats['total_signals']) * 100
        failure_rate = (stats['ai_failures'] / stats['total_signals']) * 100
        
        logger.info(f"Confirmation Rate     : {confirmation_rate:.1f}%")
        logger.info(f"Rejection Rate        : {rejection_rate:.1f}%")
        logger.info(f"AI Failure Rate       : {failure_rate:.1f}%")
    
    print(f"{'=' * 60}")

def print_waiting_message(next_check_time):
    """Print a clear waiting message to show bot is active and waiting for next check"""
    print(f"{'=' * 60}")
    logger.info(f"{Fore.BLUE}🤖 THREE-MODEL ENSEMBLE BOT STATUS : MONITORING {TICKER}{Style.RESET_ALL}")
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
    
    logger.info(f"🤖 AI Models     : LR + RF + LSTM ({'ACTIVE' if ENABLE_AI_VALIDATION else 'INACTIVE'})")
    print(f"{'=' * 60}")

def run_live_trading():
    """Run the Ensemble AI-Enhanced Zero Lag MACD strategy continuously in live trading mode"""
    # Setup custom logging
    global logger
    logger = setup_logging()
    
    # Print the bot header
    print_bot_header()
    
    # Test ensemble model loading on startup
    logger.info(f"🔍 Testing ensemble model loading...")
    test_models = load_ensemble_models()
    if test_models:
        model_names = list(test_models.keys())
        logger.info(f"✅ Loaded {len(model_names)} models: {', '.join(model_names)}")
    else:
        logger.warning(f"⚠️ No models loaded - will use fallback")
    
    last_check_hour = None
    signal_count = 0
    waiting_message_shown = False

    # Print instructions for stopping the bot
    logger.info(f"{Fore.YELLOW}To stop the bot and view ensemble validation summary, hold the 'q' key for a few seconds.{Style.RESET_ALL}")
        
    try:
        while True:
            try:
                # Check if user pressed 'q' to quit
                if keyboard.is_pressed('q'):
                    logger.info("Bot stopped by user. Displaying ensemble validation summary...")
                    print_ai_validation_stats()
                    calculate_roi_summary()
                    break
                
                # Get current hour
                current_datetime = datetime.now()
                current_hour = current_datetime.replace(minute=0, second=0, microsecond=0)

                # Only process once per hour for 1-hour timeframe
                if last_check_hour != current_hour:
                    # Reset waiting message flag when starting a new hour check
                    waiting_message_shown = False

                    df = get_historical_data()
                    signals_df = calculate_signals(df)
                    latest_bar = signals_df.iloc[-1]
                    current_price = latest_bar['Close']
                    
                    current_position = get_current_position()

                    signal_count += 1
                    print_signal_update(latest_bar, current_price, current_position, signal_count)
                    
                    # Get extended historical data for ensemble validation
                    historical_data = fetch_and_prepare_data_for_ai(TICKER, days_back=10)
                    
                    # Process buy signal with ensemble validation
                    if latest_bar['Buy']:
                        if current_position > 0:
                            logger.info(f"Already holding {current_position} {TICKER}, no action taken.")
                        else:
                            # Validate with ensemble
                            should_execute, execution_reason = validate_signal_with_ensemble(
                                'buy', current_price, historical_data
                            )
                            
                            if should_execute:
                                logger.info(f"🤖 ENSEMBLE VALIDATION PASSED: Opening new long position at ${current_price:.4f}")
                                execute_trade(OrderSide.BUY, current_price, execution_reason)
                            else:
                                logger.info(f"🤖 ENSEMBLE VALIDATION FAILED: Buy signal rejected - {execution_reason}")
                    
                    # Process sell signal with ensemble validation
                    elif latest_bar['Sell']:
                        if current_position > 0:
                            # Validate with ensemble
                            should_execute, execution_reason = validate_signal_with_ensemble(
                                'sell', current_price, historical_data
                            )
                            
                            if should_execute:
                                logger.info(f"🤖 ENSEMBLE VALIDATION PASSED: Selling {current_position} {TICKER} at ${current_price:.4f}")
                                execute_trade(OrderSide.SELL, current_price, execution_reason)
                            else:
                                logger.info(f"🤖 ENSEMBLE VALIDATION FAILED: Sell signal rejected - {execution_reason}")
                        else:
                            logger.info("No position to sell.")
                    
                    else:
                        logger.info("No new trading signals this hour")
                    
                    # Record that we checked this hour
                    last_check_hour = current_hour
                            
                # Calculate next check time (next hour)
                next_check_time = current_datetime + timedelta(hours=1)

                # Show waiting message only once after processing signals and if not already shown
                if last_check_hour == current_hour and not waiting_message_shown:
                    print_waiting_message(next_check_time)
                    waiting_message_shown = True
                
                time.sleep(30)
                
            except Exception as e:
                logger.error(f"Error in live trading execution: {e}")
                logger.error(f"{'=' * 60}")
                time.sleep(60)

    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        logger.info("Bot stopped by user. Displaying ensemble validation summary...")
        print_ai_validation_stats()
        calculate_roi_summary()

def run_signal_test():
    """Run a test to print all historical signals without executing trades"""
    # Setup custom logging
    global logger
    logger = setup_logging()
    
    # Print the bot header
    print_bot_header()
    logger.info("🤖 RUNNING IN ENSEMBLE SIGNAL TEST MODE - NO TRADES WILL BE EXECUTED")
    
    df = get_historical_data()
    signals_df = calculate_signals(df)
    print_detailed_signals(signals_df)
    latest_bar = signals_df.iloc[-1]
    current_price = latest_bar['Close']

    current_position = get_current_position()
    
    print_signal_update(latest_bar, current_price, current_position, signal_count=1)
    
    buy_signals = signals_df['Buy'].sum()
    sell_signals = signals_df['Sell'].sum()
    
    print(f"{'=' * 60}")
    logger.info(f"🤖 ENSEMBLE SIGNAL SUMMARY")
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
    interval_counter = 1
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