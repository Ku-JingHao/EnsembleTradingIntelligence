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
from sklearn.preprocessing import MinMaxScaler
from darts.utils.missing_values import fill_missing_values

# Initialize colorama for colored terminal output
colorama.init(autoreset=True)
warnings.filterwarnings('ignore')

# AI Enhancement Configuration - FLEXIBLE MODEL SELECTION
ENABLE_AI_VALIDATION = True  # Set to True to enable AI validation
MODELS_DIR = 'models'  # Directory containing deployed models

# 🔥 NEW: Configurable ML Model Selection
# Change this to switch between your deployed models
SELECTED_ML_MODEL = 'linear_regression'  # Options: 'linear_regression', 'random_forest', 'xgboost'

# Model configuration mapping
ML_MODEL_CONFIG = {
    'linear_regression': {
        'display_name': 'Linear Regression',
        'model_file': 'linear_regression_model.pkl',
        'config_file': 'linear_regression_config.pkl',
        'type': 'traditional_ml'
    },
    'random_forest': {
        'display_name': 'Random Forest',
        'model_file': 'random_forest_model.pkl',
        'config_file': 'random_forest_config.pkl',
        'type': 'traditional_ml'
    },
    'xgboost': {
        'display_name': 'XGBoost',
        'model_file': 'xgboost_model.pkl',
        'config_file': 'xgboost_config.pkl',
        'type': 'traditional_ml'
    }
}

# Set AI model name based on selection
AI_MODEL_NAME = ML_MODEL_CONFIG[SELECTED_ML_MODEL]['display_name']

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
    file_handler = logging.FileHandler("ai_macd_trading.log", encoding='utf-8')
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

# Global variables to track trades and performance in 30-minute intervals
trade_history = []
performance_metrics = {}
ai_validation_stats = {
    'total_signals': 0,
    'confirmed_signals': 0,
    'rejected_signals': 0,
    'ai_failures': 0,
    'fallback_executions': 0
}

# Zero Lag MACD Parameters (replacing UT Bot parameters)
FAST_LENGTH = 12
SLOW_LENGTH = 26
SIGNAL_LENGTH = 9
MACD_EMA_LENGTH = 9
USE_EMA = True
USE_OLD_ALGO = False

TICKER = "CRVUSD"  
YFINANCE_TICKER = "CRV-USD"  # Yahoo Finance uses different format for crypto
INTERVAL = "1m"  # Keep as 1m for minute bot

# Alpaca API credentials
ALPACA_API_KEY = "PKJNP12V0TGGUIMFIYAN"
ALPACA_SECRET_KEY = "T31sbuEEnNUmuoUaD5pYddBJalY7zDcKbyG2ETEV"

# Initialize Alpaca clients
trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)
data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

def load_selected_ml_model():
    """Load the currently selected ML model - FLEXIBLE VERSION"""
    try:
        # Get configuration for selected model
        model_config = ML_MODEL_CONFIG.get(SELECTED_ML_MODEL)
        if not model_config:
            logger.error(f"❌ Unknown model selection: {SELECTED_ML_MODEL}")
            logger.error(f"Available models: {list(ML_MODEL_CONFIG.keys())}")
            return None
        
        model_path = f'{MODELS_DIR}/{model_config["model_file"]}'
        config_path = f'{MODELS_DIR}/{model_config["config_file"]}'
        
        if not os.path.exists(model_path) or not os.path.exists(config_path):
            logger.warning(f"❌ {model_config['display_name']} model files not found")
            logger.warning(f"Expected: {model_path} and {config_path}")
            return None
        
        # Load config first
        with open(config_path, 'rb') as f:
            unpickler = CustomUnpickler(f)
            config = unpickler.load()
        
        # Load model using pickle
        with open(model_path, 'rb') as f:
            unpickler = CustomUnpickler(f)
            model = unpickler.load()
        
        logger.info(f"✅ {model_config['display_name']} model loaded successfully")
        logger.info(f"📁 Model file: {model_config['model_file']}")
        
        return {
            'model': model,
            'config': config,
            'model_type': SELECTED_ML_MODEL,
            'display_name': model_config['display_name']
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to load {AI_MODEL_NAME} model: {e}")
        return None

def prepare_traditional_ml_data(data, interval="1m"):
    """Prepare data for traditional ML models with robust NaN handling - ENHANCED DEBUG VERSION"""
    try:
        logger.info(f"📊 DEBUG - Initial data: {len(data)} points, NaN count: {data.isnull().sum().sum()}")
        
        # Handle NaN values before any processing
        clean_data = data.copy()
        
        # Forward fill NaN values first, then backward fill any remaining
        clean_data = clean_data.ffill().bfill()
        clean_data = clean_data.dropna()
        
        if clean_data.empty:
            logger.error("❌ DEBUG - No valid data after NaN handling")
            return None, None, None
        
        logger.info(f"✅ DEBUG - After NaN handling: {len(clean_data)} points")
        
        # Prepare target and covariates from clean data
        target_data = clean_data[['Close']].copy()
        covariate_data = clean_data[['Open', 'High', 'Low', 'Volume']].copy()
        
        # 🔥 ENHANCED DEBUG: Log target data statistics
        close_values = target_data['Close'].values
        logger.info(f"🔍 DEBUG - Target data analysis:")
        logger.info(f"    Close values count: {len(close_values)}")
        logger.info(f"    Close min: {close_values.min():.8f}")
        logger.info(f"    Close max: {close_values.max():.8f}")
        logger.info(f"    Close mean: {close_values.mean():.8f}")
        logger.info(f"    Close std: {close_values.std():.8f}")
        logger.info(f"    Unique values: {len(np.unique(close_values))}")
        logger.info(f"    Has NaN: {np.isnan(close_values).any()}")
        logger.info(f"    Has Inf: {np.isinf(close_values).any()}")
        
        # Check for constant values in target
        if len(np.unique(close_values)) == 1:
            logger.warning(f"🚨 DEBUG - All Close values are identical ({close_values[0]:.6f})")
            logger.info("🔧 DEBUG - Adding tiny noise to prevent zero variance...")
            noise = np.random.normal(0, abs(close_values[0]) * 1e-8, len(close_values))
            target_data['Close'] = close_values + noise
            close_values = target_data['Close'].values
            logger.info(f"🔧 DEBUG - After noise: min={close_values.min():.8f}, max={close_values.max():.8f}")
        
        # Create scalers
        target_scaler = MinMaxScaler()
        covariate_scaler = MinMaxScaler()
        
        # Safe scaling with validation
        try:
            logger.info(f"📊 DEBUG - Starting scaling process...")
            
            # Scale the target data
            scaled_target_values = target_scaler.fit_transform(target_data)
            
            # 🔥 ENHANCED DEBUG: Log scaling results
            logger.info(f"🔍 DEBUG - Scaling results:")
            logger.info(f"    Scaler data_min_: {target_scaler.data_min_[0]:.8f}")
            logger.info(f"    Scaler data_max_: {target_scaler.data_max_[0]:.8f}")
            logger.info(f"    Scaler scale_: {target_scaler.scale_[0]:.8f}")
            logger.info(f"    Scaled min: {scaled_target_values.min():.8f}")
            logger.info(f"    Scaled max: {scaled_target_values.max():.8f}")
            logger.info(f"    Scaled mean: {scaled_target_values.mean():.8f}")
            logger.info(f"    Scaled std: {scaled_target_values.std():.8f}")
            logger.info(f"    Scaled has NaN: {np.isnan(scaled_target_values).any()}")
            logger.info(f"    Scaled has Inf: {np.isinf(scaled_target_values).any()}")
            
            # Check for NaN in scaled target
            if np.isnan(scaled_target_values).any():
                logger.warning(f"🚨 DEBUG - Target scaling produced NaN values!")
                nan_count = np.isnan(scaled_target_values).sum()
                logger.warning(f"🚨 DEBUG - NaN count in scaled values: {nan_count}")
                
                # Manual scaling fallback for target
                data_min = target_data['Close'].min()
                data_max = target_data['Close'].max()
                if data_max == data_min:
                    scaled_target_values = np.full_like(target_data['Close'].values, 0.5).reshape(-1, 1)
                else:
                    scaled_target_values = ((target_data['Close'].values - data_min) / (data_max - data_min)).reshape(-1, 1)
                logger.info("✅ DEBUG - Used manual scaling for target")
                logger.info(f"🔧 DEBUG - Manual scaling: min={scaled_target_values.min():.8f}, max={scaled_target_values.max():.8f}")
            
            scaled_target = pd.DataFrame(
                scaled_target_values,
                columns=['Close'],
                index=target_data.index
            )
            
            logger.info(f"📊 DEBUG - Target scaling successful, range: {scaled_target_values.min():.6f} to {scaled_target_values.max():.6f}")
            
        except Exception as target_scaling_error:
            logger.error(f"❌ DEBUG - Target scaling failed: {target_scaling_error}")
            return None, None, None
        
        # Safe covariate scaling
        try:
            logger.info(f"📊 DEBUG - Scaling covariates...")
            
            # 🔥 ENHANCED DEBUG: Log covariate data before scaling
            logger.info(f"🔍 DEBUG - Covariate data analysis:")
            for col in covariate_data.columns:
                col_values = covariate_data[col].values
                logger.info(f"    {col}: min={col_values.min():.6f}, max={col_values.max():.6f}, "
                          f"NaN={np.isnan(col_values).any()}, Inf={np.isinf(col_values).any()}")
            
            scaled_covariate_values = covariate_scaler.fit_transform(covariate_data)
            
            # 🔥 ENHANCED DEBUG: Log covariate scaling results
            logger.info(f"🔍 DEBUG - Covariate scaling results:")
            logger.info(f"    Scaled shape: {scaled_covariate_values.shape}")
            logger.info(f"    Scaled min: {scaled_covariate_values.min():.8f}")
            logger.info(f"    Scaled max: {scaled_covariate_values.max():.8f}")
            logger.info(f"    Scaled has NaN: {np.isnan(scaled_covariate_values).any()}")
            logger.info(f"    Scaled has Inf: {np.isinf(scaled_covariate_values).any()}")
            
            # Check for NaN in scaled covariates
            if np.isnan(scaled_covariate_values).any():
                logger.warning(f"🚨 DEBUG - Covariate scaling produced NaN values!")
                nan_indices = np.where(np.isnan(scaled_covariate_values))
                logger.warning(f"🚨 DEBUG - NaN positions: rows={nan_indices[0][:5]}, cols={nan_indices[1][:5]} (showing first 5)")
                
                # Manual scaling for each covariate column
                scaled_covariate_values = np.zeros_like(covariate_data.values)
                for i, col in enumerate(covariate_data.columns):
                    col_data = covariate_data[col].values
                    col_min, col_max = col_data.min(), col_data.max()
                    if col_max == col_min:
                        scaled_covariate_values[:, i] = 0.5
                        logger.info(f"🔧 DEBUG - {col}: constant values, set to 0.5")
                    else:
                        scaled_covariate_values[:, i] = (col_data - col_min) / (col_max - col_min)
                        logger.info(f"🔧 DEBUG - {col}: manual scaling successful")
                logger.info("✅ DEBUG - Used manual scaling for covariates")
            
            scaled_covariates = pd.DataFrame(
                scaled_covariate_values,
                columns=covariate_data.columns,
                index=covariate_data.index
            )
            
            logger.info(f"📊 DEBUG - Covariate scaling successful, range: {scaled_covariate_values.min():.6f} to {scaled_covariate_values.max():.6f}")
            
        except Exception as cov_scaling_error:
            logger.error(f"❌ DEBUG - Covariate scaling failed: {cov_scaling_error}")
            return None, None, None
        
        # 🔥 FIX: Convert to TimeSeries with frequency mapping like model_loader.py
        # 🔥 FIX: Convert to TimeSeries with frequency mapping like model_loader.py
        try:
            logger.info(f"📊 DEBUG - Creating TimeSeries...")
            
            # 🔥 Handle timezone issues (like model_loader.py)
            if scaled_target.index.tz is not None:
                logger.info("🔧 DEBUG - Removing timezone for Darts compatibility")
                scaled_target.index = scaled_target.index.tz_localize(None)
                scaled_covariates.index = scaled_covariates.index.tz_localize(None)
            
            # 🔥 ENHANCED DEBUG: Log DataFrame details before TimeSeries creation
            logger.info(f"🔍 DEBUG - DataFrame analysis before TimeSeries:")
            logger.info(f"    Target DataFrame shape: {scaled_target.shape}")
            logger.info(f"    Target index type: {type(scaled_target.index)}")
            logger.info(f"    Target index range: {scaled_target.index[0]} to {scaled_target.index[-1]}")
            logger.info(f"    Target DataFrame NaN count: {scaled_target.isnull().sum().sum()}")
            logger.info(f"    Target DataFrame Inf count: {np.isinf(scaled_target.select_dtypes(include=[np.number])).sum().sum()}")
            
            # Sample some values for inspection
            sample_indices = [0, len(scaled_target)//4, len(scaled_target)//2, -1]
            logger.info(f"🔍 DEBUG - Sample target values:")
            for idx in sample_indices:
                if 0 <= idx < len(scaled_target):
                    val = scaled_target.iloc[idx]['Close']
                    logger.info(f"    Index {idx}: {val:.8f}")
            
            # 🔥 FREQUENCY MAPPING: Match model_loader.py approach
            freq_map = {
                "1m": "T",      # 🔥 ADD: Minute frequency mapping
                "1h": "H", 
                "1d": "B"
            }
            freq = freq_map.get(interval, "T")  # Default to minute for trading bot
            
            logger.info(f"🔧 DEBUG - Using frequency '{freq}' for interval '{interval}'")
            
            # 🔥 SOLUTION: Use fill_missing_dates=True and fillna for minute data gaps
            logger.info(f"🔧 DEBUG - Creating target TimeSeries with gap filling...")
            target_ts = TimeSeries.from_dataframe(
                scaled_target, 
                freq=freq,
                fill_missing_dates=True,
                fillna_value=None  # Let Darts fill with NaN first, then we'll handle it
            )
            logger.info(f"✅ DEBUG - Target TimeSeries created successfully")
            
            logger.info(f"🔧 DEBUG - Creating covariate TimeSeries with gap filling...")
            covariate_ts = TimeSeries.from_dataframe(
                scaled_covariates, 
                freq=freq,
                fill_missing_dates=True,
                fillna_value=None  # Let Darts fill with NaN first, then we'll handle it
            )
            logger.info(f"✅ DEBUG - Covariate TimeSeries created successfully")
            
            # 🔥 POST-PROCESSING: Fill NaN values in TimeSeries
            logger.info(f"🔧 DEBUG - Checking for NaN values in TimeSeries...")
            
            target_values = target_ts.values()
            covariate_values = covariate_ts.values()
            
            # Log initial state
            target_nan_count = np.isnan(target_values).sum()
            covariate_nan_count = np.isnan(covariate_values).sum()
            
            logger.info(f"🔍 DEBUG - Initial TimeSeries state:")
            logger.info(f"    Target TS shape: {target_values.shape}")
            logger.info(f"    Target TS NaN count: {target_nan_count}")
            logger.info(f"    Covariate TS shape: {covariate_values.shape}")
            logger.info(f"    Covariate TS NaN count: {covariate_nan_count}")
            
            # 🔥 FILL NaN VALUES if they exist
            # 🔥 FILL NaN VALUES if they exist
            # 🔥 FILL NaN VALUES if they exist
            if target_nan_count > 0 or covariate_nan_count > 0:
                logger.info(f"🔧 DEBUG - Filling NaN values in TimeSeries...")
                
                # Import Darts missing values utility
                from darts.utils.missing_values import fill_missing_values
                
                # 🔥 DIRECT NaN FILLING: Use correct Darts methods for TimeSeries
                if target_nan_count > 0:
                    logger.info(f"🔧 DEBUG - Filling {target_nan_count} NaN values in target TimeSeries...")
                    # Use Darts built-in gap filling methods
                    try:
                        target_ts = fill_missing_values(target_ts, fill='auto')
                    except:
                        # Alternative: try forward fill
                        try:
                            target_ts = fill_missing_values(target_ts, fill='ffill')
                        except:
                            # Last resort: use backward fill
                            target_ts = fill_missing_values(target_ts, fill='bfill')
                
                if covariate_nan_count > 0:
                    logger.info(f"🔧 DEBUG - Filling {covariate_nan_count} NaN values in covariate TimeSeries...")
                    # Use Darts built-in gap filling methods
                    try:
                        covariate_ts = fill_missing_values(covariate_ts, fill='auto')
                    except:
                        # Alternative: try forward fill
                        try:
                            covariate_ts = fill_missing_values(covariate_ts, fill='ffill')
                        except:
                            # Last resort: use backward fill
                            covariate_ts = fill_missing_values(covariate_ts, fill='bfill')
                
                # Verify NaN removal
                target_values = target_ts.values()
                covariate_values = covariate_ts.values()
                
                final_target_nan = np.isnan(target_values).sum()
                final_covariate_nan = np.isnan(covariate_values).sum()
                
                logger.info(f"✅ DEBUG - After NaN filling:")
                logger.info(f"    Target TS NaN count: {final_target_nan}")
                logger.info(f"    Covariate TS NaN count: {final_covariate_nan}")
                
                if final_target_nan > 0 or final_covariate_nan > 0:
                    logger.warning(f"🔧 DEBUG - Some NaN values remain, trying alternative approach...")
                    
                    # 🔥 ALTERNATIVE: Convert to DataFrame, fill NaN, convert back
                    if final_target_nan > 0:
                        logger.info(f"🔧 DEBUG - Using DataFrame approach for target...")
                        try:
                            # Convert to DataFrame
                            target_df = target_ts.pd_dataframe()
                            # Fill NaN using pandas methods
                            target_df = target_df.ffill().bfill()
                            # Convert back to TimeSeries
                            target_ts = TimeSeries.from_dataframe(target_df, freq=freq)
                        except:
                            logger.error("❌ DEBUG - DataFrame approach failed for target")
                    
                    if final_covariate_nan > 0:
                        logger.info(f"🔧 DEBUG - Using DataFrame approach for covariates...")
                        try:
                            # Convert to DataFrame
                            covariate_df = covariate_ts.pd_dataframe()
                            # Fill NaN using pandas methods
                            covariate_df = covariate_df.ffill().bfill()
                            # Convert back to TimeSeries
                            covariate_ts = TimeSeries.from_dataframe(covariate_df, freq=freq)
                        except:
                            logger.error("❌ DEBUG - DataFrame approach failed for covariates")
                    
                    # Final check after alternative approach
                    target_values = target_ts.values()
                    covariate_values = covariate_ts.values()
                    
                    final_target_nan = np.isnan(target_values).sum()
                    final_covariate_nan = np.isnan(covariate_values).sum()
                    
                    logger.info(f"🔧 DEBUG - After alternative filling:")
                    logger.info(f"    Target TS NaN count: {final_target_nan}")
                    logger.info(f"    Covariate TS NaN count: {final_covariate_nan}")
                    
                    if final_target_nan > 0 or final_covariate_nan > 0:
                        logger.error(f"🚨 DEBUG - Cannot remove all NaN values!")
                        return None, None, None
            
            # 🔥 ENHANCED DEBUG: Final validation with detailed analysis
            logger.info(f"🔍 DEBUG - Final TimeSeries validation:")
            logger.info(f"    Target TS shape: {target_values.shape}")
            logger.info(f"    Target TS min: {target_values.min():.8f}")
            logger.info(f"    Target TS max: {target_values.max():.8f}")
            logger.info(f"    Target TS mean: {target_values.mean():.8f}")
            logger.info(f"    Target TS std: {target_values.std():.8f}")
            logger.info(f"    Target TS NaN count: {np.isnan(target_values).sum()}")
            logger.info(f"    Target TS Inf count: {np.isinf(target_values).sum()}")
            
            logger.info(f"    Covariate TS shape: {covariate_values.shape}")
            logger.info(f"    Covariate TS min: {covariate_values.min():.8f}")
            logger.info(f"    Covariate TS max: {covariate_values.max():.8f}")
            logger.info(f"    Covariate TS NaN count: {np.isnan(covariate_values).sum()}")
            logger.info(f"    Covariate TS Inf count: {np.isinf(covariate_values).sum()}")
            
            # 🔥 FINAL CHECK: Ensure no invalid values
            if np.isnan(target_values).any():
                nan_positions = np.where(np.isnan(target_values))
                logger.error(f"🚨 DEBUG - Target TimeSeries still has NaN positions: {nan_positions[0][:5]}")
                return None, None, None
            
            if np.isinf(target_values).any():
                inf_positions = np.where(np.isinf(target_values))
                logger.error(f"🚨 DEBUG - Target TimeSeries has Inf positions: {inf_positions[0][:5]}")
                return None, None, None
            
            if np.isnan(covariate_values).any():
                nan_positions = np.where(np.isnan(covariate_values))
                logger.error(f"🚨 DEBUG - Covariate TimeSeries still has NaN positions: {nan_positions[0][:5]}")
                return None, None, None
            
            if np.isinf(covariate_values).any():
                inf_positions = np.where(np.isinf(covariate_values))
                logger.error(f"🚨 DEBUG - Covariate TimeSeries has Inf positions: {inf_positions[0][:5]}")
                return None, None, None
            
            logger.info(f"✅ DEBUG - Traditional ML data prepared successfully:")
            logger.info(f"    Target TimeSeries: Length={len(target_ts)}, Range={target_values.min():.6f} to {target_values.max():.6f}")
            logger.info(f"    Covariate TimeSeries: Length={len(covariate_ts)}, Features={covariate_values.shape[1]}")
            logger.info(f"    Frequency: {target_ts.freq}")
            
            return target_ts, covariate_ts, target_scaler
            
        except Exception as ts_error:
            logger.error(f"❌ DEBUG - TimeSeries creation failed: {ts_error}")
            logger.error(f"📊 DEBUG - Full traceback: {traceback.format_exc()}")
            
            # 🔥 FALLBACK: Try with fill_missing_dates if explicit freq fails
            try:
                logger.info("🔄 DEBUG - Trying fallback with fill_missing_dates=True")
                
                target_ts = TimeSeries.from_dataframe(
                    scaled_target, 
                    fill_missing_dates=True,
                    freq=freq
                )
                
                covariate_ts = TimeSeries.from_dataframe(
                    scaled_covariates, 
                    fill_missing_dates=True,
                    freq=freq
                )
                
                # 🔥 ENHANCED DEBUG: Check fallback results
                target_values = target_ts.values()
                covariate_values = covariate_ts.values()
                
                logger.info(f"🔍 DEBUG - Fallback TimeSeries validation:")
                logger.info(f"    Target fallback NaN count: {np.isnan(target_values).sum()}")
                logger.info(f"    Target fallback Inf count: {np.isinf(target_values).sum()}")
                logger.info(f"    Covariate fallback NaN count: {np.isnan(covariate_values).sum()}")
                logger.info(f"    Covariate fallback Inf count: {np.isinf(covariate_values).sum()}")
                
                if np.isnan(target_values).any() or np.isinf(target_values).any():
                    logger.error(f"🚨 DEBUG - Fallback target still has invalid values!")
                    return None, None, None
                
                if np.isnan(covariate_values).any() or np.isinf(covariate_values).any():
                    logger.error(f"🚨 DEBUG - Fallback covariate still has invalid values!")
                    return None, None, None
                
                logger.info(f"✅ DEBUG - Fallback TimeSeries creation successful")
                return target_ts, covariate_ts, target_scaler
                
            except Exception as fallback_error:
                logger.error(f"❌ DEBUG - Fallback TimeSeries creation also failed: {fallback_error}")
                logger.error(f"📊 DEBUG - Fallback traceback: {traceback.format_exc()}")
                return None, None, None
            
    except Exception as e:
        logger.error(f"❌ DEBUG - Error preparing traditional ML data: {e}")
        logger.error(f"📊 DEBUG - Full traceback: {traceback.format_exc()}")
        return None, None, None

def fetch_and_prepare_data_for_ai(ticker_symbol, minutes_back=1440):
    """Fetch extended historical data for AI prediction"""
    try:
        # Calculate dates for minute data
        end_date = datetime.now()
        start_date = end_date - timedelta(minutes=minutes_back)
        
        # Use the Yahoo Finance ticker format for crypto
        yf_ticker = YFINANCE_TICKER if ticker_symbol == TICKER else ticker_symbol
        
        # Fetch data using yfinance (more reliable for extended history)
        df = yf.download(yf_ticker, start=start_date, end=end_date, interval="1m")
        
        if df.empty:
            logger.error("❌ No data fetched from yfinance")
            return pd.DataFrame()
        
        # Reset index to make Date a column if needed
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

def generate_ml_prediction(current_price, historical_data):
    """Generate prediction using the selected ML model - FLEXIBLE VERSION"""
    try:
        # Load the selected ML model
        model_info = load_selected_ml_model()
        if model_info is None:
            logger.warning(f"⚠️ {AI_MODEL_NAME} model not available")
            return None, "Model not available"
        
        model = model_info['model']
        config = model_info['config']
        model_type = model_info['model_type']
        
        # Get required parameters from config
        lags = config['params'].get('lags', 7)
        
        logger.info(f"📊 DEBUG - Using {model_info['display_name']} model")
        logger.info(f"📊 DEBUG - Model type: {model_type}")
        logger.info(f"📊 DEBUG - Required lags: {lags}")
        
        # Prepare data for traditional ML models
        target_ts, covariate_ts, scaler = prepare_traditional_ml_data(historical_data, interval="1m")
        if target_ts is None:
            logger.warning(f"⚠️ Failed to prepare data for {AI_MODEL_NAME}")
            return None, "Data preparation failed"
        
        # Check if we have enough data
        if len(target_ts) < lags:
            logger.warning(f"⚠️ Insufficient data. Need {lags}, have {len(target_ts)}")
            return None, f"Insufficient data ({len(target_ts)} < {lags})"
        
        logger.info(f"📊 DEBUG - Generating prediction with {len(target_ts)} data points...")
        
        # Generate prediction using the selected ML model
        predictions = model.predict(
            n=1,
            series=target_ts,
            past_covariates=covariate_ts
        )
        
        # Inverse transform prediction
        pred_values = predictions.values().reshape(-1, 1)
        inverse_pred = scaler.inverse_transform(pred_values)
        predicted_price = inverse_pred.flatten()[0]
        
        logger.info(f"🤖 {AI_MODEL_NAME} Prediction: ${predicted_price:.4f} (Current: ${current_price:.4f})")
        
        return predicted_price, "Success"
        
    except Exception as e:
        logger.error(f"❌ {AI_MODEL_NAME} prediction failed: {e}")
        logger.error(f"📊 DEBUG - Full traceback: {traceback.format_exc()}")
        return None, str(e)

def validate_signal_with_ai(signal_type, current_price, historical_data):
    """
    Validate buy/sell signal using the selected AI model - FLEXIBLE VERSION
    
    Returns:
        - True: Execute the trade
        - False: Reject the trade
        - execution_reason: Detailed reason for the decision
    """
    global ai_validation_stats
    
    ai_validation_stats['total_signals'] += 1
    
    # Log signal detection
    logger.info(f"🔍 AI VALIDATION: {signal_type.upper()} signal detected at ${current_price:.4f}")
    logger.info(f"🤖 Using {AI_MODEL_NAME} for validation")
    
    if not ENABLE_AI_VALIDATION:
        logger.info(f"⚙️ AI validation disabled, executing traditional signal")
        ai_validation_stats['fallback_executions'] += 1
        return True, "AI validation disabled - Traditional signal executed"
    
    # Generate prediction using selected ML model
    predicted_price, prediction_status = generate_ml_prediction(current_price, historical_data)
    
    if predicted_price is None:
        # AI prediction failed - fallback to traditional signal
        logger.warning(f"⚠️ AI prediction failed: {prediction_status}")
        logger.info(f"🔄 FALLBACK: Executing traditional {signal_type} signal")
        ai_validation_stats['ai_failures'] += 1
        ai_validation_stats['fallback_executions'] += 1
        return True, f"AI failed ({prediction_status}) - Traditional signal executed"
    
    # AI validation logic
    if signal_type.lower() == 'buy':
        # Buy signal: Execute if predicted price > current price
        if predicted_price > current_price:
            price_diff = predicted_price - current_price
            price_diff_pct = (price_diff / current_price) * 100
            
            logger.info(f"✅ {AI_MODEL_NAME} CONFIRMED BUY: Predicted=${predicted_price:.4f} > Current=${current_price:.4f}")
            logger.info(f"📈 Expected gain: ${price_diff:.4f} (+{price_diff_pct:.2f}%)")
            
            ai_validation_stats['confirmed_signals'] += 1
            return True, f"{AI_MODEL_NAME} confirmed - Expected gain: +{price_diff_pct:.2f}%"
        else:
            price_diff = current_price - predicted_price
            price_diff_pct = (price_diff / current_price) * 100
            
            logger.info(f"❌ {AI_MODEL_NAME} REJECTED BUY: Predicted=${predicted_price:.4f} < Current=${current_price:.4f}")
            logger.info(f"📉 Expected loss: ${price_diff:.4f} (-{price_diff_pct:.2f}%)")
            
            ai_validation_stats['rejected_signals'] += 1
            return False, f"{AI_MODEL_NAME} rejected - Expected loss: -{price_diff_pct:.2f}%"
    
    elif signal_type.lower() == 'sell':
        # Sell signal: Execute if predicted price < current price
        if predicted_price < current_price:
            price_diff = current_price - predicted_price
            price_diff_pct = (price_diff / current_price) * 100
            
            logger.info(f"✅ {AI_MODEL_NAME} CONFIRMED SELL: Predicted=${predicted_price:.4f} < Current=${current_price:.4f}")
            logger.info(f"📉 Expected decline: ${price_diff:.4f} (-{price_diff_pct:.2f}%)")
            
            ai_validation_stats['confirmed_signals'] += 1
            return True, f"{AI_MODEL_NAME} confirmed - Expected decline: -{price_diff_pct:.2f}%"
        else:
            price_diff = predicted_price - current_price
            price_diff_pct = (price_diff / current_price) * 100
            
            logger.info(f"❌ {AI_MODEL_NAME} REJECTED SELL: Predicted=${predicted_price:.4f} > Current=${current_price:.4f}")
            logger.info(f"📈 Expected rise: ${price_diff:.4f} (+{price_diff_pct:.2f}%)")
            
            ai_validation_stats['rejected_signals'] += 1
            return False, f"{AI_MODEL_NAME} rejected - Expected rise: +{price_diff_pct:.2f}%"
    
    # Should not reach here
    logger.error(f"❌ Unknown signal type: {signal_type}")
    ai_validation_stats['fallback_executions'] += 1
    return True, f"Unknown signal type - Traditional signal executed"

def get_historical_data():
    """Fetch historical data using Yahoo Finance (no subscription required)"""
    try:
        # Calculate dates for minute data
        end_date = datetime.now()
        start_date = end_date - timedelta(hours=24)  # Get 24 hours of minute data
        
        # Fetch data using yfinance (free and reliable) with crypto ticker format
        df = yf.download(YFINANCE_TICKER, start=start_date, end=end_date, interval="1m")
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
    
        if df.empty:
            logger.error("❌ No data fetched from Yahoo Finance")
            return pd.DataFrame()
        
        # Reset index to make Date a column
        df = df.reset_index()
        
        # Rename columns to match expected format
        df = df.rename(columns={
            'Datetime': 'Date',  # For minute data
            'Open': 'Open',
            'High': 'High',
            'Low': 'Low',
            'Close': 'Close',
            'Volume': 'Volume'
        })
        
        # Handle missing values
        df = df.fillna(method='ffill').fillna(method='bfill')
        df = df.dropna()
        
        logger.info(f"✅ Fetched {len(df)} minute bars for {TICKER} from Yahoo Finance")
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
            if position.symbol == TICKER:  # Changed to use TICKER variable (CRVUSD)
                return float(position.qty)
        return 0
    except Exception as e:
        logger.error(f"Error getting position: {e}")
        return 0

def execute_trade(side, price, execution_reason="Traditional signal"):
    """Execute a trade on Alpaca with AI validation tracking"""
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
            symbol="CRVUSD",  
            qty=quantity,
            side=side,
            time_in_force=TimeInForce.GTC
        )
        
        # Submit order
        order = trading_client.submit_order(order_data)
        
        # Record the trade in history with AI validation info
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
        logger.info(f"🤖 AI-ENHANCED TRADE EXECUTED - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Type           : {side.name}")
        logger.info(f"Symbol         : {TICKER}")
        logger.info(f"Shares         : {quantity:.6f}")
        logger.info(f"Price          : ${price:.2f}")
        logger.info(f"Total Value    : ${quantity * price:.2f}")
        logger.info(f"Cash Left      : ${cash_left:.2f}")
        logger.info(f"Order ID       : {order.id}")
        logger.info(f"AI Validation  : {execution_reason}")
        print(f"{'=' * 60}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error executing trade: {e}")
        return False

def print_bot_header():
    """Print a nicely formatted header when the bot starts - UPDATED"""
    header = f"""
{'=' * 60}
{Fore.CYAN}
 🤖 AI-ENHANCED ZERO LAG MACD TRADING BOT 🤖
 
 ██╗     ███████╗████████╗███╗   ███╗    ███╗   ███╗ █████╗  ██████╗██████╗ 
 ██║     ██╔════╝╚══██╔══╝████╗ ████║    ████╗ ████║██╔══██╗██╔════╝██╔══██╗
 ██║     ███████╗   ██║   ██╔████╔██║    ██╔████╔██║███████║██║     ██║  ██║
 ██║     ╚════██║   ██║   ██║╚██╔╝██║    ██║╚██╔╝██║██╔══██║██║     ██║  ██║
 ███████╗███████║   ██║   ██║ ╚═╝ ██║    ██║ ╚═╝ ██║██║  ██║╚██████╗██████╔╝
 ╚══════╝╚══════╝   ╚═╝   ╚═╝     ╚═╝    ╚═╝     ╚═╝╚═╝  ╚═╝ ╚═════╝╚═════╝ 
{Style.RESET_ALL}
{'=' * 60}
"""
    print(header)
    
    logger.info(f"🤖 STRATEGY       : AI-Enhanced Zero Lag MACD Trading Bot")
    logger.info(f"📊 SYMBOL         : {TICKER} (CRV/USD)")
    logger.info(f"⏰ TIMEFRAME      : {INTERVAL}")
    logger.info(f"⚡ FAST LENGTH    : {FAST_LENGTH}")
    logger.info(f"🐌 SLOW LENGTH    : {SLOW_LENGTH}")
    logger.info(f"📡 SIGNAL LENGTH  : {SIGNAL_LENGTH}")
    logger.info(f"🤖 AI MODEL       : {AI_MODEL_NAME} ({SELECTED_ML_MODEL})")
    logger.info(f"✅ AI VALIDATION  : {'ENABLED' if ENABLE_AI_VALIDATION else 'DISABLED'}")
    logger.info(f"📈 TRADING MODE   : Paper Trading (Alpaca)")
    logger.info(f"💰 ASSET TYPE     : Cryptocurrency")
    
    # Show available models
    available_models = list(ML_MODEL_CONFIG.keys())
    logger.info(f"📋 AVAILABLE MODELS: {', '.join(available_models)}")

def list_available_models():
    """Utility function to show available models"""
    print(f"{'=' * 60}")
    print(f"📋 AVAILABLE ML MODELS:")
    for key, config in ML_MODEL_CONFIG.items():
        status = "✅ ACTIVE" if key == SELECTED_ML_MODEL else "⚪ Available"
        print(f"   {config['display_name']} ({key}) - {status}")
    print(f"{'=' * 60}")
    print(f"💡 To switch models, change SELECTED_ML_MODEL = '{SELECTED_ML_MODEL}' in the config")
    print(f"{'=' * 60}")

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
        # Format date (include time for minute data)
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
    logger.info(f"🤖 AI-ENHANCED SIGNAL UPDATE #{signal_count} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Format datetime for minute data
    if isinstance(latest_bar['Date'], pd.Timestamp):
        date_str = latest_bar['Date'].strftime('%Y-%m-%d %H:%M:%S')
    else:
        date_str = str(latest_bar['Date'])
    
    logger.info(f"Date Time      : {date_str}")
    logger.info(f"Price          : ${current_price:.4f}")  # More decimals for crypto
    logger.info(f"ZL MACD        : {latest_bar['ZL_MACD']:.4f}")
    logger.info(f"Signal Line    : {latest_bar['Signal']:.4f}")
    logger.info(f"Histogram      : {latest_bar['Histogram']:.4f}")
    logger.info(f"Position       : {current_position} CRV")  # Changed to CRV
    
    # Use different colored signals based on buy/sell
    if latest_bar['Buy']:
        logger.info(f"{Fore.CYAN}SIGNAL         : BUY 🔵{Style.RESET_ALL}")
    elif latest_bar['Sell']:
        logger.info(f"{Fore.MAGENTA}SIGNAL         : SELL 🔴{Style.RESET_ALL}")
    else:
        logger.info(f"{Fore.YELLOW}SIGNAL         : NO SIGNAL ⚪{Style.RESET_ALL}")

def print_ai_validation_stats():
    """Print AI validation statistics"""
    stats = ai_validation_stats
    print(f"{'=' * 60}")
    logger.info(f"🤖 AI VALIDATION STATISTICS")
    logger.info(f"Total Signals      : {stats['total_signals']}")
    logger.info(f"AI Confirmed       : {stats['confirmed_signals']}")
    logger.info(f"AI Rejected        : {stats['rejected_signals']}")
    logger.info(f"AI Failures        : {stats['ai_failures']}")
    logger.info(f"Fallback Executions: {stats['fallback_executions']}")
    
    if stats['total_signals'] > 0:
        confirmation_rate = (stats['confirmed_signals'] / stats['total_signals']) * 100
        rejection_rate = (stats['rejected_signals'] / stats['total_signals']) * 100
        failure_rate = (stats['ai_failures'] / stats['total_signals']) * 100
        
        logger.info(f"Confirmation Rate  : {confirmation_rate:.1f}%")
        logger.info(f"Rejection Rate     : {rejection_rate:.1f}%")
        logger.info(f"AI Failure Rate    : {failure_rate:.1f}%")
    
    print(f"{'=' * 60}")

def print_waiting_message(next_check_time):
    """Print a clear waiting message to show bot is active and waiting for next check"""
    print(f"{'=' * 60}")
    logger.info(f"{Fore.BLUE}🤖 AI BOT STATUS   : MONITORING {TICKER}{Style.RESET_ALL}")
    logger.info(f"Current Time   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Next Check     : {next_check_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Calculate time remaining
    time_remaining = next_check_time - datetime.now()
    seconds = time_remaining.seconds
    
    logger.info(f"Time Remaining : {seconds:.0f} seconds")
    
    # Show current market status - crypto markets are 24/7
    logger.info(f"{Fore.GREEN}Market Status  : 24/7 Crypto Trading Active{Style.RESET_ALL}")
    
    logger.info(f"🤖 AI Model      : {AI_MODEL_NAME} ({'ACTIVE' if ENABLE_AI_VALIDATION else 'INACTIVE'})")
    print(f"{'=' * 60}")

def run_live_trading():
    """Run the AI-Enhanced Zero Lag MACD strategy continuously in live trading mode"""
    # Setup custom logging
    global logger
    logger = setup_logging()
    
    # Print the bot header and available models
    print_bot_header()
    list_available_models()
    
    # Test model loading on startup
    logger.info(f"🔍 Testing {AI_MODEL_NAME} model loading...")
    test_model = load_selected_ml_model()
    if test_model:
        logger.info(f"✅ {AI_MODEL_NAME} model ready for use")
    else:
        logger.warning(f"⚠️ {AI_MODEL_NAME} model failed to load - will use fallback")
    
    last_check_minute = None
    signal_count = 0  
    waiting_message_shown = False

    # Print instructions for stopping the bot
    logger.info(f"{Fore.YELLOW}To stop the bot and view AI validation summary, hold the 'q' key for a few seconds.{Style.RESET_ALL}")
        
    try:
        while True:
            try:
                # Check if user pressed 'q' to quit
                if keyboard.is_pressed('q'):
                    logger.info("Bot stopped by user. Displaying AI validation summary...")
                    print_ai_validation_stats()
                    calculate_roi_summary()
                    break
                
                # Get current minute
                current_datetime = datetime.now()
                current_minute = current_datetime.replace(second=0, microsecond=0)

                # Only process once per minute for 1-minute timeframe
                if last_check_minute != current_minute:
                    # Reset waiting message flag when starting a new minute check
                    waiting_message_shown = False

                    df = get_historical_data() # Get historical data
                    signals_df = calculate_signals(df) # Calculate signals
                    latest_bar = signals_df.iloc[-1] # Check latest signal (most recent bar)
                    current_price = latest_bar['Close']
                    
                    current_position = get_current_position() # Get current position

                    signal_count += 1 # Increment signal counter
                    print_signal_update(latest_bar, current_price, current_position, signal_count)
                    
                    # Get extended historical data for AI validation
                    historical_data = fetch_and_prepare_data_for_ai(TICKER, minutes_back=1440)
                    
                    # Process buy signal with AI validation
                    if latest_bar['Buy']:
                        if current_position > 0:
                            logger.info(f"Already holding {current_position} CRV, no action taken.")
                        else:
                            # Validate with AI
                            should_execute, execution_reason = validate_signal_with_ai(
                                'buy', current_price, historical_data
                            )
                            
                            if should_execute:
                                logger.info(f"🤖 AI VALIDATION PASSED: Opening new long position at ${current_price:.4f}")
                                execute_trade(OrderSide.BUY, current_price, execution_reason)
                            else:
                                logger.info(f"🤖 AI VALIDATION FAILED: Buy signal rejected - {execution_reason}")
                    
                    # Process sell signal with AI validation
                    elif latest_bar['Sell']:
                        if current_position > 0:
                            # Validate with AI
                            should_execute, execution_reason = validate_signal_with_ai(
                                'sell', current_price, historical_data
                            )
                            
                            if should_execute:
                                logger.info(f"🤖 AI VALIDATION PASSED: Selling {current_position} CRV at ${current_price:.4f}")
                                execute_trade(OrderSide.SELL, current_price, execution_reason)
                            else:
                                logger.info(f"🤖 AI VALIDATION FAILED: Sell signal rejected - {execution_reason}")
                        else:
                            logger.info("No position to sell.")
                    
                    else:
                        logger.info("No new trading signals this minute")
                    
                    # Record that we checked this minute
                    last_check_minute = current_minute
                            
                # Calculate next check time (next minute)
                next_check_time = current_datetime + timedelta(minutes=1)

                # Show waiting message only once after processing signals and if not already shown
                if last_check_minute == current_minute and not waiting_message_shown:
                    print_waiting_message(next_check_time)
                    waiting_message_shown = True  
                
                time.sleep(1)  # Check every second (more frequent for 1-min strategy)
                
            except Exception as e:
                logger.error(f"Error in live trading execution: {e}")
                logger.error(f"{'=' * 60}")
                time.sleep(10)  # Shorter retry time for 1-min strategy

    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        logger.info("Bot stopped by user. Displaying AI validation summary...")
        print_ai_validation_stats()
        calculate_roi_summary()

def run_signal_test():
    """Run a test to print all historical signals without executing trades"""
    # Setup custom logging
    global logger
    logger = setup_logging()
    
    # Print the bot header
    print_bot_header()
    list_available_models()
    logger.info("🤖 RUNNING IN AI SIGNAL TEST MODE - NO TRADES WILL BE EXECUTED")
    
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
    logger.info(f"🤖 AI SIGNAL SUMMARY")
    logger.info(f"Total bars     : {len(signals_df)}")
    logger.info(f"Buy signals    : {buy_signals} ({buy_signals/len(signals_df)*100:.2f}%)")
    logger.info(f"Sell signals   : {sell_signals} ({sell_signals/len(signals_df)*100:.2f}%)")
    logger.info(f"No signals     : {len(signals_df) - buy_signals - sell_signals} ({(len(signals_df) - buy_signals - sell_signals)/len(signals_df)*100:.2f}%)")
    print(f"{'=' * 60}")

def calculate_roi_summary():
    """Calculate and print ROI summary for each 30-minute interval"""
    if not trade_history:
        logger.info("No trades executed. ROI summary not available.")
        return
    
    # Group trades by 30-minute intervals
    for trade in trade_history:
        # Create a 30-minute interval key 
        interval_time = trade['timestamp'].replace(
            minute=30 * (trade['timestamp'].minute // 30),
            second=0,
            microsecond=0
        )
        interval_key = interval_time.strftime('%Y-%m-%d %H:%M')
        
        if interval_key not in performance_metrics:
            performance_metrics[interval_key] = {
                'initial_cash': trade['cash_left'] + (trade['quantity'] * trade['price'] if trade['side'] == OrderSide.SELL else trade['cash_left']),
                'final_cash': trade['cash_left'],
                'trades': []
            }
        performance_metrics[interval_key]['trades'].append(trade)
        performance_metrics[interval_key]['final_cash'] = trade['cash_left']
    
    # Calculate ROI for each 30-minute interval
    interval_counter = 1
    print(f"{'=' * 60}")
    logger.info(f"{Fore.CYAN}🤖 AI-ENHANCED PERFORMANCE SUMMARY BY 30-MINUTE INTERVALS{Style.RESET_ALL}")
    
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
            
        logger.info(f"Interval {interval_counter}: {interval}")
        logger.info(f"Initial Cash   : ${initial_cash:.2f}")
        logger.info(f"Final Cash     : ${final_cash:.2f}")
        logger.info(f"ROI            : {roi_color}{roi:+.2f}%{Style.RESET_ALL}")
        logger.info(f"Number of Trades: {len(data['trades'])}")
        
        # Show AI validation info for trades in this interval
        ai_trades = [t for t in data['trades'] if t.get('ai_validation', False)]
        if ai_trades:
            logger.info(f"AI-Enhanced Trades: {len(ai_trades)}")
        
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
                logger.info(f"Open position  : {current_position} CRV valued at ${position_value:.2f}")
            except Exception as e:
                logger.error(f"Error getting current position value: {e}")
                overall_final = last_trade['cash_left']
        else:
            overall_final = last_trade['cash_left']
            
        overall_roi = ((overall_final - overall_initial) / overall_initial) * 100
        
        logger.info(f"{Fore.CYAN}🤖 OVERALL AI-ENHANCED PERFORMANCE{Style.RESET_ALL}")
        logger.info(f"Starting Capital: ${overall_initial:.2f}")
        logger.info(f"Current Capital : ${overall_final:.2f}")
        
        if overall_roi > 0:
            logger.info(f"Total ROI       : {Fore.GREEN}+{overall_roi:.2f}%{Style.RESET_ALL}")
        elif overall_roi < 0:
            logger.info(f"Total ROI       : {Fore.RED}{overall_roi:.2f}%{Style.RESET_ALL}")
        else:
            logger.info(f"Total ROI       : {Fore.YELLOW}{overall_roi:.2f}%{Style.RESET_ALL}")
            
        logger.info(f"Total Trades    : {len(trade_history)}")
        
        # Count AI-enhanced trades
        ai_enhanced_trades = [t for t in trade_history if t.get('ai_validation', False)]
        logger.info(f"AI-Enhanced Trades: {len(ai_enhanced_trades)}")
    
    print(f"{'=' * 60}")

if __name__ == "__main__":
    #run_signal_test()  # Use this to test signals without trading
    run_live_trading()  # Use this for actual trading