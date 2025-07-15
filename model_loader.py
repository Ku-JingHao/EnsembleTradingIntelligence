import pickle
import os
import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.models import RNNModel
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import warnings
import sys
warnings.filterwarnings('ignore')

# FIX: Define MetricsCallback locally BEFORE any imports or loading
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

# Create a custom pickle loader
class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'MetricsCallback':
            return MetricsCallback
        return super().find_class(module, name)

class ModelLoader:
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        self.models = {}
        self.deployment_info = None
        self.load_deployment_info()
    
    def load_deployment_info(self):
        """Load deployment information with dynamic naming"""
        try:
            # Try to load deployment info for specific stock/interval combination
            # For now, we'll try the first available deployment info file
            deployment_files = [f for f in os.listdir(self.models_dir) if f.startswith('deployment_info_') and f.endswith('.pkl')]
            
            if deployment_files:
                # Use the first available deployment info file
                deployment_file = deployment_files[0]
                with open(f'{self.models_dir}/{deployment_file}', 'rb') as f:
                    unpickler = CustomUnpickler(f)
                    self.deployment_info = unpickler.load()
                print(f"✅ Deployment info loaded from {deployment_file}. Status: {self.deployment_info['status']}")
            else:
                # Fallback to generic deployment info
                with open(f'{self.models_dir}/deployment_info.pkl', 'rb') as f:
                    unpickler = CustomUnpickler(f)
                    self.deployment_info = unpickler.load()
                print(f"✅ Deployment info loaded. Status: {self.deployment_info['status']}")
        except FileNotFoundError:
            print("❌ No deployment info found. Please run model deployment first.")
            self.deployment_info = None
        except Exception as e:
            print(f"⚠️ Error loading deployment info: {e}")
            self.deployment_info = None
    
    def load_model(self, model_name, stock_name=None, interval=None):
        """Load a specific model with dynamic naming based on model_name, stock_name, and interval"""
        # Create cache key for this specific model combination
        cache_key = f"{model_name}_{stock_name}_{interval}" if stock_name and interval else model_name
        
        if cache_key in self.models:
            print(f"✅ Model {model_name} for {stock_name}_{interval} loaded from cache")
            return self.models[cache_key]
        
        try:
            # Handle ensemble separately (only has config, no model file)
            if model_name == 'ensemble':
                print(f"🤖 Loading Ensemble model (Linear Regression + LSTM) for {stock_name}_{interval}...")
                
                # Load Linear Regression model
                lr_model_info = self.load_model('linear_regression', stock_name=stock_name, interval=interval)
                if lr_model_info is None:
                    print(f"❌ Failed to load Linear Regression model for ensemble")
                    return None
                
                # Load LSTM model
                lstm_model_info = self.load_model('lstm', stock_name=stock_name, interval=interval)
                if lstm_model_info is None:
                    print(f"❌ Failed to load LSTM model for ensemble")
                    return None
                
                # Create ensemble config
                ensemble_config = {
                    'type': 'ensemble',
                    'name': 'LR + LSTM',
                    'models': ['linear_regression', 'lstm'],
                    'weights': [0.5, 0.5],  # Equal weights
                    'stock_name': stock_name,
                    'interval': interval
                }
                
                self.models[cache_key] = {
                    'model': None,  # Ensemble has no single model
                    'config': ensemble_config,
                    'lr_model': lr_model_info['model'],
                    'lstm_model': lstm_model_info['model'],
                    'lr_config': lr_model_info['config'],
                    'lstm_config': lstm_model_info['config']
                }
                
                print(f"✅ Ensemble model (LR + LSTM) loaded successfully for {stock_name}_{interval}")
                print(f"   📊 Linear Regression: {lr_model_info['config']['name']}")
                print(f"   🧠 LSTM: {lstm_model_info['config']['name']}")
                return self.models[cache_key]
            
            # For LSTM and GRU - Use .pt files with Darts native load
            if model_name in ['lstm', 'gru']:
                print(f"🧠 Loading {model_name.upper()} model for {stock_name}_{interval}...")
                
                # Dynamic model path based on stock_name and interval
                if stock_name and interval:
                    model_path = f'{self.models_dir}/{model_name}_{stock_name}_{interval}_model.pt'
                else:
                    # Fallback to old naming convention
                    model_path = f'{self.models_dir}/{model_name}_model.pt'
                
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Model file {model_path} not found")
                
                try:
                    # Load using Darts native method - includes all configuration
                    model = RNNModel.load(model_path)
                    print(f"✅ {model_name.upper()} model loaded successfully from: {model_path}")
                    
                    # Create a minimal config for compatibility
                    config = {
                        'type': 'deep_learning',
                        'name': model_name.upper(),
                        'save_method': 'darts_native',
                        'file_extension': '.pt',
                        'params': {
                            'input_chunk_length': model.input_chunk_length,
                            'output_chunk_length': model.output_chunk_length
                        }
                    }
                    
                    self.models[cache_key] = {
                        'model': model,
                        'config': config
                    }
                    
                    return self.models[cache_key]
                    
                except Exception as e:
                    print(f"❌ Failed to load {model_name} with Darts native method: {e}")
                    raise e
            
            # For traditional ML models - load model + separate config
            else:
                print(f"📊 Loading {model_name} model for {stock_name}_{interval}...")
                
                # Dynamic config and model paths based on stock_name and interval
                if stock_name and interval:
                    # Handle special case for linear_regression vs linear_regression
                    if model_name == 'linear_regression':
                        # Check for both naming conventions
                        config_path1 = f'{self.models_dir}/linear_regression_{stock_name}_{interval}_config.pkl'
                        config_path2 = f'{self.models_dir}/linear_{stock_name}_{interval}_regression_config.pkl'
                        model_path1 = f'{self.models_dir}/linear_regression_{stock_name}_{interval}_model.pkl'
                        model_path2 = f'{self.models_dir}/linear_{stock_name}_{interval}_regression_model.pkl'
                        
                        # Try the standard naming first
                        if os.path.exists(config_path1) and os.path.exists(model_path1):
                            config_path = config_path1
                            model_path = model_path1
                        elif os.path.exists(config_path2) and os.path.exists(model_path2):
                            config_path = config_path2
                            model_path = model_path2
                        else:
                            raise FileNotFoundError(f"Neither {config_path1} nor {config_path2} found")
                    else:
                        config_path = f'{self.models_dir}/{model_name}_{stock_name}_{interval}_config.pkl'
                        model_path = f'{self.models_dir}/{model_name}_{stock_name}_{interval}_model.pkl'
                else:
                    # Fallback to old naming convention
                    config_path = f'{self.models_dir}/{model_name}_config.pkl'
                    model_path = f'{self.models_dir}/{model_name}_model.pkl'
                
                # Load config first
                with open(config_path, 'rb') as f:
                    unpickler = CustomUnpickler(f)
                    config = unpickler.load()
                
                # Load model using pickle
                with open(model_path, 'rb') as f:
                    unpickler = CustomUnpickler(f)
                    model = unpickler.load()
                
                self.models[cache_key] = {
                    'model': model,
                    'config': config
                }
                
                print(f"✅ {config['name']} model loaded successfully from: {model_path}")
                return self.models[cache_key]
            
        except FileNotFoundError as e:
            print(f"❌ Model {model_name} not found: {str(e)}")
            return None
        except Exception as e:
            print(f"❌ Error loading model {model_name}: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            return None
    
    def get_available_models(self):
        """Get list of available models"""
        if self.deployment_info:
            return self.deployment_info['models']
        else:
            # Scan directory for model files
            models = []
            
            # Check for traditional ML models (have config files)
            for file in os.listdir(self.models_dir):
                if file.endswith('_config.pkl') and not file.startswith('ensemble'):
                    model_name = file.replace('_config.pkl', '')
                    models.append(model_name)
            
            # Check for deep learning models (.pt files)
            for file in os.listdir(self.models_dir):
                if file.endswith('_model.pt'):
                    model_name = file.replace('_model.pt', '')
                    models.append(model_name)
            
            # Check for ensemble
            if 'ensemble_config.pkl' in os.listdir(self.models_dir):
                models.append('ensemble')
            
            return list(set(models))  # Remove duplicates

    def generate_predictions(self, model_name, ticker, start_date, end_date, interval="1d", forecast_horizon=1):
        """Generate predictions using specified model with dynamic naming"""
        try:
            # Load model with dynamic naming
            model_info = self.load_model(model_name, stock_name=ticker, interval=interval)
            if model_info is None:
                return {
                    'predictions': None,
                    'timestamps': None,
                    'model_used': model_name,
                    'success': False,
                    'error': f'Failed to load model {model_name} for {ticker}_{interval}'
                }
            
            # Handle ensemble model separately
            if model_name == 'ensemble':
                return self.generate_ensemble_predictions_new(ticker, start_date, end_date, interval, forecast_horizon, model_info)
            
            # Fetch and prepare data
            data = self.fetch_and_prepare_data(ticker, start_date, end_date, interval)
            if data.empty:
                return {
                    'predictions': None,
                    'timestamps': None,
                    'model_used': model_name,
                    'success': False,
                    'error': 'No data fetched'
                }
            
            # Prepare data differently for deep learning vs traditional ML
            model = model_info['model']
            config = model_info['config']
            
            if config['type'] == 'deep_learning':
                # For LSTM/GRU - use simpler data preparation without past_covariates
                target_ts, scaler = self.prepare_deep_learning_data(data, interval)
                if target_ts is None:
                    return {
                        'predictions': None,
                        'timestamps': None,
                        'model_used': model_name,
                        'success': False,
                        'error': 'Failed to prepare deep learning data'
                    }
                
                input_chunk_length = config['params'].get('input_chunk_length', 7)
                if len(target_ts) < input_chunk_length:
                    raise ValueError(f"Insufficient data. Need at least {input_chunk_length} data points.")
                
                # Generate predictions using only target series (no past_covariates)
                predictions = model.predict(
                    n=forecast_horizon,
                    series=target_ts
                )
                
            else:
                # For traditional ML models
                target_ts, covariate_ts, scaler = self.prepare_traditional_ml_data(data, model_name, interval, ticker)
                if target_ts is None:
                    return {
                        'predictions': None,
                        'timestamps': None,
                        'model_used': model_name,
                        'success': False,
                        'error': 'Failed to prepare traditional ML data'
                    }
                
                lags = config['params']['lags']
                if len(target_ts) < lags:
                    raise ValueError(f"Insufficient data. Need at least {lags} data points.")
                
                predictions = model.predict(
                    n=forecast_horizon,
                    series=target_ts,
                    past_covariates=covariate_ts
                )
            
            # Inverse transform predictions
            pred_values = predictions.values().reshape(-1, 1)
            inverse_pred = scaler.inverse_transform(pred_values)
            
            return {
                'predictions': inverse_pred.flatten(),
                'timestamps': predictions.time_index,
                'model_used': config['name'],
                'success': True,
                'raw_predictions': predictions
            }
            
        except Exception as e:
            print(f"❌ Error generating predictions with {model_name}: {str(e)}")
            return {
                'predictions': None,
                'timestamps': None,
                'model_used': model_name,
                'success': False,
                'error': str(e)
            }
    
    def prepare_deep_learning_data(self, data, interval="1d"):
        """Prepare data specifically for deep learning models (LSTM/GRU) with ROBUST NaN handling"""
        try:
            # Create a new scaler for the target data
            scaler = MinMaxScaler()
            target_data = data[['Close']].copy()
            
            print(f"📊 Data preparation: {len(target_data)} points, NaN count: {target_data.isnull().sum().sum()}")
            
            # 🔥 STEP 1: Handle NaN values before scaling
            # Forward fill NaN values first, then backward fill any remaining
            target_data = target_data.fillna(method='ffill').fillna(method='bfill')
            
            # Drop any remaining NaN rows (if entire rows are NaN)
            target_data = target_data.dropna()
            
            if target_data.empty:
                print("❌ No valid data after NaN handling")
                return None, None
            
            print(f"✅ After NaN handling: {len(target_data)} points")
            
            # 🔥 STEP 2: Check for constant/zero variance data
            close_values = target_data['Close'].values
            
            # Check if all values are identical (zero variance)
            if len(np.unique(close_values)) == 1:
                print(f"🚨 CRITICAL: All values are identical ({close_values[0]:.6f})")
                print("🔧 Adding tiny noise to prevent zero variance...")
                # Add minimal noise to prevent zero variance
                noise = np.random.normal(0, abs(close_values[0]) * 1e-8, len(close_values))
                close_values = close_values + noise
                target_data['Close'] = close_values
            
            # 🔥 STEP 3: Check for infinite values
            if np.isinf(close_values).any():
                print(f"🚨 CRITICAL: Infinite values detected!")
                # Replace infinite values with boundary values
                finite_mask = np.isfinite(close_values)
                if finite_mask.any():
                    finite_values = close_values[finite_mask]
                    close_values[~finite_mask] = np.median(finite_values)
                    target_data['Close'] = close_values
                    print(f"✅ Replaced infinite values with median: {np.median(finite_values):.6f}")
                else:
                    print("❌ All values are infinite, cannot proceed")
                    return None, None
            
            # 🔥 STEP 4: Check variance before scaling
            variance = np.var(close_values)
            print(f"📊 Price variance: {variance:.8f}")
            
            if variance < 1e-10:  # Extremely low variance
                print(f"🚨 WARNING: Very low variance detected ({variance:.2e})")
                print("🔧 Adding proportional noise to increase variance...")
                mean_price = np.mean(close_values)
                noise_scale = max(abs(mean_price) * 1e-6, 1e-6)  # At least 1e-6 noise
                noise = np.random.normal(0, noise_scale, len(close_values))
                close_values = close_values + noise
                target_data['Close'] = close_values
                print(f"✅ Variance after noise: {np.var(close_values):.8f}")
            
            # 🔥 STEP 5: Safe scaling with validation
            try:
                print(f"📊 Pre-scaling stats:")
                print(f"    Min: {close_values.min():.6f}, Max: {close_values.max():.6f}")
                print(f"    Mean: {close_values.mean():.6f}, Std: {close_values.std():.6f}")
                
                # Fit and transform the scaler
                scaled_values = scaler.fit_transform(target_data[['Close']])
                
                # 🔥 STEP 6: Validate scaled output
                if np.isnan(scaled_values).any():
                    print(f"🚨 CRITICAL: Scaler produced NaN values!")
                    print(f"📊 Scaler info:")
                    print(f"    Scale: {scaler.scale_[0]:.10f}")
                    print(f"    Min: {scaler.min_[0]:.10f}")
                    print(f"    Data min: {scaler.data_min_[0]:.6f}")
                    print(f"    Data max: {scaler.data_max_[0]:.6f}")
                    
                    # Try manual scaling as fallback
                    print("🔧 Attempting manual scaling fallback...")
                    data_min = close_values.min()
                    data_max = close_values.max()
                    data_range = data_max - data_min
                    
                    if data_range == 0:
                        # All values are the same after noise, use simple normalization
                        scaled_values = np.full_like(close_values, 0.5).reshape(-1, 1)
                        print("✅ Used constant scaling (0.5) for identical values")
                    else:
                        scaled_values = ((close_values - data_min) / data_range).reshape(-1, 1)
                        print(f"✅ Manual scaling successful, range: {scaled_values.min():.6f} to {scaled_values.max():.6f}")
                    
                    # Create a manual scaler object for inverse transform
                    class ManualScaler:
                        def __init__(self, data_min, data_max):
                            self.data_min_ = np.array([data_min])
                            self.data_max_ = np.array([data_max])
                            self.scale_ = np.array([1.0 / (data_max - data_min) if data_max != data_min else 1.0])
                            self.min_ = np.array([-data_min / (data_max - data_min) if data_max != data_min else 0.0])
                        
                        def inverse_transform(self, X):
                            if self.data_max_[0] == self.data_min_[0]:
                                return np.full_like(X, self.data_min_[0])
                            return X * (self.data_max_[0] - self.data_min_[0]) + self.data_min_[0]
                    
                    scaler = ManualScaler(data_min, data_max)
                
                if np.isinf(scaled_values).any():
                    print(f"🚨 CRITICAL: Scaler produced infinite values!")
                    return None, None
                
                print(f"✅ Scaling successful:")
                print(f"    Scaled range: {scaled_values.min():.6f} to {scaled_values.max():.6f}")
                print(f"    Scaled mean: {scaled_values.mean():.6f}")
                print(f"    Scaled std: {scaled_values.std():.6f}")
                
            except Exception as scaling_error:
                print(f"❌ Scaling failed: {scaling_error}")
                return None, None
            
            # 🔥 STEP 7: Create scaled DataFrame
            scaled_target = pd.DataFrame(
                scaled_values,
                columns=['Close'],
                index=target_data.index
            )
            
            # 🔥 STEP 8: Convert to TimeSeries with validation
            try:
                freq_map = {"1h": "H", "1d": "B"}  # REMOVED: "15m": "15T", "30m": "30T"
                freq = freq_map.get(interval, "B")
                
                target_ts = TimeSeries.from_dataframe(scaled_target, freq=freq)
                
                # Final validation of TimeSeries
                ts_values = target_ts.values()
                if np.isnan(ts_values).any() or np.isinf(ts_values).any():
                    print(f"🚨 CRITICAL: TimeSeries contains invalid values!")
                    return None, None
                
                print(f"✅ TimeSeries created successfully:")
                print(f"    Length: {len(target_ts)}")
                print(f"    Value range: {ts_values.min():.6f} to {ts_values.max():.6f}")
                
                return target_ts, scaler
                
            except Exception as ts_error:
                print(f"❌ TimeSeries creation failed: {ts_error}")
                return None, None
                
        except Exception as e:
            print(f"❌ Error in data preparation: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def prepare_traditional_ml_data(self, data, model_name, interval="1d", stock_name=None):
        """Prepare data for traditional ML models with ROBUST NaN handling"""
        try:
            model_info = self.load_model(model_name, stock_name=stock_name, interval=interval)
            if not model_info:
                return None, None, None
            
            # For traditional ML models, try to get scaler from config
            scaler = model_info['config'].get('scaler', None)
            
            # 🔥 FIX: Handle NaN values before any processing
            clean_data = data.copy()
            print(f"📊 Data preparation: {len(clean_data)} points, NaN count: {clean_data.isnull().sum().sum()}")
            
            # Forward fill NaN values first, then backward fill any remaining
            clean_data = clean_data.fillna(method='ffill').fillna(method='bfill')
            
            # Drop any remaining NaN rows
            clean_data = clean_data.dropna()
            
            if clean_data.empty:
                print("❌ No valid data after NaN handling")
                return None, None, None
            
            print(f"✅ After NaN handling: {len(clean_data)} points")
            
            # Prepare target and covariates from clean data
            target_data = clean_data[['Close']]
            covariate_data = clean_data[['Open', 'High', 'Low', 'Volume']]
            
            # 🔥 STEP 2: Check for constant values in target
            close_values = target_data['Close'].values
            if len(np.unique(close_values)) == 1:
                print(f"🚨 WARNING: All Close values are identical ({close_values[0]:.6f})")
                print("🔧 Adding tiny noise to prevent zero variance...")
                noise = np.random.normal(0, abs(close_values[0]) * 1e-8, len(close_values))
                target_data['Close'] = close_values + noise
            
            # Create or use existing scaler
            if scaler is None:
                print(f"Warning: No scaler found for {model_name}, creating new one")
                scaler = MinMaxScaler()
                scaler.fit(target_data)
            
            # 🔥 STEP 3: Safe scaling with validation
            try:
                # Scale the target data
                scaled_target_values = scaler.transform(target_data)
                
                # Check for NaN in scaled target
                if np.isnan(scaled_target_values).any():
                    print(f"🚨 CRITICAL: Target scaling produced NaN values!")
                    # Manual scaling fallback for target
                    data_min = target_data['Close'].min()
                    data_max = target_data['Close'].max()
                    if data_max == data_min:
                        scaled_target_values = np.full_like(target_data['Close'].values, 0.5).reshape(-1, 1)
                    else:
                        scaled_target_values = ((target_data['Close'].values - data_min) / (data_max - data_min)).reshape(-1, 1)
                    print("✅ Used manual scaling for target")
                
                scaled_target = pd.DataFrame(
                    scaled_target_values,
                    columns=['Close'],
                    index=target_data.index
                )
                
            except Exception as target_scaling_error:
                print(f"❌ Target scaling failed: {target_scaling_error}")
                return None, None, None
            
            # 🔥 STEP 4: Safe covariate scaling
            try:
                # For covariates, create a simple scaler if not available
                covariate_scaler = MinMaxScaler()
                covariate_scaler.fit(covariate_data)
                
                scaled_covariate_values = covariate_scaler.transform(covariate_data)
                
                # Check for NaN in scaled covariates
                if np.isnan(scaled_covariate_values).any():
                    print(f"🚨 WARNING: Covariate scaling produced NaN values!")
                    # Manual scaling for each covariate column
                    scaled_covariate_values = np.zeros_like(covariate_data.values)
                    for i, col in enumerate(covariate_data.columns):
                        col_data = covariate_data[col].values
                        col_min, col_max = col_data.min(), col_data.max()
                        if col_max == col_min:
                            scaled_covariate_values[:, i] = 0.5
                        else:
                            scaled_covariate_values[:, i] = (col_data - col_min) / (col_max - col_min)
                    print("✅ Used manual scaling for covariates")
                
                scaled_covariates = pd.DataFrame(
                    scaled_covariate_values,
                    columns=['Open', 'High', 'Low', 'Volume'],
                    index=covariate_data.index
                )
                
            except Exception as cov_scaling_error:
                print(f"❌ Covariate scaling failed: {cov_scaling_error}")
                return None, None, None
            
            # 🔥 STEP 5: Convert to TimeSeries with validation
            try:
                freq_map = {"1h": "H", "1d": "B"}  # REMOVED: "15m": "15T", "30m": "30T"
                freq = freq_map.get(interval, "B")
                
                target_ts = TimeSeries.from_dataframe(scaled_target, freq=freq)
                covariate_ts = TimeSeries.from_dataframe(scaled_covariates, freq=freq)
                
                # Final validation
                if np.isnan(target_ts.values()).any() or np.isinf(target_ts.values()).any():
                    print(f"🚨 CRITICAL: Target TimeSeries contains invalid values!")
                    return None, None, None
                
                if np.isnan(covariate_ts.values()).any() or np.isinf(covariate_ts.values()).any():
                    print(f"🚨 CRITICAL: Covariate TimeSeries contains invalid values!")
                    return None, None, None
                
                print(f"✅ Traditional ML data prepared successfully:")
                print(f"    Target range: {target_ts.values().min():.6f} to {target_ts.values().max():.6f}")
                print(f"    Covariate range: {covariate_ts.values().min():.6f} to {covariate_ts.values().max():.6f}")
                
                return target_ts, covariate_ts, scaler
                
            except Exception as ts_error:
                print(f"❌ TimeSeries creation failed: {ts_error}")
                return None, None, None
                
        except Exception as e:
            print(f"❌ Error preparing traditional ML data: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None

    def fetch_and_prepare_data(self, ticker, start_date, end_date, interval="1d"):
        """Fetch and prepare data for prediction with ROBUST NaN handling"""
        try:
            df = yf.download(ticker, start=start_date, end=end_date, interval=interval)
            
            if df.empty:
                print("❌ No data fetched from yfinance")
                return pd.DataFrame()
            
            # 🔥 FIX: Enhanced data cleaning and validation
            print(f"📊 Raw data: {len(df)} points, NaN count: {df.isnull().sum().sum()}")
            
            # 🔥 STEP 1: Check for completely missing columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                print(f"❌ Missing required columns: {missing_columns}")
                return pd.DataFrame()
            
            # 🔥 STEP 2: Set proper frequency and handle missing values
            freq_map = {"1h": "H", "1d": "B"}  # REMOVED: "15m": "15T", "30m": "30T"
            df.index = pd.DatetimeIndex(df.index)
            df = df.asfreq(freq_map.get(interval, "B"))
            
            # 🔥 STEP 3: Intelligent gap filling
            # Forward fill first (carries last known price forward)
            df = df.fillna(method='ffill')
            
            # For any remaining NaN at the beginning, backward fill
            df = df.fillna(method='bfill')
            
            # 🔥 STEP 4: Handle any remaining NaN values
            original_len = len(df)
            
            # Check for rows that are still completely NaN
            completely_nan_mask = df.isnull().all(axis=1)
            if completely_nan_mask.any():
                print(f"⚠️ Dropping {completely_nan_mask.sum()} completely empty rows")
                df = df[~completely_nan_mask]
            
            # Check for partial NaN values and handle them
            partial_nan_mask = df.isnull().any(axis=1)
            if partial_nan_mask.any():
                print(f"⚠️ Found {partial_nan_mask.sum()} rows with partial NaN values")
                
                # For partial NaN, use interpolation
                for col in required_columns:
                    if df[col].isnull().any():
                        print(f"    Interpolating {df[col].isnull().sum()} NaN values in {col}")
                        df[col] = df[col].interpolate(method='linear')
                        
                        # If still NaN (at edges), use forward/backward fill
                        df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
            
            # 🔥 STEP 5: Final cleanup - drop any remaining NaN rows
            df = df.dropna()
            
            if len(df) < original_len:
                print(f"⚠️ Dropped {original_len - len(df)} rows with NaN values")
            
            # 🔥 STEP 6: Validate data quality
            if df.empty:
                print("❌ No valid data after cleaning")
                return pd.DataFrame()
            
            # Check for reasonable price values
            for price_col in ['Open', 'High', 'Low', 'Close']:
                zero_or_negative = (df[price_col] <= 0)
                if zero_or_negative.any():
                    print(f"⚠️ Found zero or negative prices in {price_col}")
                    # Replace with nearby valid values
                    df[price_col] = df[price_col].mask(zero_or_negative).interpolate(method='linear')
                    df[price_col] = df[price_col].fillna(method='ffill').fillna(method='bfill')
            
            # Check for infinite values
            if np.isinf(df.select_dtypes(include=[np.number])).any().any():
                print("⚠️ Found infinite values, replacing with finite alternatives")
                df = df.replace([np.inf, -np.inf], np.nan)
                df = df.fillna(method='ffill').fillna(method='bfill')
            
            print(f"✅ Clean data: {len(df)} points, final NaN count: {df.isnull().sum().sum()}")
            
            # 🔥 STEP 7: Final validation
            if df.isnull().any().any():
                print(f"🚨 WARNING: Still have NaN values after cleaning!")
                print(f"NaN counts: {df.isnull().sum()}")
                # As last resort, drop remaining NaN rows
                df = df.dropna()
                print(f"Final data after NaN removal: {len(df)} points")
            
            return df
            
        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    def generate_ensemble_predictions(self, ticker, start_date, end_date, interval="1d", forecast_horizon=1):
        """Generate ensemble predictions by averaging all individual models"""
        try:
            # Load ensemble config
            with open(f'{self.models_dir}/ensemble_config.pkl', 'rb') as f:
                unpickler = CustomUnpickler(f)
                ensemble_config = unpickler.load()
            
            model_names = ensemble_config['models']
            weights = ensemble_config['weights']
            
            all_predictions = []
            successful_models = []
            
            # Get predictions from each individual model
            for model_name in model_names:
                pred_result = self.generate_predictions(model_name, ticker, start_date, end_date, interval, forecast_horizon)
                if pred_result and pred_result['success']:
                    all_predictions.append(pred_result['predictions'])
                    successful_models.append(model_name)
                else:
                    print(f"⚠️ {model_name} failed in ensemble: {pred_result.get('error', 'Unknown error') if pred_result else 'No result'}")
            
            if not all_predictions:
                return {
                    'predictions': None,
                    'timestamps': None,
                    'model_used': 'Ensemble',
                    'success': False,
                    'error': 'No individual models succeeded'
                }
            
            # Calculate weighted average
            ensemble_pred = np.zeros_like(all_predictions[0])
            total_weight = 0
            
            for i, pred in enumerate(all_predictions):
                if i < len(weights):
                    weight = weights[i]
                else:
                    weight = 1.0 / len(all_predictions)
                
                ensemble_pred += pred * weight
                total_weight += weight
            
            ensemble_pred /= total_weight
            
            # Get timestamps from any successful prediction
            timestamps = None
            for model_name in successful_models:
                pred_result = self.generate_predictions(model_name, ticker, start_date, end_date, interval, forecast_horizon)
                if pred_result and pred_result['timestamps'] is not None:
                    timestamps = pred_result['timestamps']
                    break
            
            return {
                'predictions': ensemble_pred,
                'timestamps': timestamps,
                'model_used': f'Ensemble ({len(successful_models)} models)',
                'success': True,
                'individual_models': successful_models
            }
            
        except Exception as e:
            return {
                'predictions': None,
                'timestamps': None,
                'model_used': 'Ensemble',
                'success': False,
                'error': str(e)
            }
    
    def get_model_info(self):
        """Get information about all available models"""
        info = {
            'deployment_date': self.deployment_info['deployment_date'] if self.deployment_info else 'Unknown',
            'available_models': [],
            'status': self.deployment_info['status'] if self.deployment_info else 'Unknown'
        }
        
        # For general model info, we'll try to load the first available model for each type
        # This is just for display purposes, not for actual predictions
        model_types = ['linear_regression', 'random_forest', 'xgboost', 'lstm', 'gru', 'ensemble']
        
        for model_name in model_types:
            try:
                # Try to load any available model of this type (without specific stock/interval)
                # This will use the fallback naming convention
                model_info = self.load_model(model_name)
                if model_info:
                    info['available_models'].append({
                        'name': model_name,
                        'display_name': model_info['config']['name'],
                        'type': model_info['config']['type']
                    })
            except Exception as e:
                # If loading fails, add a generic entry
                display_names = {
                    'linear_regression': 'Linear Regression',
                    'random_forest': 'Random Forest',
                    'xgboost': 'XGBoost',
                    'lstm': 'LSTM',
                    'gru': 'GRU',
                    'ensemble': 'Ensemble'
                }
                info['available_models'].append({
                    'name': model_name,
                    'display_name': display_names.get(model_name, model_name.title()),
                    'type': 'deep_learning' if model_name in ['lstm', 'gru'] else 'traditional_ml' if model_name != 'ensemble' else 'ensemble'
                })
        
        return info
    
    def generate_ensemble_predictions_new(self, ticker, start_date, end_date, interval="1d", forecast_horizon=1, model_info=None):
        """Generate ensemble predictions by averaging Linear Regression + LSTM predictions"""
        try:
            print(f"🤖 Generating ensemble predictions (LR + LSTM) for {ticker}_{interval}...")
            
            # Get predictions from Linear Regression
            lr_result = self.generate_predictions('linear_regression', ticker, start_date, end_date, interval, forecast_horizon)
            if not lr_result['success']:
                print(f"❌ Linear Regression failed: {lr_result.get('error', 'Unknown error')}")
                return lr_result
            
            # Get predictions from LSTM
            lstm_result = self.generate_predictions('lstm', ticker, start_date, end_date, interval, forecast_horizon)
            if not lstm_result['success']:
                print(f"❌ LSTM failed: {lstm_result.get('error', 'Unknown error')}")
                return lstm_result
            
            # Average the predictions
            lr_preds = lr_result['predictions']
            lstm_preds = lstm_result['predictions']
            
            if lr_preds is None or lstm_preds is None:
                return {
                    'predictions': None,
                    'timestamps': None,
                    'model_used': 'LR + LSTM',
                    'success': False,
                    'error': 'One or both individual models failed to generate predictions'
                }
            
            # Ensure both predictions have the same length
            min_length = min(len(lr_preds), len(lstm_preds))
            lr_preds = lr_preds[:min_length]
            lstm_preds = lstm_preds[:min_length]
            
            # Calculate ensemble predictions (simple average)
            ensemble_preds = (lr_preds + lstm_preds) / 2
            
            print(f"✅ Ensemble predictions generated successfully:")
            print(f"   📊 Linear Regression predictions: {len(lr_preds)} points")
            print(f"   🧠 LSTM predictions: {len(lstm_preds)} points")
            print(f"   🤖 Ensemble predictions: {len(ensemble_preds)} points")
            
            return {
                'predictions': ensemble_preds,
                'timestamps': lr_result['timestamps'][:min_length] if lr_result['timestamps'] is not None else None,
                'model_used': 'LR + LSTM',
                'success': True,
                'individual_models': ['linear_regression', 'lstm'],
                'lr_predictions': lr_preds,
                'lstm_predictions': lstm_preds
            }
            
        except Exception as e:
            print(f"❌ Error in ensemble prediction: {str(e)}")
            return {
                'predictions': None,
                'timestamps': None,
                'model_used': 'LR + LSTM',
                'success': False,
                'error': str(e)
            }
    