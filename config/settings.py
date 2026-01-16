import os
from datetime import datetime, timedelta

# API Configuration
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_KEY')

# Data Collection
DATA_DIR = 'data'
LOOKBACK_DAYS = 730  # 2 years of data
INTERVAL = '1d'      # Daily data

# Token Evaluation Criteria
MIN_MARKET_CAP = 1e9        # $1B minimum
MIN_DAILY_VOLUME = 10e6     # $10M minimum
MIN_VOLATILITY = 0.20       # 20% annualized
MAX_VOLATILITY = 1.50       # 150% annualized
MAX_CORRELATION = 0.75      # Between any two tokens

# Portfolio Configuration
MIN_TOKENS = 2
MAX_TOKENS = 10
PREFERRED_TOKENS = 5        # Sweet spot

# Rebalancing Parameters
DRIFT_THRESHOLDS = [0.08, 0.10, 0.12, 0.15, 0.20, 0.25]  # Test range
MIN_REBALANCE_INTERVAL = 5  # Days
MAX_REBALANCE_INTERVAL = 21 # Days
TRANSACTION_FEE = 0.0006    # 0.06% per trade

# Portfolio Allocation Methods
ALLOCATION_METHODS = ['equal_weight', 'market_cap', 'risk_parity', 'volatility_weighted']

# Backtesting
BACKTEST_START = datetime.now() - timedelta(days=LOOKBACK_DAYS)
BACKTEST_END = datetime.now() - timedelta(days=30)  # Leave recent data for live testing
INITIAL_CAPITAL = 10000     # $10k starting capital

# Parallel Processing Configuration
# Safe default for parallel processing - prevents excessive memory usage and SIGKILL on macOS
# Can be overridden via PORTFOLIO_N_JOBS environment variable
PORTFOLIO_N_JOBS = int(os.getenv('PORTFOLIO_N_JOBS', '8'))  # Conservative default: 8 cores
PORTFOLIO_BACKEND = os.getenv('PORTFOLIO_BACKEND', 'loky')  # joblib backend: 'loky' or 'threading'
MAX_N_JOBS_WARNING_THRESHOLD = 16  # Warn if n_jobs exceeds this value
