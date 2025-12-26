"""
Backtesting configuration file for CryptoQuant Pro v0.2.0

This file defines various backtesting scenarios and configurations
for comprehensive strategy evaluation.
"""

# Walk-forward analysis settings
WALK_FORWARD_CONFIG = {
    'training_window_days': 365,  # 1 year training
    'test_window_days': 90,       # 3 months testing
    'step_days': 30,              # Monthly steps
    'min_training_samples': 250   # Minimum data points
}

# Monte Carlo simulation settings
MONTE_CARLO_CONFIG = {
    'n_simulations': 10000,       # Number of simulation paths
    'simulation_days': 365,       # 1 year ahead
    'confidence_levels': [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99],
    'random_seed': 42             # For reproducibility
}

# Stress test scenarios
STRESS_TEST_SCENARIOS = [
    {
        'name': 'Market Crash (-40%)',
        'description': '2008-style market crash scenario',
        'volatility_multiplier': 3.0,
        'correlation_shift': 0.3,     # Correlations increase in crashes
        'mean_return_shift': -0.002   # Negative drift
    },
    {
        'name': 'High Volatility Regime',
        'description': 'Extended period of high volatility',
        'volatility_multiplier': 2.5,
        'correlation_shift': 0.0,
        'mean_return_shift': 0.0
    },
    {
        'name': 'Correlation Breakdown',
        'description': 'All assets become highly correlated',
        'volatility_multiplier': 1.5,
        'correlation_shift': 0.5,
        'mean_return_shift': 0.0
    },
    {
        'name': 'Black Swan Event',
        'description': 'Extreme negative event',
        'volatility_multiplier': 5.0,
        'correlation_shift': 0.6,
        'mean_return_shift': -0.005
    },
    {
        'name': 'Bull Market Rally',
        'description': 'Strong positive trend',
        'volatility_multiplier': 0.8,
        'correlation_shift': -0.1,
        'mean_return_shift': 0.003
    }
]

# Portfolio rebalancing strategies to test
REBALANCING_STRATEGIES = {
    'threshold_based': {
        'drift_thresholds': [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30],
        'min_rebalance_interval': 5,
        'max_rebalance_interval': 30
    },
    'time_based': {
        'rebalance_frequencies': [7, 14, 30, 60, 90],  # days
        'allowed_drift': 0.05  # Maximum drift allowed
    },
    'hybrid': {
        'time_check_frequency': 7,
        'drift_threshold': 0.12,
        'min_interval': 5,
        'max_interval': 30
    }
}

# Transaction cost scenarios
TRANSACTION_COST_SCENARIOS = {
    'low_cost': 0.0003,      # 0.03% (market maker rebate)
    'standard': 0.0006,      # 0.06% (typical exchange fee)
    'high_cost': 0.0015,     # 0.15% (retail trading)
    'slippage_included': 0.0025  # 0.25% (fee + slippage)
}

# Market regimes for testing
MARKET_REGIMES = {
    'bull': {
        'min_return': 0.20,
        'max_volatility': 0.50,
        'description': 'Strong uptrend with manageable volatility'
    },
    'bear': {
        'max_return': -0.10,
        'min_volatility': 0.30,
        'description': 'Sustained downtrend with elevated volatility'
    },
    'high_vol': {
        'min_volatility': 0.80,
        'description': 'Extreme price swings regardless of direction'
    },
    'low_vol': {
        'max_volatility': 0.30,
        'description': 'Stable market with minimal price movement'
    },
    'crash_recovery': {
        'min_drawdown': -0.30,
        'recovery_required': True,
        'description': 'Sharp decline followed by recovery'
    },
    'sideways': {
        'max_abs_return': 0.15,
        'max_volatility': 0.40,
        'description': 'Range-bound market with no clear trend'
    }
}

# Performance benchmarks
PERFORMANCE_TARGETS = {
    'minimum_sharpe': 1.0,
    'minimum_return': 0.15,        # 15% annual return
    'maximum_drawdown': -0.25,     # 25% max drawdown
    'minimum_win_rate': 0.55,      # 55% winning periods
    'maximum_volatility': 0.60     # 60% annual volatility
}

# Data quality requirements
DATA_QUALITY_CONFIG = {
    'min_history_days': 365,
    'max_missing_data_pct': 0.05,  # 5% max missing
    'min_daily_volume_usd': 1_000_000,
    'outlier_threshold': 5.0       # Standard deviations
}
