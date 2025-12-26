"""
pytest conftest.py - Shared fixtures and configuration for all tests.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


@pytest.fixture(scope="session")
def test_config():
    """Test configuration fixture."""
    return {
        'lookback_days': 365,
        'min_tokens': 2,
        'max_tokens': 10,
        'risk_free_rate': 0.02
    }


@pytest.fixture
def sample_dates():
    """Generate sample date range for testing."""
    return pd.date_range(start='2023-01-01', periods=400, freq='D')


@pytest.fixture
def generate_price_series():
    """Factory fixture to generate price series."""
    def _generate(n_days=400, initial_price=100, volatility=0.02, seed=None):
        if seed is not None:
            np.random.seed(seed)
        returns = np.random.randn(n_days) * volatility + 0.0005  # Small positive drift
        prices = initial_price * np.cumprod(1 + returns)
        return prices
    return _generate


@pytest.fixture
def generate_token_data(sample_dates, generate_price_series):
    """Factory fixture to generate complete token data."""
    def _generate(tokens=None, n_days=400):
        if tokens is None:
            tokens = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
        
        data = {}
        for i, token in enumerate(tokens):
            prices = generate_price_series(n_days=n_days, initial_price=100*(i+1), seed=i)
            dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D')
            
            data[token] = pd.DataFrame({
                'open': prices * (1 + np.random.randn(n_days) * 0.005),
                'high': prices * (1 + np.abs(np.random.randn(n_days)) * 0.01),
                'low': prices * (1 - np.abs(np.random.randn(n_days)) * 0.01),
                'close': prices,
                'volume': np.random.rand(n_days) * 1000000
            }, index=dates)
        
        return data
    return _generate


@pytest.fixture
def sample_portfolio_config():
    """Sample portfolio configuration for testing."""
    return {
        'tokens': ['BTCUSDT', 'ETHUSDT', 'BNBUSDT'],
        'allocations': {
            'BTCUSDT': 0.4,
            'ETHUSDT': 0.35,
            'BNBUSDT': 0.25
        },
        'drift_threshold': 0.10,
        'min_interval': 5,
        'max_interval': 30
    }


# Markers for test categorization
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "requires_api: mark test as requiring API credentials"
    )
    config.addinivalue_line(
        "markers", "requires_data: mark test as requiring cached data"
    )


# Skip tests that require API if no credentials
def pytest_collection_modifyitems(config, items):
    """Modify test collection to handle conditional skips."""
    for item in items:
        if "requires_api" in item.keywords:
            # Check if API credentials are available
            try:
                from config.settings import BINANCE_API_KEY
                if not BINANCE_API_KEY:
                    item.add_marker(pytest.mark.skip(reason="No API credentials available"))
            except ImportError:
                item.add_marker(pytest.mark.skip(reason="Config module not available"))
