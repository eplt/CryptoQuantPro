"""Unit tests for data processing module."""

import pytest
from data_processing import (
    get_manual_token_list, validate_manual_tokens, collect_tokens_by_mode
)
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch


@pytest.fixture
def sample_price_data():
    """Create sample price data for testing."""
    dates = pd.date_range(start='2023-01-01', periods=400, freq='D')
    data = {}
    
    for token in ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']:
        data[token] = pd.DataFrame({
            'close': np.random.randn(400).cumsum() + 100,
            'volume': np.random.rand(400) * 1000000
        }, index=dates)
    
    return data


@pytest.fixture
def sample_market_data():
    """Create sample market data for testing."""
    return {
        'BTCUSDT': {
            'symbol': 'BTCUSDT',
            'price': 45000,
            'volume_24h': 1e9,
            'market_cap': 1e12,
            'rank': 1
        },
        'ETHUSDT': {
            'symbol': 'ETHUSDT',
            'price': 3000,
            'volume_24h': 5e8,
            'market_cap': 5e11,
            'rank': 2
        },
        'BNBUSDT': {
            'symbol': 'BNBUSDT',
            'price': 300,
            'volume_24h': 1e8,
            'market_cap': 5e10,
            'rank': 4
        }
    }


class TestValidateManualTokens:
    """Tests for validate_manual_tokens function."""
    
    def test_all_valid_tokens(self, sample_price_data, sample_market_data):
        """Test validation with all valid tokens."""
        tokens = ['BTCUSDT', 'ETHUSDT']
        result = validate_manual_tokens(tokens, sample_price_data, sample_market_data)
        
        assert result is not None
        assert len(result) == 2
        assert 'BTCUSDT' in result
        assert 'ETHUSDT' in result
    
    def test_invalid_token(self, sample_price_data, sample_market_data):
        """Test validation with invalid token."""
        tokens = ['BTCUSDT', 'INVALIDUSDT']
        result = validate_manual_tokens(tokens, sample_price_data, sample_market_data)
        
        assert result is not None
        assert len(result) == 1
        assert 'BTCUSDT' in result
        assert 'INVALIDUSDT' not in result
    
    def test_insufficient_valid_tokens(self, sample_price_data, sample_market_data):
        """Test validation with insufficient valid tokens."""
        tokens = ['INVALIDUSDT', 'NONEUSDT']
        result = validate_manual_tokens(tokens, sample_price_data, sample_market_data)
        
        assert result is None
    
    def test_insufficient_data(self, sample_price_data, sample_market_data):
        """Test validation with token having insufficient data."""
        # Add token with insufficient data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')  # Only 100 days
        sample_price_data['SHORTUSDT'] = pd.DataFrame({
            'close': np.random.randn(100).cumsum() + 100,
            'volume': np.random.rand(100) * 1000000
        }, index=dates)
        
        tokens = ['BTCUSDT', 'SHORTUSDT']
        result = validate_manual_tokens(tokens, sample_price_data, sample_market_data)
        
        assert result is not None
        assert 'SHORTUSDT' not in result


class TestCollectTokensByMode:
    """Tests for collect_tokens_by_mode function."""
    
    @pytest.fixture
    def mock_collector(self, sample_price_data, sample_market_data):
        """Create mock DataCollector."""
        collector = Mock()
        collector.collect_all_data.return_value = (sample_price_data, sample_market_data)
        return collector
    
    def test_auto_mode(self, mock_collector):
        """Test auto mode collection."""
        price_data, market_data = collect_tokens_by_mode(mock_collector, 'auto', [])
        
        assert price_data is not None
        assert market_data is not None
        mock_collector.collect_all_data.assert_called_once()
    
    def test_manual_mode(self, mock_collector):
        """Test manual mode collection."""
        tokens = ['BTCUSDT', 'ETHUSDT']
        price_data, market_data = collect_tokens_by_mode(mock_collector, 'manual', tokens)
        
        assert price_data is not None
        assert market_data is not None
        mock_collector.collect_all_data.assert_called_once_with(symbols=tokens)
    
    def test_hybrid_mode_no_fill_needed(self, mock_collector, sample_price_data, sample_market_data):
        """Test hybrid mode when no additional tokens needed."""
        # Create data with 10 tokens
        extended_price_data = {f'TOKEN{i}USDT': sample_price_data['BTCUSDT'] 
                              for i in range(10)}
        extended_market_data = {f'TOKEN{i}USDT': sample_market_data['BTCUSDT'] 
                               for i in range(10)}
        
        mock_collector.collect_all_data.return_value = (extended_price_data, extended_market_data)
        
        tokens = [f'TOKEN{i}USDT' for i in range(10)]
        price_data, market_data = collect_tokens_by_mode(mock_collector, 'hybrid', tokens)
        
        assert len(price_data) >= len(tokens)


class TestDataProcessingEdgeCases:
    """Tests for edge cases in data processing."""
    
    def test_empty_token_list(self, sample_price_data, sample_market_data):
        """Test with empty token list."""
        result = validate_manual_tokens([], sample_price_data, sample_market_data)
        assert result is None
    
    def test_single_valid_token(self, sample_price_data, sample_market_data):
        """Test with single valid token (should fail minimum requirement)."""
        result = validate_manual_tokens(['BTCUSDT'], sample_price_data, sample_market_data)
        assert result is None  # Need at least 2 tokens


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
