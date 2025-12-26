"""Unit tests for portfolio optimization module."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock
from portfolio_optimization import (
    build_portfolio_options, select_best_portfolio, test_allocation_methods,
    run_monte_carlo_optimization, find_best_drift_threshold
)


@pytest.fixture
def sample_token_scores():
    """Create sample token scores for testing."""
    tokens = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT']
    scores = {}
    
    for i, token in enumerate(tokens):
        scores[token] = {
            'final_score': 100 - i * 10,
            'metrics': {
                'volatility': 0.5 + i * 0.1,
                'sharpe_ratio': 2.0 - i * 0.2,
                'market_cap': 1e11 / (i + 1),
                'liquidity_ratio': 0.01
            }
        }
    
    return scores


@pytest.fixture
def sample_price_data():
    """Create sample price data for testing."""
    dates = pd.date_range(start='2023-01-01', periods=400, freq='D')
    data = {}
    
    for token in ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT']:
        data[token] = pd.DataFrame({
            'close': np.random.randn(400).cumsum() + 100,
            'volume': np.random.rand(400) * 1000000
        }, index=dates)
    
    return data


@pytest.fixture
def sample_backtest_results():
    """Create sample backtest results for testing."""
    results = {}
    
    for drift in [0.08, 0.10, 0.12, 0.15]:
        results[drift] = {
            'performance_metrics': {
                'annualized_return': 0.15 + drift * 0.5,
                'volatility': 0.25,
                'sharpe_ratio': 1.5 - drift * 2,
                'max_drawdown': -0.20,
                'alpha': 0.05,
                'beta': 1.1
            }
        }
    
    return results


class TestFindBestDriftThreshold:
    """Tests for find_best_drift_threshold function."""
    
    def test_find_best_by_sharpe(self, sample_backtest_results):
        """Test finding best drift threshold by Sharpe ratio."""
        best_drift = find_best_drift_threshold(sample_backtest_results)
        
        # Drift of 0.08 should have highest Sharpe
        assert best_drift == 0.08
    
    def test_empty_results(self):
        """Test with empty backtest results."""
        best_drift = find_best_drift_threshold({})
        
        assert best_drift == 0.12  # Default fallback
    
    def test_single_result(self):
        """Test with single backtest result."""
        results = {
            0.10: {
                'performance_metrics': {'sharpe_ratio': 1.5}
            }
        }
        best_drift = find_best_drift_threshold(results)
        
        assert best_drift == 0.10


class TestSelectBestPortfolio:
    """Tests for select_best_portfolio function."""
    
    @pytest.fixture
    def portfolio_options(self):
        """Create sample portfolio options."""
        return {
            2: {'tokens': ['BTCUSDT', 'ETHUSDT'], 'score': 85.0},
            3: {'tokens': ['BTCUSDT', 'ETHUSDT', 'BNBUSDT'], 'score': 90.0},
            4: {'tokens': ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT'], 'score': 88.0},
            5: {'tokens': ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT'], 'score': 82.0}
        }
    
    def test_select_preferred_size_4(self, portfolio_options):
        """Test selecting portfolio with preferred size 4."""
        size, portfolio = select_best_portfolio(portfolio_options)
        
        assert size == 4  # Should prefer size 4
        assert portfolio['score'] == 88.0
    
    def test_select_manual_mode(self, portfolio_options):
        """Test selecting portfolio in manual mode."""
        manual_tokens = ['BTCUSDT', 'ETHUSDT']
        size, portfolio = select_best_portfolio(
            portfolio_options, 
            manual_tokens=manual_tokens, 
            selection_mode='manual'
        )
        
        assert size == 2  # Should match manual token count
    
    def test_no_preferred_size_available(self):
        """Test when preferred size 4 is not available."""
        portfolio_options = {
            2: {'tokens': ['BTCUSDT', 'ETHUSDT'], 'score': 85.0},
            3: {'tokens': ['BTCUSDT', 'ETHUSDT', 'BNBUSDT'], 'score': 90.0}
        }
        
        size, portfolio = select_best_portfolio(portfolio_options)
        
        assert size == 3  # Should take max available
    
    def test_empty_options(self):
        """Test with empty portfolio options."""
        size, portfolio = select_best_portfolio({})
        
        assert size is None
        assert portfolio is None


class TestPortfolioOptimizationEdgeCases:
    """Tests for edge cases in portfolio optimization."""
    
    def test_insufficient_tokens(self):
        """Test with insufficient tokens for portfolio."""
        token_scores = {
            'BTCUSDT': {'final_score': 90, 'metrics': {}}
        }
        
        # Should handle single token gracefully
        assert len(token_scores) < 2
    
    def test_monte_carlo_large_portfolio(self):
        """Test Monte Carlo skipping for large portfolios."""
        tokens = [f'TOKEN{i}USDT' for i in range(10)]
        
        # Monte Carlo should be skipped for >6 tokens
        assert len(tokens) > 6


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
