"""Unit tests for enhanced metrics module."""

import pytest
import numpy as np
from analysis.enhanced_metrics import EnhancedMetrics


@pytest.fixture
def sample_returns():
    """Create sample returns data for testing."""
    np.random.seed(42)
    return np.random.randn(252) * 0.02  # Daily returns with 2% std


@pytest.fixture
def sample_portfolio_values():
    """Create sample portfolio values for testing."""
    np.random.seed(42)
    returns = np.random.randn(252) * 0.02 + 0.001  # Positive drift
    values = np.cumprod(1 + returns) * 10000
    return np.insert(values, 0, 10000)  # Start at $10,000


class TestSharpeRatio:
    """Tests for Sharpe ratio calculation."""
    
    def test_sharpe_positive_returns(self, sample_returns):
        """Test Sharpe ratio with positive returns."""
        sharpe = EnhancedMetrics.calculate_sharpe_ratio(sample_returns)
        
        assert isinstance(sharpe, float)
        assert -5 < sharpe < 5  # Reasonable range
    
    def test_sharpe_zero_volatility(self):
        """Test Sharpe ratio with zero volatility."""
        returns = np.zeros(252)
        sharpe = EnhancedMetrics.calculate_sharpe_ratio(returns)
        
        # Zero volatility with negative mean (due to risk-free rate) gives 0
        assert sharpe == 0.0 or sharpe < 0  # Can be 0 or slightly negative
    
    def test_sharpe_custom_rf_rate(self, sample_returns):
        """Test Sharpe ratio with custom risk-free rate."""
        sharpe1 = EnhancedMetrics.calculate_sharpe_ratio(sample_returns, risk_free_rate=0.02)
        sharpe2 = EnhancedMetrics.calculate_sharpe_ratio(sample_returns, risk_free_rate=0.05)
        
        assert sharpe1 != sharpe2


class TestSortinoRatio:
    """Tests for Sortino ratio calculation."""
    
    def test_sortino_positive_returns(self, sample_returns):
        """Test Sortino ratio with positive returns."""
        sortino = EnhancedMetrics.calculate_sortino_ratio(sample_returns)
        
        assert isinstance(sortino, float)
    
    def test_sortino_no_downside(self):
        """Test Sortino ratio with no downside returns."""
        returns = np.abs(np.random.randn(252)) * 0.01  # All positive
        sortino = EnhancedMetrics.calculate_sortino_ratio(returns)
        
        # With no downside, sortino should be 0 or infinite depending on implementation
        assert sortino == 0.0 or np.isnan(sortino) or np.isinf(sortino)


class TestCalmarRatio:
    """Tests for Calmar ratio calculation."""
    
    def test_calmar_normal_case(self, sample_returns):
        """Test Calmar ratio calculation."""
        max_dd = -0.20
        calmar = EnhancedMetrics.calculate_calmar_ratio(sample_returns, max_dd)
        
        assert isinstance(calmar, float)
        # Can be positive or negative depending on returns
    
    def test_calmar_zero_drawdown(self, sample_returns):
        """Test Calmar ratio with zero drawdown."""
        calmar = EnhancedMetrics.calculate_calmar_ratio(sample_returns, 0.0)
        
        assert calmar == 0.0


class TestOmegaRatio:
    """Tests for Omega ratio calculation."""
    
    def test_omega_normal_case(self, sample_returns):
        """Test Omega ratio calculation."""
        omega = EnhancedMetrics.calculate_omega_ratio(sample_returns)
        
        assert isinstance(omega, float)
        assert omega > 0
    
    def test_omega_all_positive(self):
        """Test Omega ratio with all positive returns."""
        returns = np.abs(np.random.randn(252)) * 0.01
        omega = EnhancedMetrics.calculate_omega_ratio(returns)
        
        # With all positive returns, omega should be very high or infinite
        assert omega > 10 or np.isinf(omega)


class TestVaRCVaR:
    """Tests for VaR and CVaR calculation."""
    
    def test_var_cvar_95(self, sample_returns):
        """Test VaR and CVaR at 95% confidence."""
        var, cvar = EnhancedMetrics.calculate_var_cvar(sample_returns, 0.95)
        
        assert isinstance(var, float)
        assert isinstance(cvar, float)
        assert cvar <= var  # CVaR should be more extreme
    
    def test_var_cvar_99(self, sample_returns):
        """Test VaR and CVaR at 99% confidence."""
        var95, cvar95 = EnhancedMetrics.calculate_var_cvar(sample_returns, 0.95)
        var99, cvar99 = EnhancedMetrics.calculate_var_cvar(sample_returns, 0.99)
        
        assert var99 <= var95  # Higher confidence = more extreme
        assert cvar99 <= cvar95


class TestMaximumDrawdown:
    """Tests for maximum drawdown calculation."""
    
    def test_max_drawdown_decreasing(self):
        """Test max drawdown with decreasing values."""
        values = np.array([100, 90, 80, 70, 60])
        max_dd, peak_idx, trough_idx, recovery_idx = \
            EnhancedMetrics.calculate_maximum_drawdown(values)
        
        assert max_dd < 0
        assert peak_idx == 0
        assert trough_idx == 4
    
    def test_max_drawdown_with_recovery(self):
        """Test max drawdown with recovery."""
        values = np.array([100, 90, 80, 90, 100, 110])
        max_dd, peak_idx, trough_idx, recovery_idx = \
            EnhancedMetrics.calculate_maximum_drawdown(values)
        
        assert max_dd < 0
        assert recovery_idx is not None
        assert values[recovery_idx] >= values[peak_idx]
    
    def test_max_drawdown_increasing(self):
        """Test max drawdown with increasing values (no drawdown)."""
        values = np.array([100, 110, 120, 130])
        max_dd, peak_idx, trough_idx, recovery_idx = \
            EnhancedMetrics.calculate_maximum_drawdown(values)
        
        assert max_dd == 0.0


class TestAlphaBeta:
    """Tests for alpha and beta calculation."""
    
    @pytest.fixture
    def benchmark_returns(self):
        """Create sample benchmark returns."""
        np.random.seed(43)
        return np.random.randn(252) * 0.015
    
    def test_alpha_beta_calculation(self, sample_returns, benchmark_returns):
        """Test alpha and beta calculation."""
        alpha, beta = EnhancedMetrics.calculate_alpha_beta(sample_returns, benchmark_returns)
        
        assert isinstance(alpha, float)
        assert isinstance(beta, float)
        assert -1 < beta < 3  # Reasonable beta range
    
    def test_alpha_beta_perfect_correlation(self):
        """Test alpha and beta with perfect correlation."""
        returns = np.random.randn(252) * 0.02
        benchmark = returns.copy()  # Perfect correlation
        
        alpha, beta = EnhancedMetrics.calculate_alpha_beta(returns, benchmark)
        
        assert abs(beta - 1.0) < 0.01  # Should be close to 1


class TestInformationRatio:
    """Tests for information ratio calculation."""
    
    def test_information_ratio(self, sample_returns):
        """Test information ratio calculation."""
        benchmark_returns = sample_returns * 0.9  # Slightly different
        info_ratio = EnhancedMetrics.calculate_information_ratio(
            sample_returns, benchmark_returns
        )
        
        assert isinstance(info_ratio, float)
    
    def test_information_ratio_identical(self, sample_returns):
        """Test information ratio with identical returns."""
        info_ratio = EnhancedMetrics.calculate_information_ratio(
            sample_returns, sample_returns
        )
        
        assert info_ratio == 0.0


class TestWinRateStats:
    """Tests for win rate statistics."""
    
    def test_win_rate_mixed(self, sample_returns):
        """Test win rate with mixed returns."""
        stats = EnhancedMetrics.calculate_win_rate_stats(sample_returns)
        
        assert 0 <= stats['win_rate'] <= 1
        assert stats['avg_win'] >= 0
        assert stats['avg_loss'] <= 0
        assert stats['profit_factor'] >= 0
    
    def test_win_rate_all_positive(self):
        """Test win rate with all positive returns."""
        returns = np.abs(np.random.randn(252)) * 0.01
        stats = EnhancedMetrics.calculate_win_rate_stats(returns)
        
        assert stats['win_rate'] == 1.0
        assert stats['avg_loss'] == 0.0


class TestComprehensiveMetrics:
    """Tests for comprehensive metrics calculation."""
    
    def test_all_metrics_without_benchmark(self, sample_portfolio_values):
        """Test calculating all metrics without benchmark."""
        metrics = EnhancedMetrics.calculate_all_metrics(sample_portfolio_values)
        
        # Check all expected keys are present
        expected_keys = [
            'total_return', 'annualized_return', 'volatility',
            'sharpe_ratio', 'sortino_ratio', 'calmar_ratio',
            'max_drawdown', 'var_95', 'cvar_95',
            'skewness', 'kurtosis', 'win_rate'
        ]
        
        for key in expected_keys:
            assert key in metrics
    
    def test_all_metrics_with_benchmark(self, sample_portfolio_values):
        """Test calculating all metrics with benchmark."""
        benchmark_values = sample_portfolio_values * 0.95  # Similar but lower
        
        metrics = EnhancedMetrics.calculate_all_metrics(
            sample_portfolio_values, 
            benchmark_values
        )
        
        # Check benchmark-relative metrics
        assert 'alpha' in metrics
        assert 'beta' in metrics
        assert 'information_ratio' in metrics


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
