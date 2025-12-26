"""Enhanced performance metrics module with advanced risk-return analysis."""

import numpy as np
import pandas as pd
from scipy import stats


class EnhancedMetrics:
    """Calculate advanced performance and risk metrics."""
    
    @staticmethod
    def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
        """Calculate Sharpe ratio.
        
        Args:
            returns: Array of returns
            risk_free_rate: Annual risk-free rate
            
        Returns:
            float: Sharpe ratio
        """
        excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
        if np.std(excess_returns) == 0:
            return 0.0
        return np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)
    
    @staticmethod
    def calculate_sortino_ratio(returns, risk_free_rate=0.02, target_return=0):
        """Calculate Sortino ratio (downside deviation).
        
        Args:
            returns: Array of returns
            risk_free_rate: Annual risk-free rate
            target_return: Target return threshold
            
        Returns:
            float: Sortino ratio
        """
        excess_returns = returns - (risk_free_rate / 252)
        downside_returns = returns[returns < target_return]
        
        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            return 0.0
        
        downside_deviation = np.std(downside_returns)
        return np.sqrt(252) * np.mean(excess_returns) / downside_deviation
    
    @staticmethod
    def calculate_calmar_ratio(returns, max_drawdown):
        """Calculate Calmar ratio (return / max drawdown).
        
        Args:
            returns: Array of returns
            max_drawdown: Maximum drawdown (negative value)
            
        Returns:
            float: Calmar ratio
        """
        annual_return = np.mean(returns) * 252
        if max_drawdown == 0:
            return 0.0
        return annual_return / abs(max_drawdown)
    
    @staticmethod
    def calculate_omega_ratio(returns, threshold=0):
        """Calculate Omega ratio.
        
        Args:
            returns: Array of returns
            threshold: Return threshold
            
        Returns:
            float: Omega ratio
        """
        returns_above = returns[returns > threshold]
        returns_below = returns[returns < threshold]
        
        if len(returns_below) == 0 or np.sum(np.abs(returns_below)) == 0:
            return np.inf if len(returns_above) > 0 else 1.0
        
        return np.sum(returns_above - threshold) / np.sum(np.abs(returns_below - threshold))
    
    @staticmethod
    def calculate_var_cvar(returns, confidence_level=0.95):
        """Calculate Value at Risk and Conditional Value at Risk.
        
        Args:
            returns: Array of returns
            confidence_level: Confidence level (default 95%)
            
        Returns:
            tuple: (VaR, CVaR) at specified confidence level
        """
        var = np.percentile(returns, (1 - confidence_level) * 100)
        cvar = np.mean(returns[returns <= var])
        return var, cvar
    
    @staticmethod
    def calculate_maximum_drawdown(portfolio_values):
        """Calculate maximum drawdown.
        
        Args:
            portfolio_values: Array of portfolio values
            
        Returns:
            tuple: (max_drawdown, peak_idx, trough_idx, recovery_idx)
        """
        cummax = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - cummax) / cummax
        
        max_dd = np.min(drawdown)
        trough_idx = np.argmin(drawdown)
        peak_idx = np.argmax(portfolio_values[:trough_idx + 1]) if trough_idx > 0 else 0
        
        # Find recovery point (if exists)
        recovery_idx = None
        if trough_idx < len(portfolio_values) - 1:
            future_values = portfolio_values[trough_idx + 1:]
            peak_value = portfolio_values[peak_idx]
            recovery_points = np.where(future_values >= peak_value)[0]
            if len(recovery_points) > 0:
                recovery_idx = trough_idx + 1 + recovery_points[0]
        
        return max_dd, peak_idx, trough_idx, recovery_idx
    
    @staticmethod
    def calculate_alpha_beta(portfolio_returns, benchmark_returns, risk_free_rate=0.02):
        """Calculate alpha and beta relative to benchmark.
        
        Args:
            portfolio_returns: Array of portfolio returns
            benchmark_returns: Array of benchmark returns
            risk_free_rate: Annual risk-free rate
            
        Returns:
            tuple: (alpha, beta)
        """
        # Align arrays
        min_len = min(len(portfolio_returns), len(benchmark_returns))
        portfolio_returns = portfolio_returns[:min_len]
        benchmark_returns = benchmark_returns[:min_len]
        
        # Calculate beta using covariance
        covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)
        
        if benchmark_variance == 0:
            beta = 0.0
        else:
            beta = covariance / benchmark_variance
        
        # Calculate alpha (annualized)
        daily_rf = risk_free_rate / 252
        portfolio_mean = np.mean(portfolio_returns)
        benchmark_mean = np.mean(benchmark_returns)
        
        alpha = (portfolio_mean - daily_rf - beta * (benchmark_mean - daily_rf)) * 252
        
        return alpha, beta
    
    @staticmethod
    def calculate_information_ratio(portfolio_returns, benchmark_returns):
        """Calculate information ratio (excess return / tracking error).
        
        Args:
            portfolio_returns: Array of portfolio returns
            benchmark_returns: Array of benchmark returns
            
        Returns:
            float: Information ratio
        """
        min_len = min(len(portfolio_returns), len(benchmark_returns))
        portfolio_returns = portfolio_returns[:min_len]
        benchmark_returns = benchmark_returns[:min_len]
        
        excess_returns = portfolio_returns - benchmark_returns
        tracking_error = np.std(excess_returns)
        
        if tracking_error == 0:
            return 0.0
        
        return np.sqrt(252) * np.mean(excess_returns) / tracking_error
    
    @staticmethod
    def calculate_tail_ratio(returns):
        """Calculate tail ratio (95th percentile / 5th percentile).
        
        Args:
            returns: Array of returns
            
        Returns:
            float: Tail ratio
        """
        p95 = np.percentile(returns, 95)
        p5 = np.percentile(returns, 5)
        
        if p5 == 0:
            return np.inf if p95 > 0 else 1.0
        
        return abs(p95 / p5)
    
    @staticmethod
    def calculate_skewness_kurtosis(returns):
        """Calculate skewness and kurtosis of returns.
        
        Args:
            returns: Array of returns
            
        Returns:
            tuple: (skewness, kurtosis)
        """
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        return skewness, kurtosis
    
    @staticmethod
    def calculate_win_rate_stats(returns):
        """Calculate win rate and related statistics.
        
        Args:
            returns: Array of returns
            
        Returns:
            dict: Win rate statistics
        """
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        win_rate = len(positive_returns) / len(returns) if len(returns) > 0 else 0
        avg_win = np.mean(positive_returns) if len(positive_returns) > 0 else 0
        avg_loss = np.mean(negative_returns) if len(negative_returns) > 0 else 0
        
        profit_factor = abs(np.sum(positive_returns) / np.sum(negative_returns)) \
                       if len(negative_returns) > 0 and np.sum(negative_returns) != 0 else 0
        
        return {
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'win_loss_ratio': abs(avg_win / avg_loss) if avg_loss != 0 else 0
        }
    
    @staticmethod
    def calculate_all_metrics(portfolio_values, benchmark_values=None, risk_free_rate=0.02):
        """Calculate comprehensive set of performance metrics.
        
        Args:
            portfolio_values: Array of portfolio values
            benchmark_values: Optional array of benchmark values
            risk_free_rate: Annual risk-free rate
            
        Returns:
            dict: Comprehensive metrics dictionary
        """
        # Calculate returns
        portfolio_returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        # Basic metrics
        total_return = (portfolio_values[-1] / portfolio_values[0]) - 1
        annualized_return = ((1 + total_return) ** (252 / len(portfolio_values))) - 1
        volatility = np.std(portfolio_returns) * np.sqrt(252)
        
        # Risk metrics
        sharpe = EnhancedMetrics.calculate_sharpe_ratio(portfolio_returns, risk_free_rate)
        sortino = EnhancedMetrics.calculate_sortino_ratio(portfolio_returns, risk_free_rate)
        
        max_dd, peak_idx, trough_idx, recovery_idx = \
            EnhancedMetrics.calculate_maximum_drawdown(portfolio_values)
        
        calmar = EnhancedMetrics.calculate_calmar_ratio(portfolio_returns, max_dd)
        omega = EnhancedMetrics.calculate_omega_ratio(portfolio_returns)
        
        var_95, cvar_95 = EnhancedMetrics.calculate_var_cvar(portfolio_returns, 0.95)
        
        # Distribution metrics
        skewness, kurtosis = EnhancedMetrics.calculate_skewness_kurtosis(portfolio_returns)
        tail_ratio = EnhancedMetrics.calculate_tail_ratio(portfolio_returns)
        
        # Win rate statistics
        win_stats = EnhancedMetrics.calculate_win_rate_stats(portfolio_returns)
        
        metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'omega_ratio': omega,
            'max_drawdown': max_dd,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'tail_ratio': tail_ratio,
            **win_stats
        }
        
        # Benchmark-relative metrics
        if benchmark_values is not None:
            benchmark_returns = np.diff(benchmark_values) / benchmark_values[:-1]
            alpha, beta = EnhancedMetrics.calculate_alpha_beta(
                portfolio_returns, benchmark_returns, risk_free_rate
            )
            info_ratio = EnhancedMetrics.calculate_information_ratio(
                portfolio_returns, benchmark_returns
            )
            
            metrics['alpha'] = alpha
            metrics['beta'] = beta
            metrics['information_ratio'] = info_ratio
        
        return metrics
