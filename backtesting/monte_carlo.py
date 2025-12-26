"""Monte Carlo simulation module for portfolio analysis."""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import multiprocessing as mp
from functools import partial


class MonteCarloSimulator:
    """Implements Monte Carlo simulations for portfolio analysis."""
    
    def __init__(self, price_data, n_cores=None):
        """Initialize Monte Carlo simulator.
        
        Args:
            price_data: Dictionary of price DataFrames
            n_cores: Number of CPU cores to use (default: all available)
        """
        self.price_data = price_data
        self.n_cores = n_cores or mp.cpu_count()
        
        # Calculate returns for all tokens
        self.returns = {}
        for token, df in price_data.items():
            self.returns[token] = np.log(df['close'] / df['close'].shift(1)).dropna()
    
    def simulate_returns(self, tokens, weights, n_days, n_simulations=10000):
        """Simulate portfolio returns using historical distribution.
        
        Args:
            tokens: List of token symbols
            weights: Dictionary of token weights
            n_days: Number of days to simulate
            n_simulations: Number of simulation paths
            
        Returns:
            np.array: Array of simulated portfolio values (n_simulations x n_days)
        """
        # Get returns for selected tokens
        token_returns = np.array([self.returns[token].values for token in tokens]).T
        
        # Calculate mean and covariance
        mean_returns = np.mean(token_returns, axis=0)
        cov_matrix = np.cov(token_returns.T)
        
        # Portfolio weights as array
        w = np.array([weights[token] for token in tokens])
        
        # Simulate returns
        simulated_portfolio_returns = np.zeros((n_simulations, n_days))
        
        for i in range(n_simulations):
            # Generate random returns from multivariate normal distribution
            random_returns = np.random.multivariate_normal(mean_returns, cov_matrix, n_days)
            
            # Calculate portfolio returns
            portfolio_returns = np.dot(random_returns, w)
            simulated_portfolio_returns[i, :] = portfolio_returns
        
        # Convert returns to portfolio values (starting at 1.0)
        simulated_values = np.cumprod(1 + simulated_portfolio_returns, axis=1)
        simulated_values = np.insert(simulated_values, 0, 1.0, axis=1)  # Add starting value
        
        return simulated_values
    
    def analyze_risk(self, simulated_values, initial_capital=10000):
        """Analyze risk metrics from simulated values.
        
        Args:
            simulated_values: Array of simulated portfolio values
            initial_capital: Initial portfolio value
            
        Returns:
            dict: Risk metrics
        """
        final_values = simulated_values[:, -1] * initial_capital
        
        # Calculate percentiles
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        percentile_values = np.percentile(final_values, percentiles)
        
        # Calculate probability of loss
        prob_loss = np.mean(final_values < initial_capital)
        
        # Calculate expected shortfall (CVaR at 95%)
        var_95 = np.percentile(final_values, 5)
        cvar_95 = np.mean(final_values[final_values <= var_95])
        
        # Maximum drawdown across all paths
        max_drawdowns = []
        for path in simulated_values:
            cummax = np.maximum.accumulate(path)
            drawdown = (path - cummax) / cummax
            max_drawdowns.append(np.min(drawdown))
        
        return {
            'mean_final_value': np.mean(final_values),
            'median_final_value': np.median(final_values),
            'std_final_value': np.std(final_values),
            'percentiles': dict(zip(percentiles, percentile_values)),
            'probability_of_loss': prob_loss,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'mean_max_drawdown': np.mean(max_drawdowns),
            'worst_max_drawdown': np.min(max_drawdowns)
        }
    
    def run_simulation(self, portfolio_config, n_days=365, n_simulations=10000, 
                      initial_capital=10000):
        """Run complete Monte Carlo simulation.
        
        Args:
            portfolio_config: Portfolio configuration with tokens and allocations
            n_days: Number of days to simulate
            n_simulations: Number of simulation paths
            initial_capital: Starting capital
            
        Returns:
            dict: Simulation results and risk metrics
        """
        tokens = portfolio_config['tokens']
        weights = portfolio_config['allocations']
        
        print(f"Running Monte Carlo simulation...")
        print(f"  Simulations: {n_simulations:,}")
        print(f"  Time horizon: {n_days} days")
        print(f"  Portfolio: {len(tokens)} tokens")
        
        # Run simulation
        simulated_values = self.simulate_returns(tokens, weights, n_days, n_simulations)
        
        # Analyze risk
        risk_metrics = self.analyze_risk(simulated_values, initial_capital)
        
        print(f"\n✓ Simulation complete")
        print(f"  Mean final value: ${risk_metrics['mean_final_value']:,.2f}")
        print(f"  95% VaR: ${risk_metrics['var_95']:,.2f}")
        print(f"  Probability of loss: {risk_metrics['probability_of_loss']:.1%}")
        
        return {
            'simulated_values': simulated_values,
            'risk_metrics': risk_metrics,
            'config': {
                'n_simulations': n_simulations,
                'n_days': n_days,
                'initial_capital': initial_capital
            }
        }
    
    def stress_test(self, portfolio_config, stress_scenarios, n_simulations=1000):
        """Run stress tests with extreme market scenarios.
        
        Args:
            portfolio_config: Portfolio configuration
            stress_scenarios: List of stress scenario definitions
            n_simulations: Number of simulations per scenario
            
        Returns:
            dict: Stress test results
        """
        tokens = portfolio_config['tokens']
        weights = portfolio_config['allocations']
        
        print(f"Running stress tests with {len(stress_scenarios)} scenarios...")
        
        results = {}
        
        for scenario in stress_scenarios:
            scenario_name = scenario['name']
            volatility_multiplier = scenario.get('volatility_multiplier', 1.0)
            correlation_shift = scenario.get('correlation_shift', 0.0)
            mean_return_shift = scenario.get('mean_return_shift', 0.0)
            
            print(f"\n  Scenario: {scenario_name}")
            
            # Modify returns based on scenario
            token_returns = np.array([self.returns[token].values for token in tokens]).T
            mean_returns = np.mean(token_returns, axis=0) + mean_return_shift
            cov_matrix = np.cov(token_returns.T) * (volatility_multiplier ** 2)
            
            # Adjust correlations
            if correlation_shift != 0:
                corr_matrix = np.corrcoef(token_returns.T)
                corr_matrix = np.clip(corr_matrix + correlation_shift, -1, 1)
                std_devs = np.sqrt(np.diag(cov_matrix))
                cov_matrix = np.outer(std_devs, std_devs) * corr_matrix
            
            # Portfolio weights
            w = np.array([weights[token] for token in tokens])
            
            # Simulate
            simulated_returns = []
            for _ in range(n_simulations):
                random_returns = np.random.multivariate_normal(mean_returns, cov_matrix, 90)  # 90 days
                portfolio_return = np.sum(np.dot(random_returns, w))
                simulated_returns.append(portfolio_return)
            
            simulated_returns = np.array(simulated_returns)
            
            results[scenario_name] = {
                'mean_return': np.mean(simulated_returns),
                'median_return': np.median(simulated_returns),
                'std_return': np.std(simulated_returns),
                'var_95': np.percentile(simulated_returns, 5),
                'worst_return': np.min(simulated_returns),
                'probability_negative': np.mean(simulated_returns < 0)
            }
            
            print(f"    Mean return: {results[scenario_name]['mean_return']:.2%}")
            print(f"    95% VaR: {results[scenario_name]['var_95']:.2%}")
        
        print(f"\n✓ Stress testing complete")
        
        return results
