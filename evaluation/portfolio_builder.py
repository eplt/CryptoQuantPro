import numpy as np
import pandas as pd
from itertools import combinations
from scipy.optimize import minimize
from joblib import Parallel, delayed
from functools import partial
from config.settings import *
from config.parallel_config import get_safe_n_jobs, warn_if_high_n_jobs

class PortfolioBuilder:
    def __init__(self, token_scores, price_data, n_cores=None):
        self.token_scores = token_scores
        self.price_data = price_data
        
        # Get safe n_jobs value from explicit parameter, env var, config, or safe default
        if n_cores is not None:
            # User explicitly provided n_cores, honor it
            self.n_cores = n_cores
        else:
            # Use value from config (defaults to 8, or PORTFOLIO_N_JOBS env var if set)
            self.n_cores = PORTFOLIO_N_JOBS
        
        # Get backend from config
        self.backend = PORTFOLIO_BACKEND
        
        # Warn if using high n_jobs
        warn_if_high_n_jobs(self.n_cores, threshold=MAX_N_JOBS_WARNING_THRESHOLD)
        
        print(f"Portfolio Builder using {self.n_cores} cores for parallel processing (backend: {self.backend})")
        print(f"Note: To change, set PORTFOLIO_N_JOBS env var or pass n_cores parameter")
        
    def calculate_correlation_matrix(self, symbols, lookback_days=90):
        """Calculate correlation matrix for given symbols"""
        returns_data = {}
        
        for symbol in symbols:
            returns = np.log(self.price_data[symbol]['close'] / 
                           self.price_data[symbol]['close'].shift(1)).dropna()
            returns_data[symbol] = returns.tail(lookback_days)
        
        return pd.DataFrame(returns_data).corr()
    
    def find_optimal_portfolio_size(self, candidate_tokens, max_size=10):
        """Find optimal portfolio size using parallel processing"""
        print("Finding optimal portfolio size using parallel processing...")
        
        # Prepare size range
        min_tokens = 2
        sizes_to_test = list(range(min_tokens, min(max_size + 1, len(candidate_tokens) + 1)))
        
        # Parallel execution for different portfolio sizes
        portfolio_results = Parallel(n_jobs=self.n_cores, backend=self.backend, verbose=1)(
            delayed(self._evaluate_portfolio_size)(candidate_tokens, size)
            for size in sizes_to_test
        )
        
        # Combine results
        results = {}
        for size, portfolio in zip(sizes_to_test, portfolio_results):
            if portfolio:
                results[size] = portfolio
        
        return results
    
    def _evaluate_portfolio_size(self, candidate_tokens, portfolio_size):
        """Evaluate a specific portfolio size"""
        try:
            portfolio = self.select_portfolio(candidate_tokens, portfolio_size)
            return portfolio
        except Exception as e:
            print(f"Error evaluating portfolio size {portfolio_size}: {e}")
            return None
    
    def select_portfolio(self, candidate_tokens, portfolio_size):
        """Select optimal portfolio using parallel combination testing"""
        top_candidates = list(candidate_tokens.keys())[:portfolio_size * 4]  # 4x oversampling
        
        if len(top_candidates) < portfolio_size:
            print(f"Not enough candidates for portfolio size {portfolio_size}")
            return None
        
        # Generate all combinations
        all_combinations = list(combinations(top_candidates, portfolio_size))
        
        if len(all_combinations) > 1000:  # Limit for very large combination sets
            import random
            random.seed(42)
            all_combinations = random.sample(all_combinations, 1000)
        
        print(f"Testing {len(all_combinations)} combinations for portfolio size {portfolio_size}")
        
        # Parallel evaluation of combinations
        combination_scores = Parallel(n_jobs=self.n_cores, backend=self.backend, verbose=0)(
            delayed(self._evaluate_single_combination)(combo)
            for combo in all_combinations
        )
        
        # Find best combination
        best_combination = None
        best_score = -np.inf
        
        for combo, score in zip(all_combinations, combination_scores):
            if score > best_score:
                best_score = score
                best_combination = combo
        
        if best_combination:
            return {
                'tokens': best_combination,
                'score': best_score,
                'allocations': self.calculate_allocations(best_combination)
            }
        
        return None
    
    def _evaluate_single_combination(self, combo):
        """Evaluate a single token combination - worker function"""
        try:
            return self.evaluate_portfolio_combination(combo)
        except Exception as e:
            return -np.inf
    
    def evaluate_portfolio_combination(self, tokens):
        """Evaluate a combination of tokens for portfolio suitability"""
        try:
            # Check correlation constraint
            corr_matrix = self.calculate_correlation_matrix(tokens)
            max_correlation = corr_matrix.abs().values[np.triu_indices_from(corr_matrix.values, k=1)].max()
            
            if max_correlation > 0.75:  # MAX_CORRELATION
                return -np.inf  # Reject high correlation portfolios
            
            # Calculate portfolio metrics
            portfolio_volatility = self.calculate_portfolio_volatility(tokens, corr_matrix)
            diversification_ratio = self.calculate_diversification_ratio(tokens, corr_matrix)
            rebalancing_potential = self.estimate_rebalancing_alpha(tokens)
            
            # Weighted scoring
            score = (
                50 - abs(portfolio_volatility - 0.6) * 100 +  # Prefer ~60% volatility
                diversification_ratio * 20 +                   # Reward diversification
                rebalancing_potential * 30 +                   # Reward rebalancing potential
                sum(self.token_scores[token]['final_score'] for token in tokens) / len(tokens) * 0.3
            )
            
            return score
            
        except Exception as e:
            return -np.inf
    
    def calculate_portfolio_volatility(self, tokens, corr_matrix):
        """Calculate portfolio volatility assuming equal weights"""
        weights = np.array([1/len(tokens)] * len(tokens))
        
        # Get individual volatilities
        volatilities = []
        for token in tokens:
            returns = np.log(self.price_data[token]['close'] / 
                           self.price_data[token]['close'].shift(1)).dropna()
            volatilities.append(returns.std() * np.sqrt(365))
        
        vol_array = np.array(volatilities)
        portfolio_var = np.dot(weights, np.dot(np.outer(vol_array, vol_array) * corr_matrix.values, weights))
        
        return np.sqrt(portfolio_var)
    
    def calculate_diversification_ratio(self, tokens, corr_matrix):
        """Calculate diversification ratio (higher is better)"""
        weights = np.array([1/len(tokens)] * len(tokens))
        
        # Weighted average volatility
        volatilities = []
        for token in tokens:
            returns = np.log(self.price_data[token]['close'] / 
                           self.price_data[token]['close'].shift(1)).dropna()
            volatilities.append(returns.std() * np.sqrt(365))
        
        weighted_avg_vol = np.dot(weights, volatilities)
        portfolio_vol = self.calculate_portfolio_volatility(tokens, corr_matrix)
        
        return weighted_avg_vol / portfolio_vol if portfolio_vol > 0 else 1
    
    def estimate_rebalancing_alpha(self, tokens, lookback_days=252):
        """Estimate potential rebalancing alpha based on volatility and correlations"""
        returns_data = {}
        
        for token in tokens:
            returns = np.log(self.price_data[token]['close'] / 
                           self.price_data[token]['close'].shift(1)).dropna()
            returns_data[token] = returns.tail(lookback_days)
        
        returns_df = pd.DataFrame(returns_data)
        
        # Simulate simple rebalancing alpha
        equal_weights = 1 / len(tokens)
        cumulative_returns = returns_df.cumsum()
        
        # Calculate drift from equal weights over time
        portfolio_values = np.exp(cumulative_returns)
        portfolio_weights = portfolio_values.div(portfolio_values.sum(axis=1), axis=0)
        
        # Measure average drift (proxy for rebalancing opportunities)
        weight_drift = (portfolio_weights - equal_weights).abs().mean().mean()
        
        # Higher drift = more rebalancing potential
        return min(weight_drift * 100, 10)  # Cap at 10 points
    
    def test_all_allocation_methods(self, tokens):
        """Test all allocation methods in parallel"""
        allocation_methods = ['equal_weight', 'market_cap', 'risk_parity', 'volatility_weighted']
        
        print(f"Testing {len(allocation_methods)} allocation methods in parallel...")
        
        # Parallel execution for different allocation methods
        method_results = Parallel(n_jobs=min(self.n_cores, len(allocation_methods)), backend=self.backend, verbose=1)(
            delayed(self._test_allocation_method)(tokens, method)
            for method in allocation_methods
        )
        
        # Combine results
        results = {}
        for method, result in zip(allocation_methods, method_results):
            if result:
                results[method] = result
        
        return results
    
    def _test_allocation_method(self, tokens, method):
        """Test a single allocation method"""
        try:
            allocations = self.calculate_allocations(tokens, method)
            
            # Calculate some basic metrics for this allocation
            portfolio_score = self._score_allocation(tokens, allocations)
            
            return {
                'method': method,
                'allocations': allocations,
                'score': portfolio_score
            }
            
        except Exception as e:
            print(f"Error testing allocation method {method}: {e}")
            return None
    
    def _score_allocation(self, tokens, allocations):
        """Score an allocation method"""
        try:
            # Simple scoring based on diversification and risk
            weights = np.array(list(allocations.values()))
            
            # Penalize extreme concentrations
            concentration_penalty = max(0, (max(weights) - 0.4) * 100)
            
            # Reward balanced allocations
            balance_score = 100 - np.std(weights) * 200
            
            return balance_score - concentration_penalty
            
        except Exception:
            return 0
    
    def calculate_allocations(self, tokens, method='equal_weight'):
        """Calculate optimal allocations for selected tokens"""
        if method == 'equal_weight':
            return {token: 1/len(tokens) for token in tokens}
        
        elif method == 'market_cap':
            total_mcap = sum(self.token_scores[token]['metrics']['market_cap'] for token in tokens)
            return {token: self.token_scores[token]['metrics']['market_cap'] / total_mcap 
                   for token in tokens}
        
        elif method == 'risk_parity':
            return self.calculate_risk_parity_weights(tokens)
        
        elif method == 'volatility_weighted':
            return self.calculate_inverse_volatility_weights(tokens)
        
        else:
            return {token: 1/len(tokens) for token in tokens}  # Default to equal weight
    
    def calculate_risk_parity_weights(self, tokens):
        """Calculate risk parity (inverse volatility) weights"""
        volatilities = {}
        
        for token in tokens:
            returns = np.log(self.price_data[token]['close'] / 
                           self.price_data[token]['close'].shift(1)).dropna()
            volatilities[token] = returns.std() * np.sqrt(365)
        
        inv_vol = {token: 1/vol for token, vol in volatilities.items()}
        total_inv_vol = sum(inv_vol.values())
        
        return {token: inv_vol[token] / total_inv_vol for token in tokens}
    
    def calculate_inverse_volatility_weights(self, tokens):
        """Same as risk parity for this implementation"""
        return self.calculate_risk_parity_weights(tokens)
    
    def parallel_monte_carlo_optimization(self, tokens, n_simulations=10000):
        """Run Monte Carlo portfolio optimization in parallel"""
        print(f"Running Monte Carlo optimization with {n_simulations} simulations...")
        
        # Divide simulations across cores
        sims_per_core = n_simulations // self.n_cores
        simulation_batches = [sims_per_core] * self.n_cores
        
        # Handle remainder
        remainder = n_simulations % self.n_cores
        for i in range(remainder):
            simulation_batches[i] += 1
        
        # Parallel Monte Carlo
        mc_results = Parallel(n_jobs=self.n_cores, backend=self.backend, verbose=1)(
            delayed(self._monte_carlo_batch)(tokens, batch_size)
            for batch_size in simulation_batches
        )
        
        # Combine all results
        all_results = []
        for batch_result in mc_results:
            all_results.extend(batch_result)
        
        # Find optimal portfolio from all simulations
        best_portfolio = max(all_results, key=lambda x: x['sharpe_ratio'])
        
        return {
            'best_allocation': best_portfolio['weights'],
            'expected_return': best_portfolio['return'],
            'expected_volatility': best_portfolio['volatility'],
            'sharpe_ratio': best_portfolio['sharpe_ratio'],
            'all_results': all_results
        }
    
    def _monte_carlo_batch(self, tokens, n_sims):
        """Run a batch of Monte Carlo simulations"""
        results = []
        
        for _ in range(n_sims):
            # Generate random weights
            raw_weights = np.random.random(len(tokens))
            weights = raw_weights / raw_weights.sum()
            
            # Calculate portfolio metrics
            portfolio_return, portfolio_vol = self._calculate_portfolio_metrics(tokens, weights)
            sharpe_ratio = (portfolio_return - 0.02) / portfolio_vol if portfolio_vol > 0 else -np.inf
            
            results.append({
                'weights': {token: weight for token, weight in zip(tokens, weights)},
                'return': portfolio_return,
                'volatility': portfolio_vol,
                'sharpe_ratio': sharpe_ratio
            })
        
        return results
    
    def _calculate_portfolio_metrics(self, tokens, weights):
        """Calculate portfolio return and volatility"""
        try:
            # Get returns for all tokens
            returns_data = {}
            for token in tokens:
                returns = np.log(self.price_data[token]['close'] / 
                               self.price_data[token]['close'].shift(1)).dropna()
                returns_data[token] = returns
            
            returns_df = pd.DataFrame(returns_data)
            
            # Portfolio return and volatility
            portfolio_returns = returns_df.dot(weights)
            portfolio_return = portfolio_returns.mean() * 365
            portfolio_vol = portfolio_returns.std() * np.sqrt(365)
            
            return portfolio_return, portfolio_vol
            
        except Exception:
            return 0, 1  # Safe fallback
    
    def parallel_correlation_analysis(self, tokens, time_windows=[30, 60, 90, 180]):
        """Analyze correlations across multiple time windows in parallel"""
        print(f"Analyzing correlations across {len(time_windows)} time windows...")
        
        correlation_results = Parallel(n_jobs=min(self.n_cores, len(time_windows)), backend=self.backend, verbose=1)(
            delayed(self._analyze_correlation_window)(tokens, window)
            for window in time_windows
        )
        
        # Combine results
        results = {}
        for window, corr_data in zip(time_windows, correlation_results):
            if corr_data is not None:
                results[f'{window}d'] = corr_data
        
        return results
    
    def _analyze_correlation_window(self, tokens, window_days):
        """Analyze correlations for a specific time window"""
        try:
            corr_matrix = self.calculate_correlation_matrix(tokens, window_days)
            
            # Extract upper triangle (unique pairs)
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            correlations = corr_matrix.values[mask]
            
            return {
                'matrix': corr_matrix,
                'mean_correlation': np.mean(correlations),
                'max_correlation': np.max(correlations),
                'min_correlation': np.min(correlations),
                'std_correlation': np.std(correlations)
            }
            
        except Exception as e:
            print(f"Error analyzing {window_days}d correlations: {e}")
            return None
    
    def stress_test_portfolios(self, portfolio_candidates, stress_scenarios=None):
        """Stress test multiple portfolio candidates in parallel"""
        if stress_scenarios is None:
            stress_scenarios = [
                {'name': 'Market Crash', 'shock': -0.3},
                {'name': 'Volatility Spike', 'vol_multiplier': 2.0},
                {'name': 'Correlation Spike', 'correlation': 0.9},
                {'name': 'Liquidity Crisis', 'volume_shock': -0.5}
            ]
        
        print(f"Stress testing {len(portfolio_candidates)} portfolios...")
        
        stress_results = Parallel(n_jobs=self.n_cores, backend=self.backend, verbose=1)(
            delayed(self._stress_test_single_portfolio)(portfolio, stress_scenarios)
            for portfolio in portfolio_candidates
        )
        
        return stress_results
    
    def _stress_test_single_portfolio(self, portfolio, stress_scenarios):
        """Stress test a single portfolio"""
        try:
            portfolio_id = '_'.join(sorted(portfolio['tokens']))
            results = {'portfolio_id': portfolio_id, 'tokens': portfolio['tokens']}
            
            for scenario in stress_scenarios:
                scenario_result = self._apply_stress_scenario(portfolio, scenario)
                results[scenario['name']] = scenario_result
            
            return results
            
        except Exception as e:
            print(f"Error in stress test: {e}")
            return None
    
    def _apply_stress_scenario(self, portfolio, scenario):
        """Apply a specific stress scenario to a portfolio"""
        try:
            tokens = portfolio['tokens']
            
            if 'shock' in scenario:
                # Price shock scenario
                shock_return = scenario['shock']
                portfolio_shock = sum(portfolio['allocations'][token] * shock_return for token in tokens)
                return {'portfolio_shock_return': portfolio_shock}
                
            elif 'vol_multiplier' in scenario:
                # Volatility spike scenario
                vol_multiplier = scenario['vol_multiplier']
                current_vol = self.calculate_portfolio_volatility(tokens, self.calculate_correlation_matrix(tokens))
                stressed_vol = current_vol * vol_multiplier
                return {'stressed_volatility': stressed_vol}
                
            elif 'correlation' in scenario:
                # Correlation spike scenario
                target_corr = scenario['correlation']
                # Simplified: assume all correlations spike to target level
                return {'max_correlation_stress': target_corr}
                
            elif 'volume_shock' in scenario:
                # Liquidity crisis scenario
                volume_shock = scenario['volume_shock']
                liquidity_impact = sum(
                    portfolio['allocations'][token] * 
                    (1 + volume_shock * self.token_scores[token]['metrics']['liquidity_ratio'])
                    for token in tokens
                )
                return {'liquidity_impact': liquidity_impact}
            
            return {}
            
        except Exception:
            return {}
