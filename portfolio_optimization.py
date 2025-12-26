"""Portfolio optimization module for building and optimizing cryptocurrency portfolios."""

from evaluation.portfolio_builder import PortfolioBuilder
from backtesting.backtest_engine import BacktestEngine
from datetime import datetime, timedelta
import os


def build_portfolio_options(token_scores, price_data, manual_tokens=None, selection_mode='auto'):
    """Build optimal portfolio options of different sizes.
    
    Args:
        token_scores: Dictionary of token scores from evaluation
        price_data: Dictionary of price DataFrames
        manual_tokens: Optional list of manually selected tokens
        selection_mode: Token selection mode
        
    Returns:
        dict: Portfolio options keyed by size
    """
    print(f"\nBuilding optimal portfolios with parallel processing...")
    
    try:
        builder = PortfolioBuilder(token_scores, price_data, n_cores=os.cpu_count())
        
        # Adjust max tokens based on available tokens
        max_tokens = min(10, len(token_scores))
        portfolio_options = builder.find_optimal_portfolio_size(token_scores, max_tokens)
        
        print(f"✓ Found {len(portfolio_options)} portfolio options")
        
        return portfolio_options, builder
        
    except Exception as e:
        print(f"✗ Error in portfolio construction: {e}")
        return None, None


def select_best_portfolio(portfolio_options, manual_tokens=None, selection_mode='auto'):
    """Select the best portfolio from options.
    
    Args:
        portfolio_options: Dictionary of portfolio options
        manual_tokens: Optional list of manually selected tokens
        selection_mode: Token selection mode
        
    Returns:
        tuple: (preferred_size, best_portfolio)
    """
    if not portfolio_options:
        return None, None
    
    # Select best portfolio (prefer smaller portfolios for manual selection)
    if selection_mode == 'manual' and manual_tokens:
        # For manual selection, prefer portfolio size close to number of manual tokens
        target_size = min(len(manual_tokens), max(portfolio_options.keys()))
        preferred_size = target_size if target_size in portfolio_options else max(portfolio_options.keys())
    else:
        preferred_size = 4 if 4 in portfolio_options else max(portfolio_options.keys())
    
    best_portfolio = portfolio_options[preferred_size]
    
    return preferred_size, best_portfolio


def test_allocation_methods(builder, tokens):
    """Test different allocation methods for a portfolio.
    
    Args:
        builder: PortfolioBuilder instance
        tokens: List of token symbols
        
    Returns:
        tuple: (allocation_results, best_method, best_allocations)
    """
    print(f"\nTesting allocation methods...")
    
    try:
        allocation_results = builder.test_all_allocation_methods(tokens)
        
        print(f"✓ Tested {len(allocation_results)} allocation methods")
        
        # Select best allocation method
        best_allocation_method = max(allocation_results.keys(),
                                    key=lambda k: allocation_results[k]['score'])
        best_allocations = allocation_results[best_allocation_method]['allocations']
        
        print(f"\nSelected allocation method: {best_allocation_method}")
        
        return allocation_results, best_allocation_method, best_allocations
        
    except Exception as e:
        print(f"✗ Error in allocation testing: {e}")
        # Fallback to equal weight
        best_allocation_method = 'equal_weight'
        best_allocations = {token: 1/len(tokens) for token in tokens}
        print(f"Using fallback equal weight allocation")
        
        return {}, best_allocation_method, best_allocations


def run_monte_carlo_optimization(builder, tokens, n_simulations=50000):
    """Run Monte Carlo optimization for portfolio weights.
    
    Args:
        builder: PortfolioBuilder instance
        tokens: List of token symbols
        n_simulations: Number of simulations to run
        
    Returns:
        dict or None: Monte Carlo optimization results
    """
    if len(tokens) > 6:
        print(f"\nSkipping Monte Carlo (portfolio too large: {len(tokens)} tokens)")
        return None
    
    print(f"\nRunning Monte Carlo optimization...")
    
    try:
        mc_result = builder.parallel_monte_carlo_optimization(
            tokens,
            n_simulations=n_simulations
        )
        
        print(f"✓ Monte Carlo optimization completed")
        print(f"  Optimal Sharpe ratio: {mc_result['sharpe_ratio']:.3f}")
        print(f"  Expected return: {mc_result['expected_return']:.2%}")
        print(f"  Expected volatility: {mc_result['expected_volatility']:.2%}")
        
        return mc_result
        
    except Exception as e:
        print(f"✗ Monte Carlo optimization failed: {e}")
        return None


def apply_monte_carlo_if_better(mc_result, allocation_results, best_allocation_method, 
                                best_allocations):
    """Apply Monte Carlo allocations if they are significantly better.
    
    Args:
        mc_result: Monte Carlo optimization results
        allocation_results: Dictionary of allocation method results
        best_allocation_method: Current best allocation method
        best_allocations: Current best allocations
        
    Returns:
        tuple: (updated_method, updated_allocations)
    """
    if mc_result is None:
        return best_allocation_method, best_allocations
    
    # Use MC-optimized allocations if significantly better
    current_score = allocation_results[best_allocation_method]['score']
    mc_score = mc_result['sharpe_ratio'] * 100  # Rough comparison
    
    if mc_result['sharpe_ratio'] > 1.0 and mc_score > current_score:
        print("  → Using Monte Carlo optimized allocations")
        return 'monte_carlo_optimized', mc_result['best_allocation']
    else:
        print(f"  → Keeping {best_allocation_method} allocations")
        return best_allocation_method, best_allocations


def run_backtests(price_data, portfolio_tokens, allocations, drift_thresholds=None):
    """Run backtests with multiple drift threshold configurations.
    
    Args:
        price_data: Dictionary of price DataFrames
        portfolio_tokens: List of tokens in the portfolio
        allocations: Dictionary of token allocations
        drift_thresholds: List of drift thresholds to test
        
    Returns:
        dict: Backtest results keyed by drift threshold
    """
    if drift_thresholds is None:
        drift_thresholds = [0.08, 0.10, 0.12, 0.15, 0.20, 0.25]
    
    print(f"\nRunning backtests with {len(drift_thresholds)} configurations...")
    print(f"Testing drift thresholds: {[f'{d:.1%}' for d in drift_thresholds]}")
    
    backtest_results = {}
    successful_backtests = 0
    
    for i, drift in enumerate(drift_thresholds):
        print(f"  [{i+1}/{len(drift_thresholds)}] Testing drift threshold: {drift:.1%}...", end="")
        
        portfolio_config = {
            'tokens': portfolio_tokens,
            'allocations': allocations,
            'drift_threshold': drift,
            'min_interval': 5,
            'max_interval': 21
        }
        
        try:
            engine = BacktestEngine(price_data, portfolio_config)
            results = engine.run_backtest(
                start_date=datetime.now() - timedelta(days=730),
                end_date=datetime.now() - timedelta(days=30),
                initial_capital=10000
            )
            backtest_results[drift] = results
            successful_backtests += 1
            print(" ✓")
            
        except Exception as e:
            print(f" ✗ ({str(e)[:50]}...)")
            continue
    
    print(f"✓ Backtesting completed")
    print(f"  Successful backtests: {successful_backtests}/{len(drift_thresholds)}")
    
    return backtest_results


def find_best_drift_threshold(backtest_results):
    """Find the best drift threshold based on Sharpe ratio.
    
    Args:
        backtest_results: Dictionary of backtest results
        
    Returns:
        float: Best drift threshold
    """
    if not backtest_results:
        return 0.12  # Default fallback
    
    # Find best by Sharpe ratio
    best_drift = max(backtest_results.keys(),
                    key=lambda d: backtest_results[d]['performance_metrics'].get('sharpe_ratio', -999))
    
    return best_drift
