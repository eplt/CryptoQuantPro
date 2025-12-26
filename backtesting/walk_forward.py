"""Walk-forward analysis module for robust backtesting."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from backtesting.backtest_engine import BacktestEngine


class WalkForwardAnalyzer:
    """Implements walk-forward analysis for portfolio strategies."""
    
    def __init__(self, price_data, training_window_days=365, test_window_days=90, 
                 step_days=30):
        """Initialize walk-forward analyzer.
        
        Args:
            price_data: Dictionary of price DataFrames
            training_window_days: Days of data for training/optimization
            test_window_days: Days of data for testing
            step_days: Days to step forward between windows
        """
        self.price_data = price_data
        self.training_window = training_window_days
        self.test_window = test_window_days
        self.step_days = step_days
    
    def generate_windows(self, start_date, end_date):
        """Generate training/test window pairs.
        
        Args:
            start_date: Start date for analysis
            end_date: End date for analysis
            
        Returns:
            list: List of (train_start, train_end, test_start, test_end) tuples
        """
        windows = []
        current_start = start_date
        
        while True:
            train_start = current_start
            train_end = train_start + timedelta(days=self.training_window)
            test_start = train_end
            test_end = test_start + timedelta(days=self.test_window)
            
            if test_end > end_date:
                break
            
            windows.append((train_start, train_end, test_start, test_end))
            current_start += timedelta(days=self.step_days)
        
        return windows
    
    def run_walk_forward(self, portfolio_config, start_date=None, end_date=None):
        """Run walk-forward analysis.
        
        Args:
            portfolio_config: Portfolio configuration dictionary
            start_date: Start date (default: earliest available)
            end_date: End date (default: latest available)
            
        Returns:
            dict: Walk-forward analysis results
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=730)
        if end_date is None:
            end_date = datetime.now()
        
        windows = self.generate_windows(start_date, end_date)
        
        print(f"Running walk-forward analysis with {len(windows)} windows...")
        print(f"  Training: {self.training_window} days")
        print(f"  Testing: {self.test_window} days")
        print(f"  Step: {self.step_days} days")
        
        results = {
            'windows': [],
            'in_sample_metrics': [],
            'out_of_sample_metrics': [],
            'summary': {}
        }
        
        for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
            print(f"\n  Window {i+1}/{len(windows)}:")
            print(f"    Train: {train_start.date()} to {train_end.date()}")
            print(f"    Test:  {test_start.date()} to {test_end.date()}")
            
            try:
                # Run in-sample backtest (training period)
                engine_train = BacktestEngine(self.price_data, portfolio_config)
                train_results = engine_train.run_backtest(
                    start_date=train_start,
                    end_date=train_end,
                    initial_capital=10000
                )
                
                # Run out-of-sample backtest (test period)
                engine_test = BacktestEngine(self.price_data, portfolio_config)
                test_results = engine_test.run_backtest(
                    start_date=test_start,
                    end_date=test_end,
                    initial_capital=10000
                )
                
                results['windows'].append({
                    'train_period': (train_start, train_end),
                    'test_period': (test_start, test_end)
                })
                results['in_sample_metrics'].append(train_results['performance_metrics'])
                results['out_of_sample_metrics'].append(test_results['performance_metrics'])
                
                print(f"    ✓ Train Sharpe: {train_results['performance_metrics']['sharpe_ratio']:.3f}")
                print(f"    ✓ Test Sharpe:  {test_results['performance_metrics']['sharpe_ratio']:.3f}")
                
            except Exception as e:
                print(f"    ✗ Error: {str(e)[:50]}")
                continue
        
        # Calculate summary statistics
        if results['out_of_sample_metrics']:
            oos_sharpes = [m['sharpe_ratio'] for m in results['out_of_sample_metrics']]
            oos_returns = [m['annualized_return'] for m in results['out_of_sample_metrics']]
            
            results['summary'] = {
                'total_windows': len(windows),
                'successful_windows': len(results['out_of_sample_metrics']),
                'avg_oos_sharpe': np.mean(oos_sharpes),
                'std_oos_sharpe': np.std(oos_sharpes),
                'avg_oos_return': np.mean(oos_returns),
                'std_oos_return': np.std(oos_returns),
                'consistency_score': np.mean([1 if s > 0 else 0 for s in oos_sharpes])
            }
        
        print(f"\n✓ Walk-forward analysis complete")
        print(f"  Avg OOS Sharpe: {results['summary'].get('avg_oos_sharpe', 0):.3f}")
        print(f"  Consistency: {results['summary'].get('consistency_score', 0):.1%}")
        
        return results
    
    def compare_strategies(self, strategy_configs, start_date=None, end_date=None):
        """Compare multiple strategies using walk-forward analysis.
        
        Args:
            strategy_configs: Dictionary of strategy configurations
            start_date: Start date
            end_date: End date
            
        Returns:
            dict: Comparison results for each strategy
        """
        comparison_results = {}
        
        print(f"Comparing {len(strategy_configs)} strategies...")
        
        for strategy_name, config in strategy_configs.items():
            print(f"\nAnalyzing strategy: {strategy_name}")
            results = self.run_walk_forward(config, start_date, end_date)
            comparison_results[strategy_name] = results
        
        # Rank strategies
        rankings = sorted(
            comparison_results.items(),
            key=lambda x: x[1]['summary'].get('avg_oos_sharpe', -999),
            reverse=True
        )
        
        print(f"\n=== STRATEGY RANKINGS ===")
        for rank, (name, results) in enumerate(rankings, 1):
            summary = results['summary']
            print(f"{rank}. {name}")
            print(f"   OOS Sharpe: {summary.get('avg_oos_sharpe', 0):.3f}")
            print(f"   OOS Return: {summary.get('avg_oos_return', 0):.2%}")
            print(f"   Consistency: {summary.get('consistency_score', 0):.1%}")
        
        return comparison_results
