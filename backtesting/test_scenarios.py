"""Test scenarios module for comprehensive portfolio evaluation."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from backtesting.backtest_engine import BacktestEngine


class TestScenarios:
    """Implements various test scenarios for portfolio evaluation."""
    
    def __init__(self, price_data):
        """Initialize test scenarios.
        
        Args:
            price_data: Dictionary of price DataFrames
        """
        self.price_data = price_data
        self.scenarios = self._define_scenarios()
    
    def _define_scenarios(self):
        """Define standard test scenarios.
        
        Returns:
            dict: Dictionary of scenario definitions
        """
        return {
            'bull_market': {
                'name': 'Bull Market',
                'description': 'Strong uptrend with low volatility',
                'conditions': {
                    'min_return': 0.20,  # 20% minimum return
                    'max_volatility': 0.50
                }
            },
            'bear_market': {
                'name': 'Bear Market',
                'description': 'Sustained downtrend',
                'conditions': {
                    'max_return': -0.10,  # Negative return
                    'min_volatility': 0.30
                }
            },
            'high_volatility': {
                'name': 'High Volatility',
                'description': 'Extreme price swings',
                'conditions': {
                    'min_volatility': 0.80
                }
            },
            'low_volatility': {
                'name': 'Low Volatility',
                'description': 'Stable market conditions',
                'conditions': {
                    'max_volatility': 0.30
                }
            },
            'crash_recovery': {
                'name': 'Crash & Recovery',
                'description': 'Sharp decline followed by recovery',
                'conditions': {
                    'min_drawdown': -0.30,  # At least 30% drawdown
                    'recovery': True
                }
            },
            'sideways': {
                'name': 'Sideways Market',
                'description': 'Range-bound with no clear trend',
                'conditions': {
                    'max_abs_return': 0.15,  # Less than 15% absolute return
                    'max_volatility': 0.40
                }
            }
        }
    
    def identify_periods(self, window_days=90, min_periods=3):
        """Identify historical periods matching each scenario.
        
        Args:
            window_days: Rolling window size for analysis
            min_periods: Minimum number of matching periods required
            
        Returns:
            dict: Periods matching each scenario
        """
        # Get first token for reference dates
        reference_token = list(self.price_data.keys())[0]
        df = self.price_data[reference_token]
        
        identified_periods = {name: [] for name in self.scenarios.keys()}
        
        print(f"Identifying scenario periods (window: {window_days} days)...")
        
        # Slide window through historical data
        for i in range(len(df) - window_days):
            window_start = df.index[i]
            window_end = df.index[i + window_days]
            
            # Calculate metrics for this window
            window_returns = []
            window_volatilities = []
            window_drawdowns = []
            
            for token in self.price_data.keys():
                token_df = self.price_data[token].loc[window_start:window_end]
                
                if len(token_df) < window_days * 0.9:  # Need at least 90% of data
                    continue
                
                returns = np.log(token_df['close'] / token_df['close'].shift(1)).dropna()
                total_return = (token_df['close'].iloc[-1] / token_df['close'].iloc[0]) - 1
                volatility = returns.std() * np.sqrt(365)
                
                # Calculate drawdown
                cummax = token_df['close'].cummax()
                drawdown = (token_df['close'] - cummax) / cummax
                max_drawdown = drawdown.min()
                
                window_returns.append(total_return)
                window_volatilities.append(volatility)
                window_drawdowns.append(max_drawdown)
            
            if not window_returns:
                continue
            
            avg_return = np.mean(window_returns)
            avg_volatility = np.mean(window_volatilities)
            avg_drawdown = np.mean(window_drawdowns)
            
            # Check each scenario
            for scenario_name, scenario in self.scenarios.items():
                conditions = scenario['conditions']
                match = True
                
                if 'min_return' in conditions and avg_return < conditions['min_return']:
                    match = False
                if 'max_return' in conditions and avg_return > conditions['max_return']:
                    match = False
                if 'min_volatility' in conditions and avg_volatility < conditions['min_volatility']:
                    match = False
                if 'max_volatility' in conditions and avg_volatility > conditions['max_volatility']:
                    match = False
                if 'min_drawdown' in conditions and avg_drawdown > conditions['min_drawdown']:
                    match = False
                if 'max_abs_return' in conditions and abs(avg_return) > conditions['max_abs_return']:
                    match = False
                
                if match:
                    identified_periods[scenario_name].append({
                        'start': window_start,
                        'end': window_end,
                        'avg_return': avg_return,
                        'avg_volatility': avg_volatility,
                        'avg_drawdown': avg_drawdown
                    })
        
        # Report findings
        print(f"\n=== IDENTIFIED SCENARIO PERIODS ===")
        for scenario_name, periods in identified_periods.items():
            print(f"\n{self.scenarios[scenario_name]['name']}: {len(periods)} periods")
            if periods:
                for i, period in enumerate(periods[:3], 1):  # Show first 3
                    print(f"  {i}. {period['start'].date()} to {period['end'].date()}")
                    print(f"     Return: {period['avg_return']:.2%}, Vol: {period['avg_volatility']:.2%}")
        
        return identified_periods
    
    def test_portfolio_in_scenarios(self, portfolio_config, identified_periods=None):
        """Test portfolio performance in each scenario.
        
        Args:
            portfolio_config: Portfolio configuration
            identified_periods: Pre-identified periods (if None, will identify)
            
        Returns:
            dict: Performance results for each scenario
        """
        if identified_periods is None:
            identified_periods = self.identify_periods()
        
        print(f"\n=== TESTING PORTFOLIO IN SCENARIOS ===")
        
        results = {}
        
        for scenario_name, periods in identified_periods.items():
            if not periods:
                print(f"\n{scenario_name}: No matching periods found")
                continue
            
            print(f"\n{scenario_name}: Testing {len(periods)} periods...")
            
            scenario_results = []
            
            for period in periods:
                try:
                    engine = BacktestEngine(self.price_data, portfolio_config)
                    backtest_result = engine.run_backtest(
                        start_date=period['start'],
                        end_date=period['end'],
                        initial_capital=10000
                    )
                    scenario_results.append(backtest_result['performance_metrics'])
                except Exception as e:
                    continue
            
            if scenario_results:
                # Aggregate results
                results[scenario_name] = {
                    'n_periods': len(scenario_results),
                    'avg_return': np.mean([r['annualized_return'] for r in scenario_results]),
                    'avg_sharpe': np.mean([r['sharpe_ratio'] for r in scenario_results]),
                    'avg_drawdown': np.mean([r['max_drawdown'] for r in scenario_results]),
                    'win_rate': np.mean([1 if r['annualized_return'] > 0 else 0 for r in scenario_results])
                }
                
                print(f"  Avg Return: {results[scenario_name]['avg_return']:.2%}")
                print(f"  Avg Sharpe: {results[scenario_name]['avg_sharpe']:.3f}")
                print(f"  Win Rate: {results[scenario_name]['win_rate']:.1%}")
        
        return results
    
    def generate_scenario_report(self, portfolio_config):
        """Generate comprehensive scenario analysis report.
        
        Args:
            portfolio_config: Portfolio configuration
            
        Returns:
            dict: Complete scenario analysis
        """
        print(f"\n{'='*60}")
        print("COMPREHENSIVE SCENARIO ANALYSIS")
        print(f"{'='*60}")
        
        # Identify periods
        identified_periods = self.identify_periods()
        
        # Test portfolio
        test_results = self.test_portfolio_in_scenarios(portfolio_config, identified_periods)
        
        # Generate summary
        summary = {
            'scenarios': self.scenarios,
            'identified_periods': identified_periods,
            'test_results': test_results,
            'overall_robustness': self._calculate_robustness_score(test_results)
        }
        
        print(f"\n{'='*60}")
        print(f"OVERALL ROBUSTNESS SCORE: {summary['overall_robustness']:.1f}/100")
        print(f"{'='*60}")
        
        return summary
    
    def _calculate_robustness_score(self, test_results):
        """Calculate overall portfolio robustness score.
        
        Args:
            test_results: Results from scenario testing
            
        Returns:
            float: Robustness score (0-100)
        """
        if not test_results:
            return 0.0
        
        scores = []
        
        for scenario_name, results in test_results.items():
            # Score based on Sharpe ratio, win rate, and consistency
            sharpe_score = min(100, max(0, (results['avg_sharpe'] + 1) * 30))
            win_rate_score = results['win_rate'] * 100
            scenario_score = (sharpe_score + win_rate_score) / 2
            scores.append(scenario_score)
        
        return np.mean(scores) if scores else 0.0
