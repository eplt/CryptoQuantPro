import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
from config.settings import TRANSACTION_FEE

class PerformanceAnalyzer:
    def __init__(self, backtest_results):
        self.results = backtest_results
        
    def generate_performance_report(self):
        """Generate comprehensive performance report"""
        metrics = self.results['performance_metrics']
        portfolio_values = self.results.get('portfolio_values', [])
        first_entry = portfolio_values[0] if portfolio_values else {}
        last_entry = portfolio_values[-1] if portfolio_values else {}
        start_date = first_entry.get('date')
        end_date = last_entry.get('date')
        total_days = (end_date - start_date).days if start_date and end_date else 0
        initial_capital = first_entry.get('total_value', 0)
        start_label = start_date.strftime('%Y-%m-%d') if start_date else "N/A"
        end_label = end_date.strftime('%Y-%m-%d') if end_date else "N/A"
        
        try:
            risk_metrics = self.calculate_risk_metrics()
        except Exception as e:
            print(f"Could not calculate additional risk metrics for report: {e}")
            risk_metrics = {}
        
        trade_analysis = self.generate_trade_analysis()
        if 'total_trades' in trade_analysis:
            trade_details = f"""
TRADE DETAILS:
- Average Trade Size: ${trade_analysis['avg_trade_size']:,.0f}
- Largest Trade: ${trade_analysis['largest_trade']:,.0f}
- Buy/Sell Ratio: {trade_analysis['buy_vs_sell_ratio']:.2f}
- Avg Days Between Rebalances: {trade_analysis['avg_days_between_rebalances']:.1f}
"""
        else:
            trade_details = f"""
TRADE DETAILS:
- {trade_analysis.get('message', 'No trades executed')}
"""
        
        risk_details = ""
        if risk_metrics:
            risk_details = f"""
ADDITIONAL RISK METRICS:
- Value at Risk (5%): {risk_metrics['value_at_risk_5pct']:.3f}
- Conditional VaR (5%): {risk_metrics['conditional_var_5pct']:.3f}
- Skewness: {risk_metrics['skewness']:.3f}
- Kurtosis: {risk_metrics['kurtosis']:.3f}
- Positive Periods: {risk_metrics['positive_periods']:.1%}
- Max Consecutive Losses: {risk_metrics['max_consecutive_losses']} periods
- Calmar Ratio: {risk_metrics['calmar_ratio']:.3f}
"""
        
        report = f"""
=== PORTFOLIO PERFORMANCE REPORT ===

BACKTEST WINDOW:
- Start Date: {start_label}
- End Date: {end_label}
- Total Days: {total_days}
- Starting Capital: ${initial_capital:,.0f}
- Fee Rate (per trade): {TRANSACTION_FEE:.3%}

RETURNS:
- Total Return: {metrics['total_return']:.2%}
- Annualized Return: {metrics['annualized_return']:.2%}
- Benchmark Return: {metrics['benchmark_return']:.2%}
- Alpha: {metrics['alpha']:.2%}

RISK METRICS:
- Volatility: {metrics['volatility']:.2%}
- Sharpe Ratio: {metrics['sharpe_ratio']:.3f}
- Maximum Drawdown: {metrics['max_drawdown']:.2%}

TRADING METRICS:
- Total Trades: {metrics['total_trades']}
- Rebalances: {metrics['rebalance_frequency']}
- Total Fees: ${metrics['total_fees']:,.2f}
- Fees as % of Portfolio: {metrics['fees_pct_of_portfolio']:.3%}

EFFICIENCY:
- Return per Unit Risk: {metrics['annualized_return']/metrics['volatility']:.3f}
- Return per Dollar of Fees: {metrics['total_return']/(metrics['fees_pct_of_portfolio']+0.001):.1f}
{trade_details}{risk_details}
"""
        
        return report
    
    def compare_strategies(self, other_results: List[Dict]):
        """Compare multiple strategy results"""
        comparison_data = []
        
        # Add current strategy
        comparison_data.append({
            'Strategy': 'Current',
            **self.results['performance_metrics']
        })
        
        # Add other strategies
        for i, other in enumerate(other_results):
            comparison_data.append({
                'Strategy': f'Strategy_{i+1}',
                **other['performance_metrics']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Calculate rankings
        ranking_cols = ['total_return', 'sharpe_ratio', 'alpha']
        for col in ranking_cols:
            comparison_df[f'{col}_rank'] = comparison_df[col].rank(ascending=False)
        
        return comparison_df
    
    def plot_performance_charts(self, save_path=None):
        """Generate performance visualization charts"""
        try:
            portfolio_df = pd.DataFrame(self.results['portfolio_values'])
            portfolio_df.set_index('date', inplace=True)
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Portfolio value over time
            axes[0, 0].plot(portfolio_df.index, portfolio_df['total_value'])
            axes[0, 0].set_title('Portfolio Value Over Time')
            axes[0, 0].set_ylabel('Portfolio Value ($)')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # Rolling Sharpe ratio
            returns = portfolio_df['total_value'].pct_change().dropna()
            rolling_sharpe = returns.rolling(30).mean() / returns.rolling(30).std() * np.sqrt(365)
            axes[0, 1].plot(portfolio_df.index[30:], rolling_sharpe.dropna())
            axes[0, 1].set_title('30-Day Rolling Sharpe Ratio')
            axes[0, 1].axhline(y=1, color='r', linestyle='--', label='Target=1.0')
            axes[0, 1].legend()
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Drawdown chart
            peak = portfolio_df['total_value'].expanding().max()
            drawdown = (portfolio_df['total_value'] - peak) / peak
            axes[1, 0].fill_between(portfolio_df.index, 0, drawdown, alpha=0.3, color='red')
            axes[1, 0].set_title('Portfolio Drawdown')
            axes[1, 0].set_ylabel('Drawdown %')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Weight allocation over time
            weights_df = pd.DataFrame(self.results['weights_history'])
            weights_df.set_index('date', inplace=True)
            
            for token in weights_df.columns:
                axes[1, 1].plot(weights_df.index, weights_df[token], label=token)
            axes[1, 1].set_title('Portfolio Weights Over Time')
            axes[1, 1].set_ylabel('Weight')
            axes[1, 1].legend()
            axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            return fig
            
        except Exception as e:
            print(f"Error creating performance charts: {e}")
            return None
    
    def calculate_risk_metrics(self):
        """Calculate additional risk metrics"""
        portfolio_df = pd.DataFrame(self.results['portfolio_values'])
        portfolio_df.set_index('date', inplace=True)
        returns = portfolio_df['total_value'].pct_change().dropna()
        
        risk_metrics = {
            'value_at_risk_5pct': returns.quantile(0.05),
            'conditional_var_5pct': returns[returns <= returns.quantile(0.05)].mean(),
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis(),
            'positive_periods': (returns > 0).sum() / len(returns),
            'max_consecutive_losses': self._calculate_max_consecutive_losses(returns),
            'calmar_ratio': self.results['performance_metrics']['annualized_return'] / abs(self.results['performance_metrics']['max_drawdown']),
        }
        
        return risk_metrics
    
    def _calculate_max_consecutive_losses(self, returns):
        """Calculate maximum consecutive losing periods"""
        losing_periods = (returns < 0).astype(int)
        consecutive_losses = []
        current_streak = 0
        
        for loss in losing_periods:
            if loss == 1:
                current_streak += 1
            else:
                if current_streak > 0:
                    consecutive_losses.append(current_streak)
                current_streak = 0
        
        if current_streak > 0:
            consecutive_losses.append(current_streak)
        
        return max(consecutive_losses) if consecutive_losses else 0
    
    def generate_trade_analysis(self):
        """Analyze trading patterns and efficiency"""
        if not self.results['trades']:
            return {"message": "No trades executed"}
        
        trades_df = pd.DataFrame(self.results['trades'])
        
        trade_analysis = {
            'total_trades': len(trades_df),
            'total_fees': trades_df['fee'].sum(),
            'avg_trade_size': trades_df['amount'].mean(),
            'largest_trade': trades_df['amount'].max(),
            'trades_by_token': trades_df['token'].value_counts().to_dict(),
            'buy_vs_sell_ratio': (trades_df['side'] == 'buy').sum() / len(trades_df),
            'avg_days_between_rebalances': self._calculate_avg_rebalance_interval(),
        }
        
        return trade_analysis
    
    def _calculate_avg_rebalance_interval(self):
        """Calculate average days between rebalances"""
        if len(self.results['rebalance_dates']) < 2:
            return 0
        
        intervals = []
        dates = sorted(self.results['rebalance_dates'])
        
        for i in range(1, len(dates)):
            interval = (dates[i] - dates[i-1]).days
            intervals.append(interval)
        
        return sum(intervals) / len(intervals) if intervals else 0
    
    def export_results(self, filename='backtest_results.csv'):
        """Export results to CSV for further analysis"""
        try:
            # Portfolio values
            portfolio_df = pd.DataFrame(self.results['portfolio_values'])
            portfolio_df.to_csv(f'portfolio_{filename}', index=False)
            
            # Weights history
            weights_df = pd.DataFrame(self.results['weights_history'])
            weights_df.to_csv(f'weights_{filename}', index=False)
            
            # Trades
            if self.results['trades']:
                trades_df = pd.DataFrame(self.results['trades'])
                trades_df.to_csv(f'trades_{filename}', index=False)
            
            # Performance metrics
            metrics_df = pd.DataFrame([self.results['performance_metrics']])
            metrics_df.to_csv(f'metrics_{filename}', index=False)
            
            print(f"Results exported to CSV files with prefix: {filename}")
            
        except Exception as e:
            print(f"Error exporting results: {e}")
