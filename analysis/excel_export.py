"""Excel export module for generating detailed Excel reports."""

import pandas as pd
import numpy as np
from datetime import datetime
import os


class ExcelReportGenerator:
    """Generates comprehensive Excel reports with multiple sheets."""
    
    def __init__(self, output_dir='results'):
        """Initialize Excel report generator.
        
        Args:
            output_dir: Directory for output files
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_comprehensive_report(self, analysis_data, filename=None):
        """Generate comprehensive Excel report with multiple sheets.
        
        Args:
            analysis_data: Dictionary containing all analysis results
            filename: Optional custom filename
            
        Returns:
            str: Path to generated Excel file
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M')
            filename = f'cryptoquant_analysis_{timestamp}.xlsx'
        
        filepath = os.path.join(self.output_dir, filename)
        
        print(f"Generating Excel report: {filename}")
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Sheet 1: Executive Summary
            self._write_executive_summary(writer, analysis_data)
            
            # Sheet 2: Token Scores
            if 'token_scores' in analysis_data:
                self._write_token_scores(writer, analysis_data['token_scores'])
            
            # Sheet 3: Portfolio Options
            if 'portfolio_options' in analysis_data:
                self._write_portfolio_options(writer, analysis_data['portfolio_options'])
            
            # Sheet 4: Allocation Comparison
            if 'allocation_results' in analysis_data:
                self._write_allocation_comparison(writer, analysis_data['allocation_results'])
            
            # Sheet 5: Backtest Results
            if 'backtest_results' in analysis_data:
                self._write_backtest_results(writer, analysis_data['backtest_results'])
            
            # Sheet 6: Risk Metrics
            if 'backtest_results' in analysis_data:
                self._write_risk_metrics(writer, analysis_data['backtest_results'])
            
            # Sheet 7: Performance Timeline
            if 'backtest_results' in analysis_data:
                self._write_performance_timeline(writer, analysis_data['backtest_results'])
        
        print(f"✓ Excel report generated: {filepath}")
        return filepath
    
    def _write_executive_summary(self, writer, analysis_data):
        """Write executive summary sheet."""
        summary_data = []
        
        summary_data.append(['CryptoQuant Pro Analysis Report', ''])
        summary_data.append(['Generated:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
        summary_data.append(['', ''])
        
        # Portfolio details
        if 'best_portfolio' in analysis_data:
            portfolio = analysis_data['best_portfolio']
            summary_data.append(['Portfolio Configuration', ''])
            summary_data.append(['Number of Tokens:', len(portfolio.get('tokens', []))])
            summary_data.append(['Tokens:', ', '.join(portfolio.get('tokens', []))])
            summary_data.append(['Portfolio Score:', f"{portfolio.get('score', 0):.2f}"])
            summary_data.append(['', ''])
        
        # Allocation method
        if 'best_allocation_method' in analysis_data:
            summary_data.append(['Allocation Method:', analysis_data['best_allocation_method']])
            summary_data.append(['', ''])
        
        # Best performance metrics
        if 'best_drift_metrics' in analysis_data:
            metrics = analysis_data['best_drift_metrics']
            summary_data.append(['Best Performance Metrics', ''])
            summary_data.append(['Annual Return:', f"{metrics.get('annualized_return', 0):.2%}"])
            summary_data.append(['Volatility:', f"{metrics.get('volatility', 0):.2%}"])
            summary_data.append(['Sharpe Ratio:', f"{metrics.get('sharpe_ratio', 0):.3f}"])
            summary_data.append(['Max Drawdown:', f"{metrics.get('max_drawdown', 0):.2%}"])
            summary_data.append(['Alpha:', f"{metrics.get('alpha', 0):.2%}"])
            summary_data.append(['Beta:', f"{metrics.get('beta', 0):.3f}"])
        
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_excel(writer, sheet_name='Executive Summary', index=False, header=False)
    
    def _write_token_scores(self, writer, token_scores):
        """Write token scores sheet."""
        scores_data = []
        
        for token, data in sorted(token_scores.items(), 
                                  key=lambda x: x[1]['final_score'], 
                                  reverse=True):
            metrics = data.get('metrics', {})
            scores_data.append({
                'Token': token,
                'Final Score': data['final_score'],
                'Volatility': metrics.get('volatility', 0),
                'Sharpe Ratio': metrics.get('sharpe_ratio', 0),
                'Market Cap': metrics.get('market_cap', 0),
                'Liquidity Ratio': metrics.get('liquidity_ratio', 0),
                '1Y Return': metrics.get('returns_1y', 0),
                'Max Drawdown': metrics.get('max_drawdown', 0)
            })
        
        df_scores = pd.DataFrame(scores_data)
        df_scores.to_excel(writer, sheet_name='Token Scores', index=False)
    
    def _write_portfolio_options(self, writer, portfolio_options):
        """Write portfolio options sheet."""
        options_data = []
        
        for size, portfolio in sorted(portfolio_options.items()):
            options_data.append({
                'Portfolio Size': size,
                'Score': portfolio['score'],
                'Tokens': ', '.join(portfolio['tokens']),
                'Avg Correlation': portfolio.get('avg_correlation', 0),
                'Expected Sharpe': portfolio.get('expected_sharpe', 0)
            })
        
        df_options = pd.DataFrame(options_data)
        df_options.to_excel(writer, sheet_name='Portfolio Options', index=False)
    
    def _write_allocation_comparison(self, writer, allocation_results):
        """Write allocation comparison sheet."""
        alloc_data = []
        
        for method, result in allocation_results.items():
            row = {'Method': method, 'Score': result['score']}
            for token, weight in result['allocations'].items():
                row[token] = weight
            alloc_data.append(row)
        
        df_alloc = pd.DataFrame(alloc_data)
        df_alloc.to_excel(writer, sheet_name='Allocation Comparison', index=False)
    
    def _write_backtest_results(self, writer, backtest_results):
        """Write backtest results sheet."""
        results_data = []
        
        for drift, result in sorted(backtest_results.items()):
            metrics = result['performance_metrics']
            results_data.append({
                'Drift Threshold': f"{drift:.1%}",
                'Annual Return': metrics.get('annualized_return', 0),
                'Volatility': metrics.get('volatility', 0),
                'Sharpe Ratio': metrics.get('sharpe_ratio', 0),
                'Max Drawdown': metrics.get('max_drawdown', 0),
                'Calmar Ratio': metrics.get('calmar_ratio', 0),
                'Win Rate': metrics.get('win_rate', 0),
                'Avg Win': metrics.get('avg_win', 0),
                'Avg Loss': metrics.get('avg_loss', 0)
            })
        
        df_results = pd.DataFrame(results_data)
        df_results.to_excel(writer, sheet_name='Backtest Results', index=False)
    
    def _write_risk_metrics(self, writer, backtest_results):
        """Write risk metrics sheet."""
        risk_data = []
        
        for drift, result in sorted(backtest_results.items()):
            metrics = result['performance_metrics']
            risk_data.append({
                'Drift Threshold': f"{drift:.1%}",
                'VaR 95%': metrics.get('var_95', 0),
                'CVaR 95%': metrics.get('cvar_95', 0),
                'Max Drawdown': metrics.get('max_drawdown', 0),
                'Avg Drawdown': metrics.get('avg_drawdown', 0),
                'Downside Deviation': metrics.get('downside_deviation', 0),
                'Sortino Ratio': metrics.get('sortino_ratio', 0),
                'Beta': metrics.get('beta', 0),
                'Alpha': metrics.get('alpha', 0)
            })
        
        df_risk = pd.DataFrame(risk_data)
        df_risk.to_excel(writer, sheet_name='Risk Metrics', index=False)
    
    def _write_performance_timeline(self, writer, backtest_results):
        """Write performance timeline sheet (if available)."""
        # Get the best performing drift threshold
        best_drift = max(backtest_results.keys(), 
                        key=lambda d: backtest_results[d]['performance_metrics'].get('sharpe_ratio', -999))
        
        result = backtest_results[best_drift]
        
        if 'portfolio_values' in result:
            timeline_data = []
            for i, value in enumerate(result['portfolio_values']):
                timeline_data.append({
                    'Day': i,
                    'Portfolio Value': value,
                    'Return': (value / result['portfolio_values'][0] - 1) if i > 0 else 0
                })
            
            df_timeline = pd.DataFrame(timeline_data)
            df_timeline.to_excel(writer, sheet_name='Performance Timeline', index=False)
