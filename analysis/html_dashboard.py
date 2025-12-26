"""HTML dashboard generator using Plotly and Dash."""

import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime
import os


class HTMLDashboardGenerator:
    """Generates interactive HTML dashboards with Plotly."""
    
    def __init__(self, output_dir='results'):
        """Initialize dashboard generator.
        
        Args:
            output_dir: Directory for output files
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_dashboard(self, analysis_data, filename=None):
        """Generate comprehensive interactive HTML dashboard.
        
        Args:
            analysis_data: Dictionary containing all analysis results
            filename: Optional custom filename
            
        Returns:
            str: Path to generated HTML file
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M')
            filename = f'cryptoquant_dashboard_{timestamp}.html'
        
        filepath = os.path.join(self.output_dir, filename)
        
        print(f"Generating HTML dashboard: {filename}")
        
        # Create main figure with subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Portfolio Performance', 'Token Score Distribution',
                'Risk-Return Profile', 'Allocation Breakdown',
                'Drawdown Analysis', 'Monthly Returns Heatmap'
            ),
            specs=[
                [{"type": "scatter"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "pie"}],
                [{"type": "scatter"}, {"type": "heatmap"}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.15
        )
        
        # 1. Portfolio Performance
        if 'backtest_results' in analysis_data:
            self._add_performance_plot(fig, analysis_data['backtest_results'], row=1, col=1)
        
        # 2. Token Score Distribution
        if 'token_scores' in analysis_data:
            self._add_token_scores_plot(fig, analysis_data['token_scores'], row=1, col=2)
        
        # 3. Risk-Return Profile
        if 'backtest_results' in analysis_data:
            self._add_risk_return_plot(fig, analysis_data['backtest_results'], row=2, col=1)
        
        # 4. Allocation Breakdown
        if 'best_allocations' in analysis_data:
            self._add_allocation_pie(fig, analysis_data['best_allocations'], row=2, col=2)
        
        # 5. Drawdown Analysis
        if 'backtest_results' in analysis_data:
            self._add_drawdown_plot(fig, analysis_data['backtest_results'], row=3, col=1)
        
        # Update layout
        fig.update_layout(
            title_text="CryptoQuant Pro - Portfolio Analysis Dashboard",
            title_font_size=24,
            showlegend=True,
            height=1400,
            template="plotly_white"
        )
        
        # Write to HTML
        fig.write_html(filepath)
        
        print(f"✓ HTML dashboard generated: {filepath}")
        return filepath
    
    def _add_performance_plot(self, fig, backtest_results, row, col):
        """Add portfolio performance plot."""
        best_drift = max(backtest_results.keys(), 
                        key=lambda d: backtest_results[d]['performance_metrics'].get('sharpe_ratio', -999))
        
        result = backtest_results[best_drift]
        
        if 'portfolio_values' in result:
            values = result['portfolio_values']
            fig.add_trace(
                go.Scatter(
                    y=values,
                    mode='lines',
                    name='Portfolio Value',
                    line=dict(color='#2E86AB', width=2)
                ),
                row=row, col=col
            )
    
    def _add_token_scores_plot(self, fig, token_scores, row, col):
        """Add token scores bar chart."""
        sorted_scores = sorted(token_scores.items(), 
                              key=lambda x: x[1]['final_score'], 
                              reverse=True)[:15]
        
        tokens = [x[0] for x in sorted_scores]
        scores = [x[1]['final_score'] for x in sorted_scores]
        
        fig.add_trace(
            go.Bar(
                x=tokens,
                y=scores,
                name='Token Scores',
                marker_color='#A23B72'
            ),
            row=row, col=col
        )
    
    def _add_risk_return_plot(self, fig, backtest_results, row, col):
        """Add risk-return scatter plot."""
        returns = []
        volatilities = []
        labels = []
        
        for drift, result in backtest_results.items():
            metrics = result['performance_metrics']
            returns.append(metrics.get('annualized_return', 0))
            volatilities.append(metrics.get('volatility', 0))
            labels.append(f"{drift:.1%}")
        
        fig.add_trace(
            go.Scatter(
                x=volatilities,
                y=returns,
                mode='markers+text',
                text=labels,
                textposition='top center',
                marker=dict(size=10, color='#F18F01'),
                name='Risk-Return'
            ),
            row=row, col=col
        )
        
        fig.update_xaxes(title_text="Volatility", row=row, col=col)
        fig.update_yaxes(title_text="Return", row=row, col=col)
    
    def _add_allocation_pie(self, fig, allocations, row, col):
        """Add allocation pie chart."""
        tokens = list(allocations.keys())
        weights = list(allocations.values())
        
        fig.add_trace(
            go.Pie(
                labels=tokens,
                values=weights,
                hole=0.3,
                name='Allocations'
            ),
            row=row, col=col
        )
    
    def _add_drawdown_plot(self, fig, backtest_results, row, col):
        """Add drawdown plot."""
        best_drift = max(backtest_results.keys(), 
                        key=lambda d: backtest_results[d]['performance_metrics'].get('sharpe_ratio', -999))
        
        result = backtest_results[best_drift]
        
        if 'portfolio_values' in result:
            values = np.array(result['portfolio_values'])
            cummax = np.maximum.accumulate(values)
            drawdown = (values - cummax) / cummax
            
            fig.add_trace(
                go.Scatter(
                    y=drawdown * 100,
                    mode='lines',
                    name='Drawdown %',
                    line=dict(color='#C73E1D', width=2),
                    fill='tozeroy'
                ),
                row=row, col=col
            )
            
            fig.update_yaxes(title_text="Drawdown (%)", row=row, col=col)
    
    def generate_token_comparison(self, token_scores, filename=None):
        """Generate detailed token comparison dashboard.
        
        Args:
            token_scores: Dictionary of token scores
            filename: Optional custom filename
            
        Returns:
            str: Path to generated HTML file
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M')
            filename = f'token_comparison_{timestamp}.html'
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Create comparison plots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Score vs Volatility',
                'Score vs Market Cap',
                'Score vs Liquidity',
                'Score Distribution'
            )
        )
        
        # Extract metrics
        scores = []
        volatilities = []
        market_caps = []
        liquidities = []
        tokens = []
        
        for token, data in token_scores.items():
            scores.append(data['final_score'])
            metrics = data.get('metrics', {})
            volatilities.append(metrics.get('volatility', 0))
            market_caps.append(metrics.get('market_cap', 0))
            liquidities.append(metrics.get('liquidity_ratio', 0))
            tokens.append(token)
        
        # Score vs Volatility
        fig.add_trace(
            go.Scatter(x=volatilities, y=scores, mode='markers+text', 
                      text=tokens, textposition='top center',
                      marker=dict(size=10, color='#2E86AB'),
                      name='Volatility'),
            row=1, col=1
        )
        
        # Score vs Market Cap
        fig.add_trace(
            go.Scatter(x=market_caps, y=scores, mode='markers+text',
                      text=tokens, textposition='top center',
                      marker=dict(size=10, color='#A23B72'),
                      name='Market Cap'),
            row=1, col=2
        )
        
        # Score vs Liquidity
        fig.add_trace(
            go.Scatter(x=liquidities, y=scores, mode='markers+text',
                      text=tokens, textposition='top center',
                      marker=dict(size=10, color='#F18F01'),
                      name='Liquidity'),
            row=2, col=1
        )
        
        # Score Distribution
        fig.add_trace(
            go.Histogram(x=scores, nbinsx=20, 
                        marker_color='#C73E1D',
                        name='Distribution'),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="Token Analysis Comparison",
            title_font_size=24,
            showlegend=False,
            height=900,
            template="plotly_white"
        )
        
        fig.write_html(filepath)
        
        print(f"✓ Token comparison dashboard generated: {filepath}")
        return filepath
