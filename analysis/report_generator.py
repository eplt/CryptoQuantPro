import json
import os
import requests
import matplotlib.pyplot as plt
from datetime import datetime
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from config.settings import TRANSACTION_FEE
import pandas as pd
import numpy as np
from analysis.performance_metrics import PerformanceAnalyzer

class ReportGenerator:
    def __init__(self, ollama_url="http://localhost:11434"):
        self.ollama_url = ollama_url
        self.ollama_model = "gemma3n:latest"
        self.analysis_data = {}
        self.execution_log = []
        self.charts = []
        
        # Check if Ollama is available
        self.ollama_available = self._check_ollama()
        
    def _check_ollama(self):
        """Check if Ollama is available"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                available_models = [model['name'] for model in models]
                print(f"Ollama available with models: {available_models}")
                return True
        except Exception as e:
            print(f"Ollama not available: {e}")
        return False
    
    def generate_ai_explanation(self, prompt, context=""):
        """Generate AI explanation using Ollama"""
        if not self.ollama_available:
            return "AI explanation not available (Ollama not running)"
        
        try:
            full_prompt = f"""
            You are a cryptocurrency quantitative analyst writing a professional investment report. 
            
            Context: {context}
            
            Task: {prompt}
            
            Please provide a clear, professional explanation that would be suitable for an institutional investor. 
            Focus on the quantitative aspects and risk management implications.
            Keep the response concise but informative (2-3 paragraphs maximum).
            """
            
            payload = {
                "model": self.ollama_model,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "max_tokens": 500
                }
            }
            
            response = requests.post(f"{self.ollama_url}/api/generate", 
                                   json=payload, 
                                   timeout=30)
            
            if response.status_code == 200:
                return response.json().get('response', 'No response generated')
            else:
                return f"AI generation failed: HTTP {response.status_code}"
                
        except Exception as e:
            return f"AI explanation error: {str(e)}"
    
    def log_step(self, step_name, details, duration=None):
        """Log execution step"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'step': step_name,
            'details': details,
            'duration_seconds': duration
        }
        self.execution_log.append(log_entry)
    
    def add_analysis_data(self, key, data):
        """Add analysis data for report"""
        self.analysis_data[key] = data
    
    def add_chart(self, chart_path, caption, chart_type="performance"):
        """Add chart for inclusion in report"""
        self.charts.append({
            'path': chart_path,
            'caption': caption,
            'type': chart_type
        })
    
    def create_portfolio_composition_chart(self, allocations, filename):
        """Create portfolio composition pie chart"""
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(10, 8))
        
        tokens = list(allocations.keys())
        weights = list(allocations.values())
        
        # Create pie chart
        colors_list = plt.cm.Set3(np.linspace(0, 1, len(tokens)))
        wedges, texts, autotexts = ax.pie(weights, labels=tokens, autopct='%1.1f%%',
                                         colors=colors_list, startangle=90)
        
        # Beautify
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax.set_title('Portfolio Composition', fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename
    
    def create_performance_summary_chart(self, backtest_results, filename):
        """Create performance summary chart"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Extract metrics
        drifts = list(backtest_results.keys())
        returns = [results['performance_metrics']['annualized_return'] for results in backtest_results.values()]
        sharpes = [results['performance_metrics']['sharpe_ratio'] for results in backtest_results.values()]
        drawdowns = [results['performance_metrics']['max_drawdown'] for results in backtest_results.values()]
        trades = [results['performance_metrics']['total_trades'] for results in backtest_results.values()]
        
        # Return vs Drift
        ax1.plot([d*100 for d in drifts], [r*100 for r in returns], 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Drift Threshold (%)')
        ax1.set_ylabel('Annualized Return (%)')
        ax1.set_title('Return vs Drift Threshold')
        ax1.grid(True, alpha=0.3)
        
        # Sharpe vs Drift
        ax2.plot([d*100 for d in drifts], sharpes, 'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel('Drift Threshold (%)')
        ax2.set_ylabel('Sharpe Ratio')
        ax2.set_title('Sharpe Ratio vs Drift Threshold')
        ax2.grid(True, alpha=0.3)
        
        # Max Drawdown vs Drift
        ax3.plot([d*100 for d in drifts], [dd*100 for dd in drawdowns], 'go-', linewidth=2, markersize=8)
        ax3.set_xlabel('Drift Threshold (%)')
        ax3.set_ylabel('Max Drawdown (%)')
        ax3.set_title('Max Drawdown vs Drift Threshold')
        ax3.grid(True, alpha=0.3)
        
        # Trading Frequency vs Drift
        ax4.bar([d*100 for d in drifts], trades, color='orange', alpha=0.7)
        ax4.set_xlabel('Drift Threshold (%)')
        ax4.set_ylabel('Number of Trades')
        ax4.set_title('Trading Frequency vs Drift Threshold')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename
    
    def create_token_scores_chart(self, token_scores, filename, top_n=15):
        """Create token scores visualization"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Get top tokens
        top_tokens = list(token_scores.items())[:top_n]
        tokens = [token for token, _ in top_tokens]
        scores = [data['final_score'] for _, data in top_tokens]
        volatilities = [data['metrics']['volatility'] for _, data in top_tokens]
        
        # Token scores bar chart
        bars = ax1.barh(range(len(tokens)), scores, color='steelblue', alpha=0.7)
        ax1.set_yticks(range(len(tokens)))
        ax1.set_yticklabels(tokens)
        ax1.set_xlabel('Composite Score')
        ax1.set_title(f'Top {top_n} Token Scores')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Add score labels on bars
        for i, (bar, score) in enumerate(zip(bars, scores)):
            ax1.text(score + 1, i, f'{score:.1f}', va='center', fontsize=9)
        
        # Volatility vs Score scatter
        ax2.scatter(volatilities, scores, alpha=0.7, s=100, color='red')
        ax2.set_xlabel('Volatility (Annualized)')
        ax2.set_ylabel('Composite Score')
        ax2.set_title('Token Score vs Volatility')
        ax2.grid(True, alpha=0.3)
        
        # Add token labels to scatter points
        for i, token in enumerate(tokens):
            ax2.annotate(token, (volatilities[i], scores[i]), 
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.8)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename
    
    def generate_pdf_report(self, output_filename="crypto_portfolio_analysis_report.pdf"):
        """Generate comprehensive PDF report"""
        doc = SimpleDocTemplate(output_filename, pagesize=A4,
                              rightMargin=72, leftMargin=72,
                              topMargin=72, bottomMargin=18)
        
        # Container for the 'Flowable' objects
        story = []
        
        # Define styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.darkblue
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            spaceBefore=20,
            spaceAfter=12,
            textColor=colors.darkblue
        )
        
        subheading_style = ParagraphStyle(
            'CustomSubheading',
            parent=styles['Heading3'],
            fontSize=14,
            spaceBefore=15,
            spaceAfter=8,
            textColor=colors.darkblue
        )
        
        # Title Page
        story.append(Paragraph("Cryptocurrency Portfolio Analysis Report", title_style))
        story.append(Spacer(1, 0.2*inch))
        story.append(Paragraph(f"Generated on {datetime.now().strftime('%B %d, %Y at %H:%M')}", 
                             styles['Normal']))
        story.append(Spacer(1, 0.5*inch))
        
        # Executive Summary
        story.append(Paragraph("Executive Summary", heading_style))
        
        if 'best_portfolio' in self.analysis_data and 'best_results' in self.analysis_data:
            portfolio = self.analysis_data['best_portfolio']
            results = self.analysis_data['best_results']
            metrics = results['performance_metrics']
            
            exec_summary = f"""
            This report presents a comprehensive analysis of cryptocurrency portfolio strategies using quantitative methods 
            and multiprocessing optimization. The analysis evaluated {len(self.analysis_data.get('token_scores', {}))} 
            cryptocurrency tokens and tested multiple portfolio configurations using parallel processing across 
            {os.cpu_count()} CPU cores.
            
            <b>Key Findings:</b><br/>
            â€¢ Optimal Portfolio: {len(portfolio['tokens'])} tokens ({', '.join(portfolio['tokens'])})<br/>
            â€¢ Expected Annual Return: {metrics.get('annualized_return', 0):.2%}<br/>
            â€¢ Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}<br/>
            â€¢ Maximum Drawdown: {metrics.get('max_drawdown', 0):.2%}<br/>
            â€¢ Optimal Rebalancing Threshold: {self.analysis_data.get('best_drift', 0):.1%}<br/>
            â€¢ Total Trades (Backtest): {metrics.get('total_trades', 0)}<br/>
            â€¢ Transaction Cost Impact: {metrics.get('fees_pct_of_portfolio', 0):.3%} of portfolio<br/>
            """
            
            story.append(Paragraph(exec_summary, styles['Normal']))
            
            # AI-enhanced executive summary
            if self.ollama_available:
                ai_summary = self.generate_ai_explanation(
                    "Provide an executive summary interpretation of these portfolio analysis results, focusing on risk-adjusted performance and practical implementation",
                    f"Portfolio: {portfolio['tokens']}, Return: {metrics.get('annualized_return', 0):.2%}, "
                    f"Sharpe: {metrics.get('sharpe_ratio', 0):.3f}, Drawdown: {metrics.get('max_drawdown', 0):.2%}, "
                    f"Rebalancing threshold: {self.analysis_data.get('best_drift', 0):.1%}"
                )
                story.append(Spacer(1, 12))
                story.append(Paragraph("<b>AI Investment Analysis:</b>", subheading_style))
                story.append(Paragraph(ai_summary, styles['Normal']))
        
        story.append(PageBreak())
        
        # Methodology
        story.append(Paragraph("Methodology", heading_style))
        
        methodology_text = """
        <b>1. Data Collection and Caching:</b><br/>
        Historical price data was collected from Binance API with intelligent caching to minimize API calls. 
        Data freshness was maintained with a 7-day cache expiration policy. Only tokens with market cap > $1B 
        and daily volume > $10M were considered for analysis.<br/><br/>
        
        <b>2. Token Evaluation Framework:</b><br/>
        Each token was scored using a multi-factor model incorporating:<br/>
        â€¢ Liquidity metrics (25% weight): Volume consistency, liquidity ratios<br/>
        â€¢ Volatility analysis (20% weight): Preferring moderate volatility (30-80% annualized)<br/>
        â€¢ Stability measures (20% weight): Maximum drawdown, Value at Risk<br/>
        â€¢ Market position (15% weight): Market capitalization and ranking<br/>
        â€¢ Mean reversion properties (10% weight): Hurst exponent analysis<br/>
        â€¢ Historical performance (10% weight): Risk-adjusted returns<br/><br/>
        
        <b>3. Portfolio Optimization:</b><br/>
        Multiple portfolio sizes (2-10 tokens) and allocation methods were tested using parallel processing 
        across all available CPU cores. Monte Carlo simulation (50,000 iterations) was employed for 
        portfolios with â‰¤6 tokens to optimize risk-adjusted returns.<br/><br/>
        
        <b>4. Rebalancing Strategy Testing:</b><br/>
        Six different drift thresholds (8%, 10%, 12%, 15%, 20%, 25%) were backtested with realistic 
        transaction costs of {TRANSACTION_FEE:.2%} per trade. Minimum rebalancing interval of 5 days and maximum 
        of 21 days were enforced to prevent over-trading.<br/><br/>
        
        <b>5. Risk Management:</b><br/>
        Comprehensive risk metrics including VaR, Conditional VaR, skewness, kurtosis, and maximum 
        consecutive losses were calculated. Correlation analysis was performed to ensure portfolio diversification.
        """
        
        story.append(Paragraph(methodology_text, styles['Normal']))
        
        # AI-enhanced methodology explanation
        if self.ollama_available:
            ai_methodology = self.generate_ai_explanation(
                "Evaluate the robustness and institutional suitability of this cryptocurrency portfolio analysis methodology",
                "Multi-factor scoring, parallel processing, Monte Carlo optimization, realistic transaction costs, comprehensive risk metrics"
            )
            story.append(Spacer(1, 12))
            story.append(Paragraph("<b>Methodology Assessment:</b>", subheading_style))
            story.append(Paragraph(ai_methodology, styles['Normal']))
        
        story.append(PageBreak())
        
        # Token Analysis Results
        story.append(Paragraph("Token Analysis Results", heading_style))
        
        if 'token_scores' in self.analysis_data:
            # Create token scores visualization
            token_chart_filename = "token_scores_analysis.png"
            self.create_token_scores_chart(self.analysis_data['token_scores'], token_chart_filename)
            
            img = Image(token_chart_filename, width=7*inch, height=3.5*inch)
            story.append(img)
            story.append(Spacer(1, 12))
            
            # Create top tokens table
            top_tokens = list(self.analysis_data['token_scores'].items())[:10]
            
            table_data = [['Rank', 'Token', 'Score', 'Volatility', 'Market Cap', 'Liquidity Ratio']]
            
            for i, (token, score_data) in enumerate(top_tokens):
                table_data.append([
                    str(i+1),
                    token,
                    f"{score_data['final_score']:.1f}",
                    f"{score_data['metrics']['volatility']:.3f}",
                    f"${score_data['metrics']['market_cap']:.2e}",
                    f"{score_data['metrics']['liquidity_ratio']:.6f}"
                ])
            
            table = Table(table_data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(table)
            story.append(Spacer(1, 12))
            
            # AI analysis of token selection
            if self.ollama_available:
                top_3_tokens = [token for token, _ in top_tokens[:3]]
                ai_token_analysis = self.generate_ai_explanation(
                    f"Analyze why these tokens ranked highest in the quantitative scoring: {', '.join(top_3_tokens)}. What does this reveal about current market conditions?",
                    f"Top tokens by composite score considering liquidity, volatility, stability, and market position. Scores range from {top_tokens[-1][1]['final_score']:.1f} to {top_tokens[0][1]['final_score']:.1f}"
                )
                story.append(Paragraph("<b>Token Selection Analysis:</b>", subheading_style))
                story.append(Paragraph(ai_token_analysis, styles['Normal']))
        
        story.append(PageBreak())
        
        # Portfolio Composition
        story.append(Paragraph("Portfolio Composition", heading_style))
        
        if 'best_allocations' in self.analysis_data:
            allocations = self.analysis_data['best_allocations']
            
            # Create and add portfolio composition chart
            chart_filename = "portfolio_composition.png"
            self.create_portfolio_composition_chart(allocations, chart_filename)
            
            # Add chart to story
            img = Image(chart_filename, width=6*inch, height=4.8*inch)
            story.append(img)
            story.append(Spacer(1, 12))
            
            # Allocation table
            alloc_data = [['Token', 'Allocation', 'Weight', 'Expected Vol*']]
            for token, weight in allocations.items():
                token_vol = self.analysis_data.get('token_scores', {}).get(token, {}).get('metrics', {}).get('volatility', 0)
                alloc_data.append([
                    token, 
                    f"{weight:.1%}", 
                    f"{weight:.4f}",
                    f"{token_vol:.3f}"
                ])
            
            # Add portfolio-level statistics
            alloc_data.append(['Portfolio Total', '100.0%', '1.0000', ''])
            
            alloc_table = Table(alloc_data)
            alloc_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('BACKGROUND', (-1, -1), (-1, -1), colors.lightgrey),
                ('FONTNAME', (-1, -1), (-1, -1), 'Helvetica-Bold'),
            ]))
            
            story.append(alloc_table)
            story.append(Paragraph("*Individual token annualized volatility", styles['Normal']))
            story.append(Spacer(1, 12))
            
            # Portfolio diversification metrics
            if 'best_portfolio' in self.analysis_data:
                portfolio_info = f"""
                <b>Portfolio Diversification Metrics:</b><br/>
                â€¢ Number of tokens: {len(allocations)}<br/>
                â€¢ Allocation method: {self.analysis_data.get('best_allocation_method', 'Not specified')}<br/>
                â€¢ Maximum single allocation: {max(allocations.values()):.1%}<br/>
                â€¢ Minimum single allocation: {min(allocations.values()):.1%}<br/>
                â€¢ Portfolio score: {self.analysis_data['best_portfolio'].get('score', 0):.1f}<br/>
                """
                story.append(Paragraph(portfolio_info, styles['Normal']))
            
            # AI portfolio analysis
            if self.ollama_available:
                weights_summary = {k: v for k, v in sorted(allocations.items(), key=lambda x: x[1], reverse=True)}
                ai_portfolio = self.generate_ai_explanation(
                    "Analyze this portfolio composition from a risk management and diversification perspective. Comment on the allocation balance and potential concentration risks.",
                    f"Portfolio allocation (sorted by weight): {weights_summary}. "
                    f"Method: {self.analysis_data.get('best_allocation_method', 'Equal weight')}"
                )
                story.append(Spacer(1, 12))
                story.append(Paragraph("<b>Portfolio Composition Analysis:</b>", subheading_style))
                story.append(Paragraph(ai_portfolio, styles['Normal']))
        
        story.append(PageBreak())
        
        # Performance Analysis
        story.append(Paragraph("Performance Analysis", heading_style))
        
        if 'backtest_results' in self.analysis_data:
            # Create performance summary chart
            perf_chart_filename = "performance_summary.png"
            self.create_performance_summary_chart(self.analysis_data['backtest_results'], perf_chart_filename)
            
            img = Image(perf_chart_filename, width=7*inch, height=5.25*inch)
            story.append(img)
            story.append(Spacer(1, 12))
            
            # Performance metrics table
            results = self.analysis_data['backtest_results']
            perf_data = [['Drift Threshold', 'Annual Return', 'Volatility', 'Sharpe Ratio', 'Max Drawdown', 'Total Trades', 'Fees %']]
            
            for drift, result in results.items():
                metrics = result['performance_metrics']
                perf_data.append([
                    f"{drift:.1%}",
                    f"{metrics['annualized_return']:.2%}",
                    f"{metrics['volatility']:.2%}",
                    f"{metrics['sharpe_ratio']:.3f}",
                    f"{metrics['max_drawdown']:.2%}",
                    str(metrics['total_trades']),
                    f"{metrics['fees_pct_of_portfolio']:.3%}"
                ])
            
            perf_table = Table(perf_data)
            perf_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(perf_table)
            story.append(Spacer(1, 12))
            
            # Performance summary
            if 'best_results' in self.analysis_data:
                best_metrics = self.analysis_data['best_results']['performance_metrics']
                best_drift = self.analysis_data.get('best_drift', 0)
                
                perf_summary = f"""
                <b>Optimal Strategy Performance:</b><br/>
                â€¢ Best drift threshold: {best_drift:.1%}<br/>
                â€¢ Annualized return: {best_metrics.get('annualized_return', 0):.2%}<br/>
                â€¢ Portfolio volatility: {best_metrics.get('volatility', 0):.2%}<br/>
                â€¢ Sharpe ratio: {best_metrics.get('sharpe_ratio', 0):.3f}<br/>
                â€¢ Maximum drawdown: {best_metrics.get('max_drawdown', 0):.2%}<br/>
                â€¢ Alpha vs benchmark: {best_metrics.get('alpha', 0):.2%}<br/>
                â€¢ Total rebalancing trades: {best_metrics.get('total_trades', 0)}<br/>
                â€¢ Transaction cost drag: {best_metrics.get('fees_pct_of_portfolio', 0):.3%}<br/>
                """
                
                story.append(Paragraph(perf_summary, styles['Normal']))
            
            # AI performance analysis
            if self.ollama_available and 'best_results' in self.analysis_data:
                best_metrics = self.analysis_data['best_results']['performance_metrics']
                ai_performance = self.generate_ai_explanation(
                    "Interpret these backtesting results and their implications for practical portfolio implementation. Address the risk-return trade-offs and rebalancing frequency optimization.",
                    f"Optimal strategy: {best_metrics['annualized_return']:.2%} return, "
                    f"{best_metrics['sharpe_ratio']:.3f} Sharpe, {best_metrics['max_drawdown']:.2%} max drawdown, "
                    f"{best_metrics['total_trades']} trades, {best_metrics['fees_pct_of_portfolio']:.3%} cost drag"
                )
                story.append(Spacer(1, 12))
                story.append(Paragraph("<b>Performance Interpretation:</b>", subheading_style))
                story.append(Paragraph(ai_performance, styles['Normal']))
        
        # Add existing performance charts if available
        for chart in self.charts:
            if os.path.exists(chart['path']):
                story.append(PageBreak())
                story.append(Paragraph(f"Additional Analysis: {chart['caption']}", heading_style))
                try:
                    img = Image(chart['path'], width=7*inch, height=5.25*inch)
                    story.append(img)
                except:
                    story.append(Paragraph(f"Chart could not be loaded: {chart['path']}", styles['Normal']))
        
        story.append(PageBreak())
        
        # Risk Analysis
        story.append(Paragraph("Risk Analysis", heading_style))
        
        if 'risk_metrics' in self.analysis_data:
            risk = self.analysis_data['risk_metrics']
            
            risk_text = f"""
            <b>Tail Risk Metrics:</b><br/>
            â€¢ Value at Risk (5%): {risk.get('value_at_risk_5pct', 0):.3f}<br/>
            â€¢ Conditional VaR (5%): {risk.get('conditional_var_5pct', 0):.3f}<br/>
            
            <b>Distribution Characteristics:</b><br/>
            â€¢ Skewness: {risk.get('skewness', 0):.3f}<br/>
            â€¢ Kurtosis: {risk.get('kurtosis', 0):.3f}<br/>
            
            <b>Consistency Metrics:</b><br/>
            â€¢ Positive return periods: {risk.get('positive_periods', 0):.1%}<br/>
            â€¢ Maximum consecutive losses: {risk.get('max_consecutive_losses', 0)} periods<br/>
            
            <b>Risk-Adjusted Performance:</b><br/>
            â€¢ Calmar ratio: {risk.get('calmar_ratio', 0):.3f}<br/>
            """
            
            story.append(Paragraph(risk_text, styles['Normal']))
            
            # Risk interpretation
            risk_interpretation = """
            <b>Risk Metric Interpretation:</b><br/>
            â€¢ VaR (5%): Expected maximum loss on worst 5% of days<br/>
            â€¢ Conditional VaR: Average loss when VaR threshold is exceeded<br/>
            â€¢ Skewness: Distribution asymmetry (negative = left tail risk)<br/>
            â€¢ Kurtosis: Tail thickness (>3 = fatter tails than normal distribution)<br/>
            â€¢ Calmar ratio: Annual return / Maximum drawdown<br/>
            """
            
            story.append(Spacer(1, 12))
            story.append(Paragraph(risk_interpretation, styles['Normal']))
            
            # AI risk analysis
            if self.ollama_available:
                ai_risk = self.generate_ai_explanation(
                    "Analyze these risk metrics and their implications for institutional portfolio management. Address tail risk, distribution characteristics, and risk-adjusted performance.",
                    f"VaR(5%): {risk.get('value_at_risk_5pct', 0):.3f}, "
                    f"Skewness: {risk.get('skewness', 0):.3f}, "
                    f"Kurtosis: {risk.get('kurtosis', 0):.3f}, "
                    f"Max consecutive losses: {risk.get('max_consecutive_losses', 0)}, "
                    f"Calmar ratio: {risk.get('calmar_ratio', 0):.3f}"
                )
                story.append(Spacer(1, 12))
                story.append(Paragraph("<b>Risk Assessment:</b>", subheading_style))
                story.append(Paragraph(ai_risk, styles['Normal']))
        else:
            story.append(Paragraph("Risk metrics not available - ensure risk analysis is completed in main script.", styles['Normal']))
        
        story.append(PageBreak())
        
        # Implementation Guidelines
        story.append(Paragraph("Implementation Guidelines", heading_style))
        
        if 'best_portfolio' in self.analysis_data:
            implementation_text = f"""
            <b>Recommended Portfolio Configuration:</b><br/>
            â€¢ Tokens: {', '.join(self.analysis_data['best_portfolio']['tokens'])}<br/>
            â€¢ Allocation Method: {self.analysis_data.get('best_allocation_method', 'Equal Weight')}<br/>
            â€¢ Rebalancing Threshold: {self.analysis_data.get('best_drift', 0.12):.1%}<br/>
            â€¢ Expected Sharpe Ratio: {self.analysis_data.get('best_results', {}).get('performance_metrics', {}).get('sharpe_ratio', 0):.3f}<br/>
            â€¢ Minimum Rebalancing Interval: 5 days<br/>
            â€¢ Maximum Rebalancing Interval: 21 days<br/>
            
            <b>Operational Considerations:</b><br/>
            â€¢ Monitor correlation breakdown during market stress periods<br/>
            â€¢ Reassess token fundamentals and scoring quarterly<br/>
            â€¢ Consider reducing position sizes during high volatility regimes (>80% annualized)<br/>
            â€¢ Maintain adequate cash reserves (2-5%) for rebalancing operations<br/>
            â€¢ Review and update the token universe semi-annually<br/>
            
            <b>Risk Management Protocols:</b><br/>
            â€¢ Implement position size limits (maximum 35% in any single token)<br/>
            â€¢ Monitor cross-token correlations weekly<br/>
            â€¢ Set portfolio-level stop-loss at 25% drawdown<br/>
            â€¢ Scale down during correlation spike periods (>0.8 average correlation)<br/>
            â€¢ Regular stress testing against historical crash scenarios<br/>
            
            <b>Performance Monitoring:</b><br/>
            â€¢ Track rebalancing frequency and costs monthly<br/>
            â€¢ Compare actual vs expected Sharpe ratios quarterly<br/>
            â€¢ Monitor alpha decay and token ranking changes<br/>
            â€¢ Review allocation method performance annually<br/>
            """
            
            story.append(Paragraph(implementation_text, styles['Normal']))
            
            # AI implementation recommendations
            if self.ollama_available:
                ai_implementation = self.generate_ai_explanation(
                    "Provide strategic implementation recommendations for this cryptocurrency rebalancing strategy, focusing on practical execution challenges and risk management in volatile markets.",
                    f"Portfolio: {self.analysis_data['best_portfolio']['tokens']}, "
                    f"Optimal rebalancing: {self.analysis_data.get('best_drift', 0.12):.1%}, "
                    f"Expected Sharpe: {self.analysis_data.get('best_results', {}).get('performance_metrics', {}).get('sharpe_ratio', 0):.3f}"
                )
                story.append(Spacer(1, 12))
                story.append(Paragraph("<b>Strategic Implementation Analysis:</b>", subheading_style))
                story.append(Paragraph(ai_implementation, styles['Normal']))
        
        story.append(PageBreak())
        
        # Execution Summary and Technical Details
        story.append(Paragraph("Execution Summary", heading_style))
        
        if self.execution_log:
            total_time = sum(log.get('duration_seconds', 0) for log in self.execution_log if log.get('duration_seconds'))
            
            exec_summary_text = f"""
            <b>Analysis Execution Details:</b><br/>
            â€¢ Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>
            â€¢ Total execution time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)<br/>
            â€¢ Parallel processing: {os.cpu_count()} CPU cores utilized<br/>
            â€¢ Tokens analyzed: {len(self.analysis_data.get('token_scores', {}))}<br/>
            â€¢ Portfolio configurations tested: {len(self.analysis_data.get('backtest_results', {}))}<br/>
            â€¢ AI explanations: {'Enabled (Ollama)' if self.ollama_available else 'Disabled'}<br/>
            """
            
            story.append(Paragraph(exec_summary_text, styles['Normal']))
            
            # Execution step breakdown
            if len(self.execution_log) > 0:
                story.append(Spacer(1, 12))
                story.append(Paragraph("<b>Processing Steps:</b>", subheading_style))
                
                step_data = [['Step', 'Duration (s)', 'Details']]
                for log in self.execution_log:
                    step_data.append([
                        log['step'],
                        f"{log.get('duration_seconds', 0):.1f}" if log.get('duration_seconds') else 'N/A',
                        str(log['details'])[:50] + '...' if len(str(log['details'])) > 50 else str(log['details'])
                    ])
                
                step_table = Table(step_data)
                step_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 9),
                    ('FONTSIZE', (0, 1), (-1, -1), 8),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                story.append(step_table)
        
        # Disclaimer
        story.append(Spacer(1, 0.3*inch))
        disclaimer = """
        <b>Important Disclaimer:</b> This analysis is for informational purposes only and does not constitute investment advice. 
        Cryptocurrency investments are highly volatile and risky. Past performance does not guarantee future results. 
        The quantitative models and backtests presented are based on historical data and may not predict future performance. 
        Investors should conduct their own research and consider their risk tolerance before making investment decisions. 
        The authors are not responsible for any financial losses incurred from using this analysis.
        """
        
        story.append(Paragraph(disclaimer, styles['Normal']))
        
        # Build PDF
        try:
            doc.build(story)
            print(f"âœ“ PDF report generated successfully: {output_filename}")
            return output_filename
        except Exception as e:
            print(f"âœ— Error building PDF: {e}")
            return None
