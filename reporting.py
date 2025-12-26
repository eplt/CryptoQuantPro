"""Reporting module for generating comprehensive analysis reports."""

import os
import time
from datetime import datetime


def prompt_for_report_generation():
    """Prompt user for report generation options.
    
    Returns:
        str: Report type ('full', 'basic', or 'none')
    """
    print(f"\n" + "="*60)
    print("REPORT GENERATION OPTIONS")
    print("="*60)
    
    while True:
        print("\nWould you like to generate a comprehensive PDF report?")
        print("1. Yes - Generate full PDF report with AI insights (requires Ollama)")
        print("2. Yes - Generate PDF report without AI insights")
        print("3. No - Skip report generation")
        print("4. View report features before deciding")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            return 'full'
        elif choice == '2':
            return 'basic'
        elif choice == '3':
            return 'none'
        elif choice == '4':
            print("\nPDF REPORT FEATURES:")
            print("• Executive Summary with quantitative insights")
            print("• Detailed methodology explanation") 
            print("• Token analysis with scoring charts")
            print("• Portfolio composition visualization")
            print("• Performance analysis across drift thresholds")
            print("• Comprehensive risk metrics")
            print("• Implementation guidelines")
            print("• Professional formatting (15-25 pages)")
            if check_ollama_available():
                print("• AI-enhanced explanations (Ollama detected)")
            else:
                print("• AI explanations unavailable (Ollama not running)")
            continue
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")


def check_ollama_available():
    """Quick check if Ollama is available for AI report generation.
    
    Returns:
        bool: True if Ollama is accessible
    """
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except:
        return False


def generate_report_if_requested(report_type, analysis_data, execution_log, charts):
    """Generate report based on user choice.
    
    Args:
        report_type: Type of report to generate ('full', 'basic', or 'none')
        analysis_data: Dictionary of analysis results
        execution_log: List of execution steps and timings
        charts: List of chart paths and captions
        
    Returns:
        str or None: Path to generated report file or None
    """
    if report_type == 'none':
        print("Skipping report generation as requested.")
        return None
    
    try:
        # Import here to avoid dependency issues if not needed
        from analysis.report_generator import ReportGenerator
        
        print(f"\nGenerating PDF report...")
        report_start = time.time()
        
        # Initialize report generator
        force_no_ai = (report_type == 'basic')
        if force_no_ai:
            print("Generating report without AI insights...")
            report_gen = ReportGenerator()
            report_gen.ollama_available = False
        else:
            print("Generating report with AI insights...")
            report_gen = ReportGenerator()
        
        # Add all analysis data
        for key, value in analysis_data.items():
            report_gen.add_analysis_data(key, value)
        
        # Add execution log
        report_gen.execution_log = execution_log
        
        # Add charts
        for chart in charts:
            report_gen.add_chart(chart['path'], chart['caption'], chart.get('type', 'performance'))
        
        # Generate report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        pdf_filename = f"cryptoquant_pro_report_{timestamp}.pdf"
        
        generated_file = report_gen.generate_pdf_report(pdf_filename)
        report_time = time.time() - report_start
        
        if generated_file:
            file_size = os.path.getsize(generated_file) / (1024 * 1024)  # MB
            print(f"✓ PDF report generated in {report_time:.1f} seconds")
            print(f"📄 Report saved as: {generated_file}")
            print(f"📊 Report size: {file_size:.1f} MB")
            return generated_file
        else:
            print(f"✗ Failed to generate PDF report")
            return None
            
    except ImportError:
        print("✗ Report generation requires additional dependencies:")
        print("  pip install reportlab requests matplotlib")
        return None
    except Exception as e:
        print(f"✗ Error generating PDF report: {e}")
        return None


def save_results_to_json(results_dir, backtest_results, best_portfolio, best_allocations,
                        best_allocation_method, start_time):
    """Save backtest results to JSON file.
    
    Args:
        results_dir: Directory to save results
        backtest_results: Dictionary of backtest results
        best_portfolio: Best portfolio configuration
        best_allocations: Best allocation weights
        best_allocation_method: Best allocation method name
        start_time: Analysis start time
        
    Returns:
        str or None: Path to saved file or None if error
    """
    import json
    import numpy as np
    
    try:
        os.makedirs(results_dir, exist_ok=True)
        
        serializable_results = {}
        for drift, results in backtest_results.items():
            try:
                # Convert numpy types for JSON serialization
                serializable_results[str(drift)] = {
                    'performance_metrics': {
                        k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                        for k, v in results['performance_metrics'].items()
                    },
                    'portfolio_config': {
                        'tokens': list(best_portfolio['tokens']),
                        'allocations': {k: float(v) for k, v in best_allocations.items()},
                        'drift_threshold': float(drift),
                        'allocation_method': best_allocation_method,
                        'portfolio_score': float(best_portfolio['score'])
                    },
                    'execution_info': {
                        'timestamp': datetime.now().isoformat(),
                        'total_runtime_seconds': time.time() - start_time
                    }
                }
            except Exception as e:
                print(f"Error serializing results for drift {drift}: {e}")
        
        results_file = os.path.join(results_dir, 'backtest_results.json')
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"✓ Results saved to {results_file}")
        return results_file
        
    except Exception as e:
        print(f"✗ Could not save JSON results: {e}")
        return None


def export_detailed_csv(best_drift, backtest_results):
    """Export detailed results to CSV.
    
    Args:
        best_drift: Best drift threshold value
        backtest_results: Dictionary of backtest results
        
    Returns:
        bool: True if successful
    """
    try:
        from analysis.performance_metrics import PerformanceAnalyzer
        
        if best_drift in backtest_results:
            analyzer = PerformanceAnalyzer(backtest_results[best_drift])
            analyzer.export_results('detailed_results.csv')
            print(f"✓ Detailed CSV results exported")
            return True
    except Exception as e:
        print(f"Could not export CSV results: {e}")
        return False


def generate_performance_charts(plots_dir, best_drift, backtest_results):
    """Generate and save performance charts.
    
    Args:
        plots_dir: Directory to save plots
        best_drift: Best drift threshold value
        backtest_results: Dictionary of backtest results
        
    Returns:
        list: List of generated chart dictionaries
    """
    import os
    from datetime import datetime
    
    charts = []
    
    try:
        from analysis.performance_metrics import PerformanceAnalyzer
        
        os.makedirs(plots_dir, exist_ok=True)
        
        chart_file = os.path.join(plots_dir, f'performance_charts_{datetime.now().strftime("%Y%m%d_%H%M")}.png')
        
        if best_drift in backtest_results:
            analyzer = PerformanceAnalyzer(backtest_results[best_drift])
            analyzer.plot_performance_charts(chart_file)
            print(f"✓ Performance charts saved to {chart_file}")
            
            charts.append({
                'path': chart_file,
                'caption': 'Portfolio Performance Analysis',
                'type': 'performance'
            })
            
    except Exception as e:
        print(f"Could not generate charts: {e}")
    
    return charts
