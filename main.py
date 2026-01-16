from data.data_collector import DataCollector
from evaluation.token_evaluator import TokenEvaluator
from evaluation.portfolio_builder import PortfolioBuilder
from backtesting.backtest_engine import BacktestEngine, build_backtest_windows
from analysis.performance_metrics import PerformanceAnalyzer
from config.settings import *
from __version__ import __version__, __release__
import json
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime, timedelta

def get_token_selection_mode():
    """Get user preference for token selection"""
    print(f"\n" + "="*60)
    print("TOKEN SELECTION MODE")
    print("="*60)
    
    while True:
        print("\nHow would you like to select tokens for analysis?")
        print("1. Auto - Use top tokens by market cap (recommended for discovery)")
        print("2. Manual - Enter specific token symbols (up to 10 tokens)")
        print("3. Hybrid - Manual tokens + top performers to fill portfolio")
        print("4. View example token formats")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            return 'auto', []
        elif choice == '2':
            return 'manual', get_manual_token_list()
        elif choice == '3':
            return 'hybrid', get_manual_token_list()
        elif choice == '4':
            show_token_examples()
            continue
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")

def show_token_examples():
    """Show examples of token symbol formats"""
    print("\nTOKEN SYMBOL EXAMPLES:")
    print("• Bitcoin: BTC or BTCUSDT")
    print("• Ethereum: ETH or ETHUSDT") 
    print("• XRP: XRP or XRPUSDT")
    print("• Solana: SOL or SOLUSDT")
    print("• Cardano: ADA or ADAUSDT")
    print("• Binance Coin: BNB or BNBUSDT")
    print("• Polygon: MATIC or MATICUSDT")
    print("• Chainlink: LINK or LINKUSDT")
    print("• Polkadot: DOT or DOTUSDT")
    print("• Avalanche: AVAX or AVAXUSDT")
    print("\nNOTES:")
    print("• You can use either format (BTC or BTCUSDT)")
    print("• System will automatically add USDT if needed")
    print("• Maximum 10 tokens per analysis")
    print("• Minimum 2 tokens required for portfolio analysis")

def get_manual_token_list():
    """Get manual token list from user"""
    while True:
        tokens_input = input("\nEnter token symbols (comma-separated, max 10): ").strip()
        
        if not tokens_input:
            print("Please enter at least one token symbol.")
            continue
            
        # Parse and clean token list
        raw_tokens = [t.strip().upper() for t in tokens_input.split(',')]
        raw_tokens = [t for t in raw_tokens if t]  # Remove empty strings
        
        if len(raw_tokens) > 10:
            print(f"Too many tokens ({len(raw_tokens)}). Maximum is 10.")
            continue
            
        if len(raw_tokens) < 1:
            print("Please enter at least one token symbol.")
            continue
        
        # Normalize token symbols (add USDT if not present)
        normalized_tokens = []
        for token in raw_tokens:
            if token.endswith('USDT'):
                normalized_tokens.append(token)
            else:
                normalized_tokens.append(f"{token}USDT")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_tokens = []
        for token in normalized_tokens:
            if token not in seen:
                seen.add(token)
                unique_tokens.append(token)
        
        print(f"\nNormalized tokens: {', '.join(unique_tokens)}")
        
        # Confirm selection
        confirm = input("Proceed with these tokens? (y/n): ").strip().lower()
        if confirm in ['y', 'yes']:
            return unique_tokens
        else:
            print("Please re-enter your token selection.")

def validate_manual_tokens(tokens, price_data, market_data):
    """Validate that manual tokens have sufficient data"""
    print(f"\nValidating {len(tokens)} manually selected tokens...")
    
    valid_tokens = []
    invalid_tokens = []
    insufficient_data = []
    
    for token in tokens:
        if token not in price_data:
            invalid_tokens.append(token)
            continue
            
        # Check data quality
        df = price_data[token]
        if len(df) < 365:  # Need at least 1 year
            insufficient_data.append(f"{token} ({len(df)} days)")
            continue
            
        valid_tokens.append(token)
    
    # Report validation results
    if valid_tokens:
        print(f"✓ Valid tokens ({len(valid_tokens)}): {', '.join(valid_tokens)}")
    
    if invalid_tokens:
        print(f"✗ Invalid/unavailable tokens ({len(invalid_tokens)}): {', '.join(invalid_tokens)}")
        
    if insufficient_data:
        print(f"⚠ Insufficient data tokens: {', '.join(insufficient_data)}")
    
    if len(valid_tokens) < 2:
        print(f"\n❌ Error: Need at least 2 valid tokens for portfolio analysis.")
        print(f"   Only {len(valid_tokens)} valid tokens found.")
        return None
        
    return valid_tokens

def collect_tokens_by_mode(collector, mode, manual_tokens):
    """Collect token data based on selection mode"""
    print(f"\nCollecting data for {mode} token selection...")
    
    if mode == 'auto':
        # Original behavior - get top tokens by market cap
        print("Using automatic token selection (top by market cap)")
        return collector.collect_all_data()
        
    elif mode == 'manual':
        # Only collect data for manually specified tokens
        print(f"Collecting data for {len(manual_tokens)} manually selected tokens")
        return collector.collect_all_data(symbols=manual_tokens)
        
    elif mode == 'hybrid':
        # Get manual tokens + top performers to fill gaps
        print(f"Using hybrid approach: {len(manual_tokens)} manual + top performers")
        
        # First get manual tokens
        manual_price_data, manual_market_data = collector.collect_all_data(symbols=manual_tokens)
        
        # If we have fewer than 10 total tokens, add top performers
        if len(manual_price_data) < 10:
            needed = 10 - len(manual_price_data)
            print(f"Adding {needed} top performers to reach 10 tokens...")
            
            # Get top tokens, excluding ones we already have
            all_price_data, all_market_data = collector.collect_all_data()
            
            # Combine datasets
            combined_price_data = manual_price_data.copy()
            combined_market_data = manual_market_data.copy()
            
            added = 0
            for token in all_price_data:
                if token not in combined_price_data and added < needed:
                    combined_price_data[token] = all_price_data[token]
                    if token in all_market_data:
                        combined_market_data[token] = all_market_data[token]
                    added += 1
            
            print(f"✓ Added {added} additional tokens")
            return combined_price_data, combined_market_data
        else:
            return manual_price_data, manual_market_data

def suggest_rebalancing_token_sets(token_scores, price_data, max_sets=3, target_size=4, lookback_days=90):
    """Suggest token mixes for rebalancing backtests using scoring, correlation, and volatility.
    
    Args:
        token_scores: Ordered token score mapping from evaluation.
        price_data: Dict of price history DataFrames keyed by token.
        max_sets: Maximum number of token sets to return.
        target_size: Desired number of tokens per suggestion.
        lookback_days: Lookback window for correlation analysis.
    
    Returns:
        List of dicts with 'label' and 'tokens' keys.
    """
    if not token_scores:
        return []
    
    tokens_by_score = list(token_scores.keys())
    target_size = max(2, min(target_size, len(tokens_by_score)))
    candidate_pool_size = min(len(tokens_by_score), max(target_size * 3, target_size))
    candidate_pool = tokens_by_score[:candidate_pool_size]
    suggestions = []
    seen = set()
    
    def add_suggestion(label, tokens):
        tokens = list(dict.fromkeys(tokens))
        if len(tokens) < 2:
            return
        token_key = tuple(tokens)
        if token_key in seen:
            return
        suggestions.append({'label': label, 'tokens': tokens})
        seen.add(token_key)
    
    add_suggestion("Top score mix", candidate_pool[:target_size])
    
    if len(candidate_pool) >= 2:
        try:
            returns_data = {}
            for symbol in candidate_pool:
                returns = np.log(price_data[symbol]['close'] / price_data[symbol]['close'].shift(1)).dropna()
                returns_data[symbol] = returns.tail(lookback_days)
            
            corr_matrix = pd.DataFrame(returns_data).corr().abs().fillna(1)
            selected = [candidate_pool[0]]
            remaining = [token for token in candidate_pool if token not in selected]
            
            while len(selected) < target_size and remaining:
                avg_corr = corr_matrix.loc[remaining, selected].mean(axis=1)
                next_token = avg_corr.idxmin()
                selected.append(next_token)
                remaining.remove(next_token)
            
            add_suggestion("Low correlation mix", selected)
        except Exception as e:
            print(f"Skipping correlation-based suggestions: {e}")
    
    high_vol_tokens = sorted(
        candidate_pool,
        key=lambda token: token_scores[token]['metrics'].get('volatility', 0),
        reverse=True
    )
    add_suggestion("High volatility mix", high_vol_tokens[:target_size])
    
    return suggestions[:max_sets]

def prompt_for_report_generation():
    """Prompt user for report generation options"""
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
    """Quick check if Ollama is available"""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except:
        return False

def generate_report_if_requested(report_type, analysis_data, execution_log, charts):
    """Generate report based on user choice"""
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

def main():
    start_time = time.time()
    
    print("=== CryptoQuant Pro - Advanced Portfolio Analysis ===")
    print(f"Version {__version__} - {__release__}")
    print(f"Using {os.cpu_count()} CPU cores for parallel processing")
    print(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Initialize data storage for potential report generation
    analysis_data = {}
    execution_log = []
    charts = []
    
    # Step 0: Token Selection Mode
    selection_mode, manual_tokens = get_token_selection_mode()
    
    if selection_mode != 'auto':
        print(f"\n✓ Token selection mode: {selection_mode}")
        if manual_tokens:
            print(f"   Manual tokens: {', '.join(manual_tokens)}")
    
    # Step 1: Data Collection with Smart Caching
    print(f"\n1. Collecting data ({selection_mode} mode)...")
    data_start = time.time()
    collector = DataCollector()
    
    # Show cache status
    cache_status = collector.get_cache_status()
    if 'total_files' in cache_status:
        print(f"Cache status: {cache_status['fresh_files']} fresh, "
              f"{cache_status['old_files']} old files "
              f"({cache_status['cache_size_mb']} MB total)")
    
    # Clean old cache files
    try:
        cleaned = collector.clean_old_cache(max_age_days=30)
        if cleaned['files_removed'] > 0:
            print(f"Cleaned {cleaned['files_removed']} old cache files")
    except Exception as e:
        print(f"Cache cleaning failed: {e}")
    
    # Collect data based on selection mode
    try:
        price_data, market_data = collect_tokens_by_mode(collector, selection_mode, manual_tokens)
        data_time = time.time() - data_start
        
        print(f"✓ Data ready for {len(price_data)} tokens in {data_time:.1f} seconds")
        
        # Validate manual tokens if applicable
        if selection_mode in ['manual', 'hybrid'] and manual_tokens:
            valid_manual = validate_manual_tokens(manual_tokens, price_data, market_data)
            if valid_manual is None:
                print("Exiting due to insufficient valid manual tokens.")
                return
            
            # Store which tokens were manually selected for reporting
            analysis_data['manual_token_selection'] = {
                'mode': selection_mode,
                'requested_tokens': manual_tokens,
                'valid_tokens': valid_manual
            }
        
        execution_log.append({
            'timestamp': datetime.now().isoformat(),
            'step': 'Data Collection',
            'details': f'Collected {len(price_data)} tokens ({selection_mode} mode)',
            'duration_seconds': data_time
        })
        
        if len(price_data) < 2:
            print("Error: Need at least 2 tokens for portfolio analysis.")
            return
            
    except Exception as e:
        print(f"✗ Error collecting  {e}")
        return
    
    # Step 2: Token Evaluation (Parallel)
    print("\n2. Evaluating tokens with multiprocessing...")
    eval_start = time.time()
    
    try:
        evaluator = TokenEvaluator(price_data, market_data, n_cores=os.cpu_count())
        token_scores = evaluator.evaluate_all_tokens()
        
        eval_time = time.time() - eval_start
        print(f"✓ Token evaluation completed in {eval_time:.1f} seconds")
        
        execution_log.append({
            'timestamp': datetime.now().isoformat(),
            'step': 'Token Evaluation',
            'details': f'Evaluated {len(token_scores)} tokens',
            'duration_seconds': eval_time
        })
        analysis_data['token_scores'] = token_scores
        
        if not token_scores:
            print("✗ No tokens were successfully evaluated")
            return
        
    except Exception as e:
        print(f"✗ Error in token evaluation: {e}")
        return
    
    # Display token results
    print(f"\nToken Analysis Results ({len(token_scores)} tokens):")
    print("Rank | Token        | Score | Volatility | Market Cap    | Liquidity")
    print("-----|--------------|-------|------------|---------------|----------")
    
    for i, (token, score_data) in enumerate(list(token_scores.items())[:min(15, len(token_scores))]):
        vol = score_data['metrics']['volatility']
        mcap = score_data['metrics']['market_cap']
        liq_ratio = score_data['metrics']['liquidity_ratio']
        
        # Highlight manually selected tokens
        marker = "🎯" if selection_mode != 'auto' and token in manual_tokens else "  "
        
        print(f"{i+1:4d} | {token:12s} | {score_data['final_score']:5.1f} | "
              f"{vol:8.3f}   | ${mcap:11.2e} | {liq_ratio:8.6f} {marker}")
    
    if selection_mode != 'auto':
        print("🎯 = Manually selected token")
    
    suggestions = suggest_rebalancing_token_sets(token_scores, price_data)
    if suggestions:
        analysis_data['rebalancing_token_suggestions'] = suggestions
        print("\nRebalancing backtest token suggestions:")
        for suggestion in suggestions:
            print(f"  {suggestion['label']}: {', '.join(suggestion['tokens'])}")
    
    # Step 3: Portfolio Construction (Parallel)
    print(f"\n3. Building optimal portfolios with parallel processing...")
    portfolio_start = time.time()
    
    try:
        builder = PortfolioBuilder(token_scores, price_data, n_cores=os.cpu_count())
        
        # Adjust max tokens based on available tokens
        max_tokens = min(10, len(token_scores))
        portfolio_options = builder.find_optimal_portfolio_size(token_scores, max_tokens)
        
        portfolio_time = time.time() - portfolio_start
        print(f"✓ Portfolio construction completed in {portfolio_time:.1f} seconds")
        
        execution_log.append({
            'timestamp': datetime.now().isoformat(),
            'step': 'Portfolio Construction',
            'details': f'Built {len(portfolio_options)} portfolio options',
            'duration_seconds': portfolio_time
        })
        analysis_data['portfolio_options'] = portfolio_options
        
    except Exception as e:
        print(f"✗ Error in portfolio construction: {e}")
        return
    
    if not portfolio_options:
        print("✗ No valid portfolio options found")
        return
    
    print(f"\nPortfolio options found:")
    for size, portfolio in portfolio_options.items():
        tokens_str = ', '.join(portfolio['tokens'][:3])
        if len(portfolio['tokens']) > 3:
            tokens_str += f"... (+{len(portfolio['tokens'])-3} more)"
        print(f"  {size} tokens: {tokens_str} (Score: {portfolio['score']:.1f})")
    
    # Select best portfolio (prefer smaller portfolios for manual selection)
    if selection_mode == 'manual':
        # For manual selection, prefer portfolio size close to number of manual tokens
        target_size = min(len(manual_tokens), max(portfolio_options.keys()))
        preferred_size = target_size if target_size in portfolio_options else max(portfolio_options.keys())
    else:
        preferred_size = 4 if 4 in portfolio_options else max(portfolio_options.keys())
    
    best_portfolio = portfolio_options[preferred_size]
    print(f"\nSelected portfolio ({preferred_size} tokens): {', '.join(best_portfolio['tokens'])}")
    analysis_data['best_portfolio'] = best_portfolio
    
    # Step 4: Test Allocation Methods
    print(f"\n4. Testing allocation methods...")
    allocation_start = time.time()
    
    try:
        allocation_results = builder.test_all_allocation_methods(best_portfolio['tokens'])
        
        allocation_time = time.time() - allocation_start
        print(f"✓ Allocation method testing completed in {allocation_time:.1f} seconds")
        
        execution_log.append({
            'timestamp': datetime.now().isoformat(),
            'step': 'Allocation Testing',
            'details': f'Tested {len(allocation_results)} methods',
            'duration_seconds': allocation_time
        })
        analysis_data['allocation_results'] = allocation_results
        
        print("\nAllocation method comparison:")
        for method, result in allocation_results.items():
            print(f"  {method:20s}: Score {result['score']:6.1f}")
            # Show top 3 allocations
            sorted_allocations = sorted(result['allocations'].items(), 
                                      key=lambda x: x[1], reverse=True)
            for token, weight in sorted_allocations[:3]:
                print(f"    {token}: {weight:.1%}")
            if len(result['allocations']) > 3:
                print(f"    ... (+{len(result['allocations'])-3} more)")
        
        # Select best allocation method
        best_allocation_method = max(allocation_results.keys(), 
                                    key=lambda k: allocation_results[k]['score'])
        best_allocations = allocation_results[best_allocation_method]['allocations']
        
        print(f"\nSelected allocation method: {best_allocation_method}")
        analysis_data['best_allocation_method'] = best_allocation_method
        analysis_data['best_allocations'] = best_allocations
        
    except Exception as e:
        print(f"✗ Error in allocation testing: {e}")
        # Fallback to equal weight
        best_allocation_method = 'equal_weight'
        best_allocations = {token: 1/len(best_portfolio['tokens']) 
                           for token in best_portfolio['tokens']}
        print(f"Using fallback equal weight allocation")
        analysis_data['best_allocation_method'] = best_allocation_method
        analysis_data['best_allocations'] = best_allocations
    
    # Step 5: Optional Monte Carlo Optimization
    if len(best_portfolio['tokens']) <= 6:
        print(f"\n5. Running Monte Carlo optimization...")
        mc_start = time.time()
        
        try:
            mc_result = builder.parallel_monte_carlo_optimization(
                best_portfolio['tokens'], 
                n_simulations=50000
            )
            
            mc_time = time.time() - mc_start
            print(f"✓ Monte Carlo optimization completed in {mc_time:.1f} seconds")
            print(f"  Optimal Sharpe ratio: {mc_result['sharpe_ratio']:.3f}")
            print(f"  Expected return: {mc_result['expected_return']:.2%}")
            print(f"  Expected volatility: {mc_result['expected_volatility']:.2%}")
            
            execution_log.append({
                'timestamp': datetime.now().isoformat(),
                'step': 'Monte Carlo Optimization',
                'details': '50,000 simulations completed',
                'duration_seconds': mc_time
            })
            analysis_data['monte_carlo_result'] = mc_result
            
            # Use MC-optimized allocations if significantly better
            current_score = allocation_results[best_allocation_method]['score']
            mc_score = mc_result['sharpe_ratio'] * 100  # Rough comparison
            
            if mc_result['sharpe_ratio'] > 1.0 and mc_score > current_score:
                best_allocations = mc_result['best_allocation']
                best_allocation_method = 'monte_carlo_optimized'
                print("  → Using Monte Carlo optimized allocations")
                analysis_data['best_allocation_method'] = best_allocation_method
                analysis_data['best_allocations'] = best_allocations
            else:
                print(f"  → Keeping {best_allocation_method} allocations")
                
        except Exception as e:
            print(f"✗ Monte Carlo optimization failed: {e}")
    else:
        print(f"\n5. Skipping Monte Carlo (portfolio too large: {len(best_portfolio['tokens'])} tokens)")
    
    # Step 6: Backtesting with Multiple Configurations
    print(f"\n6. Running backtests with multiple configurations...")
    print(f"Transaction fee applied: {TRANSACTION_FEE:.2%} per trade")
    backtest_start = time.time()
    
    backtest_results = {}
    backtest_batch_results = {}
    backtest_window_errors = {}
    drift_thresholds = DRIFT_THRESHOLDS
    
    backtest_window_mode = BACKTEST_WINDOW_MODE
    
    try:
        backtest_windows = build_backtest_windows(
            backtest_window_mode,
            BACKTEST_START,
            BACKTEST_END,
            BACKTEST_WINDOW_DAYS,
            BACKTEST_WINDOW_STEP_DAYS
        )
    except ValueError as e:
        print(f"Invalid backtest window mode '{BACKTEST_WINDOW_MODE}': {e}. Using single window.")
        backtest_window_mode = 'single'
        try:
            backtest_windows = build_backtest_windows(
                backtest_window_mode,
                BACKTEST_START,
                BACKTEST_END,
                BACKTEST_WINDOW_DAYS,
                BACKTEST_WINDOW_STEP_DAYS
            )
        except ValueError as fallback_error:
            print(f"Backtest window fallback failed: {fallback_error}. Using full window.")
            backtest_windows = build_backtest_windows('single', BACKTEST_START, BACKTEST_END, None, None)
    
    if backtest_window_mode == 'single':
        primary_window = backtest_windows[0]
    else:
        primary_window = backtest_windows[-1]
    primary_window_label = primary_window['label']
    
    analysis_data['backtest_window_mode'] = backtest_window_mode
    analysis_data['backtest_window_label'] = primary_window_label
    analysis_data['backtest_window_range'] = {
        'start': primary_window['start_date'].date().isoformat(),
        'end': primary_window['end_date'].date().isoformat()
    }
    analysis_data['backtest_windows'] = [window['label'] for window in backtest_windows]
    
    window_label = "window" if len(backtest_windows) == 1 else "windows"
    print(f"Testing {len(drift_thresholds)} drift thresholds across "
          f"{len(backtest_windows)} {window_label} ({backtest_window_mode} mode):")
    if len(backtest_windows) > 1:
        print(f"Primary window for summary: {primary_window_label}")
    
    successful_backtests = 0
    
    for i, drift in enumerate(drift_thresholds):
        print(f"  [{i+1}/{len(drift_thresholds)}] Testing drift threshold: {drift:.1%}...", end="")
        
        portfolio_config = {
            'tokens': best_portfolio['tokens'],
            'allocations': best_allocations,
            'drift_threshold': drift,
            'min_interval': MIN_REBALANCE_INTERVAL,
            'max_interval': MAX_REBALANCE_INTERVAL
        }
        
        try:
            engine = BacktestEngine(price_data, portfolio_config)
            batch_results, batch_errors = engine.run_backtest_batch(
                backtest_windows,
                initial_capital=INITIAL_CAPITAL
            )
            
            if not batch_results:
                print(" ✗ (no successful windows)")
                continue
            
            backtest_batch_results[drift] = {
                label: results['performance_metrics']
                for label, results in batch_results.items()
            }
            
            if batch_errors:
                backtest_window_errors[drift] = batch_errors
            
            primary_results = batch_results.get(primary_window_label)
            if primary_results is None:
                print(" ✗ (primary window unavailable)")
                continue
            
            backtest_results[drift] = primary_results
            successful_backtests += 1
            window_summary = ""
            if len(backtest_windows) > 1:
                window_summary = f" ({len(batch_results)}/{len(backtest_windows)} windows)"
            print(f" ✓{window_summary}")
            
        except Exception as e:
            print(f" ✗ ({str(e)[:50]}...)")
            continue
    
    backtest_time = time.time() - backtest_start
    print(f"✓ Backtesting completed in {backtest_time:.1f} seconds")
    print(f"  Successful backtests: {successful_backtests}/{len(drift_thresholds)}")
    
    execution_log.append({
        'timestamp': datetime.now().isoformat(),
        'step': 'Backtesting',
        'details': (f'Completed {successful_backtests} backtests '
                    f'({len(backtest_windows)} windows, mode: {backtest_window_mode})'),
        'duration_seconds': backtest_time
    })
    analysis_data['backtest_results'] = backtest_results
    analysis_data['backtest_batch_results'] = backtest_batch_results
    if backtest_window_errors:
        analysis_data['backtest_window_errors'] = backtest_window_errors
    
    # Step 7: Performance Analysis
    print(f"\n7. Analyzing results...")
    
    if not backtest_results:
        print("✗ No successful backtests to analyze")
        print("This might be due to insufficient data or configuration issues.")
        return
    
    best_drift = None
    best_sharpe = -np.inf
    best_overall_score = -np.inf
    
    print(f"\nBacktest Results Summary:")
    print("Drift   | Return | Sharpe | Max DD | Trades | Fees  | Alpha  | Score")
    print("--------|--------|--------|--------|--------|-------|--------|-------")
    
    for drift, results in backtest_results.items():
        try:
            metrics = results['performance_metrics']
            
            # Calculate combined score
            return_score = min(metrics['annualized_return'] * 100, 50)  # Cap at 50%
            sharpe_score = min(metrics['sharpe_ratio'] * 20, 40)  # Cap at 2.0 Sharpe
            drawdown_score = min((1 + metrics['max_drawdown']) * 20, 20)  # Less negative DD
            alpha_score = min(metrics['alpha'] * 100, 20)  # Cap at 20%
            fee_score = max(0, 10 - metrics['fees_pct_of_portfolio'] * 1000)  # Penalize high fees
            
            overall_score = return_score + sharpe_score + drawdown_score + alpha_score + fee_score
            
            if metrics['sharpe_ratio'] > best_sharpe:
                best_sharpe = metrics['sharpe_ratio']
                best_drift = drift
            
            if overall_score > best_overall_score:
                best_overall_score = overall_score
            
            print(f"{drift:6.1%} | {metrics['annualized_return']:5.1%} | "
                  f"{metrics['sharpe_ratio']:5.2f} | {metrics['max_drawdown']:5.1%} | "
                  f"{metrics['total_trades']:6d} | {metrics['fees_pct_of_portfolio']:4.2%} | "
                  f"{metrics['alpha']:5.1%} | {overall_score:5.1f}")
                  
        except Exception as e:
            print(f"{drift:6.1%} | ERROR: {str(e)[:50]}...")
    
    print(f"\n✓ Best configuration: {best_drift:.1%} drift threshold (Sharpe: {best_sharpe:.3f})")
    analysis_data['best_drift'] = best_drift
    analysis_data['best_results'] = backtest_results[best_drift]
    
    # Step 8: Detailed Analysis of Best Strategy
    try:
        best_results = backtest_results[best_drift]
        analyzer = PerformanceAnalyzer(best_results)
        
        print(f"\n=== DETAILED PERFORMANCE REPORT ===")
        report = analyzer.generate_performance_report()
        print(report)
        
        # Calculate additional risk metrics
        try:
            risk_metrics = analyzer.calculate_risk_metrics()
            analysis_data['risk_metrics'] = risk_metrics
            
            print(f"\nADDITIONAL RISK METRICS:")
            print(f"- Value at Risk (5%): {risk_metrics['value_at_risk_5pct']:.3f}")
            print(f"- Conditional VaR (5%): {risk_metrics['conditional_var_5pct']:.3f}")
            print(f"- Skewness: {risk_metrics['skewness']:.3f}")
            print(f"- Kurtosis: {risk_metrics['kurtosis']:.3f}")
            print(f"- Positive Periods: {risk_metrics['positive_periods']:.1%}")
            print(f"- Max Consecutive Losses: {risk_metrics['max_consecutive_losses']} periods")
            print(f"- Calmar Ratio: {risk_metrics['calmar_ratio']:.3f}")
        except Exception as e:
            print(f"Could not calculate additional risk metrics: {e}")
        
        # Trading analysis
        try:
            trade_analysis = analyzer.generate_trade_analysis()
            if 'total_trades' in trade_analysis:
                print(f"\nTRADING ANALYSIS:")
                print(f"- Total Trades: {trade_analysis['total_trades']}")
                print(f"- Average Trade Size: ${trade_analysis['avg_trade_size']:,.0f}")
                print(f"- Largest Trade: ${trade_analysis['largest_trade']:,.0f}")
                print(f"- Buy/Sell Ratio: {trade_analysis['buy_vs_sell_ratio']:.2f}")
                print(f"- Avg Days Between Rebalances: {trade_analysis['avg_days_between_rebalances']:.1f}")
        except Exception as e:
            print(f"Could not generate trade analysis: {e}")
            
    except Exception as e:
        print(f"Error in detailed analysis: {e}")
    
    # Step 9: Save Results and Generate Charts
    print(f"\n9. Saving results and generating charts...")
    
    try:
        # Create results directory
        results_dir = 'results'
        plots_dir = 'plots'
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)
        
        # Save backtest results
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
                        'total_runtime_seconds': time.time() - start_time,
                        'data_age_days': cache_status.get('newest_file', {}).get('age_days', 0),
                        'backtest_window_mode': analysis_data.get('backtest_window_mode'),
                        'backtest_window_label': analysis_data.get('backtest_window_label'),
                        'backtest_window_range': analysis_data.get('backtest_window_range')
                    }
                }
            except Exception as e:
                print(f"Error serializing results for drift {drift}: {e}")
        
        results_file = os.path.join(results_dir, 'backtest_results.json')
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"✓ Results saved to {results_file}")
        
    except Exception as e:
        print(f"✗ Could not save JSON results: {e}")
    
    # Save detailed results as CSV
    try:
        if best_drift in backtest_results:
            analyzer = PerformanceAnalyzer(backtest_results[best_drift])
            analyzer.export_results('detailed_results.csv')
            print(f"✓ Detailed CSV results exported")
    except Exception as e:
        print(f"Could not export CSV results: {e}")
    
    # Generate performance charts
    save_plots = True
    if save_plots:
        try:
            chart_file = os.path.join(plots_dir, f'performance_charts_{datetime.now().strftime("%Y%m%d_%H%M")}.png')
            if best_drift in backtest_results:
                analyzer = PerformanceAnalyzer(backtest_results[best_drift])
                analyzer.plot_performance_charts(chart_file)
                print(f"✓ Performance charts saved to {chart_file}")
                
                # Add chart for potential report generation
                charts.append({
                    'path': chart_file,
                    'caption': 'Portfolio Performance Analysis',
                    'type': 'performance'
                })
                
        except Exception as e:
            print(f"Could not generate charts: {e}")
    
    # Step 10: Final Summary (before report generation)
    total_time = time.time() - start_time
    
    print(f"\n=== ANALYSIS COMPLETE ===")
    print(f"Total execution time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"  Data collection: {data_time:.1f}s")
    print(f"  Token evaluation: {eval_time:.1f}s")
    print(f"  Portfolio construction: {portfolio_time:.1f}s")
    print(f"  Allocation testing: {allocation_time:.1f}s")
    if 'mc_time' in locals():
        print(f"  Monte Carlo optimization: {mc_time:.1f}s")
    print(f"  Backtesting: {backtest_time:.1f}s")
    
    print(f"\n=== RECOMMENDED CONFIGURATION ===")
    print(f"Portfolio tokens: {', '.join(best_portfolio['tokens'])}")
    print(f"Allocation method: {best_allocation_method}")
    print(f"Optimal drift threshold: {best_drift:.1%}")
    
    if best_drift in backtest_results:
        best_metrics = backtest_results[best_drift]['performance_metrics']
        print(f"Expected performance:")
        print(f"  - Annual Return: {best_metrics['annualized_return']:.2%}")
        print(f"  - Volatility: {best_metrics['volatility']:.2%}")
        print(f"  - Sharpe Ratio: {best_metrics['sharpe_ratio']:.3f}")
        print(f"  - Max Drawdown: {best_metrics['max_drawdown']:.2%}")
        print(f"  - Alpha vs Benchmark: {best_metrics['alpha']:.2%}")
    
    print(f"\n🚀 Core analysis finished!")
    print(f"📁 Results saved in: results/")
    print(f"📊 Charts saved in: plots/")
    
    # Step 11: Optional Report Generation
    report_choice = prompt_for_report_generation()
    
    if report_choice != 'none':
        generated_file = generate_report_if_requested(report_choice, analysis_data, execution_log, charts)
        
        if generated_file:
            print(f"\n📄 AI-enhanced PDF report: {generated_file}")
        else:
            print(f"\n⚠️  Report generation failed or was skipped")
    
    # Final next steps
    print(f"\n=== NEXT STEPS ===")
    print(f"1. Review the analysis results and performance charts")
    if report_choice != 'none':
        print(f"2. Examine the comprehensive PDF report")
    print(f"3. Consider paper trading with the recommended configuration")
    print(f"4. Monitor correlation breakdowns and regime changes")
    print(f"5. Re-run analysis weekly with fresh data")
    print(f"6. Validate results with out-of-sample testing")



if __name__ == "__main__":
    # Set optimal multiprocessing settings
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1' 
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['NUMBA_NUM_THREADS'] = '1'
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Execution interrupted by user")
    except Exception as e:
        print(f"\n\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
