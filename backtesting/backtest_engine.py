import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from backtesting.rebalancer import Rebalancer
from config.settings import *

def build_backtest_windows(mode, start_date, end_date, window_days=None, step_days=None):
    """Build backtest windows based on mode."""
    normalized_mode = (mode or 'single').lower()
    
    if normalized_mode == 'single' or window_days is None or window_days <= 0:
        return [{
            'label': 'full',
            'start_date': start_date,
            'end_date': end_date
        }]
    
    if step_days is None:
        step_days = window_days
    if step_days <= 0:
        raise ValueError("Backtest window step_days must be positive.")
    windows = []
    
    if normalized_mode == 'rolling':
        window_start = start_date
        while window_start + timedelta(days=window_days) <= end_date:
            window_end = window_start + timedelta(days=window_days)
            windows.append({
                'label': format_backtest_window_label(window_start, window_end),
                'start_date': window_start,
                'end_date': window_end
            })
            window_start += timedelta(days=step_days)
    elif normalized_mode == 'expanding':
        window_end = start_date + timedelta(days=window_days)
        while window_end <= end_date:
            windows.append({
                'label': format_backtest_window_label(start_date, window_end),
                'start_date': start_date,
                'end_date': window_end
            })
            window_end += timedelta(days=step_days)
    else:
        raise ValueError(f"Unsupported backtest window mode: {mode}")
    
    if not windows:
        return [{
            'label': 'full',
            'start_date': start_date,
            'end_date': end_date
        }]
    
    return windows

def format_backtest_window_label(start_date, end_date):
    """Format a window label for backtests."""
    return f"{start_date:%Y%m%d}-{end_date:%Y%m%d}"

class BacktestEngine:
    def __init__(self, price_data, portfolio_config):
        self.price_data = price_data
        self.portfolio_config = portfolio_config
        self.rebalancer = Rebalancer(portfolio_config)
        
    def run_backtest(self, start_date=BACKTEST_START, end_date=BACKTEST_END, 
                     initial_capital=INITIAL_CAPITAL):
        """Run comprehensive backtest"""
        
        # Prepare data
        portfolio_data = self.prepare_portfolio_data(start_date, end_date)
        
        results = {
            'trades': [],
            'portfolio_values': [],
            'weights_history': [],
            'rebalance_dates': [],
            'performance_metrics': {}
        }
        
        # Initialize portfolio
        current_weights = self.portfolio_config['allocations'].copy()
        current_values = {token: initial_capital * weight 
                         for token, weight in current_weights.items()}
        cash = 0
        
        last_rebalance_date = start_date
        
        # Daily simulation
        for date in portfolio_data.index:
            daily_data = portfolio_data.loc[date]
            
            # Update portfolio values based on price changes
            for token in current_values:
                if date > start_date:  # Skip first day
                    price_change = daily_data[f'{token}_return'] + 1
                    current_values[token] *= price_change
            
            total_value = sum(current_values.values()) + cash
            current_weights = {token: value / total_value 
                             for token, value in current_values.items()}
            
            # Check rebalancing conditions
            should_rebalance, rebalance_info = self.rebalancer.should_rebalance(
                current_weights, date, last_rebalance_date
            )
            
            if should_rebalance:
                # Execute rebalancing
                trades, new_values, new_cash = self.execute_rebalance(
                    current_values, cash, daily_data, total_value
                )
                
                current_values = new_values
                cash = new_cash
                last_rebalance_date = date
                
                results['trades'].extend(trades)
                results['rebalance_dates'].append(date)
            
            # Record daily state
            results['portfolio_values'].append({
                'date': date,
                'total_value': total_value,
                'cash': cash,
                **current_values
            })
            
            results['weights_history'].append({
                'date': date,
                **current_weights
            })
        
        # Calculate performance metrics
        results['performance_metrics'] = self.calculate_performance_metrics(results)
        
        return results

    def run_backtest_batch(self, windows, initial_capital=INITIAL_CAPITAL):
        """Run backtests across multiple windows."""
        batch_results = {}
        batch_errors = {}
        
        for window in windows:
            label = window.get('label') or format_backtest_window_label(
                window['start_date'],
                window['end_date']
            )
            
            try:
                batch_results[label] = self.run_backtest(
                    start_date=window['start_date'],
                    end_date=window['end_date'],
                    initial_capital=initial_capital
                )
            except Exception as e:
                batch_errors[label] = str(e)
        
        return batch_results, batch_errors
    
    def prepare_portfolio_data(self, start_date, end_date):
        """Prepare aligned price data for all portfolio tokens"""
        portfolio_data = pd.DataFrame()
        
        for token in self.portfolio_config['tokens']:
            token_data = self.price_data[token].loc[start_date:end_date]
            
            # Calculate returns
            returns = token_data['close'].pct_change()
            
            portfolio_data[f'{token}_price'] = token_data['close']
            portfolio_data[f'{token}_return'] = returns
            portfolio_data[f'{token}_volume'] = token_data['volume']
        
        return portfolio_data.dropna()
    
    def execute_rebalance(self, current_values, cash, daily_data, total_value):
        """Execute rebalancing trades"""
        target_allocations = self.portfolio_config['allocations']
        trades = []
        new_values = current_values.copy()
        new_cash = cash
        
        # Calculate target values
        target_values = {token: total_value * weight 
                        for token, weight in target_allocations.items()}
        
        # Execute trades
        for token in current_values:
            current_val = current_values[token]
            target_val = target_values[token]
            trade_amount = target_val - current_val
            
            if abs(trade_amount) > total_value * 0.001:  # Only trade if >0.1% of portfolio
                # Calculate fees
                fee = abs(trade_amount) * TRANSACTION_FEE
                
                if trade_amount > 0:  # Buy
                    new_values[token] = target_val - fee
                    new_cash -= trade_amount
                else:  # Sell
                    new_values[token] = target_val
                    new_cash -= trade_amount - fee  # Add cash, subtract fee
                
                trades.append({
                    'date': daily_data.name,
                    'token': token,
                    'side': 'buy' if trade_amount > 0 else 'sell',
                    'amount': abs(trade_amount),
                    'fee': fee,
                    'price': daily_data[f'{token}_price']
                })
        
        return trades, new_values, new_cash
    
    def calculate_performance_metrics(self, results):
        """Calculate comprehensive performance metrics"""
        portfolio_df = pd.DataFrame(results['portfolio_values'])
        portfolio_df.set_index('date', inplace=True)
        
        # Calculate returns
        portfolio_returns = portfolio_df['total_value'].pct_change().dropna()
        
        # Basic metrics
        total_return = (portfolio_df['total_value'].iloc[-1] / portfolio_df['total_value'].iloc[0]) - 1
        annualized_return = (1 + total_return) ** (365 / len(portfolio_df)) - 1
        volatility = portfolio_returns.std() * np.sqrt(365)
        
        # Risk metrics
        max_drawdown = self.calculate_max_drawdown(portfolio_df['total_value'])
        sharpe_ratio = (annualized_return - 0.02) / volatility if volatility > 0 else 0
        
        # Trading metrics
        total_trades = len(results['trades'])
        total_fees = sum(trade['fee'] for trade in results['trades'])
        rebalance_frequency = len(results['rebalance_dates'])
        
        # Calculate benchmark (buy and hold)
        benchmark_return = self.calculate_benchmark_return(portfolio_df.index[0], portfolio_df.index[-1])
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': total_trades,
            'total_fees': total_fees,
            'fees_pct_of_portfolio': total_fees / INITIAL_CAPITAL,
            'rebalance_frequency': rebalance_frequency,
            'benchmark_return': benchmark_return,
            'alpha': total_return - benchmark_return
        }
    
    def calculate_max_drawdown(self, portfolio_values):
        """Calculate maximum drawdown"""
        peak = portfolio_values.expanding().max()
        drawdown = (portfolio_values - peak) / peak
        return drawdown.min()
    
    def calculate_benchmark_return(self, start_date, end_date):
        """Calculate buy-and-hold return for the portfolio"""
        target_allocations = self.portfolio_config['allocations']
        benchmark_return = 0
        
        for token, weight in target_allocations.items():
            token_data = self.price_data[token].loc[start_date:end_date]
            token_return = (token_data['close'].iloc[-1] / token_data['close'].iloc[0]) - 1
            benchmark_return += weight * token_return
        
        return benchmark_return
