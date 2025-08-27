import numpy as np
from datetime import timedelta
from config.settings import *

class Rebalancer:
    def __init__(self, portfolio_config):
        self.portfolio_config = portfolio_config
        self.drift_threshold = portfolio_config.get('drift_threshold', 0.15)
        self.min_interval = portfolio_config.get('min_interval', MIN_REBALANCE_INTERVAL)
        self.max_interval = portfolio_config.get('max_interval', MAX_REBALANCE_INTERVAL)
        
    def should_rebalance(self, current_weights, current_date, last_rebalance_date):
        """Determine if portfolio should be rebalanced"""
        target_weights = self.portfolio_config['allocations']
        
        # Calculate days since last rebalance
        days_since_rebalance = (current_date - last_rebalance_date).days
        
        # Check minimum interval
        if days_since_rebalance < self.min_interval:
            return False, {'reason': 'min_interval_not_met'}
        
        # Calculate weight drifts
        weight_drifts = {}
        total_drift = 0
        max_drift = 0
        
        for token in target_weights:
            target_weight = target_weights[token]
            current_weight = current_weights.get(token, 0)
            drift = abs(current_weight - target_weight)
            
            weight_drifts[token] = {
                'current': current_weight,
                'target': target_weight,
                'drift': drift,
                'drift_pct': drift / target_weight if target_weight > 0 else 0
            }
            
            total_drift += drift
            max_drift = max(max_drift, drift)
        
        # Rebalancing conditions
        conditions = {
            'max_drift_exceeded': max_drift > self.drift_threshold,
            'total_drift_exceeded': total_drift > self.drift_threshold * 2,
            'max_interval_reached': days_since_rebalance >= self.max_interval,
            'significant_deviation': any(d['drift_pct'] > 0.5 for d in weight_drifts.values())
        }
        
        should_rebalance = any(conditions.values())
        
        rebalance_info = {
            'weight_drifts': weight_drifts,
            'total_drift': total_drift,
            'max_drift': max_drift,
            'days_since_rebalance': days_since_rebalance,
            'conditions': conditions,
            'trigger_reason': [k for k, v in conditions.items() if v]
        }
        
        return should_rebalance, rebalance_info
    
    def calculate_rebalancing_trades(self, current_values, target_allocations, total_value):
        """Calculate required trades for rebalancing"""
        trades = []
        
        for token in target_allocations:
            current_value = current_values.get(token, 0)
            target_value = total_value * target_allocations[token]
            trade_amount = target_value - current_value
            
            if abs(trade_amount) > total_value * 0.001:  # Only trade if significant
                trades.append({
                    'token': token,
                    'current_value': current_value,
                    'target_value': target_value,
                    'trade_amount': trade_amount,
                    'side': 'buy' if trade_amount > 0 else 'sell'
                })
        
        return trades
    
    def optimize_trade_sequence(self, trades):
        """Optimize the sequence of trades to minimize market impact"""
        # Sort trades: sells first, then buys
        sells = [t for t in trades if t['side'] == 'sell']
        buys = [t for t in trades if t['side'] == 'buy']
        
        # Sort sells by largest amount first (free up cash)
        sells.sort(key=lambda x: abs(x['trade_amount']), reverse=True)
        
        # Sort buys by smallest amount first (use available cash efficiently)
        buys.sort(key=lambda x: abs(x['trade_amount']))
        
        return sells + buys
