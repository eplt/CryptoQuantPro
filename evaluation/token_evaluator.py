import pandas as pd
import numpy as np
import multiprocessing as mp
from functools import partial
import os

class TokenEvaluator:
    def __init__(self, price_data, market_data, n_cores=None):
        self.price_data = price_data
        self.market_data = market_data
        self.scores = {}
        
        # Use all cores by default on Apple Silicon
        if n_cores is None:
            self.n_cores = os.cpu_count()  # Should be 20 on your machine
        else:
            self.n_cores = min(n_cores, os.cpu_count())
            
        print(f"Using {self.n_cores} cores for parallel processing")
        
    def calculate_metrics(self, symbol):
        """Calculate comprehensive metrics for a token"""
        df = self.price_data[symbol]
        returns = np.log(df['close'] / df['close'].shift(1)).dropna()
        
        metrics = {
            'volatility': returns.std() * np.sqrt(365),
            'max_drawdown': self._calculate_max_drawdown(df['close']),
            'var_95': returns.quantile(0.05),
            'sharpe_ratio': self._calculate_sharpe(returns),
            'avg_volume_usd': (df['volume'] * df['close']).mean(),
            'volume_consistency': self._calculate_volume_consistency(df),
            'liquidity_ratio': self.market_data[symbol]['volume_24h'] / self.market_data[symbol]['market_cap'],
            'returns_1y': (df['close'].iloc[-1] / df['close'].iloc[0]) - 1,
            'returns_6m': (df['close'].iloc[-1] / df['close'].iloc[-180]) - 1 if len(df) > 180 else 0,
            'trend_strength': self._calculate_trend_strength(df['close']),
            'market_cap': self.market_data[symbol]['market_cap'],
            'market_rank': self.market_data[symbol]['rank'],
            'mean_reversion': self._calculate_mean_reversion(returns),
        }
        return metrics
    
    def _calculate_volume_consistency(self, df):
        volume_usd = df['volume'] * df['close']
        mean_vol = volume_usd.mean()
        if mean_vol == 0:
            return 10
        std_vol = volume_usd.std()
        return std_vol / mean_vol
    
    def _calculate_max_drawdown(self, prices):
        peak = prices.expanding().max()
        drawdown = (prices - peak) / peak
        return drawdown.min()
    
    def _calculate_sharpe(self, returns, risk_free_rate=0.02):
        excess_returns = returns.mean() * 365 - risk_free_rate
        volatility = returns.std() * np.sqrt(365)
        return excess_returns / volatility if volatility > 0 else 0
    
    def _calculate_trend_strength(self, prices, window=50):
        if len(prices) < window:
            return 0.5
        ma_short = prices.rolling(window // 2).mean()
        ma_long = prices.rolling(window).mean()
        trend_signals = (ma_short > ma_long).astype(int)
        return trend_signals.rolling(window).mean().iloc[-1]
    
    def _calculate_mean_reversion(self, returns, window=20):
        returns_series = returns.dropna()
        
        if len(returns_series) < window * 3:
            return 0.5
        
        returns_series = returns_series[np.isfinite(returns_series)]
        
        if len(returns_series) < window * 2:
            return 0.5
            
        try:
            max_lag = min(window, len(returns_series) // 3)
            lags = list(range(2, max_lag))
            
            if len(lags) < 3:
                return 0.5
            
            tau = []
            for lag in lags:
                if lag >= len(returns_series):
                    continue
                    
                diff = np.subtract(returns_series[lag:], returns_series[:-lag])
                
                if len(np.unique(diff)) < 2:
                    continue
                    
                var_diff = np.var(diff)
                if var_diff <= 0:
                    continue
                    
                tau_val = np.sqrt(var_diff)
                if tau_val > 0:
                    tau.append(tau_val)
            
            if len(tau) < 3:
                return 0.5
            
            tau = np.array(tau)
            tau = tau[tau > 1e-10]
            
            if len(tau) < 3:
                return 0.5
            
            lags = lags[:len(tau)]
            
            epsilon = 1e-10
            log_lags = np.log(np.array(lags) + epsilon)
            log_tau = np.log(tau + epsilon)
            
            if not np.all(np.isfinite(log_lags)) or not np.all(np.isfinite(log_tau)):
                return 0.5
            
            poly = np.polyfit(log_lags, log_tau, 1)
            hurst = poly[0] * 2
            
            return np.clip(hurst, 0.1, 0.9)
            
        except Exception:
            return 0.5
    
    def score_token(self, symbol):
        metrics = self.calculate_metrics(symbol)
        
        weights = {
            'liquidity': 0.25,
            'volatility': 0.20,
            'stability': 0.20,
            'market_position': 0.15,
            'mean_reversion': 0.10,
            'performance': 0.10
        }
        
        scores = {
            'liquidity': self._score_liquidity(metrics),
            'volatility': self._score_volatility(metrics['volatility']),
            'stability': self._score_stability(metrics['max_drawdown'], metrics['var_95']),
            'market_position': self._score_market_position(metrics['market_cap'], metrics['market_rank']),
            'mean_reversion': self._score_mean_reversion(metrics['mean_reversion']),
            'performance': self._score_performance(metrics['returns_1y'], metrics['sharpe_ratio'])
        }
        
        final_score = sum(scores[component] * weights[component] for component in scores)
        
        return {
            'final_score': final_score,
            'component_scores': scores,
            'metrics': metrics
        }
    
    def _score_liquidity(self, metrics):
        volume_score = min(100, metrics['avg_volume_usd'] / 1e6)
        consistency_score = max(0, 100 - metrics['volume_consistency'] * 50)
        liquidity_ratio_score = min(100, metrics['liquidity_ratio'] * 1000)
        return (volume_score + consistency_score + liquidity_ratio_score) / 3
    
    def _score_volatility(self, volatility):
        if 0.3 <= volatility <= 0.8:
            return 100
        elif volatility < 0.3:
            return 50 + (volatility / 0.3) * 50
        else:
            return max(0, 100 - (volatility - 0.8) * 100)
    
    def _score_stability(self, max_drawdown, var_95):
        drawdown_score = max(0, 100 + max_drawdown * 200)
        var_score = max(0, 100 + var_95 * 500)
        return (drawdown_score + var_score) / 2
    
    def _score_market_position(self, market_cap, rank):
        cap_score = min(100, np.log10(market_cap / 1e9) * 25) if market_cap > 0 else 0
        rank_score = max(0, 100 - rank) if rank and rank <= 100 else 0
        return (cap_score + rank_score) / 2
    
    def _score_mean_reversion(self, hurst_exponent):
        if 0.3 <= hurst_exponent <= 0.4:
            return 100
        else:
            return max(0, 100 - abs(hurst_exponent - 0.35) * 200)
    
    def _score_performance(self, returns_1y, sharpe_ratio):
        return_score = min(100, max(0, (returns_1y + 1) * 50))
        sharpe_score = min(100, max(0, (sharpe_ratio + 2) * 25))
        return (return_score + sharpe_score) / 2
    
    def evaluate_all_tokens(self):
        """Evaluate all tokens using multiprocessing"""
        # Filter symbols that have market data
        valid_symbols = [symbol for symbol in self.price_data.keys() 
                        if symbol in self.market_data]
        
        print(f"Evaluating {len(valid_symbols)} tokens using {self.n_cores} cores...")
        
        if self.n_cores == 1:
            # Single-threaded fallback
            return self._evaluate_single_threaded(valid_symbols)
        
        # Multiprocessing approach
        try:
            with mp.Pool(processes=self.n_cores) as pool:
                # Create partial function with bound self
                score_func = partial(self._score_token_worker, 
                                   price_data=self.price_data, 
                                   market_data=self.market_data)
                
                # Process symbols in parallel
                results_list = pool.map(score_func, valid_symbols)
                
                # Combine results
                results = {}
                for symbol, result in zip(valid_symbols, results_list):
                    if result is not None:
                        results[symbol] = result
                
        except Exception as e:
            print(f"Multiprocessing failed: {e}")
            print("Falling back to single-threaded execution...")
            return self._evaluate_single_threaded(valid_symbols)
        
        # Sort by final score
        sorted_results = dict(sorted(results.items(), 
                                   key=lambda x: x[1]['final_score'], 
                                   reverse=True))
        
        return sorted_results
    
    def _evaluate_single_threaded(self, symbols):
        """Fallback single-threaded evaluation"""
        results = {}
        for symbol in symbols:
            try:
                results[symbol] = self.score_token(symbol)
            except Exception as e:
                print(f"Error evaluating {symbol}: {e}")
        
        return dict(sorted(results.items(), 
                         key=lambda x: x[1]['final_score'], 
                         reverse=True))
    
    @staticmethod
    def _score_token_worker(symbol, price_data, market_data):
        """Worker function for multiprocessing"""
        try:
            # Create a temporary evaluator instance for this worker
            evaluator = TokenEvaluator(price_data, market_data, n_cores=1)
            return evaluator.score_token(symbol)
        except Exception as e:
            print(f"Error evaluating {symbol}: {e}")
            return None
