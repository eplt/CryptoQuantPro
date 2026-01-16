"""
Unit tests for parallel processing configuration.

Tests the safety features and configuration options for portfolio optimization
parallel processing, ensuring proper defaults and environment variable overrides.
"""

import os
import unittest
import warnings
from unittest.mock import patch
from config.parallel_config import (
    configure_blas_threads,
    get_safe_n_jobs,
    warn_if_high_n_jobs
)


class TestParallelConfig(unittest.TestCase):
    """Test parallel processing configuration utilities."""
    
    def test_get_safe_n_jobs_default_cap(self):
        """Test that n_jobs is capped at max_default when not explicitly requested."""
        # With 64 CPUs available, should cap at default (8)
        n_jobs = get_safe_n_jobs(requested_n_jobs=None, max_default=8, cpu_count=64)
        self.assertEqual(n_jobs, 8)
        
        # With 4 CPUs available, should use all 4
        n_jobs = get_safe_n_jobs(requested_n_jobs=None, max_default=8, cpu_count=4)
        self.assertEqual(n_jobs, 4)
    
    def test_get_safe_n_jobs_explicit_request(self):
        """Test that explicitly requested n_jobs is honored."""
        # User explicitly requests 20, should get 20
        n_jobs = get_safe_n_jobs(requested_n_jobs=20, max_default=8, cpu_count=64)
        self.assertEqual(n_jobs, 20)
        
        # User explicitly requests 1, should get 1
        n_jobs = get_safe_n_jobs(requested_n_jobs=1, max_default=8, cpu_count=64)
        self.assertEqual(n_jobs, 1)
    
    def test_get_safe_n_jobs_minimum_one(self):
        """Test that n_jobs is at least 1."""
        n_jobs = get_safe_n_jobs(requested_n_jobs=0, max_default=8, cpu_count=64)
        self.assertEqual(n_jobs, 1)
        
        n_jobs = get_safe_n_jobs(requested_n_jobs=-5, max_default=8, cpu_count=64)
        self.assertEqual(n_jobs, 1)
    
    def test_warn_if_high_n_jobs_triggers_warning(self):
        """Test that high n_jobs values trigger a warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warn_if_high_n_jobs(n_jobs=20, threshold=16)
            
            # Should have triggered a warning
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[0].category, RuntimeWarning))
            self.assertIn("20 parallel jobs", str(w[0].message))
            self.assertIn("memory usage", str(w[0].message))
    
    def test_warn_if_high_n_jobs_no_warning_below_threshold(self):
        """Test that n_jobs below threshold doesn't trigger warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warn_if_high_n_jobs(n_jobs=8, threshold=16)
            
            # Should not have triggered a warning
            self.assertEqual(len(w), 0)
    
    def test_configure_blas_threads_sets_env_vars(self):
        """Test that configure_blas_threads sets expected environment variables."""
        # Clear any existing thread env vars
        thread_vars = [
            'OMP_NUM_THREADS',
            'OPENBLAS_NUM_THREADS',
            'MKL_NUM_THREADS',
            'NUMEXPR_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS'
        ]
        
        original_values = {}
        for var in thread_vars:
            if var in os.environ:
                original_values[var] = os.environ[var]
                del os.environ[var]
        
        try:
            # Configure with 1 thread
            configure_blas_threads(num_threads=1)
            
            # Check that all variables are set
            for var in thread_vars:
                self.assertIn(var, os.environ)
                self.assertEqual(os.environ[var], '1')
        finally:
            # Restore original values
            for var in thread_vars:
                if var in original_values:
                    os.environ[var] = original_values[var]
                elif var in os.environ:
                    del os.environ[var]
    
    def test_configure_blas_threads_respects_existing_vars(self):
        """Test that configure_blas_threads doesn't override existing env vars."""
        # Set one variable explicitly
        os.environ['OMP_NUM_THREADS'] = '4'
        
        try:
            configure_blas_threads(num_threads=1)
            
            # Should not have overridden the existing value
            self.assertEqual(os.environ['OMP_NUM_THREADS'], '4')
        finally:
            # Clean up
            if 'OMP_NUM_THREADS' in os.environ:
                del os.environ['OMP_NUM_THREADS']


class TestPortfolioBuilderConfig(unittest.TestCase):
    """Test PortfolioBuilder configuration with environment variables."""
    
    def setUp(self):
        """Save original environment variables."""
        self.original_n_jobs = os.environ.get('PORTFOLIO_N_JOBS')
        self.original_backend = os.environ.get('PORTFOLIO_BACKEND')
    
    def tearDown(self):
        """Restore original environment variables."""
        if self.original_n_jobs is not None:
            os.environ['PORTFOLIO_N_JOBS'] = self.original_n_jobs
        elif 'PORTFOLIO_N_JOBS' in os.environ:
            del os.environ['PORTFOLIO_N_JOBS']
        
        if self.original_backend is not None:
            os.environ['PORTFOLIO_BACKEND'] = self.original_backend
        elif 'PORTFOLIO_BACKEND' in os.environ:
            del os.environ['PORTFOLIO_BACKEND']
    
    def test_portfolio_n_jobs_env_var_override(self):
        """Test that PORTFOLIO_N_JOBS environment variable is respected."""
        # Set environment variable
        os.environ['PORTFOLIO_N_JOBS'] = '12'
        
        # Need to reload settings module to pick up new env var
        import importlib
        import config.settings
        importlib.reload(config.settings)
        
        from config.settings import PORTFOLIO_N_JOBS
        self.assertEqual(PORTFOLIO_N_JOBS, 12)
    
    def test_portfolio_backend_env_var_override(self):
        """Test that PORTFOLIO_BACKEND environment variable is respected."""
        # Set environment variable
        os.environ['PORTFOLIO_BACKEND'] = 'threading'
        
        # Need to reload settings module to pick up new env var
        import importlib
        import config.settings
        importlib.reload(config.settings)
        
        from config.settings import PORTFOLIO_BACKEND
        self.assertEqual(PORTFOLIO_BACKEND, 'threading')
    
    def test_default_values_without_env_vars(self):
        """Test default values when environment variables are not set."""
        # Ensure env vars are not set
        if 'PORTFOLIO_N_JOBS' in os.environ:
            del os.environ['PORTFOLIO_N_JOBS']
        if 'PORTFOLIO_BACKEND' in os.environ:
            del os.environ['PORTFOLIO_BACKEND']
        
        # Need to reload settings module
        import importlib
        import config.settings
        importlib.reload(config.settings)
        
        from config.settings import PORTFOLIO_N_JOBS, PORTFOLIO_BACKEND
        
        # Should use safe defaults
        self.assertEqual(PORTFOLIO_N_JOBS, 8)
        self.assertEqual(PORTFOLIO_BACKEND, 'loky')


if __name__ == '__main__':
    unittest.main()
