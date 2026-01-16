"""
Parallel processing configuration utilities.

This module handles thread limits for BLAS/OMP libraries to avoid nested parallelism
issues when using joblib's multiprocessing, particularly on Apple Silicon (M1/M2/M3)
where excessive parallelism can lead to memory pressure and SIGKILL errors.
"""

import os
import warnings


def configure_blas_threads(num_threads=1):
    """
    Configure BLAS and OpenMP thread limits to avoid nested parallelism.
    
    This function sets environment variables that control threading in various
    numerical libraries (BLAS, OpenMP, MKL, NumExpr, etc.). When using joblib
    with many processes, having each process spawn additional threads can lead
    to oversubscription and memory issues.
    
    Args:
        num_threads (int): Number of threads for each worker process (default: 1)
    
    Note:
        This should be called early in the program, before importing numpy or other
        numerical libraries. If environment variables are already set, they will
        not be overridden.
    """
    thread_env_vars = {
        'OMP_NUM_THREADS': str(num_threads),        # OpenMP
        'OPENBLAS_NUM_THREADS': str(num_threads),   # OpenBLAS
        'MKL_NUM_THREADS': str(num_threads),        # Intel MKL
        'NUMEXPR_NUM_THREADS': str(num_threads),    # NumExpr
        'VECLIB_MAXIMUM_THREADS': str(num_threads), # macOS Accelerate framework
    }
    
    set_vars = []
    already_set = []
    
    for var, value in thread_env_vars.items():
        if var not in os.environ:
            os.environ[var] = value
            set_vars.append(var)
        else:
            already_set.append(f"{var}={os.environ[var]}")
    
    if set_vars:
        print(f"Configured BLAS/OMP thread limits: {', '.join(set_vars)} = {num_threads}")
    
    if already_set:
        print(f"Thread env vars already set (not overridden): {', '.join(already_set)}")


def get_safe_n_jobs(requested_n_jobs=None, max_default=8, cpu_count=None):
    """
    Get a safe number of parallel jobs, capping at a reasonable default.
    
    Args:
        requested_n_jobs (int, optional): User-requested number of jobs
        max_default (int): Maximum default if not explicitly requested (default: 8)
        cpu_count (int, optional): Available CPU count (defaults to os.cpu_count())
    
    Returns:
        int: Safe number of jobs to use
    """
    if cpu_count is None:
        cpu_count = os.cpu_count() or 1
    
    if requested_n_jobs is not None:
        # User explicitly requested this value, honor it
        n_jobs = requested_n_jobs
    else:
        # Use conservative default, capped at max_default
        n_jobs = min(cpu_count, max_default)
    
    return max(1, n_jobs)  # Ensure at least 1


def warn_if_high_n_jobs(n_jobs, threshold=16):
    """
    Warn if n_jobs is set to a high value that might cause issues.
    
    Args:
        n_jobs (int): Number of parallel jobs
        threshold (int): Threshold above which to warn (default: 16)
    """
    if n_jobs > threshold:
        warnings.warn(
            f"Using {n_jobs} parallel jobs may cause excessive memory usage and "
            f"potential crashes on some systems (especially macOS with Apple Silicon). "
            f"Consider setting PORTFOLIO_N_JOBS environment variable to a lower value "
            f"(e.g., 8-12) if you experience issues.",
            RuntimeWarning,
            stacklevel=2
        )
