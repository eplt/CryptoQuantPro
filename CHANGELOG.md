# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-12-26

### Added

#### Core Architecture
- Modular codebase structure with separate modules:
  - `data_processing.py`: Token selection and data validation utilities
  - `portfolio_optimization.py`: Portfolio construction and optimization utilities
  - `reporting.py`: Report generation and export utilities
  - `data/` module: Complete data collection and caching system

#### Enhanced Backtesting
- `backtesting/walk_forward.py`: Walk-forward analysis with rolling windows
  - Configurable training/testing window sizes
  - Multiple strategy comparison
  - Out-of-sample performance validation
- `backtesting/monte_carlo.py`: Monte Carlo simulations
  - 10,000+ simulation paths
  - VaR and CVaR calculation
  - Stress testing scenarios
  - Risk distribution analysis
- `backtesting/test_scenarios.py`: Scenario-based testing
  - Bull, bear, crash, and sideways market scenarios
  - Automatic period identification
  - Robustness scoring
- `config/backtesting_config.py`: Comprehensive backtesting configurations

#### Reporting Enhancements
- `analysis/excel_export.py`: Excel report generation
  - Multi-sheet workbooks
  - Executive summary, token scores, portfolio options
  - Backtest results and risk metrics
  - Performance timeline data
- `analysis/html_dashboard.py`: Interactive HTML dashboards
  - Plotly-based visualizations
  - Portfolio performance charts
  - Risk-return profiles
  - Token comparison tools
- `analysis/enhanced_metrics.py`: Advanced performance metrics
  - Sharpe, Sortino, Calmar, Omega ratios
  - Alpha, beta, information ratio
  - VaR, CVaR at multiple confidence levels
  - Skewness, kurtosis, tail ratio
  - Win rate and profit factor statistics

#### Testing Framework
- Comprehensive pytest-based testing suite
- Unit tests for all major modules
- Test fixtures and configuration
- Coverage reporting (target: 80%)
- Integration test structure
- Continuous integration support

#### Documentation
- Updated README.md with v0.2.0 features
- Examples directory with usage documentation
- Test documentation and guidelines
- Configuration guides

### Changed
- Reduced main.py from 841 to 568 lines through modularization
- Improved token selection with Auto/Manual/Hybrid modes
- Enhanced error handling throughout codebase
- Better progress reporting and logging
- Optimized performance with parallel processing

### Fixed
- Corrected Binance package name in requirements (python-binance)
- Improved data caching mechanism
- Better handling of missing data
- Enhanced validation for manual token selection

### Dependencies
- Added openpyxl>=3.0.0 for Excel export
- Added plotly>=5.0.0 for interactive visualizations
- Added dash>=2.0.0 for dashboard support
- Added pytest>=7.0.0 for testing
- Added pytest-cov>=3.0.0 for coverage reporting
- Added pytest-mock>=3.6.0 for mocking support
- Updated python-binance>=0.3.0 (fixed package name)

## [0.1.0] - 2025-12-25

### Added
- Initial public release of CryptoQuant Pro
- Multi-factor scoring system for cryptocurrency token evaluation
  - Liquidity scoring
  - Volatility analysis
  - Stability metrics
  - Market dominance evaluation
  - Mean reversion indicators
- Portfolio optimization engine
  - Support for 2-10 token portfolios
  - Monte Carlo simulations for weight optimization (up to 6 tokens)
  - Multiple allocation schemes (equal weight, optimized, risk-parity)
- Realistic backtesting framework
  - Transaction cost modeling
  - Rebalance constraints
  - Multiple rebalancing strategies
- Advanced risk analytics
  - Value at Risk (VaR)
  - Conditional Value at Risk (CVaR)
  - Drawdown analysis
  - Skewness and kurtosis metrics
- Performance optimization
  - Full multiprocessing support (up to 20+ CPU cores)
  - Apple Silicon (M1/M2/M3) optimization
  - Intelligent 7-day data caching
- AI-powered reporting
  - Local Ollama integration with Gemma2 model
  - Professional PDF report generation
  - Investment commentary and strategic recommendations
- Interactive command-line interface
  - Manual, auto, and hybrid token selection modes
  - User-friendly parameter selection
  - Real-time progress feedback
- Data collection and caching system
  - Binance API integration
  - Robust error handling
  - Rate limit management
- Modular and extensible architecture
  - Clean Python codebase
  - Configurable settings module
  - Secure API key management

### Documentation
- Comprehensive README with installation instructions
- Configuration guidelines
- Usage examples
- Performance benchmarks
- Project structure overview

[0.2.0]: https://github.com/eplt/CryptoQuantPro/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/eplt/CryptoQuantPro/releases/tag/v0.1.0
