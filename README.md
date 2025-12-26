# CryptoQuant Pro 🚀

**Advanced Cryptocurrency Portfolio Optimization with AI-Enhanced Analytics**

**Version 0.2.0 - Major Refactoring Release**

---

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Version](https://img.shields.io/badge/version-0.2.0-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey.svg)
![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-Optimized-orange.svg)

---

## Overview

CryptoQuant Pro is a high-performance, comprehensive cryptocurrency portfolio analysis and rebalancing system. It integrates multi-factor token evaluation, robust portfolio optimization, realistic backtesting, and advanced risk analytics. Version 0.2.0 introduces major architectural improvements, enhanced backtesting capabilities, and rich reporting features.

Ideal for quants, crypto funds, and advanced retail traders, CryptoQuant Pro combines state-of-the-art quantitative finance with modern software engineering optimized for Apple Silicon and multi-core setups.

---

## What's New in v0.2.0

### 🏗️ **Modular Architecture**
- **Refactored Codebase**: Main script reduced from 841 to 568 lines
- **Separate Modules**: `data_processing`, `portfolio_optimization`, `reporting`
- **Enhanced Data Module**: Complete data collection and caching system

### 📊 **Advanced Backtesting**
- **Walk-Forward Analysis**: Robust out-of-sample testing with rolling windows
- **Monte Carlo Simulations**: Risk assessment with 10,000+ simulation paths
- **Scenario Testing**: Bull, bear, crash, and sideways market scenarios
- **Stress Testing**: Extreme market condition analysis

### 📈 **Rich Reporting**
- **Excel Export**: Multi-sheet workbooks with comprehensive analysis
- **HTML Dashboards**: Interactive Plotly-based visualizations
- **Enhanced Metrics**: 20+ performance metrics including Sortino, Calmar, Omega ratios
- **Risk Analytics**: VaR, CVaR, alpha, beta, information ratio

### 🎯 **Improved Features**
- Better token selection modes (Auto/Manual/Hybrid)
- Configurable backtesting scenarios
- Performance benchmarking tools
- Enhanced logging and error handling

---

## Features

### Core Analytics
- Multi-factor scoring on liquidity, volatility, stability, market dominance, and mean reversion
- Portfolio size optimization between 2 and 10 tokens with multiple allocation schemes
- Monte Carlo simulations for portfolio weight optimization
- Realistic backtesting including transaction costs and rebalance constraints
- Walk-forward analysis for robust strategy validation
- Extensive risk metrics including VaR, CVaR, drawdown, skewness, and kurtosis

### Performance & Scalability
- Fully parallel and multiprocessing-enabled (up to 20+ CPU cores)
- Apple Silicon (M1/M2/M3) optimized for high throughput and low latency
- Intelligent 7-day data caching to minimize API calls and speed workflows
- Robust error handling for API limits and data inconsistencies

### AI-Driven Reporting
- Local Ollama-based AI integration (Gemma2 model) for generating insightful investment commentary
- Interactive command-line interface for user-friendly parameter selection
- Detailed PDF reports with embedded charts and AI insights
- Excel workbooks with multiple analysis sheets
- Interactive HTML dashboards with Plotly

### Extensibility & Modularity
- Clean, modular Python codebase designed for easy customization
- Configurable parameters loaded from dedicated settings file
- Secure API key management via separate secrets module
- Easy to extend with custom strategies and metrics

---

## Getting Started

### Prerequisites
- Python 3.8+
- Binance API credentials (for data collection)
- Optional: Local installation of [Ollama](https://ollama.ai/) with Gemma3n model (for AI reports)

### Installation
```bash
git clone https://github.com/eplt/CryptoQuantPro.git
cd CryptoQuantPro
pip install -r requirements.txt
```

Install Ollama if you want AI-augmented reporting:

```bash
# Follow instructions at https://ollama.ai/
ollama pull gemma3n:latest
ollama serve
```

### Configuration

1. **API Credentials**: Fill in your Binance API credentials in `config/secrets.py`:

```python
BINANCE_API_KEY = "your_api_key"
BINANCE_SECRET_KEY = "your_secret_key"
```

2. **Settings**: Adjust parameters in `config/settings.py` if needed:

```python
LOOKBACK_DAYS = 730               # Historical data length (days)
INTERVAL = '1d'                   # Price data interval
PORTFOLIO_CONFIG = {
    'min_tokens': 2,
    'max_tokens': 10,
    'preferred_tokens': 4,
    'max_correlation': 0.75
}
```

---

## Usage

### Basic Analysis

Run the main analysis with:

```bash
python main.py
```

Follow prompts to:
- Select tokens manually or auto (top by market cap)
- Review portfolio options (sizes 2–10)
- Backtest all portfolio options and compare results
- Choose the final portfolio for detailed reporting
- Generate AI-powered comprehensive PDF reports (optional)

### Advanced Features

#### Walk-Forward Analysis

```python
from backtesting.walk_forward import WalkForwardAnalyzer

analyzer = WalkForwardAnalyzer(price_data)
results = analyzer.run_walk_forward(portfolio_config)
```

#### Monte Carlo Simulation

```python
from backtesting.monte_carlo import MonteCarloSimulator

simulator = MonteCarloSimulator(price_data)
results = simulator.run_simulation(portfolio_config, n_simulations=10000)
```

#### Scenario Testing

```python
from backtesting.test_scenarios import TestScenarios

tester = TestScenarios(price_data)
results = tester.generate_scenario_report(portfolio_config)
```

#### Excel Reports

```python
from analysis.excel_export import ExcelReportGenerator

generator = ExcelReportGenerator()
filepath = generator.generate_comprehensive_report(analysis_data)
```

#### HTML Dashboards

```python
from analysis.html_dashboard import HTMLDashboardGenerator

dashboard = HTMLDashboardGenerator()
filepath = dashboard.generate_dashboard(analysis_data)
```

---

## Project Structure

```
CryptoQuant Pro/
├── data/
│   ├── data_collector.py       # Market data fetching and caching
│   └── cache/                  # Cached historical data (gitignored)
├── evaluation/
│   ├── token_evaluator.py      # Token scoring engine
│   └── portfolio_builder.py    # Portfolio optimization algorithms
├── backtesting/
│   ├── backtest_engine.py      # Core backtesting framework
│   ├── walk_forward.py         # Walk-forward analysis
│   ├── monte_carlo.py          # Monte Carlo simulations
│   ├── test_scenarios.py       # Scenario testing
│   └── rebalancer.py           # Rebalancing logic
├── analysis/
│   ├── performance_metrics.py  # Performance and risk computations
│   ├── enhanced_metrics.py     # Advanced metrics (Sortino, Calmar, etc.)
│   ├── excel_export.py         # Excel report generation
│   ├── html_dashboard.py       # Interactive HTML dashboards
│   └── report_generator.py     # AI-enhanced PDF reports
├── config/
│   ├── settings.py             # Configuration parameters
│   ├── backtesting_config.py   # Backtesting scenarios
│   └── secrets.py              # API credentials (not tracked in Git)
├── examples/                   # Example notebooks and scripts
├── data_processing.py          # Data processing utilities
├── portfolio_optimization.py   # Portfolio optimization utilities
├── reporting.py                # Reporting utilities
├── results/                    # JSON and CSV output files
├── plots/                      # Generated performance plots
├── requirements.txt            # Python dependencies
└── main.py                     # Main interactive execution script
```

---

## Examples

Check the `examples/` directory for:
- `basic_analysis.ipynb` - Complete portfolio analysis walkthrough
- `walk_forward_testing.ipynb` - Walk-forward analysis examples
- `monte_carlo_risk.ipynb` - Monte Carlo risk analysis
- `custom_strategies.ipynb` - Building custom strategies

---

## Performance Benchmarks

| System   | Cores     | Token Eval Time | Portfolio Opt Time | Total Time |
|----------|-----------|-----------------|--------------------|------------|
| M3 Max   | 20        | ~8s             | ~12s               | ~45s       |
| M2 Pro   | 12        | ~12s            | ~18s               | ~65s       |
| Intel i7 | 8         | ~25s            | ~35s               | ~120s      |

---

## Contributing

Contributions are welcome! Please fork the repo, make improvements, and submit Pull Requests.

---

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for a detailed history of changes and releases.

---

## License

MIT License — See [LICENSE](LICENSE) for details.

---

## Disclaimer

This project is for educational and research purposes only. Cryptocurrency trading involves significant risk. Past performance is not indicative of future results. Users are responsible for their own investment decisions.

---

## Support

- Issues: https://github.com/eplt/CryptoQuantPro/issues  
- Discussions: https://github.com/eplt/CryptoQuantPro/discussions  

---

**Made with ❤️ for the crypto community**

*Please star ⭐ the repository if you find it useful!*
