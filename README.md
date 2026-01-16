# CryptoQuant Pro 🚀

**Advanced Cryptocurrency Portfolio Optimization with AI-Enhanced Analytics**

**Version 0.1.0 - Initial Public Release**

---

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey.svg)
![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-Optimized-orange.svg)

---

## Overview

CryptoQuant Pro is a high-performance, comprehensive cryptocurrency portfolio analysis and rebalancing system. It integrates multi-factor token evaluation, robust portfolio optimization, realistic backtesting, and advanced risk analytics. It also features AI-powered, professional PDF report generation via local Ollama models.

Ideal for quants, crypto funds, and advanced retail traders, CryptoQuant Pro combines state-of-the-art quantitative finance with modern software engineering optimized for Apple Silicon and multi-core setups.

---

## Features

### Core Analytics
- Multi-factor scoring on liquidity, volatility, stability, market dominance, and mean reversion
- Portfolio size optimization between 2 and 10 tokens with multiple allocation schemes
- Monte Carlo simulations for portfolio weight optimization (up to 6 tokens)
- Realistic backtesting including transaction costs and rebalance constraints
- Extensive risk metrics including VaR, CVaR, drawdown, skewness, and kurtosis

### Performance & Scalability
- Fully parallel and multiprocessing-enabled (up to 20+ CPU cores)
- Apple Silicon (M1/M2/M3) optimized for high throughput and low latency
- Intelligent 7-day data caching to minimize API calls and speed workflows
- Robust error handling for API limits and data inconsistencies

### AI-Driven Reporting
- Local Ollama-based AI integration (Gemma2 model) for generating insightful investment commentary and explanations
- Interactive command-line interface for user-friendly parameter selection
- Detailed PDF reports with embedded charts and AI insights
- Customizable report generation with or without AI explanations

### Extensibility & Modularity
- Clean, modular Python codebase designed for easy customization
- Configurable parameters loaded from a dedicated settings file
- Secure API key management via separate secrets module

---

## Getting Started

### Prerequisites
- Python 3.8+
- Local installation of [Ollama](https://ollama.ai/) with Gemma3n model (optional, for AI reports)

### Installation
```
git clone https://github.com/eplt/cryptoquantpro.git
cd cryptoquantpro
pip install -r requirements.txt
```

Install Ollama if you want AI-augmented reporting:

```
# Follow instructions at https://ollama.ai/
ollama pull gemma3n:latest
ollama serve
```

### Configuration

Fill in your Binance API credentials in `config/secrets.py`:

```
BINANCE_API_KEY = "your_api_key"
BINANCE_SECRET_KEY = "your_secret_key"
```

Adjust parameters in `config/settings.py` if needed; e.g.:

```
BACKTEST_HORIZON = 'medium'        # 'medium' (6-12 months) or 'long' (2-3 years)
LOOKBACK_DAYS = 730               # Historical data length (days)
INTERVAL = '1d'                   # Price data interval
PORTFOLIO_CONFIG = {
    'min_tokens': 2,
    'max_tokens': 10,
    'preferred_tokens': 4,
    'max_correlation': 0.75
}
REBALANCING_CONFIG = {
    'drift_thresholds': [0.08, 0.10, 0.12, 0.15, 0.20, 0.25],
    'min_rebalance_interval': 5,
    'max_rebalance_interval': 21,
    'transaction_cost': 0.0006
}
BACKTEST_WINDOW_MODE = 'single'      # 'single', 'rolling', or 'expanding'
BACKTEST_WINDOW_DAYS = 365           # Window size for rolling/expanding modes
BACKTEST_WINDOW_STEP_DAYS = 90       # Step size between windows
```

---

## Usage

Run the main analysis with:

```
python main.py
```

Follow prompts to:

- Select tokens manually or auto (top by market cap)
- Review portfolio options (sizes 2–10)
- Backtest all portfolio options and compare results
- Choose the final portfolio for detailed reporting
- Generate AI-powered comprehensive PDF reports (optional)

---

## Project Structure

```
CryptoQuant Pro/
├── data/
│   ├── data_collector.py       # Market data fetching and caching
│   └── cache/                  # Cached historical data
├── evaluation/
│   ├── token_evaluator.py      # Token scoring engine
│   └── portfolio_builder.py    # Portfolio optimization algorithms
├── backtesting/
│   └── backtest_engine.py      # Backtesting framework
├── analysis/
│   ├── performance_metrics.py  # Performance and risk computations
│   └── report_generator.py     # AI-enhanced report generation
├── config/
│   ├── settings.py             # Configuration parameters
│   └── secrets.py              # API credentials (not tracked in Git)
├── results/                    # JSON and CSV output files
├── plots/                      # Generated performance & portfolio plots
├── requirements.txt            # Python dependencies
└── main.py                    # Main interactive execution script
```

---

## Performance Benchmarks

| System   | Cores     | Token Eval Time | Portfolio Opt Time | Total Time |
|----------|-----------|-----------------|--------------------|------------|
| M3 Max   | 20        | ~8s             | ~12s               | ~45s       |
| M2 Pro   | 12        | ~12s            | ~18s               | ~65s       |
| Intel i7 | 8         | ~25s            | ~35s               | ~120s      |

---

## AI Integration (Optional)

Leverages [Ollama](https://ollama.ai) for local AI-powered report generation using the `gemma3n:latest` model. Adds:

- Professional investment and risk commentary
- Clear strategic recommendations
- Contextual interpretation of backtesting results

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

- Issues: https://github.com/eplt/cryptoquantpro/issues  
- Discussions: https://github.com/eplt/cryptoquantpro/discussions  

---

**Made with ❤️ for the crypto community**

*Please star ⭐ the repository if you find it useful!*
