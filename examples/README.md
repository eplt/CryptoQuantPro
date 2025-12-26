# CryptoQuant Pro - Examples

This directory contains example Jupyter notebooks demonstrating various features of CryptoQuant Pro v0.2.0.

## Notebooks

### 1. Basic Analysis (`basic_analysis.ipynb`)
Complete walkthrough of portfolio analysis:
- Data collection and caching
- Token evaluation and scoring
- Portfolio construction and optimization
- Allocation method comparison
- Backtesting with multiple drift thresholds
- Performance reporting

### 2. Walk-Forward Testing (`walk_forward_testing.ipynb`)
Demonstrates robust strategy validation:
- Setting up walk-forward windows
- Training and testing periods
- Out-of-sample performance analysis
- Strategy comparison across windows

### 3. Monte Carlo Risk Analysis (`monte_carlo_risk.ipynb`)
Risk assessment with simulations:
- Running 10,000+ simulation paths
- VaR and CVaR calculation
- Probability distributions
- Stress testing scenarios

### 4. Custom Strategies (`custom_strategies.ipynb`)
Building your own strategies:
- Custom allocation methods
- Custom rebalancing logic
- Custom risk metrics
- Integration with existing framework

### 5. Advanced Reporting (`advanced_reporting.ipynb`)
Generating comprehensive reports:
- Excel workbook generation
- Interactive HTML dashboards
- Custom visualizations
- AI-powered insights

## Running the Notebooks

1. Install Jupyter:
```bash
pip install jupyter
```

2. Start Jupyter:
```bash
jupyter notebook
```

3. Navigate to the examples directory and open any notebook

## Requirements

All notebooks require:
- CryptoQuant Pro installed with dependencies
- Binance API credentials configured in `config/secrets.py`
- Historical data cached (run `main.py` first for initial data collection)

Optional for some notebooks:
- Ollama running locally (for AI-powered reports)

## Data Considerations

- Initial data collection may take 5-10 minutes
- Data is cached for 7 days to speed up subsequent runs
- Notebooks use the same cached data as the main application

## Support

For questions or issues with the examples:
- Open an issue: https://github.com/eplt/CryptoQuantPro/issues
- Start a discussion: https://github.com/eplt/CryptoQuantPro/discussions
