# Testing Guide for CryptoQuant Pro

## Overview

This directory contains the test suite for CryptoQuant Pro v0.2.0. The tests are organized into unit tests and integration tests, with comprehensive coverage of all major modules.

## Test Structure

```
tests/
├── conftest.py                     # Shared fixtures and configuration
├── unit/                           # Unit tests
│   ├── test_data_processing.py
│   ├── test_portfolio_optimization.py
│   ├── test_enhanced_metrics.py
│   └── ...
└── integration/                    # Integration tests
    └── ...
```

## Running Tests

### Run All Tests

```bash
pytest
```

### Run with Coverage

```bash
pytest --cov=. --cov-report=html
```

### Run Specific Test Categories

```bash
# Unit tests only
pytest tests/unit -v

# Integration tests only
pytest tests/integration -v

# Tests with specific marker
pytest -m unit
pytest -m "not slow"
```

### Run Specific Test File

```bash
pytest tests/unit/test_data_processing.py -v
```

### Run Specific Test Function

```bash
pytest tests/unit/test_data_processing.py::TestValidateManualTokens::test_all_valid_tokens -v
```

## Test Markers

Tests are categorized using pytest markers:

- `@pytest.mark.unit`: Unit tests (fast, isolated)
- `@pytest.mark.integration`: Integration tests (slower, require multiple components)
- `@pytest.mark.slow`: Slow-running tests (>5 seconds)
- `@pytest.mark.requires_api`: Tests requiring Binance API credentials
- `@pytest.mark.requires_data`: Tests requiring cached data

Example:
```python
@pytest.mark.unit
def test_something():
    pass

@pytest.mark.slow
@pytest.mark.requires_api
def test_api_integration():
    pass
```

## Writing Tests

### Basic Test Structure

```python
import pytest
from module_name import function_to_test

class TestFeatureName:
    """Tests for feature_name."""
    
    def test_normal_case(self):
        """Test normal operation."""
        result = function_to_test(input_data)
        assert result == expected_output
    
    def test_edge_case(self):
        """Test edge case."""
        result = function_to_test(edge_case_input)
        assert result is not None
    
    def test_error_handling(self):
        """Test error handling."""
        with pytest.raises(ValueError):
            function_to_test(invalid_input)
```

### Using Fixtures

```python
@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return {'key': 'value'}

def test_with_fixture(sample_data):
    """Test using fixture."""
    assert 'key' in sample_data
```

## Shared Fixtures

Common fixtures are defined in `conftest.py`:

- `test_config`: Test configuration parameters
- `sample_dates`: Date range for testing
- `generate_price_series`: Factory for generating price series
- `generate_token_data`: Factory for generating complete token data
- `sample_portfolio_config`: Sample portfolio configuration

## Coverage Goals

Target: **80% code coverage**

Current coverage can be viewed in `htmlcov/index.html` after running:

```bash
pytest --cov=. --cov-report=html
```

## Continuous Integration

Tests are automatically run in CI/CD pipelines. To ensure your changes pass:

1. Run all tests locally before committing
2. Check coverage doesn't decrease
3. Add tests for new features
4. Update tests when modifying existing features

## Best Practices

1. **Test Naming**: Use descriptive names that explain what is being tested
2. **Isolation**: Unit tests should not depend on external services or files
3. **Fixtures**: Use fixtures for common test data to avoid repetition
4. **Assertions**: Use specific assertions (assertEqual, assertIn, etc.) rather than generic assert
5. **Documentation**: Add docstrings to test classes and complex tests
6. **Mocking**: Use mocks for external dependencies (API calls, file I/O)

## Common Issues

### Import Errors

If you encounter import errors, ensure you're running pytest from the project root:

```bash
cd /path/to/CryptoQuantPro
pytest
```

### Missing Dependencies

Install test dependencies:

```bash
pip install pytest pytest-cov pytest-mock
```

### API Credentials

Tests marked with `@pytest.mark.requires_api` will be skipped if API credentials are not configured. To run these tests:

1. Configure `config/secrets.py` with valid credentials
2. Remove the marker or run with: `pytest -m requires_api`

## Contributing

When adding new features:

1. Write tests first (TDD approach recommended)
2. Ensure tests pass before submitting PR
3. Maintain or improve code coverage
4. Follow existing test patterns and conventions
