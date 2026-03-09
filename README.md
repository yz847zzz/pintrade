# PINtrade: An Alpha Factor Research & Backtesting Engine

PINtrade is a robust Python framework designed for quantitative finance enthusiasts and researchers to develop, test, and analyze various alpha trading factors. It provides a structured pipeline for data loading, factor computation (including advanced market microstructure metrics like the Probability of Informed Trading - PIN), signal generation, and vectorized backtesting. With PINtrade, you can quickly evaluate the performance of your trading strategies against historical market data.

## Key Features

-   **Flexible Data Loading**: Seamlessly pull OHLCV data for multiple tickers from Yahoo Finance.
-   **Comprehensive Factor Library**: Compute a suite of well-known alpha factors, including:
    -   Momentum (21D, 63D, 252D)
    -   Relative Strength Index (RSI_5D)
    -   Price Z-score (20D)
    -   Volatility (20D)
    -   Volume Z-score (20D)
    -   **Probability of Informed Trading (PIN)**: Calculated using the Easley-Kiefer-O'Hara-Paperman (EKOP) model.
    -   **Daily Event Labels**: Classify trading days as 'Good News' (+1), 'Bad News' (-1), or 'No Event' (0) based on the EKOP model.
-   **Modular Alpha Model Design**: Easily integrate custom alpha models with an abstract base class.
-   **Vectorized Backtesting Engine**: Efficiently simulate trading strategies with configurable rebalancing periods (monthly/weekly) and top-N stock selection.
-   **Performance Metrics**: Automatically calculate Sharpe Ratio, Maximum Drawdown, and Annualized Return.
-   **Visualization**: Generate equity curve plots to visually assess strategy performance.
-   **GitHub Actions CI/CD**: Basic setup for continuous integration to ensure code quality and test coverage.

## Project Structure

```
.github/
└── workflows/
    └── ci.yml             # GitHub Actions for CI/CD
pintrade/
├── data/
│   └── loader.py          # Data loading (e.g., from Yahoo Finance)
├── features/
│   ├── __init__.py
│   ├── factors.py         # Computation of alpha factors (Momentum, RSI, PIN, etc.)
│   └── ekop_model.py      # Implementation of the EKOP model for PIN factor
├── models/
│   ├── __init__.py
│   ├── base.py            # Abstract base class for alpha models
│   └── factor_model.py    # Factor-based alpha model implementation
├── backtest/
│   ├── __init__.py
│   └── engine.py          # Vectorized backtesting engine
└── utils/
    ├── __init__.py
    └── metrics.py         # Performance metrics (Sharpe Ratio, Max Drawdown, Annualized Return)
.env                       # Environment variables (e.g., GitHub Token)
.gitignore                 # Files and directories to ignore in Git
main.py                    # Main script to run the entire pipeline
pytest.ini                 # Pytest configuration
requirements.txt           # Python dependencies
setup_github.py            # Script to set up Git for pushing to GitHub
README.md                  # Project README
```

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/YOUR_GITHUB_USERNAME/pintrade.git
    cd pintrade
    ```

2.  **Create a Python virtual environment and activate it:**
    ```bash
    python -m venv venv
    # On Windows:
    # .\venv\Scripts\activate
    # On macOS/Linux:
    # source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Git for GitHub (Optional, for pushing changes):**
    Create a `.env` file in the root directory of the project with your GitHub Token and Username:
    ```
    GITHUB_TOKEN="YOUR_GITHUB_TOKEN"
    GITHUB_USERNAME="YOUR_GITHUB_USERNAME"
    ```
    **Note:** Your GitHub Token needs `repo` scope to push to the repository. Then, run the setup script:
    ```bash
    python setup_github.py
    ```
    This script will initialize your local Git repository, add your files, commit them, configure the remote, and push your changes to GitHub.

## How to Run the Backtest Pipeline

To execute the full PINtrade pipeline, which includes data loading, signal generation, and backtesting, simply run:

```bash
python main.py
```

This will:
-   Download historical OHLCV data for predefined tickers.
-   Compute various alpha factors, including PIN and daily event labels.
-   Generate composite alpha signals.
-   Run a vectorized backtest with the specified `top_n` stocks and `rebalance` frequency.
-   Output key backtest performance metrics (Sharpe Ratio, Max Drawdown, Annualized Return) to the console.
-   Save an equity curve plot as `backtest_equity_curve.png` in the project root directory.

## Testing

To run unit tests (if any are implemented in the future):

```bash
pytest
```
