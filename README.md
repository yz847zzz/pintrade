# PINtrade

PINtrade is an alpha factor research and backtesting engine. It allows you to define, compute, and backtest various quantitative trading factors, including market microstructure-inspired factors like the Probability of Informed Trading (PIN).

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
setup_github.py            # Script to set up GitHub repository and push initial commit
README.md                  # Project README
```

## Setup and Installation

1.  **Clone the repository (or initialize it if you're setting up for the first time):**
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

4.  **Configure GitHub (Optional, for initial repository creation and push):**
    Create a `.env` file in the root directory of the project with your GitHub Token and Username:
    ```
    GITHUB_TOKEN="YOUR_GITHUB_TOKEN"
    GITHUB_USERNAME="YOUR_GITHUB_USERNAME"
    ```
    **Note:** Your GitHub Token needs `repo` scope to create repositories. Then run:
    ```bash
    python setup_github.py
    ```

## How to Run

To run the full PINtrade pipeline (data loading, signal generation, backtesting):

```bash
python main.py
```

This will output the backtest results (Sharpe Ratio, Max Drawdown, Annualized Return) and save an equity curve plot as `backtest_equity_curve.png` in the project root.

## Testing

To run unit tests (if any are implemented):

```bash
pytest
```
