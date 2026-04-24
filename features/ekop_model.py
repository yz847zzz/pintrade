"""
pintrade/features/ekop_model.py

Standard EKOP Model (Easley-Kiefer-O'Hara-Paperman 1996)
Pure Poisson arrivals — no Inverse Gaussian mixing (VDJ)

Parameters: (alpha, delta, mu, epsi)
  alpha : probability of information event
  delta : probability of bad news | event
  mu    : informed trader arrival rate
  epsi  : uninformed trader arrival rate (buy = sell)

Scenarios:
  No event  : B ~ Pois(epsi),    S ~ Pois(epsi)
  Good news : B ~ Pois(epsi+mu), S ~ Pois(epsi)
  Bad news  : B ~ Pois(epsi),    S ~ Pois(epsi+mu)

PIN (model) = alpha*mu / (alpha*mu + 2*epsi)
PIN (empirical) = (Good days + Bad days) / Total days
"""

import numpy as np
from scipy.special import gammaln
from scipy.optimize import minimize
import pandas as pd


# ── 1. Poisson log PMF ────────────────────────────────────────────────────────

def log_poisson(k: np.ndarray, lam: float) -> np.ndarray:
    """
    Log PMF of Poisson(lam) evaluated at k.
    log g(k; lam) = k*log(lam) - lam - log(k!)
    """
    epsilon = 1e-300
    return k * np.log(lam + epsilon) - lam - gammaln(k + 1)


# ── 2. EKOP NLL (equation 3-4 from paper) ────────────────────────────────────

def ekop_nll(params: np.ndarray, buys: np.ndarray, sells: np.ndarray) -> float:
    """
    EKOP negative log-likelihood.
    params = [alpha, delta, mu, epsi]

    L_t = (1-alpha) * g(b;epsi)*g(s;epsi)
        + alpha*(1-delta) * g(b;epsi+mu)*g(s;epsi)
        + alpha*delta     * g(b;epsi)*g(s;epsi+mu)
    """
    alpha, delta, mu, epsi = params
    alpha = np.clip(alpha, 1e-6, 1-1e-6)
    delta = np.clip(delta, 1e-6, 1-1e-6)
    mu    = max(mu,   1e-6)
    epsi  = max(epsi, 1e-6)

    # Per-day log-likelihoods for each scenario
    log_none = log_poisson(buys, epsi)    + log_poisson(sells, epsi)
    log_good = log_poisson(buys, epsi+mu) + log_poisson(sells, epsi)
    log_bad  = log_poisson(buys, epsi)    + log_poisson(sells, epsi+mu)

    # Log weights
    lw_none = np.log(1.0 - alpha)
    lw_good = np.log(alpha) + np.log(1.0 - delta)
    lw_bad  = np.log(alpha) + np.log(delta)

    # Log-sum-exp per day
    log_comp = np.column_stack([
        lw_none + log_none,
        lw_good + log_good,
        lw_bad  + log_bad,
    ])
    m      = log_comp.max(axis=1, keepdims=True)
    loglik = m.squeeze() + np.log(np.exp(log_comp - m).sum(axis=1))
    loglik[np.isinf(loglik)] = -1e10

    return -np.sum(loglik)


# ── 3. MLE fitting ────────────────────────────────────────────────────────────

def fit_ekop(buys: np.ndarray, sells: np.ndarray) -> dict:
    """
    Fit EKOP via MLE with multiple initial guesses.

    Initial guesses spread across plausible parameter space.
    epsi ≈ mean daily volume (uninformed dominates)
    mu   ≈ small fraction of mean volume
    """
    buys  = np.asarray(buys,  dtype=float)
    sells = np.asarray(sells, dtype=float)

    mean_v = (buys.mean() + sells.mean()) / 2.0

    # [alpha, delta, mu, epsi]
    # epsi should be close to mean volume, mu is a fraction of it
    initial_guesses = [
        [0.3, 0.5, mean_v * 0.10, mean_v * 0.90],
        [0.5, 0.5, mean_v * 0.20, mean_v * 0.80],
        [0.2, 0.3, mean_v * 0.05, mean_v * 0.95],
        [0.7, 0.5, mean_v * 0.30, mean_v * 0.70],
        [0.4, 0.6, mean_v * 0.15, mean_v * 0.85],
    ]
    bounds = [
        (1e-6, 1-1e-6),   # alpha
        (1e-6, 1-1e-6),   # delta
        (1e-6, None),      # mu
        (1e-6, None),      # epsi
    ]

    best_nll    = np.inf
    best_params = None

    for x0 in initial_guesses:
        try:
            res = minimize(
                ekop_nll, x0, args=(buys, sells),
                method='SLSQP', bounds=bounds,
                options={'maxiter': 5000, 'ftol': 1e-10}
            )
            if res.fun < best_nll:
                best_nll    = res.fun
                best_params = res.x
        except Exception:
            continue

    if best_params is None:
        return dict(alpha=np.nan, delta=np.nan, mu=np.nan, epsi=np.nan,
                    PIN_model=np.nan, PIN_empirical=np.nan, converged=False)

    alpha, delta, mu, epsi = best_params
    pin_model = alpha * mu / (alpha * mu + 2.0 * epsi)

    return dict(alpha=float(alpha), delta=float(delta),
                mu=float(mu), epsi=float(epsi),
                PIN_model=float(pin_model),
                NLL=float(best_nll), converged=True)


# ── 4. Daily Bayesian classification ─────────────────────────────────────────

def classify_days_ekop(buys: np.ndarray, sells: np.ndarray,
                       params: dict) -> np.ndarray:
    """
    Classify each day via posterior argmax.
    Returns: +1 (Good), -1 (Bad), 0 (No Event)
    """
    alpha = params['alpha']
    delta = params['delta']
    mu    = params['mu']
    epsi  = params['epsi']

    log_comp = np.column_stack([
        np.log(alpha*(1-delta)) + log_poisson(buys, epsi+mu) + log_poisson(sells, epsi),      # Good
        np.log(alpha*delta)     + log_poisson(buys, epsi)    + log_poisson(sells, epsi+mu),    # Bad
        np.log(1.0 - alpha)     + log_poisson(buys, epsi)    + log_poisson(sells, epsi),       # None
    ])

    x_hat = np.argmax(log_comp, axis=1)
    return np.array([+1, -1, 0])[x_hat]


# ── 5. Lee-Ready estimator ────────────────────────────────────────────────────

def estimate_buy_sell_volume(df: pd.DataFrame) -> pd.DataFrame:
    """
    BVC (Bulk Volume Classification):
    buy_ratio = (Close - Low) / (High - Low)
    Gives continuous buy/sell split instead of binary Lee-Ready.
    """
    high = df['High'].values
    low  = df['Low'].values
    buy_ratio = (df['Close'].values - low) / (high - low + 1e-10)
    buy_ratio = np.clip(buy_ratio, 0.0, 1.0)
    out = df.copy()
    out['Buy_Volume']  = df['Volume'] * buy_ratio
    out['Sell_Volume'] = df['Volume'] * (1.0 - buy_ratio)
    return out


# ── 6. Wide → Long ────────────────────────────────────────────────────────────

def _wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.columns, pd.MultiIndex):
        return df
    level = 0 if df.columns.names[0] == 'Ticker' else 1
    df_long = df.stack(level=level, future_stack=True)
    df_long.index.names = ['Date', 'Ticker']
    return df_long


# ── 7. Main factor computation ────────────────────────────────────────────────

def compute_ekop_factor(
    df: pd.DataFrame,
    period: str = 'annual'   # 'annual', 'monthly', or 'both'
) -> pd.DataFrame:
    """
    For each ticker and each calendar window:
      1. Estimate buy/sell volume via Lee-Ready
      2. Fit EKOP MLE ONCE per window
      3. Classify every day: Good (+1) / Bad (-1) / No Event (0)
      4. PIN_empirical = (Good + Bad) / Total
      5. PIN_model     = alpha*mu / (alpha*mu + 2*epsi)

    Returns DataFrame (Date, Ticker):
      PIN_empirical : float ∈ (0,1) — frequency of informed days
      PIN_model     : float ∈ (0,1) — model-implied PIN
      event_label   : int ∈ {+1,-1,0} — daily classification
    """
    df_long = _wide_to_long(df)
    df_vol  = estimate_buy_sell_volume(df_long)
    tickers = df_vol.index.get_level_values('Ticker').unique()
    periods = (['annual', 'monthly'] if period == 'both' else [period])
    results = []

    for ticker in tickers:
        tk_df = df_vol.xs(ticker, level='Ticker')

        for p in periods:
            groups = (tk_df.groupby(tk_df.index.year) if p == 'annual'
                      else tk_df.groupby([tk_df.index.year, tk_df.index.month]))

            for key, group in groups:
                buys  = group['Buy_Volume'].values
                sells = group['Sell_Volume'].values

                if len(buys) < 5:
                    continue

                params = fit_ekop(buys, sells)
                if not params['converged']:
                    continue

                labels      = classify_days_ekop(buys, sells, params)
                n_event     = int(np.sum(labels != 0))
                pin_emp     = n_event / len(labels)

                print(f"  {ticker} {key}: "
                      f"alpha={params['alpha']:.3f} delta={params['delta']:.3f} "
                      f"mu={params['mu']:.0f} epsi={params['epsi']:.0f} "
                      f"PIN_model={params['PIN_model']:.3f} PIN_emp={pin_emp:.3f} "
                      f"Good={int((labels==1).sum())} "
                      f"Bad={int((labels==-1).sum())} "
                      f"None={int((labels==0).sum())}")

                for i, date in enumerate(group.index):
                    results.append({
                        'Date':        date,
                        'Ticker':      ticker,
                        'PIN':         params['PIN_model'],   # alpha*mu / (alpha*mu + 2*epsi)
                        'event_label': int(labels[i]),        # +1 Good / -1 Bad / 0 None
                    })

    if not results:
        return pd.DataFrame()

    return (pd.DataFrame(results)
              .set_index(['Date', 'Ticker'])
              .sort_index())


# ── 8. Test block ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    from pintrade.data.loader import load_ohlcv_data

    print("Loading OHLCV data...")
    df = load_ohlcv_data(['AAPL', 'MSFT', 'GOOG'], '2022-01-01', '2023-12-31')

    print("\nFitting EKOP model (annual)...")
    ekop_df = compute_ekop_factor(df, period='annual')

    print("\n--- EKOP Factor Head ---")
    print(ekop_df.head(12))

    print("\n--- Describe ---")
    print(ekop_df.describe())

    print("\n--- Event Label Distribution ---")
    print(ekop_df['event_label'].value_counts())

    print(f"\nPIN range: {ekop_df['PIN'].min():.4f} ~ {ekop_df['PIN'].max():.4f}")