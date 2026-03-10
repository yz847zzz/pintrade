"""
pintrade/features/pin_factor.py

VDJ (Volume-Decomposed) PIN Model
Translated from MATLAB — exact implementation

Model parameters: (alpha, delta, epsi, mu, invpsi)
  alpha  : probability of information event
  delta  : probability of bad news | event
  epsi   : uninformed trader arrival rate
  mu     : informed trader arrival rate
  invpsi : 1/psi (inverse of Inverse Gaussian shape param)

Three scenarios per day:
  No event  : l1 = epsi,      l2 = epsi
  Good news : l1 = epsi+mu,   l2 = epsi
  Bad news  : l1 = epsi,      l2 = epsi+mu

PIN = (Good days + Bad days) / Total days  [empirical frequency]
"""

import numpy as np
from scipy.special import gammaln
from scipy.optimize import minimize
import pandas as pd


# ── 1. DP Log Bessel K (translated exactly from MATLAB BuildLogBesselArray) ──

def build_log_besselk_array(max_trade: int, z: float) -> np.ndarray:
    """
    Exact translation of MATLAB BuildLogBesselArray.

    K*(1) = 1                          → log = 0
    K*(2) = 1 + z                      → log = log(1 + z)
    K*(t) recurrence (t >= 3):
        fac = z^2 / ((2t-3)(2t-5))
        log K*(t) = log K*(t-2) + log_sum_exp(log K*(t-1) - log K*(t-2), log(fac))

    Returns array of length max_trade where arr[t-1] = log K*(t).
    """
    Nk  = max_trade
    arr = np.zeros(Nk)

    if Nk < 1:
        return arr

    arr[0] = 0.0                          # K*(1) = 1

    if Nk >= 2:
        arr[1] = np.log(1.0 + z + 1e-300) # K*(2) = 1 + z

    for t in range(3, Nk + 1):
        fac     = z**2 / ((2.0*t - 3.0) * (2.0*t - 5.0) + 1e-300)
        log_fac = np.log(abs(fac) + 1e-300)
        d       = arr[t-2] - arr[t-3]     # log K*(t-1) - log K*(t-2)
        m       = max(d, log_fac)
        log_sum = m + np.log(np.exp(d - m) + np.exp(log_fac - m))
        arr[t-1] = arr[t-3] + log_sum     # log K*(t) = log K*(t-2) + log_sum

    return arr


def log_besselk_safe(trades: np.ndarray, z: float) -> np.ndarray:
    """
    Lookup log K*(trades[i]) from DP array.
    trades are integers (Bt+St for each day).
    """
    trades  = np.round(trades).astype(int)
    max_idx = int(trades.max())
    arr     = build_log_besselk_array(max_idx, float(z))
    return arr[trades - 1]  # arr[t-1] = log K*(t)


# ── 2. Core VDJ log-likelihood (exact translation of MATLAB log_fPIG2) ────────

def log_fPIG(Bt: np.ndarray, St: np.ndarray,
             l1: float, l2: float, psi: float) -> np.ndarray:
    """
    Exact Python translation of MATLAB log_fPIG_withTable.
    Equation (16) from VDJ paper.
    """
    epsilon = 1e-300
    Bt = np.asarray(Bt, dtype=float)
    St = np.asarray(St, dtype=float)

    # Poisson fallback: psi > 100 (EKOP limit), or counts too large for the
    # Bessel recurrence (numerically unstable for max_trade >> 1 000).
    # Must include -l1 -l2 to match the Bessel path's log_term4 limit.
    if psi > 100 or int((Bt + St).max()) > 1_000:
        return (Bt * np.log(l1 + epsilon) - l1 - gammaln(Bt + 1) +
                St * np.log(l2 + epsilon) - l2 - gammaln(St + 1))

    psi2      = psi ** 2
    sum_l     = l1 + l2
    sqrt_term = np.sqrt(psi2 + 2.0 * sum_l)
    z         = psi * sqrt_term          # scalar

    log_fact_Bt = gammaln(Bt + 1)
    log_fact_St = gammaln(St + 1)
    trades      = (Bt + St).astype(int)

    # Build Bessel lookup table once for this z
    max_trade        = int(trades.max()) + 1
    log_bessel_arr   = build_log_besselk_array(max_trade, float(z))

    # Compute Bessel term per day
    log_bessel_term = np.zeros(len(Bt))
    mask = trades > 0
    if mask.any():
        t_nz  = trades[mask]                         # integer trades
        log_K = log_bessel_arr[t_nz - 1]            # log K*(t)
        # scale term: 0.5*log(pi) + (t-1)*log(z/2) - gammaln(t-0.5)
        log_scale = (0.5 * np.log(np.pi)
                     + (t_nz - 1.0) * np.log(z / 2.0 + epsilon)
                     - gammaln(t_nz - 0.5))
        log_bessel_term[mask] = log_K - log_scale

    log_term1 = Bt * np.log(l1 + epsilon) - log_fact_Bt
    log_term2 = St * np.log(l2 + epsilon) - log_fact_St
    log_term3 = ((Bt + St) / 2.0) * np.log(
                    psi2 / (psi2 + 2.0 * sum_l + epsilon) + epsilon)
    log_term4 = psi2 - psi * sqrt_term

    result = log_term1 + log_term2 + log_term3 + log_term4 + log_bessel_term
    result[np.isinf(result)] = -1e10
    return result


# ── 3. NLL function (exact translation of MATLAB nll_function_vdj_new) ────────

def nll_function(params_scaled: np.ndarray,
                 buys: np.ndarray, sells: np.ndarray,
                 scale: np.ndarray) -> float:
    """
    Negative log-likelihood in scaled parameter space.
    Equation (17): L_VDJ = prod_t [(1-a)*h_none + a*(1-d)*h_good + a*d*h_bad]
    """
    params = params_scaled * scale
    alpha  = np.clip(params[0], 1e-6, 1-1e-6)
    delta  = np.clip(params[1], 1e-6, 1-1e-6)
    epsi   = max(params[2], 1e-10)
    mu     = max(params[3], 1e-10)
    invpsi = max(params[4], 1e-4)
    psi    = 1.0 / invpsi

    log_f_none = log_fPIG(buys, sells, epsi,      epsi,      psi)
    log_f_good = log_fPIG(buys, sells, epsi + mu, epsi,      psi)
    log_f_bad  = log_fPIG(buys, sells, epsi,      epsi + mu, psi)

    safe_log  = lambda x: np.log(max(float(x), 1e-12))
    lw_none   = safe_log(1.0 - alpha)
    lw_good   = safe_log(alpha) + safe_log(1.0 - delta)
    lw_bad    = safe_log(alpha) + safe_log(delta)

    log_components = np.column_stack([
        lw_none + log_f_none,
        lw_good + log_f_good,
        lw_bad  + log_f_bad,
    ])

    m      = log_components.max(axis=1, keepdims=True)
    loglik = m.squeeze() + np.log(np.exp(log_components - m).sum(axis=1))
    loglik[np.isinf(loglik)] = -1e10

    return -np.sum(loglik)


# ── 4. MLE parameter fitting ──────────────────────────────────────────────────

def fit_vdj_mle(buys: np.ndarray, sells: np.ndarray) -> dict:
    """
    Fit VDJ parameters via MLE.
    Normalizes Bt/St by total mean volume — keeps z manageable.
    epsi/mu in normalized space. PIN is scale-invariant.
    """
    buys  = np.asarray(buys,  dtype=float)
    sells = np.asarray(sells, dtype=float)

    # mean of total daily volume (buys + sells)
    mean_volume = (buys + sells).mean()
    if mean_volume < 1e-10:
        return dict(alpha=np.nan, delta=np.nan, epsi=np.nan,
                    mu=np.nan, invpsi=np.nan, psi=np.nan, converged=False)

    # epsi and mu are arrival rates — upper bound = 3 × mean daily volume
    maxInvPsi = 100.0
    scale = np.array([1.0, 1.0, 3.0 * mean_volume, 3.0 * mean_volume, maxInvPsi])

    # Initial guesses in scaled [0,1] space.
    # epsi ≈ 0.45 * mean_volume → scaled = 0.45/3 ≈ 0.15
    # mu   ≈ small fraction of mean_volume (like ekop_model.py)
    initial_param_sets = np.array([
        [0.3, 0.5, 0.150, 0.033, 0.005],
        [0.5, 0.5, 0.133, 0.067, 0.010],
        [0.2, 0.3, 0.117, 0.050, 0.020],
        [0.7, 0.5, 0.100, 0.100, 0.005],
        [0.4, 0.6, 0.150, 0.017, 0.010],
    ])

    bounds = [(1e-6, 1.0-1e-6),   # alpha
              (1e-6, 1.0-1e-6),   # delta
              (1e-6, 1.0),         # epsi scaled  (actual = x * 3*mean_volume)
              (1e-6, 1.0),         # mu   scaled
              (1e-6, 1.0)]         # invpsi scaled

    best_nll    = np.inf
    best_params = None

    for x0_scaled in initial_param_sets:
        try:
            res = minimize(
                nll_function,
                x0_scaled,
                args=(buys, sells, scale),   # raw volumes, no normalization
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 5000, 'ftol': 1e-9}
            )
            if res.fun < best_nll:
                best_nll    = res.fun
                best_params = res.x * scale
        except Exception:
            continue

    if best_params is None:
        return dict(alpha=np.nan, delta=np.nan, epsi=np.nan,
                    mu=np.nan, invpsi=np.nan, psi=np.nan, converged=False)

    alpha, delta, epsi, mu, invpsi = best_params
    invpsi = max(invpsi, 1e-4)
    psi    = 1.0 / invpsi

    return dict(alpha=float(alpha), delta=float(delta),
                epsi=float(epsi),   mu=float(mu),
                invpsi=float(invpsi), psi=float(psi),
                converged=True)


# ── 5. Daily Bayesian event classification ────────────────────────────────────

def classify_days(buys: np.ndarray, sells: np.ndarray,
                  params: dict) -> np.ndarray:
    """
    Posterior Bayesian classification of each day.
    Matches MATLAB: scores = exp(logf) * prior, label = argmax.
    Returns: +1 (Good), -1 (Bad), 0 (No Event)
    """
    alpha = params['alpha']
    delta = params['delta']
    epsi  = params['epsi']
    mu    = params['mu']
    psi   = params['psi']

    logf = np.column_stack([
        log_fPIG(buys, sells, epsi + mu, epsi,      psi),  # Good (+1)
        log_fPIG(buys, sells, epsi,      epsi + mu, psi),  # Bad  (-1)
        log_fPIG(buys, sells, epsi,      epsi,      psi),  # None ( 0)
    ])

    prior  = np.array([alpha*(1-delta), alpha*delta, 1.0-alpha])
    scores = np.exp(logf) * prior
    x_hat  = np.argmax(scores, axis=1)

    return np.array([+1, -1, 0])[x_hat]


# ── 6. Lee-Ready buy/sell volume estimator ────────────────────────────────────

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


# ── 7. Wide → Long format converter ──────────────────────────────────────────

def _wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.columns, pd.MultiIndex):
        return df
    if df.columns.names[0] == 'Ticker':
        df_long = df.stack(level=0, future_stack=True)
    else:
        df_long = df.stack(level=1, future_stack=True)
    df_long.index.names = ['Date', 'Ticker']
    return df_long


# ── 8. Main factor computation ────────────────────────────────────────────────

def compute_vdj_factor(
    df: pd.DataFrame,
    period: str = 'annual'
) -> pd.DataFrame:
    """
    For each ticker and each calendar window:
      1. Estimate buy/sell volume via Lee-Ready
      2. Fit VDJ MLE ONCE per window
      3. Classify every day: Good (+1) / Bad (-1) / No Event (0)
      4. PIN = (Good + Bad days) / Total days

    Returns DataFrame (Date, Ticker) with:
      PIN_annual       : float ∈ (0,1)
      VDJ_label_annual : int ∈ {+1,-1,0}
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

                params = fit_vdj_mle(buys, sells)
                if not params['converged']:
                    continue

                labels    = classify_days(buys, sells, params)

                # PIN = alpha*mu / (alpha*mu + 2*epsi) — model-implied
                pin_value = float(np.clip(
                    params['alpha'] * params['mu'] /
                    (params['alpha'] * params['mu'] + 2 * params['epsi'] + 1e-300),
                    0.0, 1.0))

                print(f"  {ticker} {key}: "
                      f"alpha={params['alpha']:.3f} delta={params['delta']:.3f} "
                      f"mu={params['mu']:.0f} epsi={params['epsi']:.0f} "
                      f"PIN={pin_value:.3f} "
                      f"Good={int((labels==1).sum())} "
                      f"Bad={int((labels==-1).sum())} "
                      f"None={int((labels==0).sum())}")

                for i, date in enumerate(group.index):
                    results.append({
                        'Date':        date,
                        'Ticker':      ticker,
                        'PIN':         pin_value,      # alpha*mu / (alpha*mu + 2*epsi)
                        'event_label': int(labels[i]), # +1 Good / -1 Bad / 0 None
                    })

    if not results:
        return pd.DataFrame()

    return (pd.DataFrame(results)
              .set_index(['Date', 'Ticker'])
              .sort_index())


# ── 9. Test block ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    from pintrade.data.loader import load_ohlcv_data

    print("Loading OHLCV data...")
    df = load_ohlcv_data(['AAPL', 'MSFT', 'GOOG'], '2022-01-01', '2023-12-31')

    print("\nFitting VDJ model (annual)...")
    vdj_df = compute_vdj_factor(df, period='annual')

    print("\n--- VDJ Factor Head ---")
    print(vdj_df.head(12))

    print("\n--- Describe ---")
    print(vdj_df.describe())

    print("\n--- Label Distribution ---")
    print(vdj_df['event_label'].value_counts())

    print(f"\nPIN range: {vdj_df['PIN'].min():.4f} ~ {vdj_df['PIN'].max():.4f}")