import pandas as pd
import numpy as np
import yfinance as yf

# ── Module-level fundamental cache (survives within a Python session) ──────────
_fundamental_cache: dict = {}

# ── IC-informed default composite weights ─────────────────────────────────────
DEFAULT_WEIGHTS = {
    'Momentum_21D':      1,   # ICIR +0.077
    'Momentum_63D':      1,   # ICIR +0.140
    'Momentum_252D':     1,   # ICIR +0.182
    'RSI_5D':            1,   # ICIR +0.095
    'Price_Zscore_20D':  1,   # ICIR +0.072
    'Volatility_20D':    1,   # ICIR +0.103
    'Volume_Zscore_20D': 0,   # ICIR -0.057 (t=-1.79, excluded)
    'Amihud_20D':       -1,   # ICIR -0.087 (liquid growth stocks dominated 2019-2023)
    'PE_Ratio':          0,   # ICIR +0.082 (t=1.45, not significant after bias fix)
    'PB_Ratio':          1,   # ICIR +0.165 (t=2.28, marginal growth premium)
    'ROE':              -1,   # ICIR -0.308 (t=-4.25, significant — growth era)
    'ROA':              -1,   # ICIR -0.179 (t=-2.48, significant — growth era)
    'PIN':              -1,   # ICIR -0.297 (confirmed negative)
    'event_label':       0,   # ICIR ~0.000 (no signal)
}

# Technical factors that must be non-NaN for a row to be kept
_TECHNICAL_COLS = [
    'Momentum_21D', 'Momentum_63D', 'Momentum_252D',
    'RSI_5D', 'Price_Zscore_20D', 'Volatility_20D',
    'Volume_Zscore_20D', 'Amihud_20D',
]


# ── Fundamental data helpers ──────────────────────────────────────────────────

def _fetch_fundamentals(ticker: str) -> dict:
    """
    Fetch annual income statement + balance sheet for one ticker via yfinance.
    Returns dict of pd.Series with fiscal-year-end dates as index:
      EPS  (annual Basic EPS from income statement — no .info snapshot)
      BVPS (book value per share = equity / shares, from balance sheet)
      ROE  (net income / equity)
      ROA  (net income / total assets)

    Shares are taken from the annual balance sheet ('Ordinary Shares Number')
    rather than .info (which is a current-date snapshot and causes lookahead).
    The 60-day reporting lag is applied in _to_daily(), so a fiscal year ending
    on date F is not usable until F + 60 days.
    Cached per ticker to avoid repeated API calls.
    """
    if ticker in _fundamental_cache:
        return _fundamental_cache[ticker]

    result = {}
    try:
        t   = yf.Ticker(ticker)
        fin = t.financials       # annual income statement; cols = fiscal-year-end dates
        bs  = t.balance_sheet    # annual balance sheet;    cols = fiscal-year-end dates

        def _get_row(df, labels):
            """Return first matching row as a sorted date-indexed Series, or None."""
            if df is None or df.empty:
                return None
            for lbl in labels:
                if lbl in df.index:
                    s = df.loc[lbl].dropna()
                    if len(s) > 0:
                        return s.sort_index()
            return None

        net_income = _get_row(fin, ['Net Income', 'NetIncome'])
        equity     = _get_row(bs,  ['Stockholders Equity', 'Total Stockholder Equity',
                                     'Common Stock Equity',
                                     'Total Equity Gross Minority Interest'])
        assets     = _get_row(bs,  ['Total Assets'])
        shares     = _get_row(bs,  ['Ordinary Shares Number', 'Share Issued'])
        eps_direct = _get_row(fin, ['Basic EPS', 'Diluted EPS'])

        # EPS: prefer income-statement figure (no share-count ambiguity)
        if eps_direct is not None and len(eps_direct) > 0:
            result['EPS'] = eps_direct
        elif net_income is not None and shares is not None:
            common = net_income.index.intersection(shares.index)
            if len(common) > 0:
                result['EPS'] = (net_income.loc[common] /
                                 shares.loc[common].replace(0, np.nan))

        # BVPS = equity / shares
        if equity is not None and shares is not None:
            common = equity.index.intersection(shares.index)
            if len(common) > 0:
                result['BVPS'] = (equity.loc[common] /
                                  shares.loc[common].replace(0, np.nan))

        # ROE = net income / equity
        if net_income is not None and equity is not None:
            common = net_income.index.intersection(equity.index)
            if len(common) > 0:
                result['ROE'] = (net_income.loc[common] /
                                 equity.loc[common].replace(0, np.nan))

        # ROA = net income / total assets
        if net_income is not None and assets is not None:
            common = net_income.index.intersection(assets.index)
            if len(common) > 0:
                result['ROA'] = (net_income.loc[common] /
                                 assets.loc[common].replace(0, np.nan))

    except Exception:
        pass

    _fundamental_cache[ticker] = result
    return result


def _to_daily(quarterly_series: pd.Series,
              daily_index: pd.DatetimeIndex,
              lag_days: int = 60) -> pd.Series:
    """
    Forward-fill a quarterly series (index = fiscal quarter-end dates)
    onto a daily DatetimeIndex, with a point-in-time reporting lag.

    A report for quarter ending on date Q is assumed available on Q + lag_days
    (default 60 days ≈ 2 months after quarter close, per SEC filing deadlines).
    On date t, carries the value from the most recent report available at t.
    """
    s = quarterly_series.copy()
    s.index = pd.DatetimeIndex(s.index) + pd.Timedelta(days=lag_days)
    s = s.sort_index()
    combined = s.reindex(s.index.union(daily_index)).sort_index().ffill()
    return combined.reindex(daily_index)


# ── Factor computation ────────────────────────────────────────────────────────

def _calculate_rsi(series, window):
    delta = series.diff()
    gain  = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs    = gain / loss
    return 100 - (100 / (1 + rs))


def _cross_sectional_zscore_normalize(factor_df):
    """
    Cross-sectional z-score per day.  NaN values in any column are preserved.
    """
    normalized_frames = []
    for date in factor_df.index.get_level_values('Date').unique():
        daily = factor_df.loc[date]
        normed = pd.DataFrame(index=daily.index, columns=daily.columns)
        for col in daily.columns:
            s = daily[col]
            if s.isnull().all():
                normed[col] = np.nan
            else:
                std = s.std()
                normed[col] = 0.0 if std == 0 else (s - s.mean()) / std

        mi = pd.MultiIndex.from_product([[date], daily.index],
                                        names=['Date', 'Ticker'])
        normed.index = mi
        normalized_frames.append(normed)

    if not normalized_frames:
        return pd.DataFrame(
            index=pd.MultiIndex.from_arrays([[], []], names=['Date', 'Ticker']))

    return pd.concat(normalized_frames)


def compute_factors(ohlcv_df: pd.DataFrame,
                    include_pin: bool = True) -> pd.DataFrame:
    """
    Compute cross-sectional alpha factors from OHLCV data.

    Technical factors (require full history, rows with NaN dropped):
      Momentum_21D/63D/252D, RSI_5D, Price_Zscore_20D,
      Volatility_20D, Volume_Zscore_20D, Amihud_20D

    Fundamental factors (annual, forward-filled; NaN kept for older dates):
      PE_Ratio, PB_Ratio, ROE, ROA

    All factors are cross-sectionally z-scored daily.

    Parameters
    ----------
    ohlcv_df   : wide OHLCV DataFrame, (Ticker, Price) MultiIndex columns
    include_pin: if True, append EKOP PIN + event_label factors

    Returns
    -------
    DataFrame indexed by (Date, Ticker)
    """
    all_factors = []
    tickers = ohlcv_df.columns.get_level_values(0).unique()

    print(f"  Fetching fundamentals for {len(tickers)} tickers...")
    for ticker in tickers:
        _fetch_fundamentals(ticker)   # pre-warm cache

    for ticker in tickers:
        df   = ohlcv_df.xs(ticker, level='Ticker', axis=1)
        rows = pd.DataFrame(index=df.index)

        # ── Momentum ──────────────────────────────────────────────────────────
        prev = df['Close'].shift(1)
        for days in [21, 63, 252]:
            rows[f'Momentum_{days}D'] = prev / df['Close'].shift(days) - 1

        # ── Mean-reversion ────────────────────────────────────────────────────
        rows['RSI_5D']           = _calculate_rsi(df['Close'], 5)
        roll20                   = df['Close'].rolling(20)
        rows['Price_Zscore_20D'] = (df['Close'] - roll20.mean()) / roll20.std()

        # ── Volatility ────────────────────────────────────────────────────────
        ret = df['Close'].pct_change()
        rows['Volatility_20D']   = ret.rolling(20).std()

        # ── Volume z-score ────────────────────────────────────────────────────
        rvol = df['Volume'].rolling(20)
        rows['Volume_Zscore_20D'] = (df['Volume'] - rvol.mean()) / rvol.std()

        # ── Amihud illiquidity ────────────────────────────────────────────────
        dollar_vol             = df['Close'] * df['Volume']
        rows['Amihud_20D']     = (
            ret.abs() / dollar_vol.replace(0, np.nan)
        ).rolling(20).mean() * 1e6   # scale to readable range

        # ── Fundamental factors (forward-filled annual data) ──────────────────
        fund  = _fundamental_cache.get(ticker, {})
        dates = df.index

        if 'EPS' in fund and len(fund['EPS']) > 0:
            eps = _to_daily(fund['EPS'], dates)
            rows['PE_Ratio'] = df['Close'] / eps.where(eps > 0)

        if 'BVPS' in fund and len(fund['BVPS']) > 0:
            bvps = _to_daily(fund['BVPS'], dates)
            rows['PB_Ratio'] = df['Close'] / bvps.where(bvps > 0)

        if 'ROE' in fund and len(fund['ROE']) > 0:
            rows['ROE'] = _to_daily(fund['ROE'], dates)

        if 'ROA' in fund and len(fund['ROA']) > 0:
            rows['ROA'] = _to_daily(fund['ROA'], dates)

        all_factors.append(rows.assign(Ticker=ticker))

    if not all_factors:
        return pd.DataFrame(
            index=pd.MultiIndex.from_arrays([[], []], names=['Date', 'Ticker']))

    combined = pd.concat(all_factors)
    combined = combined.set_index([combined.index, 'Ticker']).sort_index()
    combined.index.names = ['Date', 'Ticker']

    # Drop rows missing any technical factor (fundamentals may remain NaN)
    combined = combined.dropna(subset=_TECHNICAL_COLS)

    # Cross-sectional z-score normalise all columns
    factor_df = _cross_sectional_zscore_normalize(combined)

    if include_pin:
        from pintrade.features.ekop_model import compute_ekop_factor
        pin_df    = compute_ekop_factor(ohlcv_df, period='annual')
        factor_df = factor_df.join(pin_df, how='left')

    return factor_df


def get_composite_score(factor_df: pd.DataFrame,
                        weights: dict = None) -> pd.Series:
    """
    Weighted composite alpha score.

    Default weights are IC-informed (see DEFAULT_WEIGHTS):
      - Momentum / quality / illiquidity factors: +1
      - PIN: -1 (ICIR = -0.21)
      - Volume_Zscore_20D, event_label: 0 (excluded)
      - Value factors (PE, PB): -1 (low ratio → high return)

    Factors missing from factor_df are silently skipped.
    NaN values in a factor contribute 0 to that row's score.

    Parameters
    ----------
    factor_df : (Date, Ticker) MultiIndex DataFrame of z-scored factors
    weights   : optional dict overriding DEFAULT_WEIGHTS

    Returns
    -------
    Series indexed by (Date, Ticker)
    """
    if factor_df.empty:
        return pd.Series(
            dtype=float,
            index=pd.MultiIndex.from_arrays([[], []], names=['Date', 'Ticker']))

    w = weights if weights is not None else DEFAULT_WEIGHTS

    # Build weighted sum — skip zero-weight and absent factors
    active = {f: wt for f, wt in w.items() if wt != 0 and f in factor_df.columns}

    if not active:
        # Fallback: equal weight all present columns
        return factor_df.mean(axis=1)

    weighted = pd.DataFrame(
        {f: factor_df[f].fillna(0) * wt for f, wt in active.items()},
        index=factor_df.index,
    )
    return weighted.sum(axis=1)
