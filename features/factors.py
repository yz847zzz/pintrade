import sqlite3
from pathlib import Path

import pandas as pd
import numpy as np
import yfinance as yf

# ── Module-level fundamental cache (survives within a Python session) ──────────
_fundamental_cache: dict = {}

# ── IC-informed default composite weights ─────────────────────────────────────
# Updated 2024 — validated on 25 S&P 500 large caps (2023-01-01 → 2024-12-31)
# Only factors with |t-stat| > 2.0 are given non-zero weight.
# Amihud_20D and Volume_Zscore_20D are REVERSAL factors for large-cap growth stocks.
DEFAULT_WEIGHTS = {
    'Momentum_21D':      0,   # ICIR -0.036, t=-0.79  — not significant
    'Momentum_63D':      0,   # ICIR +0.043, t=+0.95  — not significant
    'Momentum_252D':     1,   # ICIR +0.135, t=+2.96  — significant
    'RSI_5D':            0,   # ICIR -0.003, t=-0.07  — not significant
    'Price_Zscore_20D':  0,   # ICIR -0.074, t=-1.61  — not significant
    'Volatility_20D':    1,   # ICIR +0.425, t=+9.32  — strong (vol premium)
    'Volume_Zscore_20D':-1,   # ICIR -0.096, t=-2.10  — reversal: high vol → lower ret
    'Amihud_20D':       -1,   # ICIR -0.342, t=-7.48  — reversal for large-cap growth
    'PE_Ratio':          0,   # ICIR +0.037, t=+0.81  — not significant
    'PB_Ratio':          1,   # ICIR +0.270, t=+5.91  — significant (growth premium)
    'ROE':               0,   # ICIR +0.069, t=+1.52  — not significant
    'ROA':               0,   # ICIR +0.028, t=+0.60  — not significant
    'PIN':              -1,   # ICIR -0.297 (prior study, confirmed negative)
    'event_label':       0,   # ICIR ~0.000 (no signal)
    'News_Sentiment':    1,   # FinBERT: positive news tone → positive return (Tetlock 2007)
    'Filing_Sentiment':  1,   # FinBERT: positive 8-K/10-K MD&A tone → positive return
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


# ── Sentiment factor loader ───────────────────────────────────────────────────

def load_sentiment_factor(
    tickers: list,
    daily_index: pd.DatetimeIndex,
    sentiment_db: str | Path = "data/sentiment.db",
    news_ffill_days: int = 5,
    filing_ffill_days: int = 90,
) -> pd.DataFrame:
    """
    Load FinBERT sentiment scores from sentiment.db and align to daily trading index.

    Two separate signals are returned:
      News_Sentiment    — from yfinance + RSS news (forward-fill 5 days / 1 week)
      Filing_Sentiment  — from 8-K earnings releases + 10-K MD&A (forward-fill 90 days)

    Lookahead bias prevention:
      Scores are shifted forward by 1 trading day — sentiment observed on day T
      is only usable for trading on day T+1 (after market close on T).

    Parameters
    ----------
    tickers          : list of ticker strings to load
    daily_index      : DatetimeIndex of trading days to align to (from OHLCV data)
    sentiment_db     : path to sentiment.db created by the pipeline
    news_ffill_days  : max days to carry forward a news score (default 5)
    filing_ffill_days: max days to carry forward a filing score (default 90)

    Returns
    -------
    DataFrame indexed by (Date, Ticker) with columns: News_Sentiment, Filing_Sentiment
    NaN where no sentiment data exists within the forward-fill window.
    """
    db_path = Path(sentiment_db)
    if not db_path.exists():
        return pd.DataFrame(
            index=pd.MultiIndex.from_arrays([[], []], names=['Date', 'Ticker'])
        )

    # ── Load raw scores from SQLite ───────────────────────────────────────────
    ticker_list = "', '".join(tickers)
    sql = f"""
        SELECT
            ticker,
            date,
            doc_type,
            AVG(compound) AS compound
        FROM sentiment
        WHERE ticker IN ('{ticker_list}')
        GROUP BY ticker, date, doc_type
        ORDER BY ticker, date
    """
    conn = sqlite3.connect(str(db_path))
    raw = pd.read_sql_query(sql, conn)
    conn.close()

    if raw.empty:
        return pd.DataFrame(
            index=pd.MultiIndex.from_arrays([[], []], names=['Date', 'Ticker'])
        )

    raw['date'] = pd.to_datetime(raw['date'])

    # ── Split into news vs filing signals ─────────────────────────────────────
    news_mask    = raw['doc_type'] == 'news'
    filing_mask  = raw['doc_type'].isin(['10-K', '10-Q', '8-K'])

    def _to_daily_wide(df_subset, ffill_limit: int) -> pd.DataFrame:
        """
        Pivot (ticker, date, compound) to wide Date×Ticker, reindex to daily_index,
        forward-fill up to ffill_limit days, shift 1 day for lookahead prevention.
        Mirrors _to_daily() pattern used for fundamental factors.
        """
        if df_subset.empty:
            return pd.DataFrame(index=daily_index, columns=tickers, dtype=float)

        # Aggregate multiple scores on same (ticker, date) → mean
        agg = (
            df_subset.groupby(['date', 'ticker'])['compound']
            .mean()
            .unstack('ticker')          # → Date × Ticker wide table
            .reindex(columns=tickers)
        )
        # Align to full daily trading index, forward-fill, then shift 1 day
        daily = (
            agg
            .reindex(agg.index.union(daily_index))
            .sort_index()
            .ffill(limit=ffill_limit)
            .reindex(daily_index)
            .shift(1)                   # T+1 shift: no lookahead
        )
        return daily

    news_wide    = _to_daily_wide(raw[news_mask],    news_ffill_days)
    filing_wide  = _to_daily_wide(raw[filing_mask],  filing_ffill_days)

    # ── Stack back to (Date, Ticker) MultiIndex ───────────────────────────────
    frames = []
    for ticker in tickers:
        ticker_df = pd.DataFrame({
            'News_Sentiment':   news_wide[ticker]   if ticker in news_wide.columns   else np.nan,
            'Filing_Sentiment': filing_wide[ticker] if ticker in filing_wide.columns else np.nan,
        }, index=daily_index)
        ticker_df.index.name = 'Date'
        ticker_df['Ticker'] = ticker
        frames.append(ticker_df.reset_index().set_index(['Date', 'Ticker']))

    return pd.concat(frames).sort_index()


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


def _sector_neutral_zscore_normalize(factor_df, sector_map: dict):
    """
    Sector-neutral cross-sectional z-score per day.

    Each factor is z-scored WITHIN each GICS sector separately, removing
    common sector-level variation.  Tickers not present in sector_map are
    grouped into an implicit 'Unknown' sector and z-scored among themselves.

    Sectors with only one valid observation for a factor receive NaN for that
    factor on that day (can't compute a meaningful z-score).

    Parameters
    ----------
    factor_df  : (Date, Ticker) MultiIndex DataFrame of raw factor values
    sector_map : dict mapping ticker → sector string

    Returns
    -------
    DataFrame with same shape/index as factor_df, values replaced by
    within-sector z-scores.
    """
    normalized_frames = []

    for date in factor_df.index.get_level_values('Date').unique():
        daily = factor_df.loc[date]
        normed = pd.DataFrame(np.nan, index=daily.index, columns=daily.columns,
                              dtype=float)

        # Build sector → [ticker] groups (one pass per date, not per factor)
        sector_groups: dict[str, list] = {}
        for ticker in daily.index:
            sec = sector_map.get(ticker, 'Unknown')
            sector_groups.setdefault(sec, []).append(ticker)

        for col in daily.columns:
            s = daily[col]
            for sec, sec_tickers in sector_groups.items():
                sec_vals = s.loc[sec_tickers].dropna()
                if len(sec_vals) < 2:
                    # Single valid ticker in sector → can't z-score; leave NaN
                    continue
                mean = sec_vals.mean()
                std  = sec_vals.std()
                if std == 0:
                    normed.loc[sec_tickers, col] = 0.0
                else:
                    normed.loc[sec_tickers, col] = (s.loc[sec_tickers] - mean) / std

        mi = pd.MultiIndex.from_product([[date], daily.index],
                                        names=['Date', 'Ticker'])
        normed.index = mi
        normalized_frames.append(normed)

    if not normalized_frames:
        return pd.DataFrame(
            index=pd.MultiIndex.from_arrays([[], []], names=['Date', 'Ticker']))

    return pd.concat(normalized_frames)


def compute_factors(ohlcv_df: pd.DataFrame,
                    include_pin: bool = True,
                    include_sentiment: bool = False,
                    sentiment_db: str | Path = "data/sentiment.db",
                    sector_map: dict | None = None) -> pd.DataFrame:
    """
    Compute cross-sectional alpha factors from OHLCV data.

    Technical factors (require full history, rows with NaN dropped):
      Momentum_21D/63D/252D, RSI_5D, Price_Zscore_20D,
      Volatility_20D, Volume_Zscore_20D, Amihud_20D

    Fundamental factors (annual, forward-filled; NaN kept for older dates):
      PE_Ratio, PB_Ratio, ROE, ROA

    By default all factors are cross-sectionally z-scored daily across the
    full universe.  When sector_map is provided the z-scoring is done WITHIN
    each GICS sector (sector-neutral normalization), removing common
    sector-level variation before composite scoring.

    Parameters
    ----------
    ohlcv_df           : wide OHLCV DataFrame, (Ticker, Price) MultiIndex columns
    include_pin        : if True, append EKOP PIN + event_label factors
    include_sentiment  : if True, load FinBERT News_Sentiment + Filing_Sentiment
                         from sentiment_db. Requires pipeline to have been run first.
    sentiment_db       : path to sentiment.db produced by data/pipeline/pipeline.py
    sector_map         : optional dict {ticker: sector_string}.  When provided,
                         factors are z-scored within each sector separately
                         (sector-neutral mode) instead of across the full universe.

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

    # Z-score normalise: sector-neutral if sector_map provided, else full cross-section
    if sector_map:
        factor_df = _sector_neutral_zscore_normalize(combined, sector_map)
    else:
        factor_df = _cross_sectional_zscore_normalize(combined)

    if include_pin:
        from pintrade.features.ekop_model import compute_ekop_factor
        pin_df    = compute_ekop_factor(ohlcv_df, period='annual')
        factor_df = factor_df.join(pin_df, how='left')

    if include_sentiment:
        daily_index  = ohlcv_df.index
        ticker_list  = list(ohlcv_df.columns.get_level_values('Ticker').unique())
        sent_df      = load_sentiment_factor(ticker_list, daily_index, sentiment_db)

        # Keep only the rows that survived technical-factor dropna
        sent_df      = sent_df.reindex(factor_df.index)

        # Z-score sentiment columns cross-sectionally and join
        sent_normed  = _cross_sectional_zscore_normalize(sent_df)
        factor_df    = factor_df.join(sent_normed, how='left')

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
