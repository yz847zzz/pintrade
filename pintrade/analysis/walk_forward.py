"""
Walk-forward backtest with per-fold factor selection and Long/Short portfolios.

Protocol (strict no-lookahead):
  For each annual fold —
    Training window (1yr IS):
      1. Compute factor IC vs 21-day forward returns on IS prices
      2. Select factors where |t-stat| > t_threshold
      3. Assign weight = sign(ICIR) for selected, 0 for the rest
      4. Run IS backtest (L/S) → record IS Sharpe

    Test window (1yr OOS):
      5. Apply IS-fitted weights to OOS factor signals
      6. Run OOS backtest → track Long, Short, L/S separately
      7. Record OOS Sharpe / Ann Return / Max DD / Market Beta for each leg

Annual folds (train_years=1, test_years=1, rolling by 1 yr):
    Fold 1 : Train 2019  →  OOS 2020
    Fold 2 : Train 2020  →  OOS 2021
    Fold 3 : Train 2021  →  OOS 2022   ← bear market: did short rescue it?
    Fold 4 : Train 2022  →  OOS 2023
    Fold 5 : Train 2023  →  OOS 2024

Cross-sectional z-scoring inside compute_factors() is per-day → no lookahead.
Factors are computed once on the full dataset for efficiency.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pandas.tseries.offsets import DateOffset

from pintrade.data.loader import load_ohlcv_data
from pintrade.features.factors import compute_factors, get_composite_score
from pintrade.backtest.engine import run_backtest_ls
from pintrade.backtest.regime import compute_regime_multiplier, compute_regime_detail
from pintrade.analysis.ic_analysis import compute_ic, compute_ic_summary


# ── Date window generation ────────────────────────────────────────────────────

def _annual_folds(
    start: pd.Timestamp,
    end: pd.Timestamp,
    train_years: int = 1,
    test_years: int = 1,
) -> list[tuple[pd.Timestamp, ...]]:
    """
    Non-overlapping (train_start, train_end, test_start, test_end) tuples,
    rolling forward by test_years each iteration. train_end < test_start always.
    """
    folds = []
    test_start = start + DateOffset(years=train_years)
    while True:
        train_start = test_start - DateOffset(years=train_years)
        train_end   = test_start - pd.Timedelta(days=1)
        test_end    = test_start + DateOffset(years=test_years) - pd.Timedelta(days=1)
        if test_end > end:
            break
        folds.append((
            pd.Timestamp(train_start),
            pd.Timestamp(train_end),
            pd.Timestamp(test_start),
            pd.Timestamp(test_end),
        ))
        test_start = test_start + DateOffset(years=test_years)
    return folds


# ── Factor selection ──────────────────────────────────────────────────────────

def _select_weights_from_ic(
    ic_df: pd.DataFrame,
    t_threshold: float = 2.0,
) -> tuple[dict, list[str], pd.DataFrame]:
    """
    Derive per-fold factor weights from IS IC.

    Rules:
      |t-stat| >= t_threshold  →  weight = sign(ICIR)  (+1 or -1)
      |t-stat| <  t_threshold  →  weight = 0            (excluded)
      NaN                      →  weight = 0

    Returns: weights dict, list of selected factor names, full IC summary.
    """
    ic_sum = compute_ic_summary(ic_df)
    weights: dict   = {}
    selected: list  = []

    for factor in ic_sum.index:
        t    = ic_sum.loc[factor, 't-stat']
        icir = ic_sum.loc[factor, 'ICIR']
        if pd.isna(t) or abs(t) < t_threshold:
            weights[factor] = 0
        elif icir >= 0:
            weights[factor] = 1
            selected.append(factor)
        else:
            weights[factor] = -1
            selected.append(factor)

    return weights, selected, ic_sum


# ── Quiet L/S backtest (suppresses per-fold PNG saves) ───────────────────────

def _run_ls_quiet(signals, prices, top_n, rebalance,
                  regime_multiplier: pd.Series | None = None):
    """run_backtest_ls without writing any PNG files."""
    with patch('matplotlib.pyplot.savefig'):
        result = run_backtest_ls(signals, prices, top_n=top_n,
                                 rebalance=rebalance,
                                 regime_multiplier=regime_multiplier)
    plt.close('all')
    return result


# ── Main walk-forward ─────────────────────────────────────────────────────────

def run_walk_forward(
    tickers: list[str],
    start: str,
    end: str,
    train_years: int = 1,
    test_years: int = 1,
    top_n: int = 5,
    rebalance: str = 'monthly',
    include_pin: bool = False,
    t_threshold: float = 2.0,
    momentum_buffer_days: int = 300,
    include_sentiment: bool = False,
    sentiment_db: str | None = None,
    use_regime: bool = False,
    vix_threshold: float = 25.0,
    sector_map: dict | None = None,
) -> dict:
    """
    Rolling walk-forward backtest tracking Long, Short, and L/S portfolios.

    Per-fold process:
      IS window  → IC → select factors → IS L/S backtest
      OOS window → apply IS weights → OOS Long / Short / L/S backtest

    Parameters
    ----------
    tickers               : universe of stock tickers
    start                 : first training-window start (ISO date)
    end                   : last test-window end (ISO date)
    train_years           : IS window length in years
    test_years            : OOS window length (= rolling step) in years
    top_n                 : long positions = short positions = top_n
    rebalance             : 'monthly' | 'weekly'
    include_pin           : compute EKOP PIN factor (slow)
    t_threshold           : |t-stat| cutoff for factor selection
    momentum_buffer_days  : extra history before first fold for Momentum_252D
    include_sentiment     : include FinBERT sentiment factors
    sentiment_db          : path to sentiment.db

    Returns
    -------
    dict:
      'windows'      : per-fold DataFrame with IS/OOS metrics for all 3 legs
      'summary'      : aggregate statistics (mean/overall OOS Sharpe etc.)
      'oos_equity'   : dict {'long', 'short', 'ls'} → stitched OOS equity Series
    """
    start_ts = pd.Timestamp(start)
    end_ts   = pd.Timestamp(end)

    folds = _annual_folds(start_ts, end_ts, train_years, test_years)
    if not folds:
        raise ValueError(
            f"Date range {start}→{end} too short for "
            f"train={train_years}yr / test={test_years}yr windows."
        )

    print(f"Walk-forward: {len(folds)} folds | "
          f"train={train_years}yr  test={test_years}yr  N={top_n}  |t|>{t_threshold}")

    # ── Load full OHLCV once ──────────────────────────────────────────────────
    load_start = folds[0][0] - pd.Timedelta(days=momentum_buffer_days)
    print(f"Loading OHLCV  {load_start.date()} → {end_ts.date()}...")
    full_ohlcv = load_ohlcv_data(tickers, str(load_start.date()), end)

    # ── Load regime multiplier (once for full period) ─────────────────────────
    full_regime: pd.Series | None = None
    if use_regime:
        print(f"Computing regime indicators (VIX>{vix_threshold}, 200MA, 12M mom)...")
        full_regime = compute_regime_multiplier(
            start=str(load_start.date()),
            end=end,
            vix_threshold=vix_threshold,
        )
        pct_short = (full_regime > 0).mean() * 100
        pct_full  = (full_regime == 1.0).mean() * 100
        pct_half  = (full_regime == 0.5).mean() * 100
        print(f"  Regime: {pct_short:.1f}% days with short active  "
              f"({pct_full:.1f}% full, {pct_half:.1f}% half)")

    # ── Compute factors once ──────────────────────────────────────────────────
    kw: dict = dict(include_pin=include_pin)
    if include_sentiment and sentiment_db:
        kw['include_sentiment'] = True
        kw['sentiment_db']      = sentiment_db
    if sector_map:
        kw['sector_map'] = sector_map
    mode = 'sector-neutral' if sector_map else 'cross-sectional'
    print(f"Computing factors (pin={include_pin}, sentiment={include_sentiment}, zscore={mode})...")
    full_factors = compute_factors(full_ohlcv, **kw)

    # ── Iterate folds ─────────────────────────────────────────────────────────
    fold_rows:  list[dict]    = []
    oos_pieces: dict[str, list[pd.Series]] = {'long': [], 'short': [], 'ls': []}

    for i, (train_start, train_end, test_start, test_end) in enumerate(folds):
        fold_num = i + 1
        print(f"\n{'─'*66}")
        print(f"  Fold {fold_num}: IS {train_start.date()} → {train_end.date()} | "
              f"OOS {test_start.date()} → {test_end.date()}")

        # Date masks
        is_dates  = full_ohlcv.index[
            (full_ohlcv.index >= train_start) & (full_ohlcv.index <= train_end)]
        oos_dates = full_ohlcv.index[
            (full_ohlcv.index >= test_start)  & (full_ohlcv.index <= test_end)]

        if len(is_dates) < 30 or len(oos_dates) < 10:
            print(f"  Fold {fold_num}: skipped (IS={len(is_dates)}, OOS={len(oos_dates)} trading days)")
            continue

        is_prices  = full_ohlcv.loc[is_dates]
        oos_prices = full_ohlcv.loc[oos_dates]

        is_mask  = full_factors.index.get_level_values('Date').isin(is_dates)
        oos_mask = full_factors.index.get_level_values('Date').isin(oos_dates)
        is_factors  = full_factors[is_mask]
        oos_factors = full_factors[oos_mask]

        # ── IS IC → weight selection ──────────────────────────────────────────
        # forward_days=21 at train boundary: last ~21 IS days look slightly into
        # test period for IC estimation only — no OOS price data used in signals.
        ic_df = compute_ic(is_factors, full_ohlcv, forward_days=21)
        weights, selected, _ = _select_weights_from_ic(ic_df, t_threshold)

        if not selected:
            print(f"  Fold {fold_num}: no factors passed |t|>{t_threshold} — skipped")
            continue

        selected_str = ', '.join(
            f"{f}({'+'if weights[f]>0 else'-'})" for f in selected
        )

        # ── Slice regime to IS / OOS windows ─────────────────────────────────
        is_regime  = None
        oos_regime = None
        if full_regime is not None:
            is_regime  = full_regime.reindex(is_dates,  method='ffill').dropna()
            oos_regime = full_regime.reindex(oos_dates, method='ffill').dropna()
            oos_pct_full = (oos_regime == 1.0).mean() * 100
            oos_pct_half = (oos_regime == 0.5).mean() * 100
            oos_pct_off  = (oos_regime == 0.0).mean() * 100
            print(f"  Regime OOS: {oos_pct_full:.0f}% full-short  "
                  f"{oos_pct_half:.0f}% half-short  "
                  f"{oos_pct_off:.0f}% long-only")

        # ── IS backtest (L/S, for IS Sharpe reference) ───────────────────────
        is_signals = get_composite_score(is_factors, weights=weights)
        is_ls_res  = _run_ls_quiet(is_signals, is_prices, top_n, rebalance,
                                   regime_multiplier=is_regime)
        is_sharpe  = is_ls_res['ls']['sharpe_ratio']

        # ── OOS backtest ──────────────────────────────────────────────────────
        oos_signals = get_composite_score(oos_factors, weights=weights)
        if oos_signals.empty:
            print(f"  Fold {fold_num}: no OOS signals — skipped")
            continue

        oos = _run_ls_quiet(oos_signals, oos_prices, top_n, rebalance,
                            regime_multiplier=oos_regime)

        # Stitch equity pieces (normalise each fold to start at 1.0)
        for leg in ('long', 'short', 'ls'):
            eq = oos[leg]['equity_curve']
            if not eq.empty and eq.iloc[0] != 0:
                oos_pieces[leg].append(eq / eq.iloc[0])

        # Print fold summary
        def _fmt(leg):
            r = oos[leg]
            return (f"SR={r['sharpe_ratio']:+.3f}  "
                    f"AR={r['annualized_return']*100:+.1f}%  "
                    f"MDD={r['max_drawdown']*100:.1f}%  "
                    f"β={r['beta']:.2f}")

        print(f"  IS  L/S Sharpe = {is_sharpe:+.3f}")
        print(f"  OOS Long : {_fmt('long')}")
        print(f"  OOS Short: {_fmt('short')}")
        print(f"  OOS L/S  : {_fmt('ls')}")
        print(f"  Selected ({len(selected)}): {selected_str}")

        oos_short_pct = (
            (oos_regime > 0).mean() * 100
            if oos_regime is not None else 100.0
        )
        fold_rows.append({
            'fold':            fold_num,
            'train_start':     train_start.date(),
            'train_end':       train_end.date(),
            'test_start':      test_start.date(),
            'test_end':        test_end.date(),
            'n_selected':      len(selected),
            'selected':        selected_str,
            'short_active_%':  round(oos_short_pct, 1),
            # IS reference
            'is_ls_sharpe':    is_sharpe,
            # OOS — Long leg
            'long_sharpe':  oos['long']['sharpe_ratio'],
            'long_annret':  oos['long']['annualized_return'],
            'long_mdd':     oos['long']['max_drawdown'],
            'long_beta':    oos['long']['beta'],
            # OOS — Short leg
            'short_sharpe': oos['short']['sharpe_ratio'],
            'short_annret': oos['short']['annualized_return'],
            'short_mdd':    oos['short']['max_drawdown'],
            'short_beta':   oos['short']['beta'],
            # OOS — L/S combined
            'ls_sharpe':    oos['ls']['sharpe_ratio'],
            'ls_annret':    oos['ls']['annualized_return'],
            'ls_mdd':       oos['ls']['max_drawdown'],
            'ls_beta':      oos['ls']['beta'],
        })

    if not fold_rows:
        print("No valid folds completed.")
        return {'windows': pd.DataFrame(), 'summary': {}, 'oos_equity': {}}

    results_df = pd.DataFrame(fold_rows).set_index('fold')

    # Stitch OOS equity for each leg
    oos_equity = {leg: _stitch_equity(oos_pieces[leg]) for leg in ('long', 'short', 'ls')}

    summary = {
        'n_folds':                len(results_df),
        'overall_oos_ls_sharpe':  _sharpe_from_equity(oos_equity['ls']),
        'mean_oos_ls_sharpe':     results_df['ls_sharpe'].mean(),
        'mean_oos_long_sharpe':   results_df['long_sharpe'].mean(),
        'mean_oos_short_sharpe':  results_df['short_sharpe'].mean(),
        'mean_is_ls_sharpe':      results_df['is_ls_sharpe'].mean(),
        'pct_positive_oos_ls':    (results_df['ls_sharpe'] > 0).mean(),
        'mean_oos_ls_ann_return': results_df['ls_annret'].mean(),
        'mean_oos_ls_max_dd':     results_df['ls_mdd'].mean(),
        'mean_oos_ls_beta':       results_df['ls_beta'].mean(),
    }

    return {
        'windows':   results_df,
        'summary':   summary,
        'oos_equity': oos_equity,
    }


# ── Equity stitching ──────────────────────────────────────────────────────────

def _stitch_equity(pieces: list[pd.Series]) -> pd.Series:
    """Chain OOS equity pieces; each piece is already normalised to start at 1."""
    if not pieces:
        return pd.Series(dtype=float)
    stitched = pieces[0].copy()
    for piece in pieces[1:]:
        scale = stitched.iloc[-1]
        stitched = pd.concat([stitched, piece * scale])
    return stitched


def _sharpe_from_equity(equity: pd.Series) -> float:
    if equity.empty or len(equity) < 2:
        return float('nan')
    rets = equity.pct_change().dropna()
    return float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0.0


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_walk_forward(results: dict, save_path: str = 'walk_forward.png'):
    """
    Five-panel walk-forward diagnostic:
    1. IS L/S Sharpe vs OOS L/S Sharpe (grouped bars)
    2. OOS Sharpe for Long / Short / L/S per fold (grouped bars)
    3. Factor selection heatmap (fold × factor, colour = weight sign)
    4. OOS annualised return for all 3 legs per fold
    5. Stitched OOS equity curves: Long / Short / L/S on one chart
    """
    df         = results['windows']
    oos_equity = results.get('oos_equity', {})

    if df.empty:
        print("No results to plot.")
        return

    n  = len(df)
    fi = list(df.index)
    x  = np.arange(n)
    w  = 0.25

    fig = plt.figure(figsize=(max(12, n * 2.5), 24))
    gs  = gridspec.GridSpec(5, 1, hspace=0.5)
    axes = [fig.add_subplot(gs[i]) for i in range(5)]

    # ── Panel 1: IS vs OOS L/S Sharpe ────────────────────────────────────────
    ax = axes[0]
    ax.bar(x - w/2, df['is_ls_sharpe'],  w, label='IS L/S Sharpe',
           color='steelblue', alpha=0.85)
    ax.bar(x + w/2, df['ls_sharpe'], w, label='OOS L/S Sharpe',
           color=['forestgreen' if v >= 0 else 'salmon' for v in df['ls_sharpe']],
           alpha=0.85)
    ax.axhline(0, color='black', lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([f"F{i}" for i in fi])
    ax.set_title('IS vs OOS L/S Sharpe per Fold')
    ax.set_ylabel('Sharpe')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.4)

    # ── Panel 2: OOS Sharpe for Long / Short / L/S ───────────────────────────
    ax = axes[1]
    ax.bar(x - w,   df['long_sharpe'],  w, label='Long',  color='steelblue',  alpha=0.8)
    ax.bar(x,       df['short_sharpe'], w, label='Short', color='darkorange', alpha=0.8)
    ax.bar(x + w,   df['ls_sharpe'],    w, label='L/S',   color='forestgreen',alpha=0.8)
    ax.axhline(0, color='black', lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([f"F{i}" for i in fi])
    ax.set_title('OOS Sharpe: Long / Short / L/S per Fold')
    ax.set_ylabel('Sharpe')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.4)

    # ── Panel 3: Factor selection heatmap ─────────────────────────────────────
    ax = axes[2]
    all_factors: set[str] = set()
    fold_weights: list[dict] = []
    for _, row in df.iterrows():
        d: dict = {}
        if row['selected']:
            for part in row['selected'].split(', '):
                part = part.strip()
                if part.endswith('(+)'):
                    d[part[:-3]] = 1
                elif part.endswith('(-)'):
                    d[part[:-3]] = -1
        fold_weights.append(d)
        all_factors.update(d.keys())

    factor_list = sorted(all_factors)
    if factor_list:
        mat = np.array([[fw.get(f, 0) for f in factor_list]
                        for fw in fold_weights], dtype=float)
        im = ax.imshow(mat.T, aspect='auto', cmap='RdYlGn', vmin=-1, vmax=1,
                       extent=[-0.5, n - 0.5, len(factor_list) - 0.5, -0.5])
        ax.set_xticks(range(n))
        ax.set_xticklabels([f"F{i}" for i in fi])
        ax.set_yticks(range(len(factor_list)))
        ax.set_yticklabels(factor_list, fontsize=8)
        ax.set_title('Factor Selection Heatmap  (green=+1  red=−1  white=excluded)')
        plt.colorbar(im, ax=ax, fraction=0.015, pad=0.02, label='Weight')
    else:
        ax.text(0.5, 0.5, 'No factors selected', ha='center', va='center',
                transform=ax.transAxes)

    # ── Panel 4: OOS Ann Return for all 3 legs ───────────────────────────────
    ax = axes[3]
    ax.bar(x - w,   df['long_annret']  * 100, w, label='Long',  color='steelblue',   alpha=0.8)
    ax.bar(x,       df['short_annret'] * 100, w, label='Short', color='darkorange',  alpha=0.8)
    ax.bar(x + w,   df['ls_annret']    * 100, w, label='L/S',   color='forestgreen', alpha=0.8)
    ax.axhline(0, color='black', lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([f"F{i}" for i in fi])
    ax.set_title('OOS Annualised Return (%): Long / Short / L/S per Fold')
    ax.set_ylabel('Ann Return (%)')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.4)

    # ── Panel 5: Stitched OOS equity curves ───────────────────────────────────
    ax = axes[4]
    colours = {'long': 'steelblue', 'short': 'darkorange', 'ls': 'forestgreen'}
    styles  = {'long': '-',         'short': '--',          'ls': '-'}
    labels  = {'long': 'Long-only', 'short': 'Short-only',  'ls': 'L/S Combined'}

    for leg in ('long', 'short', 'ls'):
        eq = oos_equity.get(leg, pd.Series(dtype=float))
        if eq.empty:
            continue
        eq.plot(ax=ax, label=labels[leg], color=colours[leg],
                linestyle=styles[leg], linewidth=1.4)

    ax.axhline(1.0, color='black', lw=0.7, linestyle=':')

    # Shade alternating OOS fold bands
    for j, (_, row) in enumerate(df.iterrows()):
        for leg in ('long', 'short', 'ls'):
            eq = oos_equity.get(leg, pd.Series(dtype=float))
            if eq.empty:
                break
            ts = max(pd.Timestamp(str(row['test_start'])), eq.index[0])
            te = min(pd.Timestamp(str(row['test_end'])),   eq.index[-1])
            if ts < te:
                ax.axvspan(ts, te, alpha=0.05,
                           color='steelblue' if j % 2 == 0 else 'grey')
            break  # shade once (shared axis)

    ax.set_title('Stitched OOS Equity Curves — Long / Short / L/S')
    ax.set_ylabel('Portfolio Value')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"Walk-forward plot saved → {save_path}")


# ── Regime diagnostic plot ────────────────────────────────────────────────────

def _plot_regime_detail(detail: pd.DataFrame, save_path: str = 'regime_detail.png'):
    """Four-panel regime diagnostic: VIX, SPY vs MA, bear count, multiplier."""
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.suptitle('Market Regime Indicators', fontsize=12)

    ax = axes[0]
    detail['vix_level'].plot(ax=ax, color='crimson', lw=0.9)
    ax.axhline(25, color='grey', linestyle='--', lw=0.9, label='VIX=25')
    ax.fill_between(detail.index, 25, detail['vix_level'],
                    where=detail['vix_level'] > 25, alpha=0.25, color='crimson')
    ax.set_title('VIX Level  (bear = above 25)')
    ax.set_ylabel('VIX')
    ax.legend(fontsize=7)

    ax = axes[1]
    detail['spx_close'].plot(ax=ax, color='steelblue', lw=0.9, label='S&P500')
    (detail['spx_close'].rolling(200, min_periods=1).mean()
     .plot(ax=ax, color='darkorange', lw=1.1, linestyle='--', label='200-day MA'))
    ax.fill_between(detail.index,
                    detail['spx_close'].min(), detail['spx_close'].max(),
                    where=(detail['ma'] == 1), alpha=0.12, color='salmon')
    ax.set_title('S&P500 vs 200-day MA  (shaded = price below MA)')
    ax.set_ylabel('Price')
    ax.legend(fontsize=7)

    ax = axes[2]
    color_map = {0: 'steelblue', 1: 'gold', 2: 'darkorange', 3: 'crimson'}
    bar_colors = [color_map.get(int(v), 'grey') for v in detail['bear_count']]
    ax.bar(detail.index, detail['bear_count'], color=bar_colors, width=1.5, alpha=0.85)
    ax.set_title('Bear Count  (0=bull  1=caution  2=half-short  3=full-short)')
    ax.set_ylabel('Count')
    ax.set_yticks([0, 1, 2, 3])
    ax.set_ylim(-0.1, 3.4)

    ax = axes[3]
    mult_color_map = {0.0: 'steelblue', 0.5: 'darkorange', 1.0: 'crimson'}
    mult_colors = [mult_color_map.get(float(v), 'grey') for v in detail['multiplier']]
    ax.bar(detail.index, detail['multiplier'], color=mult_colors, width=1.5, alpha=0.9)
    ax.set_title('Short-Leg Multiplier  (blue=0 long-only | orange=0.5 half | red=1.0 full)')
    ax.set_ylabel('Multiplier')
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_ylim(-0.05, 1.1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"Regime detail plot saved -> {save_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    SP25 = [
        "AAPL", "MSFT", "GOOG", "AMZN", "NVDA",
        "META", "TSLA", "BRK-B", "JPM", "JNJ",
        "V", "PG", "UNH", "HD", "MA",
        "DIS", "BAC", "XOM", "CVX", "WMT",
        "NFLX", "ADBE", "CRM", "AMD", "INTC",
    ]

    _common = dict(
        tickers=SP25,
        start='2019-01-01',
        end='2024-12-31',
        train_years=1,
        test_years=1,
        top_n=5,
        rebalance='monthly',
        include_pin=False,
        t_threshold=2.0,
    )

    pd.set_option('display.float_format', '{:.3f}'.format)
    pd.set_option('display.max_columns', 25)
    pd.set_option('display.width', 220)

    # ── Run 1: unconditional L/S (baseline) ──────────────────────────────────
    print("\n" + "="*70)
    print("RUN 1: UNCONDITIONAL L/S (no regime filter)")
    print("="*70)
    res_uncond = run_walk_forward(**_common, use_regime=False)

    # ── Run 2: regime-conditional L/S ────────────────────────────────────────
    print("\n" + "="*70)
    print("RUN 2: REGIME-CONDITIONAL L/S (VIX>25 | 200MA | 12M mom)")
    print("="*70)
    res_cond = run_walk_forward(**_common, use_regime=True, vix_threshold=25.0)

    # ── Side-by-side comparison table ────────────────────────────────────────
    df_u = res_uncond['windows']
    df_c = res_cond['windows']

    print(f"\n{'='*70}")
    print("UNCONDITIONAL vs REGIME-CONDITIONAL -- L/S Sharpe per Fold")
    print(f"{'='*70}")
    print(f"  {'Fold':<6} {'OOS Year':<10} {'Long SR':>8} "
          f"{'Uncond SR':>10} {'Cond SR':>9} "
          f"{'Short%':>7} {'Delta':>7}")
    print(f"  {'-'*64}")
    for fold in df_u.index:
        yr   = str(df_u.loc[fold, 'test_start'])[:4]
        lsr  = df_u.loc[fold, 'long_sharpe']
        u_sr = df_u.loc[fold, 'ls_sharpe']
        c_sr = df_c.loc[fold, 'ls_sharpe']   if fold in df_c.index else float('nan')
        spct = df_c.loc[fold, 'short_active_%'] if fold in df_c.index else float('nan')
        delta = c_sr - u_sr
        note = ''
        if fold == 3 and delta > 0.05:
            note = '  F3: short still active in bear'
        elif fold == 4 and delta > 0.5:
            note = '  F4: RECOVERED -- shorts mostly off in 2023 bull'
        elif fold == 4 and abs(delta) < 0.1:
            note = '  F4: minimal change (regime check the Short% column)'
        print(f"  F{fold:<5} {yr:<10} {lsr:>+8.3f} "
              f"{u_sr:>+10.3f} {c_sr:>+9.3f} "
              f"{spct:>6.0f}% {delta:>+7.3f}{note}")

    print(f"\n  Overall OOS L/S Sharpe (unconditional): "
          f"{res_uncond['summary']['overall_oos_ls_sharpe']:+.4f}")
    print(f"  Overall OOS L/S Sharpe (conditional):   "
          f"{res_cond['summary']['overall_oos_ls_sharpe']:+.4f}")

    # ── F3 / F4 spotlight ─────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SPOTLIGHT: F3 (2022 bear) and F4 (2023 bull) -- regime impact")
    print(f"{'='*70}")
    for fold, label in [(3, '2022 (bear)'), (4, '2023 (bull)')]:
        if fold not in df_u.index:
            continue
        u = df_u.loc[fold]
        c = df_c.loc[fold] if fold in df_c.index else None
        print(f"\n  Fold {fold}  OOS {label}:")
        print(f"    Long-only  SR={u['long_sharpe']:+.3f}  AnnRet={u['long_annret']*100:+.1f}%")
        print(f"    Short days active: {c['short_active_%']:.0f}%" if c is not None else "")
        print(f"    Uncond L/S  SR={u['ls_sharpe']:+.3f}  AnnRet={u['ls_annret']*100:+.1f}%  MDD={u['ls_mdd']*100:.1f}%")
        if c is not None:
            print(f"    Cond   L/S  SR={c['ls_sharpe']:+.3f}  AnnRet={c['ls_annret']*100:+.1f}%  MDD={c['ls_mdd']*100:.1f}%")
            delta = c['ls_sharpe'] - u['ls_sharpe']
            verdict = 'IMPROVED' if delta > 0.05 else ('WORSENED' if delta < -0.05 else 'FLAT')
            print(f"    Delta (cond-uncond): {delta:+.3f}  [{verdict}]")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print()
    plot_walk_forward(res_uncond, save_path='walk_forward_unconditional.png')
    plot_walk_forward(res_cond,   save_path='walk_forward_conditional.png')

    print("\nGenerating regime detail plot...")
    detail = compute_regime_detail('2019-01-01', '2024-12-31')
    _plot_regime_detail(detail, save_path='regime_detail.png')
