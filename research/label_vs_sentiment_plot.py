"""
Visualization: EKOP Event Label vs Filing Sentiment
Tickers: AAPL, NVDA, MSFT   |   Years: 2022 (bear) and 2023 (bull)

Layout per figure:
  2 rows x 3 columns  (row 0 = event_label bars, row 1 = filing sentiment line)
  Shared X axis within each column (ticker)

Outputs (saved to pintrade/research/):
  label_vs_sentiment_2022.png
  label_vs_sentiment_2023.png

Run from project root:
    cd E:/emo/workspace && python pintrade/research/label_vs_sentiment_plot.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch

from pintrade.data.loader import load_ohlcv_data
from pintrade.features.ekop_model import compute_ekop_factor
from pintrade.features.factors import load_sentiment_factor

# ── Config ────────────────────────────────────────────────────────────────────
TICKERS  = ["AAPL", "NVDA", "MSFT"]
YEARS    = [2022, 2023]
SENT_DB  = Path(__file__).parent.parent / "data" / "sentiment.db"
OUT_DIR  = Path(__file__).parent

# Colors
C_GOOD    = "#2ca02c"   # green
C_BAD     = "#d62728"   # red
C_NOEVENT = "#bbbbbb"   # gray
C_MA      = "#1f77b4"   # blue (MA line)
C_RAW     = "#999999"   # light gray (raw sentiment)

# ── Load data (once for all tickers and years) ────────────────────────────────
# Need 2021-06-01 as load start: EKOP annual fitting needs full year 2022/2023,
# and we load a little earlier to avoid edge effects.
LOAD_START = "2021-06-01"
LOAD_END   = "2023-12-31"

print(f"Loading OHLCV {LOAD_START} -> {LOAD_END} for {TICKERS}...")
ohlcv = load_ohlcv_data(TICKERS, LOAD_START, LOAD_END)
daily_index = ohlcv.index

print("Computing EKOP event labels (annual)...")
ekop_df = compute_ekop_factor(ohlcv, period='annual')

print("Loading Filing_Sentiment from sentiment.db...")
sent_df = load_sentiment_factor(TICKERS, daily_index, SENT_DB,
                                news_ffill_days=5, filing_ffill_days=90)

# ── Helper: extract one ticker, one year ──────────────────────────────────────

def _get_ticker_year(ticker: str, year: int):
    """
    Returns (dates, labels, sentiment_raw, sentiment_ma) for one ticker/year.
    labels: pd.Series indexed by date, values in {-1, 0, +1}
    """
    year_str = str(year)

    # Event label
    try:
        lbl = (ekop_df['event_label']
               .xs(ticker, level='Ticker')
               .loc[year_str]
               .sort_index())
    except KeyError:
        lbl = pd.Series(dtype=int)

    # Filing sentiment
    try:
        raw = (sent_df['Filing_Sentiment']
               .xs(ticker, level='Ticker')
               .loc[year_str]
               .sort_index())
    except KeyError:
        raw = pd.Series(dtype=float)

    # Align to union of dates
    dates = lbl.index.union(raw.index)
    lbl   = lbl.reindex(dates)
    raw   = raw.reindex(dates).fillna(0.0)
    ma20  = raw.rolling(20, min_periods=1).mean()

    return dates, lbl, raw, ma20


# ── Draw one figure (one year) ────────────────────────────────────────────────

def draw_figure(year: int, save_path: Path):
    bear_bull = "Bear Market" if year == 2022 else "Bull Market"
    print(f"\nDrawing {year} ({bear_bull})...")

    n_tickers = len(TICKERS)
    fig, axes = plt.subplots(
        2, n_tickers,
        figsize=(5.5 * n_tickers, 7),
        sharex='col',
        gridspec_kw={'height_ratios': [1, 1.4], 'hspace': 0.06, 'wspace': 0.12},
    )

    fig.suptitle(
        f"EKOP Event Label vs Filing Sentiment  --  {year} ({bear_bull})",
        fontsize=14, fontweight='bold', y=0.99,
    )

    for col, ticker in enumerate(TICKERS):
        ax_top = axes[0, col]
        ax_bot = axes[1, col]

        dates, labels, raw_sent, ma_sent = _get_ticker_year(ticker, year)

        if dates.empty:
            ax_top.text(0.5, 0.5, 'No data', ha='center', va='center',
                        transform=ax_top.transAxes)
            ax_bot.text(0.5, 0.5, 'No data', ha='center', va='center',
                        transform=ax_bot.transAxes)
            continue

        # ── Panel 1: Event Label bars ─────────────────────────────────────────
        bar_colors = [
            C_GOOD    if v == 1
            else C_BAD     if v == -1
            else C_NOEVENT
            for v in labels.fillna(0).astype(int)
        ]
        ax_top.bar(dates, labels.fillna(0).values,
                   color=bar_colors, width=1.4, alpha=0.85, linewidth=0)

        ax_top.axhline(0, color='black', linewidth=0.6, zorder=5)
        ax_top.set_ylim(-1.6, 1.6)
        ax_top.set_yticks([-1, 0, 1])
        ax_top.set_yticklabels(['-1\nBad', '0\nNoEvent', '+1\nGood'], fontsize=7)
        ax_top.set_ylabel('Event Label', fontsize=9)
        ax_top.grid(axis='y', alpha=0.25, linestyle=':')
        ax_top.spines['bottom'].set_visible(False)

        # Column title (ticker name)
        ax_top.set_title(ticker, fontsize=13, fontweight='bold', pad=6)

        # Legend only on first column
        if col == 0:
            legend_patches = [
                Patch(facecolor=C_GOOD,    label='Good (+1)'),
                Patch(facecolor=C_NOEVENT, label='NoEvent (0)'),
                Patch(facecolor=C_BAD,     label='Bad (-1)'),
            ]
            ax_top.legend(handles=legend_patches, fontsize=7,
                          loc='upper left', framealpha=0.8,
                          handlelength=1.2, handleheight=0.9)

        # Stats annotation (top-right)
        n_good    = (labels == 1).sum()
        n_bad     = (labels == -1).sum()
        n_none    = (labels == 0).sum()
        n_total   = len(labels.dropna())
        ax_top.text(0.99, 0.97,
                    f"G={n_good}  B={n_bad}  N={n_none}",
                    transform=ax_top.transAxes,
                    ha='right', va='top', fontsize=7,
                    color='dimgray',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white',
                              ec='none', alpha=0.7))

        # ── Panel 2: Filing Sentiment ─────────────────────────────────────────
        # Fill area under raw sentiment
        ax_bot.fill_between(dates, 0, raw_sent,
                            where=raw_sent >= 0,
                            alpha=0.18, color=C_GOOD, linewidth=0)
        ax_bot.fill_between(dates, 0, raw_sent,
                            where=raw_sent < 0,
                            alpha=0.18, color=C_BAD, linewidth=0)

        # Raw sentiment (thin gray)
        ax_bot.plot(dates, raw_sent, color=C_RAW,
                    linewidth=0.7, alpha=0.7, zorder=2)

        # 20-day MA (bold colored line)
        ax_bot.plot(dates, ma_sent, color=C_MA,
                    linewidth=2.0, zorder=4, label='20d MA')

        ax_bot.axhline(0, color='black', linewidth=0.8,
                       linestyle='--', alpha=0.5, zorder=3)

        ax_bot.set_ylim(-0.18, 0.18)
        ax_bot.set_ylabel('Filing Sentiment', fontsize=9)
        ax_bot.grid(axis='y', alpha=0.25, linestyle=':')
        ax_bot.spines['top'].set_visible(False)

        # Mean sentiment annotation
        mean_sent = raw_sent[raw_sent != 0].mean() if (raw_sent != 0).any() else 0
        ax_bot.axhline(mean_sent, color=C_MA, linewidth=0.9,
                       linestyle=':', alpha=0.7)
        ax_bot.text(0.99, 0.04,
                    f"mean={mean_sent:+.4f}",
                    transform=ax_bot.transAxes,
                    ha='right', va='bottom', fontsize=7, color=C_MA,
                    bbox=dict(boxstyle='round,pad=0.2', fc='white',
                              ec='none', alpha=0.7))

        if col == 0:
            ax_bot.legend(fontsize=7, loc='upper left', framealpha=0.8)

        # ── X axis formatting ─────────────────────────────────────────────────
        ax_bot.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
        ax_bot.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        ax_bot.xaxis.set_minor_locator(mdates.MonthLocator())
        plt.setp(ax_bot.get_xticklabels(), rotation=0, fontsize=8)

        # Shade Q2 and Q4 lightly for calendar reference
        for q_start, q_end in [
            (f'{year}-04-01', f'{year}-06-30'),
            (f'{year}-10-01', f'{year}-12-31'),
        ]:
            for ax in (ax_top, ax_bot):
                ax.axvspan(pd.Timestamp(q_start), pd.Timestamp(q_end),
                           alpha=0.04, color='steelblue', zorder=0)

        print(f"  {ticker} {year}: {n_total} label days | "
              f"G={n_good} B={n_bad} N={n_none} | "
              f"sent mean={mean_sent:+.4f}")

    # Shared x-label
    fig.text(0.5, 0.01, f'Date ({year})', ha='center', fontsize=10)

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


# ── Generate both figures ─────────────────────────────────────────────────────
for year in YEARS:
    draw_figure(year, OUT_DIR / f"label_vs_sentiment_{year}.png")

print("\nDone.")
