"""
data/taq_vs_leeready.py

Compare two buy/sell volume estimators against TAQ ground truth:
  - Lee-Ready (midpoint rule): sign trade by comparing price to prev midpoint
  - BVC (Bulk Volume Classification): buy_ratio = (Close - Low) / (High - Low)

TAQ source: E:/emo/workspace/data/bs2009daily.csv
  columns: symbol, date, dbuys, dsells, dtotbuys, dtotsells
  dbuys/dsells  = trade counts  (number of buy/sell transactions)
  dtotbuys/dtotsells = share volume classified as buy/sell

We compare estimators against both count-based and volume-based ground truth.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import yfinance as yf
from scipy.stats import pearsonr

# ── Config ────────────────────────────────────────────────────────────────────

TAQ_PATH  = "E:/emo/workspace/data/bs2009daily.csv"
OUT_PATH  = "E:/emo/workspace/pintrade/data/taq_comparison.png"
TICKERS   = ["AAPL", "MSFT", "GE", "JPM", "XOM"]
START     = "2009-01-01"
END       = "2009-12-31"


# ── 1. Load TAQ ground truth ──────────────────────────────────────────────────

def load_taq(path: str, tickers: list) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"],
                     usecols=["symbol", "date", "dbuys", "dsells",
                               "dtotbuys", "dtotsells"])
    df = df[df["symbol"].isin(tickers)].copy()
    df = df.rename(columns={"symbol": "Ticker"})
    df = df.set_index(["date", "Ticker"]).sort_index()
    df.index.names = ["Date", "Ticker"]
    return df


# ── 2. Download OHLCV ─────────────────────────────────────────────────────────

def load_ohlcv(tickers: list, start: str, end: str) -> pd.DataFrame:
    raw = yf.download(tickers, start=start, end=end,
                      auto_adjust=True, progress=False)
    if not isinstance(raw.columns, pd.MultiIndex):
        raw.columns = pd.MultiIndex.from_product(
            [tickers, raw.columns], names=["Ticker", "Price"])
    else:
        if raw.columns.names[0] != "Ticker":
            raw = raw.swaplevel(axis=1).sort_index(axis=1)
        raw.columns.names = ["Ticker", "Price"]
    raw.index.name = "Date"
    # stack to long: (Date, Ticker) index
    long = raw.stack(level="Ticker", future_stack=True)
    long.index.names = ["Date", "Ticker"]
    return long


# ── 3. Lee-Ready (midpoint rule) ──────────────────────────────────────────────

def lee_ready(ohlcv_long: pd.DataFrame) -> pd.DataFrame:
    """
    Sign each day using prev-day midpoint as proxy for quote midpoint.
    Mid_t-1 = (High_t-1 + Low_t-1) / 2
    If Close_t > Mid_t-1  → buy day  (buy_ratio = 1)
    If Close_t < Mid_t-1  → sell day (buy_ratio = 0)
    If Close_t == Mid_t-1 → use prior day's direction (tick rule)

    Returns long DataFrame with Buy_Volume and Sell_Volume columns.
    """
    results = []
    for ticker, grp in ohlcv_long.groupby(level="Ticker"):
        grp = grp.copy()
        mid_prev = ((grp["High"] + grp["Low"]) / 2).shift(1)
        close    = grp["Close"]
        volume   = grp["Volume"]

        direction = np.where(close > mid_prev, 1,
                    np.where(close < mid_prev, -1, 0))

        # tick rule for ties (direction=0): carry forward last non-zero
        filled = pd.Series(direction, index=grp.index)
        filled = filled.replace(0, np.nan).ffill().fillna(1)

        buy_ratio  = (filled + 1) / 2          # +1 → 1.0, -1 → 0.0
        buy_volume  = volume * buy_ratio
        sell_volume = volume * (1 - buy_ratio)

        # trade count proxy: split total daily trades equally by direction
        # (TAQ dbuys/dsells are counts; we don't have intraday tick data)
        # We estimate count fraction using the same direction ratio
        grp["LR_buy_vol"]  = buy_volume
        grp["LR_sell_vol"] = sell_volume
        grp["LR_buy_ratio"] = buy_ratio.values
        results.append(grp)

    return pd.concat(results).sort_index()


# ── 4. BVC (Bulk Volume Classification) ───────────────────────────────────────

def bvc(ohlcv_long: pd.DataFrame) -> pd.DataFrame:
    """
    buy_ratio = (Close - Low) / (High - Low)
    Continuous split of daily volume into buy/sell.
    """
    df = ohlcv_long.copy()
    hl_range = (df["High"] - df["Low"]).replace(0, np.nan)
    buy_ratio = ((df["Close"] - df["Low"]) / hl_range).clip(0, 1)
    df["BVC_buy_ratio"] = buy_ratio
    df["BVC_buy_vol"]   = df["Volume"] * buy_ratio
    df["BVC_sell_vol"]  = df["Volume"] * (1 - buy_ratio)
    return df


# ── 5. Merge and evaluate ─────────────────────────────────────────────────────

def evaluate(taq: pd.DataFrame, est: pd.DataFrame) -> pd.DataFrame:
    """
    Join TAQ ground truth with estimator columns.
    Computes per-ticker and overall correlation + MAE for volume estimates.
    """
    merged = taq.join(est[["LR_buy_vol", "LR_sell_vol",
                            "LR_buy_ratio",
                            "BVC_buy_vol", "BVC_sell_vol",
                            "BVC_buy_ratio"]], how="inner")
    merged = merged.dropna()
    return merged


def summary_stats(merged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ticker in TICKERS:
        try:
            m = merged.xs(ticker, level="Ticker")
        except KeyError:
            continue

        for method, buy_col, sell_col in [
            ("LR",  "LR_buy_vol",  "LR_sell_vol"),
            ("BVC", "BVC_buy_vol", "BVC_sell_vol"),
        ]:
            # Volume comparison against dtotbuys / dtotsells
            r_buy,  _ = pearsonr(m["dtotbuys"],  m[buy_col])
            r_sell, _ = pearsonr(m["dtotsells"], m[sell_col])
            mae_buy   = (m["dtotbuys"]  - m[buy_col]).abs().mean()
            mae_sell  = (m["dtotsells"] - m[sell_col]).abs().mean()
            rows.append({
                "Ticker": ticker, "Method": method,
                "r_buy":    round(r_buy,  3),
                "r_sell":   round(r_sell, 3),
                "MAE_buy":  round(mae_buy  / 1e6, 2),   # millions of shares
                "MAE_sell": round(mae_sell / 1e6, 2),
            })
    return pd.DataFrame(rows).set_index(["Ticker", "Method"])


# ── 6. Plot ───────────────────────────────────────────────────────────────────

def plot_comparison(merged: pd.DataFrame, out_path: str):
    """
    Grid: rows = tickers, cols = [LR buy, LR sell, BVC buy, BVC sell]
    Each cell = scatter of estimated vs TAQ volume.
    """
    n_tickers = len(TICKERS)
    fig = plt.figure(figsize=(16, 3.5 * n_tickers))
    gs  = gridspec.GridSpec(n_tickers, 4, hspace=0.45, wspace=0.35)

    col_labels = ["LR — Buy Vol", "LR — Sell Vol",
                  "BVC — Buy Vol", "BVC — Sell Vol"]
    pairs = [
        ("dtotbuys",  "LR_buy_vol"),
        ("dtotsells", "LR_sell_vol"),
        ("dtotbuys",  "BVC_buy_vol"),
        ("dtotsells", "BVC_sell_vol"),
    ]

    for row, ticker in enumerate(TICKERS):
        try:
            m = merged.xs(ticker, level="Ticker").dropna()
        except KeyError:
            continue

        for col, (true_col, est_col) in enumerate(pairs):
            ax = fig.add_subplot(gs[row, col])
            x  = m[true_col]  / 1e6
            y  = m[est_col]   / 1e6

            r, _ = pearsonr(x, y)
            mae  = (x - y).abs().mean()

            ax.scatter(x, y, alpha=0.55, s=18, color="steelblue")

            lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
            ax.plot([lo, hi], [lo, hi], "r--", linewidth=0.9, label="y=x")

            method = "LR" if col < 2 else "BVC"
            side   = "Buy" if col % 2 == 0 else "Sell"
            ax.set_title(f"{ticker} · {method} {side}\nr={r:.3f}  MAE={mae:.1f}M",
                         fontsize=8.5)
            ax.set_xlabel("TAQ (M shares)", fontsize=7.5)
            ax.set_ylabel("Estimated (M shares)", fontsize=7.5)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3)

            if row == 0:
                ax.set_title(f"{col_labels[col]}\n"
                             f"{ticker} · r={r:.3f}  MAE={mae:.1f}M",
                             fontsize=8.5)

    fig.suptitle("TAQ Ground Truth vs Lee-Ready / BVC Estimates (2009)\n"
                 "Volume in millions of shares", fontsize=12, y=1.01)
    plt.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"Plot saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading TAQ ground truth...")
    taq = load_taq(TAQ_PATH, TICKERS)
    print(f"  {len(taq)} rows, {taq.index.get_level_values('Ticker').nunique()} tickers")
    print(f"  date range: {taq.index.get_level_values('Date').min().date()} → "
          f"{taq.index.get_level_values('Date').max().date()}")

    print("\nDownloading OHLCV from yfinance...")
    ohlcv = load_ohlcv(TICKERS, START, END)
    print(f"  {len(ohlcv)} rows")

    print("\nApplying Lee-Ready...")
    lr  = lee_ready(ohlcv)

    print("Applying BVC...")
    est = bvc(lr)   # adds BVC columns on top of LR output

    print("\nMerging with TAQ...")
    merged = evaluate(taq, est)
    print(f"  {len(merged)} matched rows after join + dropna")

    stats = summary_stats(merged)
    print("\n=== Volume correlation & MAE vs TAQ (MAE in millions of shares) ===")
    print(stats.to_string())

    print("\nGenerating comparison plot...")
    plot_comparison(merged, OUT_PATH)

    # Overall pooled stats
    print("\n=== Pooled across all tickers ===")
    for method, buy_col, sell_col in [
        ("Lee-Ready", "LR_buy_vol",  "LR_sell_vol"),
        ("BVC",       "BVC_buy_vol", "BVC_sell_vol"),
    ]:
        rb, _ = pearsonr(merged["dtotbuys"],  merged[buy_col])
        rs, _ = pearsonr(merged["dtotsells"], merged[sell_col])
        mb    = (merged["dtotbuys"]  - merged[buy_col]).abs().mean()  / 1e6
        ms    = (merged["dtotsells"] - merged[sell_col]).abs().mean() / 1e6
        print(f"  {method:12s}  r_buy={rb:.3f}  r_sell={rs:.3f}  "
              f"MAE_buy={mb:.1f}M  MAE_sell={ms:.1f}M")
