"""
analysis/pin_regime_analysis.py

Regime-conditional PIN-return analysis using the cached VDJ PIN data.

Market regime for month M+1 is defined by SPY's return in M+1:
  Bull:    SPY return > +2%
  Bear:    SPY return < -2%
  Neutral: -2% <= SPY return <= +2%

For each regime we compute:
  - Average next-month stock return per PIN quintile (Q1 low .. Q5 high)
  - Monthly IC (Spearman rank correlation: PIN vs next-month return)
  - ICIR and t-stat

Hypothesis: PIN has negative IC in Bull markets (high-PIN stocks are avoided
by institutions → miss rallies), zero/positive IC in Bear markets (high-PIN
stocks may have already priced in bad news).
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import spearmanr

sys.path.insert(0, "E:/emo/workspace")

# ── Config ────────────────────────────────────────────────────────────────────

CACHE_PATH = "E:/emo/workspace/pintrade/analysis/pin_return_cache.csv"
OUT_PATH   = "E:/emo/workspace/pintrade/analysis/pin_regime_analysis.png"

BULL_THR  =  0.02   # SPY monthly return threshold for Bull
BEAR_THR  = -0.02   # SPY monthly return threshold for Bear

PRICE_START = "2004-12-01"
PRICE_END   = "2011-03-31"


# ── 1. Load cached PIN ────────────────────────────────────────────────────────

def load_pin_cache(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["ym"] = pd.PeriodIndex(df["ym"], freq="M")
    print(f"  Loaded {len(df):,} PIN observations, "
          f"{df['symbol'].nunique()} symbols, "
          f"{df['ym'].nunique()} months")
    return df


# ── 2. Download stock and SPY monthly returns ─────────────────────────────────

def _to_monthly_returns(raw: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    """Normalise yfinance MultiIndex output → long (ym, symbol, return)."""
    if not isinstance(raw.columns, pd.MultiIndex):
        raw.columns = pd.MultiIndex.from_product(
            [[symbols[0]], raw.columns], names=["Ticker", "Price"]
        )
    else:
        if raw.columns.names[0] != "Ticker":
            raw = raw.swaplevel(axis=1).sort_index(axis=1)
        raw.columns.names = ["Ticker", "Price"]

    close   = raw.xs("Close", level="Price", axis=1)
    monthly = close.resample("ME").last()
    ret     = monthly.pct_change()
    ret.index = ret.index.to_period("M")
    ret.index.name = "ym"

    return (
        ret.stack()
           .rename("return")
           .reset_index()
           .rename(columns={"Ticker": "symbol"})
    )


def get_returns(symbols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (stock_ret_df, spy_ret_df); both long with (ym, symbol, return)."""
    print(f"  Downloading stock prices ({len(symbols)} symbols)...")
    raw_stocks = yf.download(
        symbols, start=PRICE_START, end=PRICE_END,
        auto_adjust=True, progress=False,
    )
    stock_ret = _to_monthly_returns(raw_stocks, symbols)

    print("  Downloading SPY prices...")
    raw_spy = yf.download(
        "SPY", start=PRICE_START, end=PRICE_END,
        auto_adjust=True, progress=False,
    )
    spy_ret = _to_monthly_returns(raw_spy, ["SPY"])
    spy_ret = spy_ret.rename(columns={"return": "spy_return"}).drop(
        columns="symbol", errors="ignore"
    )
    return stock_ret, spy_ret


# ── 3. Build merged dataset with regime labels ────────────────────────────────

def build_dataset(pin_df: pd.DataFrame,
                  stock_ret: pd.DataFrame,
                  spy_ret: pd.DataFrame) -> pd.DataFrame:
    """
    PIN in month M  →  forward return = stock return in M+1
                       regime         = SPY return in M+1
    """
    # stock: (ym=earn_month, symbol, return) → shift to pin month
    stock_ret = stock_ret.copy()
    stock_ret["ym_pin"] = stock_ret["ym"] - 1

    merged = pin_df.merge(
        stock_ret[["symbol", "ym_pin", "return"]],
        left_on=["symbol", "ym"],
        right_on=["symbol", "ym_pin"],
        how="inner",
    ).drop(columns="ym_pin")

    # spy: (ym=earn_month) → shift to pin month
    spy_ret = spy_ret.copy()
    spy_ret["ym_pin"] = spy_ret["ym"] - 1

    merged = merged.merge(
        spy_ret[["ym_pin", "spy_return"]],
        left_on="ym",
        right_on="ym_pin",
        how="left",
    ).drop(columns="ym_pin")

    merged = merged.dropna(subset=["PIN", "return", "spy_return"])

    # Regime label based on SPY return in the forward month
    def _regime(r):
        if r > BULL_THR:
            return "Bull"
        if r < BEAR_THR:
            return "Bear"
        return "Neutral"

    merged["regime"] = merged["spy_return"].map(_regime)
    return merged


# ── 4. Quintile analysis per regime ──────────────────────────────────────────

def quintile_ic(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      quint_ret  : avg return per quintile (index 1-5)
      ic_summary : IC mean, std, ICIR, t-stat for this regime subset
    """
    # Cross-sectional quintile assignment per month
    def _assign(grp):
        grp = grp.copy()
        try:
            grp["quintile"] = pd.qcut(
                grp["PIN"], q=5, labels=[1, 2, 3, 4, 5], duplicates="drop"
            )
        except ValueError:
            grp["quintile"] = np.nan
        return grp

    df = df.groupby("ym", group_keys=False).apply(_assign)
    df = df.dropna(subset=["quintile"])
    df["quintile"] = df["quintile"].astype(int)

    quint_ret = (
        df.groupby("quintile")["return"]
        .agg(["mean", "std", "count"])
        .rename(columns={"mean": "avg_ret", "std": "std_ret", "count": "n"})
    )
    quint_ret["avg_ret_pct"] = quint_ret["avg_ret"] * 100
    quint_ret["se_pct"]      = quint_ret["std_ret"] / np.sqrt(quint_ret["n"]) * 100
    quint_ret["t_stat"]      = quint_ret["avg_ret_pct"] / quint_ret["se_pct"]

    # Monthly IC within this regime
    def _ic(grp):
        if len(grp) < 5:
            return np.nan
        r, _ = spearmanr(grp["PIN"], grp["return"])
        return float(r)

    ic_series = df.groupby("ym").apply(_ic).dropna()
    n = len(ic_series)

    ic_summary = {
        "IC_mean":   ic_series.mean(),
        "IC_std":    ic_series.std(),
        "ICIR":      ic_series.mean() / ic_series.std() if ic_series.std() > 0 else np.nan,
        "t_stat":    (ic_series.mean() / (ic_series.std() / np.sqrt(n))
                      if n > 1 else np.nan),
        "IC>0_pct":  (ic_series > 0).mean() * 100,
        "n_months":  n,
        "n_obs":     len(df),
    }

    return quint_ret, pd.Series(ic_summary)


# ── 5. Plot ───────────────────────────────────────────────────────────────────

def plot_regimes(results: dict[str, tuple], out_path: str):
    """
    Three-panel figure: one quintile bar chart per regime.
    results = {"Bull": (quint_ret, ic_summary), "Bear": ..., "Neutral": ...}
    """
    regimes     = ["Bull", "Neutral", "Bear"]
    regime_cols = {
        "Bull":    ("#2166ac", "#92c5de"),   # dark/light blue
        "Neutral": ("#4d4d4d", "#bababa"),   # dark/light grey
        "Bear":    ("#b2182b", "#f4a582"),   # dark/light red
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), sharey=False)

    for ax, regime in zip(axes, regimes):
        if regime not in results:
            ax.set_visible(False)
            continue

        quint_ret, ic_sum = results[regime]
        dark, light = regime_cols[regime]

        xs   = quint_ret.index.tolist()
        vals = quint_ret["avg_ret_pct"].values
        ses  = quint_ret["se_pct"].values

        # Gradient: Q1 light → Q5 dark
        bar_colors = [light, light, dark, dark, dark]
        bars = ax.bar(xs, vals, color=bar_colors,
                      edgecolor="black", linewidth=0.7, zorder=3)
        ax.errorbar(xs, vals, yerr=1.96 * ses,
                    fmt="none", color="black", capsize=4,
                    linewidth=1.1, zorder=4)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

        # Value labels
        for bar, v, se in zip(bars, vals, ses):
            yoff = 0.05 if v >= 0 else -0.18
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + yoff,
                    f"{v:.2f}%", ha="center", va="bottom", fontsize=8.5)

        spread = vals[0] - vals[4]    # Q1 − Q5
        ic_txt = (f"IC={ic_sum['IC_mean']:.3f}  "
                  f"ICIR={ic_sum['ICIR']:.2f}  "
                  f"t={ic_sum['t_stat']:.2f}\n"
                  f"Q1-Q5 spread={spread:+.2f}%  "
                  f"n_months={ic_sum['n_months']:.0f}")

        spy_label = (">+2%" if regime == "Bull" else
                     "<-2%" if regime == "Bear" else "-2% to +2%")

        ax.set_title(f"{regime} Months  (SPY {spy_label})\n{ic_txt}",
                     fontsize=9.5)
        ax.set_xticks(xs)
        ax.set_xticklabels(["Q1\nLow", "Q2", "Q3", "Q4", "Q5\nHigh"])
        ax.set_xlabel("PIN Quintile", fontsize=10)
        ax.set_ylabel("Avg Next-Month Return (%)", fontsize=10)
        ax.grid(axis="y", alpha=0.35, zorder=0)

    fig.suptitle(
        "PIN Quintile Returns by Market Regime  (TAQ 2005-2010, VDJ model)\n"
        "Regime defined by SPY next-month return",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved -> {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # 1. Load cached PIN
    print("=== 1. Loading cached PIN data ===")
    pin_df  = load_pin_cache(CACHE_PATH)
    symbols = pin_df["symbol"].unique().tolist()

    # 2. Download prices
    print("\n=== 2. Downloading prices ===")
    stock_ret, spy_ret = get_returns(symbols)
    print(f"  Stock returns: {len(stock_ret):,} obs")
    print(f"  SPY returns:   {len(spy_ret):,} months")

    # 3. Build merged dataset
    print("\n=== 3. Building dataset with regime labels ===")
    merged = build_dataset(pin_df, stock_ret, spy_ret)
    print(f"  Total obs: {len(merged):,}")

    # Regime distribution
    regime_counts = merged.groupby("regime").agg(
        n_obs=("return", "count"),
        n_months=("ym", "nunique"),
        avg_spy_ret=("spy_return", "mean"),
    )
    print("\n  Regime distribution:")
    print(regime_counts.to_string())

    # 4. Quintile + IC per regime
    print("\n=== 4. Regime-conditional quintile & IC analysis ===")
    results = {}
    for regime in ["Bull", "Neutral", "Bear"]:
        sub = merged[merged["regime"] == regime]
        if len(sub) < 50:
            print(f"  {regime}: too few obs ({len(sub)}), skipping")
            continue
        q, ic = quintile_ic(sub)
        results[regime] = (q, ic)
        print(f"\n  -- {regime} ({ic['n_months']:.0f} months, "
              f"{ic['n_obs']:.0f} obs) --")
        print(q[["avg_ret_pct", "se_pct", "t_stat", "n"]].to_string(
            float_format=lambda x: f"{x:.3f}"))

    # 5. IC comparison table
    print("\n=== 5. IC Summary Across Regimes ===")
    ic_table = pd.DataFrame(
        {r: results[r][1] for r in results}
    ).T[["IC_mean", "IC_std", "ICIR", "t_stat", "IC>0_pct", "n_months", "n_obs"]]
    ic_table.index.name = "Regime"
    print(ic_table.to_string(float_format=lambda x: f"{x:.4f}"))

    # Q1-Q5 spread per regime
    print("\n  Q1-Q5 monthly return spread (Low PIN minus High PIN):")
    for regime, (q, _) in results.items():
        spread = q.loc[1, "avg_ret_pct"] - q.loc[5, "avg_ret_pct"]
        print(f"    {regime:8s}: {spread:+.3f}%/month")

    # 6. Plot
    print("\n=== 6. Saving plot ===")
    plot_regimes(results, OUT_PATH)

    csv_path = OUT_PATH.replace(".png", "_ic_table.csv")
    ic_table.to_csv(csv_path)
    print(f"IC table CSV -> {csv_path}")
