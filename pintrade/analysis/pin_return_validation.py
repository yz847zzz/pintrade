"""
analysis/pin_return_validation.py

Validates PIN (VDJ model) as a return predictor using TAQ ground-truth trade counts.

For each stock-month in the TAQ data (2005-2010):
  1. Fit VDJ MLE on that month's daily dbuys/dsells → PIN
  2. Get next-month return from yfinance daily prices (month-end to month-end)
  3. Sort into 5 PIN quintiles per month; compute avg return per quintile
  4. Compute monthly rank IC (Spearman) and ICIR

Hypothesis: high PIN (information asymmetry) → adverse selection → low next-month return

Runtime estimate (joblib parallel):
  300 symbols × ~50 months × 0.67s / n_cpu
  8 CPUs  → ~630s ≈ 10 min
  16 CPUs → ~315s ≈  5 min
"""

import glob
import multiprocessing
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from joblib import Parallel, delayed
from scipy.stats import spearmanr

sys.path.insert(0, "E:/emo/workspace")
from pintrade.features.pin_factor import fit_vdj_mle

# ── Config ────────────────────────────────────────────────────────────────────

TAQ_DIR     = "E:/emo/workspace/data"
OUT_PATH    = "E:/emo/workspace/pintrade/analysis/pin_return_validation.png"
CACHE_PATH  = "E:/emo/workspace/pintrade/analysis/pin_return_cache.csv"

MIN_DAYS    = 15     # min trading days per stock-month to fit VDJ
MIN_TRADES  = 50     # min mean(dbuys + dsells) per day (VDJ design regime)
MIN_MONTHS  = 24     # min months a symbol must appear (data continuity)
MAX_SYMBOLS = 300    # cap universe for runtime (~10 min on 8 cores)
RANDOM_SEED = 42
N_JOBS      = -1     # joblib workers; -1 = all CPUs


# ── 1. Load TAQ ───────────────────────────────────────────────────────────────

def load_taq(taq_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(f"{taq_dir}/bs20*.csv"))
    dfs = []
    for f in files:
        df = pd.read_csv(
            f,
            usecols=["symbol", "date", "dbuys", "dsells"],
            parse_dates=["date"],
        )
        dfs.append(df)
        print(f"  {os.path.basename(f)}: {len(df):,} rows")

    taq = pd.concat(dfs, ignore_index=True)
    taq = taq.dropna(subset=["dbuys", "dsells"])
    taq["dbuys"]  = taq["dbuys"].astype(float)
    taq["dsells"] = taq["dsells"].astype(float)
    taq["total"]  = taq["dbuys"] + taq["dsells"]
    taq["ym"]     = taq["date"].dt.to_period("M")
    return taq


# ── 2. Select symbols ─────────────────────────────────────────────────────────

def select_symbols(taq: pd.DataFrame) -> list[str]:
    stats = taq.groupby(["symbol", "ym"])["total"].agg(["count", "mean"])
    valid = stats[(stats["count"] >= MIN_DAYS) & (stats["mean"] >= MIN_TRADES)]

    sym_months = valid.groupby("symbol").size()
    eligible   = sym_months[sym_months >= MIN_MONTHS].index.tolist()
    print(f"  Eligible symbols (>={MIN_MONTHS} qualifying months): {len(eligible):,}")

    rng    = np.random.default_rng(RANDOM_SEED)
    sample = rng.choice(eligible, size=min(MAX_SYMBOLS, len(eligible)), replace=False)
    return sorted(sample.tolist())


# ── 3. Fit VDJ per stock-month ────────────────────────────────────────────────

def _fit_one(symbol: str, ym: object, buys: np.ndarray, sells: np.ndarray):
    """Single VDJ fit; returns a result dict or None on non-convergence."""
    params = fit_vdj_mle(buys, sells)
    if not params["converged"]:
        return None
    pin = (params["alpha"] * params["mu"]
           / (params["alpha"] * params["mu"] + 2.0 * params["epsi"]))
    return {"symbol": symbol, "ym": ym, "PIN": float(pin)}


def compute_pin_all(taq: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    sub = taq[taq["symbol"].isin(symbols)]

    # Build task list: only valid stock-months
    tasks = []
    for (sym, ym), grp in sub.groupby(["symbol", "ym"]):
        if len(grp) < MIN_DAYS or grp["total"].mean() < MIN_TRADES:
            continue
        tasks.append((sym, ym, grp["dbuys"].values, grp["dsells"].values))

    n_cpu = multiprocessing.cpu_count() if N_JOBS == -1 else N_JOBS
    est   = len(tasks) * 0.67 / n_cpu
    print(f"  Tasks: {len(tasks):,} stock-months  |  "
          f"Estimated: {est/60:.0f} min on {n_cpu} CPUs")

    results = Parallel(n_jobs=N_JOBS, verbose=1)(
        delayed(_fit_one)(sym, ym, buys, sells)
        for sym, ym, buys, sells in tasks
    )

    records = [r for r in results if r is not None]
    df = pd.DataFrame(records)
    df["ym"] = pd.PeriodIndex(df["ym"], freq="M")
    return df


# ── 4. Monthly prices → next-month returns ────────────────────────────────────

def get_monthly_returns(symbols: list[str],
                        start: str = "2004-12-01",
                        end:   str = "2011-03-31") -> pd.DataFrame:
    print(f"  Downloading daily prices for {len(symbols)} symbols "
          f"({start} → {end})...")
    raw = yf.download(symbols, start=start, end=end,
                      auto_adjust=True, progress=False)

    # Normalize to (Ticker, Price) MultiIndex regardless of yfinance version
    if not isinstance(raw.columns, pd.MultiIndex):
        raw.columns = pd.MultiIndex.from_product(
            [[symbols[0]], raw.columns], names=["Ticker", "Price"]
        )
    else:
        if raw.columns.names[0] != "Ticker":
            raw = raw.swaplevel(axis=1).sort_index(axis=1)
        raw.columns.names = ["Ticker", "Price"]

    close = raw.xs("Close", level="Price", axis=1)   # (Date × Ticker)

    # Month-end close → monthly return
    monthly = close.resample("ME").last()
    ret     = monthly.pct_change()
    ret.index = ret.index.to_period("M")
    ret.index.name = "ym"

    long = (
        ret.stack()
           .rename("return")
           .reset_index()
           .rename(columns={"Ticker": "symbol"})
    )
    return long                                       # (ym, symbol, return)


# ── 5. Merge PIN with next-month return ───────────────────────────────────────

def merge_pin_return(pin_df: pd.DataFrame, ret_df: pd.DataFrame) -> pd.DataFrame:
    """
    PIN measured in month M → match with return IN month M+1.
    ret_df['ym'] is the month the return is earned, so we shift by -1.
    """
    ret_df = ret_df.copy()
    ret_df["ym_pin"] = ret_df["ym"] - 1   # month when PIN was computed

    merged = pin_df.merge(
        ret_df[["symbol", "ym_pin", "return"]],
        left_on=["symbol", "ym"],
        right_on=["symbol", "ym_pin"],
        how="inner",
    ).drop(columns="ym_pin")

    return merged.dropna(subset=["PIN", "return"])


# ── 6. Quintile analysis + IC ─────────────────────────────────────────────────

def quintile_analysis(merged: pd.DataFrame):
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

    merged = merged.groupby("ym", group_keys=False).apply(_assign)
    merged = merged.dropna(subset=["quintile"])
    merged["quintile"] = merged["quintile"].astype(int)

    # Average return per quintile
    quint_ret = (
        merged.groupby("quintile")["return"]
        .agg(["mean", "std", "count"])
        .rename(columns={"mean": "avg_ret", "std": "std_ret", "count": "n"})
    )
    quint_ret["avg_ret_pct"] = quint_ret["avg_ret"] * 100
    quint_ret["se_pct"]      = quint_ret["std_ret"] / np.sqrt(quint_ret["n"]) * 100
    quint_ret["t_stat"]      = quint_ret["avg_ret_pct"] / quint_ret["se_pct"]

    # Monthly IC (Spearman rank correlation: PIN vs next-month return)
    def _ic(grp):
        if len(grp) < 5:
            return np.nan
        r, _ = spearmanr(grp["PIN"], grp["return"])
        return float(r)

    ic_series = merged.groupby("ym").apply(_ic).dropna()

    ic_summary = pd.DataFrame({
        "IC_mean":  [ic_series.mean()],
        "IC_std":   [ic_series.std()],
        "ICIR":     [ic_series.mean() / ic_series.std()],
        "t_stat":   [ic_series.mean() / (ic_series.std() / np.sqrt(len(ic_series)))],
        "IC>0 pct": [(ic_series > 0).mean() * 100],
        "n_months": [len(ic_series)],
    })

    return quint_ret, ic_summary, ic_series


# ── 7. Plot ───────────────────────────────────────────────────────────────────

def plot_results(quint_ret: pd.DataFrame,
                 ic_series: pd.Series,
                 out_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Left: quintile bar chart ──────────────────────────────────────────────
    ax = axes[0]
    colors = ["#4575b4", "#91bfdb", "#fee090", "#fc8d59", "#d73027"]
    xs = quint_ret.index.tolist()
    bars = ax.bar(xs, quint_ret["avg_ret_pct"], color=colors,
                  edgecolor="black", linewidth=0.7, zorder=3)
    ax.errorbar(xs, quint_ret["avg_ret_pct"],
                yerr=1.96 * quint_ret["se_pct"],
                fmt="none", color="black", capsize=5, linewidth=1.2, zorder=4)
    ax.axhline(0, color="black", linewidth=0.9, linestyle="--")

    for bar, (_, row) in zip(bars, quint_ret.iterrows()):
        ypos = bar.get_height() + 0.04 if bar.get_height() >= 0 else bar.get_height() - 0.13
        ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                f"{row['avg_ret_pct']:.2f}%\n(t={row['t_stat']:.1f})",
                ha="center", va="bottom", fontsize=8.5)

    ax.set_xticks(xs)
    ax.set_xticklabels(["Q1\n(Low PIN)", "Q2", "Q3", "Q4", "Q5\n(High PIN)"])
    ax.set_xlabel("PIN Quintile", fontsize=11)
    ax.set_ylabel("Avg Next-Month Return (%)", fontsize=11)
    ax.set_title("PIN Quintile vs Next-Month Return\n"
                 "TAQ 2005–2010, VDJ model, 95% CI", fontsize=11)
    ax.grid(axis="y", alpha=0.4, zorder=0)

    # ── Right: monthly IC bar chart ───────────────────────────────────────────
    ax2 = axes[1]
    x    = [p.to_timestamp() for p in ic_series.index]
    cols = np.where(ic_series.values >= 0, "steelblue", "salmon")
    ax2.bar(x, ic_series.values, color=cols, width=20, zorder=3)
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.axhline(ic_series.mean(), color="darkred", linewidth=1.8,
                label=f"Mean IC = {ic_series.mean():.3f}")
    ax2.set_xlabel("Month", fontsize=11)
    ax2.set_ylabel("Rank IC (Spearman: PIN vs next return)", fontsize=11)
    ax2.set_title("Monthly PIN–Return IC\n"
                  "(negative = high PIN stocks underperform)", fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(axis="y", alpha=0.4, zorder=0)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    n_cpu = multiprocessing.cpu_count()
    print(f"CPUs available: {n_cpu}")

    # ── 1. Load TAQ ──────────────────────────────────────────────────────────
    print("\n=== 1. Loading TAQ data (2005–2010) ===")
    taq = load_taq(TAQ_DIR)
    print(f"  Combined rows: {len(taq):,}  |  "
          f"unique symbols: {taq['symbol'].nunique():,}")

    # ── 2. Select universe ───────────────────────────────────────────────────
    print("\n=== 2. Selecting symbol universe ===")
    symbols = select_symbols(taq)
    print(f"  Selected: {len(symbols)} symbols")

    # ── 3. Fit VDJ (or load cache) ───────────────────────────────────────────
    print("\n=== 3. Fitting VDJ model per stock-month (parallel) ===")
    if os.path.exists(CACHE_PATH):
        print(f"  Loading cached PIN data from {CACHE_PATH}")
        pin_df = pd.read_csv(CACHE_PATH)
        pin_df["ym"] = pd.PeriodIndex(pin_df["ym"], freq="M")
    else:
        pin_df = compute_pin_all(taq, symbols)
        pin_df.to_csv(CACHE_PATH, index=False)
        print(f"  PIN data cached to {CACHE_PATH}")
    conv_rate = len(pin_df) / (len(pin_df) + 1) * 100   # approximate
    print(f"  Converged fits: {len(pin_df):,}")
    print(f"  PIN  mean={pin_df['PIN'].mean():.3f}  "
          f"median={pin_df['PIN'].median():.3f}  "
          f"min={pin_df['PIN'].min():.3f}  max={pin_df['PIN'].max():.3f}")

    # ── 4. Download prices ───────────────────────────────────────────────────
    print("\n=== 4. Downloading monthly prices ===")
    ret_df = get_monthly_returns(symbols)
    print(f"  Return observations: {len(ret_df):,}")

    # ── 5. Merge ─────────────────────────────────────────────────────────────
    print("\n=== 5. Merging PIN → next-month return ===")
    merged = merge_pin_return(pin_df, ret_df)
    print(f"  Matched obs:    {len(merged):,}")
    print(f"  Months covered: {merged['ym'].nunique()}")
    print(f"  Avg stocks/mo:  {merged.groupby('ym').size().mean():.0f}")

    # ── 6. Analysis ──────────────────────────────────────────────────────────
    print("\n=== 6. Quintile analysis & IC ===")
    quint_ret, ic_summary, ic_series = quintile_analysis(merged)

    spread = quint_ret.loc[1, "avg_ret_pct"] - quint_ret.loc[5, "avg_ret_pct"]

    print("\n--- Quintile Returns ---")
    print(quint_ret[["avg_ret_pct", "se_pct", "t_stat", "n"]].to_string(
        float_format=lambda x: f"{x:.3f}"))
    print(f"\nQ1-Q5 spread: {spread:+.3f}%/month")

    print("\n--- Information Coefficient (PIN vs next-month return) ---")
    print(ic_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ── 7. Plot ──────────────────────────────────────────────────────────────
    print("\n=== 7. Saving plot ===")
    plot_results(quint_ret, ic_series, OUT_PATH)

    csv_path = OUT_PATH.replace(".png", "_results.csv")
    quint_ret.to_csv(csv_path)
    print(f"Results CSV → {csv_path}")
