"""
features/vdj_taq.py

Fit the VDJ (Inverse Gaussian-Poisson mixture) PIN model directly on
TAQ trade counts: dbuys / dsells from bs2009daily.csv.

No OHLCV, no BVC, no normalization — raw daily trade counts fed straight
into the VDJ MLE, exactly as the model was designed.

Expected ballpark (Duarte-Young-Joon 2009, liquid stocks):
  PIN  ~ 18%     alpha*mu / (alpha*mu + 2*epsi)
  epsi ~ 73 trades/day   uninformed arrival rate
  mu   ~ 28 trades/day   informed arrival rate
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from pintrade.features.pin_factor import fit_vdj_mle, classify_days

# ── Config ────────────────────────────────────────────────────────────────────

TAQ_PATH = "E:/emo/workspace/data/bs2009daily.csv"
# Pick 5 stocks with mean ~150 total trades/day (epsi~73 regime).
# AAPL/MSFT/GE/JPM/XOM have 4 000–9 000 trades/day — too large for
# the VDJ Bessel path and yield trivial PIN=1/3.
TICKERS  = ["MGEE", "ZF", "PLFE", "RNST", "MKTX"]
OUT_PATH = "E:/emo/workspace/pintrade/features/vdj_taq.png"


# ── Load ──────────────────────────────────────────────────────────────────────

def load_taq(path: str, tickers: list) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"],
                     usecols=["symbol", "date", "dbuys", "dsells"])
    df = df[df["symbol"].isin(tickers)].dropna(subset=["dbuys", "dsells"])
    df["dbuys"]  = df["dbuys"].astype(float)
    df["dsells"] = df["dsells"].astype(float)
    return df


# ── Fit VDJ per ticker ────────────────────────────────────────────────────────

def fit_all(df: pd.DataFrame) -> list[dict]:
    records = []
    for ticker in TICKERS:
        sub   = df[df["symbol"] == ticker].sort_values("date")
        buys  = sub["dbuys"].values
        sells = sub["dsells"].values

        print(f"\n{ticker}  ({len(buys)} days)  "
              f"mean buys={buys.mean():.1f}  mean sells={sells.mean():.1f}")

        params = fit_vdj_mle(buys, sells)
        if not params["converged"]:
            print(f"  [WARN] VDJ did not converge for {ticker}")
            continue

        labels  = classify_days(buys, sells, params)
        n_good  = int((labels ==  1).sum())
        n_bad   = int((labels == -1).sum())
        n_none  = int((labels ==  0).sum())
        pin_emp = (n_good + n_bad) / len(labels)

        pin_model = (params["alpha"] * params["mu"] /
                     (params["alpha"] * params["mu"] + 2 * params["epsi"]))

        print(f"  alpha={params['alpha']:.3f}  delta={params['delta']:.3f}  "
              f"epsi={params['epsi']:.1f}  mu={params['mu']:.1f}  "
              f"psi={params['psi']:.3f}")
        print(f"  PIN_model={pin_model:.3f}  PIN_empirical={pin_emp:.3f}  "
              f"Good={n_good}  Bad={n_bad}  None={n_none}")

        records.append({
            "Ticker":        ticker,
            "alpha":         params["alpha"],
            "delta":         params["delta"],
            "epsi":          params["epsi"],
            "mu":            params["mu"],
            "psi":           params["psi"],
            "PIN_model":     pin_model,
            "PIN_empirical": pin_emp,
            "n_good":        n_good,
            "n_bad":         n_bad,
            "n_none":        n_none,
            "buys":          buys,
            "sells":         sells,
            "labels":        labels,
            "dates":         sub["date"].values,
        })
    return records


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_results(records: list[dict], out_path: str):
    """
    Two-panel layout per ticker:
    Left  — time series of daily buy/sell counts with event shading
    Right — histogram of buy−sell imbalance coloured by event label
    """
    n = len(records)
    fig = plt.figure(figsize=(14, 3.8 * n))
    gs  = gridspec.GridSpec(n, 2, hspace=0.55, wspace=0.3)

    cmap = {1: "steelblue", -1: "salmon", 0: "lightgrey"}
    label_name = {1: "Good", -1: "Bad", 0: "None"}

    for row, rec in enumerate(records):
        ticker = rec["Ticker"]
        dates  = pd.to_datetime(rec["dates"])
        buys   = rec["buys"]
        sells  = rec["sells"]
        labels = rec["labels"]

        # ── Left: time series with event shading ──────────────────────────
        ax1 = fig.add_subplot(gs[row, 0])
        ax1.plot(dates, buys,  color="steelblue", lw=0.8, label="dbuys")
        ax1.plot(dates, sells, color="salmon",    lw=0.8, label="dsells")

        for lbl, color in cmap.items():
            mask = labels == lbl
            ax1.fill_between(dates, 0, np.where(mask, buys, np.nan),
                             alpha=0.15, color=color)

        ax1.set_title(f"{ticker}  PIN={rec['PIN_model']:.3f}  "
                      f"α={rec['alpha']:.2f}  ε={rec['epsi']:.0f}  "
                      f"μ={rec['mu']:.0f}  ψ={rec['psi']:.2f}",
                      fontsize=9)
        ax1.set_ylabel("Trade count / day")
        ax1.legend(fontsize=7, loc="upper right")
        ax1.grid(True, alpha=0.3)

        # ── Right: imbalance histogram by label ───────────────────────────
        ax2 = fig.add_subplot(gs[row, 1])
        imbalance = buys - sells
        for lbl in [1, -1, 0]:
            mask = labels == lbl
            if mask.sum() == 0:
                continue
            ax2.hist(imbalance[mask], bins=25, alpha=0.6,
                     color=cmap[lbl],
                     label=f"{label_name[lbl]} (n={mask.sum()})")
        ax2.axvline(0, color="black", lw=0.8, linestyle="--")
        ax2.set_title(f"{ticker} — daily buy−sell imbalance by event", fontsize=9)
        ax2.set_xlabel("dbuys − dsells (trades)")
        ax2.set_ylabel("Frequency")
        ax2.legend(fontsize=7)
        ax2.grid(True, alpha=0.3)

    fig.suptitle("VDJ PIN Model fitted on TAQ trade counts (2009)\n"
                 "Shading: Good=blue / Bad=red / None=grey",
                 fontsize=11, y=1.005)
    plt.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"\nPlot saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading TAQ trade counts...")
    df = load_taq(TAQ_PATH, TICKERS)
    print(f"Loaded {len(df)} rows for {df['symbol'].nunique()} tickers")

    print("\n=== Fitting VDJ MLE (raw trade counts, no normalization) ===")
    records = fit_all(df)

    if records:
        summary = pd.DataFrame([{
            k: v for k, v in r.items()
            if k not in ("buys", "sells", "labels", "dates")
        } for r in records]).set_index("Ticker")

        print("\n=== Summary ===")
        print(summary[["alpha", "delta", "epsi", "mu", "psi",
                        "PIN_model", "PIN_empirical",
                        "n_good", "n_bad", "n_none"]].to_string())

        plot_results(records, OUT_PATH)
