# region imports
from AlgorithmImports import *
# endregion

import numpy as np
from scipy.special import gammaln
from scipy.optimize import minimize


# ── EKOP PIN model (Easley-Kiefer-O'Hara-Paperman 1996) ──────────────────────
# Inlined from pintrade/features/ekop_model.py

def _log_poisson(k: np.ndarray, lam: float) -> np.ndarray:
    """Log PMF of Poisson(lam) at k."""
    return k * np.log(lam + 1e-300) - lam - gammaln(k + 1)


def _ekop_nll(params: np.ndarray, buys: np.ndarray, sells: np.ndarray) -> float:
    """EKOP negative log-likelihood for params = [alpha, delta, mu, epsi]."""
    alpha, delta, mu, epsi = params
    alpha = np.clip(alpha, 1e-6, 1 - 1e-6)
    delta = np.clip(delta, 1e-6, 1 - 1e-6)
    mu    = max(mu,   1e-6)
    epsi  = max(epsi, 1e-6)

    log_none = _log_poisson(buys, epsi)       + _log_poisson(sells, epsi)
    log_good = _log_poisson(buys, epsi + mu)  + _log_poisson(sells, epsi)
    log_bad  = _log_poisson(buys, epsi)       + _log_poisson(sells, epsi + mu)

    log_comp = np.column_stack([
        np.log(1.0 - alpha)               + log_none,
        np.log(alpha) + np.log(1 - delta) + log_good,
        np.log(alpha) + np.log(delta)     + log_bad,
    ])
    m      = log_comp.max(axis=1, keepdims=True)
    loglik = m.squeeze() + np.log(np.exp(log_comp - m).sum(axis=1))
    loglik = np.where(np.isinf(loglik), -1e10, loglik)
    return -np.sum(loglik)


def _fit_ekop(buys: np.ndarray, sells: np.ndarray) -> float:
    """
    Fit EKOP via MLE with 5 restarts.
    Returns PIN_model = alpha*mu / (alpha*mu + 2*epsi), or NaN on failure.
    """
    mean_v = (buys.mean() + sells.mean()) / 2.0
    x0s = [
        [0.3, 0.5, mean_v * 0.10, mean_v * 0.90],
        [0.5, 0.5, mean_v * 0.20, mean_v * 0.80],
        [0.2, 0.3, mean_v * 0.05, mean_v * 0.95],
        [0.7, 0.5, mean_v * 0.30, mean_v * 0.70],
        [0.4, 0.6, mean_v * 0.15, mean_v * 0.85],
    ]
    bounds = [(1e-6, 1 - 1e-6), (1e-6, 1 - 1e-6), (1e-6, None), (1e-6, None)]

    best_nll, best_params = np.inf, None
    for x0 in x0s:
        try:
            res = minimize(_ekop_nll, x0, args=(buys, sells),
                           method="SLSQP", bounds=bounds,
                           options={"maxiter": 5000, "ftol": 1e-10})
            if res.fun < best_nll:
                best_nll, best_params = res.fun, res.x
        except Exception:
            continue

    if best_params is None:
        return np.nan
    alpha, delta, mu, epsi = best_params
    return float(alpha * mu / (alpha * mu + 2.0 * epsi))


# ─────────────────────────────────────────────────────────────────────────────

class PintradeQC(QCAlgorithm):
    """
    Factor-based long-only equity strategy ported from pintrade.

    Factors (cross-sectionally z-scored monthly):
      Momentum_21D  (+)   Momentum_63D   (+)   Momentum_252D (+)
      RSI_5D        (+)   Price_Zscore   (+)   Volatility    (+)
      Amihud        (-)   PE_Ratio       (0)   PB_Ratio      (+)
      ROE           (-)   ROA            (-)

    Rebalance: monthly, long top-N by composite score, equal weight.
    """

    # ── Default IC-informed weights ───────────────────────────────────────────
    WEIGHTS = {
        "Momentum_21D":     1,
        "Momentum_63D":     1,
        "Momentum_252D":    1,
        "RSI_5D":           1,
        "Price_Zscore":     1,
        "Volatility":       1,
        "Amihud":          -1,
        "PE_Ratio":         0,
        "PB_Ratio":         1,
        "ROE":             -1,
        "ROA":             -1,
        "PIN":             -1,   # ICIR -0.297; high PIN → avoid
    }

    TICKERS   = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA",
                 "META", "NVDA", "JPM",  "BAC",  "XOM"]
    TOP_N     = 3      # long positions
    LOOKBACK  = 252    # trading days of history needed
    PIN_WINDOW = 60    # trading days used to fit EKOP each month

    def initialize(self):
        self.set_start_date(2020, 1, 1)
        self.set_end_date(2024, 12, 31)
        self.set_cash(100_000)
        self.set_benchmark("SPY")

        self.set_brokerage_model(
            BrokerageName.INTERACTIVE_BROKERS_BROKERAGE,
            AccountType.MARGIN,
        )

        # Subscribe to daily equity data
        self._symbols = []
        for ticker in self.TICKERS:
            symbol = self.add_equity(ticker, Resolution.DAILY).symbol
            self._symbols.append(symbol)

        # Rolling window: LOOKBACK+30 days of OHLCV per ticker
        win = self.LOOKBACK + 30
        self._history: dict[Symbol, RollingWindow] = {
            s: RollingWindow[TradeBar](win) for s in self._symbols
        }

        # Warm up engine so windows are full on day 1
        self.set_warm_up(win, Resolution.DAILY)

        # Monthly rebalance
        self.schedule.on(
            self.date_rules.month_start(self.TICKERS[0]),
            self.time_rules.after_market_open(self.TICKERS[0], 30),
            self._rebalance,
        )

        self._last_scores: dict[Symbol, float] = {}

    def on_data(self, data: Slice):
        for symbol in self._symbols:
            bar = data.bars.get(symbol)
            if bar is not None:
                self._history[symbol].add(bar)

    # ── Factor computation ────────────────────────────────────────────────────

    def _bars_to_arrays(self, symbol: Symbol):
        """Return (close, high, low, volume) numpy arrays, oldest→newest."""
        win = self._history[symbol]
        if not win.is_ready:
            return None
        bars = [win[i] for i in range(win.count - 1, -1, -1)]
        close  = np.array([b.close  for b in bars], dtype=float)
        high   = np.array([b.high   for b in bars], dtype=float)
        low    = np.array([b.low    for b in bars], dtype=float)
        volume = np.array([b.volume for b in bars], dtype=float)
        return close, high, low, volume

    @staticmethod
    def _rsi(close: np.ndarray, window: int = 5) -> float:
        delta = np.diff(close[-window - 1:])
        gain  = np.where(delta > 0, delta, 0).mean()
        loss  = np.where(delta < 0, -delta, 0).mean()
        if loss == 0:
            return 100.0
        return 100 - 100 / (1 + gain / loss)

    def _compute_factors(self, symbol: Symbol) -> dict | None:
        result = self._bars_to_arrays(symbol)
        if result is None:
            return None
        close, high, low, volume = result
        n = len(close)
        if n < self.LOOKBACK + 1:
            return None

        ret = np.diff(close) / close[:-1]   # daily returns

        # Momentum (shift(1) convention: use close[-2] as "yesterday")
        prev = close[-2]
        mom = {
            "Momentum_21D":  prev / close[-22]  - 1 if n >= 22  else np.nan,
            "Momentum_63D":  prev / close[-64]  - 1 if n >= 64  else np.nan,
            "Momentum_252D": prev / close[-253] - 1 if n >= 253 else np.nan,
        }

        # RSI-5
        rsi = self._rsi(close, 5) if n >= 7 else np.nan

        # Price z-score (20-day)
        roll20 = close[-20:]
        price_z = ((close[-1] - roll20.mean()) / roll20.std()
                   if roll20.std() > 0 else 0.0)

        # Volatility-20
        vol = ret[-20:].std() if len(ret) >= 20 else np.nan

        # Amihud-20 (|ret| / dollar_vol, scaled)
        dollar_vol = close[-21:-1] * volume[-20:]
        dollar_vol = np.where(dollar_vol == 0, np.nan, dollar_vol)
        amihud = (np.abs(ret[-20:]) / dollar_vol).mean() * 1e6

        # Fundamental factors via LEAN's fundamentals feed
        fund = self.securities[symbol].fundamentals
        pe, pb, roe, roa = np.nan, np.nan, np.nan, np.nan
        if fund is not None:
            op = fund.operation_ratios
            vr = fund.valuation_ratios
            try:
                pe  = vr.pe_ratio
                pb  = vr.pb_ratio
                roe = op.roe.three_months
                roa = op.roa.three_months
            except Exception:
                pass

        return {
            **mom,
            "RSI_5D":      rsi,
            "Price_Zscore": price_z,
            "Volatility":  vol,
            "Amihud":      amihud,
            "PE_Ratio":    pe,
            "PB_Ratio":    pb,
            "ROE":         roe,
            "ROA":         roa,
        }

    def _compute_pin(self, symbol: Symbol) -> float:
        """
        Estimate PIN via EKOP MLE using the last PIN_WINDOW days of OHLCV.

        Buy/sell volume split via BVC:
          buy_ratio  = (Close - Low) / (High - Low)
          buys       = Volume * buy_ratio
          sells      = Volume * (1 - buy_ratio)
        """
        win = self._history[symbol]
        n_needed = self.PIN_WINDOW + 1  # +1 so we have PIN_WINDOW complete bars
        if win.count < n_needed:
            return np.nan

        # Collect the most recent PIN_WINDOW bars (oldest → newest)
        bars = [win[i] for i in range(self.PIN_WINDOW - 1, -1, -1)]

        close  = np.array([b.close  for b in bars], dtype=float)
        high   = np.array([b.high   for b in bars], dtype=float)
        low    = np.array([b.low    for b in bars], dtype=float)
        volume = np.array([b.volume for b in bars], dtype=float)

        hl_range  = np.where(high - low > 0, high - low, 1e-10)
        buy_ratio = np.clip((close - low) / hl_range, 0.0, 1.0)
        buys      = volume * buy_ratio
        sells     = volume * (1.0 - buy_ratio)

        if buys.mean() < 1.0 or sells.mean() < 1.0:
            return np.nan

        return _fit_ekop(buys, sells)

    @staticmethod
    def _zscore(values: dict[Symbol, float]) -> dict[Symbol, float]:
        """Cross-sectional z-score; NaN → 0."""
        symbols = list(values)
        arr = np.array([values[s] for s in symbols], dtype=float)
        valid = arr[~np.isnan(arr)]
        if len(valid) < 2 or valid.std() == 0:
            return {s: 0.0 for s in symbols}
        mean, std = valid.mean(), valid.std()
        return {s: (values[s] - mean) / std if not np.isnan(values[s]) else 0.0
                for s in symbols}

    def _score(self, factor_rows: dict[Symbol, dict]) -> dict[Symbol, float]:
        """Weighted composite score per symbol."""
        scores = {s: 0.0 for s in factor_rows}
        for factor, weight in self.WEIGHTS.items():
            if weight == 0:
                continue
            raw = {s: row.get(factor, np.nan) for s, row in factor_rows.items()}
            zs  = self._zscore(raw)
            for s in scores:
                scores[s] += weight * zs[s]
        return scores

    # ── Rebalance ─────────────────────────────────────────────────────────────

    def _rebalance(self):
        if self.is_warming_up:
            return

        factor_rows = {}
        for symbol in self._symbols:
            row = self._compute_factors(symbol)
            if row is not None:
                row["PIN"] = self._compute_pin(symbol)
                factor_rows[symbol] = row

        if not factor_rows:
            return

        scores = self._score(factor_rows)
        self._last_scores = scores

        # Rank and select top-N
        ranked = sorted(scores, key=scores.__getitem__, reverse=True)
        longs  = set(ranked[: self.TOP_N])
        target = 1.0 / len(longs)

        # Liquidate dropped holdings
        for symbol in self._symbols:
            if symbol not in longs and self.portfolio[symbol].invested:
                self.liquidate(symbol)

        # Enter / resize longs
        for symbol in longs:
            self.set_holdings(symbol, target)

        self.log(
            f"Rebalance {self.time.date()}: "
            + ", ".join(f"{s.value}={scores[s]:.2f}" for s in longs)
        )
