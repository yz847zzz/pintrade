# Roadmap

## Research Completed ✅

- [x] Filing Sentiment PEAD analysis
      (`research/filing_pead_analysis.py`, `research/PEAD_RESEARCH.md`)
- [x] event_label PEAD analysis — does EKOP order-flow classify predict forward returns?
      (`research/event_label_pead_analysis.py`, `research/EVENT_LABEL_RESEARCH.md`)
- [x] Two-layer aggregation robustness check — corrects for within-ticker serial correlation
      (`research/event_label_twolayer.py`)
- [x] event_label vs Filing_Sentiment relationship — MI analysis, lead-lag
      (`research/event_sentiment_analysis.py`)
- [x] Benchmark analysis — vs SPY base rate and random SP25 baseline
      (`research/event_label_benchmark.py`)

**Key finding:** EKOP Bad (−1) days predict higher forward returns than Good (+1) days
(contrarian mean-reversion, not informed bearish trading). NoEvent (0) = worst forward
returns. Market-adjusted pure alpha: ~2.4% annualised. Robust across 23/25 tickers.
See `docs/EKOP_MODEL.md` for full summary.

---

## Short Term

- [ ] Re-run SP100 Walk-Forward after SEC download completes
      (sentiment coverage 25→100 tickers, expect Sharpe improvement)
- [ ] Fix 2022 bear market long-side failure
      (implement regime-aware factor weights:
       bull regime → momentum/growth factors
       bear regime → low-vol/value/defensive factors)

## Medium Term

- [ ] Expand universe to 200+ stocks
- [ ] Event-driven framework (trade on 8-K release day, PEAD strategy)
      - Capture Post-Earnings Announcement Drift
      - Hold 3-10 days after filing
      - Independent backtest from monthly framework

## Long Term

- [ ] Combine monthly + event-driven strategies
      (low correlation → higher combined Sharpe)
- [ ] Connect to live trading
- [ ] Multi-agent architecture
      (data agent, quant agent, NLP agent coordinated by PM agent)
