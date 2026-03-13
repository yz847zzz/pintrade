# Roadmap

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
