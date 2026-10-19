# composite_rank_paper  (LOCKED 2026-10-20)

## 1. Idea
Weekly long/short on the 8-stock F&O universe using the composite rank of options-derived features
(pcr_atm, otm_put_iv_skew, pcr_atm_chg_5d). Run by paper_trade.py. Signal #1 was 2026-10-01.

## 2. Mechanism
Honest status: UNPROVEN. Hypothesis only: options positioning (put/call ratio, put skew) carries short-horizon
information about stocks. Other side of the trade would be uninformed flow. The only backtest evidence is weak:
honest out-of-sample Sharpe +0.22, CI straddling zero. This paper trade exists to MEASURE, not to confirm.

## 3. Prediction (written in advance)
Long top 2 / short bottom 2 by composite rank, 5 trading days. Expected gross edge: small, unknown.
Realistic expectation: indistinguishable from zero after costs.

## 4. Exact specification (frozen from 2026-10-01 signal; patched paper_trade.py version)
- Universe: the 8 symbols in paper_trade.py. Ranking: composite of the 3 features. Tie-break: alphabetical.
- Entry: close of first trading day after the signal date. Exit: close of the 5th trading day later.
- Positions: long 2, short 2, equal weight. One open trade at a time (no overlap).
- NO changes to symbols, features, weights, hold, or tie-break while this test runs. Any change = new hypothesis, restart count.

## 5. Costs
- 16 bps round trip assumed (same as backtest). Log real fills and slippage vs close where possible.

## 6. Data split
- Everything from the 2026-10-01 signal forward is the holdout. No backfilled or edited trades. Never delete a trade.

## 7. Decision rules (written before outcomes)
- Review ONLY after 12 closed trades (about 60 trading days). Do not peek or react before that.
- KILL if mean net per trade <= 0 bps after 12 trades.
- CONTINUE (not pass) if net > 0: keep running to 50 trades (about one year).
- CAPITAL discussion only if after 50 trades: mean net > 0, net t >= 2, and the result holds in both halves.
- With 12 trades this test can only rule the strategy out. It cannot prove it works. Do not size real money on 12 trades.

## 8. Result log
- Trade log: data_repo/paper_logs. Append one line here per review. Final decision (date): ___

## AMENDMENT 2026-10-20 (before any result was seen)
The fills log (data_repo/paper_logs/fills_*.csv) shows an INTRADAY version: entry fill at 9:15, exit fill at 15:25,
one basket per signal_date, legs long/short by composite_rank. Sections 4, 5 and 7 above are replaced by:
- Spec: per signal_date, long the top-ranked and short the bottom-ranked legs in the log; entry fill 9:15, exit fill 15:25 the same day.
  Basket gross = mean over legs of sign x (exit/entry - 1), equal weight.
- Costs: 10 bps round trip assumed (cash intraday: STT on sell, brokerage, charges, slippage). Measure real fills vs indicative_908.
- Decision gates: review only after 60 CLOSED baskets (about 3 months). KILL if mean net <= 0. Otherwise CONTINUE.
- Capital discussion only after 250 closed baskets with net t >= 2 and both halves positive.
- Tool: paper_review.py stays blind until the gate. Do not look at running P&L before that.
