# Hypothesis: overnight_gap_persistence
Status: LOCKED before running gap_check.py   (written 2026-10-18)

## 1. Who is on the other side, and why are they there?
Overnight returns persist day to day, intraday returns reverse (documented in US data, "tug of war").
Stocks that gap up at the open keep gapping up the next morning because the same overnight demand
(news flow, retail/arbitrage positioning for the next open) repeats. Intraday traders then fade it.

## 2. Why would this persist even though people know about it?
Unknown. Cost and capacity are the likely limits: the effect may be smaller than trading cost.
Expect it to vanish if the net edge is zero after real fills.

## 3. Prediction written BEFORE testing
- Direction: long stocks with the highest gap(t), short the lowest, hold overnight
- Expected gross edge per day: about +20 bps (cash quintile spread was +22 bps on F&O names)
- Holding period: close(t) to open(t+1)
- Universe: F&O stocks, top 100 by daily cash turnover

## 4. Cost model, fixed now
- 4 trades x c bps (c = 2, 3 or 4) + 2 sells x futures STT
- STT 2 bps per sell before 2026-04-01, 5 bps after
- Break-even gross edge: about 16-20 bps per day at c = 3

## 5. Spec locked in advance
- Signal: gap(t) = OPEN(t)/PREV_CLOSE(t) - 1, cross-sectional rank
- Basket: top 10% long, bottom 10% short, equal weight, rebalanced daily
- Entry: close(t) (30-min VWAP). Exit: OPEN(t+1)
- Parameters: none tuned (10% and top-100 are fixed, not optimised)
- Maximum variants allowed: 8 (the diagnostics in gap_check.py)

## 6. Data split
- Exploration: 2022-11-01 to 2026-10-16 (already seen: quintile spreads, not this basket)
- True holdout: paper-trade period from 2026-10-19 onward

## 7. Decision rules
- PASS only if ALL: net Sharpe at c=3 >= 1.0; positive net in both before and after 2025-12-08;
  positive net in both liquidity halves; last-trade-close gross >= 70% of VWAP-close gross;
  deflated Sharpe >= 0.9 after counting all trials
- KILL if: net at c=3 <= 0, or last-trade-close gross < 50% of VWAP-close gross
- If PASS: paper trade 60 trading days, log every fill vs the 9:08 indicative price, stop if
  measured cost per trade exceeds the break-even printed by gap_check.py

## 8. Notes
KILLED 2026-10-18: futures net@3bp = -8.3 bps/day, break-even 0.9 bps per trade. Cash +26 bps did not carry to futures.
KILLED 2026-10-18: futures net@3bp = -8.3 bps/day, break-even 0.9 bps per trade. Cash +26 bps did not carry to futures.
