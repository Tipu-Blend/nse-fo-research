# Testing Trading Edges Honestly: NSE F&O Research

A hypothesis-first research workflow for finding, and mostly killing, trading ideas on NSE F&O and cash data (Nov 2022 to Oct 2026).

**Headline result:** about 112 trials, no strategy that survives realistic costs. The process that showed this, and the bugs it caught along the way, are the point of the project.

## Method
1. Write a hypothesis card first: who is on the other side of the trade, and why do they lose?
2. Lock the exact spec (universe, signal, entry and exit prices, costs) before running anything.
3. Pre-register PASS and KILL rules with numbers.
4. Log every variant as a trial and adjust for luck with the deflated Sharpe ratio.
5. Treat only forward data as a true holdout.

## Results
| Idea | Result | Verdict |
| --- | --- | --- |
| Futures opening-print theory (pre-open auction) | Cash stocks show the same pattern; the auction changed nothing | Falsified |
| Overnight gap persistence (futures) | Net -8.3 bps per day; breakeven cost 0.9 bps per trade | Killed |
| Swing archetypes | 8 tests, best Sharpe +0.22, deflated Sharpe probability 0.00 | Null |
| Short straddles (volatility risk premium) | Sharpe 3.40 on 26 periods fell to 0.01 once blow-ups were included; locked test net -8.4 bps per cycle | Killed |
| News-spike drift | Gross +48 bps, net +18 bps at 30 bps cost, t = 1.0 | Parked |
| Options-based composite rank | Out-of-sample Sharpe +0.22, interval straddles zero | Running as a paper trade |

## Mistakes the process caught
- Cost model charged 2 trades per round trip when 4 are needed.
- Ranking by same-day turnover biased the benchmark upward.
- Dropping blown-up trades turned a straddle Sharpe of 0.01 into 3.40.
- A positional shift skipped missing strike rows, so exits landed on the wrong day.

## Contents
- `research/hypotheses/` - locked hypothesis cards with decision rules
- `research/test_log.csv` - every logged trial
- `research_log.py` - trial log and deflated Sharpe
- `gap_check.py`, `fut_check.py`, `event_drift_check.py`, `straddle_check.py` - the locked-spec tests
- `paper_review.py` - blind review gate for the forward paper trade

## Data
Data is not included. The scripts expect NSE bhavcopy files (public, from the NSE archives) in a local `data_repo/` folder.

Educational research only. Nothing here is investment advice.
