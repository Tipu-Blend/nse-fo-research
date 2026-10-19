"""
paper_review.py - blind review gate for research/hypotheses/composite_rank_paper.md

Reads data_repo/paper_logs/fills_*.csv (one row per leg: signal_date, symbol, side, entry_fill_915, exit_fill_1525).
A basket = all legs sharing a signal_date. It is CLOSED when every leg has both fills.
Basket gross = mean over legs of (+1 long / -1 short) x (exit/entry - 1). Net = gross - COST_BPS.
Shows NOTHING about performance until GATE_KILL closed baskets exist.

Run from ~/algo-scanner:   python3 paper_review.py
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

GATE_KILL = 60
GATE_CAPITAL = 250
DEFAULT_COST_BPS = 10.0


def load(dir_):
    fs = sorted(Path(dir_).glob("fills_*.csv"))
    if not fs:
        raise SystemExit(f"no fills_*.csv in {dir_}")
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    d = d.drop_duplicates(["signal_date", "symbol", "side"], keep="last")
    d["sign"] = np.where(d["side"].astype(str).str.upper().str.startswith("L"), 1.0, -1.0)
    d["entry"] = pd.to_numeric(d["entry_fill_915"], errors="coerce")
    d["exit"] = pd.to_numeric(d["exit_fill_1525"], errors="coerce")
    d["leg"] = d["sign"] * (d["exit"] / d["entry"] - 1.0) * 1e4
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data_repo/paper_logs")
    ap.add_argument("--cost", type=float, default=DEFAULT_COST_BPS, help="round-trip cost in bps of gross notional")
    a = ap.parse_args()
    d = load(a.dir)
    g = d.groupby("signal_date")
    closed = g["leg"].apply(lambda s: s.notna().all())
    n_open = int((~closed).sum())
    x = g["leg"].mean()[closed].sort_index().values - a.cost
    n = len(x)
    print(f"closed baskets: {n}  (open/unfilled: {n_open})")
    if n < GATE_KILL:
        print(f"BLIND: {GATE_KILL - n} more closed baskets before the first review. Performance is hidden on purpose.")
        return
    m, s = x.mean(), x.std(ddof=1)
    se = s / np.sqrt(n)
    print(f"\nREVIEW (n={n}, cost {a.cost:.0f} bps): mean net {m:+.1f} bps/basket (SE {se:.1f}, t {m / se:+.1f}), "
          f"win {(x > 0).mean() * 100:.0f}%, worst {x.min():+.0f}, best {x.max():+.0f}")
    if m <= 0:
        print("KILL: mean net <= 0 after the pre-registered number of baskets.")
        return
    print("CONTINUE (not a pass): net > 0.")
    if n >= GATE_CAPITAL:
        h = n // 2
        ok = m / se >= 2 and x[:h].mean() > 0 and x[h:].mean() > 0
        print(f"halves {x[:h].mean():+.1f} / {x[h:].mean():+.1f}")
        print("CAPITAL DISCUSSION allowed" if ok else "capital rule NOT met: keep running or stop.")
    else:
        print(f"Next gate at {GATE_CAPITAL} baskets ({GATE_CAPITAL - n} to go).")


if __name__ == "__main__":
    main()
