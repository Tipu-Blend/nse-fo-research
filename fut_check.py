"""
fut_check.py - the SAME locked spec as gap_check.py, but earning FUTURES overnight returns.

Same universe (top 100 F&O names by cash turnover), same signal (cash gap), same basket (top/bottom 10%).
Only the return changes: near-month futures close(t) -> same contract open(t+1).
Days where the exchange adjusted the cash prev close (dividends/splits) are dropped, as in gap_check.py.
This is a CHECK of the locked hypothesis, not a new strategy: count it as one more trial.

Run from ~/algo-scanner (needs gap_check.py in the same folder):   python3 fut_check.py
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from gap_check import (AUCTION_START, STT_NEW, STT_OLD, STT_STEP, TOP_N, hdr, line, load_fno,
                       load_panel, net_series, nw_se, run, sharpe)


def load_futures():
    fp = Path("data_repo/fut_ohlc_master.csv")
    if not fp.exists():
        return None
    cols = ["date", "symbol", "expiry_dt", "fut_open", "fut_close"]
    fut = pd.read_csv(fp, parse_dates=["date", "expiry_dt"], usecols=cols)
    fut["symbol"] = fut["symbol"].astype(str).str.strip()
    fut = fut[(fut["fut_open"] > 0) & (fut["fut_close"] > 0)]
    fut = fut.sort_values(["symbol", "expiry_dt", "date"])
    g = fut.groupby(["symbol", "expiry_dt"])
    fut["n_date"] = g["date"].shift(-1)          # next row of the SAME contract
    fut["n_open"] = g["fut_open"].shift(-1)
    fut = fut[fut["expiry_dt"] > fut["date"]]    # contract must still exist tomorrow
    fut = (fut.sort_values(["symbol", "date", "expiry_dt"])
              .drop_duplicates(["symbol", "date"], keep="first"))   # near month
    fut["ov_fut_raw"] = fut["n_open"] / fut["fut_close"] - 1
    return fut[["date", "symbol", "n_date", "ov_fut_raw"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data_repo/cash_bhavcopies")
    a = ap.parse_args()

    p = load_panel(Path(a.dir))
    if p is None:
        print("no cash files found")
        return
    fno = load_fno()
    if fno:
        p = p[p["symbol"].isin(fno)]
    fut = load_futures()
    if fut is None:
        print("data_repo/fut_ohlc_master.csv not found")
        return

    p = p.copy()
    p["liq_rank"] = p.groupby("date")["turnover"].rank(ascending=False)
    u = p[p["liq_rank"] <= TOP_N].copy()
    u = u.merge(fut, on=["date", "symbol"], how="left")
    u["ov_fut"] = u["ov_fut_raw"].where((u["n_date"] == u["exit_date"]) & u["ov_vwap"].notna())
    u.loc[u["ov_fut"].abs() > 0.5, "ov_fut"] = np.nan
    cover = u["ov_fut"].notna().groupby(u["date"]).sum()
    print(f"futures coverage: on average {cover.mean():.0f} of the top-{TOP_N} names per day have a usable futures return")

    hdr("1. LOCKED SPEC EARNING FUTURES RETURNS (near-month, close(t) -> open(t+1))")
    fut_res = run(u, "ov_fut")
    line("futures overnight", fut_res)
    if len(fut_res) > 20:
        g = fut_res["gross"].values * 1e4
        stt_mean = np.where(fut_res["exit_date"] < STT_STEP, STT_OLD, STT_NEW).mean()
        print(f"   break-even cost per trade: {(g.mean() - 2 * stt_mean) / 4:+.2f} bps")

    hdr("2. SAME DAYS AND NAMES: cash return vs futures return (gap between them = basis effect)")
    both = u.dropna(subset=["ov_vwap", "ov_fut"])
    rc, rf = run(both, "ov_vwap"), run(both, "ov_fut")
    line("cash overnight", rc)
    line("futures overnight", rf)
    m = rc.merge(rf, on="exit_date", suffixes=("_cash", "_fut"))
    if len(m) > 20:
        d = (m["gross_fut"] - m["gross_cash"]).values * 1e4
        print(f"   futures minus cash: {d.mean():+.2f} bps/day (SE {nw_se(d):.2f})")

    hdr("3. BEFORE vs AFTER the futures pre-open auction (2025-12-08), same days and names")
    for label, mask in [("before 2025-12-08", lambda r: r["exit_date"] < AUCTION_START),
                        ("from 2025-12-08", lambda r: r["exit_date"] >= AUCTION_START)]:
        line(f"cash    {label}", rc[mask(rc)])
        line(f"futures {label}", rf[mask(rf)])

    print("\n" + "=" * 100)
    print("Count this as 1 more trial. Suggested log line:")
    if len(fut_res) > 20:
        print(f"  python3 research_log.py log --hyp overnight_gap_persistence --desc \"futures version, net@3bp\" "
              f"--sharpe {sharpe(net_series(fut_res, 3.0)):.2f} --days {len(fut_res)} --net")
    print("Pass rule from the hypothesis file still applies; futures is what you would actually trade.")


if __name__ == "__main__":
    main()
