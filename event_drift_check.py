"""
event_drift_check.py - locked-spec test: do stocks keep drifting after a big news-type day? (cash, long-only)

SPEC (locked, see research/hypotheses/news_spike_drift.md):
  universe : top 300 EQ names by PRIOR 20-day median cash turnover (known before the day)
  event    : day t with return >= +3% (close vs adjusted prev close) AND volume >= 3x the median of the previous 20 days
  one event per stock per 20 trading days (later events inside that window are ignored)
  trade    : buy at close(t+1), sell at close(t+21)  (20 trading days; you cannot trade the event day itself)
  measure  : excess return = stock return minus equal-weight return of the same universe over the same days
  costs    : all-in round trip 20 / 30 / 40 bps (cash delivery STT 10 bps on buy AND sell + other costs); main = 30
Standard errors cluster by entry month (events in the same month share market and results-season shocks).
Diagnostic only: the mirror (return <= -3%) event, shown as the SHORT side (needs futures to trade).

Run from ~/algo-scanner:   python3 event_drift_check.py
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

TOP_N = 300
RET_CUT = 0.03
VOL_MULT = 3.0
HOLD = 20
COSTS_RT = (20.0, 30.0, 40.0)
MAIN_COST = 30.0
N_VARIANTS = 4   # main + short-side mirror + 2 liquidity halves


def load_panel(dir_):
    files = sorted(dir_.glob("sec_bhavdata_full_*.csv"))
    print(f"cash files: {len(files)}")
    rows = []
    for fp in files:
        m = re.search(r"(\d{2})(\d{2})(\d{4})", fp.name)
        if not m:
            continue
        d = pd.Timestamp(int(m.group(3)), int(m.group(2)), int(m.group(1)))
        try:
            df = pd.read_csv(fp, skipinitialspace=True)
        except Exception:
            continue
        df.columns = [str(c).strip().upper() for c in df.columns]
        if not {"SYMBOL", "SERIES", "PREV_CLOSE", "CLOSE_PRICE", "TTL_TRD_QNTY"} <= set(df.columns):
            continue
        df = df[df["SERIES"].astype(str).str.strip() == "EQ"]
        num = lambda c: pd.to_numeric(df[c], errors="coerce") if c in df.columns else np.nan
        rows.append(pd.DataFrame({
            "date": d, "symbol": df["SYMBOL"].astype(str).str.strip(),
            "prev_close": num("PREV_CLOSE"), "close": num("CLOSE_PRICE"),
            "vol": num("TTL_TRD_QNTY"), "turnover": num("TURNOVER_LACS")}))
    if not rows:
        return None
    p = pd.concat(rows, ignore_index=True)
    p = p[(p["prev_close"] > 0) & (p["close"] > 0) & (p["vol"] > 0)]
    p = p.drop_duplicates(["date", "symbol"]).sort_values(["symbol", "date"]).reset_index(drop=True)
    dates = sorted(p["date"].unique())
    p["di"] = p["date"].map({d: i for i, d in enumerate(dates)})
    p["r"] = p["close"] / p["prev_close"] - 1            # exchange-adjusted daily return
    p.loc[p["r"].abs() > 0.5, "r"] = np.nan
    p["lr"] = np.log1p(p["r"])
    p["vol_med20"] = p.groupby("symbol")["vol"].transform(
        lambda s: s.shift(1).rolling(20, min_periods=15).median())
    # Rank by the PRIOR 20-day median turnover (known before the day). Ranking by same-day turnover
    # selects stocks that moved a lot that day and biases the market benchmark upward.
    p["tov_med20"] = p.groupby("symbol")["turnover"].transform(
        lambda s: s.shift(1).rolling(20, min_periods=15).median())
    p["liq_rank"] = p.groupby("date")["tov_med20"].rank(ascending=False)
    print(f"days: {len(dates)} ({dates[0]:%Y-%m-%d} to {dates[-1]:%Y-%m-%d}), names/day ~{len(p) / len(dates):.0f}")
    return p, len(dates)


def build_events(p, ndates, sign):
    univ = p["liq_rank"] <= TOP_N
    # equal-weight universe daily return, cumulative log by global date index
    mk = p[univ].groupby("di")["r"].mean().reindex(range(ndates))
    cm = np.concatenate([[0.0], np.cumsum(np.log1p(mk.fillna(0.0)).values)])
    if sign > 0:
        ev_flag = (p["r"] >= RET_CUT) & (p["vol"] >= VOL_MULT * p["vol_med20"]) & univ
    else:
        ev_flag = (p["r"] <= -RET_CUT) & (p["vol"] >= VOL_MULT * p["vol_med20"]) & univ
    p = p.assign(event=ev_flag.fillna(False).values)

    out = []
    for sym, g in p.groupby("symbol", sort=False):
        ev = np.flatnonzero(g["event"].values)
        if len(ev) == 0:
            continue
        di = g["di"].values
        lr = g["lr"].values
        ok = np.isfinite(lr)
        cs = np.concatenate([[0.0], np.cumsum(np.where(ok, lr, 0.0))])
        cn = np.concatenate([[0], np.cumsum(~ok)])
        dts = g["date"].values
        liq = g["liq_rank"].values
        last = -10 ** 9
        for i in ev:
            if i - last <= HOLD:
                continue
            last = i                                    # suppression must not depend on the future
            j0, j1 = i + 1, i + 1 + HOLD
            if j1 >= len(g) or di[j1] - di[i] != HOLD + 1:
                continue
            if cn[j1 + 1] - cn[j0 + 1] > 0:
                continue
            stock = np.exp(cs[j1 + 1] - cs[j0 + 1]) - 1.0
            k0 = di[j0]
            mkt = np.exp(cm[k0 + HOLD + 1] - cm[k0 + 1]) - 1.0
            out.append((sym, dts[i], dts[j0], (stock - mkt) * 1e4, liq[i]))
    return pd.DataFrame(out, columns=["symbol", "event_date", "entry_date", "excess_bps", "liq_rank"])


def stats(x, clusters):
    n = len(x)
    m = x.mean()
    s = pd.Series(x - m).groupby(clusters).sum()
    se = np.sqrt((s ** 2).sum()) / n
    return n, m, se


def line(label, ev, sign=1):
    if len(ev) < 30:
        print(f"{label:<34} too few events ({len(ev)})")
        return None
    x = sign * ev["excess_bps"].values
    cl = pd.to_datetime(ev["entry_date"]).dt.to_period("M").values
    n, m, se = stats(x, cl)
    s = f"{label:<34}{n:6d}{m:+9.1f} ({se:5.1f}){m / se:+7.1f}{(x > 0).mean() * 100:6.0f}%"
    for c in COSTS_RT:
        s += f"{m - c:+9.1f}"
    print(s)
    return n, m, se


def hdr(title):
    print("\n" + "=" * 100 + f"\n{title}\n" + "=" * 100)
    print(f"{'':<34}{'events':>6}{'excess bps (SE)':>17}{'t':>7}{'win':>6}"
          + "".join(f"{'net@' + str(int(c)):>9}" for c in COSTS_RT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data_repo/cash_bhavcopies")
    a = ap.parse_args()
    res = load_panel(Path(a.dir))
    if res is None:
        print("no cash files found")
        return
    p, ndates = res

    pos = build_events(p, ndates, +1)
    hdr("1. LOCKED SPEC: long after a +3% / 3x-volume day, buy close(t+1), sell close(t+21), vs universe")
    main_row = line("all positive events", pos)
    if main_row:
        n, m, se = main_row
        years = (pd.to_datetime(pos["entry_date"]).max() - pd.to_datetime(pos["entry_date"]).min()).days / 365.25
        print(f"   net@{int(MAIN_COST)} = {m - MAIN_COST:+.1f} bps per trade, t = {(m - MAIN_COST) / se:+.1f} (month-clustered); "
              f"approx annualised Sharpe-equivalent {((m - MAIN_COST) / se) / np.sqrt(years):+.2f} over {years:.1f} years")

    hdr("2. STABILITY (same events, split)")
    if len(pos):
        yr = pd.to_datetime(pos["entry_date"]).dt.year
        for y in sorted(yr.unique()):
            line(f"entry year {y}", pos[yr == y])
        line("liquidity rank 1-150", pos[pos["liq_rank"] <= 150])
        line("liquidity rank 151-300", pos[pos["liq_rank"] > 150])

    neg = build_events(p, ndates, -1)
    hdr("3. DIAGNOSTIC: mirror event (-3% / 3x volume) as a SHORT (needs futures; names may not be F&O)")
    line("negative events, short side", neg, sign=-1)

    print("\n" + "=" * 100)
    print(f"TRIAL COUNT: count this run as {N_VARIANTS} trials of ONE idea. Log with the Sharpe-equivalent above, e.g.:")
    if main_row:
        n, m, se = main_row
        sr = ((m - MAIN_COST) / se) / np.sqrt(years)
        print(f"  python3 research_log.py log --hyp news_spike_drift --desc \"locked spec, net@30bp\" "
              f"--sharpe {sr:.2f} --days {int(years * 252)} --net")
    print("Read it against the PASS/KILL rules in the hypothesis file, written before this run.")


if __name__ == "__main__":
    main()
