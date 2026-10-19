"""
straddle_check.py - locked-spec test for research/hypotheses/short_straddle_vrp.md

SPEC (locked 2026-10-20): short 5-day ATM straddle (nearest strike within 3%) on the top 10% (min 3 names)
of ex_ante_vrp = atm_iv - rv20. Entry = straddle close on t+1, exit = straddle close 5 trading days later.
Liquidity: min(CE,PE) volume >= 5 and OI >= 1000, dte at entry >= 7. Non-overlapping cycles (every 5th date).
Return = (entry - exit straddle) / futures price. Costs 15 / 25 bps of notional.
Missing exit option price (no row that day) is NOT assumed to be a free win. Three treatments:
  (a) exclude the trade   (b) intrinsic value only   (c) zero P&L for the trade
"""
from pathlib import Path

import numpy as np
import pandas as pd

HOLD = 5
Q = 0.10
MIN_NAMES = 30
MIN_VOL, MIN_OI = 5, 1000
MIN_DTE = 7
MAX_STRIKE_DIST = 0.03
SPLIT_GUARD = 0.40
COSTS = (15.0, 25.0)
WORST_LIMIT = -400.0
MIN_CYCLES = 100
N_VARIANTS = 3

FUT = Path("data_repo/fut_ohlc_master.csv")
FEATS = Path("data_repo/heavy_features_master_v31.csv")
STRAD = Path("data_repo/atm_straddles_master.csv")


def load():
    for f in (FUT, FEATS, STRAD):
        if not f.exists():
            raise SystemExit(f"missing {f}")
    fut = pd.read_csv(FUT, parse_dates=["date", "expiry_dt"], usecols=["date", "symbol", "expiry_dt", "fut_close"])
    fut["symbol"] = fut["symbol"].astype(str).str.strip()
    fut = fut[fut["fut_close"] > 0].drop_duplicates(["date", "symbol", "expiry_dt"])
    fut["dte"] = (fut["expiry_dt"] - fut["date"]).dt.days
    feats = pd.read_csv(FEATS, parse_dates=["trade_date"], usecols=["trade_date", "symbol", "atm_iv"]).rename(
        columns={"trade_date": "date"})
    feats["symbol"] = feats["symbol"].astype(str).str.strip()
    st = pd.read_csv(STRAD, parse_dates=["date", "expiry_dt"])
    st["symbol"] = st["symbol"].astype(str).str.strip()
    return fut, feats, st


def build(fut, feats, st):
    dates = sorted(fut["date"].unique())
    di = {d: i for i, d in enumerate(dates)}
    near = fut[(fut["dte"] >= 2) & (fut["dte"] <= 40)].copy()
    near["pref"] = np.where(near["dte"] >= MIN_DTE, 0, 1)
    near = near.sort_values(["date", "symbol", "pref", "expiry_dt"]).drop_duplicates(["date", "symbol"])
    near["di"] = near["date"].map(di)

    m = near[["date", "symbol", "fut_close", "di"]].merge(feats, on=["date", "symbol"], how="inner")
    m = m.sort_values(["symbol", "date"]).reset_index(drop=True)
    scale = 100.0 if m["atm_iv"].median() > 2.0 else 1.0
    m["iv"] = (m["atm_iv"] / scale).clip(0.05, 2.00)
    m["ret1"] = m.groupby("symbol")["fut_close"].pct_change().where(lambda s: s.abs() <= 0.25, 0.0)
    m["rv20"] = m.groupby("symbol")["ret1"].transform(lambda s: s.rolling(20, min_periods=15).std()) * np.sqrt(252.0)
    m["vrp"] = m["iv"] - m["rv20"]
    m = m.dropna(subset=["vrp"])
    m["entry_di"] = m["di"] + 1

    st = st.copy()
    st["di"] = st["date"].map(di)
    st = st.dropna(subset=["di"])
    st["di"] = st["di"].astype(int)
    e = st[(st["min_vol"] >= MIN_VOL) & (st["min_oi"] >= MIN_OI)]
    e = e.merge(near[["di", "symbol", "expiry_dt", "fut_close", "dte"]].rename(columns={"fut_close": "f_entry"}),
                on=["di", "symbol", "expiry_dt"], how="inner")
    e = e[e["dte"] >= MIN_DTE]
    e["dist"] = (e["strike_pr"] - e["f_entry"]).abs() / e["f_entry"]
    e = e[e["dist"] <= MAX_STRIKE_DIST].sort_values(["di", "symbol", "dist"]).drop_duplicates(["di", "symbol"])
    e = e.rename(columns={"di": "entry_di", "straddle_close": "s_entry"})

    e["exit_di"] = e["entry_di"] + HOLD
    x = st[["di", "symbol", "expiry_dt", "strike_pr", "straddle_close"]].rename(
        columns={"di": "exit_di", "straddle_close": "s_exit"})
    e = e.merge(x, on=["exit_di", "symbol", "expiry_dt", "strike_pr"], how="left")
    fx = fut[["date", "symbol", "expiry_dt", "fut_close"]].copy()
    fx["exit_di"] = fx["date"].map(di)
    fx = fx.dropna(subset=["exit_di"]).drop(columns="date")
    fx["exit_di"] = fx["exit_di"].astype(int)
    e = e.merge(fx.rename(columns={"fut_close": "f_exit"}), on=["exit_di", "symbol", "expiry_dt"], how="left")
    e = e.dropna(subset=["f_exit"])
    move = (e["f_exit"] / e["f_entry"] - 1).abs()
    n_split = int((move > SPLIT_GUARD).sum())
    e = e[move <= SPLIT_GUARD].copy()
    intrinsic = (e["strike_pr"] - e["f_exit"]).abs()
    e["missing"] = e["s_exit"].isna()
    e["ret_a"] = np.where(e["missing"], np.nan, (e["s_entry"] - np.fmax(e["s_exit"], intrinsic)) / e["f_entry"])
    e["ret_b"] = (e["s_entry"] - np.fmax(e["s_exit"].fillna(0.0), intrinsic)) / e["f_entry"]
    e["ret_c"] = np.where(e["missing"], 0.0, (e["s_entry"] - np.fmax(e["s_exit"], intrinsic)) / e["f_entry"])
    e["prem"] = e["s_entry"] / e["f_entry"]

    t = m[["entry_di", "symbol", "date", "vrp"]].merge(
        e[["entry_di", "symbol", "ret_a", "ret_b", "ret_c", "missing", "prem", "exit_di"]],
        on=["entry_di", "symbol"], how="inner")
    t["entry_date"] = t["entry_di"].map(lambda i: dates[i])
    return t, n_split


def cycle_returns(t, col, offset=0):
    ds = sorted(t["entry_di"].unique())[offset::HOLD]
    out = []
    for d in ds:
        day = t[t["entry_di"] == d].dropna(subset=[col])
        if len(day) < MIN_NAMES:
            continue
        n = max(3, int(len(day) * Q))
        out.append((day["entry_date"].iloc[0], day.nlargest(n, "vrp")[col].mean() * 1e4))
    return pd.DataFrame(out, columns=["date", "gross"])


def summarize(label, c, years):
    g = c["gross"].values
    n = len(g)
    if n < 10:
        print(f"{label:<30} too few cycles ({n})")
        return None
    ann = np.sqrt(n / years)
    out = {"n": n, "mean": g.mean(), "t": g.mean() / (g.std(ddof=1) / np.sqrt(n)), "worst": g.min()}
    s = f"{label:<30}{n:7d}{g.mean():+9.1f}{out['t']:+7.1f}{(g > 0).mean() * 100:6.0f}%{g.min():+9.0f}"
    for cst in COSTS:
        x = g - cst
        out[f"net{int(cst)}"] = x.mean()
        out[f"sr{int(cst)}"] = x.mean() / x.std(ddof=1) * ann
        s += f"{x.mean():+9.1f}{out[f'sr{int(cst)}']:+7.2f}"
    h = n // 2
    out["h1"] = (g[:h] - COSTS[1]).mean()
    out["h2"] = (g[h:] - COSTS[1]).mean()
    s += f"   halves net@25: {out['h1']:+.1f} / {out['h2']:+.1f}"
    print(s)
    return out


def main():
    fut, feats, st = load()
    t, n_split = build(fut, feats, st)
    print(f"straddle trades: {len(t)}, dates: {t['entry_di'].nunique()}, avg premium {t['prem'].mean() * 100:.2f}% of futures price")
    print(f"missing exit option price: {t['missing'].mean() * 100:.1f}% of trades; dropped as split/corp-action guard: {n_split}")
    years = (t["entry_date"].max() - t["entry_date"].min()).days / 365.25

    print("\n" + "=" * 130)
    print("LOCKED SPEC: short 5d ATM straddle, top 10% ex-ante VRP. Gross/net in bps of futures notional per 5-day cycle")
    print("=" * 130)
    print(f"{'treatment':<30}{'cycles':>7}{'gross':>9}{'t':>7}{'win':>6}{'worst':>9}"
          + "".join(f"{'net@' + str(int(c)):>9}{'SR':>7}" for c in COSTS))
    res = {}
    for lab, col in [("(a) exclude missing exits", "ret_a"), ("(b) intrinsic at missing", "ret_b"),
                     ("(c) zero P&L at missing", "ret_c")]:
        res[col] = summarize(lab, cycle_returns(t, col), years)

    allrows = t.dropna(subset=["ret_c"])
    base = allrows.groupby("entry_di")["ret_c"].mean().mean() * 1e4
    print(f"\nreference, ALL liquid straddles equal-weight (treatment c): {base:+.1f} bps/cycle gross")
    print("diagnostic: other cycle start offsets (same spec), net@25 under (c):")
    for off in range(HOLD):
        c = cycle_returns(t, "ret_c", off)
        if len(c) > 10:
            print(f"   offset {off}: cycles {len(c)}, gross {c['gross'].mean():+.1f}, net@25 {(c['gross'] - 25).mean():+.1f}")

    print("\n" + "=" * 130)
    print("DECISION (rules from the hypothesis file)")
    a, c = res.get("ret_a"), res.get("ret_c")
    if a is None or c is None:
        print("not enough cycles to judge")
        return
    kills = []
    if c["net25"] <= 0: kills.append("net@25 <= 0 under (c)")
    if a["t"] < 2: kills.append(f"gross t {a['t']:.1f} < 2 under (a)")
    if min(a["worst"], c["worst"]) < WORST_LIMIT: kills.append(f"worst cycle {min(a['worst'], c['worst']):.0f} < {WORST_LIMIT:.0f} bps")
    if kills:
        print("KILL: " + "; ".join(kills))
    else:
        ok = (a["net25"] > 0 and c["net25"] > 0 and min(a["n"], c["n"]) >= MIN_CYCLES
              and min(a["h1"], a["h2"], c["h1"], c["h2"]) > 0)
        print("PASS for PAPER TRADING only (not capital)" if ok else
              "NO KILL, NO PASS: fails cycle-count or half-sample rule. Do nothing until forward paper data exists.")
    print(f"\nTRIAL COUNT: count this run as {N_VARIANTS} trials. Suggested log:")
    print(f"  python3 research_log.py log --hyp short_straddle_vrp --desc \"locked spec, (c) net@25bp\" "
          f"--sharpe {c['sr25']:.2f} --days {c['n'] * HOLD} --net")


if __name__ == "__main__":
    main()
