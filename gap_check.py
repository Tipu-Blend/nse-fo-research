"""
gap_check.py - locked-spec test of "overnight persistence" in F&O stocks (cash data as proxy).

SPEC (locked, see research/hypotheses/overnight_gap_persistence.md):
  universe : F&O stocks, top 100 by that day's cash turnover
  signal   : gap(t) = OPEN(t) / PREV_CLOSE(t) - 1, ranked across the universe
  trade    : at close(t) long top 10% / short bottom 10%, exit at OPEN(t+1)  (overnight leg only)
  costs    : 4 trades x c bps + 2 sells x futures STT (2 bps before 2026-04-01, 5 bps after)

Run from ~/algo-scanner:   python3 gap_check.py
Diagnostics below the main result are SPLITS of the same spec, not new strategies.
Rows where the exchange adjusted PREV_CLOSE (dividends, splits) are dropped.
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

AUCTION_START = pd.Timestamp("2025-12-08")
STT_STEP = pd.Timestamp("2026-04-01")
STT_OLD, STT_NEW = 2.0, 5.0
TOP_N = 100
Q = 0.10
NEWS_CUT = 0.03
MIN_NAMES = 20
COSTS = (2.0, 3.0, 4.0)
N_VARIANTS = 8   # main spec + last-price close + 2 news splits + pre/post split + 2 liquidity halves + 1 buffer


def nw_se(x, lag=5):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < lag + 5:
        return np.nan
    e = x - x.mean()
    s = e @ e / n
    for k in range(1, lag + 1):
        s += 2.0 * (1.0 - k / (lag + 1.0)) * (e[k:] @ e[:-k]) / n
    return np.sqrt(s / n) if s > 0 else np.nan


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
        if not {"SYMBOL", "SERIES", "PREV_CLOSE", "OPEN_PRICE", "CLOSE_PRICE"} <= set(df.columns):
            continue
        df = df[df["SERIES"].astype(str).str.strip() == "EQ"]
        num = lambda c: pd.to_numeric(df[c], errors="coerce") if c in df.columns else np.nan
        rows.append(pd.DataFrame({
            "date": d, "symbol": df["SYMBOL"].astype(str).str.strip(),
            "prev_close": num("PREV_CLOSE"), "open": num("OPEN_PRICE"),
            "close": num("CLOSE_PRICE"), "last": num("LAST_PRICE"),
            "turnover": num("TURNOVER_LACS")}))
    if not rows:
        return None
    p = pd.concat(rows, ignore_index=True)
    p = p[(p["prev_close"] > 0) & (p["open"] > 0) & (p["close"] > 0)]
    p = p.drop_duplicates(["date", "symbol"]).sort_values(["symbol", "date"]).reset_index(drop=True)
    dates = sorted(p["date"].unique())
    p["di"] = p["date"].map({d: i for i, d in enumerate(dates)})
    g = p.groupby("symbol")
    nxt = g["di"].shift(-1) == p["di"] + 1
    for c in ["open", "prev_close"]:
        p["n_" + c] = g[c].shift(-1).where(nxt)
    p["exit_date"] = g["date"].shift(-1).where(nxt)
    noadj = (p["n_prev_close"] / p["close"] - 1).abs() < 0.001      # no dividend/split adjustment
    p["x_gap"] = p["open"] / p["prev_close"] - 1
    p["ov_vwap"] = (p["n_open"] / p["close"] - 1).where(noadj)
    p["ov_last"] = (p["n_open"] / p["last"] - 1).where(noadj & (p["last"] > 0))
    for c in ["x_gap", "ov_vwap", "ov_last"]:
        p.loc[p[c].abs() > 0.5, c] = np.nan
    print(f"days: {len(dates)} ({dates[0]:%Y-%m-%d} to {dates[-1]:%Y-%m-%d})")
    return p


def load_fno():
    fp = Path("data_repo/fut_ohlc_master.csv")
    if not fp.exists():
        return None
    try:
        return set(pd.read_csv(fp, usecols=["symbol"])["symbol"].astype(str).str.strip().unique())
    except Exception:
        return None


def run(u, ycol, mask=None):
    """Daily long-short gross return (decimal) for the locked basket."""
    out = []
    for _, day in u.groupby("date"):
        if mask is not None:
            day = day[mask(day)]
        day = day.dropna(subset=["x_gap", ycol])
        if len(day) < MIN_NAMES:
            continue
        n = max(2, int(round(len(day) * Q)))
        r = day.nlargest(n, "x_gap")[ycol].mean() - day.nsmallest(n, "x_gap")[ycol].mean()
        out.append((day["exit_date"].iloc[0], r))
    return pd.DataFrame(out, columns=["exit_date", "gross"])


def net_series(res, c):
    stt = np.where(res["exit_date"] < STT_STEP, STT_OLD, STT_NEW)
    return res["gross"].values * 1e4 - (4.0 * c + 2.0 * stt)


def sharpe(x):
    return x.mean() / x.std(ddof=1) * np.sqrt(252) if len(x) > 5 and x.std() > 0 else np.nan


def line(label, res, show_net=True):
    if len(res) < 20:
        print(f"{label:<44} too few days ({len(res)})")
        return
    g = res["gross"].values * 1e4
    s = f"{label:<44}{len(res):5d}{g.mean():+9.2f} ({nw_se(g):4.2f})"
    if show_net:
        for c in COSTS:
            s += f"{net_series(res, c).mean():+8.2f}"
        s += f"{sharpe(net_series(res, 3.0)):+8.2f}"
    print(s)


def hdr(title):
    print("\n" + "=" * 100 + f"\n{title}\n" + "=" * 100)
    print(f"{'':<44}{'days':>5}{'gross bps (SE)':>16}{'net@2':>8}{'net@3':>8}{'net@4':>8}{'SR@3':>8}")


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
        print(f"universe restricted to {len(fno)} F&O symbols")
    else:
        print("WARNING: fut_ohlc_master.csv not found, using all EQ names (not the F&O universe)")
    p = p.copy()
    p["liq_rank"] = p.groupby("date")["turnover"].rank(ascending=False)
    u = p[p["liq_rank"] <= TOP_N]

    hdr("1. LOCKED SPEC: overnight leg, top/bottom 10%, top-100 liquid F&O names (VWAP close)")
    main_res = run(u, "ov_vwap")
    line("locked spec", main_res)
    if len(main_res) > 20:
        g = main_res["gross"].values * 1e4
        stt_mean = np.where(main_res["exit_date"] < STT_STEP, STT_OLD, STT_NEW).mean()
        print(f"   break-even cost per trade: {(g.mean() - 2 * stt_mean) / 4:+.2f} bps "
              f"(you need real fills cheaper than this)")

    hdr("2. DIAGNOSTIC: is it just the 30-min VWAP close being stale? (same days, last trade as the close)")
    both = u.dropna(subset=["ov_vwap", "ov_last"])
    line("close = 30-min VWAP", run(both, "ov_vwap"))
    line("close = last traded price", run(both, "ov_last"))

    hdr("3. DIAGNOSTIC: news/earnings-type gaps (|gap| > 3%) vs ordinary gaps")
    line("all names", main_res)
    line("exclude |gap| > 3% names", run(u, "ov_vwap", lambda d: d["x_gap"].abs() <= NEWS_CUT))
    line("only |gap| > 3% names", run(u, "ov_vwap", lambda d: d["x_gap"].abs() > NEWS_CUT))

    hdr("4. STABILITY: by period (exit date) and by liquidity")
    if len(main_res):
        for y in sorted(main_res["exit_date"].dt.year.unique()):
            line(f"year {y}", main_res[main_res["exit_date"].dt.year == y])
        line("before 2025-12-08", main_res[main_res["exit_date"] < AUCTION_START])
        line("from 2025-12-08", main_res[main_res["exit_date"] >= AUCTION_START])
    line("liquidity rank 1-50", run(u[u["liq_rank"] <= 50], "ov_vwap"))
    line("liquidity rank 51-100", run(u[u["liq_rank"] > 50], "ov_vwap"))

    print("\n" + "=" * 100)
    print(f"TRIAL COUNT: count this run as {N_VARIANTS} trials of ONE idea. Log the main result and add the rest:")
    if len(main_res) > 20:
        print(f"  python3 research_log.py log --hyp overnight_gap_persistence --desc \"locked spec, net@3bp\" "
              f"--sharpe {sharpe(net_series(main_res, 3.0)):.2f} --days {len(main_res)} --net")
    print(f"  python3 research_log.py summary --extra-trials <your earlier count + {N_VARIANTS - 1}>")
    print("Read it against the PASS/KILL rules you wrote BEFORE running this, not after.")


if __name__ == "__main__":
    main()
