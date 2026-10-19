"""
research_log.py  -  write the hypothesis first, log every test, deflate the result.

Setup (once, from ~/algo-scanner):
    python3 research_log.py init
    python3 research_log.py hyp post_results_drift

After EVERY test you run, including the ones you throw away:
    python3 research_log.py log --hyp post_results_drift --desc "20d hold, top 10%" \
        --sharpe 1.4 --days 205 --sample OOS --net

See where you stand:
    python3 research_log.py summary --extra-trials 40
    python3 research_log.py deflate --sharpe 1.6 --days 209 --trials 36
"""
import argparse
import csv
import math
import sys
from datetime import datetime
from pathlib import Path
from statistics import NormalDist

ROOT = Path("research")
HYP_DIR = ROOT / "hypotheses"
LOG = ROOT / "test_log.csv"
FIELDS = ["ts", "hyp", "desc", "sample", "net", "sharpe_ann", "days", "skew", "kurt", "notes"]
EULER = 0.5772156649015329
ND = NormalDist()

TEMPLATE = """# Hypothesis: {name}
Created: {date}      Status: DRAFT   (change to LOCKED before you look at ANY result)

## 1. Who is on the other side of this trade, and why are they there?
(forced seller, index rebalancer, slow reactor, insurance seller... one paragraph, no data yet)

## 2. Why would this persist even though people know about it?
(limits to arbitrage, capacity, risk, cost, effort. If you cannot answer, expect it to be gone.)

## 3. Prediction written BEFORE testing
- Direction / sign:
- Expected gross edge per trade (bps):
- Holding period:
- Universe:
- Expected trades per year:

## 4. Cost model, fixed now
- Per-trade cost (bps: spread + impact + fees):
- STT / taxes (check current rates):
- Round-trip break-even gross edge (bps):

## 5. Spec locked in advance
- Data fields:
- Signal formula:
- Entry / exit rule:
- Every tunable parameter (each one you tune adds to your trial count):
- Maximum number of variants I allow myself:

## 6. Data split, dates fixed now
- Exploration / in-sample:
- Holdout (touched ONCE, at the end):
- Forward / paper period:

## 7. Decision rules, written before results
- PASS if (e.g. net Sharpe in holdout >= ?, deflated Sharpe >= 0.9, positive in both halves):
- KILL if:
- If PASS: paper-trade size, duration, stop condition:

## 8. Notes
(every test is also logged with `python3 research_log.py log ...`)
"""


def need_init():
    if not LOG.exists():
        print("Run `python3 research_log.py init` first.")
        sys.exit(1)


def read_log():
    need_init()
    with LOG.open(newline="") as f:
        return list(csv.DictReader(f))


def expected_max_sr(var_sr, n):
    if n <= 1:
        return 0.0
    return math.sqrt(var_sr) * ((1 - EULER) * ND.inv_cdf(1 - 1 / n) + EULER * ND.inv_cdf(1 - 1 / (n * math.e)))


def deflated_sharpe(sr_ann, days, trials, var_ann=None, skew=0.0, kurt=3.0):
    if days < 3:
        return float("nan"), float("nan")
    if var_ann is None or var_ann <= 0:
        var_ann = 252.0 / days
    sr = sr_ann / math.sqrt(252)
    sr0 = expected_max_sr(var_ann / 252, trials)
    denom = math.sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4 * sr * sr))
    z = (sr - sr0) * math.sqrt(days - 1) / denom
    return ND.cdf(z), sr0 * math.sqrt(252)


def sample_var(vals):
    n = len(vals)
    if n < 3:
        return None
    m = sum(vals) / n
    return sum((v - m) ** 2 for v in vals) / (n - 1)


def cmd_init(_):
    HYP_DIR.mkdir(parents=True, exist_ok=True)
    (ROOT / "TEMPLATE.md").write_text(TEMPLATE.format(name="NAME", date="YYYY-MM-DD"))
    if not LOG.exists():
        with LOG.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()
    print(f"created {ROOT}/ (TEMPLATE.md, hypotheses/, test_log.csv)")


def cmd_hyp(a):
    HYP_DIR.mkdir(parents=True, exist_ok=True)
    fp = HYP_DIR / f"{a.name}.md"
    if fp.exists():
        print(f"{fp} already exists; not overwriting.")
        return
    fp.write_text(TEMPLATE.format(name=a.name, date=datetime.now().strftime("%Y-%m-%d")))
    print(f"created {fp}\nFill in sections 1-7 and set Status: LOCKED before you run any test.")


def cmd_log(a):
    need_init()
    row = {"ts": datetime.now().strftime("%Y-%m-%d %H:%M"), "hyp": a.hyp, "desc": a.desc,
           "sample": a.sample, "net": "net" if a.net else "gross", "sharpe_ann": a.sharpe,
           "days": a.days, "skew": a.skew, "kurt": a.kurt, "notes": a.notes}
    with LOG.open("a", newline="") as f:
        csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
    print(f"logged. total tests so far: {len(read_log())}")


def cmd_summary(a):
    rows = read_log()
    if not rows:
        print("No tests logged yet.")
        return
    sharpes = [float(r["sharpe_ann"]) for r in rows]
    trials = len(rows) + a.extra_trials
    var_ann = sample_var(sharpes)
    print(f"tests logged: {len(rows)} | extra trials before logging: {a.extra_trials} | total trials: {trials}")
    print(f"spread of logged Sharpes (std, annualised): {math.sqrt(var_ann):.2f}" if var_ann else
          "spread of logged Sharpes: need >= 3 tests, using sampling variance instead")
    hyps = sorted({r["hyp"] for r in rows})
    print(f"\n{'hypothesis':<28}{'tests':>6}{'best SR':>9}{'days':>6}{'luck bar':>10}{'DSR prob':>10}")
    for h in hyps:
        sub = [r for r in rows if r["hyp"] == h]
        best = max(sub, key=lambda r: float(r["sharpe_ann"]))
        dsr, bar = deflated_sharpe(float(best["sharpe_ann"]), int(best["days"]), trials, var_ann,
                                   float(best["skew"] or 0), float(best["kurt"] or 3))
        print(f"{h:<28}{len(sub):6d}{float(best['sharpe_ann']):+9.2f}{int(best['days']):6d}{bar:+10.2f}{dsr:10.2f}")
    print("\nluck bar = Sharpe you would expect from the best of that many tries with NO real edge.")
    print("DSR prob = chance the true Sharpe is above zero after that adjustment (want >= ~0.9).")
    print("Counting every variant is the point: a result that needs the best of many is not evidence.")


def cmd_deflate(a):
    dsr, bar = deflated_sharpe(a.sharpe, a.days, a.trials, a.var, a.skew, a.kurt)
    print(f"Sharpe {a.sharpe:+.2f} over {a.days} days, {a.trials} trials")
    print(f"  Sharpe luck alone could reach with that many tries: {bar:+.2f}")
    print(f"  Deflated Sharpe probability (true SR > 0): {dsr:.2f}")
    print(f"  Sharpe standard error for {a.days} days: +/-{math.sqrt(252 / a.days):.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("init").set_defaults(fn=cmd_init)

    h = sp.add_parser("hyp")
    h.add_argument("name")
    h.set_defaults(fn=cmd_hyp)

    l = sp.add_parser("log")
    l.add_argument("--hyp", required=True)
    l.add_argument("--desc", required=True)
    l.add_argument("--sharpe", type=float, required=True, help="annualised Sharpe of this test")
    l.add_argument("--days", type=int, required=True, help="number of daily observations")
    l.add_argument("--sample", default="IS", choices=["IS", "OOS", "PAPER"])
    l.add_argument("--net", action="store_true", help="Sharpe is after costs (otherwise gross)")
    l.add_argument("--skew", type=float, default=0.0)
    l.add_argument("--kurt", type=float, default=3.0)
    l.add_argument("--notes", default="")
    l.set_defaults(fn=cmd_log)

    s = sp.add_parser("summary")
    s.add_argument("--extra-trials", type=int, default=0,
                   help="variants you tried before you started logging (be honest)")
    s.set_defaults(fn=cmd_summary)

    d = sp.add_parser("deflate")
    d.add_argument("--sharpe", type=float, required=True)
    d.add_argument("--days", type=int, required=True)
    d.add_argument("--trials", type=int, required=True)
    d.add_argument("--var", type=float, default=None, help="variance of Sharpes across trials (annualised)")
    d.add_argument("--skew", type=float, default=0.0)
    d.add_argument("--kurt", type=float, default=3.0)
    d.set_defaults(fn=cmd_deflate)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
