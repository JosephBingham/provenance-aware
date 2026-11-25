"""Summarise the 999-draw source-balanced nulls (99 from bal99.py + 900 from bal999.py) per held-out leak:
null mean and upper quantiles, one- and two-sided permutation p of the first (pre-specified) real fit, number of the
ten real fits above the null's 97.5% quantile, and Holm-adjusted two-sided p across the four leaks.
Usage: python3 bal999_summary.py <results_dir>"""
import sys, json, numpy as np
R = sys.argv[1]
LEAKS = ["Panama Papers", "Paradise Papers", "Offshore Leaks", "Bahamas Leaks"]
b99 = json.load(open(f"{R}/full8_bal99.json")); out = {}
for k, L in enumerate(LEAKS):
    n = np.array(b99[L]["null_draws"] + json.load(open(f"{R}/bal900_{k}.json"))); real = np.array(b99[L]["real_draws"])
    x = real[0]; B = len(n)
    pu = (1 + (n >= x).sum()) / (B + 1); pl = (1 + (n <= x).sum()) / (B + 1)
    out[L] = dict(B=B, real_first=x, real_median=float(np.median(real)), null_mean=n.mean(), q975=np.quantile(n, .975),
                  q995=np.quantile(n, .995), nmax=n.max(), p_one=pu, p_two=min(1.0, 2 * min(pu, pl)),
                  beyond=int((real > np.quantile(n, .975)).sum()))
order = sorted(LEAKS, key=lambda L: out[L]["p_two"]); run = 0.0
for i, L in enumerate(order):
    run = max(run, min(1.0, (len(LEAKS) - i) * out[L]["p_two"])); out[L]["holm"] = run
json.dump(out, open(f"{R}/full8_bal999_summary.json", "w"), indent=1, default=float)
for L in LEAKS: print(L, {k: round(float(v), 4) for k, v in out[L].items()})
