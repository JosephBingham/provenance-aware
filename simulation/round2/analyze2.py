"""Summarise sim2: call rates of three readings of the source-balanced fit, with Clopper-Pearson 95% intervals.
Usage: python3 analyze2.py <out.json> sim2_0.jsonl sim2_1.jsonl"""
import sys, json, collections
from scipy.stats import beta
OUT, files = sys.argv[1], sys.argv[2:]
L = [json.loads(l) for f in files for l in open(f)]
a = 0.05
def cp(k, n):
    lo = 0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1); hi = 1.0 if k == n else beta.ppf(0.975, k + 1, n - k)
    return [k, n, k / n, float(lo), float(hi)]
def calls(r):
    tu, td = r["p_train_up"] <= a / 2, r["p_train_dn"] <= a / 2
    hu, hd = r["p_held_up"] <= a / 2, r["p_held_dn"] <= a / 2
    return {"training": "up" if tu else "down" if td else None, "heldout": "up" if hu else "down" if hd else None,
            "rule": "up" if (tu and hu) else "down" if (td and hd) else None}
res = {}
for scen in ["absent", "specific", "shared", "reversed"]:
    R = [r for r in L if r["scen"] == scen]; n = len(R); d = {"n": n}
    if n == 0: continue
    for k in ["training", "heldout", "rule"]:
        c = [calls(r)[k] for r in R]
        d[k] = {"any": cp(sum(x is not None for x in c), n), "up": cp(sum(x == "up" for x in c), n),
                "down": cp(sum(x == "down" for x in c), n)}
    res[scen] = d
# absent: by whether training has a shortcut-relevant structure (rho>1 and Delta>0) and by signal strength
for key, f in [("absent_by_delta", lambda r: r["delta"]), ("absent_by_shortcut", lambda r: r["rho"] > 1 and r["Delta"] > 0)]:
    g = collections.defaultdict(list)
    for r in L:
        if r["scen"] == "absent": g[str(f(r))].append(r)
    res[key] = {k: {m: cp(sum(calls(r)[m] is not None for r in v), len(v)) for m in ["training", "heldout", "rule"]} for k, v in g.items()}
json.dump(res, open(OUT, "w"), indent=1)
for s in [s for s in ["absent", "specific", "shared", "reversed"] if s in res]:
    print(s, res[s]["n"], {k: [round(x, 3) if isinstance(x, float) else x for x in res[s][k]["any"]] for k in ["training", "heldout", "rule"]},
          "up/down rule", res[s]["rule"]["up"][0], res[s]["rule"]["down"][0])
for key in ["absent_by_delta", "absent_by_shortcut"]:
    print(key, {k: {m: v[m][:2] for m in v} for k, v in res[key].items()})
