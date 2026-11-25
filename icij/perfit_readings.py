"""Reading of every one of the ten source-balanced fits under the rule of Section 3.2 (response to the third review, M2).
Training-label test: each fit against the 999-draw balanced null (bal99 + bal900); held-out-label test: heldout_assoc_icij.json.
Usage: python3 perfit_readings.py <results_dir> <out.json>"""
import sys, json, numpy as np
R, OUT = sys.argv[1], sys.argv[2]
LEAKS = ["Panama Papers", "Paradise Papers", "Offshore Leaks", "Bahamas Leaks"]
b99 = json.load(open(f"{R}/full8_bal99.json")); ha = json.load(open(f"{R}/heldout_assoc_icij.json"))
out = {}
p2 = np.zeros((4, 10)); reads = {}
for k, L in enumerate(LEAKS):
    null = np.array(b99[L]["null_draws"] + json.load(open(f"{R}/bal900_{k}.json"))); B = len(null)
    for r, (a, f) in enumerate(zip(b99[L]["real_draws"], ha[L]["fits"])):
        assert abs(a - f["auc"]) < 1e-9
        tu = (1 + (null >= a).sum()) / (B + 1); td = (1 + (null <= a).sum()) / (B + 1)
        p2[k, r] = min(1, 2 * min(max(tu, f["p_up"]), max(td, f["p_down"])))
        reads[(k, r)] = "up" if max(tu, f["p_up"]) <= max(td, f["p_down"]) else "down"
for r in range(10):  # Holm across the four leaks, separately for each fit index
    o = np.argsort(p2[:, r]); adj = np.maximum.accumulate([min(1, (4 - i) * p2[o[i], r]) for i in range(4)])
    for i, k in enumerate(o): p2[k, r] = adj[i]
for k, L in enumerate(LEAKS):
    rd = ["transfer" if p2[k, r] <= 0.05 and reads[(k, r)] == "up" else "reversal" if p2[k, r] <= 0.05 else "inconclusive" for r in range(10)]
    out[L] = dict(holm=[float(x) for x in p2[k]], reading=rd, n_transfer=rd.count("transfer"), n_reversal=rd.count("reversal"))
    print(L, out[L]["n_transfer"], out[L]["n_reversal"], np.round(p2[k], 3), rd)
json.dump(out, open(OUT, "w"), indent=1)
