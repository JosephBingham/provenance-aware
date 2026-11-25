"""E2 evaluation, same protocol as ICIJ baseline.py, storing all real and null draws.
Usage: python3 e2eval.py <rep> <R> <B> <Rbal> <Bbal> <out.json>"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sys, json, time
from e2lib import *
rep = sys.argv[1]; R, B, Rb, Bb = map(int, sys.argv[2:6]); OUT = sys.argv[6]
X, y, src = load(rep); rng = np.random.default_rng(7); t0 = time.time()
res = {"rep": rep, "n": len(y), "pos": int(y.sum()),
       "rates": {s: float(y[src == s].mean()) for s in SOURCES}, "sizes": {s: int((src == s).sum()) for s in SOURCES}}
res["source_acc"] = source_accuracy_r(X, src, rng); res["thm2_pred"] = theorem2_r(X, y, src, rng)
res["majority"] = float(max((src == s).mean() for s in SOURCES))
print(rep, "source acc", res["source_acc"], "thm2", res["thm2_pred"], flush=True)
def summ(real, null):
    s = summarize(real, null); s["real_draws"] = real[:, 0].tolist(); s["null_draws"] = null[:, 0].tolist()
    s["real_ap_draws"] = real[:, 1].tolist(); s["null_ap_draws"] = null[:, 1].tolist(); return s
for matched in [False, True]:
    real, null = run_incorpus(X, y, src, 5, 15, seed=100 + matched, matched=matched)
    k = "incorpus_matched" if matched else "incorpus_uniform"; res[k] = summ(real, null)
    print(k, round(res[k]["real"], 3), round(res[k]["null"], 3), f"{time.time()-t0:.0f}s", flush=True)
    json.dump(res, open(OUT, "w"), indent=1, default=float)
for k_, held in enumerate(SOURCES):
    pool, ep, eu = lolo_sets(y, src, held, np.random.default_rng(13))
    real, null = run(X, y, src, pool, ep, eu, R, B, seed=200 + k_)
    s = summ(real, null); s["source_only"] = source_only_r(X, y, src, pool, ep, eu, rng)
    s["n_pos"] = int(len(ep)); s["n_unl"] = int(len(eu)); s["base_rate"] = len(ep) / (len(ep) + len(eu))
    rngp = np.random.default_rng(5); op = []
    for _ in range(9):
        yp = y.copy(); yp[pool] = y[pool][rngp.permutation(len(pool))]; op.append(fit_eval(X, yp, src, pool, ep, eu, rngp)[0])
    s["ordinary_perm"] = float(np.mean(op)); res[f"lolo_{held}"] = s
    if Rb:
        rb, nb = run(X, y, src, pool, ep, eu, Rb, Bb, seed=300 + k_, balanced=True); res[f"lolo_bal_{held}"] = summ(rb, nb)
    print(held, "real", round(s["real"], 3), "ref", round(s["null"], 3), "p", round(s["p"], 3), "src-only", round(s["source_only"], 3),
          "ord", round(s["ordinary_perm"], 3), "| bal", round(res.get(f"lolo_bal_{held}", {}).get("real", np.nan), 3),
          round(res.get(f"lolo_bal_{held}", {}).get("null", np.nan), 3), f"{time.time()-t0:.0f}s", flush=True)
    json.dump(res, open(OUT, "w"), indent=1, default=float)
