"""Reproduce manuscript Tables 9-12 (in-corpus, leave-one-leak-out) for given feature set.
Usage: python3 baseline.py <work_dir> <featset> <R> <B> <Rbal> <Bbal> <out_json> [X_npy]"""
import sys, json, time, numpy as np, pandas as pd
from evallib import *

W, fs, R, B, Rb, Bb, OUT = sys.argv[1], sys.argv[2], *map(int, sys.argv[3:7]), sys.argv[7]
df = pd.read_parquet(f"{W}/officers.parquet")
y = df.y.values.astype(int); leak = df.leak.values
F8 = ["ent_deg", "log_ent_deg", "n_addr", "n_inter", "n_jur", "role_div", "ent_per_addr", "ent_per_inter"]
SETS = {"conn": ["ent_deg", "n_addr", "n_inter"], "full8": F8,
        "full12": F8 + ["pagerank", "kcore", "nbr_deg_mean", "nbr_deg_max"]}
if fs in SETS:
    X = df[SETS[fs]].values.astype(np.float32)
else:
    X = np.load(sys.argv[8]).astype(np.float32)
res = {"featset": fs}
rng = np.random.default_rng(7)
t0 = time.time()
res["source_acc"] = float(source_accuracy(X, leak, rng))
res["thm3_pred"] = float(theorem3(X, y, leak, rng))
print(fs, "source acc", res["source_acc"], "thm3", res["thm3_pred"], flush=True)
for matched in [False, True]:
    real, null = run_incorpus(X, y, leak, max(R // 2, 3), 15 if B else 0, seed=100 + matched, matched=matched)
    key = "incorpus_matched" if matched else "incorpus_uniform"
    res[key] = summarize(real, null) if B else {"real": real[:, 0].mean()}
    print(key, {k: round(v, 3) for k, v in res[key].items()}, f"{time.time()-t0:.0f}s", flush=True)
for held in LEAKS:
    pool, ep, eu = lolo_sets(y, leak, held, np.random.default_rng(13))
    real, null = run(X, y, leak, pool, ep, eu, R, B, seed=200 + LEAKS.index(held))
    s = summarize(real, null); s["source_only"] = float(source_only_heldout(X, y, leak, pool, ep, eu, rng))
    # ordinary (ignore-source) permutation reference
    rngp = np.random.default_rng(5); op = []
    for _ in range(9):
        yp = y.copy(); yp[pool] = y[pool][rngp.permutation(len(pool))]
        op.append(fit_eval(X, yp, leak, pool, ep, eu, rngp)[0])
    s["ordinary_perm"] = float(np.mean(op))
    res[f"lolo_{held}"] = s
    if Rb:
        realb, nullb = run(X, y, leak, pool, ep, eu, Rb, Bb, seed=300 + LEAKS.index(held), balanced=True)
        res[f"lolo_bal_{held}"] = summarize(realb, nullb)
    print(held, {k: round(v, 3) for k, v in s.items()}, "| bal",
          {k: round(v, 3) for k, v in res.get(f'lolo_bal_{held}', {}).items() if k in ('real', 'null', 'null_sd', 'p')},
          f"{time.time()-t0:.0f}s", flush=True)
json.dump(res, open(OUT, "w"), indent=1, default=float)
