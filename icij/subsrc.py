"""ICIJ: sensitivity of the reference and of the held-out readings to a finer source partition (response to the second
review, 4.2). Sub-sources = officer-node source tags (e.g. 'Paradise Papers - Appleby'), nested in the four leaks.
In-corpus: reference (15 draws) and Theorem 2 plug-in with sub-source strata. Held out (eight features): pooled reference
with training labels permuted within sub-sources (39 draws); training-label test of the source-balanced pipeline (balanced
by leak as before) with permutations within sub-sources (199 draws), combined with the held-out-label test of the same
first fit (results/heldout_assoc_icij.json). Usage: python3 subsrc.py <out.json>"""
import sys, json, numpy as np, pandas as pd
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import WORK, RESULTS, ICIJ
from evallib import *
OUT = sys.argv[1]
df = pd.read_parquet(WORK + "/officers.parquet"); y = df.y.values.astype(int); leak = df.leak.values
tag = pd.read_csv(ICIJ + "/nodes-officers.csv", dtype=str, keep_default_na=False, usecols=["node_id", "sourceID"]).set_index("node_id").sourceID
t = df.officer.astype(str).map(tag).fillna("")
tv = t.values.astype(str); lv = np.asarray(leak).astype(str)
sub = np.array([a if a.startswith(b) else b for a, b in zip(tv, lv)], dtype=object)
F8 = ["ent_deg", "log_ent_deg", "n_addr", "n_inter", "n_jur", "role_div", "ent_per_addr", "ent_per_inter"]
X = df[F8].values.astype(np.float32)
res = {"strata": {s: {"n": int((sub == s).sum()), "pos": int(y[sub == s].sum())} for s in np.unique(sub)}}
print(json.dumps(res["strata"], indent=0), flush=True)
rng = np.random.default_rng(7)
res["source_acc_sub"] = float(source_accuracy(X, sub, rng)); res["thm2_sub"] = float(theorem3(X, y, sub, rng))
real, null = run_incorpus(X, y, sub, 5, 15, seed=100, matched=False)
res["incorpus_sub"] = {"real": float(real[:, 0].mean()), "ref": float(null[:, 0].mean()), "ref_sd": float(null[:, 0].std(ddof=1))}
print("in-corpus sub-sources", res["incorpus_sub"], "thm2", res["thm2_sub"], "acc", res["source_acc_sub"], flush=True)
json.dump(res, open(OUT, "w"), indent=1)
HA = json.load(open(RESULTS + "/heldout_assoc_icij.json")); B99 = json.load(open(RESULTS + "/full8_bal99.json"))
for k, L in enumerate(LEAKS):
    pool, ep, eu = lolo_sets(y, leak, L, np.random.default_rng(13))
    g = np.random.default_rng(900 + k)
    pooled = np.array([fit_eval(X, permute_within(y, sub, pool, g), leak, pool, ep, eu, g)[0] for _ in range(39)])
    bal = np.array([fit_eval(X, permute_within(y, sub, pool, g), leak, pool, ep, eu, g, True)[0] for _ in range(199)])
    a0 = B99[L]["real_draws"][0]; f0 = HA[L]["fits"][0]
    pu, pd_ = (1 + (bal >= a0).sum()) / 200, (1 + (bal <= a0).sum()) / 200
    comb = min(1, 2 * min(max(pu, f0["p_up"]), max(pd_, f0["p_down"])))
    res[L] = {"pooled_ref": float(pooled.mean()), "pooled_lo": float(np.quantile(pooled, .025)), "pooled_hi": float(np.quantile(pooled, .975)),
              "bal_null_mean": float(bal.mean()), "bal_q025": float(np.quantile(bal, .025)), "bal_q975": float(np.quantile(bal, .975)),
              "bal_first": a0, "pT_up": float(pu), "pT_down": float(pd_), "p2": float(comb), "bal_null": bal.tolist(), "pooled_null": pooled.tolist()}
    print(L, {k_: (round(v, 3) if isinstance(v, float) else None) for k_, v in res[L].items() if not isinstance(v, list)}, flush=True)
    json.dump(res, open(OUT, "w"), indent=1)
p2 = [res[L]["p2"] for L in LEAKS]; o = np.argsort(p2); adj = np.empty(4); run_ = 0
for j, i in enumerate(o): run_ = max(run_, min(1, (4 - j) * p2[i])); adj[i] = run_
for L, a in zip(LEAKS, adj): res[L]["holm"] = float(a)
print("Holm", dict(zip(LEAKS, np.round(adj, 3))), flush=True); json.dump(res, open(OUT, "w"), indent=1)
