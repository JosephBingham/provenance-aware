"""Functional nominee label (ICIJ 'underlying' Nominee ... of edges, start node; Offshore Leaks only).
(a) Pooled in-corpus evaluation of a label recorded by one source: real vs within-source reference vs Theorem 3,
    uniform and source-matched unlabelled samples (8 structural features).
(b) Replication of the Array checks: name-trained model and raw degree on the name-independent functional nominees.
Usage: python3 funclabel.py <out.json>"""
import sys, json, numpy as np, pandas as pd
from config import WORK, ICIJ
from evallib import *
from sklearn.metrics import roc_auc_score
OUT = sys.argv[1]
rel = pd.read_csv(ICIJ + "/relationships.csv", dtype=str, keep_default_na=False,
                  usecols=["node_id_start", "rel_type", "link", "sourceID"])
nom = rel[(rel.rel_type == "underlying") & rel.link.str.contains("nominee", case=False) & (rel.sourceID != "")]
df = pd.read_parquet(WORK + "/officers.parquet")
fn = df.officer.isin(set(nom.node_id_start)).values.astype(int)
leak = df.leak.values; yname = df.y.values
F8 = ["ent_deg", "log_ent_deg", "n_addr", "n_inter", "n_jur", "role_div", "ent_per_addr", "ent_per_inter"]
X = df[F8].values.astype(np.float32)
res = {"n_functional": int(fn.sum()), "n_functional_not_name": int((fn & (1 - yname)).sum()),
       "by_leak": {L: int(fn[leak == L].sum()) for L in LEAKS}}
print(res, flush=True)
rng = np.random.default_rng(3)
res["source_acc"] = float(source_accuracy(X, leak, rng)); res["thm3_pred"] = float(theorem3(X, fn, leak, rng))
# Offshore Leaks vs rest: how well do the features identify the one labelling source?
for matched in [False, True]:
    real, null = run_incorpus(X, fn, leak, 10, 19, seed=500 + matched, matched=matched)
    k = "incorpus_matched" if matched else "incorpus_uniform"
    res[k] = summarize(real, null); res[k]["null_draws"] = null[:, 0].tolist(); res[k]["real_draws"] = real[:, 0].tolist()
    print(k, {a: round(b, 3) for a, b in res[k].items() if not isinstance(b, list)}, flush=True)
print("thm3", round(res["thm3_pred"], 3), "source acc", round(res["source_acc"], 3), flush=True)
# (b) name-trained model (pooled, all leaks, 1:3 pseudo-negatives) on name-independent functional nominees in Offshore Leaks
ol = np.where(leak == "Offshore Leaks")[0]
pos = ol[(fn[ol] == 1) & (yname[ol] == 0)]; unl = ol[(fn[ol] == 0)]
P = np.where(yname == 1)[0]; P = P[~np.isin(P, pos)]
Uall = np.setdiff1d(np.where(yname == 0)[0], np.r_[pos, unl])
sc = []
for s in range(5):
    r2 = np.random.default_rng(10 + s); N = r2.choice(Uall, 3 * len(P), replace=False)
    m = gbm(s).fit(X[np.r_[P, N]], np.r_[np.ones(len(P)), np.zeros(len(N))])
    sc.append(roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(unl))], m.predict_proba(X[np.r_[pos, unl]])[:, 1]))
res["name_model_on_functional"] = float(np.mean(sc))
res["raw_degree_on_functional"] = float(roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(unl))], df.ent_deg.values[np.r_[pos, unl]]))
print("name-trained model on name-independent functional nominees:", round(res["name_model_on_functional"], 3),
      "| raw entity degree:", round(res["raw_degree_on_functional"], 3), "| n pos", len(pos), flush=True)
json.dump(res, open(OUT, "w"), indent=1, default=float)
