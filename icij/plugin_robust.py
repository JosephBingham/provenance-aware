"""Robustness of the Theorem 2 plug-in to the source classifier (response to the second review, 4.6).
For each representation: plug-in reference with several classifiers, fitted in-sample (as reported) or cross-fitted
(two folds: posteriors for a node come from a classifier that did not see it), with held-out log loss and accuracy.
Usage: python3 plugin_robust.py <icij|os|os2> <out.json>   (os = plan's first cohort, os2 = count-invariant cohort)"""
import os, sys, json, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(os.path.dirname(HERE), "opensanctions"))
from config import WORK, RESULTS
import config_os
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import log_loss
W, OUT = sys.argv[1], sys.argv[2]
def hgb(**kw):
    a = dict(max_iter=200, max_depth=3, learning_rate=0.1, early_stopping=False, random_state=1); a.update(kw)
    return lambda: HistGradientBoostingClassifier(**a)
CLFS = {"boosting (as reported)": hgb(), "boosting, L2 1": hgb(l2_regularization=1.0), "boosting, L2 10": hgb(l2_regularization=10.0),
        "boosting, depth 6, 400 trees": hgb(max_depth=6, max_iter=400),
        "logistic regression": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))}
def gmd(v):
    v = np.sort(v); k = len(v); i = np.arange(1, k + 1); return 2 * np.sum((2 * i - k - 1) * v) / (k * (k - 1))
def plugin(X, y, s, make, rng, cross, per=20000, napply=60000):
    S = np.unique(s); pi = np.array([y[s == c].mean() for c in S]); pib = y.mean(); n = len(y)
    fold = rng.integers(0, 2, n) if cross else np.zeros(n, int)
    rs, ll, acc = [], [], []
    for k in ([0, 1] if cross else [0]):
        tr_pool = np.where(fold != k)[0] if cross else np.arange(n); ap_pool = np.where(fold == k)[0] if cross else np.arange(n)
        samp, w = [], []
        for c in S:
            ii = tr_pool[s[tr_pool] == c]; m_ = min(per, len(ii)); samp.append(rng.choice(ii, m_, replace=False)); w.append(len(ii) / m_)
        samp = np.concatenate(samp); Xt = np.log1p(np.clip(X[samp], 0, None)) if LOG else X[samp]
        m = make().fit(Xt, s[samp]); cls = list(m.classes_); assert cls == list(S)
        ap = rng.choice(ap_pool, min(napply // (2 if cross else 1), len(ap_pool)), replace=False)
        Xa = np.log1p(np.clip(X[ap], 0, None)) if LOG else X[ap]
        pr = m.predict_proba(Xa) * np.array(w); pr /= pr.sum(1, keepdims=True)
        rs.append(pr @ pi)
        if cross:
            ll.append(log_loss(s[ap], pr, labels=cls)); acc.append(float((np.array(cls)[pr.argmax(1)] == s[ap]).mean()))
    r = np.concatenate(rs)
    return float(0.5 + gmd(r) / (4 * pib * (1 - pib))), (float(np.mean(ll)) if ll else None), (float(np.mean(acc)) if acc else None)
reps = {}
if W == "icij":
    df = pd.read_parquet(WORK + "/officers.parquet"); y = df.y.values.astype(int); s = df.leak.values.astype(str)
    F8 = ["ent_deg", "log_ent_deg", "n_addr", "n_inter", "n_jur", "role_div", "ent_per_addr", "ent_per_inter"]
    obs = lambda f: json.load(open(f"{RESULTS}/{f}.json"))["incorpus_uniform"]["null"]
    reps = {"Connectivity (3)": (df[["ent_deg", "n_addr", "n_inter"]].values, obs("conn")), "Structural (8)": (df[F8].values, obs("full8")),
            "Structural + graph (12)": (df[F8 + ["pagerank", "kcore", "nbr_deg_mean", "nbr_deg_max"]].values, obs("full12")),
            "TransE": (np.load(WORK + "/emb_transe.npy"), obs("transe")), "RDF2Vec": (np.load(WORK + "/emb_rdf2vec.npy"), obs("rdf2vec"))}
else:
    base = config_os.WORK_FIRST if W == "os" else config_os.WORK_CELLS
    resd = os.path.join(os.path.dirname(RESULTS), "opensanctions_first" if W == "os" else "opensanctions") if False else (config_os._R + ("/opensanctions_first" if W == "os" else "/opensanctions"))
    d = pd.read_parquet(f"{base}/persons.parquet"); sc = open(f"{base}/struct_cols.txt").read().split("\n")
    y = d.y.values.astype(int); s = d.src.values.astype(str)
    reps = {"OpenSanctions structural": (d[sc].values, json.load(open(f"{resd}/struct.json"))["incorpus_uniform"]["null"])}
res = {}
for name, (X, ob) in reps.items():
    X = X.astype(np.float32); res[name] = {"observed": ob}
    for cname, make in CLFS.items():
        LOG = cname == "logistic regression" and name not in ("TransE", "RDF2Vec")
        for cross in [False, True]:
            if cname != "boosting (as reported)" and not cross: continue
            v, ll, acc = plugin(X, y, s, make, np.random.default_rng(11), cross)
            res[name][f"{cname}{', cross-fitted' if cross else ''}"] = {"plugin": v, "logloss": ll, "acc": acc}
            print(name, cname, "cross" if cross else "in-sample", round(v, 4), ll and round(ll, 3), acc and round(acc, 3), "observed", round(ob, 4), flush=True)
    json.dump(res, open(OUT, "w"), indent=1)
