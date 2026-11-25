"""OpenSanctions (E2) helpers on top of the ICIJ protocol (evallib). The pipeline under test is unchanged; only the
diagnostic source classifier adds L2 regularisation (1.0): with near-separable sources the unregularised multiclass
booster diverged (training accuracy 0.70 instead of 0.998)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_os import WORK
import sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "icij"))
from evallib import *
from sklearn.ensemble import HistGradientBoostingClassifier
SOURCES = ["ext_ru_egrul", "us_sam_exclusions", "wikidata", "gb_coh_disqualified", "ru_acf_bribetakers"]
SHORT = {"ext_ru_egrul": "EGRUL", "us_sam_exclusions": "SAM", "wikidata": "Wikidata", "gb_coh_disqualified": "UK disq.",
         "ru_acf_bribetakers": "ACF"}

def src_clf(seed):
    return HistGradientBoostingClassifier(max_iter=200, max_depth=3, learning_rate=0.1, l2_regularization=1.0,
                                          early_stopping=False, random_state=seed)

def source_posteriors_r(X, leak, train_idx, apply_idx, rng, per_source=20000):
    srcs = np.unique(leak[train_idx]); samp = []; w = {}
    for s in srcs:
        ii = train_idx[leak[train_idx] == s]; k = min(per_source, len(ii))
        samp.append(rng.choice(ii, k, replace=False)); w[s] = len(ii) / k
    samp = np.concatenate(samp)
    m = src_clf(int(rng.integers(1 << 30))).fit(X[samp], leak[samp])
    pr = m.predict_proba(X[apply_idx]) * np.array([w[c] for c in m.classes_]); pr /= pr.sum(1, keepdims=True)
    return m.classes_, pr

def source_accuracy_r(X, leak, rng, n=80000):
    idx = rng.choice(len(leak), n, replace=False); a, b = idx[: n // 2], idx[n // 2:]
    m = src_clf(1).fit(X[a], leak[a]); return float((m.predict(X[b]) == leak[b]).mean())

def gmd(v):
    v = np.sort(v); k = len(v); i = np.arange(1, k + 1); return 2 * np.sum((2 * i - k - 1) * v) / (k * (k - 1))

def theorem2_r(X, y, leak, rng, n=60000):
    allidx = np.arange(len(y))
    cls, pr = source_posteriors_r(X, leak, allidx, rng.choice(allidx, n, replace=False), rng)
    pi = np.array([y[leak == c].mean() for c in cls]); r = pr @ pi; pib = y.mean()
    return float(0.5 + gmd(r) / (4 * pib * (1 - pib)))

def source_only_r(X, y, leak, pool, ep, eu, rng):
    cls, pr = source_posteriors_r(X, leak, pool, np.concatenate([ep, eu]), rng)
    pi = np.array([y[pool][leak[pool] == c].mean() for c in cls])
    return float(roc_auc_score(np.r_[np.ones(len(ep)), np.zeros(len(eu))], pr @ pi))

def load(rep):
    d = pd.read_parquet(WORK + "/persons.parquet")
    sc = open(WORK + "/struct_cols.txt").read().split("\n")
    tc = [c for c in d.columns if c.startswith("topic_")]
    if rep == "struct": X = d[sc].values
    elif rep == "struct_topics": X = d[sc + tc].values
    elif rep == "topics": X = d[tc].values
    else: X = np.load(f"{WORK}/emb_{rep}.npy")
    return X.astype(np.float32), d.y.values.astype(int), d.src.values
