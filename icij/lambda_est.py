"""Estimates of lambda = sup_{x,s} P(l=1 | x, s) for Theorem 4(c) (response to review 4.4, Q3). Cross-fitted
(2-fold) gradient boosting on the representation plus a one-hot source indicator; reports high quantiles of the
out-of-fold probabilities, their calibration in the top percentile, and the implied factor (1-lambda)^-2.
Usage: python3 lambda_est.py icij|os <out.json>"""
import os, sys, json, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "opensanctions"))
from sklearn.ensemble import HistGradientBoostingClassifier
W, OUT = sys.argv[1], sys.argv[2]
if W == "icij":
    df = pd.read_parquet(__import__("config").WORK + "/officers.parquet")
    F8 = ["ent_deg", "log_ent_deg", "n_addr", "n_inter", "n_jur", "role_div", "ent_per_addr", "ent_per_inter"]
    X = df[F8].values; y = df.y.values.astype(int); s = df.leak.values
else:
    os.environ["OS_COHORT"] = "cells" if W == "os2" else "first"; from e2lib import load
    X, y, s = load("struct")
s = np.asarray(s, dtype=object); S = np.unique(s); Xs = np.c_[X, (s[:, None] == S[None, :]).astype(float)].astype(np.float32)
rng = np.random.default_rng(3); fold = rng.integers(0, 2, len(y)); p = np.zeros(len(y))
for k in range(2):
    m = HistGradientBoostingClassifier(max_iter=200, max_depth=3, learning_rate=0.1, early_stopping=False,
                                       random_state=k).fit(Xs[fold != k], y[fold != k])
    p[fold == k] = m.predict_proba(Xs[fold == k])[:, 1]
qs = {str(q): float(np.quantile(p, q)) for q in [0.5, 0.9, 0.99, 0.999]}
top = p >= np.quantile(p, 0.99)
res = dict(quantiles=qs, max=float(p.max()), top1pct_mean_pred=float(p[top].mean()), top1pct_obs_rate=float(y[top].mean()),
           per_source_max={str(c): float(p[s == c].max()) for c in S},
           per_source_q999={str(c): float(np.quantile(p[s == c], 0.999)) for c in S})
for key, lam in [("q999", qs["0.999"]), ("max", res["max"])]:
    res[f"factor_{key}"] = float((1 - lam) ** -2) if lam < 1 else float("inf")
json.dump(res, open(OUT, "w"), indent=1); print(json.dumps(res, indent=1))
