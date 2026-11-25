"""Held-out-label permutation test for fixed source-balanced models (response to review, point 4.1).
Re-creates the real source-balanced fits of the original runs (same seeds, so the AUCs match the recorded ones), keeps
each fitted model's scores on the held-out source fixed, and permutes the held-out source's labels (9,999 draws).
Under H0_A (held-out labels exchangeable given the fitted scores, e.g. no signal in the held-out source) this test is
exact whatever the training sources contain. Usage: python3 heldout_assoc.py icij|os|os2 <out.json>   (os = plan's first cohort, os2 = count-invariant cohort)"""
import sys, json, numpy as np, pandas as pd
from scipy.stats import rankdata
import os
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "opensanctions"))
from evallib import train_set, gbm, lolo_sets, LEAKS
WHICH, OUT = sys.argv[1], sys.argv[2]
if WHICH == "icij":
    df = pd.read_parquet(__import__("config").WORK + "/officers.parquet"); y = df.y.values; src = df.leak.values
    F8 = ["ent_deg", "log_ent_deg", "n_addr", "n_inter", "n_jur", "role_div", "ent_per_addr", "ent_per_inter"]
    X = df[F8].values.astype(np.float32); SRC = LEAKS; seed = lambda k: 400 + k; R = 10
    rec = json.load(open(__import__("config").RESULTS + "/full8_bal99.json")); recorded = lambda L: rec[L]["real_draws"]
else:
    os.environ["OS_COHORT"] = "cells" if WHICH == "os2" else "first"
    from e2lib import load, SOURCES
    X, y, src = load("struct"); SRC = SOURCES; seed = lambda k: 300 + k; R = 10
    rec = json.load(open(__import__("config_os").RESULTS + "/struct.json")); recorded = lambda L: rec[f"lolo_bal_{L}"]["real_draws"]
BH = 9999; out = {}
for k, L in enumerate(SRC):
    pool, ep, eu = lolo_sets(y, src, L, np.random.default_rng(13))
    rng = np.random.default_rng(seed(k)); ev = np.concatenate([ep, eu]); n1, n0 = len(ep), len(eu)
    fits = []
    for r in range(R):
        P, N = train_set(y, src, pool, rng, True)
        tr = np.concatenate([P, N]); t = np.r_[np.ones(len(P)), np.zeros(len(N))]
        s = gbm(int(rng.integers(1 << 30))).fit(X[tr], t).predict_proba(X[ev])[:, 1]
        rk = rankdata(s); auc = (rk[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
        prng = np.random.default_rng(777 + r)
        held = np.array([(rk[prng.choice(n1 + n0, n1, replace=False)].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) for _ in range(BH)])
        pu = (1 + (held >= auc).sum()) / (BH + 1); pd_ = (1 + (held <= auc).sum()) / (BH + 1)
        fits.append(dict(auc=float(auc), p_up=float(pu), p_down=float(pd_), p_two=float(min(1, 2 * min(pu, pd_))),
                         held_q025=float(np.quantile(held, .025)), held_q975=float(np.quantile(held, .975))))
    out[L] = dict(fits=fits, recorded_match=float(max(abs(f["auc"] - a) for f, a in zip(fits, recorded(L)))))
    f0 = fits[0]
    print(L, "first fit AUC %.3f (recorded %.3f) | held-out-label p two-sided %.4f | null 95%% [%.3f, %.3f] | all fits p<=0.025 up: %d/%d"
          % (f0["auc"], recorded(L)[0], f0["p_two"], f0["held_q025"], f0["held_q975"], sum(f["p_up"] <= 0.025 for f in fits), R), flush=True)
json.dump(out, open(OUT, "w"), indent=1)
