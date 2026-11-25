"""Source-balanced leave-one-leak-out readings for TransE and RDF2Vec on ICIJ with enough draws for a two-sided
Holm-adjusted test (response to review, 5.2): 5 real balanced fits, 199 balanced within-source permutation draws, and the
held-out-label permutation test (9,999 draws) of every real fit. Usage: python3 learned_bal.py <emb.npy> <out.json>"""
import sys, json, numpy as np, pandas as pd
from scipy.stats import rankdata
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import WORK
from evallib import train_set, gbm, lolo_sets, permute_within, LEAKS
EMB, OUT = sys.argv[1], sys.argv[2]
df = pd.read_parquet(WORK + "/officers.parquet"); y = df.y.values.astype(int); leak = df.leak.values
X = np.load(EMB).astype(np.float32); assert len(X) == len(y)
R, B, BH = 5, 199, 9999; out = {}

def fit_scores(ylab, pool, ev, rng):
    P, N = train_set(ylab, leak, pool, rng, True)
    tr = np.concatenate([P, N]); t = np.r_[np.ones(len(P)), np.zeros(len(N))]
    return gbm(int(rng.integers(1 << 30))).fit(X[tr], t).predict_proba(X[ev])[:, 1]

def auc(rk, n1, n0, idx=None):
    s = rk[:n1].sum() if idx is None else rk[idx].sum()
    return (s - n1 * (n1 + 1) / 2) / (n1 * n0)

for k, L in enumerate(LEAKS):
    pool, ep, eu = lolo_sets(y, leak, L, np.random.default_rng(13)); ev = np.concatenate([ep, eu]); n1, n0 = len(ep), len(eu)
    rng = np.random.default_rng(600 + k); fits = []
    for r in range(R):
        rk = rankdata(fit_scores(y, pool, ev, rng)); a = auc(rk, n1, n0)
        prng = np.random.default_rng(888 + r)
        held = np.array([auc(rk, n1, n0, prng.choice(n1 + n0, n1, replace=False)) for _ in range(BH)])
        fits.append(dict(auc=float(a), p_up=float((1 + (held >= a).sum()) / (BH + 1)), p_down=float((1 + (held <= a).sum()) / (BH + 1))))
    null = []
    for b in range(B):
        yp = permute_within(y, leak, pool, rng)
        null.append(float(auc(rankdata(fit_scores(yp, pool, ev, rng)), n1, n0)))
    nl = np.array(null); a0 = fits[0]["auc"]
    pu = (1 + (nl >= a0).sum()) / (B + 1); pdn = (1 + (nl <= a0).sum()) / (B + 1)
    out[L] = dict(fits=fits, null=null, real_first=a0, real_median=float(np.median([f["auc"] for f in fits])),
                  null_mean=float(nl.mean()), null_q025=float(np.quantile(nl, .025)), null_q975=float(np.quantile(nl, .975)),
                  p_train_up=float(pu), p_train_down=float(pdn))
    print(L, "first %.3f median %.3f | null mean %.3f [%.3f, %.3f] | p_train up %.3f down %.3f | p_held up %.4f down %.4f"
          % (a0, out[L]["real_median"], nl.mean(), out[L]["null_q025"], out[L]["null_q975"], pu, pdn, fits[0]["p_up"], fits[0]["p_down"]), flush=True)
    json.dump(out, open(OUT, "w"), indent=1)
