"""Aggregate test over the ten source-balanced fits (response to the third review, M2).
Statistic: the median held-out AUC of K=10 source-balanced fits, the summary already reported in Table 4.
Training-label test: for each of B within-leak permutations of the training labels, K fresh balanced fits on that same
permuted labelling, median AUC (valid by Theorem 1: K i.i.d. pipeline runs per labelling are one pipeline run).
Held-out-label test: the ten real fits are re-created (same seeds as bal99.py / heldout_assoc.py), their held-out scores
fixed, and one permutation of the held-out labels is applied to all ten before taking the median (valid under H0_A).
Writes partial results every 25 draws. Usage: python3 agg_fits.py <shard> <nshards> <B> <outdir>"""
import sys, os, json, numpy as np, pandas as pd
from scipy.stats import rankdata
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from config import WORK
from evallib import train_set, gbm, lolo_sets, permute_within, fit_eval, LEAKS
shard, nsh, B, OUT = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
K, BH = 10, 9999
df = pd.read_parquet(WORK + "/officers.parquet"); y = df.y.values; leak = df.leak.values
F8 = ["ent_deg", "log_ent_deg", "n_addr", "n_inter", "n_jur", "role_div", "ent_per_addr", "ent_per_inter"]
X = df[F8].values.astype(np.float32)
for k, L in enumerate(LEAKS):
    if k % nsh != shard: continue
    pool, ep, eu = lolo_sets(y, leak, L, np.random.default_rng(13))
    ev = np.concatenate([ep, eu]); n1, n0 = len(ep), len(eu)
    # re-create the ten real fits (same rng stream as bal99.py's real fits)
    rng = np.random.default_rng(400 + k); ranks, aucs = [], []
    for r in range(K):
        P, N = train_set(y, leak, pool, rng, True)
        tr = np.concatenate([P, N]); t = np.r_[np.ones(len(P)), np.zeros(len(N))]
        s = gbm(int(rng.integers(1 << 30))).fit(X[tr], t).predict_proba(X[ev])[:, 1]
        rk = rankdata(s); ranks.append(rk); aucs.append((rk[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    ranks = np.array(ranks); real_med = float(np.median(aucs))
    prng = np.random.default_rng(888 + k); held = np.empty(BH)
    for b in range(BH):
        idx = prng.choice(n1 + n0, n1, replace=False)
        held[b] = np.median((ranks[:, idx].sum(1) - n1 * (n1 + 1) / 2) / (n1 * n0))
    pA_up = (1 + (held >= real_med).sum()) / (BH + 1); pA_dn = (1 + (held <= real_med).sum()) / (BH + 1)
    # training-label null: K fresh balanced fits per within-leak permutation
    rng = np.random.default_rng(9000 + k); null = []
    res = dict(leak=L, K=K, real_aucs=[float(a) for a in aucs], real_median=real_med,
               heldout=dict(BH=BH, p_up=float(pA_up), p_down=float(pA_dn), q025=float(np.quantile(held, .025)), q975=float(np.quantile(held, .975))))
    for b in range(B):
        yp = permute_within(y, leak, pool, rng)
        null.append(float(np.median([fit_eval(X, yp, leak, pool, ep, eu, rng, True)[0] for _ in range(K)])))
        if (b + 1) % 25 == 0 or b + 1 == B:
            a = np.array(null); res["training"] = dict(B=len(null), null=null,
                p_up=float((1 + (a >= real_med).sum()) / (len(a) + 1)), p_down=float((1 + (a <= real_med).sum()) / (len(a) + 1)))
            json.dump(res, open(f"{OUT}/agg_fits_{k}.json", "w"))
            print(L, b + 1, "median real %.3f null mean %.3f q97.5 %.3f p_up %.4f" % (real_med, a.mean(), np.quantile(a, .975), res["training"]["p_up"]), flush=True)
