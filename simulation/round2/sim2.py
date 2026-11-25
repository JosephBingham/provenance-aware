"""Decision-rule simulation, second round (response to review): error rates under the no-transfer nulls, with replicates.

Scenarios (target = held-out source at the origin; training sources at distance Delta on axes 0..K-1):
  absent   : training sources share a signal (positives shifted by delta*w), the target has none          (null)
  specific : each training source has its own signal along axis 6 with a random sign per source; the target's
             positives are shifted along axis 7, a direction no training source uses                          (null)
  shared   : the target has the training signal                                                               (power)
  reversed : the target has the opposite signal                                                               (power)

Per replicate: one real source-balanced fit and B balanced within-source permutation fits (the training-permutation
test), plus a held-out-label permutation test of the fixed real fit (labels of the target permuted, scores fixed).
Rules: (old) training-permutation test alone, two-sided at alpha; (IUT) both tests reject at alpha/2 in the same direction.
Usage: python3 sim2.py <shard> <nshards> <out.jsonl>"""
import sys, json, itertools, time, numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from scipy.stats import rankdata

SHARD, NSH, OUT = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
B, BH, D, N, P0, LAB = 39, 1999, 8, 20000, 0.03, 0.5
JOBS = []
for K, rho, Delta, theta, delta in itertools.product([3, 6], [1, 4, 8], [0, 1, 2], [0, 180], [0.25, 0.5, 1.0]):
    JOBS += [("absent", K, rho, Delta, theta, delta, rep) for rep in range(5)]
    JOBS += [("shared", K, rho, Delta, theta, delta, 0), ("reversed", K, rho, Delta, theta, delta, 0)]
for K, rho, Delta, delta in itertools.product([3, 6], [1, 4, 8], [0, 1, 2], [0.5, 1.0]):
    JOBS += [("specific", K, rho, Delta, 0, delta, rep) for rep in range(5)]


def gen(scen, K, rho, Delta, theta, delta, rng):
    w = np.zeros(D); w[0] = np.cos(np.radians(theta))
    signs = rng.choice([-1.0, 1.0], K + 1)
    X, lab, src = [], [], []
    mus = [np.zeros(D)] + [Delta * np.eye(D)[k] for k in range(K)]
    for s, mu in enumerate(mus):
        y = rng.random(N) < P0 * (rho if s == 1 else 1)
        if scen == "specific":
            shift = delta * (np.eye(D)[7] if s == 0 else signs[s] * np.eye(D)[6])
        elif s > 0:
            shift = delta * w
        else:
            shift = {"shared": delta * w, "absent": 0 * w, "reversed": -delta * w}[scen]
        X.append(mu + rng.standard_normal((N, D)) + np.outer(y, shift))
        lab.append(y & (rng.random(N) < LAB)); src.append(np.full(N, s))
    return np.vstack(X).astype(np.float32), np.concatenate(lab).astype(int), np.concatenate(src)


def fit_scores(X, ylab, src, rng):
    tr = np.where(src > 0)[0]; P, U = tr[ylab[tr] == 1], tr[ylab[tr] == 0]
    ss = np.unique(src[tr]); m = min((src[P] == s).sum() for s in ss)
    P = np.concatenate([rng.choice(P[src[P] == s], m, replace=False) for s in ss])
    Nn = np.concatenate([rng.choice(U[src[U] == s], 3 * m, replace=False) for s in ss])
    idx = np.r_[P, Nn]; t = np.r_[np.ones(len(P)), np.zeros(len(Nn))]
    mdl = HistGradientBoostingClassifier(max_iter=100, max_depth=3, learning_rate=0.1, early_stopping=False,
                                         random_state=int(rng.integers(1 << 30))).fit(X[idx], t)
    return mdl.predict_proba(X[src == 0])[:, 1]


def auc_from_ranks(ranks, pos_idx, n1, n0):
    return (ranks[pos_idx].sum(-1) - n1 * (n1 + 1) / 2) / (n1 * n0)


def permute_within(ylab, src, rng):
    y2 = ylab.copy()
    for s in np.unique(src[src > 0]):
        ii = np.where(src == s)[0]; y2[ii] = ylab[ii][rng.permutation(len(ii))]
    return y2


done = set()
try:
    done = {json.loads(l)["j"] for l in open(OUT)}
except FileNotFoundError:
    pass
with open(OUT, "a") as f:
    for j, (scen, K, rho, Delta, theta, delta, rep) in enumerate(JOBS):
        if j % NSH != SHARD or j in done:
            continue
        t0 = time.time(); rng = np.random.default_rng(50000 + j)
        X, lab, src = gen(scen, K, rho, Delta, theta, delta, rng)
        yt = lab[src == 0]; n1 = int(yt.sum()); n0 = len(yt) - n1
        sc = fit_scores(X, lab, src, rng); ranks = rankdata(sc)
        real = auc_from_ranks(ranks, np.where(yt == 1)[0], n1, n0)
        null = [auc_from_ranks(rankdata(fit_scores(X, permute_within(lab, src, rng), src, rng)), np.where(yt == 1)[0], n1, n0)
                for _ in range(B)]
        held = np.array([auc_from_ranks(ranks, rng.choice(len(yt), n1, replace=False), n1, n0) for _ in range(BH)])
        rec = dict(j=j, scen=scen, K=K, rho=rho, Delta=Delta, theta=theta, delta=delta, rep=rep, n1=n1, real=float(real),
                   null=[float(v) for v in null],
                   p_train_up=(1 + sum(v >= real for v in null)) / (B + 1), p_train_dn=(1 + sum(v <= real for v in null)) / (B + 1),
                   p_held_up=float((1 + (held >= real).sum()) / (BH + 1)), p_held_dn=float((1 + (held <= real).sum()) / (BH + 1)),
                   sec=time.time() - t0)
        f.write(json.dumps(rec) + "\n"); f.flush()
