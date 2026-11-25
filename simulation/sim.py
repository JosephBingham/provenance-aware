"""Re-evaluation of the decision rule in simulation (re-implemented from the Array manuscript's description).
Grid: training sources K in {3,6} x rate ratio rho in {1,4,8} x separation Delta in {0,1,2} x alignment theta in {0,180}
x signal delta in {0.25,0.5,1} x target relation in {shared, absent, reversed} = 324 settings.
Per setting: 1 real + B within-source null fits, pooled and source-balanced (B=19). Learner: gradient boosting, 100 trees,
depth 3, 1:3 pseudo-negatives. Usage: python3 sim.py <shard> <nshards> <out.jsonl> [B]"""
import sys, json, itertools, time, numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

SHARD, NSH, OUT = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
B = int(sys.argv[4]) if len(sys.argv) > 4 else 19
D, N, P0, LAB = 8, 20000, 0.03, 0.5
GRID = list(itertools.product([3, 6], [1, 4, 8], [0, 1, 2], [0, 180], [0.25, 0.5, 1.0], ["shared", "absent", "reversed"]))


def gen(K, rho, Delta, theta, delta, rel, rng):
    w = np.zeros(D); w[0] = np.cos(np.radians(theta))          # signal axis: along H (=source 1) at 0, against at 180
    X, lab, src = [], [], []
    mus = [np.zeros(D)] + [Delta * np.eye(D)[k] for k in range(K)]   # 0 = held-out target T, 1 = H, 2.. = others
    for s, mu in enumerate(mus):
        p = P0 * (rho if s == 1 else 1)
        y = rng.random(N) < p
        shift = delta * w if s > 0 else {"shared": delta * w, "absent": 0 * w, "reversed": -delta * w}[rel]
        x = mu + rng.standard_normal((N, D)) + np.outer(y, shift)
        X.append(x); lab.append(y & (rng.random(N) < LAB)); src.append(np.full(N, s))
    return np.vstack(X).astype(np.float32), np.concatenate(lab).astype(int), np.concatenate(src), mus


def pop_reference(X, lab, src, mus, K, rho):
    """AUC on the target of the population score r(x) = sum_s pi_s P(s|x) over training sources."""
    t = src == 0; x = X[t]
    pis = np.array([P0 * (rho if s == 1 else 1) * LAB for s in range(1, K + 1)])
    ll = np.stack([-0.5 * ((x - mus[s]) ** 2).sum(1) for s in range(1, K + 1)], 1)
    ll -= ll.max(1, keepdims=True); post = np.exp(ll); post /= post.sum(1, keepdims=True)
    return roc_auc_score(lab[t], post @ pis)


def fit_score(X, ylab, src, rng, balanced):
    tr = np.where(src > 0)[0]
    P, U = tr[ylab[tr] == 1], tr[ylab[tr] == 0]
    if balanced:
        m = min((src[P] == s).sum() for s in np.unique(src[tr]))
        P = np.concatenate([rng.choice(P[src[P] == s], m, replace=False) for s in np.unique(src[tr])])
        Nn = np.concatenate([rng.choice(U[src[U] == s], 3 * m, replace=False) for s in np.unique(src[tr])])
    else:
        Nn = rng.choice(U, 3 * len(P), replace=False)
    idx = np.r_[P, Nn]; t = np.r_[np.ones(len(P)), np.zeros(len(Nn))]
    mdl = HistGradientBoostingClassifier(max_iter=100, max_depth=3, learning_rate=0.1, early_stopping=False,
                                         random_state=int(rng.integers(1 << 30))).fit(X[idx], t)
    te = src == 0
    return roc_auc_score(ylab[te], mdl.predict_proba(X[te])[:, 1])


def permute_within(ylab, src, rng):
    y2 = ylab.copy()
    for s in np.unique(src[src > 0]):
        ii = np.where(src == s)[0]; y2[ii] = ylab[ii][rng.permutation(len(ii))]
    return y2


done = set()
try:
    done = {json.loads(l)["i"] for l in open(OUT)}
except FileNotFoundError:
    pass
with open(OUT, "a") as f:
    for i, (K, rho, Delta, theta, delta, rel) in enumerate(GRID):
        if i % NSH != SHARD or i in done:
            continue
        t0 = time.time(); rng = np.random.default_rng(1000 + i)
        X, lab, src, mus = gen(K, rho, Delta, theta, delta, rel, rng)
        rec = dict(i=i, K=K, rho=rho, Delta=Delta, theta=theta, delta=delta, rel=rel,
                   pop_ref=pop_reference(X, lab, src, mus, K, rho))
        for bal in [False, True]:
            key = "bal" if bal else "pool"
            rec[f"{key}_real"] = fit_score(X, lab, src, rng, bal)
            rec[f"{key}_null"] = [fit_score(X, permute_within(lab, src, rng), src, rng, bal) for _ in range(B)]
        rec["sec"] = time.time() - t0
        f.write(json.dumps(rec) + "\n"); f.flush()
