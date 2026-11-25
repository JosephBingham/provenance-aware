"""Evaluation protocol of the manuscript: real fits, within-source permutation nulls,
source-balanced training, source-only plug-in baseline, Theorem 3 closed form."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

LEAKS = ["Panama Papers", "Paradise Papers", "Offshore Leaks", "Bahamas Leaks"]


def gbm(seed):
    return HistGradientBoostingClassifier(max_iter=200, max_depth=3, learning_rate=0.1,
                                          early_stopping=False, random_state=seed)


def permute_within(y, groups, idx, rng):
    """Return copy of y with labels permuted within each group, restricted to positions idx."""
    y2 = y.copy()
    for g in np.unique(groups[idx]):
        ii = idx[groups[idx] == g]
        y2[ii] = y[ii][rng.permutation(len(ii))]
    return y2


def train_set(ylab, leak, pool, rng, balanced=False, ratio=3, matched=False):
    """Positives + pseudo-negatives (unlabelled) from pool indices."""
    P = pool[ylab[pool] == 1]; U = pool[ylab[pool] == 0]
    if matched:
        neg = np.concatenate([rng.choice(U[leak[U] == s], ratio * (leak[P] == s).sum(), replace=False)
                              for s in np.unique(leak[P])])
        return P, neg
    if not balanced:
        neg = rng.choice(U, size=min(len(U), ratio * len(P)), replace=False)
        return P, neg
    srcs = np.unique(leak[pool])
    m = min((leak[P] == s).sum() for s in srcs)
    ps, ns = [], []
    for s in srcs:
        Ps = P[leak[P] == s]; Us = U[leak[U] == s]
        ps.append(rng.choice(Ps, m, replace=False)); ns.append(rng.choice(Us, ratio * m, replace=False))
    return np.concatenate(ps), np.concatenate(ns)


def fit_eval(X, ylab, leak, pool, ev_pos, ev_unl, rng, balanced=False, fitter=None, matched=False):
    P, N = train_set(ylab, leak, pool, rng, balanced, matched=matched)
    tr = np.concatenate([P, N]); t = np.r_[np.ones(len(P)), np.zeros(len(N))]
    seed = int(rng.integers(1 << 30))
    if fitter is None:
        m = gbm(seed).fit(X[tr], t)
        ev = np.concatenate([ev_pos, ev_unl])
        s = m.predict_proba(X[ev])[:, 1]
    else:
        ev = np.concatenate([ev_pos, ev_unl])
        s = fitter(tr, t, ev, seed)
    e = np.r_[np.ones(len(ev_pos)), np.zeros(len(ev_unl))]
    return roc_auc_score(e, s), average_precision_score(e, s)


def lolo_sets(y, leak, held, rng_eval, max_unl=40000):
    ev = np.where(leak == held)[0]
    pos = ev[y[ev] == 1]; unl = ev[y[ev] == 0]
    if len(unl) > max_unl:
        unl = rng_eval.choice(unl, max_unl, replace=False)
    pool = np.where(leak != held)[0]
    return pool, pos, unl


def incorpus_sets(y, leak, rng_eval, matched=False, n_unl=20000, test_frac=0.3):
    P = np.where(y == 1)[0]; U = np.where(y == 0)[0]
    P = rng_eval.permutation(P); nt = int(test_frac * len(P))
    test_pos = P[:nt]
    if not matched:
        unl = rng_eval.choice(U, n_unl, replace=False)
    else:
        parts = []
        for s in np.unique(leak):
            k = int(round(n_unl * (leak[test_pos] == s).mean()))
            parts.append(rng_eval.choice(U[leak[U] == s], k, replace=False))
        unl = np.concatenate(parts)
    mask = np.ones(len(y), bool); mask[test_pos] = False; mask[unl] = False
    return np.where(mask)[0], test_pos, unl


def run(X, y, leak, pool, ev_pos, ev_unl, R, B, seed, balanced=False, fitter=None, matched=False):
    """R real fits and B within-source permutation fits. Returns arrays (auc, ap)."""
    rng = np.random.default_rng(seed)
    real = np.array([fit_eval(X, y, leak, pool, ev_pos, ev_unl, rng, balanced, fitter, matched) for _ in range(R)])
    null = []
    for _ in range(B):
        yp = permute_within(y, leak, pool, rng)
        null.append(fit_eval(X, yp, leak, pool, ev_pos, ev_unl, rng, balanced, fitter, matched))
    return real, np.array(null).reshape(-1, 2)


def run_incorpus(X, y, leak, R, B, seed, matched, fitter=None, perm_eval=True):
    """In-corpus: real fits on a fixed 70/30 split; null permutes labels within leak over ALL officers
    (perm_eval=True: evaluation labels permuted too, as Theorem 3 assumes) or only over the training pool."""
    pool, ep, eu = incorpus_sets(y, leak, np.random.default_rng(seed), matched=matched)
    rng = np.random.default_rng(seed + 1)
    real = np.array([fit_eval(X, y, leak, pool, ep, eu, rng, False, fitter, matched) for _ in range(R)])
    null = []
    for b in range(B):
        if perm_eval:
            yp = permute_within(y, leak, np.arange(len(y)), rng)
            pl, p2, u2 = incorpus_sets(yp, leak, np.random.default_rng(seed + 1000 + b), matched=matched)
            null.append(fit_eval(X, yp, leak, pl, p2, u2, rng, False, fitter, matched))
        else:
            yp = permute_within(y, leak, pool, rng)
            null.append(fit_eval(X, yp, leak, pool, ep, eu, rng, False, fitter, matched))
    return real, np.array(null).reshape(-1, 2)


def source_posteriors(X, leak, train_idx, apply_idx, rng, per_source=20000):
    """Source classifier trained on up to per_source nodes per source, re-weighted to pool composition."""
    srcs = np.unique(leak[train_idx]); samp = []; w = {}
    for s in srcs:
        ii = train_idx[leak[train_idx] == s]
        k = min(per_source, len(ii)); samp.append(rng.choice(ii, k, replace=False)); w[s] = len(ii) / k
    samp = np.concatenate(samp)
    m = gbm(int(rng.integers(1 << 30))).fit(X[samp], leak[samp])
    pr = m.predict_proba(X[apply_idx]) * np.array([w[c] for c in m.classes_])
    pr /= pr.sum(1, keepdims=True)
    return m.classes_, pr


def source_accuracy(X, leak, rng, n=80000):
    """Balanced 4-way accuracy of predicting the leak from X (held-out half)."""
    idx = rng.choice(len(leak), n, replace=False); a, b = idx[: n // 2], idx[n // 2:]
    m = gbm(1).fit(X[a], leak[a]); return (m.predict(X[b]) == leak[b]).mean()


def theorem3(X, y, leak, rng, n=60000):
    """Closed-form in-corpus reference: 1/2 + E|r-r'| / (4 pibar (1-pibar)), plug-in."""
    allidx = np.arange(len(y))
    cls, pr = source_posteriors(X, leak, allidx, rng.choice(allidx, n, replace=False), rng)
    pi = np.array([y[leak == c].mean() for c in cls]); r = pr @ pi
    rs = np.sort(r); k = len(rs); i = np.arange(1, k + 1)
    gmd = 2 * np.sum((2 * i - k - 1) * rs) / (k * (k - 1))
    pib = y.mean()
    return 0.5 + gmd / (4 * pib * (1 - pib))


def source_only_heldout(X, y, leak, pool, ev_pos, ev_unl, rng):
    cls, pr = source_posteriors(X, leak, pool, np.concatenate([ev_pos, ev_unl]), rng)
    pi = np.array([y[pool][leak[pool] == c].mean() for c in cls]); r = pr @ pi
    e = np.r_[np.ones(len(ev_pos)), np.zeros(len(ev_unl))]
    return roc_auc_score(e, r)


def summarize(real, null):
    ra, na = real[:, 0], null[:, 0]
    p = (1 + (na >= ra[0]).sum()) / (1 + len(na))
    return dict(real=ra.mean(), real_sd=ra.std(ddof=1) if len(ra) > 1 else 0, null=na.mean(), null_sd=na.std(ddof=1),
                null_lo=np.quantile(na, .025), null_hi=np.quantile(na, .975), diff=ra.mean() - na.mean(), p=p,
                real_ap=real[:, 1].mean(), null_ap=null[:, 1].mean(), null_ap_sd=null[:, 1].std(ddof=1), B=len(na))
