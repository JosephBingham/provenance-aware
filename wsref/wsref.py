"""wsref: the within-source permutation reference for evaluating learning on multi-source data.

Minimal usage
-------------
    import wsref
    # evaluate(labels) must retrain the WHOLE pipeline on `labels` (a full-length 0/1 vector) and return the
    # evaluation statistic (AUC, AP, MRR, ...) computed against the REAL labels of the evaluation set.
    ref = wsref.reference(evaluate, y, sources, B=99, permute_idx=train_idx)
    real = evaluate(y)
    p = wsref.p_value(real, ref.null)                      # one-sided, upper tail
    # transfer: both the training-label test and the held-out-label test of ONE source-balanced fit must agree
    pt = wsref.two_tailed(bal_real, bal_ref.null)                      # training-label test (refits)
    pa = wsref.heldout_label_test(bal_scores_eval, y_eval)             # held-out-label test (no refits)
    call = wsref.read_transfer(pt, pa)                    # 'transfers' | 'reversal' | 'inconclusive'

Only numpy is required. The pipeline is yours; this module never sees features.
"""
from dataclasses import dataclass
import numpy as np

__all__ = ["permute_within", "reference", "p_value", "two_tailed", "heldout_label_test", "read_transfer", "combined_p2",
           "read_balanced", "holm", "min_draws",
           "incorpus_reference_closed_form", "provenance_score", "Reference"]


def permute_within(y, sources, rng, idx=None):
    """Copy of y with labels permuted within each source, only among positions idx (default: all)."""
    y = np.asarray(y); sources = np.asarray(sources)
    idx = np.arange(len(y)) if idx is None else np.asarray(idx)
    out = y.copy()
    for s in np.unique(sources[idx]):
        ii = idx[sources[idx] == s]
        out[ii] = y[ii][rng.permutation(len(ii))]
    return out


@dataclass
class Reference:
    null: np.ndarray

    @property
    def mean(self):
        return float(np.mean(self.null))

    def quantiles(self, q=(0.025, 0.975)):
        return tuple(float(v) for v in np.quantile(self.null, q))


def reference(evaluate, y, sources, B=99, seed=0, permute_idx=None):
    """Within-source permutation reference: B evaluations of the pipeline on within-source permuted labels.
    For held-out evaluation pass permute_idx = training positions. For in-corpus evaluation permute all
    positions (labels are permuted before the train/test split, as Theorem 2 assumes)."""
    rng = np.random.default_rng(seed)
    return Reference(np.array([evaluate(permute_within(y, sources, rng, permute_idx)) for _ in range(B)]))


def p_value(real, null, tail="upper"):
    """Permutation p-value, (1 + #{null at least as extreme}) / (1 + B). Valid for any statistic (Theorem 1)."""
    null = np.asarray(null)
    k = (null >= real).sum() if tail == "upper" else (null <= real).sum()
    return (1 + int(k)) / (1 + len(null))


def min_draws(alpha=0.05, k=1):
    """Smallest B for which a two-sided reading at level alpha, Holm-adjusted over k sources, can succeed."""
    return int(np.ceil(2 * k / alpha - 1))


def two_tailed(real, null):
    """(upper-tail, lower-tail) permutation p-values of `real` against `null`."""
    return p_value(real, null, "upper"), p_value(real, null, "lower")


def auc_from_scores(scores, labels):
    """Rank AUC (ties count one half) of 0/1 `labels` under `scores`."""
    from scipy.stats import rankdata
    labels = np.asarray(labels).astype(bool); rk = rankdata(scores); n1 = labels.sum(); n0 = len(labels) - n1
    return (rk[labels].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def heldout_label_test(scores_eval, labels_eval, M=9999, seed=0, statistic=None):
    """Held-out-label permutation test of a FIXED fitted model: permute the held-out labels among the held-out
    evaluation nodes, keep the scores. Exact under H0^A (held-out labels exchangeable given the scores), whatever the
    training sources contain; needs no refitting. Returns (upper-tail, lower-tail) p-values.
    statistic(scores, labels) defaults to the AUC."""
    stat = statistic or auc_from_scores
    rng = np.random.default_rng(seed); labels_eval = np.asarray(labels_eval)
    real = stat(scores_eval, labels_eval)
    null = np.array([stat(scores_eval, labels_eval[rng.permutation(len(labels_eval))]) for _ in range(M)])
    return two_tailed(real, null)


def combined_p2(p_train, p_held):
    """Two-sided p-value of the intersection-union rule: min(1, 2 min(max(pT_up, pA_up), max(pT_dn, pA_dn))).
    Valid under H0^T (no training signal) and under H0^A (no held-out association); use it for Holm."""
    return min(1.0, 2 * min(max(p_train[0], p_held[0]), max(p_train[1], p_held[1])))


def read_transfer(p_train, p_held, alpha=0.05):
    """Decision rule of the paper (Corollary 2): transfer if both upper-tail p-values are <= alpha/2, reversal if both
    lower-tail p-values are, otherwise inconclusive. p_train, p_held: (upper, lower) pairs for the same fit."""
    if max(p_train[0], p_held[0]) <= alpha / 2:
        return "transfers"
    if max(p_train[1], p_held[1]) <= alpha / 2:
        return "reversal"
    return "inconclusive"


def read_balanced(balanced_real, balanced_null, alpha=0.05, center=0.5):
    """Training-label test alone (two-sided, alpha/2 per tail). Exact only under H0^T (no training signal): it does
    NOT control false transfer calls when training carries signal that the held-out source lacks. Prefer
    read_transfer; kept for comparison."""
    if balanced_real > center and p_value(balanced_real, balanced_null, "upper") <= alpha / 2:
        return "transfers"
    if balanced_real < center and p_value(balanced_real, balanced_null, "lower") <= alpha / 2:
        return "reversal"
    return "inconclusive"


def holm(pvals):
    """Holm step-down adjusted p-values, in input order."""
    p = np.asarray(pvals, float); order = np.argsort(p); m = len(p)
    adj = np.empty(m); running = 0.0
    for j, i in enumerate(order):
        running = max(running, min(1.0, (m - j) * p[i])); adj[i] = running
    return adj


def provenance_score(posteriors, label_rates):
    """r(x) = sum_s pi_s P(s|x). posteriors: (n, S) source posteriors re-weighted to pool composition."""
    return np.asarray(posteriors) @ np.asarray(label_rates)


def incorpus_reference_closed_form(r, pooled_rate):
    """Theorem 2(a): AUC(r) = 1/2 + E|r(X) - r(X')| / (4 pibar (1 - pibar)), X, X' iid from the pool.
    r: provenance scores of a uniform sample of the pool."""
    rs = np.sort(np.asarray(r, float)); n = len(rs); i = np.arange(1, n + 1)
    gmd = 2 * np.sum((2 * i - n - 1) * rs) / (n * (n - 1))
    return 0.5 + gmd / (4 * pooled_rate * (1 - pooled_rate))
