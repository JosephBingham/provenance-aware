"""Theorem 2 plug-in by rate regression: r(x) = E[pi_{s(V)} | X = x], estimated by cross-fitted gradient-boosting
regression of each node's source rate on its representation (no multiclass source classifier needed)."""
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

def gmd(v):
    v = np.sort(v); k = len(v); i = np.arange(1, k + 1); return 2 * np.sum((2 * i - k - 1) * v) / (k * (k - 1))

def rate_regression_r(X, y, src, rng, folds=2, l2=0.0):
    """Plug-in r(x) = E[pi_{s(V)} | X = x] by cross-fitted gradient-boosting regression of each node's source rate on
    its representation. Needs no multiclass source classifier, so it scales to many sources."""
    s = pd.Series(src); pi = s.map(pd.Series(y).groupby(s).mean()).values
    f = rng.integers(0, folds, len(y)); r = np.zeros(len(y))
    for k in range(folds):
        m = HistGradientBoostingRegressor(max_iter=200, max_depth=3, learning_rate=0.1, l2_regularization=l2,
                                          early_stopping=False, random_state=k).fit(X[f != k], pi[f != k])
        r[f == k] = m.predict(X[f == k])
    return np.clip(r, 0, 1)

def thm2_from_r(r, y, rng, n=60000):
    idx = rng.choice(len(r), min(n, len(r)), replace=False); pib = y.mean()
    return float(0.5 + gmd(r[idx]) / (4 * pib * (1 - pib)))

