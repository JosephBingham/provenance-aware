"""Bootstrap 95% intervals for the held-out AUC of the tested (first) source-balanced fit on each leak (response to the
third review, minor 3). Re-creates the first fit as bal99.py / heldout_assoc.py do and resamples held-out positives and
unlabelled officers separately (2,000 resamples). The interval describes evaluation-sample uncertainty for a fixed fit.
Usage: python3 boot_icij.py <out.json>"""
import sys, os, json, numpy as np, pandas as pd
from scipy.stats import rankdata
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from config import WORK
from evallib import train_set, gbm, lolo_sets, LEAKS
df = pd.read_parquet(WORK + "/officers.parquet"); y = df.y.values; leak = df.leak.values
F8 = ["ent_deg", "log_ent_deg", "n_addr", "n_inter", "n_jur", "role_div", "ent_per_addr", "ent_per_inter"]
X = df[F8].values.astype(np.float32); out = {}
for k, L in enumerate(LEAKS):
    pool, ep, eu = lolo_sets(y, leak, L, np.random.default_rng(13)); rng = np.random.default_rng(400 + k)
    P, N = train_set(y, leak, pool, rng, True); tr = np.concatenate([P, N]); t = np.r_[np.ones(len(P)), np.zeros(len(N))]
    m = gbm(int(rng.integers(1 << 30))).fit(X[tr], t)
    sp, su = m.predict_proba(X[ep])[:, 1], m.predict_proba(X[eu])[:, 1]
    def auc(a, b):
        rk = rankdata(np.r_[a, b]); n1, n0 = len(a), len(b); return (rk[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    a0 = auc(sp, su); br = np.random.default_rng(55 + k)
    bs = np.array([auc(sp[br.integers(0, len(sp), len(sp))], su[br.integers(0, len(su), len(su))]) for _ in range(2000)])
    out[L] = dict(auc=float(a0), lo=float(np.quantile(bs, .025)), hi=float(np.quantile(bs, .975)))
    print(L, "%.3f [%.3f, %.3f]" % (a0, out[L]["lo"], out[L]["hi"]), flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
