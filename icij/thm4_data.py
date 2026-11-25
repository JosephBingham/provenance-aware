"""Plug-in provenance score r(x) = sum_s pi_s P(s|x) on each held-out leak (8 structural features), trained on the
other three leaks; saves scores of held-out positives and unlabelled officers. Usage: python3 thm4_data.py <out.npz>"""
import sys, numpy as np, pandas as pd
from config import WORK
from evallib import *
from sklearn.metrics import roc_auc_score
df = pd.read_parquet(WORK + "/officers.parquet"); y = df.y.values; leak = df.leak.values
F8 = ["ent_deg", "log_ent_deg", "n_addr", "n_inter", "n_jur", "role_div", "ent_per_addr", "ent_per_inter"]
X = df[F8].values.astype(np.float32)
out = {}
for k, L in enumerate(LEAKS):
    pool, ep, eu = lolo_sets(y, leak, L, np.random.default_rng(13))
    rng = np.random.default_rng(77 + k)
    cls, pr = source_posteriors(X, leak, pool, np.concatenate([ep, eu]), rng)
    pi = np.array([y[pool][leak[pool] == c].mean() for c in cls]); r = pr @ pi
    rp, ru = r[: len(ep)], r[len(ep):]
    auc = roc_auc_score(np.r_[np.ones(len(rp)), np.zeros(len(ru))], r)
    # which training leak do held-out positives resemble more than held-out unlabelled officers?
    post_pos, post_unl = pr[: len(ep)].mean(0), pr[len(ep):].mean(0)
    print(f"{L:16s} AUC(r) {auc:.3f} | mean P(source|x), positives vs unlabelled:",
          {c: (round(a, 3), round(b, 3)) for c, a, b in zip(cls, post_pos, post_unl)}, "| pi", dict(zip(cls, np.round(pi, 4))), flush=True)
    out[f"pos_{k}"], out[f"unl_{k}"], out[f"auc_{k}"] = rp, ru, auc
    j = int(np.argmax(pi)); out[f"hi_{k}"] = str(cls[j]); out[f"hipos_{k}"] = post_pos[j]; out[f"hiunl_{k}"] = post_unl[j]
np.savez(sys.argv[1], **out)
