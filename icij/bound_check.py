"""Proposition 2 (size of the shortcut) on ICIJ: for each representation, the Theorem 2 plug-in reference and the
Proposition 2 upper bound 1/2 + sum_s |pi_s - pibar| E|P(s|X) - q_s| / (2 pibar (1 - pibar)), from the same leak-classifier
posteriors on a uniform sample of 60,000 officers. Usage: python3 bound_check.py <out.json>"""
import sys, json, numpy as np, pandas as pd
from config import WORK, ICIJ
from evallib import source_posteriors
df = pd.read_parquet(WORK + "/officers.parquet"); y = df.y.values; leak = df.leak.values
F8 = ["ent_deg", "log_ent_deg", "n_addr", "n_inter", "n_jur", "role_div", "ent_per_addr", "ent_per_inter"]
REPS = {"conn": lambda: df[["ent_deg", "n_addr", "n_inter"]].values, "full8": lambda: df[F8].values,
        "full12": lambda: df[F8 + ["pagerank", "kcore", "nbr_deg_mean", "nbr_deg_max"]].values,
        "transe": lambda: np.load(WORK + "/emb_transe.npy"),
        "rdf2vec": lambda: np.load(WORK + "/emb_rdf2vec.npy")}
allidx = np.arange(len(y)); pib = y.mean(); out = {}
for name, get in REPS.items():
    X = get().astype(np.float32); rng = np.random.default_rng(11)
    cls, pr = source_posteriors(X, leak, allidx, rng.choice(allidx, 60000, replace=False), rng)
    pi = np.array([y[leak == c].mean() for c in cls]); q = np.array([(leak == c).mean() for c in cls])
    r = pr @ pi; rs = np.sort(r); k = len(rs); i = np.arange(1, k + 1)
    gmd = 2 * np.sum((2 * i - k - 1) * rs) / (k * (k - 1))
    closed = 0.5 + gmd / (4 * pib * (1 - pib))
    bound = 0.5 + np.sum(np.abs(pi - pib) * np.abs(pr - q).mean(0)) / (2 * pib * (1 - pib))
    out[name] = {"closed_form": closed, "bound": bound}
    print(f"{name:8s} closed form {closed:.3f}  Prop. 2 bound {bound:.3f}", flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
