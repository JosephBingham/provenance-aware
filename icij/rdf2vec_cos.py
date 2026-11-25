"""RDF2Vec sanity check: mean cosine similarity of officers that share an entity vs random officer pairs.
Usage: python3 rdf2vec_cos.py <icij_dir> <out.json>"""
import sys, json, numpy as np, pandas as pd
from config import WORK
D = sys.argv[1]
off = pd.read_parquet(WORK + "/officers.parquet"); E = np.load(WORK + "/emb_rdf2vec.npy")
E = E / np.linalg.norm(E, axis=1, keepdims=True); pos = pd.Series(np.arange(len(off)), index=off.officer.astype(str))
rel = pd.read_csv(f"{D}/relationships.csv", dtype=str, keep_default_na=False, usecols=["node_id_start", "node_id_end", "rel_type"])
oo = rel[(rel.rel_type == "officer_of") & rel.node_id_start.isin(pos.index)][["node_id_start", "node_id_end"]].drop_duplicates()
oo["i"] = pos.reindex(oo.node_id_start).values
rng = np.random.default_rng(5)
g = oo.groupby("node_id_end").i.apply(np.array); g = g[g.apply(len) >= 2]
ents = rng.choice(len(g), 50000)
pairs = np.array([rng.choice(g.iloc[e], 2, replace=False) for e in ents])
share = np.mean(np.sum(E[pairs[:, 0]] * E[pairs[:, 1]], 1))
a, b = rng.integers(0, len(E), 50000), rng.integers(0, len(E), 50000); keep = a != b
rand = np.mean(np.sum(E[a[keep]] * E[b[keep]], 1))
out = {"cos_shared_entity": float(share), "cos_random": float(rand), "n_pairs": 50000}
print(out); json.dump(out, open(sys.argv[2], "w"), indent=1)
