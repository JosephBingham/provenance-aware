"""RDF2Vec-style embeddings (Ristoski & Paulheim 2016) of the label-free ICIJ graph.
Random walks of depth D (hops) from every assigned officer over the typed graph, following each relation in both
directions (inverse relations get their own token), emitting node and relation tokens; skip-gram word2vec on the walks.
Usage: python3 rdf2vec.py <icij_dir> <work_dir> <walks_per_root> <depth> <dim> <epochs> <out.npy>"""
import sys, os, time, numpy as np, pandas as pd
from config import ICIJ

D, W = sys.argv[1], sys.argv[2]
NW, DEPTH, DIM, EP, OUT = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), sys.argv[7]
rng = np.random.default_rng(0); t0 = time.time()
rel = pd.read_csv(f"{D}/relationships.csv", dtype=str, keep_default_na=False,
                  usecols=["node_id_start", "node_id_end", "rel_type", "link"])
rel = rel[rel.rel_type.isin(["officer_of", "registered_address", "intermediary_of"])]
ben = (rel.rel_type == "officer_of") & rel.link.str.lower().str.contains("beneficial|beneficiary")
rname = rel.rel_type.where(~ben, "beneficial_owner_of").values
h = rel.node_id_start.astype(np.int64).values; t = rel.node_id_end.astype(np.int64).values
del rel
nodes, inv = np.unique(np.r_[h, t], return_inverse=True); n = len(nodes)
hi, ti = inv[: len(h)], inv[len(h):]
rels = ["officer_of", "beneficial_owner_of", "registered_address", "intermediary_of"]
rcode = np.array([rels.index(x) for x in rname], dtype=np.int8)
E = np.unique(np.stack([hi, ti, rcode.astype(np.int64)], 1), axis=0)       # deduplicate typed triples
src = np.r_[E[:, 0], E[:, 1]]; dst = np.r_[E[:, 1], E[:, 0]]
rr = np.r_[E[:, 2], E[:, 2] + len(rels)]                                    # inverse relations: code + 4
order = np.argsort(src, kind="stable"); src, dst, rr = src[order], dst[order], rr[order]
indptr = np.zeros(n + 1, np.int64); np.add.at(indptr, src + 1, 1); indptr = np.cumsum(indptr)
print(f"graph {n} nodes, {len(E)} typed triples, {time.time()-t0:.0f}s", flush=True)

off = pd.read_parquet(f"{W}/officers.parquet", columns=["officer"]).officer.astype(np.int64).values
roots = np.searchsorted(nodes, off); assert (nodes[roots] == off).all()
corpus = f"{W}/rdf2vec_walks.txt"
with open(corpus, "w") as f:
    for w in range(NW):
        cur = roots.copy(); toks = [cur]; rtoks = []
        alive = np.ones(len(cur), bool)
        for d in range(DEPTH):
            deg = indptr[cur + 1] - indptr[cur]
            alive &= deg > 0
            j = indptr[cur] + np.floor(rng.random(len(cur)) * np.maximum(deg, 1)).astype(np.int64)
            nxt = np.where(alive, dst[np.minimum(j, len(dst) - 1)], -1); r = np.where(alive, rr[np.minimum(j, len(dst) - 1)], -1)
            rtoks.append(r); toks.append(nxt); cur = np.where(alive, nxt, cur)
        T = np.stack(toks, 1); Rt = np.stack(rtoks, 1)
        lines = []
        for k in range(len(roots)):
            parts = [f"n{T[k,0]}"]
            for d in range(DEPTH):
                if T[k, d + 1] < 0: break
                parts.append(f"r{Rt[k,d]}"); parts.append(f"n{T[k,d+1]}")
            lines.append(" ".join(parts))
        f.write("\n".join(lines) + "\n")
        print(f"walk round {w+1}/{NW} {time.time()-t0:.0f}s", flush=True)

from gensim.models import Word2Vec
m = Word2Vec(corpus_file=corpus, vector_size=DIM, window=5, sg=1, negative=5, min_count=1,
             sample=1e-3, epochs=EP, workers=2, seed=0)
print(f"word2vec done {time.time()-t0:.0f}s; vocab {len(m.wv)}", flush=True)
X = np.stack([m.wv[f"n{r}"] for r in roots]).astype(np.float32)
np.save(OUT, X); os.remove(corpus)
# sanity: nearest neighbours of an officer should share its entity neighbourhood more often than random
print("saved", X.shape, f"{time.time()-t0:.0f}s", flush=True)
