"""RDF2Vec on the OpenSanctions S-graph (relationships asserted by the five sources), same walk scheme and word2vec
settings as for ICIJ. Usage: python3 rdf2vec_os.py <work_dir> <walks_per_root> <depth> <dim> <epochs> <out.npy>"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sys, os, time, numpy as np, pandas as pd
W = sys.argv[1]; NW, DEPTH, DIM, EP, OUT = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
rng = np.random.default_rng(0); t0 = time.time()
e = pd.read_parquet(f"{W}/sgraph_edges.parquet")
p = pd.read_parquet(f"{W}/persons.parquet", columns=["id"]).id.values
nodes, inv = np.unique(np.r_[e.src.values, e.dst.values, p], return_inverse=True); n = len(nodes)
hi, ti = inv[: len(e)], inv[len(e): 2 * len(e)]
rels = sorted(e.schema.unique()); rcode = np.searchsorted(rels, e.schema.values)
src = np.r_[hi, ti]; dst = np.r_[ti, hi]; rr = np.r_[rcode, rcode + len(rels)]
order = np.argsort(src, kind="stable"); src, dst, rr = src[order], dst[order], rr[order]
indptr = np.zeros(n + 1, np.int64); np.add.at(indptr, src + 1, 1); indptr = np.cumsum(indptr)
print(f"graph {n} nodes, {len(e)} typed triples, relations {rels}, {time.time()-t0:.0f}s", flush=True)
roots = np.searchsorted(nodes, p); assert (nodes[roots] == p).all()
corpus = f"{W}/rdf2vec_walks.txt"
with open(corpus, "w") as f:
    for w in range(NW):
        cur = roots.copy(); toks = [cur]; rtoks = []; alive = np.ones(len(cur), bool)
        for d in range(DEPTH):
            deg = indptr[cur + 1] - indptr[cur]; alive &= deg > 0
            j = indptr[cur] + np.floor(rng.random(len(cur)) * np.maximum(deg, 1)).astype(np.int64)
            jj = np.minimum(j, len(dst) - 1)
            nxt = np.where(alive, dst[jj], -1); r = np.where(alive, rr[jj], -1)
            rtoks.append(r); toks.append(nxt); cur = np.where(alive, nxt, cur)
        T = np.stack(toks, 1); Rt = np.stack(rtoks, 1); lines = []
        for k in range(len(roots)):
            parts = [f"n{T[k,0]}"]
            for d in range(DEPTH):
                if T[k, d + 1] < 0: break
                parts.append(f"r{Rt[k,d]}"); parts.append(f"n{T[k,d+1]}")
            lines.append(" ".join(parts))
        f.write("\n".join(lines) + "\n"); print(f"walk round {w+1}/{NW} {time.time()-t0:.0f}s", flush=True)
from gensim.models import Word2Vec
m = Word2Vec(corpus_file=corpus, vector_size=DIM, window=5, sg=1, negative=5, min_count=1, sample=1e-3,
             epochs=EP, workers=2, seed=0)
print(f"word2vec done {time.time()-t0:.0f}s; vocab {len(m.wv)}", flush=True)
X = np.stack([m.wv[f"n{r}"] for r in roots]).astype(np.float32); np.save(OUT, X); os.remove(corpus)
print("saved", X.shape, f"{time.time()-t0:.0f}s", flush=True)
