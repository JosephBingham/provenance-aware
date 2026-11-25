import sys, numpy as np, pandas as pd
from config import WORK, ICIJ
from evallib import *; import rgcn
X, adj, off_node = rgcn.build(ICIJ, WORK)
adj = [a.tocsr() for a in adj]
H = np.empty((X.shape[0], X.shape[1] * (1 + len(adj))), np.float32); H[:, :X.shape[1]] = X
for k, a in enumerate(adj):
    H[:, X.shape[1] * (k + 1): X.shape[1] * (k + 2)] = a @ X           # 1-hop mean aggregates per relation
H1 = H[off_node]
S = np.zeros_like(H1)
for a in adj: S += a[off_node] @ H                                     # 2-hop, summed over relations
del H
H2 = np.concatenate([H1, S], 1)
df = pd.read_parquet(WORK + "/officers.parquet"); y = df.y.values; leak = df.leak.values
rng = np.random.default_rng(7)
print("1-hop source acc", source_accuracy(H1, leak, rng), flush=True)
print("2-hop source acc", source_accuracy(H2, leak, rng), flush=True)
print("thm3 (2-hop receptive field)", theorem3(H2, y, leak, rng), flush=True)
