"""Full-batch R-GCN (Schlichtkrull et al. 2018) on the ICIJ graph, CPU, plain torch sparse ops.
Relations: officer_of, beneficial_owner_of, registered_address, intermediary_of, each in both directions,
mean-normalised, plus a self-loop. Input: node-type one-hot + log1p degree per relation-direction (label-free).
Hyperparameters fixed in advance (no tuning): hidden 32, 2 layers, Adam lr 0.01, wd 5e-4, 100 epochs, BCE."""
import os, numpy as np, pandas as pd, torch, scipy.sparse as sp
from config import ICIJ

torch.set_num_threads(2)


def build(D, W):
    cache = f"{W}/rgcn_graph.npz"
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        return z["X"], list(z["adj"]), z["off_node"]
    rel = pd.read_csv(f"{D}/relationships.csv", dtype=str, keep_default_na=False,
                      usecols=["node_id_start", "node_id_end", "rel_type", "link"])
    rel = rel[rel.rel_type.isin(["officer_of", "registered_address", "intermediary_of"])]
    ben = (rel.rel_type == "officer_of") & rel.link.str.lower().str.contains("beneficial|beneficiary")
    r = rel.rel_type.where(~ben, "beneficial_owner_of").values
    h = rel.node_id_start.astype(np.int64).values; t = rel.node_id_end.astype(np.int64).values
    nodes, inv = np.unique(np.r_[h, t], return_inverse=True); n = len(nodes)
    hi, ti = inv[: len(h)], inv[len(h):]
    adj, degs = [], []
    for rn in ["officer_of", "beneficial_owner_of", "registered_address", "intermediary_of"]:
        m = r == rn
        for a, b in [(hi[m], ti[m]), (ti[m], hi[m])]:          # messages b -> a
            A = sp.coo_matrix((np.ones(m.sum(), np.float32), (a, b)), shape=(n, n)).tocsr()
            A.data[:] = 1.0; A.sum_duplicates()
            d = np.asarray(A.sum(1)).ravel(); degs.append(np.log1p(d))
            A = sp.diags(1 / np.maximum(d, 1)).astype(np.float32) @ A
            adj.append(A.tocoo())
    types = np.zeros((n, 5), np.float32)
    for k, fn in enumerate(["officers", "entities", "intermediaries", "addresses", "others"]):
        ids = pd.read_csv(f"{D}/nodes-{fn}.csv", dtype=str, usecols=["node_id"]).node_id.astype(np.int64).values
        pos = np.searchsorted(nodes, ids); ok = (pos < n) & (nodes[np.minimum(pos, n - 1)] == ids)
        types[pos[ok], k] = 1
    X = np.concatenate([types, np.stack(degs, 1).astype(np.float32)], 1)
    X[:, 5:] = (X[:, 5:] - X[:, 5:].mean(0)) / (X[:, 5:].std(0) + 1e-6)
    off = pd.read_parquet(f"{W}/officers.parquet", columns=["officer"]).officer.astype(np.int64).values
    off_node = np.searchsorted(nodes, off); assert (nodes[off_node] == off).all()
    np.savez(cache, X=X, adj=np.array(adj, dtype=object), off_node=off_node)
    return X, adj, off_node


class RGCN(torch.nn.Module):
    def __init__(self, nin, nrel, hid=32):
        super().__init__()
        self.W1 = torch.nn.ModuleList([torch.nn.Linear(nin, hid, bias=False) for _ in range(nrel)])
        self.S1 = torch.nn.Linear(nin, hid)
        self.W2 = torch.nn.ModuleList([torch.nn.Linear(hid, hid, bias=False) for _ in range(nrel)])
        self.S2 = torch.nn.Linear(hid, hid)
        self.out = torch.nn.Linear(hid, 1)

    def forward(self, X, AX, A_rows, rows):
        h1 = self.S1(X) + sum(W(ax) for W, ax in zip(self.W1, AX))      # AX precomputed: layer-1 input is fixed
        h1 = torch.relu(h1)
        h2 = self.S2(h1[rows]) + sum(W(torch.sparse.mm(A, h1)) for W, A in zip(self.W2, A_rows))
        return self.out(torch.relu(h2)).squeeze(-1)


def to_torch(A):
    A = A.tocoo()
    return torch.sparse_coo_tensor(np.vstack([A.row, A.col]), A.data, A.shape).coalesce()


class Fitter:
    """fitter(tr, t, ev, seed) -> scores for ev (officer indices)."""
    def __init__(self, D, W, epochs=100):
        X, adj, self.off_node = build(D, W)
        self.adj = [a.tocsr() for a in adj]
        self.X = torch.from_numpy(X)
        self.AX = [torch.from_numpy(np.asarray(a @ X, dtype=np.float32)) for a in self.adj]
        self.epochs = epochs

    def __call__(self, tr, t, ev, seed):
        torch.manual_seed(seed)
        rows = np.concatenate([self.off_node[tr], self.off_node[ev]])
        # restrict layer 1 to the 1-hop neighbourhood of the rows (exactly equivalent output)
        sub = [a[rows] for a in self.adj]
        nb = np.unique(np.concatenate([rows] + [s.indices for s in sub]))
        loc = np.searchsorted(nb, rows)
        A_rows = [to_torch(s[:, nb]) for s in sub]
        X, AX = self.X[nb], [ax[nb] for ax in self.AX]
        rows_t = torch.as_tensor(loc); ntr = len(tr)
        y = torch.as_tensor(t, dtype=torch.float32)
        model = RGCN(self.X.shape[1], len(self.adj))
        opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        pw = (y == 0).sum() / max(int((y == 1).sum()), 1)
        for _ in range(self.epochs):
            opt.zero_grad()
            out = model(X, AX, A_rows, rows_t)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(out[:ntr], y, pos_weight=pw)
            loss.backward(); opt.step()
        with torch.no_grad():
            out = model(X, AX, A_rows, rows_t)
        self.last_loss = float(loss.detach()); self.n_sub = len(nb)
        return out[ntr:].numpy()
