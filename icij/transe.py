"""Lean label-free TransE (Bordes et al. 2013) on the ICIJ graph: L1 distance, margin ranking loss (margin 1),
one corrupted head-or-tail negative per triple, unit-norm entity constraint, sparse Adam.
A 0.5% held-out triple sample is used only for a sanity check (sampled MRR vs. 100 random tails).
Usage: python3 transe.py <icij_dir> <work_dir> <dim> <epochs> <out.npy>"""
import sys, time, numpy as np, pandas as pd, torch
from config import ICIJ

D, W, DIM, EP, OUT = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
torch.set_num_threads(1); torch.manual_seed(0); rng = np.random.default_rng(0)
rel = pd.read_csv(f"{D}/relationships.csv", dtype=str, keep_default_na=False,
                  usecols=["node_id_start", "node_id_end", "rel_type", "link"])
rel = rel[rel.rel_type.isin(["officer_of", "registered_address", "intermediary_of"])]
ben = (rel.rel_type == "officer_of") & rel.link.str.lower().str.contains("beneficial|beneficiary")
r = rel.rel_type.where(~ben, "beneficial_owner_of").values
h = rel.node_id_start.astype(np.int64).values; t = rel.node_id_end.astype(np.int64).values
del rel
ents, inv = np.unique(np.r_[h, t], return_inverse=True); n = len(ents)
rels, rinv = np.unique(r, return_inverse=True)
T = np.unique(np.stack([inv[: len(h)], rinv, inv[len(h):]], 1), axis=0)
del h, t, r, inv
perm = rng.permutation(len(T)); nte = len(T) // 200
test, train = T[perm[:nte]], T[perm[nte:]]
print("entities", n, "triples", len(T), "train", len(train), flush=True)
E = torch.nn.Embedding(n, DIM, sparse=True); R = torch.nn.Embedding(len(rels), DIM, sparse=True)
b = 6 / DIM ** 0.5
torch.nn.init.uniform_(E.weight, -b, b); torch.nn.init.uniform_(R.weight, -b, b)
with torch.no_grad():
    E.weight.div_(E.weight.norm(dim=1, keepdim=True)); R.weight.div_(R.weight.norm(dim=1, keepdim=True))
opt = torch.optim.SparseAdam(list(E.parameters()) + list(R.parameters()), lr=0.01)
tr = torch.as_tensor(train)


def dist(hh, rr, tt):
    return (E(hh) + R(rr) - E(tt)).abs().sum(-1)


def sampled_mrr(k=100):
    with torch.no_grad():
        te = torch.as_tensor(test[:5000])
        pos = dist(te[:, 0], te[:, 1], te[:, 2])
        neg_t = torch.randint(0, n, (len(te), k))
        neg = (E(te[:, 0]).unsqueeze(1) + R(te[:, 1]).unsqueeze(1) - E(neg_t)).abs().sum(-1)
        rank = 1 + (neg < pos.unsqueeze(1)).sum(1).float()
        return float((1 / rank).mean())


print("epoch 0 sampled MRR", round(sampled_mrr(), 4), "(random ~", round(float(np.mean(1 / np.arange(1, 102))), 4), ")", flush=True)
t0 = time.time(); bs = 32768
for ep in range(EP):
    idx = torch.randperm(len(tr)); tot = 0.0
    for i in range(0, len(tr), bs):
        bt = tr[idx[i:i + bs]]; hh, rr, tt = bt[:, 0], bt[:, 1], bt[:, 2]
        corrupt = torch.rand(len(bt)) < 0.5; rnd = torch.randint(0, n, (len(bt),))
        nh = torch.where(corrupt, rnd, hh); nt = torch.where(corrupt, tt, rnd)
        loss = torch.relu(1.0 + dist(hh, rr, tt) - dist(nh, rr, nt)).mean()
        opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss) * len(bt)
        with torch.no_grad():
            u = torch.unique(torch.cat([hh, tt, nh, nt]))
            E.weight[u] = E.weight[u] / E.weight[u].norm(dim=1, keepdim=True).clamp_min(1.0)
    print(f"epoch {ep+1} loss {tot/len(tr):.4f} sampled MRR {sampled_mrr():.4f} {time.time()-t0:.0f}s", flush=True)
off = pd.read_parquet(f"{W}/officers.parquet", columns=["officer"]).officer.astype(np.int64).values
pos = np.searchsorted(ents, off); assert (ents[pos] == off).all()
np.save(OUT, E.weight.detach().numpy()[pos].astype(np.float32)); print("saved", len(pos))
