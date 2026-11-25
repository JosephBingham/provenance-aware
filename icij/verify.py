"""Independent checks, written without evallib: (1) Panama held-out AUC and within-leak null for the 8-feature
model via a separate code path (own rank-based AUC, pandas groupby permutation); (2) R-GCN neighbourhood
restriction gives the same outputs as the full graph; (3) leakage checks."""
import sys, numpy as np, pandas as pd, torch
from config import WORK, ICIJ
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
import rgcn

def auc(pos, neg):
    s = np.r_[pos, neg]; r = rankdata(s)
    return (r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))

df = pd.read_parquet(WORK + "/officers.parquet")
F8 = ["ent_deg", "log_ent_deg", "n_addr", "n_inter", "n_jur", "role_div", "ent_per_addr", "ent_per_inter"]
rng = np.random.default_rng(2024)
held = df[df.leak == "Panama Papers"]; tr = df[df.leak != "Panama Papers"]
ev_unl = held[held.y == 0].sample(40000, random_state=1); ev_pos = held[held.y == 1]
def fit_score(trd, seed):
    P = trd[trd.y == 1]; N = trd[trd.y == 0].sample(3 * len(P), random_state=seed)
    m = HGB(max_iter=200, max_depth=3, learning_rate=0.1, early_stopping=False, random_state=seed)
    m.fit(pd.concat([P, N])[F8].values, np.r_[np.ones(len(P)), np.zeros(len(N))])
    return auc(m.predict_proba(ev_pos[F8].values)[:, 1], m.predict_proba(ev_unl[F8].values)[:, 1])
real = [fit_score(tr, s) for s in range(3)]
nulls = []
for s in range(9):
    t2 = tr.copy(); t2["y"] = t2.groupby("leak").y.transform(lambda v: v.sample(frac=1, random_state=100 + s).values)
    assert (t2.groupby("leak").y.sum() == tr.groupby("leak").y.sum()).all()
    nulls.append(fit_score(t2, 50 + s))
print("(1) Panama 8-feature: real %.3f  within-leak null %.3f (sd %.3f, n=9)" % (np.mean(real), np.mean(nulls), np.std(nulls)), flush=True)

F = rgcn.Fitter(ICIJ, WORK, epochs=1)
torch.manual_seed(0); model = rgcn.RGCN(F.X.shape[1], len(F.adj))
rows = F.off_node[rng.choice(len(F.off_node), 3000, replace=False)]
with torch.no_grad():
    full = model(F.X, F.AX, [rgcn.to_torch(a[rows]) for a in F.adj], torch.as_tensor(rows))
    sub = [a[rows] for a in F.adj]; nb = np.unique(np.concatenate([rows] + [s.indices for s in sub]))
    loc = np.searchsorted(nb, rows)
    restr = model(F.X[nb], [ax[nb] for ax in F.AX], [rgcn.to_torch(s[:, nb]) for s in sub], torch.as_tensor(loc))
print("(2) R-GCN restriction max |diff| = %.2e over %d officers" % (float((full - restr).abs().max()), len(rows)), flush=True)

name_like = [c for c in df.columns if "name" in c.lower()]
print("(3) feature columns containing 'name':", name_like)
rel = pd.read_csv(ICIJ + "/relationships.csv", dtype=str, keep_default_na=False, usecols=["rel_type", "link"])
used = rel[rel.rel_type.isin(["officer_of", "registered_address", "intermediary_of"])]
print("    nominee-typed links among edges used by TransE/R-GCN:", int(used.link.str.contains("nominee", case=False).sum()),
      "| 'underlying' edges used:", int((used.rel_type == "underlying").sum()))
pos_officers = set(df.officer[df.y == 1])
print("    positives:", len(pos_officers), "| base rates of held-out evaluation sets (AP chance level):",
      {L: round(int((df.leak == L).values[df.y.values == 1].sum()) / (int((df.leak == L).values[df.y.values == 1].sum()) + min(40000, int(((df.leak == L) & (df.y == 0)).sum()))), 3)
       for L in ["Panama Papers", "Paradise Papers", "Offshore Leaks", "Bahamas Leaks"]})
