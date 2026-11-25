"""Build officer table: leak, label, 8 tabular + 4 graph features. Follows the Array manuscript's description.
Usage: python3 prep.py <icij_data_dir> <out_dir>
"""
import sys, os, re, numpy as np, pandas as pd, igraph as ig

D, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
rel = pd.read_csv(f"{D}/relationships.csv", dtype=str, keep_default_na=False,
                  usecols=["node_id_start", "node_id_end", "rel_type", "link", "sourceID"])
off = pd.read_csv(f"{D}/nodes-officers.csv", dtype=str, keep_default_na=False, usecols=["node_id", "name", "sourceID"])
ent = pd.read_csv(f"{D}/nodes-entities.csv", dtype=str, keep_default_na=False, usecols=["node_id", "jurisdiction"])
officer_ids = set(off.node_id)

# --- graph over three relation types, undirected, deduplicated
g = rel[rel.rel_type.isin(["officer_of", "registered_address", "intermediary_of"])]
a = g.node_id_start.astype(np.int64).values; b = g.node_id_end.astype(np.int64).values
lo, hi = np.minimum(a, b), np.maximum(a, b)
E = np.unique(np.stack([lo, hi], 1), axis=0)
E = E[E[:, 0] != E[:, 1]]
nodes, inv = np.unique(E.ravel(), return_inverse=True)
Ei = inv.reshape(-1, 2)
print("graph", len(nodes), "nodes", len(Ei), "edges", flush=True)
G = ig.Graph(n=len(nodes), edges=Ei.tolist(), directed=False)
deg = np.array(G.degree())
pr = np.array(G.pagerank(damping=0.85, implementation="prpack"))
core = np.array(G.coreness())
# mean / max neighbour degree
nd_sum = np.zeros(len(nodes)); nd_max = np.zeros(len(nodes))
np.add.at(nd_sum, Ei[:, 0], deg[Ei[:, 1]]); np.add.at(nd_sum, Ei[:, 1], deg[Ei[:, 0]])
np.maximum.at(nd_max, Ei[:, 0], deg[Ei[:, 1]]); np.maximum.at(nd_max, Ei[:, 1], deg[Ei[:, 0]])
nd_mean = nd_sum / np.maximum(deg, 1)
gfeat = pd.DataFrame({"nid": nodes, "pagerank": pr, "kcore": core, "nbr_deg_mean": nd_mean, "nbr_deg_max": nd_max})
pd.DataFrame(Ei, columns=["u", "v"]).to_parquet(f"{OUT}/edges_idx.parquet")
pd.Series(nodes).to_frame("nid").to_parquet(f"{OUT}/node_index.parquet")

# --- officer_of edges, excluding beneficial-ownership sub-types
oo = rel[rel.rel_type == "officer_of"].copy()
ben = oo.link.str.lower().str.contains("beneficial|beneficiary")
print("beneficial share of officer_of", round(ben.mean(), 4))
oo = oo[~ben & oo.node_id_start.isin(officer_ids)]
oo["role"] = oo.link.str.lower().str.strip().str.replace(r"\s+of$", "", regex=True)
oo["leak"] = oo.sourceID.where(~oo.sourceID.str.startswith("Paradise"), "Paradise Papers")
tag = oo[oo.sourceID != ""]
cnt = tag.groupby(["node_id_start", "leak"]).size().reset_index(name="n")
cnt = cnt.sort_values("n", ascending=False).drop_duplicates("node_id_start")
src = cnt.set_index("node_id_start").leak

# tabular features
ej = ent.set_index("node_id").jurisdiction
oo["jur"] = oo.node_id_end.map(ej)
inter = rel[rel.rel_type == "intermediary_of"][["node_id_start", "node_id_end"]].rename(columns={"node_id_start": "inter", "node_id_end": "ent"})
oe = oo[["node_id_start", "node_id_end"]].drop_duplicates().rename(columns={"node_id_start": "off", "node_id_end": "ent"})
f = pd.DataFrame(index=src.index)
f["ent_deg"] = oe.groupby("off").ent.nunique()
f["log_ent_deg"] = np.log1p(f.ent_deg)
ra = rel[(rel.rel_type == "registered_address")]
f["n_addr"] = ra.groupby("node_id_start").node_id_end.nunique().reindex(f.index).fillna(0)
oi = oe.merge(inter, on="ent")
f["n_inter"] = oi.groupby("off").inter.nunique().reindex(f.index).fillna(0)
f["n_jur"] = oo[oo.jur.notna() & (oo.jur != "")].groupby("node_id_start").jur.nunique().reindex(f.index).fillna(0)
f["role_div"] = oo.groupby("node_id_start").role.nunique().reindex(f.index).fillna(0)
f["ent_per_addr"] = f.ent_deg / np.maximum(f.n_addr, 1)
f["ent_per_inter"] = f.ent_deg / np.maximum(f.n_inter, 1)
f = f.fillna(0)
gf = gfeat.set_index(gfeat.nid.astype(str))
for c in ["pagerank", "kcore", "nbr_deg_mean", "nbr_deg_max"]:
    f[c] = gf[c].reindex(f.index).fillna(0).values
f["degree"] = pd.Series(deg, index=nodes.astype(str)).reindex(f.index).fillna(0).values

name = off.set_index("node_id").name
f["y"] = name.reindex(f.index).fillna("").str.contains("NOMINEE|NOMINEES|TRUSTEE|FIDUCIAR|NOMINI", case=False, regex=True).astype(int)
f["leak"] = src
f.index.name = "officer"
f.reset_index().to_parquet(f"{OUT}/officers.parquet")
print(f.groupby("leak").agg(n=("y", "size"), pos=("y", "sum"), rate=("y", "mean")))
print("total", len(f), "pos", f.y.sum())
print(f.describe().T[["mean", "50%", "max"]])
