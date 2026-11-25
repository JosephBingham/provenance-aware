"""Sensitivity analysis for the single-source restriction on OpenSanctions (response to the reviews; label-free assignments): adds the 7,887
persons recorded by two or more of the five sources and assigns them by three label-free rules -- the provenance combination
as its own stratum, the largest and the smallest member source (by single-source persons). Structural representation; in-corpus pooled score
(uniform unlabelled sampling), within-source reference and Theorem 2 plug-in for each partition.
Usage: python3 os_multi_lf.py <out.json>"""
import os, sys, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_os import WORK_FIRST as WORK
from e2lib import *
OUT = sys.argv[1]
S = SOURCES
g = lambda d: "wikidata" if (d == "wikidata" or d.startswith("wd_")) else d
n = pd.read_parquet(WORK + "/nodes.parquet"); t = pd.read_parquet(WORK + "/topics.parquet")
sanc = set(t[t.topic == "sanction"].canonical)
p = n[n.schema == "Person"].copy()
p["s"] = p.datasets.str.split("|").apply(lambda ds: tuple(sorted({g(d) for d in ds} & set(S))))
p = p[p.s.str.len() >= 1][["id", "s"]].reset_index(drop=True); p["y"] = p.id.isin(sanc).astype(int)
# features exactly as build.py (same S-graph, same column list)
e = pd.read_parquet(WORK + "/sgraph_edges.parquet"); sc = open(WORK + "/struct_cols.txt").read().split("\n")
deg = pd.concat([e.src, e.dst]).value_counts(); f = pd.DataFrame(index=p.id)
for c in sc:
    if c.endswith("_out") or c.endswith("_in"):
        sch, side = c.rsplit("_", 1); col = "src" if side == "out" else "dst"
        f[c] = e[e.schema == sch][col].value_counts().reindex(f.index).fillna(0).values
und = pd.concat([e[["src", "dst"]], e[["dst", "src"]].rename(columns={"dst": "src", "src": "dst"})]).drop_duplicates()
und = und[und.src.isin(set(p.id))]; und["nd"] = und.dst.map(deg).fillna(0)
agg = und.groupby("src").nd.agg(["size", "mean", "max", "sum"])
f["n_nbrs"] = agg["size"].reindex(f.index).fillna(0).values; f["nbr_deg_mean"] = agg["mean"].reindex(f.index).fillna(0).values
f["nbr_deg_max"] = agg["max"].reindex(f.index).fillna(0).values; f["two_hop"] = agg["sum"].reindex(f.index).fillna(0).values
f["log_n_nbrs"] = np.log1p(f.n_nbrs)
X = f[sc].values.astype(np.float32); y = p.y.values; multi = (p.s.str.len() >= 2).values
# check: single-source rows reproduce the main cohort's features
ref = pd.read_parquet(WORK + "/persons.parquet").set_index("id")
chk = float(np.abs(f.loc[ref.index, sc].values - ref[sc].values).max())
# label-free assignments only (Proposition 3): sizes are numbers of single-source persons, no labels used
size1 = {s: int(((~multi) & (p.s.str[0] == s).values).sum()) for s in S}
parts = {"combination": p.s.str.join("+").values,
         "largest": p.s.apply(lambda c: max(c, key=lambda s: size1[s])).values,
         "smallest": p.s.apply(lambda c: min(c, key=lambda s: size1[s])).values}
res = {"n": int(len(y)), "pos": int(y.sum()), "n_multi": int(multi.sum()), "pos_multi": int(y[multi].sum()),
       "rate_single": float(y[~multi].mean()), "rate_multi": float(y[multi].mean()), "feature_check_maxabs": chk}
rng = np.random.default_rng(7)
for name, src in parts.items():
    real, null = run_incorpus(X, y, src, 5, 15, seed=100, matched=False)
    s = summarize(real, null)
    res[name] = {"strata": int(len(np.unique(src))), "real": float(s["real"]), "ref": float(s["null"]),
                 "ref_sd": float(s["null_sd"]), "thm2": theorem2_r(X, y, src, rng)}
    print(name, res[name], flush=True); json.dump(res, open(OUT, "w"), indent=1)
