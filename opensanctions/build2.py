"""OpenSanctions, count-invariant cohort (response to the second review, 4.1).
Provenance cell of a person = the set of non-label datasets (datasets that never assert the topic `sanction`) that record
it, ungrouped. Sources: non-label person datasets that are not sanctions or export-control publications, whose singleton
cell has at least 150 positives. Cohort: persons whose cell is a singleton {s} with s a source. The selection then depends
on the labels only through the positive counts of the analysis strata, which every within-source permutation preserves.
Features exactly as build.py (structural, from relationships asserted by a source dataset).
Usage: python3 build2.py $OS_WORK_CELLS_DIR   (after build.py, which writes nodes/topics tables to OS_WORK_DIR)"""
import os, sys, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_os import EXTRACT, WORK_FIRST
OUT = sys.argv[1]
n = pd.read_parquet(WORK_FIRST + "/nodes.parquet", columns=["id", "schema", "datasets"])
t = pd.read_parquet(WORK_FIRST + "/topics.parquet", columns=["canonical", "topic", "dataset"])
L = set(t[t.topic == "sanction"].dataset); sanc = set(t[t.topic == "sanction"].canonical)
SANCTION_PORTALS = {"ua_war_sanctions", "tw_shtc", "ext_us_ofac_press_releases"}   # sanctions publications that do not emit the topic (PREREG)
p = n[n.schema == "Person"][["id", "datasets"]]
ex = p.assign(d=p.datasets.str.split("|")).explode("d").drop_duplicates(["id", "d"])
nl = ex[~ex.d.isin(L)]
k = nl.groupby("id").d.nunique()
single = nl[nl.id.map(k) == 1][["id", "d"]].copy(); single["y"] = single.id.isin(sanc).astype(int)
cells = single.groupby("d").agg(n=("id", "size"), pos=("y", "sum")).sort_values("pos", ascending=False)
elig = cells[(cells.pos >= 150) & ~cells.index.isin(SANCTION_PORTALS)]
S = list(elig.index)
print("sources:", elig.to_dict("index"))
cand = cells[cells.pos >= 10].copy()
cand["reason"] = np.where(cand.index.isin(SANCTION_PORTALS), "sanctions publication", np.where(cand.pos >= 150, "source", "< 150 positives in its cell"))
cand.to_csv(f"{OUT}/candidate_cells.csv")
pp = single[single.d.isin(S)].rename(columns={"d": "src"}).reset_index(drop=True)
# S-graph: relationships asserted by at least one source dataset (no address links, as in build.py)
e = pd.read_csv(EXTRACT + "/edges.tsv.gz", sep="\t", dtype=str, keep_default_na=False, quoting=3)
keep = e.datasets.str.split("|").apply(lambda ds: any(d in S for d in ds))
e = e[keep & (e.src != e.dst) & (e.schema != "addressEntity")].drop_duplicates(["src", "dst", "schema"])
print("S-graph edges", len(e), e.schema.value_counts().to_dict())
e[["src", "dst", "schema"]].to_parquet(f"{OUT}/sgraph_edges.parquet")
deg = pd.concat([e.src, e.dst]).value_counts(); f = pd.DataFrame(index=pp.id); ids = set(pp.id)
for sch in sorted(e.schema.unique()):
    for side in ["src", "dst"]:
        c = e[e.schema == sch][side].value_counts()
        if c.index.isin(ids).sum() >= 50:
            f[f"{sch}_{'out' if side == 'src' else 'in'}"] = c.reindex(f.index).fillna(0).values
und = pd.concat([e[["src", "dst"]], e[["dst", "src"]].rename(columns={"dst": "src", "src": "dst"})]).drop_duplicates()
und = und[und.src.isin(ids)]; und["nd"] = und.dst.map(deg).fillna(0)
agg = und.groupby("src").nd.agg(["size", "mean", "max", "sum"])
f["n_nbrs"] = agg["size"].reindex(f.index).fillna(0).values; f["nbr_deg_mean"] = agg["mean"].reindex(f.index).fillna(0).values
f["nbr_deg_max"] = agg["max"].reindex(f.index).fillna(0).values; f["two_hop"] = agg["sum"].reindex(f.index).fillna(0).values
f["log_n_nbrs"] = np.log1p(f.n_nbrs)
sc = list(f.columns)
ts = t[t.dataset.isin(S) & ~t.topic.str.startswith(("sanction", "export")) & t.canonical.isin(ids)]
top = ts.groupby("topic").canonical.nunique(); top = top[top >= 50].index
tf = pd.crosstab(ts[ts.topic.isin(top)].canonical, ts[ts.topic.isin(top)].topic).clip(upper=1)
for c in tf.columns:
    f[f"topic_{c}"] = tf[c].reindex(f.index).fillna(0).values
print("topics", list(tf.columns))
out = pd.concat([pp, f.reset_index(drop=True)], axis=1)
out.to_parquet(f"{OUT}/persons.parquet"); open(f"{OUT}/struct_cols.txt", "w").write("\n".join(sc))
summ = {"sources": S, "cells": cells.head(20).to_dict("index"), "n": int(len(out)), "pos": int(out.y.sum()),
        "by_source": out.groupby("src").y.agg(["size", "sum", "mean"]).to_dict("index"), "struct_cols": sc,
        "isolated_by_source_y": out.assign(iso=out.n_nbrs == 0).groupby(["src", "y"]).iso.mean().to_dict().__repr__()}
json.dump(summ, open(f"{OUT}/cohort.json", "w"), indent=1, default=str); print(json.dumps(summ["by_source"], indent=1)); print(summ["isolated_by_source_y"])
