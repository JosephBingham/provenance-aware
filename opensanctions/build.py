"""E2: person table, S-graph, features. Usage: python3 build.py <work_dir>   (reads the extract in OS_EXTRACT_DIR)"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_os import EXTRACT, WORK_FIRST as WORK
import sys, numpy as np, pandas as pd, collections
OUT = sys.argv[1]
D = EXTRACT
S = ["ext_ru_egrul", "us_sam_exclusions", "wikidata", "gb_coh_disqualified", "ru_acf_bribetakers"]
g = lambda d: "wikidata" if (d == "wikidata" or d.startswith("wd_")) else d
os.makedirs(OUT, exist_ok=True)
rd = lambda f: pd.read_csv(f"{D}/{f}", sep="\t", dtype=str, keep_default_na=False, quoting=3)
n = rd("nodes.tsv.gz"); t = rd("topics_by_dataset.tsv.gz")
n.to_parquet(f"{OUT}/nodes.parquet"); t.to_parquet(f"{OUT}/topics.parquet")
L = set(t[t.topic == "sanction"].dataset); sanc = set(t[t.topic == "sanction"].canonical)
assert not (set(S) & L)
p = n[n.schema == "Person"].copy()
p["s"] = p.datasets.str.split("|").apply(lambda ds: sorted({g(d) for d in ds} & set(S)))
p = p[p.s.str.len() == 1].assign(src=lambda x: x.s.str[0])[["id", "src"]].reset_index(drop=True)
p["y"] = p.id.isin(sanc).astype(int)
print(p.groupby("src").y.agg(["size", "sum", "mean"]))
# S-graph: relationships asserted by at least one S dataset
e = pd.read_csv(f"{D}/edges.tsv.gz", sep="\t", dtype=str, keep_default_na=False, quoting=3)
keep = e.datasets.str.split("|").apply(lambda ds: any(g(d) in S for d in ds))
e = e[keep & (e.src != e.dst) & (e.schema != "addressEntity")].drop_duplicates(["src", "dst", "schema"])
print("S-graph edges", len(e), e.schema.value_counts().to_dict())
e[["src", "dst", "schema"]].to_parquet(f"{OUT}/sgraph_edges.parquet")
deg = pd.concat([e.src, e.dst]).value_counts()
f = pd.DataFrame(index=p.id)
for sch in sorted(e.schema.unique()):
    for side, other in [("src", "dst"), ("dst", "src")]:
        c = e[e.schema == sch][side].value_counts()
        if c.index.isin(p.id).sum() >= 50:
            f[f"{sch}_{'out' if side == 'src' else 'in'}"] = c.reindex(f.index).fillna(0).values
und = pd.concat([e[["src", "dst"]], e[["dst", "src"]].rename(columns={"dst": "src", "src": "dst"})]).drop_duplicates()
und = und[und.src.isin(set(p.id))]
und["nd"] = und.dst.map(deg).fillna(0)
agg = und.groupby("src").nd.agg(["size", "mean", "max", "sum"])
f["n_nbrs"] = agg["size"].reindex(f.index).fillna(0).values
f["nbr_deg_mean"] = agg["mean"].reindex(f.index).fillna(0).values
f["nbr_deg_max"] = agg["max"].reindex(f.index).fillna(0).values
f["two_hop"] = agg["sum"].reindex(f.index).fillna(0).values      # walks of length 2 (upper bound on 2-hop reach)
f["log_n_nbrs"] = np.log1p(f.n_nbrs)
struct_cols = list(f.columns)
# context topics asserted by S datasets
ts = t[t.dataset.map(g).isin(S) & ~t.topic.str.startswith(("sanction", "export"))]
ts = ts[ts.canonical.isin(set(p.id))]
top = ts.groupby("topic").canonical.nunique(); top = top[top >= 50].index
tf = pd.crosstab(ts[ts.topic.isin(top)].canonical, ts[ts.topic.isin(top)].topic).clip(upper=1)
for c in tf.columns:
    f[f"topic_{c}"] = tf[c].reindex(f.index).fillna(0).values
print("structural features", len(struct_cols), "topics", list(tf.columns))
f = f.reset_index(drop=True)
out = pd.concat([p, f], axis=1)
out.to_parquet(f"{OUT}/persons.parquet")
open(f"{OUT}/struct_cols.txt", "w").write("\n".join(struct_cols))
print(out.shape)
