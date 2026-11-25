"""OpenSanctions cohort flow and multi-source persons (response to review 4.5, 5.5). Usage: python3 os_cohort.py <out.json>"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_os import WORK_FIRST as WORK
import json, collections, pandas as pd
S = ["ext_ru_egrul", "us_sam_exclusions", "wikidata", "gb_coh_disqualified", "ru_acf_bribetakers"]
g = lambda d: "wikidata" if (d == "wikidata" or d.startswith("wd_")) else d
n = pd.read_parquet(WORK + "/nodes.parquet"); t = pd.read_parquet(WORK + "/topics.parquet")
sanc = set(t[t.topic == "sanction"].canonical); L = sorted(set(t[t.topic == "sanction"].dataset))
out = {"entities": int(len(n)), "label_datasets": len(L)}
p = n[n.schema == "Person"].copy(); out["persons"] = int(len(p))
p["y"] = p.id.isin(sanc).astype(int); out["persons_pos"] = int(p.y.sum())
p["s"] = p.datasets.str.split("|").apply(lambda ds: tuple(sorted({g(d) for d in ds} & set(S))))
p["k"] = p.s.str.len()
out["by_k"] = {int(k): {"n": int(len(d)), "pos": int(d.y.sum())} for k, d in p.groupby("k")}
m = p[p.k >= 2]
out["multi_combos"] = {"+".join(c): {"n": int(len(d)), "pos": int(d.y.sum())} for c, d in m.groupby("s")}
one = p[p.k == 1]
out["single"] = {c[0]: {"n": int(len(d)), "pos": int(d.y.sum())} for c, d in one.groupby("s")}
# persons whose only label-relevant datasets are sanction lists (not in S) are outside the cohort
out["persons_no_S"] = {"n": int((p.k == 0).sum()), "pos": int(p[p.k == 0].y.sum())}
json.dump(out, open(sys.argv[1], "w"), indent=1); print(json.dumps(out, indent=1))
