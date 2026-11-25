"""Facts about the OpenSanctions extract quoted in the paper, recomputed from the extract.
Usage: python3 e2_audit.py <extract_dir> <work_dir> <sparql_audit.log> <out.json>"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sys, json, re, pandas as pd, numpy as np
X, W, LOG, OUT = sys.argv[1:5]
summ = json.load(open(f"{X}/summary.json")); prov = json.load(open(f"{X}/provenance.json"))
t = pd.read_csv(f"{X}/topics_by_dataset.tsv.gz", sep="\t", dtype=str, keep_default_na=False, quoting=3)
p = pd.read_parquet(f"{W}/persons.parquet"); e = pd.read_parquet(f"{W}/sgraph_edges.parquet")
out = {"version": prov["published_version"], "n_datasets": len(summ["datasets"]),
       "n_label_datasets": int(t[t.topic == "sanction"].dataset.nunique()),
       "persons": len(p), "positives": int(p.y.sum()),
       "rates": p.groupby("src").y.mean().to_dict(), "sgraph_edges": len(e), "sgraph_by_schema": e.schema.value_counts().to_dict(),
       "isolated_by_src_y": {f"{s}|{y}": float(v) for (s, y), v in (p.n_nbrs == 0).groupby([p.src, p.y]).mean().items()},
       "checksums_match": all(v.get("checksum_matches") for v in prov["files"].values())}
log = open(LOG).read(); out["quads_loaded"] = int(re.search(r"quads (\d+)", log).group(1))
out["sparql_rows"] = {re.search(r"graph/([^>]+)>", l).group(1): [int(x) for x in re.findall(r'"(\d+)"\^\^<http://www.w3.org/2001/XMLSchema#integer>', l)]
                      for l in log.splitlines() if "graph/" in l}
# address-link leak (pre-fix): share of SAM persons, by label, with an addressEntity link whose entity datasets include an S source
g = lambda d: "wikidata" if (d == "wikidata" or d.startswith("wd_")) else d
S = {"ext_ru_egrul", "us_sam_exclusions", "wikidata", "gb_coh_disqualified", "ru_acf_bribetakers"}
ed = pd.read_csv(f"{X}/edges.tsv.gz", sep="\t", dtype=str, keep_default_na=False, quoting=3)
ad = ed[(ed.schema == "addressEntity") & ed.datasets.str.split("|").apply(lambda ds: any(g(d) in S for d in ds))]
has = p.id.isin(set(ad.src)); sam = p.src == "us_sam_exclusions"
out["sam_address_share"] = {"pos": float(has[sam & (p.y == 1)].mean()), "unl": float(has[sam & (p.y == 0)].mean())}
json.dump(out, open(OUT, "w"), indent=1, default=float); print(json.dumps(out, indent=1, default=float)[:1500])
