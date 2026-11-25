"""Coverage of each OpenSanctions source in the count-invariant cohort (response to the third review, M4): share of
positives and of other persons with at least one relationship, and the number of distinct feature vectors under the
structural representation (i) and with topics (ii). Usage: python3 os_coverage.py <persons.parquet> <struct_cols.txt> <out.json>"""
import sys, json, numpy as np, pandas as pd
d = pd.read_parquet(sys.argv[1]); sc = open(sys.argv[2]).read().split("\n"); tc = [c for c in d.columns if c.startswith("topic_")]
out = {}
for s in ["ext_ru_egrul", "us_sam_exclusions", "wikidata", "gb_coh_disqualified", "ru_acf_bribetakers"]:
    x = d[d.src == s]; rel = (x.n_nbrs > 0).values; yy = x.y.values == 1
    out[s] = dict(n=int(len(x)), pos=int(yy.sum()), rel_pos=float(rel[yy].mean()), rel_oth=float(rel[~yy].mean()),
                  distinct_i=int(len(np.unique(x[sc].values, axis=0))), distinct_ii=int(len(np.unique(x[sc + tc].values, axis=0))),
                  linked_labelled=float(yy[rel].mean()) if rel.any() else None)
    print(s, out[s])
json.dump(out, open(sys.argv[3], "w"), indent=1)
