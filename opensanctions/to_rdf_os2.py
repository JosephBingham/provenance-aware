"""N-Quads for the count-invariant OpenSanctions cohort (response to the second review): every person's membership in
every non-label dataset, one named graph per dataset; each `sanction` topic statement in the graph of the dataset that
asserted it; the five sources marked in a meta graph. The SPARQL audit (audit_os2.rq) then recomputes the provenance
cells, the per-source counts and rates. Usage: python3 to_rdf_os2.py <out.nq>"""
import os, sys, re, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_os import WORK_FIRST
OUT = sys.argv[1]
B = "https://w3id.org/anon-e2/"; FTM = "https://w3id.org/anon-e2/ftm#"
G = lambda d: f"<{B}graph/{re.sub(r'[^A-Za-z0-9_]', '_', d)}>"
node = lambda i: f"<{B}entity/{re.sub(r'[^A-Za-z0-9_.-]', '_', i)}>"
S = ["ext_ru_egrul", "us_sam_exclusions", "wikidata", "gb_coh_disqualified", "ru_acf_bribetakers"]
n = pd.read_parquet(WORK_FIRST + "/nodes.parquet", columns=["id", "schema", "datasets"]); n = n[n.schema == "Person"]
t = pd.read_parquet(WORK_FIRST + "/topics.parquet", columns=["canonical", "topic", "dataset"])
L = set(t[t.topic == "sanction"].dataset); k = 0
ex = n.assign(d=n.datasets.str.split("|")).explode("d").drop_duplicates(["id", "d"]); ex = ex[~ex.d.isin(L)]
with open(OUT, "w") as f:
    for i, d in zip(ex.id, ex.d):
        f.write(f"{node(i)} <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <{FTM}Person> {G(d)} .\n"); k += 1
    tt = t[(t.topic == "sanction")].drop_duplicates(["canonical", "dataset"])
    tt = tt[tt.canonical.isin(set(n.id))]
    for i, d in zip(tt.canonical, tt.dataset):
        f.write(f'{node(i)} <{FTM}topic> "sanction" {G(d)} .\n'); k += 1
    for d in sorted(set(ex.d)):
        f.write(f'{G(d)} <{FTM}role> "{"source" if d in S else "provenance"}" <{B}graph/meta> .\n'); k += 1
print("quads", k)
