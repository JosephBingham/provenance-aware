"""OpenSanctions E2 subset as N-Quads: one named graph per source dataset. Person membership and relationships go
into the graph of each S dataset that contributed them; each `sanction` topic statement goes into the graph of the
dataset that asserted it (statement-level provenance). Usage: python3 to_rdf_os.py <work_dir> <out.nq>"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_os import EXTRACT, WORK_FIRST as WORK
import sys, pandas as pd, re
W, OUT = sys.argv[1], sys.argv[2]
B = "https://w3id.org/anon-e2/"; FTM = "https://w3id.org/anon-e2/ftm#"
g = lambda d: "wikidata" if (d == "wikidata" or d.startswith("wd_")) else d
G = lambda d: f"<{B}graph/{re.sub(r'[^A-Za-z0-9_]', '_', d)}>"
node = lambda i: f"<{B}entity/{re.sub(r'[^A-Za-z0-9_.-]', '_', i)}>"
S = {"ext_ru_egrul", "us_sam_exclusions", "wikidata", "gb_coh_disqualified", "ru_acf_bribetakers"}
p = pd.read_parquet(f"{W}/persons.parquet", columns=["id", "src"])
n = pd.read_parquet(WORK + "/nodes.parquet", columns=["id", "datasets"]).set_index("id")
e = pd.read_csv(EXTRACT + "/edges.tsv.gz", sep="\t", dtype=str, keep_default_na=False, quoting=3)
e = e[e.schema != "addressEntity"]
t = pd.read_parquet(WORK + "/topics.parquet")
ids = set(p.id); k = 0
with open(OUT, "w") as f:
    for i in p.id:
        for d in {g(x) for x in n.at[i, "datasets"].split("|")} & S:
            f.write(f"{node(i)} <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <{FTM}Person> {G(d)} .\n"); k += 1
    ee = e[e.src.isin(ids) | e.dst.isin(ids)]
    for s_, o, sch, ds in zip(ee.src, ee.dst, ee.schema, ee.datasets):
        for d in {g(x) for x in ds.split("|")} & S:
            f.write(f"{node(s_)} <{FTM}{sch}> {node(o)} {G(d)} .\n"); k += 1
    tt = t[(t.topic == "sanction") & t.canonical.isin(ids)].drop_duplicates(["canonical", "dataset"])
    for i, d in zip(tt.canonical, tt.dataset):
        f.write(f'{node(i)} <{FTM}topic> "sanction" {G(d)} .\n'); k += 1
    for d in sorted(S):
        f.write(f'{G(d)} <{FTM}role> "source" <{B}graph/meta> .\n'); k += 1
print("quads", k)
