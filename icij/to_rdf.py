"""ICIJ Offshore Leaks CSV -> N-Quads. One named graph per ICIJ sourceID; node typing in a 'nodes' graph;
the name-derived proxy label in its own graph so representations can be computed label-free; PROV metadata.
Usage: python3 to_rdf.py <icij_data_dir> <work_dir> <out.nq>"""
import sys, re, hashlib, pandas as pd
from config import ICIJ

D, W, OUT = sys.argv[1:4]
B = "https://w3id.org/anon-icij/"
EX = B + "ont#"
def node(i): return f"<{B}node/{i}>"
def gname(s): return f"<{B}graph/{re.sub(r'[^A-Za-z0-9]+', '_', s).strip('_') or 'untagged'}>"
G_NODES, G_LABELS, G_META = f"<{B}graph/nodes>", f"<{B}graph/labels>", f"<{B}graph/meta>"
RDF_TYPE = "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>"
PROV = "http://www.w3.org/ns/prov#"
n = 0
with open(OUT, "w") as f:
    def q(s, p, o, g):
        global n; f.write(f"{s} {p} {o} {g} .\n"); n += 1
    for fn, cls in [("officers", "Officer"), ("entities", "Entity"), ("intermediaries", "Intermediary"),
                    ("addresses", "Address"), ("others", "Other")]:
        for nid in pd.read_csv(f"{D}/nodes-{fn}.csv", dtype=str, usecols=["node_id"]).node_id:
            q(node(nid), RDF_TYPE, f"<{EX}{cls}>", G_NODES)
    rel = pd.read_csv(f"{D}/relationships.csv", dtype=str, keep_default_na=False,
                      usecols=["node_id_start", "node_id_end", "rel_type", "link", "sourceID"])
    rel = rel[rel.rel_type.isin(["officer_of", "registered_address", "intermediary_of"])]
    ben = (rel.rel_type == "officer_of") & rel.link.str.lower().str.contains("beneficial|beneficiary")
    pred = rel.rel_type.map({"officer_of": "officerOf", "registered_address": "registeredAddress",
                             "intermediary_of": "intermediaryOf"}).where(~ben, "beneficialOwnerOf")
    for s, p, o, src in zip(rel.node_id_start, pred, rel.node_id_end, rel.sourceID):
        q(node(s), f"<{EX}{p}>", node(o), gname(src))
    srcs = sorted(set(rel.sourceID))
    act = f"<{B}activity/csv-to-rdf>"
    sha = hashlib.sha256(open(f"{D}/relationships.csv", "rb").read()).hexdigest()
    q(act, RDF_TYPE, f"<{PROV}Activity>", G_META)
    q(act, f"<{PROV}used>", f"<{B}source/relationships.csv>", G_META)
    q(f"<{B}source/relationships.csv>", f"<{EX}sha256>", f'"{sha}"', G_META)
    for s in srcs:
        g = gname(s)
        leak = "Paradise Papers" if s.startswith("Paradise") else (s or "untagged")
        q(g, f"<{PROV}wasGeneratedBy>", act, G_META)
        q(g, f"<{EX}icijSourceID>", f'"{s}"', G_META)
        q(g, f"<{EX}leak>", f'"{leak}"', G_META)
    off = pd.read_parquet(f"{W}/officers.parquet", columns=["officer", "y"])
    for o in off.officer[off.y == 1]:
        q(node(o), RDF_TYPE, f"<{EX}NameMatchedNominee>", G_LABELS)
print("quads", n)
