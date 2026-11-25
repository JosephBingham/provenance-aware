"""Sensitivity of the ICIJ in-corpus reference to the granularity of the source partition (response to the second
review, 4.2). Partitions: the four leaks (paper); ICIJ's finer sourceID, which splits the Paradise Papers into the
Appleby law firm and eight corporate registries; one stratum (an ordinary permutation); and each leak split in two at
random, a finer partition unrelated to how the data were collected. Eight structural features, uniform unlabelled
sampling, 5 real fits, 15 draws, plus Theorem 2's plug-in by rate regression.
Usage: python3 granularity_icij.py <icij_csv_dir> <out.json>"""
import sys, json, numpy as np, pandas as pd
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import WORK
from evallib import run_incorpus, summarize
from ratereg import rate_regression_r, thm2_from_r
CSV, OUT = sys.argv[1], sys.argv[2]
df = pd.read_parquet(WORK + "/officers.parquet")
F8 = ["ent_deg", "log_ent_deg", "n_addr", "n_inter", "n_jur", "role_div", "ent_per_addr", "ent_per_inter"]
X = df[F8].values.astype(np.float32); y = df.y.values.astype(int)
sid = pd.read_csv(f"{CSV}/nodes-officers.csv", usecols=["node_id", "sourceID"], dtype=str).set_index("node_id").sourceID
fine = df.officer.map(sid).values.astype(object); assert pd.isna(fine).sum() == 0
assert all(str(f).startswith(str(l).split()[0]) for f, l in zip(fine[:5000], df.leak.values[:5000]))
half = np.random.default_rng(99).integers(0, 2, len(y))
parts = {"leak (paper)": df.leak.values.astype(object), "sourceID": fine, "one stratum": np.full(len(y), "all", dtype=object),
         "leak, random halves": np.array([f"{l}|{h}" for l, h in zip(df.leak.values, half)], dtype=object)}
res = {"strata": {}}
for name, src in parts.items():
    tab = pd.DataFrame({"s": src, "y": y}).groupby("s").y.agg(["size", "sum", "mean"])
    real, null = run_incorpus(X, y, src, 5, 15, seed=100, matched=False); sm = summarize(real, null)
    rng = np.random.default_rng(7); r = rate_regression_r(X, y, src, rng)
    res[name] = dict(n_strata=int(len(tab)), real=float(sm["real"]), ref=float(sm["null"]), ref_sd=float(sm["null_sd"]), thm2=thm2_from_r(r, y, rng),
                     rate_min=float(tab["mean"].min()), rate_max=float(tab["mean"].max()))
    res["strata"][name] = {str(k): [int(v["size"]), int(v["sum"])] for k, v in tab.iterrows()}
    print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res[name].items()}, flush=True)
    json.dump(res, open(OUT, "w"), indent=1)
