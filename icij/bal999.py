"""Extend the 8-feature source-balanced nulls from 99 to 999 draws per held-out leak (fresh seeds).
Usage: python3 bal999.py <shard> <nshards>"""
import sys, json, numpy as np, pandas as pd
from config import WORK, RESULTS
from evallib import *
shard, nsh = int(sys.argv[1]), int(sys.argv[2])
df = pd.read_parquet(WORK + "/officers.parquet"); y = df.y.values; leak = df.leak.values
F8 = ["ent_deg","log_ent_deg","n_addr","n_inter","n_jur","role_div","ent_per_addr","ent_per_inter"]
X = df[F8].values.astype(np.float32)
for k, L in enumerate(LEAKS):
    if k % nsh != shard: continue
    pool, ep, eu = lolo_sets(y, leak, L, np.random.default_rng(13))
    _, null = run(X, y, leak, pool, ep, eu, 0, 900, seed=7000 + k, balanced=True)
    json.dump(null[:, 0].tolist(), open(f"{RESULTS}/bal900_{k}.json", "w"))
    print(L, "done", flush=True)
