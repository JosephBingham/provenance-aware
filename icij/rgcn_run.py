"""R-GCN under the within-source reference. Writes results incrementally.
Usage: python3 rgcn_run.py <out_json> <R> <B> <Rbal> <Bbal> <Rin> <Bin>"""
import sys, json, time, numpy as np, pandas as pd, torch
from config import WORK, ICIJ
from evallib import *; import rgcn
torch.set_num_threads(1)
OUT = sys.argv[1]; R, B, Rb, Bb, Ri, Bi = map(int, sys.argv[2:8])
F = rgcn.Fitter(ICIJ, WORK, epochs=100)
df = pd.read_parquet(WORK + "/officers.parquet"); y = df.y.values.astype(int); leak = df.leak.values
res = {}; t0 = time.time()
def save(): json.dump(res, open(OUT, "w"), indent=1, default=float)
for held in LEAKS:
    if not R: break
    pool, ep, eu = lolo_sets(y, leak, held, np.random.default_rng(13))
    real, null = run(None, y, leak, pool, ep, eu, R, B, seed=200 + LEAKS.index(held), fitter=F)
    res[f"lolo_{held}"] = summarize(real, null); res[f"lolo_{held}"]["null_draws"] = null[:, 0].tolist()
    print(held, {k: round(v, 3) for k, v in res[f"lolo_{held}"].items() if k != "null_draws"}, f"{time.time()-t0:.0f}s", flush=True); save()
for held in LEAKS:
    if not Rb: break
    pool, ep, eu = lolo_sets(y, leak, held, np.random.default_rng(13))
    real, null = run(None, y, leak, pool, ep, eu, Rb, Bb, seed=300 + LEAKS.index(held), balanced=True, fitter=F)
    res[f"lolo_bal_{held}"] = summarize(real, null); res[f"lolo_bal_{held}"]["null_draws"] = null[:, 0].tolist()
    print("bal", held, {k: round(v, 3) for k, v in res[f"lolo_bal_{held}"].items() if k != "null_draws"}, f"{time.time()-t0:.0f}s", flush=True); save()
if Ri:
    real, null = run_incorpus(None, y, leak, Ri, Bi, seed=100, matched=False, fitter=F)
    res["incorpus_uniform"] = summarize(real, null)
    print("incorpus_uniform", {k: round(v, 3) for k, v in res["incorpus_uniform"].items()}, f"{time.time()-t0:.0f}s", flush=True); save()
