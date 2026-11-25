import sys, numpy as np, pandas as pd
from config import WORK
from evallib import *
df = pd.read_parquet(WORK + "/officers.parquet"); y = df.y.values; leak = df.leak.values
F8 = ["ent_deg","log_ent_deg","n_addr","n_inter","n_jur","role_div","ent_per_addr","ent_per_inter"]
X = df[F8].values.astype(np.float32)
for matched in [False, True]:
    for pe in [True, False]:
        r, n = run_incorpus(X, y, leak, 3, 5, seed=100+matched, matched=matched, perm_eval=pe)
        print("matched" if matched else "uniform", "perm_eval" if pe else "real_eval_labels", round(r[:,0].mean(),3), round(n[:,0].mean(),3), round(n[:,0].std(),3), flush=True)
