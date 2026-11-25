import sys, json, numpy as np, pandas as pd
from config import WORK, RESULTS
from evallib import *
df = pd.read_parquet(WORK + "/officers.parquet"); y = df.y.values; leak = df.leak.values
F8 = ["ent_deg","log_ent_deg","n_addr","n_inter","n_jur","role_div","ent_per_addr","ent_per_inter"]
X = df[F8].values.astype(np.float32); res = {}
for L in LEAKS:
    pool, ep, eu = lolo_sets(y, leak, L, np.random.default_rng(13))
    real, null = run(X, y, leak, pool, ep, eu, 10, 99, seed=400 + LEAKS.index(L), balanced=True)
    s = summarize(real, null); s["null_draws"] = null[:, 0].tolist(); s["real_draws"] = real[:, 0].tolist(); res[L] = s
    na = null[:, 0]
    print(L, "real %.3f (sd %.3f) | null mean %.3f sd %.3f q2.5 %.3f q97.5 %.3f max %.3f | mean+2sd %.3f | p(first) %.2f | share of real draws > all nulls %.1f"%(
        s["real"], s["real_sd"], na.mean(), na.std(ddof=1), np.quantile(na,.025), np.quantile(na,.975), na.max(), na.mean()+2*na.std(ddof=1), s["p"], (real[:,0] > na.max()).mean()), flush=True)
    print("   null histogram", np.histogram(na, bins=np.arange(0.25, 0.80, 0.05))[0].tolist(), flush=True)
json.dump(res, open(RESULTS + "/full8_bal99.json", "w"), default=float)
