"""Add extra source-balanced null draws to existing settings (same data: generator re-seeded identically).
Usage: python3 extend.py <shard> <nshards> <in.jsonl...> --out <out.jsonl> --extra 20"""
import sys, json, numpy as np
import os
sys.argv = [a for a in sys.argv]
out = sys.argv[sys.argv.index("--out") + 1]; extra = int(sys.argv[sys.argv.index("--extra") + 1])
shard, nsh = int(sys.argv[1]), int(sys.argv[2]); ins = sys.argv[3:sys.argv.index("--out")]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "sim.py")).read().split("done = set()")[0]
sys.argv = ["sim.py", "0", "1", "/dev/null"]; exec(src)          # defines GRID, gen, fit_score, permute_within
recs = {json.loads(l)["i"]: json.loads(l) for f in ins for l in open(f)}
done = set()
try: done = {json.loads(l)["i"] for l in open(out)}
except FileNotFoundError: pass
with open(out, "a") as f:
    for i in sorted(recs):
        if i % nsh != shard or i in done: continue
        r = recs[i]; K, rho, Delta, theta, delta, rel = GRID[i]
        X, lab, s, mus = gen(K, rho, Delta, theta, delta, rel, np.random.default_rng(1000 + i))
        assert abs(r["K"] - K) + abs(r["rho"] - rho) == 0
        rng = np.random.default_rng(900000 + i)
        r["bal_null"] = r["bal_null"] + [fit_score(X, permute_within(lab, s, rng), s, rng, True) for _ in range(extra)]
        f.write(json.dumps(r) + "\n"); f.flush()
