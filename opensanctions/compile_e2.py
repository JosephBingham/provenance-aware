"""Summarise E2 results: in-corpus table, held-out table with two-sided balanced p (Holm across sources).
Usage: python3 compile_e2.py <results_dir>"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sys, json, numpy as np
R = sys.argv[1]
S = ["ext_ru_egrul", "us_sam_exclusions", "wikidata", "gb_coh_disqualified", "ru_acf_bribetakers"]
out = {}
for rep in ["struct", "struct_topics", "rdf2vec"]:
    d = json.load(open(f"{R}/{rep}.json")); o = {"source_acc": d["source_acc"], "majority": d["majority"], "thm2_pred": d["thm2_pred"],
         "in_U": [d["incorpus_uniform"]["real"], d["incorpus_uniform"]["null"]], "in_M": [d["incorpus_matched"]["real"], d["incorpus_matched"]["null"]]}
    ps = {}
    for s in S:
        lo, b = d[f"lolo_{s}"], d[f"lolo_bal_{s}"]; n = np.array(b["null_draws"]); x = b["real_draws"][0]; B = len(n)
        pu = (1 + (n >= x).sum()) / (B + 1); pl = (1 + (n <= x).sum()) / (B + 1); p2 = min(1.0, 2 * min(pu, pl)); ps[s] = p2
        no = np.array(lo["null_draws"]); xr = lo["real_draws"][0]
        o[s] = dict(pooled=lo["real"], ref=lo["null"], ref_lo=lo["null_lo"], ref_hi=lo["null_hi"], p_up=lo["p"],
                    p_down=(1 + (no <= xr).sum()) / (len(no) + 1), B=lo["B"], source_only=lo["source_only"], ordinary=lo["ordinary_perm"],
                    bal_first=x, bal_median=float(np.median(b["real_draws"])), bal_null_mean=float(n.mean()),
                    bal_q025=float(np.quantile(n, .025)), bal_q975=float(np.quantile(n, .975)), bal_B=B, p2=p2,
                    n_pos=lo["n_pos"], base_rate=lo["base_rate"], ap_real=lo["real_ap"], ap_ref=lo["null_ap"])
    order = sorted(S, key=lambda s: ps[s]); run = 0.0
    for i, s in enumerate(order):
        run = max(run, min(1.0, (len(S) - i) * ps[s])); o[s]["holm"] = run
    for s in S:
        x = o[s]; o[s]["reading"] = ("transfers" if x["bal_first"] > 0.5 else "reversal") if x["holm"] <= 0.05 else "inconclusive"
    out[rep] = o
    print(f"\n== {rep}: leak acc {o['source_acc']:.3f} (majority {o['majority']:.3f}); in-corpus U {o['in_U'][0]:.3f} ref {o['in_U'][1]:.3f} "
          f"closed form {o['thm2_pred']:.3f}; matched {o['in_M'][0]:.3f} ref {o['in_M'][1]:.3f}")
    for s in S:
        x = o[s]
        print(f"  {s:20s} pooled {x['pooled']:.3f} ref {x['ref']:.3f} [{x['ref_lo']:.3f},{x['ref_hi']:.3f}] p_up {x['p_up']:.2f} p_down {x['p_down']:.2f} "
              f"src-only {x['source_only']:.3f} ord {x['ordinary']:.3f} | bal {x['bal_median']:.3f} null {x['bal_null_mean']:.3f} "
              f"[{x['bal_q025']:.3f},{x['bal_q975']:.3f}] p2 {x['p2']:.3f} holm {x['holm']:.3f} {x['reading']} | AP {x['ap_real']:.3f}/{x['ap_ref']:.3f} base {x['base_rate']:.3f}")
json.dump(out, open(f"{R}/e2_summary.json", "w"), indent=1, default=float)
