"""Score decision rules on the simulated settings. Usage: python3 analyze.py ext_0.jsonl ext_1.jsonl  (writes table.json to the working directory)"""
import sys, json, numpy as np, collections
recs = [json.loads(l) for f in sys.argv[1:] for l in open(f)]
print("settings:", len(recs))
TRUTH = {"shared": "transfers", "reversed": "reversal", "absent": "inconclusive"}

def naive(r):
    x = r["pool_real"]; return "transfers" if x > 0.55 else "reversal" if x < 0.45 else "inconclusive"
def sd_rule(r, B):
    n = np.array(r["bal_null"][:B]); x = r["bal_real"]; m, s = n.mean(), n.std(ddof=1)
    if x > 0.5 and x > m + 2 * s: return "transfers"
    if x < 0.5 and x < m - 2 * s: return "reversal"
    return "inconclusive"
def perm_rule(r, B=19, alpha=0.05, two_sided=False):
    if two_sided: alpha = alpha / 2
    n = np.array(r["bal_null"][:B]); x = r["bal_real"]
    pu = (1 + (n >= x).sum()) / (B + 1); pl = (1 + (n <= x).sum()) / (B + 1)
    if x > 0.5 and pu <= alpha: return "transfers"
    if x < 0.5 and pl <= alpha: return "reversal"
    return "inconclusive"
RULES = {"against chance": naive, "2-SD rule, 9 draws": lambda r: sd_rule(r, 9),
         "2-SD rule, 39 draws": lambda r: sd_rule(r, 39),
         "permutation, 0.05 per tail, 19 draws": perm_rule,
         "permutation, 0.025 per tail, 39 draws": lambda r: perm_rule(r, 39, 0.05, True)}

def outcome(call, truth):
    if call == truth: return "correct"
    if truth == "inconclusive": return "false detection"
    if call == "inconclusive": return "missed"
    return "wrong direction"

short = [abs(r["pop_ref"] - 0.5) > 0.03 for r in recs]
flag = [abs(np.mean(r["pool_null"]) - 0.5) > 0.03 for r in recs]
print("shortcut present (population):", sum(short), "| fitted reference flags", sum(f and s for f, s in zip(flag, short)),
      "of them; false flags", sum(f and not s for f, s in zip(flag, short)), "of", len(recs) - sum(short))
cols = ["correct", "wrong direction", "false detection", "missed"]
rows = []
for scope, sel in [("shortcut present", short), ("all settings", [True] * len(recs))]:
    for name, fn in RULES.items():
        c = collections.Counter(outcome(fn(r), TRUTH[r["rel"]]) for r, s in zip(recs, sel) if s)
        rows.append((scope, name, [c[k] for k in cols], sum(sel)))
        print(f"{scope:17s} {name:28s}", [c[k] for k in cols], "n", sum(sel))
# false-detection rate among no-signal targets, and power by signal strength
print("\nfalse 'transfers'/'reversal' calls when the target has no signal (absent), by rule:")
for name, fn in RULES.items():
    a = [fn(r) != "inconclusive" for r in recs if r["rel"] == "absent"]
    a2 = [fn(r) != "inconclusive" for r, s in zip(recs, short) if r["rel"] == "absent" and s]
    print(f"  {name:28s} {np.mean(a):.3f} (n={len(a)}); with shortcut present {np.mean(a2):.3f} (n={len(a2)})")
print("\ncorrect detection of a real transfer or reversal, by signal strength (shortcut present | absent):")
for name, fn in RULES.items():
    out = []
    for d in [0.25, 0.5, 1.0]:
        for sc in [True, False]:
            v = [fn(r) == TRUTH[r["rel"]] for r, s in zip(recs, short) if r["rel"] != "absent" and r["delta"] == d and s == sc]
            w = [fn(r) not in ("inconclusive", TRUTH[r["rel"]]) for r, s in zip(recs, short) if r["rel"] != "absent" and r["delta"] == d and s == sc]
            out.append(f"d={d} {'S' if sc else 'N'}: det {np.mean(v):.2f} wrong {np.mean(w):.2f} (n={len(v)})")
    print(f"  {name}\n    " + "\n    ".join(out))
# bimodality of balanced nulls: share of settings where the 97.5% quantile exceeds mean+2SD region asymmetrically
bm = [np.quantile(r["bal_null"], .95) - np.mean(r["bal_null"]) for r in recs]
json.dump({"rows": rows}, open("table.json", "w"))
