"""Test Theorem 4's equal-mean-signal assumption on ICIJ (eight structural features).
Under P(l=1|x,s) = b_s g(x), a source-balanced model estimates g up to a constant, so ratios of e_s = E_{p_s}[g]
across leaks are identified. Labelled rates factor as pi_s = b_s e_s. We report e_s (relative), and recompute the
in-corpus closed-form reference (Theorem 2) and the held-out plug-in references with signal-adjusted rates
pi'_s = b_s * e_bar (what the rates would be if every leak had the pool's mean signal).
Usage: python3 assumption_check.py <out.json>"""
import sys, json, numpy as np, pandas as pd
from config import WORK, ICIJ
from evallib import *
from sklearn.metrics import roc_auc_score

df = pd.read_parquet(WORK + "/officers.parquet"); y = df.y.values; leak = df.leak.values
F8 = ["ent_deg", "log_ent_deg", "n_addr", "n_inter", "n_jur", "role_div", "ent_per_addr", "ent_per_inter"]
X = df[F8].values.astype(np.float32); N = len(y); allidx = np.arange(N)
q = {L: (leak == L).mean() for L in LEAKS}; pi = {L: y[leak == L].mean() for L in LEAKS}
out = {"q": q, "pi": pi}

# 1) g-hat from source-balanced models trained on all four leaks (5 fits), evaluated on a uniform sample of each leak
rng = np.random.default_rng(42)
evs = {L: rng.choice(np.where(leak == L)[0], min(40000, (leak == L).sum()), replace=False) for L in LEAKS}
logits = {L: [] for L in LEAKS}
for f in range(5):
    P, Nn = train_set(y, leak, allidx, rng, balanced=True)
    m = gbm(int(rng.integers(1 << 30))).fit(X[np.r_[P, Nn]], np.r_[np.ones(len(P)), np.zeros(len(Nn))])
    for L in LEAKS:
        p = np.clip(m.predict_proba(X[evs[L]])[:, 1], 1e-6, 1 - 1e-6); logits[L].append(np.log(p / (1 - p)))
lg = {L: np.mean(logits[L], 0) for L in LEAKS}
def e_of(transform):
    v = {L: transform(lg[L]) for L in LEAKS}; ebar = sum(q[L] * v[L] for L in LEAKS)
    return {L: v[L] / ebar for L in LEAKS}
e_mean = e_of(lambda z: np.mean(np.exp(z)))
e_wins = e_of(lambda z: np.mean(np.exp(np.clip(z, *np.quantile(z, [0.01, 0.99])))))
out["e_relative_mean"] = e_mean; out["e_relative_winsorised"] = e_wins
out["e_ratio_max_min"] = max(e_mean.values()) / min(e_mean.values())
print("relative mean signal e_s (pool = 1):", {L: round(v, 3) for L, v in e_mean.items()},
      "| winsorised:", {L: round(v, 3) for L, v in e_wins.items()}, "| max/min", round(out["e_ratio_max_min"], 2), flush=True)

# 2) in-corpus closed form with actual rates vs signal-adjusted rates
def closed_form(rates, rng):
    samp = rng.choice(allidx, 60000, replace=False)
    cls, pr = source_posteriors(X, leak, allidx, samp, rng)
    rr = pr @ np.array([rates[c] for c in cls]); pib = sum(q[c] * rates[c] for c in cls)
    rs = np.sort(rr); k = len(rs); i = np.arange(1, k + 1)
    return 0.5 + 2 * np.sum((2 * i - k - 1) * rs) / (k * (k - 1)) / (4 * pib * (1 - pib))
for name, e in [("mean", e_mean), ("winsorised", e_wins)]:
    adj = {L: pi[L] / e[L] for L in LEAKS}                     # b_s * e_bar with e_bar = 1
    out[f"rates_adjusted_{name}"] = adj
    out[f"incorpus_closed_form_actual"] = closed_form(pi, np.random.default_rng(7))
    out[f"incorpus_closed_form_adjusted_{name}"] = closed_form(adj, np.random.default_rng(7))
    print(f"[{name}] adjusted rates {dict((L, round(v, 4)) for L, v in adj.items())} | in-corpus reference: actual rates "
          f"{out['incorpus_closed_form_actual']:.3f}, signal-adjusted {out[f'incorpus_closed_form_adjusted_{name}']:.3f}", flush=True)

# 3) held-out plug-in references with actual vs adjusted rates
held = {}
for k, L in enumerate(LEAKS):
    pool, ep, eu = lolo_sets(y, leak, L, np.random.default_rng(13))
    rr = np.random.default_rng(77 + k)
    cls, pr = source_posteriors(X, leak, pool, np.concatenate([ep, eu]), rr)
    lab = np.r_[np.ones(len(ep)), np.zeros(len(eu))]
    a_act = roc_auc_score(lab, pr @ np.array([pi[c] for c in cls]))
    a_adj = roc_auc_score(lab, pr @ np.array([pi[c] / e_mean[c] for c in cls]))
    held[L] = {"actual": a_act, "adjusted": a_adj}
    print(f"held-out {L:16s} plug-in reference: actual rates {a_act:.3f} | signal-adjusted {a_adj:.3f}", flush=True)
out["heldout_plugin"] = held
json.dump(out, open(sys.argv[1], "w"), indent=1, default=float)
