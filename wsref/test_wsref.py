"""Tests for wsref: unit checks plus agreement with numbers established on the ICIJ data.
Usage: python3 test_wsref.py [--icij]"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wsref

rng = np.random.default_rng(0)
# permute_within keeps per-source counts and only touches idx
y = (rng.random(1000) < 0.1).astype(int); s = rng.integers(0, 4, 1000); idx = np.arange(500)
yp = wsref.permute_within(y, s, rng, idx)
assert all(yp[idx][s[idx] == k].sum() == y[idx][s[idx] == k].sum() for k in range(4)) and (yp[500:] == y[500:]).all()
# p-value and rule
assert wsref.p_value(1.0, np.zeros(99)) == 0.01 and wsref.p_value(0.0, np.zeros(99)) == 1.0
assert wsref.min_draws(0.05) == 39 and wsref.min_draws(0.05, 4) == 159
nul = rng.normal(0.5, 0.05, 999)
assert wsref.read_balanced(0.9, nul) == "transfers" and wsref.read_balanced(0.1, nul) == "reversal" and wsref.read_balanced(0.52, nul) == "inconclusive"
assert np.allclose(wsref.holm([0.01, 0.04, 0.03, 0.002]), [0.03, 0.06, 0.06, 0.008])
# closed form: constant r gives 1/2; matches brute-force AUC on a simulated permuted population
assert abs(wsref.incorpus_reference_closed_form(np.full(100, 0.03), 0.03) - 0.5) < 1e-12
r = rng.choice([0.01, 0.05], 200000, p=[0.6, 0.4]); lab = rng.random(len(r)) < r
from scipy.stats import rankdata
rk = rankdata(r); auc = (rk[lab].sum() - lab.sum() * (lab.sum() + 1) / 2) / (lab.sum() * (~lab).sum())
cf = wsref.incorpus_reference_closed_form(r[:20000], r.mean())
assert abs(auc - cf) < 0.01, (auc, cf)
# reference() under H0: p-values roughly uniform
def evaluate(labels):  # a 'pipeline' whose statistic ignores features: mean label in a fixed half
    return labels[:500].mean()
ref = wsref.reference(evaluate, y, s, B=199, seed=1)
assert 0 < wsref.p_value(evaluate(y), ref.null) <= 1
# held-out-label test: exact under no association, detects a real one; rule needs both tests
sc = rng.random(2000); lab0 = rng.random(2000) < 0.1
pa0 = wsref.heldout_label_test(sc, lab0, M=999, seed=2); assert 0 < pa0[0] <= 1 and 0 < pa0[1] <= 1
lab1 = rng.random(2000) < 0.05 + 0.2 * sc
pa1 = wsref.heldout_label_test(sc, lab1, M=999, seed=3); assert pa1[0] <= 0.002, pa1
assert wsref.read_transfer((0.01, 1.0), pa1) == "transfers" and wsref.read_transfer((0.5, 0.5), pa1) == "inconclusive"
assert wsref.read_transfer((0.01, 1.0), (0.4, 0.6)) == "inconclusive" and wsref.read_transfer((1.0, 0.01), (1.0, 0.001)) == "reversal"
assert wsref.combined_p2((0.01, 1.0), (0.001, 1.0)) == 0.02
# p-values of the held-out test are approximately uniform under no association
ps = [wsref.heldout_label_test(rng.random(300), rng.random(300) < 0.2, M=199, seed=k)[0] for k in range(200)]
assert 0.02 < np.mean(np.array(ps) <= 0.05) < 0.10
print("unit tests passed; closed form", round(cf, 4), "vs brute force", round(auc, 4))

if "--icij" in sys.argv:
    # Holm readings on the 999-draw balanced nulls must reproduce the paper's Table 5
    E = os.environ.get("RESULTS_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"))
    b99 = json.load(open(f"{E}/full8_bal99.json"))
    L = ["Panama Papers", "Paradise Papers", "Offshore Leaks", "Bahamas Leaks"]
    ps = []
    for k, l in enumerate(L):
        n = np.array(b99[l]["null_draws"] + json.load(open(f"{E}/bal900_{k}.json")))
        ps.append(min(1, 2 * wsref.p_value(b99[l]["real_draws"][0], n)))
        assert wsref.read_balanced(b99[l]["real_draws"][0], n) == "transfers"
    adj = wsref.holm(ps)
    assert np.allclose(adj, [0.008, 0.008, 0.016, 0.034]), adj
    print("ICIJ Holm-adjusted balanced p (training test):", np.round(adj, 3))
    # combined rule with the held-out-label test of the same first fit (paper Table 5)
    ha = json.load(open(f"{E}/heldout_assoc_icij.json")); comb = []
    for k, l in enumerate(L):
        n = np.array(b99[l]["null_draws"] + json.load(open(f"{E}/bal900_{k}.json"))); r0 = b99[l]["real_draws"][0]
        f0 = ha[l]["fits"][0]; assert abs(f0["auc"] - r0) < 1e-12
        pt, pa = wsref.two_tailed(r0, n), (f0["p_up"], f0["p_down"])
        assert wsref.read_transfer(pt, pa) == "transfers"; comb.append(wsref.combined_p2(pt, pa))
    adj = wsref.holm(comb)
    assert np.allclose(adj, [0.008, 0.008, 0.016, 0.034]), adj
    print("ICIJ Holm-adjusted combined p:", np.round(adj, 3), "(paper Table 5: 0.008 0.008 0.016 0.034)")
