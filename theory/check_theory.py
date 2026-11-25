"""Numerical checks of the new results on populations where everything is known.
T5  factorisation: pooled posterior = r*g; balanced log-odds within eps of log g; AUC perturbation bound.
P8  in-corpus reference <= dispersion x identifiability bound (and the TV form equals the E|P(s|X)-q_s| form).
P9  plug-in stability bound holds when posteriors and rates are perturbed.
C7  two-sided rule under H0 calls transfer/reversal at most alpha (Monte Carlo, logistic-regression pipeline)."""
import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
rng = np.random.default_rng(1)

def gmd(v):
    v = np.sort(v); n = len(v); i = np.arange(1, n + 1)
    return 2 * np.sum((2 * i - n - 1) * v) / (n * (n - 1))

# ---------- a 1-d population: 3 sources, Gaussian densities, shared signal g with unit mean in every source ----------
mus, q = np.array([-1.0, 0.0, 1.5]), np.array([0.5, 0.3, 0.2])
pis = np.array([0.01, 0.02, 0.06])
N = 400_000
s = rng.choice(3, N, p=q); x = rng.normal(mus[s], 1.0)
# g(x) = exp(beta*(x - mu_s)) / E_s[exp(beta*(X-mu_s))] would depend on s; to get one g with unit mean in EVERY source we use
# g(x) = 1 + a*sin(w x) with w chosen so E_s[sin(wX)] = 0 is not possible in general -> use g(x) = 1 + a*h(x) with h having
# zero mean under every source: h(x) = H2-like polynomial combination solved numerically on a grid.
grid = np.linspace(-8, 8, 20001); dens = np.stack([norm.pdf(grid, m, 1) for m in mus])
basis = np.stack([grid, grid**2, grid**3, np.sin(grid)])           # find h = c.basis with E_s[h]=0 for all s
M = (dens[:, None, :] * basis[None]).sum(-1) * (grid[1] - grid[0])  # 3 x 4
c = np.linalg.svd(M)[2][-1]                                          # null-space vector
h = lambda t: np.stack([t, t**2, t**3, np.sin(t)]).T @ c
a = 0.6 / np.max(np.abs(h(grid[np.abs(grid) < 4])))
g = np.clip(1 + a * h(x), 0.05, None)
print("E_s[g] by source:", [round(g[s == k].mean(), 3) for k in range(3)], "(should all be ~1)")
p_lab_xs = pis[s] * g                                                  # P(l=1|x,s)
lam = p_lab_xs.max(); eps = -np.log(1 - lam)
ps = np.stack([q[k] * norm.pdf(x, mus[k], 1) for k in range(3)], 1); post = ps / ps.sum(1, keepdims=True)
r = post @ pis
# pooled population posterior P(l=1|x) = sum_s P(s|x) pi_s g(x)
pooled = (post * pis).sum(1) * g
assert np.allclose(pooled, r * g), "factorisation failed"
# balanced population log-odds: positives (1/S) sum_s p_s(x) g(x) ; negatives (1/S) sum_s p_s(x)(1-pi_s g)/(1-pi_s)
pdf = np.stack([norm.pdf(x, mus[k], 1) for k in range(3)], 1)
bal = np.log((pdf.sum(1) * g) / (pdf * (1 - pis * g[:, None]) / (1 - pis)).sum(1))
delta = bal - np.log(g)
print(f"T5(c): max |delta - median| {np.max(np.abs(delta - np.median(delta))):.5f} <= eps {eps:.5f}: {np.max(np.abs(delta - np.median(delta))) <= eps}")
lab = rng.random(N) < p_lab_xs
A_bal, A_g = roc_auc_score(lab, bal), roc_auc_score(lab, np.log(g))
P, U = np.log(g[lab]), np.log(g[~lab]); pp = rng.choice(P, 200000); uu = rng.choice(U, 200000)
bound = np.mean(np.abs(pp - uu) <= 2 * eps)
print(f"T5(c): |AUC(bal) - AUC(g)| = {abs(A_bal - A_g):.5f} <= P(|dlog g| <= 2eps) = {bound:.4f}: {abs(A_bal - A_g) <= bound}")
print(f"T5(a): AUC(pooled) {roc_auc_score(lab, pooled):.4f} = AUC(r*g) {roc_auc_score(lab, r*g):.4f}; AUC(r) {roc_auc_score(lab, r):.4f}")
# SCAR identity within one source (source 2): labelled vs unlabelled AUC of g vs true AUC of g
k = 2; th = 0.2; c_k = pis[k] / th
xs = x[s == k]; gs = g[s == k]; yk = rng.random(len(xs)) < th * gs; lk = yk & (rng.random(len(xs)) < c_k)
gam = (yk & ~lk).sum() / (~lk).sum()
lhs = roc_auc_score(lk, gs) - 0.5; rhs = (1 - gam) * (roc_auc_score(yk, gs) - 0.5)
print(f"T5(d): AUC_PU - 1/2 = {lhs:.4f} vs (1-gamma)(AUC - 1/2) = {rhs:.4f}")

# ---------- P8 bound ----------
pib = (q * pis).sum()
A3 = 0.5 + gmd(rng.choice(r, 60000)) / (4 * pib * (1 - pib))
dev = np.abs(post - q).mean(0)
B8 = 0.5 + np.sum(np.abs(pis - pib) * dev) / (2 * pib * (1 - pib))
m = (q * pdf).sum(1); tv = np.array([0.5 * np.mean(np.abs(pdf[:, k] / m - 1)) for k in range(3)])  # TV(p_s, m) = E_m|p_s/m - 1|/2
B8tv = 0.5 + np.sum(q * np.abs(pis - pib) * tv) / (pib * (1 - pib))
print(f"P8: reference {A3:.4f} <= bound {B8:.4f} (TV form {B8tv:.4f})")
# brute force check of Theorem 3 on the same population
yb = rng.random(N) < r
print(f"    Theorem 3 closed form {A3:.4f} vs brute-force AUC(r) on permuted population {roc_auc_score(yb, r):.4f}")

# ---------- P9 plug-in stability ----------
for noise in [0.02, 0.05, 0.1]:
    ph = np.clip(post + rng.normal(0, noise, post.shape), 1e-4, None); ph /= ph.sum(1, keepdims=True)
    pih = pis * (1 + rng.uniform(-0.1, 0.1, 3))
    rh = ph @ pih; n = 60000; idx = rng.choice(N, n, replace=False)
    Ah = 0.5 + gmd(rh[idx]) / (4 * pib * (1 - pib))
    l1 = np.abs(ph - post).sum(1).mean(); eta = 0.05
    Rr = pih.max() - pih.min()
    B9 = (2 * (pih.max() * l1 + np.abs(pih - pis).max()) + Rr * np.sqrt(np.log(2 / eta) / (2 * (n // 2)))) / (4 * pib * (1 - pib))
    print(f"P9: noise {noise}: |A_hat - A| = {abs(Ah - A3):.4f} <= bound {B9:.4f}: {abs(Ah - A3) <= B9}")

# ---------- C7 rule validity under H0 (labels independent of x given s), logistic-regression pipeline ----------
def one_rep(seed, B=39):
    rg = np.random.default_rng(seed)
    S, n = 4, 3000
    src = np.repeat(np.arange(S), n); xx = rg.normal(src[:, None] * 0.8, 1.0, (S * n, 3))
    yy = (rg.random(S * n) < np.array([0.03, 0.06, 0.02, 0.04])[src]).astype(int)   # H0: y indep of x given s
    tr, te = src < 3, src == 3
    def stat(lbl):      # balanced pipeline
        idx = []
        mm = min(lbl[tr & (src == k)].sum() for k in range(3))
        for k in range(3):
            P = np.where(tr & (src == k) & (lbl == 1))[0]; U = np.where(tr & (src == k) & (lbl == 0))[0]
            idx += list(rg.choice(P, mm, replace=False)) + list(rg.choice(U, 3 * mm, replace=False))
        idx = np.array(idx)
        clf = LogisticRegression().fit(xx[idx], lbl[idx])
        return roc_auc_score(yy[te], clf.decision_function(xx[te]))
    t0 = stat(yy); null = []
    for _ in range(B):
        yp = yy.copy()
        for k in range(3):
            ii = np.where(tr & (src == k))[0]; yp[ii] = yy[ii][rg.permutation(len(ii))]
        null.append(stat(yp))
    null = np.array(null); pu = (1 + (null >= t0).sum()) / (B + 1); pl = (1 + (null <= t0).sum()) / (B + 1)
    return (t0 > 0.5 and pu <= 0.025) or (t0 < 0.5 and pl <= 0.025)
calls = [one_rep(sd) for sd in range(300)]
print(f"C7: false transfer/reversal calls under H0: {np.mean(calls):.3f} over 300 replicates (bound 0.05; MC s.e. {np.sqrt(0.05*0.95/300):.3f})")
