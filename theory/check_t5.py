"""Factorisation checks with a strong shared signal g that has unit mean in every source (bounded trigonometric basis)."""
import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score
rng = np.random.default_rng(2)
mus, q, pis = np.array([-1.0, 0.0, 1.5]), np.array([0.5, 0.3, 0.2]), np.array([0.01, 0.02, 0.06])
fr = [(np.sin, 1.0), (np.cos, 1.0), (np.sin, 0.5), (np.cos, 0.5), (np.sin, 0.25), (np.cos, 0.25)]
grid = np.linspace(-10, 10, 40001); dg = grid[1] - grid[0]
Bg = np.stack([f(w * grid) for f, w in fr])
M = np.stack([(norm.pdf(grid, m, 1) * Bg).sum(1) * dg for m in mus])        # E_s[basis]
c = np.linalg.svd(M)[2][-1]
hg = c @ Bg; a = 0.9 / np.abs(hg[np.abs(grid) < 6]).max()
def g_of(t): return 1 + a * (c @ np.stack([f(w * t) for f, w in fr]))
N = 1_500_000
s = rng.choice(3, N, p=q); x = rng.normal(mus[s], 1.0); g = g_of(x)
print("min g", round(g.min(), 3), "max g", round(g.max(), 3), "| E_s[g]:", [round(g[s == k].mean(), 4) for k in range(3)])
pdf = np.stack([norm.pdf(x, m, 1) for m in mus], 1); post = q * pdf; post /= post.sum(1, keepdims=True)
r = post @ pis; lam = (pis[s] * g).max(); eps = -np.log(1 - lam)
lab = rng.random(N) < pis[s] * g
print(f"labelled rates by source: {[round(lab[s==k].mean(),4) for k in range(3)]} (pi_s {pis.tolist()})")
pooled = (post * pis).sum(1) * g
bal = np.log(pdf.sum(1) * g / (pdf * (1 - pis * g[:, None]) / (1 - pis)).sum(1))
d = bal - np.log(g); d = d - np.median(d)
print(f"T5(a): AUC pooled {roc_auc_score(lab, pooled):.4f} | AUC(r*g) {roc_auc_score(lab, r*g):.4f} | AUC(r) {roc_auc_score(lab, r):.4f} | AUC(g) {roc_auc_score(lab, g):.4f}")
A_bal, A_g = roc_auc_score(lab, bal), roc_auc_score(lab, np.log(g))
P, U = np.log(g[lab]), np.log(g[~lab]); pp, uu = rng.choice(P, 400000), rng.choice(U, 400000)
print(f"T5(c): lambda {lam:.3f}, eps {eps:.4f}; max|delta| {np.abs(d).max():.4f} <= eps: {np.abs(d).max() <= eps}; "
      f"|AUC(bal)-AUC(g)| {abs(A_bal-A_g):.5f} <= P(|dlog g|<=2eps) {np.mean(np.abs(pp-uu) <= 2*eps):.4f}")
# T5(d) SCAR identity within source 2, large sample, several prevalences
for th in [0.1, 0.2]:
    k = 2; ck = pis[k] / th; xs = rng.normal(mus[k], 1.0, 2_000_000); gs = g_of(xs)
    yk = rng.random(len(xs)) < th * gs; lk = yk & (rng.random(len(xs)) < ck)
    gam = (yk & ~lk).sum() / (~lk).sum()
    lhs = roc_auc_score(lk, gs) - 0.5; rhs = (1 - gam) * (roc_auc_score(yk, gs) - 0.5)
    print(f"T5(d) theta={th}: AUC_PU-1/2 {lhs:.4f} vs (1-gamma)(AUC-1/2) {rhs:.4f} (gamma {gam:.3f})")
# unequal mean signal: E_s[g] differs -> pooled ranks by r_b * g with b_s = pi_s / E_s[g]
g2 = np.exp(0.6 * x); e = np.array([g2[s == k].mean() for k in range(3)]); b = pis / e
lab2 = rng.random(N) < np.clip(b[s] * g2, 0, 1)
rb = post @ b
print(f"general case: corr(rank pooled, rank r_b*g) = {np.corrcoef(np.argsort(np.argsort((post*b).sum(1)*g2)), np.argsort(np.argsort(rb*g2)))[0,1]:.4f}; "
      f"labelled rates {[round(lab2[s==k].mean(),4) for k in range(3)]} vs b_s*E_s[g] {np.round(b*e,4).tolist()}")
