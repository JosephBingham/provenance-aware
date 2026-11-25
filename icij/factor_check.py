"""Empirical check of the factorisation P(l=1|x) = r(x) g(x) on ICIJ, eight structural features.
For each held-out leak (training on the other three): pooled model log-odds, source-balanced model log-odds (~ log g),
plug-in log r_hat. Reports the OLS fit of pooled log-odds on (log r_hat, balanced log-odds), the AUC of the factorised
score log r_hat + balanced log-odds against the pooled model's AUC, and the mean shifts behind the masking corollary.
Usage: python3 factor_check.py <out.json>"""
import sys, json, numpy as np, pandas as pd
from config import WORK, ICIJ
from evallib import *
from sklearn.metrics import roc_auc_score

df = pd.read_parquet(WORK + "/officers.parquet"); y = df.y.values; leak = df.leak.values
F8 = ["ent_deg", "log_ent_deg", "n_addr", "n_inter", "n_jur", "role_div", "ent_per_addr", "ent_per_inter"]
X = df[F8].values.astype(np.float32)
logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6)) - np.log(np.clip(1 - p, 1e-6, 1))
out = {}
for k, L in enumerate(LEAKS):
    pool, ep, eu = lolo_sets(y, leak, L, np.random.default_rng(13))
    ev = np.concatenate([ep, eu]); lab = np.r_[np.ones(len(ep)), np.zeros(len(eu))]
    rng = np.random.default_rng(900 + k)
    def fit_avg(balanced, n=3):
        Z = []; aucs = []
        for _ in range(n):
            P, N = train_set(y, leak, pool, rng, balanced=balanced)
            m = gbm(int(rng.integers(1 << 30))).fit(X[np.r_[P, N]], np.r_[np.ones(len(P)), np.zeros(len(N))])
            z = logit(m.predict_proba(X[ev])[:, 1]); Z.append(z); aucs.append(roc_auc_score(lab, z))
        return np.mean(Z, 0), aucs
    zp, auc_p = fit_avg(False); zb, auc_b = fit_avg(True)
    cls, pr = source_posteriors(X, leak, pool, ev, rng)
    pi = np.array([y[pool][leak[pool] == c].mean() for c in cls]); lr = np.log(pr @ pi)
    A = np.c_[np.ones(len(ev)), lr, zb]; coef, *_ = np.linalg.lstsq(A, zp, rcond=None)
    pred = A @ coef; r2 = 1 - np.sum((zp - pred) ** 2) / np.sum((zp - zp.mean()) ** 2)
    f = lr + zb
    pos = lab == 1
    sh = lambda v: float(v[pos].mean() - v[~pos].mean())
    res = dict(auc_pooled_fits=auc_p, auc_pooled_avg=float(roc_auc_score(lab, zp)), auc_balanced_fits=auc_b,
               auc_balanced_avg=float(roc_auc_score(lab, zb)), auc_r=float(roc_auc_score(lab, lr)),
               auc_factorised=float(roc_auc_score(lab, f)), ols_intercept=float(coef[0]), ols_log_r=float(coef[1]),
               ols_balanced=float(coef[2]), r2=float(r2), corr_pooled_factorised=float(np.corrcoef(zp, f)[0, 1]),
               shift_log_r=sh(lr), shift_log_g=sh(zb), sd_log_r=float(lr[~pos].std()), sd_log_g=float(zb[~pos].std()))
    from scipy.stats import norm
    su, sv = lr[~pos].std(), zb[~pos].std(); rho = np.corrcoef(lr[~pos], zb[~pos])[0, 1]
    pa, pb = norm.ppf(res["auc_r"]), norm.ppf(res["auc_balanced_avg"])
    res.update(corr_logr_bal=float(np.corrcoef(lr, zb)[0, 1]), rho_unl=float(rho),
               binormal_pred=float(norm.cdf((su * pa + sv * pb) / np.sqrt(su**2 + sv**2 + 2 * rho * su * sv))),
               provenance_pull=float(su * pa), signal_push=float(sv * pb), n_distinct_log_r=int(len(np.unique(np.round(lr, 6)))))
    out[L] = res
    print(f"   corr(log r, bal) {res['corr_logr_bal']:+.2f}; sd log r {su:.2f}, sd log g {sv:.2f}; provenance pull {su*pa:+.3f} vs signal push {sv*pb:+.3f}; "
          f"rank-binormal prediction {res['binormal_pred']:.3f} vs factorised {res['auc_factorised']:.3f}; distinct log r values {res['n_distinct_log_r']}", flush=True)
    print(f"{L:16s} AUC pooled {res['auc_pooled_avg']:.3f} (fits {np.round(auc_p,3)}) | r {res['auc_r']:.3f} | balanced {res['auc_balanced_avg']:.3f} "
          f"(fits {np.round(auc_b,3)}) | factorised {res['auc_factorised']:.3f} | OLS log r {coef[1]:.2f}, bal {coef[2]:.2f}, R2 {r2:.2f} "
          f"| shifts log r {res['shift_log_r']:+.2f}, log g {res['shift_log_g']:+.2f}", flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
