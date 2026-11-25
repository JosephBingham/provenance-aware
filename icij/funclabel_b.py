"""(b) corrected: name-trained 8-feature model (pooled, all leaks) scoring name-independent functional nominees against
Offshore Leaks officers. Offshore Leaks non-functional officers are split in half: one half may be sampled as training
pseudo-negatives, the other half is the evaluation set. Five splits."""
import sys, json, numpy as np, pandas as pd
from config import WORK, ICIJ, RESULTS
from evallib import *
from sklearn.metrics import roc_auc_score
rel = pd.read_csv(ICIJ + "/relationships.csv", dtype=str, keep_default_na=False, usecols=["node_id_start", "rel_type", "link", "sourceID"])
nom = rel[(rel.rel_type == "underlying") & rel.link.str.contains("nominee", case=False) & (rel.sourceID != "")]
df = pd.read_parquet(WORK + "/officers.parquet")
fn = df.officer.isin(set(nom.node_id_start)).values.astype(int); leak = df.leak.values; yname = df.y.values
F8 = ["ent_deg", "log_ent_deg", "n_addr", "n_inter", "n_jur", "role_div", "ent_per_addr", "ent_per_inter"]
X = df[F8].values.astype(np.float32)
ol = np.where(leak == "Offshore Leaks")[0]
pos = ol[(fn[ol] == 1) & (yname[ol] == 0)]; olneg = ol[(fn[ol] == 0) & (yname[ol] == 0)]
P = np.where((yname == 1) & (fn == 0))[0]
sc, deg = [], []
for s in range(5):
    r = np.random.default_rng(20 + s); perm = r.permutation(olneg); ev, trn = perm[: len(perm) // 2], perm[len(perm) // 2:]
    U = np.r_[np.where((yname == 0) & (leak != "Offshore Leaks"))[0], trn]
    N = r.choice(U, 3 * len(P), replace=False)
    m = gbm(s).fit(X[np.r_[P, N]], np.r_[np.ones(len(P)), np.zeros(len(N))])
    lab = np.r_[np.ones(len(pos)), np.zeros(len(ev))]
    sc.append(roc_auc_score(lab, m.predict_proba(X[np.r_[pos, ev]])[:, 1]))
    deg.append(roc_auc_score(lab, df.ent_deg.values[np.r_[pos, ev]]))
print("name-trained model:", np.round(sc, 3), "mean", round(float(np.mean(sc)), 3), "| raw degree mean", round(float(np.mean(deg)), 3))
res = json.load(open(RESULTS + "/funclabel.json")); res["name_model_on_functional"] = float(np.mean(sc)); res["name_model_on_functional_draws"] = sc
res["raw_degree_on_functional"] = float(np.mean(deg)); res["note_b"] = "corrected split design (funclabel_b.py)"
json.dump(res, open(RESULTS + "/funclabel.json", "w"), indent=1, default=float)
