"""Figures for the ESWC draft. Usage: python3 -I figs.py <results_dir> <out_dir>"""
import sys, json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

R, OUT = sys.argv[1], sys.argv[2]
plt.rcParams.update({"font.family": "serif", "font.size": 7.0, "axes.titlesize": 7.0, "axes.labelsize": 6.5,
                     "xtick.labelsize": 6.0, "ytick.labelsize": 5.8, "legend.fontsize": 6.0,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
GREY, RED, BLUE, INK = "#9a9a9a", "#c0392b", "#1f5fa8", "#222222"

# ---------- Fig 1: mechanism, one-dimensional illustration ----------
rng = np.random.default_rng(3)
from scipy.stats import norm
xs = np.linspace(-4.6, 4.6, 600)
SRC = [("A", -3.1, 0.01), ("B", -1.3, 0.01), ("H", 2.5, 0.05)]       # name, centre, labelled rate
SD = 1.0
def score(rates):
    dens = np.array([norm.pdf(xs, mu, SD) for _, mu, _ in SRC]); post = dens / dens.sum(0)
    return np.array(rates) @ post
r_pooled = score([s[2] for s in SRC]); r_bal = score([0.02, 0.02, 0.02])
DELTA = 1.2
def heldout_auc(sgn):
    g = np.random.default_rng(5); xu = g.normal(0, 1, 200000); xp = g.normal(sgn * DELTA, 1, 200000)
    ru, rp = np.interp(xu, xs, r_pooled), np.interp(xp, xs, r_pooled)
    from scipy.stats import rankdata
    rk = rankdata(np.r_[rp, ru]); n1 = len(rp); return (rk[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * len(ru))
auc_toward, auc_away = heldout_auc(1), heldout_auc(-1)
print("Fig 1 held-out AUCs", round(auc_toward, 3), round(auc_away, 3))
LIGHT = {"A": "#d9d9d9", "B": "#d9d9d9", "H": "#f2c4bd"}
FS, FT, FB = 6.0, 6.6, 7.2          # text, panel title, AUC (pt at print size: the figure is drawn at \textwidth)
fig, axs = plt.subplots(1, 4, figsize=(4.8, 1.45), sharex=True)
def score_curve(ax, r, label):
    rr = (r - r_pooled.min()) / (r_pooled.max() - r_pooled.min())
    ax.plot(xs, 0.82 + 0.30 * rr, color=INK, lw=1.0)
    ax.text(-4.5, 1.17, label, fontsize=FS, color=INK, va="bottom")
def training(ax, rates_txt, rug_rates):
    for (nm, mu, _), rt, rr in zip(SRC, rates_txt, rug_rates):
        d = norm.pdf(xs, mu, SD); d = 0.50 * d / d.max()
        ax.fill_between(xs, 0, d, color=LIGHT[nm] if rr > 0.03 else "#d9d9d9", lw=0)
        pos = rng.normal(mu, SD, int(round(rr * 400)))
        ax.vlines(pos, 0.0, 0.13, color=RED, lw=0.6)
        ax.text(mu, 0.53, f"{nm}\n{rt}", ha="center", va="bottom", fontsize=FS, color=INK, linespacing=0.95)
# (a)
ax = axs[0]; training(ax, ["1%", "1%", "5%"], [0.01, 0.01, 0.05])
score_curve(ax, r_pooled, "score: “looks like H”")
ax.set_title("(a) Training sources", fontsize=FT)
ax.text(0, -0.10, "red: labelled positives", ha="center", va="top", fontsize=FS, color=RED)
# (b), (c)
for ax, sgn, auc, title in [(axs[1], 1, auc_toward, "(b) Held-out source,\npositives resemble H"),
                            (axs[2], -1, auc_away, "(c) Held-out source,\npositives resemble A, B")]:
    u = norm.pdf(xs, 0, 1); p = norm.pdf(xs, sgn * DELTA, 1)
    ax.fill_between(xs, 0, 0.50 * u / u.max(), color="#d9d9d9", lw=0)
    ax.plot(xs, 0.50 * p / p.max(), color=BLUE, lw=1.2)
    ax.text(-4.5 if sgn > 0 else 4.5, 0.36, "positives", color=BLUE, fontsize=FS, ha="left" if sgn > 0 else "right")
    ax.text(4.5 if sgn > 0 else -4.5, 0.36, "others", color="#707070", fontsize=FS, ha="right" if sgn > 0 else "left")
    score_curve(ax, r_pooled, "score from (a)")
    ax.text(0, -0.10, f"AUC = {auc:.2f}", ha="center", va="top", fontsize=FB, fontweight="bold", color=RED if auc < 0.5 else INK)
    ax.set_title(title, fontsize=FT)
# (d)
ax = axs[3]; training(ax, ["2%", "2%", "2%"], [0.02, 0.02, 0.02])
ax.plot(xs, np.full_like(xs, 0.97), color=INK, lw=1.0); ax.text(-4.5, 1.17, "score: flat", fontsize=FS, color=INK, va="bottom")
ax.text(0, -0.10, "AUC = 0.50", ha="center", va="top", fontsize=FB, fontweight="bold", color=INK)
ax.set_title("(d) Source-balanced\ntraining", fontsize=FT)
axs[0].set_title("(a) Training sources,\nno signal in the labels", fontsize=FT)
for ax in axs:
    ax.set_xlim(-4.6, 4.6); ax.set_ylim(-0.02, 1.36); ax.set_yticks([]); ax.set_xticks([])
    for sp in ["left", "top", "right"]: ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.6)
fig.tight_layout(pad=0.2, w_pad=0.5, rect=(0, 0.08, 1, 1)); fig.savefig(f"{OUT}/fig1_mechanism.pdf"); plt.close(fig)

# ---------- Fig 2: Theorem 3 prediction vs observed in-corpus reference ----------
reps = [("TransE", "transe"), ("Connectivity", "conn"), ("Structural (8)", "full8"), ("Structural+graph (12)", "full12"), ("RDF2Vec", "rdf2vec")]
pts = []
for name, f in reps:
    r = json.load(open(f"{R}/{f}.json"))
    pts.append((name, r["thm3_pred"], r["incorpus_uniform"]["null"], r["incorpus_uniform"]["null_sd"], r["source_acc"], "o"))
rg = json.load(open(f"{R}/rgcn_bal_in.json"))["incorpus_uniform"]
pts.append(("R-GCN", 0.6951560301751141, rg["null"], rg["null_sd"], 0.998875, "o"))   # from results/rgcn_ident.log
fl = json.load(open(f"{R}/funclabel.json"))
pts.append(("Functional label", fl["thm3_pred"], fl["incorpus_uniform"]["null"], fl["incorpus_uniform"]["null_sd"], fl["source_acc"], "D"))
import os
OSR = f"{R}/opensanctions"
if os.path.exists(f"{OSR}/struct.json"):
    for name, f in [("OpenSanctions (i)", "struct"), ("OpenSanctions (ii)", "struct_topics"), ("OpenSanctions (iii)", "rdf2vec")]:
        r = json.load(open(f"{OSR}/{f}.json"))
        pts.append((name, r["thm2_pred"], r["incorpus_uniform"]["null"], r["incorpus_uniform"]["null_sd"], r["source_acc"], "s"))
fig, ax = plt.subplots(figsize=(2.15, 1.85))
ax.plot([0.5, 0.85], [0.5, 0.85], lw=0.6, color=GREY, ls="--", zorder=0)
offs = {"TransE": (7, -3, "left"), "Connectivity": (7, -5, "left"), "Structural (8)": (10, -3, "left"),
        "Structural+graph (12)": (-7, 9, "right"), "RDF2Vec": (8, 8, "left"), "R-GCN": (9, -9, "left"),
        "Functional label": (-10, 2, "right"),
        "OpenSanctions (i)": (10, -30, "left"), "OpenSanctions (ii)": None, "OpenSanctions (iii)": None}
for name, pred, obs, sdv, acc, mk in pts:
    ax.errorbar(pred, obs, yerr=2 * sdv, fmt=mk, ms=2.6, color={"o": INK, "D": RED, "s": BLUE}[mk], elinewidth=0.6, capsize=1.5)
    if offs[name] is None:
        continue
    dx, dy, ha = offs[name]
    label = "OpenSanctions (i)-(iii)" if name.startswith("OpenSanctions") else (f"{name} ({acc:.3f})" if acc > 0.98 else f"{name} ({acc:.2f})")
    label = label.replace("Structural+graph (12) ", "Structural+graph\n(12) ")
    ax.annotate(label, (pred, obs), textcoords="offset points", xytext=(dx, dy), ha=ha, va="center", fontsize=5.3,
                arrowprops=dict(arrowstyle="-", lw=0.35, color=GREY, shrinkA=0, shrinkB=2))
ax.set_xlim(0.5, 0.85); ax.set_ylim(0.5, 0.85)
ax.set_xlabel("Theorem 2 prediction"); ax.set_ylabel("observed in-corpus reference")
fig.tight_layout(pad=0.3); fig.savefig(f"{OUT}/fig2_thm3.pdf"); plt.close(fig)

# ---------- Fig 3: held-out readings with both tests, ICIJ structural (8) and OpenSanctions structural ----------
LEAKS = ["Panama Papers", "Paradise Papers", "Offshore Leaks", "Bahamas Leaks"]; short = ["Panama", "Paradise", "Offshore", "Bahamas"]
f8 = json.load(open(f"{R}/full8.json")); b99 = json.load(open(f"{R}/full8_bal99.json"))
for k, L in enumerate(LEAKS):
    b99[L]["null_draws"] = b99[L]["null_draws"] + json.load(open(f"{R}/bal900_{k}.json"))
ha = json.load(open(f"{R}/heldout_assoc_icij.json"))
B9 = json.load(open(f"{R}/full8_bal999_summary.json"))
OSS = ["ext_ru_egrul", "us_sam_exclusions", "wikidata", "gb_coh_disqualified", "ru_acf_bribetakers"]
os_s = json.load(open(f"{R}/opensanctions/struct.json")); os_h = json.load(open(f"{R}/opensanctions/heldout_assoc_os.json"))
os_e = json.load(open(f"{R}/opensanctions/e2_summary.json"))["struct"]
GREEN = "#2e7d32"
panels = [
  ("ICIJ, structural (8)", ["Panama", "Paradise", "Offshore", "Bahamas"],
   [(f8[f"lolo_{L}"], b99[L]["null_draws"], b99[L]["real_draws"], ha[L]["fits"][0],
     ("masked transfer" if f8[f"lolo_{L}"]["real"] < 0.5 else "transfer") if B9[L]["holm"] <= 0.05 else "inconclusive") for L in LEAKS], (0.25, 0.85)),
  ("OpenSanctions, structural", ["EGRUL", "SAM", "Wikidata", "UK", "ACF"],
   [(os_s[f"lolo_{s}"], os_s[f"lolo_bal_{s}"]["null_draws"], os_s[f"lolo_bal_{s}"]["real_draws"], os_h[s]["fits"][0],
     "no relations" if os_e[s]["bal_q975"] == os_e[s]["bal_q025"] else ("provenance-only" if os_h[s]["fits"][0]["p_two"] <= 0.05 and os_e[s]["holm"] > 0.05 else "inconclusive"))
    for s in OSS], (-0.03, 1.03))]
fig, axs = plt.subplots(1, 2, figsize=(4.8, 2.0), gridspec_kw={"width_ratios": [3.5, 5.5]})
for ax, (title, names, rows, ylim) in zip(axs, panels):
    ax.axhline(0.5, color=GREY, lw=0.6, ls="--", zorder=0)
    for i, (p, nd, rd, f0, reading) in enumerate(rows):
        x = float(i)
        ax.vlines(x - 0.24, p["null_lo"], p["null_hi"], color=RED, lw=2.6, alpha=0.35)
        ax.plot(x - 0.24, p["null"], "_", color=RED, ms=5, mew=1.0)
        ax.plot(x - 0.24, p["real"], "o", color=RED, ms=3.0)
        nd = np.array(nd)
        ax.scatter(np.full(len(nd), x + 0.04) + rng.uniform(-0.08, 0.08, len(nd)), nd, s=0.9, color=BLUE, alpha=0.2 if len(nd) > 300 else 0.35, lw=0)
        ax.vlines(x + 0.30, f0["held_q025"], f0["held_q975"], color=GREEN, lw=2.6, alpha=0.6)
        ax.plot(np.full(len(rd) - 1, x + 0.30), rd[1:], "o", color=BLUE, ms=2.2, mfc="white", mew=0.6)
        ax.plot(x + 0.30, rd[0], "o", color=BLUE, ms=3.0)
    ax.set_xticks(range(len(names))); ax.set_xticklabels([f"{n}\n" + r[4].replace("provenance-only", "provenance-\nonly").replace("masked transfer", "masked\ntransfer").replace("no relations", "no\nrelations") for n, r in zip(names, rows)], fontsize=5.4); ax.set_title(title, fontsize=6.2)
    ax.set_xlim(-0.55, len(names) - 0.45); ax.set_ylim(*ylim)
axs[0].set_ylabel("held-out AUC")
from matplotlib.lines import Line2D
h = [Line2D([], [], marker="o", color=RED, ls="", ms=3.6, label="pooled: real"),
     Line2D([], [], color=RED, lw=3.2, alpha=0.35, label="pooled: reference 95% (bar = mean)"),
     Line2D([], [], marker="o", color=BLUE, alpha=0.5, ls="", ms=2.5, label="balanced: training-label null"),
     Line2D([], [], color=GREEN, lw=3.2, alpha=0.55, label="balanced: held-out-label null 95%"),
     Line2D([], [], marker="o", color=BLUE, ls="", ms=3.6, label="balanced: tested fit"),
     Line2D([], [], marker="o", color=BLUE, mfc="white", ls="", ms=3.0, label="other fits")]
fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.015), columnspacing=1.0, handletextpad=0.6, handlelength=1.4, fontsize=5.8)
fig.tight_layout(pad=0.25, w_pad=0.6, rect=(0, 0.165, 1, 1)); fig.savefig(f"{OUT}/fig3_heldout.pdf"); plt.close(fig)
print("figures written")

# ---------- Fig 4: Theorem 4 on ICIJ: resemblance of held-out positives to the high-rate training leak ----------
z = np.load(f"{R}/thm4_scores.npz")
refs = [f8[f"lolo_{L}"]["null"] for L in LEAKS]
fig, ax = plt.subplots(figsize=(2.75, 1.95))
x = np.arange(4); w = 0.36
pp = [float(z[f"hipos_{k}"]) for k in range(4)]; uu = [float(z[f"hiunl_{k}"]) for k in range(4)]
ax.bar(x - w / 2, pp, w, color=BLUE, label="held-out positives")
ax.bar(x + w / 2, uu, w, color=GREY, label="held-out unlabelled")
for k in range(4):
    hs = str(z[f"hi_{k}"]).split()[0]
    ax.text(x[k], -0.13, f"vs {hs}", ha="center", va="top", fontsize=5.6, color=INK, transform=ax.get_xaxis_transform())
    ax.text(x[k], max(pp[k], uu[k]) + 0.03, f"ref.\n{refs[k]:.2f}", ha="center", va="bottom", fontsize=5.8, color=RED)
ax.set_xticks(x); ax.set_xticklabels(short); ax.set_ylim(0, 1.0)
ax.set_ylabel("mean $\\hat P$(high-rate leak $\\mid x$)", fontsize=6.5)
ax.legend(loc="upper left", frameon=False, fontsize=5.6, handlelength=1.0, bbox_to_anchor=(0.0, 1.02))
fig.tight_layout(pad=0.3); fig.savefig(f"{OUT}/fig4_thm4.pdf"); plt.close(fig)
print("fig4 written")
