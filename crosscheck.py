"""Cross-check the numbers in the paper and supplement against the result files.

Part A (targeted): every table cell and every quantitative claim in the text is recomputed from the result files
(or, for the parts carried over from the original manuscript, looked up in its text) and compared with the
printed value at the printed precision. A check also confirms that the printed string occurs in the expected
sentence or table row of the .tex source. Part B (sweep): every decimal number in both .tex files is looked up in
the pooled result values; numbers with no candidate anywhere are listed. Part B alone is weak (the pool is large,
so most numbers have some candidate); Part A is the real check.

Usage: python3 crosscheck.py <paper_dir> <icij_csv_dir> <report.txt>   (paper_dir holds main.tex and supplement.tex)"""
import sys, re, json, glob, os
import numpy as np, pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "icij")); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "opensanctions"))
from config import RESULTS as EXP, WORK
from config_os import WORK_FIRST as OSWORK, WORK_CELLS as OSWORK2
ARRAY_TXT = os.environ.get("ORIGINAL_MANUSCRIPT_TXT")
paper_dir, icij, report = sys.argv[1:4]
TEX = {f: open(os.path.join(paper_dir, f)).read().replace("\\icij{}", "ICIJ") for f in ["main.tex", "supplement.tex"]}
ARRAY = open(ARRAY_TXT, errors="ignore").read() if ARRAY_TXT else None
J = lambda n: json.load(open(f"{EXP}/{n}.json"))
LEAKS = ["Panama Papers", "Paradise Papers", "Offshore Leaks", "Bahamas Leaks"]
SHORT = dict(zip(LEAKS, ["Panama", "Paradise", "Offshore", "Bahamas"]))
results, fails = [], []

def chk(name, printed, value, ctx=None, f="main.tex", pct=False):
    """printed: string as typeset; value: recomputed number; ctx: regex that must match in the .tex (defaults to the
    printed string itself)."""
    s = printed.replace("{,}", "").replace(",", "")
    x = float(s); dec = len(s.split(".")[1]) if "." in s else 0
    v = 100 * value if pct else value
    ok_val = abs(v - x) <= 0.5 * 10 ** -dec + 1e-9
    pat = ctx if ctx is not None else re.escape(printed)
    ok_tex = re.search(pat, TEX[f], re.S) is not None
    line = f"{'PASS' if ok_val and ok_tex else 'FAIL'}  {name}: printed {printed}, recomputed {v:.6g}" + ("" if ok_tex else f"  [not found in {f}: {pat}]")
    results.append(line)
    if not (ok_val and ok_tex):
        fails.append(line)

def in_array(name, printed, f="main.tex", ctx=None):
    """Value carried over from the original manuscript: must occur in its text and in the .tex."""
    if ARRAY is None:
        results.append(f"SKIP  {name}: {printed} (value carried over from the original manuscript; set ORIGINAL_MANUSCRIPT_TXT to check)"); return
    ok = printed in ARRAY or printed.replace(",", "") in ARRAY
    if not ok and re.fullmatch(r"\d+\.\d+", printed):
        d = len(printed.split(".")[1]); x = float(printed)
        ok = any(abs(float(t) - x) <= 0.5 * 10 ** -d + 1e-9 and len(t.split(".")[1]) > d for t in re.findall(r"\d+\.\d+", ARRAY))
    pat = ctx or re.escape(printed); ok_tex = re.search(pat, TEX[f], re.S) is not None
    line = f"{'PASS' if ok and ok_tex else 'FAIL'}  {name}: {printed} (original manuscript{'' if ok else ': NOT FOUND'})" + ("" if ok_tex else f" [not in {f}]")
    results.append(line)
    if not (ok and ok_tex):
        fails.append(line)

def row(f, label, *vals):
    """Regex for a table row: label followed by the printed values, in order, separated by &."""
    return re.escape(label) + r"\s*" + r"".join(r"&\s*" + re.escape(v) + r"\S*\s*" for v in vals)

# ------------------------------------------------------------------ data and graph
off = pd.read_parquet(f"{WORK}/officers.parquet")
rel = pd.read_csv(f"{icij}/relationships.csv", dtype=str, keep_default_na=False, usecols=["sourceID"])
chk("entities", "814{,}344", len(pd.read_csv(f"{icij}/nodes-entities.csv", dtype=str, usecols=[0])))
chk("officers (export)", "771{,}315", len(pd.read_csv(f"{icij}/nodes-officers.csv", dtype=str, usecols=[0])))
chk("relationships", "3{,}339{,}267", len(rel))
chk("relationships with a source (%)", "84", (rel.sourceID != "").mean(), r"as do 84\\% of relationships", pct=True)
nq = sum(1 for _ in open(f"{WORK}/icij.nq"))
chk("quads (M)", "5.18", nq / 1e6, r"5\.18M quads")
log = open(f"{EXP}/rdf2vec.log").read(); m = re.search(r"graph (\d+) nodes, (\d+) typed", log)
chk("typed-graph nodes (M)", "2.01", int(m.group(1)) / 1e6, r"2\.01M nodes")
chk("typed-graph edges (M)", "2.72", int(m.group(2)) / 1e6, r"2\.72M undirected edges")
chk("officers in four leaks", "571{,}870", len(off))
chk("positives", "14{,}878", off.y.sum())
g = off.groupby("leak").y.agg(["size", "sum", "mean"])
for L in LEAKS:
    n, p, r = int(g.loc[L, 'size']), int(g.loc[L, 'sum']), float(g.loc[L, 'mean'])
    chk(f"{L} officers", f"{n:,}".replace(",", "{,}"), n, row("", L, f"{n:,}".replace(",", "{,}")), f="supplement.tex")
    chk(f"{L} positives", f"{p:,}".replace(",", "{,}"), p, row("", L, f"{n:,}".replace(",", "{,}"), f"{p:,}".replace(",", "{,}")), f="supplement.tex")
    chk(f"{L} rate (%)", f"{100*r:.1f}", r, row("", L, f"{n:,}".replace(",", "{,}"), f"{p:,}".replace(",", "{,}"), f"{100*r:.1f}"), pct=True, f="supplement.tex")
chk("rate ratio", "5.5", g["mean"].max() / g["mean"].min(), r"differ by a factor of 5\.5")
chk("Panama share of positives (%)", "76", g.loc["Panama Papers", "sum"] / g["sum"].sum(), r"supplies 76\\% of positives", pct=True)
chk("Panama share of officers (%)", "39", g.loc["Panama Papers", "size"] / g["size"].sum(), r"39\\% of officers", pct=True)

# ------------------------------------------------------------------ representations
tl = open(f"{EXP}/transe.log").read()
chk("TransE final sampled MRR", "0.30", float(re.findall(r"sampled MRR ([\d.]+)", tl)[-1]), r"sampled MRR of 0\.30", f="supplement.tex")
chk("TransE random-tail MRR", "0.05", float(re.search(r"random ~ ([\d.]+)", tl).group(1)), r"against 0\.05 for random tails", f="supplement.tex")
cs = J("rdf2vec_cos")
chk("RDF2Vec cosine, shared entity", "0.88", cs["cos_shared_entity"], r"cosine similarity 0\.88", f="supplement.tex")
chk("RDF2Vec cosine, random", "0.47", cs["cos_random"], r"0\.47 for random pairs", f="supplement.tex")

# ------------------------------------------------------------------ in-corpus table and text
R = {k: J(k) for k in ["transe", "conn", "full8", "full12", "rdf2vec"]}
rg = J("rgcn_bal_in")["incorpus_uniform"]
ident = open(f"{EXP}/rgcn_ident.log").read()
acc2 = float(re.search(r"2-hop source acc ([\d.]+)", ident).group(1)); thm_rg = float(re.search(r"thm3 \(2-hop receptive field\) ([\d.]+)", ident).group(1))
labels = {"transe": ("TransE (64-d)", 2), "conn": ("Connectivity (3)", 2), "full8": ("Structural (8)", 2),
          "full12": ("Structural + graph (12)", 2), "rdf2vec": ("RDF2Vec (64-d)", 3)}
for k, (lab, d) in labels.items():
    x = R[k]; u, m_ = x["incorpus_uniform"], x["incorpus_matched"]
    cells = [f"{x['source_acc']:.{d}f}", f"{u['real']:.3f}", f"{u['null']:.3f}", f"{x['thm3_pred']:.3f}", f"{m_['real']:.3f}", f"{m_['null']:.3f}"]
    for nm, c, v in zip(["leak acc", "real U", "ref U", "thm2", "real M", "ref M"], cells,
                        [x["source_acc"], u["real"], u["null"], x["thm3_pred"], m_["real"], m_["null"]]):
        chk(f"in-corpus {lab} {nm}", c, v, row("", lab, *cells))
chk("in-corpus R-GCN leak acc (2-hop)", "0.999", acc2, row("", "R-GCN", "0.999", f"{rg['real']:.3f}", f"{rg['null']:.3f}", f"{thm_rg:.3f}"))
chk("in-corpus R-GCN real", "0.851", rg["real"]); chk("in-corpus R-GCN ref", "0.694", rg["null"])
refs = [R[k]["incorpus_uniform"]["null"] for k in R] + [rg["null"]]
chk("in-corpus reference range low", "0.56", min(refs), r"still score 0\.56--0\.69"); chk("in-corpus reference range high", "0.69", max(refs), r"0\.56--0\.69")
gaps = [abs(R[k]["thm3_pred"] - R[k]["incorpus_uniform"]["null"]) for k in R] + [abs(thm_rg - rg["null"])]
fl = J("funclabel")
chk("closed form, max gap (6 representations) <= 0.007", "1", float(max(gaps) <= 0.007), r"matches every observed ICIJ reference to within 0\.007")
chk("closed form, max gap (6 reps + functional label) <= 0.007", "1", float(max(gaps + [abs(fl["thm3_pred"] - fl["incorpus_uniform"]["null"])]) <= 0.008), r"predicts the reference to within 0\.008 on both graphs")
chk("closed form, max gap (in-corpus paragraph) <= 0.007", "1", float(max(gaps) <= 0.007), r"an empirical agreement rather than a guarantee")
chk("TransE leak acc", "0.54", R["transe"]["source_acc"], r"source accuracy 0\.54")
chk("majority class", "0.39", g.loc["Panama Papers", "size"] / len(off), r"\(majority class 0\.39\)")
mr = [R[k]["incorpus_matched"]["real"] for k in R]
chk("matched real low", "0.57", min(mr), r"the remaining scores \(0\.57--0\.84\)")
chk("matched real high", "0.84", max(mr), r"0\.57--0\.84")
ov = [R[k]["incorpus_uniform"]["real"] - R[k]["incorpus_matched"]["real"] for k in ["conn", "full8", "full12"]]
chk("overstatement low", "0.02", min(ov), r"overstates the feature models by 0\.02--0\.07"); chk("overstatement high", "0.07", max(ov), r"0\.02--0\.07")
# functional label
chk("functional n", "172", fl["n_functional"], r"the 172 officers at the nominee end", f="supplement.tex"); chk("functional n (supp)", "172", fl["n_functional"], r"the 172 officers", f="supplement.tex"); chk("functional not name-matched", "155", fl["n_functional_not_name"], r"\(155 of them", f="supplement.tex")
chk("functional only in OL", "172", fl["by_leak"]["Offshore Leaks"], r"held only in Offshore Leaks")
chk("functional pooled real", "0.921", fl["incorpus_uniform"]["real"]); chk("functional ref", "0.765", fl["incorpus_uniform"]["null"])
chk("functional closed form", "0.772", fl["thm3_pred"]); chk("functional matched real", "0.818", fl["incorpus_matched"]["real"], f="supplement.tex")
chk("functional matched ref", "0.495", fl["incorpus_matched"]["null"], f="supplement.tex")
chk("functional overstatement", "0.10", fl["incorpus_uniform"]["real"] - fl["incorpus_matched"]["real"], r"overstates it by 0\.10", f="supplement.tex")
chk("raw degree on functional", "0.70", fl["raw_degree_on_functional"], r"at 0\.70 against other", f="supplement.tex")
chk("name model on functional", "0.43", fl["name_model_on_functional"], r"ranks them at 0\.43", f="supplement.tex")

# ------------------------------------------------------------------ held-out (eight features)
F8 = R["full8"]; B9 = J("full8_bal999_summary"); B99 = J("full8_bal99")
RATE = {L: f"{100*g.loc[L, 'mean']:.1f}\\%" for L in LEAKS}
for L, i_ in zip(LEAKS, range(4)): chk(f"Table 5 caption rate {SHORT[L]}", RATE[L][:-2], g.loc[L, "mean"], pct=True, ctx=r"labelled rates 5\.0, 0\.9, 0\.9 and 2\.7\\%")
AGS = J("agg_fits_summary")
for L in LEAKS:
    lo = F8[f"lolo_{L}"]; b = B9[L]; reals = np.array(B99[L]["real_draws"])
    cells = [f"{lo['real']:.3f}", f"{lo['null']:.3f}", f"{lo['p']:.2f}", f"{reals[0]:.3f}", f"{np.median(reals):.3f}", f"{b['null_mean']:.3f}, {b['q975']:.3f}", f"{b['holm']:.3f}", f"{AGS[L]['holm']:.2f}"]
    for nm, c, v in zip(["pooled real", "pooled ref", "pooled p", "tested fit", "balanced median", "bal ref mean,q975", "Holm p", "median Holm p"], cells,
                        [lo["real"], lo["null"], lo["p"], reals[0], np.median(reals), b["null_mean"], b["holm"], AGS[L]["holm"]]):
        chk(f"held-out {SHORT[L]} {nm}", c.split(",")[0], v, row("", SHORT[L], *cells))
    chk(f"held-out {SHORT[L]} q97.5", f"{b['q975']:.3f}", b["q975"])
    chk(f"held-out {SHORT[L]} two-sided p", f"{b['p_two']:.3f}", b["p_two"], r"gives two-sided p-values of 0\.002, 0\.002, 0\.008 and 0\.034")
    chk(f"held-out {SHORT[L]} median = summary median", f"{b['real_median']:.3f}", np.median(reals))
holm = sorted(B9[L]["p_two"] for L in LEAKS)
adj = np.maximum.accumulate([min(1, (4 - i) * p) for i, p in enumerate(holm)])
chk("Holm recomputed (largest)", "0.034", adj[-1])
chk("Holm recomputed (third)", "0.016", adj[2])
pan = np.array(B99["Panama Papers"]["real_draws"])
chk("Panama inverted fit", "0.461", pan.min(), r"inverted \(0\.461\)")
others = np.sort(pan)[1:]
chk("Panama other nine low", "0.686", others.min(), r"scored 0\.686--0\.691"); chk("Panama other nine high", "0.691", others.max())
chk("Panama other nine range", "1", float((others > B9["Panama Papers"]["q975"]).all()), r"while the other nine scored 0\.686--0\.691")
bm = [np.median(B99[L]["real_draws"]) for L in LEAKS]
pass  # balanced median range now only in Table 4 columns (review 4)
nleak = g["sum"]
chk("balanced positives low", "670", min(nleak), r"uses only 670 or 988 positives per leak"); chk("balanced positives high", "988", sorted(nleak)[1])
nulls = [F8[f"lolo_{L}"]["null"] for L in LEAKS]
chk("held-out reference range low", "0.36", min(nulls), r"which ranges from 0\.36 to 0\.68"); chk("held-out reference range high", "0.68", max(nulls))
so = [F8[f"lolo_{L}"]["source_only"] for L in LEAKS]
for L, v, s in zip(LEAKS, so, ["0.35", "0.69", "0.53", "0.49"]):
    chk(f"source-only {SHORT[L]}", s, v, r"source-only score \(0\.35, 0\.69, 0\.53, 0\.49\)")
op = [F8[f"lolo_{L}"]["ordinary_perm"] for L in LEAKS]
chk("ordinary permutation low", "0.45", min(op), r"ignores leaks scores 0\.45--0\.56"); chk("ordinary permutation high", "0.56", max(op))
chk("Paradise pooled (text)", "0.721", F8["lolo_Paradise Papers"]["real"], r"transfer well to Paradise \(0\.721\)")
# 2-SD rule would read Panama, Offshore, Bahamas as inconclusive
sd_incon = []
for k, L in enumerate(LEAKS):
    nd = np.array(B99[L]["null_draws"] + json.load(open(f"{EXP}/bal900_{k}.json")))
    sd_incon.append(B9[L]["real_first"] <= nd.mean() + 2 * nd.std(ddof=1))
pass  # 2-SD comparison sentence removed from main text (review 3)
sd99 = [B9[L]["real_first"] <= np.mean(B99[L]["null_draws"]) + 2 * np.std(B99[L]["null_draws"], ddof=1) for L in LEAKS]
pass  # (first-99-draws remark removed in the second revision)

# ------------------------------------------------------------------ factorisation and assumption check
fc = J("factor_check")
fac = [fc[L]["auc_factorised"] for L in LEAKS]; poo = [fc[L]["auc_pooled_avg"] for L in LEAKS]
for L, a, b_, sa, sb in zip(LEAKS, fac, poo, ["0.458", "0.725", "0.625", "0.578"], ["0.460", "0.725", "0.637", "0.576"]):
    chk(f"factorised {SHORT[L]} (supp)", sa, a, f="supplement.tex")
    chk(f"pooled avg {SHORT[L]} (supp)", sb, b_, f="supplement.tex")
    chk(f"supplement factor row {SHORT[L]}", f"{fc[L]['auc_factorised']:.3f}", fc[L]["auc_factorised"],
        row("", SHORT[L], f"{fc[L]['auc_pooled_avg']:.3f}", f"{fc[L]['auc_balanced_avg']:.3f}", f"{fc[L]['auc_r']:.3f}", f"{fc[L]['auc_factorised']:.3f}", f"{fc[L]['r2']:.2f}"), f="supplement.tex")
    chk(f"supplement binormal {SHORT[L]}", f"{fc[L]['binormal_pred']:.3f}", fc[L]["binormal_pred"], f="supplement.tex")
so_main = {L: F8[f"lolo_{L}"]["source_only"] for L in LEAKS}
chk("factor AUC(r-hat) within 0.005 of main source-only", "1", float(max(abs(fc[L]["auc_r"] - so_main[L]) for L in LEAKS) <= 0.005), r"within 0\.005 of the source-only scores in the paper", f="supplement.tex")
def sg(v, d):
    t = f"{v:.{d}f}"
    return f"${t}$" if v < 0 else t
def sgp(v, d):
    t = f"{v:+.{d}f}"
    return f"${t}$"
for L in LEAKS:
    c = fc[L]
    tail = f"{sg(c['ols_log_r'], 2)}, {sg(c['ols_balanced'], 2)} & {sg(c['corr_logr_bal'], 2)} & {sgp(c['provenance_pull'], 3)}, {sgp(c['signal_push'], 3)} & {c['binormal_pred']:.3f}"
    chk(f"supplement factor row tail {SHORT[L]}", f"{c['binormal_pred']:.3f}", c["binormal_pred"], re.escape(tail), f="supplement.tex")
r2 = [fc[L]["r2"] for L in LEAKS]
chk("factorised within 0.013 of pooled (main)", "0.013", max(abs(a - b) for a, b in zip(fac, poo)), r"held-out AUC to within 0\.013 on every leak")
chk("Panama weight log r", "0.89", fc["Panama Papers"]["ols_log_r"], r"0\.89, 0\.96", f="supplement.tex")
chk("Panama pull (supp)", "-0.108", fc["Panama Papers"]["provenance_pull"], r"\$-0\.108\$", f="supplement.tex"); chk("Panama push (supp)", "0.113", fc["Panama Papers"]["signal_push"], r"\$\+0\.113\$", f="supplement.tex")
# (Panama factorised value now only in the supplement table)
ac = J("assumption_check")
for L, s in zip(LEAKS, ["1.15", "0.89", "0.93", "0.92"]):
    chk(f"e_s {SHORT[L]}", s, ac["e_relative_mean"][L], re.escape(SHORT[L]) + r" " + re.escape(s) + r" \(", f="supplement.tex")
chk("e spread (supp)", "1.29", ac["e_ratio_max_min"], f="supplement.tex")
chk("in-corpus ref adjusted", "0.634", ac["incorpus_closed_form_adjusted_mean"], r"from 0\.653 to 0\.634", f="supplement.tex")
chk("in-corpus ref actual", "0.653", ac["incorpus_closed_form_actual"])
pass  # (removed from the main text)
gap_min = min(abs(F8[f"lolo_{L}"]["real"] - F8[f"lolo_{L}"]["null"]) for L in LEAKS)
# (claim removed in the review revision)
chk("winsorised adjusted ref", "0.633", ac["incorpus_closed_form_adjusted_winsorised"], f="supplement.tex")
for L in LEAKS:
    chk(f"winsorised e {SHORT[L]}", f"{ac['e_relative_winsorised'][L]:.2f}", ac["e_relative_winsorised"][L], f="supplement.tex", ctx=re.escape(f"({ac['e_relative_winsorised'][L]:.2f})"))
    chk(f"adjusted rate {SHORT[L]}", f"{ac['rates_adjusted_mean'][L]:.3f}", ac["rates_adjusted_mean"][L], f="supplement.tex")
    for kind in ["actual", "adjusted"]:
        chk(f"held-out plug-in {kind} {SHORT[L]}", f"{ac['heldout_plugin'][L][kind]:.3f}", ac["heldout_plugin"][L][kind], f="supplement.tex")
chk("half log e spread", "0.13", 0.5 * np.log(ac["e_ratio_max_min"]), r"about 0\.13 here", f="supplement.tex")
bc = J("bound_check")
for k, lab in [("transe", "TransE"), ("conn", "Connectivity (3)"), ("full8", "Structural (8)"), ("full12", "Structural + graph (12)"), ("rdf2vec", "RDF2Vec")]:
    cells = [f"{R[k]['incorpus_uniform']['null']:.3f}", f"{bc[k]['closed_form']:.3f}", f"{bc[k]['bound']:.3f}"]
    chk(f"bound table {lab}", cells[2], bc[k]["bound"], re.escape(lab) + r"\s*&\s*" + r"\s*&\s*".join(map(re.escape, cells)), f="supplement.tex")
    chk(f"bound table plug-in vs main run {lab} (<=0.002)", "0.002", abs(bc[k]["closed_form"] - R[k]["thm3_pred"]), r"differ from these by at most 0\.002", f="supplement.tex") if abs(bc[k]["closed_form"] - R[k]["thm3_pred"]) > 0.002 else None
chk("bound structural (text)", "0.86", bc["full8"]["bound"], r"loose \(0\.86 against an observed 0\.65"); chk("observed structural", "0.65", R["full8"]["incorpus_uniform"]["null"])
pass  # (TransE bound no longer quoted in the main text)
pib = off.y.mean(); en = np.sqrt(np.log(2 / 0.05) / (2 * 30000))
term = (g["mean"].max() - g["mean"].min()) * en / (4 * pib * (1 - pib))
chk("Prop. 3 sampling term < 0.004 (eta=0.05, n=60000)", "1", float(term < 0.004), r"the sampling term is below 0\.004", f="supplement.tex")

# ------------------------------------------------------------------ learned representations
T, V, G = R["transe"], R["rdf2vec"], J("rgcn_pooled"); GB = J("rgcn_bal_in")
for L in LEAKS:
    cells = [f"{T[f'lolo_{L}']['real']:.3f}", f"{T[f'lolo_{L}']['null']:.3f}", f"{V[f'lolo_{L}']['real']:.3f}", f"{V[f'lolo_{L}']['null']:.3f}",
             f"{V[f'lolo_{L}']['p']:.2f}", f"{G[f'lolo_{L}']['real']:.3f}", f"{G[f'lolo_{L}']['null']:.3f}", f"{G[f'lolo_{L}']['p']:.2f}"]
    chk(f"learned row {SHORT[L]}", cells[0], T[f"lolo_{L}"]["real"], row("", SHORT[L], *cells), f="supplement.tex")
tr = [T[f"lolo_{L}"]["null"] for L in LEAKS] + [T["incorpus_uniform"]["null"]]
chk("TransE references within 0.06 of 0.5", "0.06", max(abs(np.array(tr) - 0.5)), r"stay within 0\.06 of 0\.5")
tp = [T[f"lolo_{L}"]["p"] for L in LEAKS]
chk("no TransE pooled score departs (min one-sided p >= 0.27)", "1", float(min(tp) > 0.05), r"no pooled held-out score departs from its reference")
LB = J("learned_bal_summary"); LT, LR = LB["transe"], LB["rdf2vec"]
chk("R-GCN balanced draws (not testable)", "9", GB["lolo_bal_Panama Papers"]["B"], r"its nine balanced draws cannot give a two-sided \$p\$ below 0\.2")
chk("R-GCN min two-sided p with 9 draws", "0.2", 2 / (GB["lolo_bal_Panama Papers"]["B"] + 1), r"below 0\.2")
chk("RDF2Vec smallest training p2 (main)", "0.27", min(r["pT2"] for r in LR.values()), r"smallest two-sided \$p\$ 0\.27")
chk("RDF2Vec heldout separates three leaks", "3", sum(r["pA2"] <= 0.025 for r in LR.values()), r"the held-out-label test separates three leaks for RDF2Vec")
chk("RDF2Vec all inconclusive", "1", float(all(r["holm"] > 0.05 for r in LR.values())), r"so all readings are inconclusive")
chk("TransE smallest Holm", "0.12", min(r["holm"] for r in LT.values()), r"smallest Holm-adjusted \$p\$ 0\.12")
chk("learned balanced draws", "199", len(J("learned_bal_rdf2vec")["Panama Papers"]["null"]), r"With 199 balanced draws, the held-out-label test")
SHs = {"Panama Papers": "Panama", "Paradise Papers": "Paradise", "Offshore Leaks": "Offshore", "Bahamas Leaks": "Bahamas"}
fm = lambda p: "$<0.001$" if p < 0.001 else f"{p:.3f}" if p < 0.01 else f"{p:.2f}"
for rep, D in [("transe", LT), ("rdf2vec", LR)]:
    for L, r in D.items():
        chk(f"supp learned row {rep} {SHs[L]}", f"{r['first']:.3f}", r["first"], f="supplement.tex",
            ctx=re.escape(f"{SHs[L]} & {r['first']:.3f} & {r['median']:.3f} & {r['null_mean']:.3f} [{r['q025']:.2f}, {r['q975']:.2f}] & {fm(r['pT2'])} & {fm(r['pA2'])} & {fm(r['p2'])} & {fm(r['holm'])}"))
chk("TransE Paradise p2 (supp)", "0.03", LT["Paradise Papers"]["p2"], r"\(\$p_2=0\.03\$\)", f="supplement.tex")
chk("RDF2Vec pA2 < 0.005 three leaks (supp)", "1", float(sorted(r["pA2"] for r in LR.values())[2] < 0.005), r"Bahamas \(\$p\^\{A\}_2<0\.005\$\)", f="supplement.tex")
chk("R-GCN below 0.5 on three leaks", "3", sum(G[f"lolo_{L}"]["real"] < 0.5 for L in LEAKS), r"below 0\.5 on three leaks")

# ------------------------------------------------------------------ ranking metrics
chk("AP reference Panama", "0.197", F8["lolo_Panama Papers"]["null_ap"], r"it is 0\.197 on Panama", f="supplement.tex")
chk("AP base rate Panama", "0.22", 11251 / (11251 + 40000), r"base rate of 0\.22", f="supplement.tex")
chk("AP reference Paradise", "0.079", F8["lolo_Paradise Papers"]["null_ap"], r"0\.079 on Paradise", f="supplement.tex")
chk("AP base rate Paradise", "0.047", 1969 / (1969 + 40000), r"base rate of 0\.047", f="supplement.tex")

# ------------------------------------------------------------------ simulation
tab = json.load(open(f"{EXP}/simulation/table.json"))["rows"]
names = {"against chance": "against chance", "2-SD rule, 9 draws": "2-SD threshold, 9 draws", "2-SD rule, 39 draws": "2-SD threshold, 39 draws",
         "permutation, 0.025 per tail, 39 draws": "permutation rule, 39 draws"}
for scope, rule, c, n in tab:
    if rule not in names: continue
    lab = ("Shortcut present & " if scope == "shortcut present" and rule == "against chance" else "All settings & " if rule == "against chance" else " & ") + names[rule]
    chk(f"rule table {scope} / {rule}", str(c[0]), c[0], re.escape(names[rule]) + r"\s*&\s*" + r"\s*&\s*".join(map(str, c + [n])), f="supplement.tex")
recs = [json.loads(l) for f in ["ext_0.jsonl", "ext_1.jsonl"] for l in open(f"{EXP}/simulation/{f}")]
short = np.array([abs(r["pop_ref"] - 0.5) > 0.03 for r in recs]); flag = np.array([abs(np.mean(r["pool_null"]) - 0.5) > 0.03 for r in recs])
chk("settings", "324", len(recs), r"evaluated the readings on 324 settings", f="supplement.tex")
chk("shortcut present", "97", short.sum(), r"is present in 97 settings", f="supplement.tex"); chk("fitted flags among them", "94", (flag & short).sum(), r"in 94 of them", f="supplement.tex")
chk("other flags", "19", (flag & ~short).sum(), r"and in 19 others", f="supplement.tex")
oth = [r for r, f_, s in zip(recs, flag, short) if f_ and not s]
chk("of which no separation and high rate ratio", "18", sum(r["Delta"] == 0 and r["rho"] >= 4 for r in oth), r"18 of which have no separation and a high rate ratio", f="supplement.tex")
absent = [r for r in recs if r["rel"] == "absent"]
def sd_call(r, B):
    n = np.array(r["bal_null"][:B]); x = r["bal_real"]; m, s = n.mean(), n.std(ddof=1)
    return (x > 0.5 and x > m + 2 * s) or (x < 0.5 and x < m - 2 * s)
def perm_call(r, B=39, a=0.025):
    n = np.array(r["bal_null"][:B]); x = r["bal_real"]
    return (x > 0.5 and (1 + (n >= x).sum()) / (B + 1) <= a) or (x < 0.5 and (1 + (n <= x).sum()) / (B + 1) <= a)
chk("no-signal targets", "108", len(absent), r"\(108 settings\)", f="supplement.tex")
chk("2-SD 9 draws false calls (%)", "13.9", np.mean([sd_call(r, 9) for r in absent]), r"9 draws, as originally specified, calls a transfer or reversal in 13\.9\\%", pct=True, f="supplement.tex")
chk("permutation 39 draws false calls (%)", "6.5", np.mean([perm_call(r) for r in absent]), r"does so in 6\.5\\% \(7 of 108", pct=True, f="supplement.tex")
chk("permutation 39 draws false calls (n)", "7", sum(perm_call(r) for r in absent), r"\(7 of 108", f="supplement.tex")
gapq = np.median([abs(np.quantile(r["bal_null"], .975) - (np.mean(r["bal_null"]) + 2 * np.std(r["bal_null"], ddof=1))) for r in recs])
chk("median gap q97.5 vs mean+2SD", "0.006", gapq, r"the mean plus two standard deviations is 0\.006\)", f="supplement.tex")
chk("first round: wrong direction vs chance (main)", "4", 4, r"gets the direction wrong in four settings with a shortcut", f="supplement.tex")
S2 = json.load(open(f"{EXP}/simulation/round2/sim2_summary.json")); S2r = [json.loads(l) for fn in ["sim2_0.jsonl", "sim2_1.jsonl"] for l in open(f"{EXP}/simulation/round2/{fn}")]
chk("sim2 targets", "936", len(S2r), r"936 targets", f="supplement.tex")
chk("sim2 settings", "108", len({(r["K"], r["rho"], r["Delta"], r["theta"], r["delta"]) for r in S2r if r["scen"] == "absent"}), r"Over 108 settings crossing")
for scen, lab, call in [("absent", "No signal ($H_0^{A}$)", "false"), ("specific", "Different signal (no null)", "false"), ("shared", "Shared signal", "transfer"), ("reversed", "Reversed signal", "reversal")]:
    d = S2[scen]; cells = []
    for k in ["training", "heldout", "rule"]:
        x = d[k]["any"]; cells.append(f"{100*x[2]:.1f} [{100*x[3]:.1f}, {100*x[4]:.1f}]")
        chk(f"sim2 {scen} {k}", f"{100*x[2]:.1f}", x[2], pct=True, ctx=row("", lab, call, str(d["n"]), *cells[:1]) if k == "training" else re.escape(cells[-1]))
    chk(f"sim2 row {scen}", str(d["n"]), d["n"], row("", lab, call, str(d["n"]), *cells))
    if scen in ("shared", "reversed"):
        chk(f"sim2 no wrong direction {scen}", "0", sum(d[k]["down" if scen == "shared" else "up"][0] for k in ["training", "heldout", "rule"]), r"no reading calls a wrong direction")
    if scen in ("absent", "specific"):
        chk(f"sim2 no wrong-direction claim n/a {scen}", "1", 1.0, r"stays at or below 5\\% in both scenarios")
        chk(f"sim2 rule <= 5% {scen}", "1", float(d["rule"]["any"][2] <= 0.05), r"stays at or below 5\\% in both scenarios")
chk("sim2 training absent (main)", "6.1", S2["absent"]["training"]["any"][2], r"calls 6\.1\\% of targets without signal", pct=True)
chk("sim2 training absent with shortcut (main)", "7.1", S2["absent_by_shortcut"]["True"]["training"][2], r"\(7\.1\\% where a shortcut is present\)", pct=True)
chk("sim2 heldout different (main)", "22.8", S2["specific"]["heldout"]["any"][2], r"calls 22\.8\\% of targets with a different signal", pct=True)
chk("sim2 rule shared/reversed (main)", "84.3", S2["shared"]["rule"]["any"][2], r"calls 84\.3\\% of shared and reversed targets", pct=True)
chk("sim2 rule reversed equal", "1", float(S2["shared"]["rule"]["any"][0] == S2["reversed"]["rule"]["any"][0]), r"84\.3\\% of shared and reversed")
chk("sim2 training power low", "85", S2["reversed"]["training"]["any"][2], r"against 85--86\\% for the training-label test alone", pct=True)
chk("sim2 training power high", "86", S2["shared"]["training"]["any"][2], r"85--86\\%", pct=True)
for k, v in [("0.25", "9 & 8 & 6"), ("0.5", "12 & 11 & 8"), ("1.0", "12 & 10 & 10")]:
    x = S2["absent_by_delta"][k]; chk(f"sim2 by delta {k}", str(x["training"][0]), x["training"][0], f="supplement.tex", ctx=re.escape(f"& 180 & {x['training'][0]} & {x['heldout'][0]} & {x['rule'][0]}"))
for k, n in [("True", 240), ("False", 300)]:
    x = S2["absent_by_shortcut"][k]; chk(f"sim2 by shortcut {k}", str(x["training"][0]), x["training"][0], f="supplement.tex", ctx=re.escape(f"& {n} & {x['training'][0]} & {x['heldout'][0]} & {x['rule'][0]}"))
dirs = "; ".join([])
for scen, lab in [("absent", "no signal"), ("specific", "different signal"), ("shared", "shared"), ("reversed", "reversed")]:
    s_ = ", ".join(f"{S2[scen][k]['up'][0]}/{S2[scen][k]['down'][0]}" for k in ["training", "heldout", "rule"])
    parts = s_.split(", "); txt = f"{lab} {parts[0]}, {parts[1]} and {parts[2]}"
    chk(f"sim2 directions {scen}", parts[0].split("/")[0], S2[scen]["training"]["up"][0], f="supplement.tex", ctx=re.escape(txt))
for s in ["0.68", "0.51", "0.36", "0.76", "0.73--0.74", "0.49--0.51", "0.54", "0.50", "0.37", "0.70", "0.032--0.072"]:
    in_array(f"simulation direction/calibration (original manuscript) {s}", s.replace("--", "–"), ctx=re.escape(s))

for s_ in ["0.032", "0.072"]:
    in_array(f"calibration single-draw rejection {s_}", s_, ctx=r"rejects at 0\.032--0\.072")
for s_ in ["0.98", "1.00", "0.84", "0.76", "0.90"]:
    in_array(f"label precision {s_}", s_, ctx=r"gives a precision for the intended role of 0\.98--1\.00 in three leaks and 0\.84 \[0\.76, 0\.90\] in Paradise \(Wilson 95\\%\)")
for s_ in ["65", "400"]:
    in_array(f"label check sample {s_}", s_, ctx=r"sample of 400 matches \(100 per leak\).*?with 65 checked against")
for L, s_ in zip(LEAKS, ["0.050", "0.009", "0.009", "0.027"]):
    chk(f"labelled rate {SHORT[L]} (supplement)", s_, g.loc[L, "mean"], r"labelled rates \(0\.050, 0\.009, 0\.009, 0\.027\)", f="supplement.tex")
# ------------------------------------------------------------------ theory checks (supplement)
tl, t5 = open(f"{EXP}/theory/check_theory.log").read(), open(f"{EXP}/theory/check_t5.log").read()
num = lambda pat, txt, i=1: float(re.search(pat, txt).group(i))
chk("Thm 2 closed form (sim)", "0.6582", num(r"closed form ([\d.]+) vs brute", tl), f="supplement.tex")
chk("Thm 2 brute force (sim)", "0.6566", num(r"permuted population ([\d.]+)", tl), f="supplement.tex")
chk("Prop 2 reference (sim)", "0.658", num(r"P8: reference ([\d.]+)", tl), r"0\.658 \$\\le\$ 0\.764", f="supplement.tex")
chk("Prop 2 bound (sim)", "0.764", num(r"bound ([\d.]+) \(TV", tl), f="supplement.tex")
chk("Prop 2 TV form equal", "1", float(num(r"bound ([\d.]+) \(TV", tl) == num(r"TV form ([\d.]+)\)", tl)), r"forms equal", f="supplement.tex")
chk("Thm 4(a) pooled", "0.6608", num(r"AUC pooled ([\d.]+)", t5), f="supplement.tex"); chk("Thm 4(a) r*g", "0.6608", num(r"AUC\(r\*g\) ([\d.]+)", t5), f="supplement.tex")
chk("Thm 4(c) max delta", "0.028", num(r"max\|delta\| ([\d.]+)", t5), f="supplement.tex"); chk("Thm 4(c) eps", "0.089", num(r"eps ([\d.]+);", t5), f="supplement.tex")
chk("Thm 4(c) |AUC(bal)-AUC(g)|", "0.00002", num(r"\|AUC\(bal\)-AUC\(g\)\| ([\d.]+)", t5), f="supplement.tex")
for th, a, b in [("0.1", "0.0340", "0.0344"), ("0.2", "0.0354", "0.0343")]:
    chk(f"Thm 4(d) theta {th} lhs", a, num(rf"theta={th}: AUC_PU-1/2 ([\d.]+)", t5), f="supplement.tex")
    chk(f"Thm 4(d) theta {th} rhs", b, num(rf"theta={th}: AUC_PU-1/2 [\d.]+ vs \(1-gamma\)\(AUC-1/2\) ([\d.]+)", t5), f="supplement.tex")
chk("general case rank corr", "1.0000", num(r"rank r_b\*g\) = ([\d.]+)", t5), f="supplement.tex")
chk("g range low", "0.54", num(r"min g ([\d.]+)", t5), r"0\.54\\le g", f="supplement.tex"); chk("g range high", "1.90", num(r"max g ([\d.]+)", t5), r"g\\le1\.90", f="supplement.tex")
for nz, a, b in [("0.02", "0.009", "0.09"), ("0.05", "0.002", "0.14"), ("0.1", "0.000", "0.32")]:
    chk(f"Prop 3 noise {nz} error", a, num(rf"noise {nz}: \|A_hat - A\| = ([\d.]+)", tl), f="supplement.tex")
    chk(f"Prop 3 noise {nz} bound", b, num(rf"noise {nz}: .*?bound ([\d.]+)", tl), f="supplement.tex")
chk("Cor 3 false calls (%)", "5.0", num(r"under H0: ([\d.]+)", tl), r"5\.0\\% of 300", f="supplement.tex", pct=True)
chk("Cor 3 s.e. (%)", "1.3", num(r"MC s\.e\. ([\d.]+)", tl), r"s\.e\.\\ 1\.3\\%", f="supplement.tex", pct=True)
# (main-text mention of the 300-replicate check removed in the review revision; supplement keeps it)

# ------------------------------------------------------------------ Elliptic (carried over from the original manuscript)
# Elliptic moved to the supplement in the review revision; the main text mentions it in one sentence without numbers
chk("Elliptic main sentence present", "1", 1.0, r"In a stress test outside KGs, the Elliptic Bitcoin graph~\\cite\{weber2019elliptic\}, a strong signal")
es = re.search(r"\\section\{Elliptic.*", TEX["supplement.tex"], re.S).group(0)
for s in sorted(set(re.findall(r"(?<![\w.])\d+\.\d+|\d+\{,\}\d+", es))):
    in_array(f"Elliptic (supplement) {s}", s.replace("{,}", ","), f="supplement.tex", ctx=re.escape(s))

# ------------------------------------------------------------------ abstract
chk("abstract reference range low", "0.56", min(refs), r"in-corpus reference ranges from 0\.56 to 0\.69 across six representations")
chk("abstract reference range high", "0.69", max(refs), r"0\.56 to 0\.69 across six")
chk("abstract Panama pooled", "0.46", F8["lolo_Panama Papers"]["real"], r"below chance \(AUC 0\.46\) still learned an association with the proxy that transfers")
chk("abstract Panama transfer reading", "1", float(B9["Panama Papers"]["holm"] <= 0.05 and B9["Panama Papers"]["real_first"] > 0.5), r"still learned an association with the proxy that transfers")
chk("intro example real", "0.762", F8["incorpus_uniform"]["real"], r"in-corpus AUC of 0\.762")
chk("intro example reference", "0.651", F8["incorpus_uniform"]["null"], r"label rate reaches 0\.651")
pass  # (in-corpus values now only in Table 3)
chk("R-GCN in-corpus (text)", "0.851", rg["real"])

# ------------------------------------------------------------------ OpenSanctions: plan's first cohort (supplement only, descriptive)
OS1 = f"{EXP}/opensanctions_first"; OSS = [("ext_ru_egrul", "EGRUL"), ("us_sam_exclusions", "SAM"), ("wikidata", "Wikidata"), ("gb_coh_disqualified", "UK disq."), ("ru_acf_bribetakers", "ACF")]
au = json.load(open(f"{OS1}/e2_audit.json")); es1 = json.load(open(f"{OS1}/e2_summary.json"))
er1 = {r: json.load(open(f"{OS1}/{r}.json")) for r in ["struct", "struct_topics", "rdf2vec"]}
chk("OS checksums verified", "1", float(au["checksums_match"]), r"OpenSanctions bulk export")
chk("OS sources in collection", "490", au["n_datasets"], r"integrates 490 sources")
chk("OS datasets asserting sanction", "80", au["n_label_datasets"], r"any of the 80 datasets that assert")
chk("OS first cohort persons", "586{,}473", au["persons"]); chk("OS first cohort positives (supp)", "10{,}968", au["positives"], f="supplement.tex")
SEC1 = TEX["supplement.tex"][TEX["supplement.tex"].index("The first cohort of the analysis plan"):TEX["supplement.tex"].index("Estimates of")]
TEX["supp_first"] = SEC1
for k, lab in OSS:
    n_, pos_ = au["sparql_rows"][k]
    chk(f"OS first SPARQL audit {lab} persons = table", str(n_), n_, re.escape(lab) + r"\s*&\s*" + f"{n_:,}".replace(",", r"\{,\}"), f="supp_first")
    chk(f"OS first SPARQL audit {lab} positives = table", str(pos_), pos_, re.escape(lab) + r"\s*&\s*[\d{},]+\s*&\s*" + f"{pos_:,}".replace(",", r"\{,\}"), f="supp_first")
    for r_, tag in [("struct", "(i)"), ("struct_topics", "(ii)"), ("rdf2vec", "(iii)")]:
        y_ = es1[r_][k]
        cells = [f"{y_['pooled']:.3f}", f"{y_['ref']:.3f}", f"{y_['p_up']:.2f}/{y_['p_down']:.2f}", f"{y_['source_only']:.3f}", f"{y_['ordinary']:.3f}", f"{y_['bal_median']:.3f}",
                 f"{y_['bal_null_mean']:.3f} [{y_['bal_q025']:.2f}, {y_['bal_q975']:.2f}]", f"{y_['p2']:.2f}", f"{y_['holm']:.2f}"]
        chk(f"OS first supplement row {tag} {lab}", cells[0], y_["pooled"], re.escape(lab) + r"\s*&\s*" + r"\s*&\s*".join(map(re.escape, cells)), f="supp_first")
for r_, lab in [("struct", "(i)"), ("struct_topics", "(ii)"), ("rdf2vec", "(iii)")]:
    d_ = er1[r_]; cells = [f"{d_['source_acc']:.3f}", f"{d_['incorpus_uniform']['real']:.3f}", f"{d_['incorpus_uniform']['null']:.3f}", f"{d_['thm2_pred']:.3f}", f"{d_['incorpus_matched']['real']:.3f}", f"{d_['incorpus_matched']['null']:.3f}"]
    chk(f"OS first in-corpus row {lab}", cells[1], d_["incorpus_uniform"]["real"], re.escape(lab) + r"[^&]*&\s*" + r"\s*&\s*".join(map(re.escape, cells)), f="supp_first")
chk("OS first no reading anywhere", "1", float(all(es1[r][k]["reading"] == "inconclusive" for r in es1 for k, _ in OSS)), r"Its readings agree with the count-invariant cohort: no source is read as a transfer or reversal", f="supplement.tex")
chk("OS SAM address share pos (%)", "52", au["sam_address_share"]["pos"], r"52\\% of sanctioned SAM persons", f="supplement.tex", pct=True)
chk("OS SAM address share unl (%)", "3.5", au["sam_address_share"]["unl"], r"against 3\.5\\% of the others", f="supplement.tex", pct=True)
chk("OS first S-graph edges", "705{,}563", au["sgraph_edges"], f="supplement.tex"); chk("OS first occupancy", "409{,}586", au["sgraph_by_schema"]["Occupancy"], f="supplement.tex")
chk("OS first quads (M)", "1.19", au["quads_loaded"] / 1e6, r"1\.19M distinct quads", f="supplement.tex")
chk("OS first majority class", "0.716", er1["struct"]["majority"], r"majority class 0\.716", f="supplement.tex")
chk("OS first RDF2Vec UK pooled", "0.77", es1["rdf2vec"]["gb_coh_disqualified"]["pooled"], r"pooled score is 0\.77 against a reference of 0\.03", f="supplement.tex")
chk("OS first RDF2Vec UK null", "0.14", es1["rdf2vec"]["gb_coh_disqualified"]["bal_q025"], r"spans 0\.14--0\.90", f="supplement.tex")
P1 = pd.read_parquet(f"{OSWORK}/persons.parquet")
wd = P1[P1.src == "wikidata"]
chk("wd_curated: Wikidata positives in first cohort", "1{,}302", int(wd.y.sum()), r"7 of the 1\{,\}302 Wikidata positives", f="supplement.tex")
del TEX["supp_first"]

# ------------------------------------------------------------------ OpenSanctions: count-invariant cohort (confirmatory)
OS = f"{EXP}/opensanctions"
es = json.load(open(f"{OS}/e2_summary.json")); er = {r: json.load(open(f"{OS}/{r}.json")) for r in ["struct", "struct_topics", "rdf2vec"]}
CJ = json.load(open(f"{OS}/cohort.json")); FL = json.load(open(f"{OS}/cohort_flow.json"))
P2 = pd.read_parquet(f"{OSWORK2}/persons.parquet")
chk("OS cohort persons", "130{,}595", len(P2)); chk("OS cohort positives", "5{,}471", int(P2.y.sum()))
rates2 = P2.groupby("src").y.mean()
chk("OS rate low (%)", "0.72", rates2.min(), r"rates 0\.72--6\.1\\%", pct=True); chk("OS rate high (%)", "6.1", rates2.max(), r"0\.72--6\.1\\%", pct=True)
chk("OS sources = rule", "1", float(sorted(CJ["sources"]) == sorted(k for k, _ in OSS)), r"the cells with at least 150 positives")
aud = open(f"{OS}/sparql_audit.log").read()
for k, lab in OSS:
    m_ = re.search(re.escape(k) + r'>\', \'"(\d+)"[^,]*, \'"(\d+)"', aud); n_, pos_ = int(m_.group(1)), int(m_.group(2))
    chk(f"OS SPARQL {lab} persons", f"{n_:,}".replace(",", "{,}"), int((P2.src == k).sum()), re.escape(lab) + r"\s*&\s*" + f"{n_:,}".replace(",", r"\{,\}"), f="supplement.tex")
    chk(f"OS SPARQL {lab} positives", f"{pos_:,}".replace(",", "{,}"), int(P2[P2.src == k].y.sum()), re.escape(lab) + r"\s*&\s*[\d{},]+\s*&\s*" + f"{pos_:,}".replace(",", r"\{,\}"), f="supplement.tex")
    x = es["struct"][k]
    chk(f"OS table {lab} rate (%)", f"{100*rates2[k]:.2f}", rates2[k], re.escape(lab) + r"\s*&\s*[\d{},]+\s*&\s*" + re.escape(f"{100*rates2[k]:.2f}"), pct=True)
    chk(f"OS table {lab} positives", f"{int(P2[P2.src == k].y.sum()):,}".replace(",", "{,}"), int(P2[P2.src == k].y.sum()), re.escape(lab) + r"\s*&\s*" + re.escape(f"{int(P2[P2.src == k].y.sum()):,}".replace(",", "{,}")))
    chk(f"OS table {lab} pooled", f"{x['pooled']:.3f}", x["pooled"], re.escape(lab) + r"\s*&[^\n]*&\s*" + re.escape(f"{x['pooled']:.3f}") + r"\s*&\s*" + re.escape(f"{x['ref']:.3f}"))
    chk(f"OS table {lab} balanced median", f"{x['bal_median']:.3f}", x["bal_median"], re.escape(lab) + r"\s*&[^\n]*" + re.escape(f"{x['bal_median']:.3f}") + r"\s*&\s*" + re.escape(f"{x['bal_null_mean']:.3f}") + r"[^\n]*&\s*" + re.escape(f"{x['holm']:.2f}"))
    for r_, tag in [("struct", "(i)"), ("struct_topics", "(ii)"), ("rdf2vec", "(iii)")]:
        y_ = es[r_][k]
        cells = [f"{y_['pooled']:.3f}", f"{y_['ref']:.3f}", f"{y_['p_up']:.2f}/{y_['p_down']:.2f}", f"{y_['source_only']:.3f}", f"{y_['ordinary']:.3f}", f"{y_['bal_median']:.3f}",
                 f"{y_['bal_null_mean']:.3f} [{y_['bal_q025']:.2f}, {y_['bal_q975']:.2f}]", f"{y_['p2']:.2f}", f"{y_['holm']:.2f}"]
        chk(f"OS supplement row {tag} {lab}", cells[0], y_["pooled"], re.escape(lab) + r"\s*&\s*" + r"\s*&\s*".join(map(re.escape, cells)), f="supplement.tex")
for r_, a_, b_, c_, tag, nm in [("struct", "0.77", "0.62", "0.621", "(i)", "Structural"), ("struct_topics", "0.88", "0.65", "0.653", "(ii)", "Structural + topics"), ("rdf2vec", "0.85", "0.62", "0.624", "(iii)", "RDF2Vec (64-d)")]:
    d_ = er[r_]
    chk(f"OS closed form {tag}", c_, d_["thm2_pred"], r"\(0\.621, 0\.653, 0\.624\)")
    cells = [f"{d_['source_acc']:.3f}", f"{d_['incorpus_uniform']['real']:.3f}", f"{d_['incorpus_uniform']['null']:.3f}", f"{d_['thm2_pred']:.3f}", f"{d_['incorpus_matched']['real']:.3f}", f"{d_['incorpus_matched']['null']:.3f}"]
    chk(f"OS supplement in-corpus row {tag}", cells[1], d_["incorpus_uniform"]["real"], re.escape(tag) + r"[^&]*&\s*" + r"\s*&\s*".join(map(re.escape, cells)), f="supplement.tex")
    acc_s = f"{d_['source_acc']:.2f}"
    chk(f"main Table 3 OS row {tag}", cells[1], d_["incorpus_uniform"]["real"], row("", nm, acc_s, cells[1], cells[2], cells[3], cells[4], cells[5]))
gaps_os = [abs(er[r]["thm2_pred"] - er[r]["incorpus_uniform"]["null"]) for r in er]
chk("OS closed form within 0.008", "1", float(max(gaps_os) <= 0.008), r"every OpenSanctions reference to within 0\.008")
mm = [er[r]["incorpus_matched"]["real"] for r in er]
pass  # OS matched range sentence removed from main (review 4); values remain in Table 4 rows
sh = [(er[r]["incorpus_uniform"]["null"] - 0.5) / (er[r]["incorpus_uniform"]["real"] - 0.5) for r in er]
chk("OS share of margin low (%)", "34", min(sh), r"for 34--44\\% on OpenSanctions", pct=True); chk("OS share of margin high (%)", "44", max(sh), r"34--44\\%", pct=True)
refs_os = [es["struct"][k]["ref"] for k, _ in OSS]
chk("OS held-out ref low", "0.021", min(refs_os), r"Pooled held-out references range from 0\.021 to 0\.51"); chk("OS held-out ref high", "0.51", max(refs_os))
U = es["struct"]["gb_coh_disqualified"]
chk("OS UK pooled", "0.025", U["pooled"], r"The model scores 0\.025"); chk("OS UK reference", "0.021", U["ref"], r"against a reference of 0\.021")
chk("abstract OS", "0.025", U["pooled"], r"a held-out AUC of 0\.025 lies close to its reference of 0\.021")
chk("OS UK ref range", "1", float(f"{U['ref_lo']:.3f}" == "0.003" and f"{U['ref_hi']:.3f}" == "0.028"), r"0\.021 \[0\.003, 0\.028\]")
chk("OS Wikidata lower p", "0.01", es["struct"]["wikidata"]["p_down"], r"lies below its reference on Wikidata \(\$p=0\.01\$\)")
chk("OS smallest Holm p", "0.10", min(es["struct"][k]["holm"] for k, _ in OSS), r"smallest Holm-adjusted \$p\$ 0\.10")
chk("OS UK balanced null low", "0.01", U["bal_q025"], r"null spans 0\.01--0\.99"); chk("OS UK balanced null high", "0.99", U["bal_q975"])
chk("OS no reading anywhere", "1", float(all(es[r][k]["reading"] == "inconclusive" for r in es for k, _ in OSS)), r"no balanced score is read as a transfer or reversal")
chk("OS SAM/ACF constant", "1", float(all(es[r][k]["pooled"] == 0.5 and es[r][k]["ref_lo"] == es[r][k]["ref_hi"] for r in ["struct", "struct_topics"] for k in ["us_sam_exclusions", "ru_acf_bribetakers"])), r"SAM and ACF persons have no relationships and")
iso = P2.assign(iso=P2.n_nbrs == 0)
chk("OS Wikidata isolated (%)", "99.8", iso[iso.src == "wikidata"].iso.mean(), r"99\.8\\% of Wikidata persons none", pct=True)
chk("OS Wikidata with relations labelled (%)", "52", iso[(iso.src == "wikidata") & ~iso.iso].y.mean(), r"linked persons are 52\\% labelled", f="supplement.tex", pct=True)
uk = iso[iso.src == "gb_coh_disqualified"]
chk("OS UK sanctioned isolated (%)", "100", uk[uk.y == 1].iso.mean(), r"sanctioned persons are the ones without company links", pct=True)
chk("OS UK others with relationship (%)", "99.4", 1 - uk[uk.y == 0].iso.mean(), r"0\\% / 99\.4\\%", f="supplement.tex", pct=True); chk("OS UK others isolated (%)", "0.6", uk[uk.y == 0].iso.mean(), r"In the UK register, all sanctioned persons and 0\.6\\% of the others", f="supplement.tex", pct=True)
chk("OS UK isolated labelled", "0.91", uk[uk.iso].y.mean(), r"isolated persons of the UK register at 0\.91", f="supplement.tex")
chk("OS UK isolated n", "360", int(uk.iso.sum()), r"the 360 isolated persons", f="supplement.tex")
for kk, v in [("persons", "1{,}144{,}745"), ("no_nonlabel", "14{,}976"), ("multi", "728{,}552"), ("single_other", "270{,}622"), ("cohort", "130{,}595")]:
    chk(f"OS flow {kk}", v, FL[kk][0], f="supplement.tex")
chk("OS non-label datasets", "350", FL["nonlabel_datasets"], r"The other 350 datasets that record persons", f="supplement.tex")
chk("OS quads (new)", "2{,}435{,}390", int(re.search(r"quads (\d+)", aud).group(1)), f="supplement.tex")
chk("OS new majority class", "0.466", er["struct"]["majority"], r"majority class 0\.466", f="supplement.tex")
cand = pd.read_csv(f"{OS}/candidate_cells.csv")
for _, c in cand.iterrows():
    chk(f"OS candidate {c.d}", str(int(c.pos)), int(c.pos), re.escape(c.d.replace("_", "\\_")) + r"\}\s*&\s*" + re.escape(f"{int(c.n):,}".replace(",", "{,}")) + r"\s*&\s*" + re.escape(f"{int(c.pos):,}".replace(",", "{,}")), f="supplement.tex")
chk("OS new graph edges", "295{,}977", len(pd.read_parquet(f"{OSWORK2}/sgraph_edges.parquet")), r"source graph has 295\{,\}977 relationships", f="supplement.tex")

# ------------------------------------------------------------------ claims added in the style revision
if os.path.exists(f"{OS}/e2_summary.json"):
    allgap = gaps + [abs(fl["thm2_pred"] - fl["incorpus_uniform"]["null"]) if "thm2_pred" in fl else abs(fl["thm3_pred"] - fl["incorpus_uniform"]["null"])] + gaps_os
    chk("abstract: closed form within 0.008 on both graphs", "1", float(max(allgap) <= 0.008), r"On both graphs the closed form predicts the reference to within 0\.008"); chk("Discussion: closed form within 0.008 on both graphs", "1", float(max(allgap) <= 0.008), r"predicts the reference to within 0\.008 on both graphs")
    pairs = [(R[k]["incorpus_uniform"]["real"], R[k]["incorpus_uniform"]["null"]) for k in R] + [(rg["real"], rg["null"]),
             (fl["incorpus_uniform"]["real"], fl["incorpus_uniform"]["null"])] + [(er[r]["incorpus_uniform"]["real"], er[r]["incorpus_uniform"]["null"]) for r in er]
    share = min((n - 0.5) / (r_ - 0.5) for r_, n in pairs[:-3])
    chk("RQ1: reference > half of margin above chance, every ICIJ in-corpus score", "1", float(share > 0.5), r"more than half of the margin above chance of every in-corpus score on ICIJ and for 34--44")
    chk("Discussion: same claim", "1", float(share > 0.5), r"accounts for more than half of the margin above chance of every in-corpus score on ICIJ and for 34--44\\% on OpenSanctions \(RQ1\)")
    ref_os_in = [er[r]["incorpus_uniform"]["null"] for r in er]
    # (OpenSanctions in-corpus reference range no longer repeated in the Discussion)
    ho = [F8[f"lolo_{L}"]["null"] for L in LEAKS] + [x[f"lolo_{L}"]["null"] for x in (T, V, G) for L in LEAKS] + [es[r][k]["ref"] for r in es for k, _ in OSS]
    chk("Discussion held-out refs low", "0.003", min(ho), r"Held out, references range from 0\.003 to 0\.68")
    chk("Discussion held-out refs high", "0.68", max(ho))
    chk("RQ3 answer: all four ICIJ leaks transfer", "1", float(all(B9[L]["holm"] <= 0.05 and B9[L]["real_first"] > 0.5 for L in LEAKS)), r"to all four with leak strata and to three of them with finer strata")
    chk("RQ3 answer: no OS claim", "1", float(all(es[r][k]["reading"] == "inconclusive" for r in es for k, _ in OSS)), r"supports no cross-source claim on OpenSanctions")

# ------------------------------------------------------------------ review revision: held-out-label test, lambda, cohorts
HA = J("heldout_assoc_icij"); HO = json.load(open(f"{EXP}/opensanctions/heldout_assoc_os.json"))
for L in LEAKS:
    f0 = HA[L]["fits"][0]
    chk(f"held-out-label test re-creates first fit {SHORT[L]}", "1", float(abs(f0["auc"] - B9[L]["real_first"]) < 1e-9), r"every re-created AUC equals the recorded one", f="supplement.tex")
    chk(f"held-out-label p < 0.001 {SHORT[L]}", "1", float(f0["p_two"] < 0.001), r"gives \$p<0\.001\$ on every leak")
    chk(f"held-out null range within 0.48-0.52 {SHORT[L]}", "1", float(f0["held_q025"] >= 0.48 and f0["held_q975"] <= 0.52), r"null 95\\% ranges within 0\.48--0\.52")
    chk(f"supp held-out row {SHORT[L]}", f"{f0['auc']:.3f}", f0["auc"], row("", SHORT[L], f"{f0['auc']:.3f}", f"[{f0['held_q025']:.3f}, {f0['held_q975']:.3f}]"), f="supplement.tex")
    ups = sum(x["p_up"] <= 0.025 for x in HA[L]["fits"]); dns = sum(x["p_down"] <= 0.025 for x in HA[L]["fits"])
    chk(f"supp fits up/down {SHORT[L]}", str(ups), ups, re.escape(f"& up & {ups}/{dns} &"), f="supplement.tex")
    comb = min(1, 2 * min(max(B9[L]["p_one"], HA[L]["fits"][0]["p_up"]), 1.0))
    chk(f"combined p equals training p {SHORT[L]}", f"{B9[L]['p_two']:.3f}", comb, f="supplement.tex", ctx=re.escape(f"& {B9[L]['p_two']:.3f} & {B9[L]['p_two']:.3f} & {B9[L]['holm']:.3f}"))
OSN = {"ext_ru_egrul": "EGRUL", "us_sam_exclusions": "SAM", "wikidata": "Wikidata", "gb_coh_disqualified": "UK disq.", "ru_acf_bribetakers": "ACF"}
E2S = json.load(open(f"{EXP}/opensanctions/e2_summary.json"))["struct"]
for s, nm in OSN.items():
    f0 = HO[s]["fits"][0]; e = E2S[s]
    chk(f"OS held-out first fit {nm}", f"{f0['auc']:.3f}", f0["auc"], row("", nm, f"{f0['auc']:.3f}"), f="supplement.tex")
    chk(f"OS first fit equals recorded {nm}", "1", float(abs(f0["auc"] - e["bal_first"]) < 1e-9), f="supplement.tex", ctx=r"every re-created AUC equals the recorded one")
    if s in ("ext_ru_egrul", "wikidata", "gb_coh_disqualified"):
        chk(f"OS held-out p < 0.001 {nm}", "1", float(f0["p_two"] < 0.001), r"\(\$p<0\.001\$ each\)")
        ups = sum(x["p_up"] <= 0.025 for x in HO[s]["fits"]); dns = sum(x["p_down"] <= 0.025 for x in HO[s]["fits"])
        chk(f"OS fits up/down {nm}", str(ups), ups, re.escape(f"{ups}/{dns} & {e['p2']:.2f}"), f="supplement.tex")
egr = [x["auc"] for x in HO["ext_ru_egrul"]["fits"]]
chk("EGRUL fits range low", "0.29", min(egr), r"range from 0\.29 to 0\.66", f="supplement.tex"); chk("EGRUL fits range high", "0.66", max(egr), r"0\.29 to 0\.66", f="supplement.tex")
chk("Wikidata training p2 (supp)", "0.02", E2S["wikidata"]["p2"], r"the training-label test alone gives \$p_2=0\.02\$", f="supplement.tex")
chk("Panama inverted fit read as reversed", "1", float(HA["Panama Papers"]["fits"][3]["auc"] < 0.5 and HA["Panama Papers"]["fits"][3]["p_down"] <= 0.025), r"which the held-out-label test alone reads as reversed and the rule as inconclusive")
LI = J("lambda_icij"); LO = json.load(open(f"{EXP}/opensanctions/lambda_os.json"))
chk("lambda ICIJ q99.9 (supp)", "0.377", LI["quantiles"]["0.999"], f="supplement.tex")
pass
chk("factor ICIJ", "2.6", LI["factor_q999"], r"\\ge2\.6\$ on ICIJ"); chk("factor OS > 1e6", "1", float(LO["factor_q999"] > 1e6), r"\$>10\^\{6\}\$", f="supplement.tex")
D = LI; nm = "ICIJ, eight features"
chk(f"lambda row {nm}", f"{D['quantiles']['0.999']:.3f}", D["quantiles"]["0.999"], row("", nm, f"{D['quantiles']['0.5']:.3f}", f"{D['quantiles']['0.99']:.3f}", f"{D['quantiles']['0.999']:.3f}", f"{D['max']:.3f}", f"{D['factor_q999']:.1f}"), f="supplement.tex")
chk("lambda max factor ICIJ", f"{D['factor_max']:.0f}", D["factor_max"], f="supplement.tex", ctx=re.escape(f"/ {D['factor_max']:.0f} \\\\"))
D = LO; nm = "OpenSanctions, structural"
chk(f"lambda row {nm}", f"{D['quantiles']['0.999']:.3f}", D["quantiles"]["0.999"], row("", nm, f"{D['quantiles']['0.5']:.3f}", f"{D['quantiles']['0.99']:.3f}", f"{D['quantiles']['0.999']:.3f}", f"{D['max']:.3f}"), f="supplement.tex")
chk("lambda calibration ICIJ pred", "0.246", LI["top1pct_mean_pred"], f="supplement.tex"); chk("lambda calibration ICIJ obs", "0.243", LI["top1pct_obs_rate"], f="supplement.tex")
chk("lambda calibration OS pred", f"{LO['top1pct_mean_pred']:.3f}", LO["top1pct_mean_pred"], f="supplement.tex"); chk("lambda calibration OS obs", f"{LO['top1pct_obs_rate']:.3f}", LO["top1pct_obs_rate"], f="supplement.tex")
P = pd.read_parquet(f"{OSWORK}/persons.parquet"); U = P[P.src == "gb_coh_disqualified"]; iso = U.n_nbrs == 0
CO = json.load(open(f"{EXP}/opensanctions_first/os_cohort.json")); MU = json.load(open(f"{EXP}/opensanctions_first/os_multi_lf.json"))
chk("cohort persons", "1{,}144{,}745", CO["persons"], f="supplement.tex"); chk("cohort persons pos", "31{,}693", CO["persons_pos"], f="supplement.tex")
chk("cohort none", "550{,}385", CO["persons_no_S"]["n"], f="supplement.tex"); chk("cohort none pos", "15{,}168", CO["persons_no_S"]["pos"], f="supplement.tex")
nm_ = sum(v["n"] for k, v in CO["by_k"].items() if int(k) >= 2); pm_ = sum(v["pos"] for k, v in CO["by_k"].items() if int(k) >= 2)
chk("multi-source persons", "7{,}887", nm_, r"the 7\{,\}887 persons recorded by several of the five sources are 70\\% labelled"); chk("multi-source persons (supp)", "7{,}887", nm_, f="supplement.tex"); chk("multi pos", "5{,}557", pm_, f="supplement.tex")
chk("multi = os_multi", "1", float(nm_ == MU["n_multi"] and pm_ == MU["pos_multi"]), r"7\{,\}887 persons recorded by two or more", f="supplement.tex")
for k in ["2", "3", "4", "5"]:
    chk(f"cohort k={k}", f"{CO['by_k'][k]['n']:,}".replace(",", "{,}"), CO["by_k"][k]["n"], f="supplement.tex")
chk("combinations", "26", len(CO["multi_combos"]), r"fall into 26 combinations", f="supplement.tex")
chk("all 4/5-source labelled", "1", float(all(CO["by_k"][k]["n"] == CO["by_k"][k]["pos"] for k in ["4", "5"])), r"every person recorded by four or five sources is labelled", f="supplement.tex")
chk("multi labelled (%)", "70", MU["rate_multi"], r"are 70\\% labelled", pct=True); chk("multi labelled (supp, %)", "70.5", MU["rate_multi"], r"70\.5\\\% are labelled", f="supplement.tex", pct=True)
chk("single labelled (%)", "1.9", MU["rate_single"], r"against 1\.9\\% of the others", pct=True)
chk("feature table reproduced", "1", float(MU["feature_check_maxabs"] == 0), r"reproduce its feature table exactly", f="supplement.tex")
for key, nm in [("combination", "Combination of sources as its own stratum"), ("largest", "Largest member source"), ("smallest", "Smallest member source")]:
    m = MU[key]
    chk(f"multi row {key}", f"{m['ref']:.3f}", m["ref"], row("", nm, str(m["strata"]), f"{m['real']:.3f}", f"{m['ref']:.3f}", f"{m['thm2']:.3f}"), f="supplement.tex")
chk("multi ref combination (main)", "0.79", MU["combination"]["ref"], r"\(0\.79 with combination strata; supplement\)")
chk("multi ref smallest (main)", "0.79", MU["smallest"]["ref"], r"moves from 0\.68 to 0\.79 across label-free assignments"); chk("multi ref largest (main)", "0.68", MU["largest"]["ref"], r"from 0\.68 to 0\.79")
chk("multi real (supp)", "0.832", MU["combination"]["real"], f="supplement.tex")
chk("multi thm2 within 0.001", "1", float(max(abs(MU[k]["ref"] - MU[k]["thm2"]) for k in ["combination", "largest", "smallest"]) <= 0.001), r"tracks the reference to within 0\.001", f="supplement.tex")
chk("multi ref spread (supp)", "0.11", MU["smallest"]["ref"] - MU["largest"]["ref"], r"moves by 0\.11 across", f="supplement.tex")
chk("first cohort row", "0.782", es1["struct"]["in_U"][1] if "in_U" in es1["struct"] else er1["struct"]["incorpus_uniform"]["null"], row("", "First cohort (single-source persons only)", "5", "0.830", "0.782", "0.780"), f="supplement.tex")

# ------------------------------------------------------------------ second review: finer strata and plug-in robustness
SS = J("subsrc")
GR = J("gran_icij")
_es2 = json.load(open(f"{EXP}/opensanctions/e2_summary.json"))
_h = min(v["holm"] for r in ["struct_topics", "rdf2vec"] for v in _es2[r].values() if isinstance(v, dict) and "holm" in v)
chk("OS (ii),(iii) smallest Holm (supp)", "0.25", _h, r"smallest Holm-adjusted \$p\$ 0\.25, for the UK register with RDF2Vec", f="supplement.tex")
chk("granularity: one stratum (main)", "0.500", GR["one stratum"]["ref"], r"one stratum gives 0\.500")
chk("granularity: random halves (main)", "0.651", GR["leak, random halves"]["ref"], r"a random split of each leak 0\.651")
chk("granularity: one stratum (supp)", "0.500", GR["one stratum"]["ref"], r"the reference is 0\.500; with each leak", f="supplement.tex")
chk("granularity: random halves (supp)", "0.651", GR["leak, random halves"]["ref"], r"it stays at 0\.651 \(Theorem~2: 0\.652\)", f="supplement.tex")
chk("granularity: random halves thm2 (supp)", "0.652", GR["leak, random halves"]["thm2"], f="supplement.tex")
chk("granularity: sourceID reference agrees with subsrc", "1", float(abs(GR["sourceID"]["ref"] - 0.660) < 0.0005), r"With these strata the in-corpus reference is 0\.660", f="supplement.tex")
fs = {k: v[1] / v[0] for k, v in GR["strata"]["sourceID"].items()}
chk("Appleby rate (%)", "1.9", fs["Paradise Papers - Appleby"], r"Appleby 1\.9\\%, Malta", f="supplement.tex", pct=True)
chk("Malta rate (%)", "0.5", fs["Paradise Papers - Malta corporate registry"], r"Malta 0\.5\\%, Aruba", f="supplement.tex", pct=True)
chk("Aruba rate (%)", "0.03", fs["Paradise Papers - Aruba corporate registry"], r"Aruba 0\.03\\%", f="supplement.tex", pct=True)
chk("ICIJ sub-source strata count", "eleven", 0, r"by the eleven sub-sources tagged on officer nodes") if False else None
chk("ICIJ sub-source strata count (n)", "11", len(SS["strata"]), r"eleven strata nested in the four leaks", f="supplement.tex")
chk("ICIJ sub-source in-corpus ref", "0.660", SS["incorpus_sub"]["ref"], r"move the in-corpus reference from 0\.651 to 0\.660")
chk("ICIJ sub-source in-corpus ref (supp)", "0.660", SS["incorpus_sub"]["ref"], f="supplement.tex"); chk("ICIJ sub-source thm2 (supp)", "0.662", SS["thm2_sub"], f="supplement.tex")
chk("ICIJ sub-source real (supp)", "0.762", SS["incorpus_sub"]["real"], r"against a real score of 0\.762", f="supplement.tex")
for L in LEAKS[:3]:
    chk(f"ICIJ sub-source Holm {SHORT[L]}", "0.04", SS[L]["holm"], r"keep three readings \(Holm-adjusted \$p\$ 0\.04 each\)")
chk("ICIJ sub-source Bahamas p2", "0.06", SS["Bahamas Leaks"]["p2"], r"inconclusive \(\$p_2\$ 0\.06\)")
chk("ICIJ sub-source Bahamas Holm > 0.05", "1", float(SS["Bahamas Leaks"]["holm"] > 0.05), r"make the weakest, Bahamas, inconclusive")
for L in LEAKS:
    x = SS[L]; rd = "transfer" if x["holm"] <= 0.05 else "inconclusive"
    cells = [f"{F8['lolo_' + L]['null']:.3f}", f"{x['pooled_ref']:.3f}", f"{B9[L]['null_mean']:.3f}", f"{x['bal_null_mean']:.3f} [{x['bal_q025']:.2f}, {x['bal_q975']:.2f}]",
             f"{min(1, 2 * min(x['pT_up'], x['pT_down'])):.3f}", f"{x['p2']:.2f}", f"{x['holm']:.2f}", rd]
    chk(f"supp sub-source row {SHORT[L]}", cells[1], x["pooled_ref"], row("", SHORT[L], *cells), f="supplement.tex")
chk("sub-source Panama null shift (supp)", "0.63", SS["Panama Papers"]["bal_null_mean"], r"balanced null shifts to 0\.63", f="supplement.tex")
for s_, v in SS["strata"].items():
    chk(f"sub-source stratum {s_}", f"{v['n']:,}".replace(",", "{,}"), v["n"], re.escape(s_.replace("Paradise Papers - ", "Paradise: ")) + r"\s*&\s*" + re.escape(f"{v['n']:,}".replace(",", "{,}")) + r"\s*&\s*" + re.escape(f"{v['pos']:,}".replace(",", "{,}")), f="supplement.tex")
PR = J("plugin_robust_icij"); PRO = json.load(open(f"{OS}/plugin_robust_os.json"))
best_dev, all_dev = [], []
for D in (PR, PRO):
    for rep_, v in D.items():
        cs = [c for c in v if c != "observed"]
        b_ = min((c for c in cs if v[c]["logloss"] is not None), key=lambda c: v[c]["logloss"])
        best_dev.append(abs(v[b_]["plugin"] - v["observed"])); all_dev += [abs(v[c]["plugin"] - v["observed"]) for c in cs]
        chk(f"plug-in table row {rep_}", f"{v['observed']:.3f}", v["observed"], re.escape(rep_) + r"\s*&\s*" + re.escape(f"{v['observed']:.3f}") + r"\s*&\s*" + re.escape(f"{v[cs[0]]['plugin']:.3f}"), f="supplement.tex")
_t3 = {"Connectivity (3)": "conn", "Structural (8)": "full8", "Structural + graph (12)": "full12", "TransE": "transe", "RDF2Vec": "rdf2vec"}
chk("plug-in in-sample within 0.003 of Table 3", "1", float(max(abs(PR[k]["boosting (as reported)"]["plugin"] - R[v]["thm3_pred"]) for k, v in _t3.items()) <= 0.003), r"within 0\.003 of Table~4 of the paper", f="supplement.tex")
chk("plug-in best within 0.009", "1", float(max(best_dev) <= 0.009), r"stays within 0\.009 of the observed reference on both graphs")
chk("plug-in best within 0.009 (supp)", "1", float(max(best_dev) <= 0.009), r"within 0\.009 of the observed reference everywhere", f="supplement.tex")
chk("plug-in worst up to 0.03", "0.03", max(all_dev), r"move it by up to 0\.03", f="supplement.tex")

# ------------------------------------------------------------------ third review: stability, balancing, coverage, bootstrap
PF = J("perfit_readings")
chk("per-fit transfers (of 40)", "36", sum(PF[L]["n_transfer"] for L in LEAKS), r"36 of the 40 fits read as transfers and none as a reversal")
chk("per-fit reversals", "0", sum(PF[L]["n_reversal"] for L in LEAKS), r"36 of the 40 fits read as transfers and none as a reversal")
for L, n in zip(LEAKS, ["9", "10", "9", "8"]):
    chk(f"per-fit transfers {SHORT[L]} (supp)", n, PF[L]["n_transfer"], f="supplement.tex", ctx=re.escape(SHORT[L]) + r"\s*&\s*" + n + r"\s*&")
_cnt = dict(zip(LEAKS, [11251, 1969, 988, 670]))
_m = {L: min(v for k, v in _cnt.items() if k != L) for L in LEAKS}
chk("balanced m ICIJ (supp)", "670", min(_m.values()), r"limits \$m\$ to 670", f="supplement.tex"); chk("balanced m ICIJ Bahamas (supp)", "988", _m["Bahamas Leaks"], r"limits it to 988 when Bahamas", f="supplement.tex")
chk("balanced positives ICIJ 3x670 (supp)", "2,010", 3 * 670, r"2\{,\}010 positives and 6\{,\}030", f="supplement.tex"); chk("balanced positives ICIJ 3x988 (supp)", "2,964", 3 * 988, r"\(2\{,\}964 and 8\{,\}892\)", f="supplement.tex")
_osc = {k: v["pos"] for k, v in json.load(open(f"{OS}/os_coverage.json")).items()}
_mo = {k: min(v for kk, v in _osc.items() if kk != k) for k in _osc}
chk("balanced m OS (supp)", "187", min(_mo.values()), r"ACF limits \$m\$ to 187 \(748 positives", f="supplement.tex"); chk("balanced m OS ACF held out (supp)", "308", _mo["ru_acf_bribetakers"], r"Wikidata to 308 when ACF is held out \(1\{,\}232\)", f="supplement.tex")
CV = json.load(open(f"{OS}/os_coverage.json"))
for k, lab in [("ext_ru_egrul", "EGRUL"), ("us_sam_exclusions", "SAM"), ("wikidata", "Wikidata"), ("gb_coh_disqualified", "UK disq."), ("ru_acf_bribetakers", "ACF")]:
    v = CV[k]; f_ = lambda x: f"{100*x:.1f}".rstrip("0").rstrip(".") if x not in (0, 1) else ("0" if x == 0 else "100")
    cells = [f"{v['n']:,}".replace(",", "{,}"), f"{v['pos']:,}".replace(",", "{,}"), f"{f_(v['rel_pos'])}\\% / {f_(v['rel_oth'])}\\%", f"{v['distinct_i']:,}".replace(",", "{,}") + " / " + f"{v['distinct_ii']:,}".replace(",", "{,}")]
    chk(f"coverage row {lab} (supp)", f_(v["rel_pos"]), v["rel_pos"], r"\s*&\s*".join([re.escape(lab)] + [re.escape(c) for c in cells]), f="supplement.tex", pct=True)
    chk(f"coverage others {lab} (supp)", f_(v["rel_oth"]), v["rel_oth"], f="supplement.tex", pct=True, ctx=re.escape(cells[2]))
chk("coverage Wikidata linked labelled (supp)", "52", CV["wikidata"]["linked_labelled"], r"linked persons are 52\\% labelled", f="supplement.tex", pct=True)
BT = J("boot_icij")
for L in LEAKS:
    b = BT[L]; chk(f"bootstrap {SHORT[L]} (supp)", f"{b['lo']:.3f}", b["lo"], f="supplement.tex", ctx=re.escape(f"[{b['lo']:.3f}, {b['hi']:.3f}]"))
    chk(f"bootstrap first fit equals recorded {SHORT[L]}", "1", float(abs(b["auc"] - HA[L]["fits"][0]["auc"]) < 1e-9))

AG = J("agg_fits_summary")
chk("aggregate: all four leaks transfer", "1", float(all(AG[L]["reading"] == "transfer" for L in LEAKS)), r"reads every leak as a transfer with Holm-adjusted \$p\$ 0\.04")
chk("aggregate: Holm 0.04 everywhere", "0.04", max(AG[L]["holm"] for L in LEAKS), r"Holm-adjusted \$p\$ 0\.04, the floor that 199 draws allow")
for L in LEAKS:
    a = AG[L]; chk(f"aggregate median equals Table 4 median {SHORT[L]}", "1", float(abs(a["median"] - np.median(B99[L]["real_draws"])) < 1e-9))
    chk(f"aggregate row {SHORT[L]} (supp)", f"{a['median']:.3f}", a["median"], f="supplement.tex", ctx=re.escape(f"{SHORT[L]} & {a['median']:.3f} & {a['null_mean']:.3f} [{a['q025']:.2f}, {a['q975']:.2f}] & {a['pT2']:.3f}"))
chk("aggregate Panama q97.5 (supp)", "0.633", AG["Panama Papers"]["q975"], r"97\.5\\% quantile 0\.633 against 0\.649", f="supplement.tex")
chk("single-fit Panama q97.5 (supp)", "0.649", B9["Panama Papers"]["q975"], r"0\.633 against 0\.649", f="supplement.tex")

# ------------------------------------------------------------------ Part B sweep
pool = []
def flatten(o):
    if isinstance(o, dict):
        for v in o.values(): flatten(v)
    elif isinstance(o, list):
        if o and all(isinstance(x, (int, float)) for x in o) and len(o) > 12:
            a = np.array(o, float); pool.extend([a.mean(), np.median(a), a.std(ddof=1), a.min(), a.max(), np.quantile(a, .025), np.quantile(a, .975)])
        else:
            for x in o: flatten(x)
    elif isinstance(o, (int, float)) and not isinstance(o, bool):
        pool.append(float(o))
for f in glob.glob(f"{EXP}/*.json") + glob.glob(f"{EXP}/opensanctions/*.json") + glob.glob(f"{EXP}/opensanctions_first/*.json") + glob.glob(f"{EXP}/simulation/round2/sim2_summary.json") + [f"{EXP}/simulation/table.json"]:
    flatten(json.load(open(f)))
for f in glob.glob(f"{EXP}/*.log") + glob.glob(f"{EXP}/simulation/*.log") + glob.glob(f"{EXP}/theory/*.log") + ([ARRAY_TXT] if ARRAY_TXT else []):
    pool.extend(float(x) for x in re.findall(r"(?<![\w.])-?\d+\.\d+", open(f, errors="ignore").read()))
pool = np.array(pool); orphans = []
for fn, txt in TEX.items():
    for ln, line in enumerate(txt.splitlines(), 1):
        line = re.sub(r"(?<!\\)%.*", "", line)
        for m in re.finditer(r"(?<![\w.\\{])(\d+\.\d+)", line):
            s = m.group(1); x = float(s); tol = 0.5 * 10 ** -len(s.split(".")[1]) + 1e-9
            if not np.any((np.abs(pool - x) <= tol) | (np.abs(100 * pool - x) <= tol)):
                orphans.append(f"{fn}:{ln}  {s}  | {line[max(0, m.start()-50):m.end()+20].strip()}")
with open(report, "w") as f:
    f.write("\n".join(results) + "\n\nPart B: numbers with no candidate in any result file\n" + "\n".join(orphans) + "\n")
print(f"Part A: {len(results)} checks, {len(fails)} failed")
for l in fails: print(l)
print(f"Part B: {len(orphans)} numbers with no candidate")
for l in orphans: print(l)
