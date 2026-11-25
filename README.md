# Within-source permutation references for multi-source knowledge graphs

Code, results and supplement for the paper *"Provenance-Aware Evaluation of Learning on Multi-Source Knowledge Graphs: Why Chance
Is the Wrong Reference"* (ESWC 2027 submission, anonymous).

Every table and figure in the paper is produced by a script in this repository from the files in `results/`,
and `crosscheck.py` recomputes every number printed in the paper from those files (see `results/crosscheck_report.txt` for the number of checks).

## Layout

| Folder | Contents |
|---|---|
| `wsref/` | Stand-alone implementation of the method (numpy, plus scipy for the AUC helper): within-source permutation, reference, p-values, the held-out-label test, the decision rule (Corollary 2), Holm adjustment, minimum number of draws, provenance score and the in-corpus closed form. `test_wsref.py` runs unit tests; `--icij` re-derives the Holm-adjusted readings of the held-out table from `results/`. |
| `icij/` | The ICIJ Offshore Leaks case study: officer table and features, RDF conversion and SPARQL audit, TransE, RDF2Vec, R-GCN, the evaluation protocol (`evallib.py`) and every analysis script. Paths are set in `icij/config.py` (environment variables `ICIJ_DIR`, `WORK_DIR`, `RESULTS_DIR`). |
| `opensanctions/` | The OpenSanctions experiment (§5.1–5.3): download and reduction script, analysis plan (`PREREG.md`), person table and source graph, RDF and SPARQL audit, RDF2Vec, evaluation and summary. Paths in `opensanctions/config_os.py` (`OS_EXTRACT_DIR`, `OS_WORK_DIR`). |
| `simulation/` | Decision-rule simulations: first round (324 settings, `sim.py`, `extend.py`, `analyze.py`) and second round (`round2/`: 936 targets with replicates, training-label and held-out-label tests; Table 2). |
| `theory/` | Numerical checks of the theoretical results on populations where everything is known. |
| `figures/` | `figs.py` draws the figures from `results/` (files `fig1_mechanism` = Fig. 1, `fig3_heldout` = Fig. 2 (ICIJ and OpenSanctions held-out readings with both tests; reads `results/heldout_assoc_icij.json` and `results/opensanctions/`), `fig4_thm4` = supplement Fig. S1, `fig2_thm3` = supplement Fig. S2). |
| `results/` | All result files (JSON), run logs, simulation records and theory-check logs. |
| `supplement/` | Supplementary material: full proofs, numerical checks, ICIJ factorisation and bound tables, full Elliptic analysis. |
| `crosscheck.py` | Recomputes every table cell and quantitative claim of the paper from `results/` and checks it against the LaTeX source at the printed precision. |
| `data/` | How to obtain the input data, with checksums. |

## Environment

Python 3.11+ and the packages in `requirements.txt`; `requirements-lock.txt` pins the exact versions used for the paper (Python 3.13). Everything ran on 2 CPU cores with 7 GB RAM and no GPU;
the R-GCN runs take 30-70 s per fit, TransE about 8 min, RDF2Vec about 9 min. When two gradient-boosting jobs run at
once, set `OMP_NUM_THREADS=1`. All hyperparameters are fixed in the scripts; nothing is tuned inside a pipeline.

## Reproducing the paper

Set `ICIJ_DIR` to the extracted CSV export (see `data/README.md`) and run from `icij/`
(`WORK_DIR` defaults to `./work`, `RESULTS_DIR` to `./results`).

| Paper item | Command | Output |
|---|---|---|
| Officer table, leak, label, 8+4 features | `python3 prep.py $ICIJ_DIR $WORK_DIR` | `work/officers.parquet` |
| Supplement leak table, RDF and SPARQL audit (§6) | `python3 to_rdf.py $ICIJ_DIR $WORK_DIR $WORK_DIR/icij.nq`; `python3 load_query.py $WORK_DIR/oxstore $WORK_DIR/icij.nq audit.rq` | per-leak label rates |
| Table 3 (ICIJ rows) and supplement Fig. S2, feature models | `python3 baseline.py $WORK_DIR full8 10 99 5 29 ../results/full8.json` (also `conn`, `full12`; draw counts are recorded in each JSON, field `B`) | `results/{conn,full8,full12}.json` |
| TransE | `python3 transe.py $ICIJ_DIR $WORK_DIR 64 20 $WORK_DIR/emb_transe.npy`; `python3 baseline.py $WORK_DIR transe 5 29 3 19 ../results/transe.json $WORK_DIR/emb_transe.npy` | `results/transe.json` |
| RDF2Vec | `python3 rdf2vec.py $ICIJ_DIR $WORK_DIR 8 4 64 5 $WORK_DIR/emb_rdf2vec.npy`; `python3 baseline.py $WORK_DIR rdf2vec 5 29 3 39 ../results/rdf2vec.json $WORK_DIR/emb_rdf2vec.npy`; `python3 rdf2vec_cos.py $ICIJ_DIR ../results/rdf2vec_cos.json` | `results/rdf2vec*.json` |
| R-GCN | `python3 rgcn_run.py ../results/rgcn_pooled.json 3 19 0 0 0 0`; `python3 rgcn_run.py ../results/rgcn_bal_in.json 0 0 2 9 2 5`; `python3 rgcn_ident.py` (leak accuracy and Theorem 2 on the two-hop receptive field) | `results/rgcn_*` |
| Table 4 and Fig. 2 (left), 999 balanced draws | `python3 bal99.py`; `python3 bal999.py 0 2` and `python3 bal999.py 1 2`; `python3 bal999_summary.py ../results` | `results/full8_bal99.json`, `results/bal900_*.json`, `results/full8_bal999_summary.json` |
| Single-source functional label | `python3 funclabel.py ../results/funclabel.json`; `python3 funclabel_b.py` | `results/funclabel.json` |
| Supplement Fig. S1 (Theorem 3 on ICIJ) | `python3 thm4_data.py ../results/thm4_scores.npz` | `results/thm4_scores.npz` |
| Theorem 4 on ICIJ (factorisation, masking) | `python3 factor_check.py ../results/factor_check.json` | `results/factor_check.json` |
| Equal-mean-signal check | `python3 assumption_check.py ../results/assumption_check.json` | `results/assumption_check.json` |
| Proposition 2 bound | `python3 bound_check.py ../results/bound_check.json` | `results/bound_check.json` |
| Independent re-implementation checks | `python3 verify.py`; `python3 check_incorpus.py` | stdout |
| Supplement Section 4, first-round decision-rule simulation | from `simulation/`: `python3 sim.py 0 2 sim_0.jsonl` and `... 1 2 sim_1.jsonl`; `python3 extend.py 0 2 sim_0.jsonl sim_1.jsonl --out ext_0.jsonl --extra 20` (and shard 1); `python3 analyze.py ext_0.jsonl ext_1.jsonl` | `results/simulation/` |
| Supplement, numerical checks | from `theory/`: `python3 check_theory.py`; `python3 check_t5.py` | `results/theory/*.log` |
| Figures | from `figures/`: `python3 figs.py ../results figs` | `fig1`-`fig4` |
| OpenSanctions extract (§5.1–5.3) | on any machine: `python3 opensanctions/get_opensanctions.py --with-statements`; unzip the package into `OS_EXTRACT_DIR` | extract with provenance and checksums |
| OpenSanctions person tables, source graphs, features | from `opensanctions/`: `python3 build2.py $OS_WORK_CELLS_DIR` (the paper's count-invariant cohort: single-dataset cells with at least 150 positives, Proposition 3; writes `candidate_cells.csv`) and `python3 build.py $OS_WORK_DIR` (the analysis plan's first cohort, supplement) | `persons.parquet`, `sgraph_edges.parquet` |
| OpenSanctions SPARQL audits | `python3 to_rdf_os2.py $OS_WORK_CELLS_DIR $OS_WORK_CELLS_DIR/e2.nq`; `python3 ../icij/load_query.py $OS_WORK_CELLS_DIR/oxstore $OS_WORK_CELLS_DIR/e2.nq audit_os2.rq` (first cohort: `to_rdf_os.py`, `audit_os.rq`) | `results/opensanctions*/sparql_audit.log` |
| OpenSanctions RDF2Vec | `python3 rdf2vec_os.py $OS_WORK_DIR 8 4 64 5 $OS_WORK_DIR/emb_rdf2vec.npy` | embedding |
| Table 3 (OpenSanctions rows), Table 5, Fig. 2 (right) and supplement Section 7 (set `OS_COHORT=first` and the `opensanctions_first` output paths for the first cohort) | `python3 e2eval.py struct 10 99 10 199 ../results/opensanctions/struct.json`; `python3 e2eval.py struct_topics 5 29 5 39 ...`; `python3 e2eval.py rdf2vec 5 29 5 39 ...`; `python3 compile_e2.py ../results/opensanctions`; `python3 e2_audit.py $OS_EXTRACT_DIR $OS_WORK_DIR ../results/opensanctions/sparql_audit.log ../results/opensanctions/e2_audit.json` | `results/opensanctions/` |
| Held-out-label test (§3.2, Tables 4 and 5, Fig. 2, supplement Section 6) | from `icij/`: `python3 heldout_assoc.py icij ../results/heldout_assoc_icij.json`; `python3 heldout_assoc.py os ../results/opensanctions/heldout_assoc_os.json` (re-creates the real balanced fits with seeds 400+k and 300+k and checks them against the recorded AUCs; 9,999 label permutations, seeds 777+fit) | `results/heldout_assoc_*.json` |
| Learned representations, balanced readings with 199 draws (supplement Section 6) | `python3 learned_bal.py $WORK_DIR/emb_transe.npy ../results/learned_bal_transe.json`; same with `emb_rdf2vec.npy` (seeds 600+k; held-out-label seeds 888+fit) | `results/learned_bal_*.json` |
| λ estimates for Theorem 4(c) (supplement) | `python3 lambda_est.py icij ../results/lambda_icij.json`; `python3 lambda_est.py os ../results/opensanctions/lambda_os.json` (fold seed 3) | `results/lambda_*.json` |
| Table 2, second-round decision-rule simulation | from `simulation/round2/`: `python3 sim2.py 0 2 sim2_0.jsonl` and `python3 sim2.py 1 2 sim2_1.jsonl` (seed 50000+job); `python3 analyze2.py sim2_summary.json sim2_0.jsonl sim2_1.jsonl` | `results/simulation/round2/` |
| OpenSanctions persons with several sources (§6, supplement Section 7) | from `opensanctions/`: `python3 os_cohort.py ../results/opensanctions_first/os_cohort.json`; `python3 os_multi_lf.py ../results/opensanctions_first/os_multi_lf.json` (label-free assignments; `os_multi.py` is the earlier, rate-based version) | `results/opensanctions_first/os_*.json` |
| Granularity of the strata on ICIJ (§6, supplement Section 5) | from `icij/`: `python3 subsrc.py ../results/subsrc.json` (eleven sub-sources: in-corpus and held-out readings); `python3 granularity_icij.py $ICIJ_DIR ../results/gran_icij.json` (leaks, sub-sources, one stratum, random halves) | `results/subsrc.json`, `results/gran_icij.json` |
| Plug-in robustness (§5.2, supplement) | from `icij/`: `python3 plugin_robust.py icij ../results/plugin_robust_icij.json`; `python3 plugin_robust.py os2 ../results/opensanctions/plugin_robust_os.json` | `results/plugin_robust_*.json` |
| Stability of the ICIJ readings across fits (§5.3, supplement Section 6) | from `icij/`: `python3 perfit_readings.py ../results ../results/perfit_readings.json` (every fit read with the rule); `OMP_NUM_THREADS=1 python3 agg_fits.py 0 2 199 ../results` and `... 1 2 199 ../results` (median of ten fits as the statistic, ten refits per null draw, seeds 9000+leak; about 40 min per shard on one core); `python3 boot_icij.py ../results/boot_icij.json` (bootstrap intervals of the tested fit) | `results/perfit_readings.json`, `results/agg_fits_{0..3}.json`, `results/boot_icij.json` |
| OpenSanctions coverage table (supplement Section 7) | from `opensanctions/`: `python3 os_coverage.py $OS_WORK_CELLS_DIR/persons.parquet $OS_WORK_CELLS_DIR/struct_cols.txt ../results/opensanctions/os_coverage.json` | `results/opensanctions/os_coverage.json` |
| Number check | `python3 crosscheck.py <dir with main.tex and supplement.tex> $ICIJ_DIR report.txt` | `results/crosscheck_report.txt` |

## Seeds

Every run is seeded; nothing depends on the order in which jobs are started. In `evallib.run(..., seed)` the real
fits are drawn first and the null draws after them, all from `numpy.random.default_rng(seed)`; each model fit takes
its own seed from that generator.

| Run | Seed |
|---|---|
| Source classifier, Theorem 2 plug-in (`baseline.py`, `e2eval.py`) | 7 |
| Held-out evaluation sample of unlabelled nodes (`lolo_sets`) | 13 |
| In-corpus split and draws (uniform / source-matched) | 100 / 101 (draws 1000+b for the permuted splits) |
| Leave-one-source-out, pooled | 200 + index of the held-out source |
| Leave-one-source-out, balanced (TransE, RDF2Vec, R-GCN and OpenSanctions first round) | 300 + index |
| Eight-feature balanced, 99 draws (`bal99.py`) / extra 900 draws (`bal999.py`) | 400 + index / 7000 + index |
| Ordinary (source-ignoring) permutation | 5 |
| Learned representations, 199 balanced draws (`learned_bal.py`) | 600 + index |
| Held-out-label permutations (`heldout_assoc.py` / `learned_bal.py`) | 777 + fit / 888 + fit |
| TransE, RDF2Vec embeddings | 0 |
| Simulation, first round / extension / second round | 1000 + setting / 900000 + setting / 50000 + job |

`evallib.py` holds the protocol. In-corpus nulls permute labels within leaks before the train/test split, so
evaluation labels are permuted too (the quantity Theorem 2 describes). Held-out nulls permute training labels
only and score against the held-out leak's real labels. Representations of a held-out leak are computed from
its own structure only; no relationship in the export connects nodes of different leaks.

Using the method on another graph needs only `wsref/`: wrap the pipeline in a function of the label vector and
pass the per-node source (for example the named graph that asserted each node's label); see `wsref/README.md`.

## Data

The ICIJ Offshore Leaks database is public (see `data/README.md`); it is not redistributed here. Appearance in the
leaks is not evidence of wrongdoing; we release no ranked lists of named entities, and no result file contains
names.
