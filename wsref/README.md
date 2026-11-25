# wsref: within-source permutation references for multi-source evaluation

When a graph is integrated from sources that record positives at different rates, and the representation reveals
the source, a model can score well (or badly) by recognising the source. Chance is then the wrong reference.
`wsref` computes the right one: the score of the *same pipeline* retrained after labels are permuted within each
source.

Only `numpy` is required (plus `scipy` for the AUC helper of the held-out-label test). The module never sees your features; you wrap your pipeline in a function.

```python
import numpy as np, wsref

def evaluate(labels):
    """Retrain the whole pipeline on `labels` (full-length 0/1 vector) and return the statistic
    (AUC, AP, MRR, ...) computed against the REAL labels of the evaluation set."""
    ...

# Held-out source: permute only the training positions
ref = wsref.reference(evaluate, y, sources, B=99, permute_idx=train_idx)
real = evaluate(y)
print(real, ref.mean, ref.quantiles(), wsref.p_value(real, ref.null))

# Does the signal transfer? Two tests of ONE pre-specified source-balanced fit must agree (Corollary 2):
#  - training-label test: the balanced pipeline refitted on within-source permuted training labels
#    (exact when the training labels carry no within-source signal)
#  - held-out-label test: the fixed fit's held-out scores against permuted held-out labels, no refits
#    (exact when the held-out labels are not associated with the scores, whatever training contains)
pt = wsref.two_tailed(balanced_real, balanced_ref.null)
pa = wsref.heldout_label_test(balanced_scores_on_heldout, y_heldout, M=9999)
call = wsref.read_transfer(pt, pa, alpha=0.05)        # 'transfers' | 'reversal' | 'inconclusive'

# Several held-out sources: Holm-adjust the combined two-sided p-values; make sure B is large enough
B_needed = wsref.min_draws(alpha=0.05, k=4)          # 159
adjusted = wsref.holm([wsref.combined_p2(pt_k, pa_k) for pt_k, pa_k in zip(train_tests, heldout_tests)])

# In-corpus reference without any permutation (Theorem 2): needs per-source label rates and
# source posteriors P(s|x) on a uniform sample of the pool (e.g. from a source classifier)
r = wsref.provenance_score(posteriors, label_rates)
print(wsref.incorpus_reference_closed_form(r, pooled_rate=y.mean()))
```

`audit.rq` is the SPARQL provenance audit (label rate per source from named graphs).

`test_wsref.py` runs unit tests; with `--icij` it also checks the paper's Table 5 readings.
