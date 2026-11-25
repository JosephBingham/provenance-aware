# E2 (OpenSanctions) analysis plan, fixed before any model was fitted (2026-11-24)

Data: OpenSanctions `default`, version 20261124065439-mxy (checksums verified against the published index).

Nodes: Person entities whose provenance includes exactly one of five sources S (Wikidata-derived datasets grouped as
"wikidata"): ext_ru_egrul, us_sam_exclusions, wikidata, gb_coh_disqualified, ru_acf_bribetakers. Chosen as the
person-recording datasets that do not assert the label and have >= 150 positives among single-source persons
(sanctions/export-control portals that do not emit the topic -- ua_war_sanctions, tw_shtc,
ext_us_ofac_press_releases -- are not eligible as sources).

Label: 1 if any dataset asserts topic `sanction` for the entity (statement-level provenance), else unlabelled.

Graph and features: only relationships and topics asserted by a dataset in S. Topics starting with `sanction` or
`export` are dropped. No dataset membership, names, countries or dates are used.

Representations: (i) structural: counts of each relation type and direction, distinct neighbours, neighbour-degree
mean/max, two-hop count; (ii) structural + context topics (multi-hot); (iii) RDF2Vec on the S-graph.
Learner and pipeline exactly as for ICIJ (gradient boosting 200 trees depth 3, 1:3 pseudo-negatives).

Confirmatory: leave-one-source-out for representation (i), all five sources; reading of the source-balanced score
under the two-sided rule with B = 199 balanced draws per source, Holm across the five sources, alpha = 0.05; pooled
score against its within-source reference (99 draws). Everything else exploratory.

## Deviation (recorded 2026-11-24, after two held-out results of representation (i) had been seen)
The extract tags `addressEntity` links with the merged entity's dataset list, not with the statement that asserted
the address, so an address contributed by a sanctions list could pass the S-filter (93% of sanctioned SAM persons
had one, against 4% of the others). Address links are dropped; all results were recomputed. The discarded run is
kept in results/discarded_addr_leak/. Relationship entities (Directorship, Ownership, Occupancy, Family, ...) carry
their own provenance and are unaffected.
Correction (same day): "93%" above is the mean number of address links per sanctioned SAM person; the share of
sanctioned SAM persons with at least one such link is 52%, against 3.5% of the other SAM persons (results/e2_audit.json).

## Implementation note
The diagnostic source classifier (used for source accuracy, Theorem 2 and the plug-in score) adds L2 regularisation
1.0: unregularised, the multiclass booster diverged on near-separable sources (accuracy 0.70 instead of 0.998). The
pipeline under test is unchanged.
