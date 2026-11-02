# Liujia computation coverage

The S24 missing-dependency report is historical. Original training, calibration,
full-phase, diagnostic and matched-comparison source was recovered and used for
S26 acceptance. The candidate now bundles the selected main-result source.

| Stage | Generated results | RF fits | Inputs from this run |
| --- | --- | ---: | --- |
| class | Full-training OOF classes/probabilities, LOOWO and LOSBO | 13 | Compiled logs and fixed groups |
| calibration | Reduced-fit raw/calibrated probabilities, 13 fitted temperatures, thresholds, folds | 13 | Original calibration-well selection |
| phases | Table 5 / Figure 3 all-phase downsampling | 13 | New full-training native cache and coarse predictions |
| triage | Class-support, stability, budgets, point caches | 13 | New calibration records/probabilities |
| paired | Five states, channel/context and run-length strata | 0 | New triage caches and rules |
| matched | Same-row class/change comparison and point/block exports | 26 | New folds/triage; fixed 33 features and sampling |

Selected primary statistics are covered, not every historical branch or figure
renderer. Multiseed sensitivity, maximum-cardinality boundary analyses,
stratigraphic association and CAD/geology construction are not newly implemented
or rerun. Figure 10's underlying calibrated-class predictions/selection counts
are generated; its publication layout/illustrative-window exporter is outside
this entry.

Legacy modules retain ancillary functions for provenance. Their old main()
functions are not public entry points. build_analysis_data() belongs to the
separate geology/CAD branch. The documented wrapper reads existing authorized
measurements and groups; it performs no new clustering or geology construction.

A lazy import and metrics import detach unused CAD modules from this workflow.
Five metric functions are copied unchanged into retained_metrics.py and checked
against source ASTs. Other source edits are English comments/docstrings.
Preservation checks do not replace numerical replay; scientific evidence remains
S26, while the new orchestration is tested at interface level.
