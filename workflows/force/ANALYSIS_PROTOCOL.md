# FORCE external diagnostic protocol

Written on 31 October 2026 before external outcome calculations. This is an
exploratory external application, not a preregistered confirmatory study.
Historical execution notes remain author-held. Selected preserved source
witnesses are included in the package-level provenance directory.

## Selection and estimand

FORCE 2020 is preferred because original ordered depth, well identity, GR and
bulk density are available with traceable source and units. The 2016 SEG facies
table lacks raw bulk density, while sliced imputation benchmarks need not retain
the physical sampling support. These are reasons for exclusion, not performance
screening. Use the full accessible official release, not favorable selected wells.
If the full Zenodo LAS download fails, use the organizers' complete train.zip and
explicitly report its smaller scope. Retain all QC-eligible wells and segments.

The estimand is portability of the diagnostic workflow for an operational rule,
NOT accuracy of Liujia geological classes in Norway. External outputs of the fixed
six-state rule are anonymous operational states 0--5, never Norwegian facies.
The interpreted FORCE lithology is retained for coverage descriptions, not used
as training targets or silently mapped onto the six operational states. No claim
of independent core truth or cross-basin geological validity follows.

## Data and reference rules

1. Verify source archive checksum, schema, curve units and depth increments.
   Exclude nonfinite depth/GR/density, GR<=0, density<=0 or density>5. Never
   interpolate across rejected records, duplicate depth or gaps >1.5 native steps.
   Retain segments of at least 21 eligible rows for the main analysis; document
   excluded rows/wells and their effects on population coverage. No outlier winsorization.
2. Preserve literal GR thresholds 20/40/60/70/80/90 API and density thresholds
   1.75/2.30 g cm^-3. Primary adaptation preserves physical support: lithology-run
   thickness from native depth spacing, heterogeneity over [-0.50,+0.50) m,
   protection windows up/down to 0.25 m. Bounds clip at QC-segment ends.
   A row-index port retaining the original 0.05-m assumptions is a rule-only
   sensitivity, not the primary external method. Verify depth-based version
   agrees with the original function on a synthetic uniform 0.05-m grid.
3. Use native sampling and strides 2/4/5, all phases, both endpoints, linear
   reconstruction to native grid. Report both stride and actual physical spacing;
   do not compare Liujia and FORCE at a common metre spacing when they differ.
4. Keep f=R(x), c=R(Tx), p=M(Tx); A/B/C/D/E partition and B-D count identity.
   Do not claim the identity itself is a new algorithm or causal decomposition.

## Fixed comparator and grouping

Only RF, no hyperparameter search. Use the prior family of 26 GR/density features,
with physical endpoint spans 0.20/0.50/1.00 m and nearest feasible symmetric odd
windows on the native grid (minimum 3 samples); document actual spans. Five
deterministic whole-well folds assigned after seed-20261031 shuffling sorted names.
No well shared by training/test. Train on at most 2,000 evenly spaced eligible
rows per training well to bound compute and prevent long wells dominating.
Test every eligible row; no outcome-based row/well sampling. Fixed 300 trees,
min_samples_leaf=2, max_features=sqrt, class_weight=balanced_subsample,
unlimited depth, seed=20261031+fold. This is a newly fitted external comparator,
not the previously calibrated Liujia model; no recalibration is needed for label
argmax accounting. No location, well ID, depth or external interpretation features.

## Endpoints and summaries

- Five-state counts and rates per held-out well/phase; average phases within
  wells then give wells equal weight. Also retain pooled-row estimates.
- GR-only, density-only, joint changes and contextual-route contributions.
- Fine-rule run length / retained spacing in <=1, (1,2], (2,4], >4 bins.
- Literal threshold distance in <=1, (1,3], >3 bins using 1 API and 0.01 g/cm3
  as analytical scales, not sensor uncertainties. No tuning to FORCE results.
- Compare signs/patterns with the Liujia diagnostics at matched strides.
- Cluster bootstrap whole-well phase means (2,000 draws, seed 20261031) for
  mean-rate intervals; the estimand is sampled wells, not new basins. No row-level
  resampling, significance screening or selection of favorable phases.
- Quantify disagreement between physical-support and row-index rule ports.
  This is an implementation transport diagnostic, not a geological label error.

## Verification and delivery

Check source hash, duplicates/gaps/units, disjoint folds, zero-stride control,
finite features, class alignment, all partitions, count identities, two independent
summary implementations, and source fingerprints. Export per-well results,
source manifest, split list and analysis records. Keep raw downloads separate;
the documented downloader identifies the frozen input without bundling
third-party full texts or private Liujia inputs. The original execution/delivery
instructions remain in the author-held historical archive rather than serving
as public user steps.
