# Public-data workflows and external inputs

Run commands from the candidate root using the installed environment's Python
(written as `python` below). Quote paths containing spaces. Use disposable
workspaces outside the candidate. Input inspection performs QC only. Commands
marked **full run** fit models and can take substantial time.
Source links are dataset locators, not software-publication claims.

## FORCE

Dataset: https://doi.org/10.5281/zenodo.4351156
Organizer: https://github.com/bolgebrygg/Force-2020-Machine-Learning-competition
Pinned commit: `c8d01ee92c1c8e1ecba36f96cca6ea7b689338a1`.
The organizer `train.zip` has 91,653,972 bytes and SHA-256
`e996c8466f2d0dc7116daa71485cc90c31fcfac6345a5bb33165026478b642e9`.

Create a workspace and either place that archive at
`data/external/FORCE_train.zip` inside it, or explicitly enable the downloader:

```text
python workflows/force/code/prepare_workspace.py --workspace "../force input check"
python -m pip install -r requirements-force-download.txt
python workflows/force/code/acquire_force.py --workspace "../force input check" --download
python workflows/force/code/run_external.py --inspect --workspace "../force input check"
```

The optional download requirements are needed only for network acquisition.
`--help` and missing `--download` never download data. The downloader checks ZIP
integrity and the pinned hash. Manually supplied input is not hash-enforced by
the analysis loader: compare it to the pinned hash before claiming paper replay.

The ZIP must contain one semicolon-delimited CSV with `WELL`, `DEPTH_MD`, `GR`,
`RHOB`, and `FORCE_2020_LITHOFACIES_LITHOLOGY`. Units are metres, API, g/cm3.
The interpretation column is retained for coverage, not used as a training target.
QC excludes nonfinite/invalid inputs and duplicate well-depth keys, breaks
context across removed rows/gaps, and retains segments of at least 21 rows.
Inspection writes population/coverage tables and `qa/DATA_INVENTORY.json`.

**Full run:** use a different fresh workspace with the same archive. Existing
results/cache/QA outputs are refused, except the downloader's source record.

```text
python workflows/force/code/run_external.py --run --workspace "../force full run"
python workflows/force/code/verify_external.py --workspace "../force full run"
```

The workflow writes folds, five trained RF models, scenario prediction caches,
paired/channel/port/strata tables and `qa/EXTERNAL_ANALYSIS_QA.json`. It uses five
whole-well folds, 300-tree RFs, eight jobs and 12 scenarios including the native
control. Retain caches until verification. The verifier recalculates accounting
and checks a scalar rule oracle; it does not fit an independent RF. Its optional
historical-baseline preservation check is unavailable in this portable layout.
The historical reference aggregates describe 98 wells and 1,009,086 eligible rows
from 1,170,511 raw rows, not all 118 wells in the broader FORCE release.
Those research-result tables are excluded from the public software package.

## KGS archive preparation

Index: https://www.kgs.ku.edu/PRS/Scans/Log_Summary/index.html
Archive: https://www.kgs.ku.edu/PRS/Scans/Log_Summary/2001_2005.zip
Terms: https://www.kgs.ku.edu/General/copyright.html

The source of this material is the Kansas Geological Survey website at
http://www.kgs.ku.edu/. All Rights Reserved.

Acquire the archive directly, then use explicit input and output paths:

```text
python workflows/kgs/source_tools/scan_kgs_archive.py --archive "../kgs inputs/2001_2005.zip" --output-dir "../kgs prepared"
python workflows/kgs/source_tools/prepare_kgs.py --archive "../kgs inputs/2001_2005.zip" --inventory "../kgs prepared/gr_density_inventory.json" --output-dir "../kgs prepared"
```

Both entry points check the frozen archive hash
`325059152a76c89fb1a798390fe955f6256e07e00e9a40305691122bcc940edd`
(44,727,461 bytes). Existing scanner/prepared outputs are not overwritten.
Use the inventory produced from that same archive; manual inventory edits are
not a supported paper-replay input.

The scanner reads nested LAS headers. Preparation requires unambiguous GR
API/GAPI, RHOB g/cm3, and depth F/FT; feet are converted by 0.3048. Missing units
are not inferred. Duplicate APIs are selected by most jointly valid rows then
LAS filename. Output: `kgs_selected_wells.csv` (`well,depth,gr,den`),
`input_manifest.json`, `input_summary.json`. Generated manifests can include
locations and raw headers: do not redistribute them without review.
Archive years are receipt years, not common logging dates or geological ages.

The retained reference input has 49 wells, 190,210 rows, 130,708 jointly valid
rows before segmentation, and 130,703 eligible rows. Normalized CSV SHA-256:
`3da3eebada775a3741505069fe369c750878af32e12d6e94b40b221a491d961f`.

## KGS-style CSV inspection and full trial

The existing external-data entry accepts another authorized CSV. Required
columns default to `well,depth,GR,DEN`; use explicit aliases for lower-case
KGS columns. Declare depth-to-metres and density-to-g/cm3 factors when needed;
units cannot be inferred from numbers. Keep original row gaps and well IDs.
The inherited loader infers identifier types; normalize numeric-only IDs
consistently and do not mix IDs differing only by leading zeros.

```text
python workflows/kgs/code/external_review_trial.py --input "../kgs prepared/kgs_selected_wells.csv" --out "../kgs input check" --gr-column gr --den-column den --max-wells 0 --inspect-only
```

Inspection writes `PROTOCOL.json`, `population.csv`, `well_selection.csv` and
`inventory.json`, but no cache or models. Invalid/duplicate rows and irregular
spacing break segments; fewer than 21 consecutive rows are excluded. Nonempty
directories are always refused, even if an earlier run was incomplete.

**Full run:** choose another fresh directory:

```text
python workflows/kgs/code/external_review_trial.py --input "../kgs prepared/kgs_selected_wells.csv" --out "../kgs full run" --gr-column gr --den-column den --dataset-name KGS_2001_2005 --max-wells 0 --train-cap-per-well-total 2000 --folds 5 --seed 42 --fold-seed 20261101 --n-jobs 4
python workflows/kgs/code/verify_external_trial.py --out "../kgs full run"
```

The full trial fits matched 33-feature class/change forests across 11 scenarios,
five whole-well folds and ten 100-tree forests, with at most 2,000 sampled rows
per training well across scenarios. It writes folds, caches, training selections,
capture/AUC summaries and `COMPLETE.json`. At least three eligible wells and both
binary training states are required. This is external retraining, not transfer of
a frozen Liujia model or a replacement for Liujia LOOWO/LOSBO/calibration.

The inherited independent verifier supports normalized KGS columns/units only,
not arbitrary aliases or conversion factors. It checks QC/targets/budgets/folds,
not RF point probabilities or a second RF fit. The analysis stores the resolved
input path so a changed working directory does not silently change the input.
Capture means distinguish 49 tested wells from 48 assessable wells.

Full raw-data replay, model fitting, confidence-interval recomputation and
performance claims require their own verification. See `VALIDATION_SCOPE.md`.
