# Rule-derived log interpretation: diagnostics and review prioritization

**MIT software | offline synthetic examples | v1.0.0**

This package pairs native rule output, resampled rule output, and model output
at the same depth to distinguish rule changes from additional model discrepancy.
Review experiments compare priority scores against a defined change target and
fixed budget. Operational rule labels are not independent geological ground truth.

Software accompanying the manuscript prepared for Computers & Geosciences:
[yqma7980/rule-derived-log-review](https://github.com/yqma7980/rule-derived-log-review).
Fixed software version:
[v1.0.0](https://github.com/yqma7980/rule-derived-log-review/tree/v1.0.0)
(`refs/tags/v1.0.0`). See [release metadata](release_metadata.json).
The manuscript is not represented as accepted or published by the journal.

## Permanent software archive

The unchanged v1.0.0 software snapshot is archived at [Zenodo DOI 10.5281/zenodo.23949723](https://doi.org/10.5281/zenodo.23949723). The archive preserves the fixed tag and commit
`aebadf57c0380a75d3b47dc8a4ce36e5d052f9a7`. Its two downloadable files were
verified anonymously against the release ZIP and SHA-256 checksum. This DOI
identifies software and documentation, not controlled Liujia measurements or
a manuscript preprint. Subsequent README/manifest updates do not alter the
immutable v1.0.0 tag.

## Install and run offline examples

Start with Python **3.12.10**. From this package directory, on Windows:

```text
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m pip check
.venv/Scripts/python -B -X utf8 workflows/liuji/run.py check-env --environment main
.venv/Scripts/python -B -X utf8 -m unittest discover -s tests -v
.venv/Scripts/python -B -X utf8 examples/run_synthetic.py --output examples/my_run
.venv/Scripts/python -B -X utf8 tools/verify_synthetic.py --synthetic-output examples/my_run
.venv/Scripts/python -B -X utf8 examples/liuji_input_check.py --output "../liuji synthetic input"
```

On POSIX, use `.venv/bin/python` instead of `.venv/Scripts/python`. Windows is
the tested platform for these interface checks; POSIX exact replay is not certified.
Installation may require network access; subsequent tests/examples do not.
No models are fitted by these commands. Do not use `python -O`, because source
assertions are part of verification. Use a fresh output directory for each run.

The pairing example writes `summary.json`, `synthetic_points.csv` and
`synthetic_blocks.csv`: 200 constructed rows, 26 features, four budgets, three
constant controls and a stride-1 control. It is not a field-performance result.
To verify the bundled fixture without generating a new one:

```text
python -B -X utf8 tools/verify_synthetic.py
python -B -X utf8 workflows/force/code/example_synthetic.py
```

The Liujia input example generates 240 rows in three synthetic wells, groups,
a local config and a QC/feature summary. It intentionally does not meet the
paper-population or matched-environment gate. Its local config contains runtime
paths and should not be committed. The legacy FORCE demonstration uses 20-row
blocks; the primary pairing example uses nominal 1 m blocks.

## Research workflows

| Workflow | Required inputs | Entry | Environment |
| --- | --- | --- | --- |
| Liujia selected main results | Authorized compiled logs and fixed well groups | `workflows/liuji/run.py` | Main for class/calibration/phases/triage/paired; matched for class/change |
| FORCE sampling/context diagnostics | Pinned organizer training archive | `workflows/force/code/run_external.py` | FORCE record |
| KGS external review evaluation | Pinned KGS archive and prepared CSV | `workflows/kgs/code/external_review_trial.py` | KGS record |

Read the [Liujia instructions](docs/LIUJI_RUN.md),
[external workflows](docs/PUBLIC_WORKFLOWS.md), [environments](docs/ENVIRONMENTS.md),
[input schema](docs/INPUT_SCHEMA.md) and
[paper-result mapping](docs/RESULT_ENTRY_INPUT_MAP.csv).
Full computation is separate from examples and requires authorized inputs and
explicit training commands. The Liujia wrapper generates new predictions and
temperatures without author folders, historical prediction caches or fitted models.
`tools/compare_arrays.py` is an optional read-only NPZ comparator, not a dependency.

## Validation and scope

The existing 42 interface/unit tests and fresh synthetic examples are checked
for this public package; machine-readable results are in `VALIDATION_SUMMARY.json`.
Fresh installation and a new full scientific replay are separate checks.
Historical S26 acceptance regenerated selected Liujia main results: 78 RF fits,
13 temperature fits, 914 checks without failures and 394 saved arrays without
differing elements. This is inherited evidence, not new training in this release.
See [validation scope](docs/VALIDATION_SCOPE.md) and [coverage](docs/LIUJI_COVERAGE.md).

All 42 baseline Python files are preserved byte-for-byte. The new
`tools/verify_synthetic.py` only exposes the existing independent synthetic
readback function; it does not alter the scientific implementation.
`SOURCE_PROVENANCE.json`, `CODE_PRESERVATION.json` and
`provenance/license_revision_qa.json` retain explicitly historical evidence.
`PACKAGE_MANIFEST.json` identifies current files, hashes and license scope.

## Distribution and license

Included: source, software documentation, environment specifications, tests,
configuration and clearly labelled generated test fixtures. Author-owned software
is licensed under [MIT](LICENSE), including documentation and synthetic test
fixtures. Citation of the associated work is appreciated, not an MIT condition.

Excluded: private logs, coordinates, fitted models, per-depth research predictions,
research-result tables, third-party data archives, author correspondence and private
approval records. Dependencies are installed separately, not vendored.
See [license scope](NOTICE.md), [third-party notices](THIRD_PARTY_NOTICES.md),
[excluded materials](docs/EXCLUDED_MATERIALS.md) and
[release checklist](docs/RELEASE_CHECKLIST.md).

The unchanged historical `tools/verify_independent.py` CLI also requires withheld
FORCE/KGS result tables and is **not** the standalone public quick-start verifier.
Use `tools/verify_synthetic.py` for the data-free example. Research tables have not
been replaced with synthetic values or relicensed as MIT.
