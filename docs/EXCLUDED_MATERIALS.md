# Excluded materials and historical dependencies

This software-only package excludes the following baseline files. Originals
remain in the author-held baseline; neither their values nor permissions were
changed. Exclusion is not a claim that the original material was unlawfully held.

| Baseline path | Reason |
| --- | --- |
| reference_results/kgs/paired_differences.csv | Research output outside the MIT software grant |
| reference_results/kgs/summary_independent.csv | Research output outside the MIT software grant |
| reference_results/kgs/summary_original.csv | Research output outside the MIT software grant |
| workflows/force/results/force_native_port_disagreement.csv | Research output outside the MIT software grant |
| workflows/force/results/force_paired_counts.csv | Research output outside the MIT software grant |
| workflows/force/results/force_paired_summary.csv | Research output outside the MIT software grant |
| third_party/KGS_terms_of_use.html | External website snapshot not needed to run the software |

The two generated CSVs under examples/verified_synthetic/ are software test
fixtures, not research-result tables, and remain included with their JSON summary.
docs/RESULT_ENTRY_INPUT_MAP.csv is documentation, not measured research data.

## Which commands work without the excluded files?

- All 42 existing unittest cases use generated inputs/source witnesses and remain
  runnable without the excluded tables or website snapshot.
- examples/run_synthetic.py, examples/liuji_input_check.py and the FORCE legacy
  synthetic example need no research tables, private inputs or fitted models.
- tools/verify_synthetic.py exposes the existing independent verify_synthetic()
  function without calling the FORCE/KGS table-readback functions.
- The historical tools/verify_independent.py CLI unconditionally calls the
  FORCE/KGS readback functions. It needs separately held result tables and is not
  a self-contained public quick-start command. Its source is retained unchanged
  rather than weakening its checks or inventing data.
- FORCE/KGS raw-input generation and verification use a separately supplied
  workspace. Excluding saved reference tables does not remove their computation
  functions. Authorized data and documented full-run requirements still apply.

SOURCE_PROVENANCE.json deliberately retains historical identities of the excluded
files. It is a lineage record, not an assertion that those paths exist in this
release. CODE_PRESERVATION.json describes earlier source recovery; the current
byte-preservation result is in VALIDATION_SUMMARY.json. S26 scientific acceptance
and the prior license-only QA record remain historical, not rerun.
