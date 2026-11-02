# Public well-log source selection

Checked 31 October 2026. Selection precedes external outcome calculations.

| Candidate | Fit to the protocol | Decision |
| --- | --- | --- |
| FORCE 2020 original LAS / organizer training CSV | Ordered measured depth, well IDs, raw GR and RHOB, traceable units and mixed lithology coverage | Selected organizer train.zip at pinned commit; no outcome-based well selection |
| SEG 2016 facies_vectors.csv | Depth and GR available; columns provide porosity-derived variables but not raw bulk-density RHOB needed for the fixed rule | Not used for this fixed-input test; not rejected for model performance |
| Well-log imputation benchmark processed slices | Relevant reproducible computing example, but processed 256-row slices require separate verification of physical depth and well continuity | Not used as an immediately interchangeable raw-depth dataset |

## Source and licence

Dataset: Bormann, P., Aursand, P., Dilib, F., Manral, S., and Dischington, P.
(2020), FORCE 2020 Well well log and lithofacies dataset for machine learning
competition. https://doi.org/10.5281/zenodo.4351156

Organizer repository: https://github.com/bolgebrygg/Force-2020-Machine-Learning-competition
Pinned commit: c8d01ee92c1c8e1ecba36f96cca6ea7b689338a1
Exact archive URL and SHA-256: qa/SOURCE_DOWNLOAD.json.

The original Norwegian government logs are stated by the dataset deposit to be
provided under NOLD 2.0 (the English licence is titled NLOD 2.0):
https://data.norge.no/nlod/en/2.0 . Attribution and identification of adaptations
are retained. No claim of organizer endorsement is made. Curated interpretations
are not re-licensed by this package. Raw archives and the third-party full text
are not redistributed in this software package; code downloads from the source
only when the documented downloader is explicitly enabled.

The cited record describes 118 wells. We do NOT claim that all 118 were analyzed:
the computational input is the complete organizer training ZIP, not the full
169.8-MB LAS archive. Acquisition identity is retained in qa/SOURCE_DOWNLOAD.json
within this workflow; complete historical records remain author-held.
Its actual well count and exclusions are computed from the CSV, not copied from
the older starter notebook (which mentions 83 wells) or the 118-well release.

## Units and interpretations

Organizer LAS header for 15/9-13 declares depth in m, GR in gAPI and RHOB in
g/cm3, with nominal step 0.152 m. Header-only evidence is saved locally; all
training-CSV well increments and curve ranges are audited separately.

The organizer geological summary, pp. 2--3 (PDF pages 3--4), states that
interpretations combine completion logs, mud logs, wireline curves and in-house
information. These are interpreted labels, not uniformly independent core truth.
They are not mapped to Liujia depositional classes or used to optimize this test.
https://www.sodir.no/globalassets/2-force/2020/seminars/contest-machine-learning/blog-post-litho-facies-competition-geological-summary.pdf

## Scope

This external application tests diagnostic portability on different logs and
physical sampling support. It does not test a frozen Liujia-trained predictor on
Norwegian geological truth. It does not validate six Liujia depositional classes
outside their original setting. Native-input classifications are reproducible
operational references, not independently verified lithology labels.
