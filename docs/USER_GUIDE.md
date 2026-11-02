# Definitions, inputs, and outputs

## Rule and paired states

Let `f = R(x)` be the native operational rule, `c = R(Tx)` the same rule on
reconstructed curves, and `p = M(Tx)` the comparator label. Labels 0 through 5
are anonymous operational states in external datasets, not validated facies.

| State | Definition |
| --- | --- |
| A | f = c = p |
| B | f = c, p differs |
| C | f differs from c, p = c |
| D | f differs from c, p = f |
| E | all three labels differ |

The rule-change count is C+D+E; comparator disagreement with native rule is
B+C+E. Their difference is B-D. This is an exact counting identity, not a causal
decomposition or a claim of scientific novelty. The tests enumerate all 216
triplets of six states, so even states absent from a particular trace are covered.

## Input and reconstruction contract

The selected functions use ordered, finite, continuous segments: GR in API,
bulk density in g/cm3, depth in metres. Do not concatenate wells or bridge QC
gaps. The depth-aware rule retains its physical support; the inherited row-index
rule assumes 0.05 m. Equivalence tests on that grid do not prove equivalence on
other spacings. Features are 26 processed GR/density statistics; actual support
spans are quantized to the available sample spacing.

Reconstruction retains every stride-th sample at the selected phase plus both
endpoints, then linearly interpolates onto the original grid. Strides 2, 4, and
5 have 2, 4, and 5 phases. Stride 1 is the zero-transformation control, not zero
metres of sampling interval. The example's depth is synthetic and relative.

## Ranking and budget

The target is `c != f`, not disagreement with independent geological truth.
The runnable example uses only processed curves to form the fraction of eight
nonzero joint shifts that change their rule state: GR shifts -1/0/+1 API and
density shifts -0.01/0/+0.01 g/cm3. These are analytical perturbations, not
instrument-error estimates. No target is supplied to score construction.

The KGS selector bins each segment from its own start into nominal 1 m blocks,
averages scores within blocks, sorts descending, and breaks ties with SHA-256
keys seeded by 20260907 and the well identifier. The final block is truncated
in original depth order to select exactly `floor(budget * N)` samples per well.
The example uses the same selector and physical-block constructor. The inherited
legacy demonstration instead uses 20-row blocks; at its 0.152 m grid these are not
1 m blocks. Both demonstrations are retained and clearly distinguished.

Capture = captured targets / all targets. Review yield = captured targets /
reviewed rows. With no targets, capture is undefined (`null` in JSON, empty in
the retained CSV), not a fabricated zero. With zero reviewed rows, yield is
undefined. A fixed sample budget does not assert equal human interpretation time.

The full KGS model workflow uses two matched 33-feature RF inputs: the 26
processed features plus six processed-rule one-hot indicators and one numeric
processed-rule state. The class-support score is `1 - P(native class = current
processed class)`, not `1 - max(P)`. The change learner estimates the operational
change target. Training targets and class weights differ even though sampled
training rows and explicit hyperparameters are shared. Neither RF is trained
in the synthetic example. Historical output keys such as `category_support`
are preserved for file compatibility; they denote class support.

## Outputs and interpretation

`examples/output_local/summary.json` records the three constant controls, a
stride-1 control, paired counts, feature spans, and budgets 0, 0.10, 0.13, 1.0.
The comparator rotates every 17th processed-rule label by one state. This is a
deliberate accounting fixture, not estimated model performance. Synthetic point
and block CSVs support independent readback, not scientific conclusions.

The separately held FORCE reference CSVs contain aggregate counts by public
well/phase and their summaries. KGS reference CSVs contain values by public well/stride:
49 tested wells, 48 with defined capture for each stride. Average phases within
well before equally averaging wells; do not pool all rows or turn missing
capture into zero. Positive and negative method differences are both retained.
The historical verifier rechecks means/signs/denominators, not RF predictions or
confidence intervals. Research tables are excluded from the public package;
use tools/verify_synthetic.py for the standalone generated example. See
EXCLUDED_MATERIALS.md before using the historical full readback CLI.
