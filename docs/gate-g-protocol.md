# Gate G protocol: finite-sample nuisance-remainder correction

## Question

Gate E2's known-mean controls validated the first-order EIF calculation, but
the feasible saturated-mean score had an additional finite-sample variance
term.  Gate G asks whether that term can be derived, estimated, and materially
reduced or correctly propagated without changing the frozen Gate E2 result.

All Gate G code, configurations, seeds, and outputs are new.  The frozen Gate
E2 source files and production artifacts remain immutable.

## Candidate mechanisms

Gate G separates two finite-sample effects.

1. **Cross-fitted mean propagation.**  Within a held-out fold, random
   assignment calibration error multiplies the saturated training-cell-mean
   error.  Although this is second order for the point estimator, observations
   scored with the same training fit are dependent.  The ordinary
   respondent-sandwich treats those scores as independent and omits this
   training influence.
2. **Residual-moment inflation.**  An out-of-fold residual equals its own
   outcome error minus an independently estimated cell-mean error.  Its raw
   second moment therefore exceeds the residual covariance by a
   design-and-count-dependent amount.  The inflation is much larger on the
   diagonal than off the diagonal when task-specific profile cells are used,
   attenuating the estimated within-respondent correlation.

The proposed corrections are:

- an exact linear-loading representation of a fixed-operator, cross-fitted
  saturated score, followed by conditional covariance propagation; and
- an elementwise count correction for the out-of-fold residual second moment,
  followed by the same positive-definite projection already used in Gate E2.
- a product-design finite-`G` operator obtained by adding the supported-cell
  mean-error covariance to the task covariance; and
- conditional saturated GLS as a finite-support benchmark that enforces
  calibration on the realized assignments.

The second correction is used only inside outer training samples.  It never
uses a held-out outcome to construct that outcome's score.

## Required theory and deterministic checks

Gate G must establish and verify all of the following.

1. For fixed fold-specific operators, express the complete cross-fitted point
   estimate exactly as a linear combination of respondent outcome vectors.
2. Show that the resulting conditional covariance includes the shared
   training-mean contribution omitted by the rowwise sandwich.
3. Derive the exact count-inflation matrix for saturated out-of-fold residual
   moments under a constant task covariance.
4. Verify algebraically that dividing by this matrix recovers the true
   covariance before numerical positive-definite projection.
5. Verify outcome leakage protections, task/profile relabeling invariance,
   full-support diagnostics, and reduction to the ordinary formulas when the
   mean is known.

Failure of any exact identity stops the computational screen.

## Fresh proof-of-concept screen

The screen uses four binary attributes, all 16 profiles, eight tasks, product
assignment, uniform task weights, and compound-symmetric residual covariance
with marginal variance one and correlation `0.80`.  It uses fresh random
streams and the existing dense task-specific mean surface.

Headline sample sizes are `G=250`, `500`, and `1000`.  The no-carryover cells
identify variance and coverage.  Paired `G=250` and `G=1000` local
least-favourable cells with radius ratio one verify that the correction also
repairs the AMSE comparison without erasing the Gate E1 local bias.

The screen compares:

- ordinary taskwise scoring;
- fixed oracle `H*` with the original rowwise sandwich;
- fixed oracle `H*` with exact count-aware mean propagation;
- the marginal-risk count-aware oracle operator (expected to change the point
  estimate only slightly);
- feasible compound-symmetry `H*` using the raw cross-fitted residual moment;
- feasible compound-symmetry `H*` using the count-corrected residual moment;
- the known-mean oracle control; and
- full-sample oracle GLS as a conditional finite-support benchmark, not as a
  general ML estimator.
- honest two-way feasible GLS, estimating the task covariance on one
  supported half and applying its weights to the other half, then reversing
  and averaging; both raw and exact residual-df covariance versions are
  retained to isolate the correction.

Same-data feasible GLS is reported only as a diagnostic. Its covariance
weights depend on the outcomes to which they are reapplied, so the
fixed-weight propagated variance is not labeled exact. Likewise, propagated
variances for learned cross-fitted operators are descriptive conditional on
the realized operators; exact variance and coverage gates apply only to
prespecified fixed operators.

Common data, assignments, folds, and random streams are used across methods.

## Prespecified proof-of-concept criteria

The screen advances to a larger production gate only if all hard criteria
hold.

1. **Exact fixed-operator variance:** in every no-carryover cell, empirical
   variance divided by the mean count-aware conditional variance lies in
   `[0.95, 1.05]` for oracle `H*`.
2. **Coverage:** count-aware oracle `H*` 95% interval coverage lies in
   `[0.93, 0.97]` in every no-carryover cell.
3. **Scaling repair:** for the `G=250/1000` pair, the corrected scaled-variance
   discrepancy is at most `0.10` of the first-order variance.  The original
   Gate E2 calculation is reported but cannot affect this new decision.
4. **AMSE repair:** in both local least-favourable cells, empirical MSE divided
   by count-aware predicted MSE lies in `[0.95, 1.05]` for fixed oracle `H*`
   and fixed blends used in the diagnostic.
5. **Covariance correction:** at `G=250` and `G=500`, the count-corrected
   covariance has at least 10% lower mean squared Frobenius error than the raw
   out-of-fold moment.
6. **Feasible safety:** the count-corrected feasible score has variance no more
   than 5% above the raw feasible score in any no-carryover cell and improves
   their geometric-mean variance ratio to the oracle bound.
7. **Numerics and leakage:** zero numerical failures, exact outer respondent
   isolation, and maximum calibration error below `1e-10`.

The proof-of-concept configuration uses 1,000 replications per headline cell.
These thresholds decide only whether a production gate is warranted; they do
not overwrite Gate E2 or establish publication readiness.

The simple finite-`G` operator minimizes the expected marginal-fold objective
under supported uniform cells. Overlapping training samples generate an
additional smaller cross-fold term. The exact propagated loading, rather than
the marginal approximation, is used for the variance and AMSE gates.

## Novelty rule

Exact influence propagation and degrees-of-freedom correction are not presumed
new.  Gate G passes as a paper-enabling feasibility result if it solves the
documented finite-sample failure while preserving the factorial EIF and
carryover conclusions.  A stand-alone novelty claim requires an additional
result specific to target-weighted repeated factorial geometry; accurate
finite-sample accounting alone is a supporting contribution.

## Pivot rule

If exact propagation works but count-corrected feasible scoring does not
improve or stabilize performance, retain the conditional variance formula for
diagnostics and inference, keep ordinary taskwise AIPW primary, and stop the
claim that feasible covariance adaptation is practically superior at moderate
cluster counts.  Do not tune thresholds or recycle Gate E2 random streams.

## Pre-run amendment log

On 2026-08-27, before any fresh Gate G replication was retained, the protocol
was amended in three ways after deterministic code audit:

1. the inherited scenario registry was made replacement-based rather than
   recursively merged with Gate E2;
2. YAML null-class labels were quoted; and
3. honest two-way feasible GLS and the marginal finite-`G` operator were added
   as diagnostic methods.

No pass threshold, seed, scenario, or replication count changed. Same-data
feasible GLS was explicitly downgraded to an exploratory benchmark rather
than given an exact finite-sample interpretation.

The first production launch was aborted after the first scenario reported 15
support failures. Diagnosis showed that a balanced outer training sample can
rarely contain an empty or singleton task-profile cell, making the auxiliary
inner out-of-fold residual fit impossible even though the frozen outer mean
learner has an explicit, linearly propagated empty-cell fallback. Before any
gate decision, the implementation was corrected to use the already-derived
exact saturated-projection degrees-of-freedom covariance correction whenever
inner splitting is impossible. If the diagnostic honest half-split itself is
unsupported, it falls back to the prespecified ordinary score. Both fallbacks
depend only on assignments, and their rates and all outer empty-cell events
are recorded. The incomplete launch
is retained as `gate-g-screen-aborted-support-bug`; the valid screen restarts
from the original data seeds, thresholds, and replication registry with new
assignment-only supported folds.
