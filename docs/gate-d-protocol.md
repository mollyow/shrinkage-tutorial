# Gate D protocol: feasible covariance-adaptive scoring

## Question

Gate D asks whether covariance-aware repeated-factorial scoring remains useful
after its residual covariance and working score are learned from data.  Gate C
showed only an oracle opportunity.  Gate D must demonstrate a feasible,
cross-fitted estimator with valid cluster inference and limited downside when
the opportunity is absent.

The candidate-recruitment application is supporting evidence, not a gatekeeper.
It may be replaced by an application with a better-documented joint randomizer
without changing this protocol.

## Frozen estimator

Every outer fold contains complete respondents.  All nuisance estimation and
score selection for that fold use only the other respondents.

1. Fit a task-specific additive outcome regression on the outer training data.
2. Within the outer training data, construct respondent-level inner-cross-fitted
   residual vectors.
3. Estimate compound-symmetric and unstructured residual second moments.  Use
   squared cluster weights when survey weights are present.
4. Convert each working covariance into an exactly calibrated factorial score.
5. Form convex blends with ordinary taskwise scoring at
   \(\tau\in\{0.25,0.50,0.75,1\}\).  Every blend remains exactly calibrated.
6. Include ordinary scoring (\(\tau=0\)) as a candidate.  Select with a
   respondent-level one-standard-error rule using inner-fold scores, preferring
   smaller \(\tau\) and then compound symmetry.
7. Refit the selected working covariance on all outer-training cross-fitted
   residuals and score the held-out respondents.
8. Average held-out scores and use their respondent-cluster sandwich covariance.

The selected nuisance may be wrong without invalidating the mean, provided the
mean no-carryover restriction and the assignment transition law used for
calibration are correct.  Covariance misspecification affects efficiency, not
the exact calibration identity.

## Frozen simulation

The production configuration is
`analysis/feasible_eif/gate-d-production.yml`.  It fixes 2,000 replications in
each of nine scenarios before the production results are examined.  Four
centered factorial coordinates are estimated on a complete 16-profile binary
factorial.  The scenarios include:

- moderate and high compound-symmetric correlation;
- nonexchangeable AR(1) correlation;
- product and uniform-without-replacement assignment;
- a smaller-sample strong-correlation cell;
- zero, weak-positive, and negative correlation controls; and
- profile-dependent heteroskedastic covariance.

The mean learner is intentionally not saturated: the truth contains interactions
while the learner contains task-specific main effects.  Exact calibration—not
correct outcome-model specification—must protect bias and coverage.

## Decision rule

Every strong-headroom cell must have selected/ordinary Monte Carlo variance at
most 0.90, with bootstrap upper endpoint below 0.98, and recover at least half
of the true constant-covariance oracle gain.  Weak-headroom cells must have an
aggregate variance ratio at most 1.05, and no safety or covariance-robustness
cell may exceed 1.10 at the point estimate.

For every mandatory no-carryover scenario, pooled coordinate coverage for both
ordinary and selected 95% intervals must lie between 93% and 97%.  RMS
standardized bias must be at most 0.10 and mean sandwich variance divided by
Monte Carlo variance must lie between 0.90 and 1.10.  Numerical failures are not
dropped: any positive failure rate fails the frozen gate.

Separate algebra tests require calibration residuals and coding-invariance
errors below \(10^{-10}\) under level permutations, reference-baseline changes,
and orthogonal rotations.

## Interpretation

Passing Gate D would justify developing the feasible estimator and its
large-sample theory.  It would not establish mean no-carryover, validate a
particular application's assignment logs, or justify inference after selecting
interactions.  Those remain Gate E and application-design questions.
