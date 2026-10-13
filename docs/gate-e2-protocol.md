# Gate E2 protocol: computational carryover sensitivity

**Status: version 3 frozen after preproduction and completion audit on
2026-08-27.**

## Preproduction amendment record

Version 3 was frozen before the first valid production run. A correctness audit
invalidated earlier partial output and made four implementation-level changes:
leave-two-fold covariance fitting for inner selector scores; outer folds fixed
independently of assignments; a recorded, support-safe empty-cell rule; and the
first-order cell-mean contribution in saturated-rescue variances. No scenario,
scientific threshold, or result-dependent tuning rule was changed. Before any
valid production output, the replication count was raised from 1,000 to 3,000
per cell because the simultaneous raw cellwise 10% AMSE tolerances were too
Monte-Carlo-unstable at 1,000 draws. Partial files produced before these
corrections are non-evidentiary and cannot be combined with version-3 output.

The first version-2 production attempt was stopped after its small-sample cell
reported two numerical failures and before a gate summary was computed. Both
were assignment-only support events: an entire task-profile cell was absent
from an outer training sample. Version 3 makes the inner anchor set cover every
cell observed in that outer training sample and applies the already-prespecified
task-mean fallback to cells absent from it. The two failing replication seeds
are regression tests. No outcome result or gate metric informed this change.

The production bundle records MD5 hashes for the configuration and every
solver, estimator, simulation, sensitivity, runner, and summarizer source file.
The decision script fails the integrity gate unless those hashes match the
current version-2 sources exactly.

## Question and decision structure

Gate E2 asks two separate questions.

1. **Hard correctness:** Do the feasible scores obey the exact Gate E1 bias
   identities, local limit, assignment-error decomposition, and respondent-level
   inference guarantees in computation?
2. **Practical value:** Once those identities hold, do feasible bias-aware
   intervals and sensitivity-indexed blends provide useful protection rather
   than merely restating an oracle calculation?

The two decisions must not be collapsed. A hard-correctness failure invalidates
the corresponding theoretical or implementation claim. A practical-value
failure can leave the identities intact, but it prevents recommending the
adaptive procedure as a useful applied method. Ordinary taskwise AIPW and its
respondent-cluster interval remain the primary unrestricted analysis regardless
of either decision.

The machine-readable production design is
`analysis/feasible_eif/gate-e2-production.yml`. No scenario, threshold, method,
or replication count may be changed after production outcomes are inspected.

## Frozen estimand and reference design

The experiment reuses Gate D's complete 16-profile binary factorial and its four
centered target coordinates. The primary scalar direction is fixed as

\[
a=e_1,
\]

the `x1_x2` coordinate. Results for the other coordinates and the joint
four-coordinate family are secondary and cannot rescue a failed primary
direction.

There are eight tasks per respondent. Unless a scenario says otherwise, the
true and analyzed assignment laws are independent uniform product assignment,
and the residual covariance is compound symmetric with marginal variance one
and correlation 0.8. The target distribution is the uniform complete
factorial. In every correct-law cell, both the score regression and the
regression used to form covariance-learning residuals are task-specific
saturated 16-profile cell means. Thus the no-carryover reference nuisance is
consistent, and the E1 efficiency and sensitivity-operator calculations are
not confounded with deliberate own-task mean misspecification.

The assignment-law mismatch cells are the sole exception. Their primary
wrong-law score deliberately uses the Gate D task-specific additive regression,
while covariance residuals continue to use task-specific saturated cell means.
Those cells also include saturated-mean and oracle-mean score comparators. This
isolates the assignment-calibration error and tests the two outcome-regression
rescue cases without changing the covariance target.

For every generated law, the simulation truth must be recomputed as

\[
\theta_{U,G}=E_q\{H_0(\mathbf A)^\mathsf Tm_G(\mathbf A)\}.
\]

It must not be copied from the no-carryover coefficient vector. This is
essential because a least-favourable carryover direction can also move the
unrestricted task-marginal target.

All folds contain complete respondents. Every outcome regression, residual
covariance, selector decision, feasible sensitivity operator, and generic
blend is trained without the respondent whose score it is used to construct.
Outer folds are a prespecified function only of respondent IDs and the random
seed; neither assignments nor outcomes can alter them. Each resulting training
sample uses exact empirical means for observed task-by-profile cells. On the
rare event that a cell is empty, its prespecified prediction is the corresponding
task's overall training mean. Empty-cell events are recorded, and folds are
never rerandomized in response to held-out assignments. Under fixed positive
overlap this fallback event vanishes exponentially and does not alter the
regular first-order limit.

Within an outer training sample, an assignment-only greedy set cover designates
a small set of support anchors for every task-profile cell observed in that
sample. Anchors enter every inner training fit and are excluded from inner
validation and selector-loss calculation. Thus every observed outer-training
cell remains represented even when it occurs only once; a cell absent from the
outer training sample uses the same prespecified task-mean fallback in every
inner fit. No respondent outcome is used to construct its own validation score,
and all respondents remain eligible for the untouched outer-fold evaluation.

For inner selector fold \(j\), covariance residuals on inner fold \(k\) are
fit after excluding both \(j\) and \(k\) (support anchors remain training-only).
Thus neither the mean nuisance nor the covariance operator used to score fold
\(j\) can depend on fold \(j\)'s outcomes. This leave-two-fold construction is
used only for honest selector and generic-blend training scores; the final
outer-training covariance may use all inner out-of-fold residuals because the
outer evaluation respondents remain untouched.

The saturated-mean assignment-law rescue is not Neyman-orthogonal when its
score operator is calibrated under the wrong assignment law. Its point
estimate therefore uses the stated wrong-law operator and saturated cell-mean
nuisance, but its sandwich variance includes the first-order influence of
estimating those cell means. If \(H\) is the wrong-law task-mixing matrix,
\(R_q\) is the true task-assignment correlation, and \(w\) contains the task
weights, the inference-only mixing matrix is
\(H+\operatorname{diag}\{w-\operatorname{diag}(H R_q)\}\). This correction is
used only for the rescue diagnostic; it assumes the true randomized assignment
law is known.

## Frozen carryover construction

At the scenario's no-carryover reference law, define

\[
D^*=H^*-H_0,\qquad
\Delta=\Omega_0-\Omega_*,\qquad
M=\Sigma^{-1},\qquad
K=T\Delta.
\]

The unit least-favourable departure in the primary direction is

\[
u_{\mathrm{LF}}(\mathbf A)
=\frac{T\Sigma D^*(\mathbf A)a}{\sqrt{a^\mathsf TKa}},
\qquad \varepsilon_M(u_{\mathrm{LF}})=1.
\]

A local cell with radius multiplier \(r\) uses

\[
m_G=B\gamma+\frac{r}{\sqrt{TG}}u,
\]

so the local carryover radius is \(r/\sqrt T\). For a least-favourable
departure, the signed normal Hausman statistic has mean shift \(r\) (and its
one-degree-of-freedom chi-squared version has noncentrality parameter \(r^2\)).
The restricted and unrestricted estimators break even in worst-case local MSE
at \(r=1\).

The lag-only departure uses the centered previous-profile `x1_x2` factorial
score for tasks 2 through 8, is projected off the no-carryover mean space in
the stated \(M\) metric, and is normalized to unit \(\varepsilon_M\). The blind
departure uses the same lag map with the unused `x3_x4` factorial score. That
feature is exactly orthogonal to the complete reported four-coordinate family,
so it lies in the kernel of the map \(c\mapsto E(D^{*\mathsf T}c)\). It is also
normalized to unit \(\varepsilon_M\). The implementation must verify that its
norm is positive and its vector bias is below \(10^{-10}\) before simulation.

The fixed-departure cell uses
\(m=B\gamma+0.10u_{\mathrm{LF}}\), rather than a shrinking triangular array.

For assignment-law error cells, let \(C_q=E_q(H^\mathsf TB)\), where \(H\) is
constructed under the analyzed law but expectation is under the true law. The
omitted no-carryover mean component is chosen before simulation from the
interaction-only space in the leading primary-direction image of
\(L-C_q\), orthogonal to the additive learner. Its scale is fixed to give a
directional assignment-error noncentrality of 1.5. This deterministic
construction must be nonzero; otherwise the configuration is invalid rather
than an invitation to choose a more favourable cell.

## Frozen scenario grid

Production uses exactly 3,000 replications in each cell. The smoke run uses 30.

| ID | \(G\) | True/analyzed assignment | Residual covariance | Carryover |
|---|---:|---|---|---|
| `null_cs_product` | 500 | product/product | CS, 0.8 | none |
| `lf_r0_5_cs_product` | 500 | product/product | CS, 0.8 | least favourable, \(r=0.5\) |
| `lf_r1_cs_product_g250` | 250 | product/product | CS, 0.8 | least favourable, \(r=1\) |
| `lf_r1_cs_product_g1000` | 1,000 | product/product | CS, 0.8 | least favourable, \(r=1\) |
| `lf_r2_cs_product` | 500 | product/product | CS, 0.8 | least favourable, \(r=2\) |
| `lf_r3_cs_product` | 500 | product/product | CS, 0.8 | least favourable, \(r=3\) |
| `lag_r1_cs_product` | 500 | product/product | CS, 0.8 | lag only, \(r=1\) |
| `lag_r1_ar1_product` | 500 | product/product | AR(1), 0.7 | lag only, \(r=1\) |
| `blind_r2_cs_product` | 500 | product/product | CS, 0.8 | blind, \(r=2\) |
| `fixed_lf_e0_10_cs_product` | 500 | product/product | CS, 0.8 | least favourable, \(\varepsilon_M=0.10\) |
| `lf_r1_cs_wor_correct` | 500 | without replacement/same | CS, 0.8 | least favourable, \(r=1\) |
| `qerr_product_as_wor_null` | 500 | product/without replacement | CS, 0.8 | none |
| `qerr_wor_as_product_null` | 500 | without replacement/product | CS, 0.8 | none |
| `qerr_wor_as_product_lag_r1` | 500 | without replacement/product | CS, 0.8 | lag only, \(r=1\) |

Uniform without-replacement assignment samples eight distinct profiles in task
order from the 16-profile support.

## Frozen methods

Every applicable method uses the same generated data and outer folds.

1. `ordinary_taskwise`: feasible taskwise AIPW with task-specific saturated
   16-profile cell means in every correct-law cell. This is the confirmatory
   estimator.
2. `oracle_unrestricted_joint`: unrestricted EIF scoring with the true complete
   conditional mean. This is an unattainable efficiency benchmark, not a
   proposed estimator.
3. `selected_adaptive`: the frozen Gate D one-standard-error covariance and
   blend selector, trained only outside each evaluation fold.
4. `oracle_adaptive`: the exact \(H^*\) under the true residual covariance and
   analyzed assignment law.
5. `oracle_tau_0.25`, `oracle_tau_0.50`, `oracle_tau_0.75`, and
   `oracle_tau_1.00`: fixed blends of \(H_0\) and oracle \(H^*\).
6. `oracle_tau_mm`: the sensitivity-indexed oracle blend. In local
   least-favourable cells it uses \(1/(1+r^2)\); in general it uses the stated
   radius and the appropriate Gate E1 quadratic form.
7. `feasible_tau_generic`: the directional clipped solution in (E1.24b), with
   \(C_{0D}\), \(V_D\), and \(K\) learned using outer-training respondents only.
8. Assignment-error cells additionally fit correct-law and wrong-law selected
   scores. The primary wrong-law score uses the deliberately misspecified
   additive regression; two rescue comparators replace it with, respectively,
   cross-fitted task-specific saturated cell means and the oracle-correct
   own-task mean. The oracle method is a diagnostic, not an applied estimator.

The Hausman statistic is never used to choose which estimate or interval is
reported.

## Quantities recorded in every replication

For each method and outer fold, the implementation computes the operator,
respondent scores, sandwich covariance, and estimated sensitivity operator in
memory. The persistent per-replication artifacts retain the resulting target
truth, estimate, standard error, standard and bias-aware interval indicators
and widths, exact bias and variance diagnostics, calibration residual, selected
library member, blend, sensitivity radius, and support diagnostics. Raw
respondent scores and full operator matrices are deliberately not persisted in
the production bundle; their algebra and provenance are covered by the frozen
unit tests and source hashes.

It must also compute the exact fold-conditional bias prediction. Under the
correct assignment law this is

\[
b_f=E_q(D_f^\mathsf Tm_G).
\]

Under an incorrect analyzed law it is

\[
b_f=(L-C_{q,f})(\eta_f-\gamma)
    +E_q(H_f^\mathsf T\delta).
\]

Fold predictions are averaged using held-out fold sizes. Expectations must use
cached exact assignment moments of the required profile pairs or triples, not
the realized outcomes or assignments. This prevents a noisy empirical bias
estimate from being relabeled an exact prediction.

Bias-aware scalar intervals widen the conventional cluster interval by

\[
\varepsilon_M\sqrt{a^\mathsf TKa}.
\]

Both oracle-\(K\) and feasible direct-\(K\) versions are recorded. The lag cells
also record the unrestricted-ball interval and the prespecified lag-subspace
interval based on \(K_Z\).

## Hard-correctness decision

Every applicable cell must pass separately. No average over favourable cells
can offset a failure.

1. **Completion and calibration.** All planned replications finish; none are
   dropped. Maximum calibration error is at most \(10^{-10}\).
2. **Exact bias.** For ordinary scoring, selected scoring, oracle \(H^*\), all
   applicable blends, and both wrong-law directions, the Monte Carlo mean of
   \((\widehat\theta-\theta_U)-b_{\mathrm{pred}}\) in the primary direction is
   at most 0.10 Monte Carlo standard deviations in magnitude and has absolute
   paired Monte Carlo \(z\)-statistic at most 3.5.
3. **Local AMSE.** In every least-favourable local cell, empirical
   \(G\)-scaled MSE for \(H_0\), \(H^*\), and each fixed blend divided by its
   exact Gate E1 AMSE lies in \([0.90,1.10]\). Between the two \(r=1\)
   product cells, scaled bias and \(G\)-scaled variance may differ by at most
   15% after accounting for their common theoretical value.
4. **Break-even ordering.** The primary-direction oracle-adaptive/ordinary MSE
   ratio is below 0.90 at \(r=0.5\), lies in \([0.90,1.10]\) at every
   \(r=1\) least-favourable cell, and exceeds 1.10 at \(r=2\) and \(r=3\).
5. **Unrestricted inference.** Ordinary primary-direction 95% coverage lies in
   \([0.925,0.975]\), absolute standardized bias is at most 0.10, and mean
   sandwich variance divided by Monte Carlo variance lies in \([0.90,1.10]\)
   in every scenario. Selected intervals meet the same requirements in the
   null and blind cells.
6. **Oracle bias-aware inference.** The oracle-\(K\) bias-aware interval covers
   at least 93% in every correct-law carryover cell.
7. **Hausman calibration.** The directional oracle test rejects between 3% and
   7% under the no-carryover and blind laws. In local carryover cells, rejection
   is within 0.04 of the noncentral-normal prediction. At \(r=1\), power must
   remain below 0.25; at \(r=2\), it must exceed 0.40. The selected-score test
   is within 0.06 of its fold-specific predicted power. The four-dimensional
   omnibus test is descriptive.
8. **Assignment-law decomposition.** Both wrong-law directions and the combined
   wrong-law-plus-lag cell satisfy the exact-bias rule. In the no-carryover
   cells, both saturated-mean and oracle-correct-mean rescue scores have absolute
   standardized bias at most 0.10 and 95% coverage in \([0.925,0.975]\). In the
   combined cell, the oracle-mean score's remaining bias must equal the direct
   carryover term under the same exact-bias tolerance; the saturated score must
   satisfy the full fold-specific decomposition.

## Practical-value decision

Practical value is evaluated only if hard correctness passes. Again, every
listed cell is mandatory.

1. The feasible direct-\(K\) bias-aware selected interval covers at least 93%
   in every correct-law carryover cell.
2. In both lag cells, the structured interval covers at least 93%, and its
   median length does not exceed the unrestricted-ball interval's median
   length.
3. In each product least-favourable local cell, the paired-bootstrap upper
   97.5% endpoint for
   `oracle_tau_mm` AMSE divided by the smaller of ordinary and oracle-adaptive
   AMSE is below 0.98.
4. `feasible_tau_generic` has point AMSE no more than 1.05 times the smaller of
   ordinary and selected AMSE in every correct-law carryover cell. In each
   \(r=1\) least-favourable cell, the paired-bootstrap upper endpoint is below
   0.98.
5. In the no-carryover cell, selected/ordinary variance is at most 0.90 and its
   paired-bootstrap upper 97.5% endpoint is below 0.98, confirming that the Gate
   D opportunity survives the E2 implementation.

Standard selected intervals under harmful carryover, interval widths, selector
frequencies, the fixed-departure inconsistency pattern, and the unrestricted
oracle-joint benchmark are required descriptive results but are not silently
averaged into either decision.

## Pass, failure, and pivot rules

- **Hard correctness fails:** stop the full sensitivity-methods route and debug
  or retract the failed identity, inference, or implementation claim.
- **Hard correctness passes but practical value fails:** retain ordinary
  unrestricted inference, the restricted EIF, and the exact sensitivity
  geometry; pivot to a technical note or theory-plus-benchmark paper without a
  general recommendation for feasible adaptation.
- **Only assignment-law components fail after the correct-law components
  pass:** restrict all claims to a verified joint randomization law and remove
  assignment-uncertainty claims.
- **Both decisions pass:** proceed toward a full methods paper, while retaining
  ordinary taskwise inference as primary and labeling covariance-adaptive
  inference as conditional on the no-carryover restriction and analyzed
  assignment law.

Smoke results never make a gate decision. Numerical failures remain in the
denominator, all methods use common random numbers, and Monte Carlo bootstrap
resampling is paired by replication.
