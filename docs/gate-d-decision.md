# Gate D decision: feasible covariance-adaptive estimator

**Decision: PASS. Proceed to Gate E assumption sensitivity.**

**Subsequent update:** Gate E1's carryover-sensitivity theory exercise passed,
but Gate E2 hard-failed two of seventeen computational checks. The feasible
adaptive estimator did not realize its first-order risk calculation reliably
at the frozen sample sizes. See `docs/gate-e1-theory.md` and
`docs/gate-e2-decision.md`; this does not revise the historical Gate D pass.

Gate C identified a large constant-working-covariance opportunity.  Gate D now
shows that a feasible estimator can learn and exploit that opportunity while
retaining valid respondent-cluster inference and falling back to ordinary
scoring when the opportunity is absent.

## Estimator evaluated

The estimator uses five respondent-level outer folds and four inner folds.
Within each outer training set it:

1. fits a task-specific additive outcome nuisance;
2. creates inner-cross-fitted residual vectors;
3. estimates compound-symmetric and unstructured covariance candidates;
4. constructs exact-calibration-preserving blends with ordinary taskwise AIPW;
5. uses a one-standard-error training-score rule, with ordinary scoring always
   available and lower blends preferred; and
6. applies the selected score only to untouched outer-fold respondents.

The simulation truth contains interactions omitted from the additive nuisance.
Consequently, validity comes from randomization calibration rather than correct
mean-model specification.

## Frozen production result

The production design used nine prespecified scenarios and exactly 2,000
replications per scenario: 18,000 replications total.  No cell was added,
removed, or extended after inspecting its result.

### Strong covariance headroom

| Scenario | Selected/ordinary variance | Bootstrap upper 97.5% | Oracle gain recovered |
|---|---:|---:|---:|
| Moderate CS, product, 4 tasks | 0.686 | 0.700 | 0.950 |
| High CS, product, 8 tasks | 0.291 | 0.302 | 1.000 |
| AR(1), product, 8 tasks | 0.482 | 0.497 | 0.976 |
| High CS, without replacement | 0.405 | 0.419 | 1.001 |
| Moderate CS, smaller sample | 0.640 | 0.653 | 0.945 |

Every strong cell cleared the frozen requirements: point ratio at most 0.90,
upper endpoint below 0.98, and at least half of the oracle gain recovered.  The
worst point ratio was 0.686, the worst upper endpoint was 0.700, and the minimum
gain recovered was 0.945.

### Safety and inference

- Weak-correlation aggregate variance ratio: **0.9963**, with upper endpoint
  **0.9971**.
- Largest safety-cell point ratio: **1.0000**.
- Pooled 95% coverage range across ordinary and selected scores: **94.48% to
  95.38%**.
- Maximum RMS standardized bias: **0.0364**.
- Sandwich-variance/Monte-Carlo-variance range: **0.948 to 1.055**.
- Maximum exact calibration residual: **1.25e-16**.
- Numerical failure rate: **0/18,000**.

The selector behaved coherently.  It chose ordinary scoring in 100% of
independence-cell outer folds.  In the weak-positive cell it chose ordinary in
59.4% and mostly a 0.25 compound-symmetry blend otherwise.  It chose full
compound-symmetry scoring in essentially every high-CS fold and selected the
unstructured model in 70.3% of AR(1) folds.

All separate correctness tests passed, including exhaustive wrong-nuisance
unbiasedness, product and without-replacement calibration, profile-level and
task permutations, reference-baseline changes, orthogonal rotations, outcome
and survey-weight scaling, and respondent-fold isolation.

## Candidate-recruitment supporting result

The real-data calculation was deliberately non-gating.  Nevertheless, it
supported the simulation result strongly:

| Assignment sensitivity | Selected/ordinary equal-family variance | Conditional bootstrap 95% interval | Oracle gain recovered |
|---|---:|---:|---:|
| Product uniform | 0.2358 | [0.2172, 0.2555] | 0.983 |
| Uniform without replacement | 0.2355 | [0.2165, 0.2552] | 0.984 |

All five outer folds selected the full compound-symmetry score.  Individual
family ratios ranged from 0.225 to 0.258, covariance stabilization was never
needed, and maximum calibration error was below 7e-17.

These application results are conditional on the mean no-carryover restriction
and on the stated product or uniform-without-replacement transition law.  They
do not resolve the missing platform-randomizer documentation and are replaceable
by a cleaner application.

## What Gate D establishes

Gate D establishes a credible feasible-estimation path:

- covariance learning can recover most of the oracle opportunity;
- exact calibration protects bias under a misspecified outcome nuisance;
- respondent-cluster sandwich intervals cover in the frozen designs; and
- the selector avoids detectable harm when cross-task covariance offers no
  useful information.

It does not yet establish robustness to conditional-mean carryover, an unknown
joint assignment mechanism, variable task counts, heavy tails, or arbitrary
high-dimensional covariance learning.  Gate E must compare with a fully robust
unrestricted-cluster estimator and quantify sensitivity to carryover and
assignment-law misspecification before the efficient score can be recommended
as a primary applied estimator.

## Reproducibility

The protocol is in `docs/gate-d-protocol.md`; the frozen configuration is
`analysis/feasible_eif/gate-d-production.yml`.  Machine-readable results and
the decision are in `results/feasible-eif/production/`, with the application
results in `results/feasible-eif/candidate/`.
