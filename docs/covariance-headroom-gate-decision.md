# Gate C decision: covariance headroom

**Decision: PASS. Proceed to the feasible-estimator gate.**

The candidate-recruitment design contains enough within-respondent covariance
headroom to justify continuing the repeated-factorial efficiency project.  This
is a sharply different conclusion from the earlier baseline-ML screen: flexible
outcome prediction added little, but covariance-calibrated use of the other
randomized tasks has a large oracle benefit.

## Frozen result

The inherited rule required an efficient/ordinary oracle residual-variance
ratio at most 0.75 and a respondent-bootstrap upper bound below 0.85.  All four
required combinations passed:

| Residual covariance | Task-assignment scenario | Ratio | Conditional bootstrap 95% interval |
|---|---:|---:|---:|
| Unstructured | Product uniform | 0.2227 | [0.2059, 0.2375] |
| Unstructured | Uniform without replacement | 0.2234 | [0.2066, 0.2383] |
| Compound symmetry | Product uniform | 0.2308 | [0.2153, 0.2482] |
| Compound symmetry | Uniform without replacement | 0.2315 | [0.2160, 0.2489] |

The worst required point estimate was 0.2315 and the worst 97.5th percentile
was 0.2489, both far inside the gate.  Equivalently, the constant-covariance
oracle indicates about 77% lower residual variance than ordinary taskwise AIPW.
Maximum numerical calibration error was (8.3\times10^{-17}).

The observed-geometry diagnostic told the same story.  With unstructured
covariance and product assignment, equal-family trace-per-dimension variance
was 0.2286 of ordinary scoring.  Individual locked-family ratios ranged from
0.2144 to 0.2557.  These geometry figures are non-gating because the same sample
was used to estimate the oracle operator.

The result was also insensitive to the simple residual model: task-only,
task-specific additive, and common two-way residualizers produced analytic
ratios of roughly 0.22--0.24.  The primary cross-fitted task-specific additive
residual correlation averaged 0.797.  No eigenvalue stabilization was needed.

## What passed—and what did not

This gate establishes only that there is substantial *oracle covariance
headroom* under the task-specific mean no-carryover model and a constant working
residual covariance.  It does not yet establish:

- that covariance weights learned on training respondents recover this gain;
- correct 95% coverage after nuisance estimation;
- robustness when covariance depends on profiles or respondent covariates;
- the mean no-carryover restriction; or
- the exact joint task-assignment law.

The last point is material.  Every one of the 3,000 respondents received eight
distinct complete profiles.  That is incompatible with literal independent
draws at any plausible chance level, even though the article and replication
materials document independent uniform randomization.  Product and uniform
without-replacement calculations agree here because the support has 1,440
profiles, but agreement is not evidence about what the platform actually did.
The ordinary taskwise estimator remains the assumption-light benchmark.

## Next gate

Gate D should implement training-fold covariance estimation and exactly
calibrated held-out scores.  The frozen requirements remain:

- selected/ordinary variance at most 0.90 with upper bound below 0.98;
- at least half of the oracle covariance gain recovered;
- no more than 5% aggregate harm in weak-correlation designs;
- 95% coverage around 93--97%; and
- coding and calibration invariance.

The candidate result alone is not enough.  Gate D needs simulation evidence,
including weak/negative correlation, covariance misspecification, and carryover
violations.  Until the joint randomizer is documented, the candidate efficient
analysis must remain a product-versus-without-replacement sensitivity exercise.

## Reproducibility

The frozen specification is in
`analysis/covariance_headroom/oracle-gate.yml`.  Exact solver checks and the
candidate run are reproduced with:

```sh
Rscript analysis/covariance_headroom/test_calibration_solver.R
Rscript analysis/covariance_headroom/run_candidate_oracle_gate.R
```

Machine-readable results are in `results/covariance-headroom/`, including the
gate rows, conditional-bootstrap summaries, moment diagnostics, family geometry,
and a serialized result bundle.
