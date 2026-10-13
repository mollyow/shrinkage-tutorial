# Narrow-ML paper gate: executed decision

**Decision date:** 2026-08-27  
**Decision:** stop the separate ML-methods paper before the full learner
benchmark. The oracle-headroom gate failed for scientific futility.

## Decision in one paragraph

The candidate application already showed no incremental gain from flexible ML
over additive adjustment. Before running a large forest benchmark, the project
therefore tested whether any meaningful gain was available even to an oracle
that knows the true conditional response function. Across 300 replications in
each of two nonlinear candidate-geometry scenarios, the pooled oracle/additive
family-balanced MSE ratio was **0.9674** (replication bootstrap 95% interval
**0.9572--0.9781**). The downstream ML gate required a ratio no larger than
0.85, while the headroom gate required an oracle ratio no larger than 0.75 and
an upper bound below 0.85. Because even the oracle's lower confidence bound is
above 0.85, the prespecified rule declares scientific futility. The full forest
benchmark was not run, as required by the stage gate.

## Locked sequence

The implemented sequence is:

1. Verify the exact candidate geometry and build a valid response-scale DGP.
2. Run an additive-versus-oracle headroom gate.
3. Only if headroom exists, run the complete learner, coverage, safety, and
   application gates.

The locked headroom configuration is
[`benchmark-headroom.yml`](../analysis/narrow_ml/benchmark-headroom.yml). The
unexecuted downstream configuration is
[`benchmark-production.yml`](../analysis/narrow_ml/benchmark-production.yml).
The latter remains useful if a substantively different application or estimand
creates genuine oracle headroom; it must not be run on a strengthened DGP
chosen after seeing this failure and then described as confirmatory.

## Design audit and corrections made before execution

The benchmark uses the seven candidate attributes, the complete 1,440-profile
grid, 600 respondents, eight tasks, the four locked interaction families, and
their 22 orthonormal coordinates. It resamples the joint distribution of six
substantively fixed respondent demographics and the empirical survey weights.

Several pilot assumptions were corrected before the gate was run:

- No observed respondent received the same complete profile twice. Under eight
  independent uniform draws from 1,440 profiles, zero repeats among 3,000
  respondents is effectively impossible. The benchmark therefore draws eight
  profiles without replacement within respondent while retaining a uniform
  marginal law. Inference clusters arbitrary within-respondent dependence and
  does not rely on row independence.
- Clipping and rounding a latent additive surface generated nonzero
  interactions on the observed scale. The production DGP instead uses a
  correlated beta-binomial construction on the ten-point rating grid whose
  conditional mean is exactly the specified factorial surface.
- The nonlinear-prognostic and nonlinear treatment-heterogeneity components
  were separated. The headroom gate uses nonlinear treatment heterogeneity,
  not a forest-favorable generic nonlinear prognostic effect.
- Uniform and shifted-target scenarios use the same generated datasets and
  folds. The shifted target changes only the estimand and has
  `max(mu/q) = 1.5`.
- Loss is the mean squared norm within each interaction family, averaged
  equally across the four families. The bootstrap resamples complete
  replications, never coordinates.

A 3,000-respondent calibration draw had weighted mean 0.324, SD 0.329, floor
mass 0.314, ceiling mass 0.060, and respondent ICC 0.769. The corresponding
application quantities are 0.329, 0.333, 0.372, 0.050, and approximately
0.785. The benchmark is close on the primary variance and dependence geometry,
although it does not exactly reproduce the application's excess zero mass.

## Oracle-headroom result

| Scenario | Oracle/additive family-balanced MSE | Bootstrap 95% interval |
|---|---:|---:|
| Nonlinear treatment heterogeneity, uniform target | 0.9693 | 0.9551--0.9842 |
| Same data-generating process, shifted target | 0.9659 | 0.9515--0.9810 |
| Equal-scenario pooled result | 0.9674 | 0.9572--0.9781 |

The oracle improves family-balanced MSE by only about 3.3%. This is an upper
bound on the practically recoverable gain from estimating the outcome model.
It is far below the 15% improvement required for the proposed paper.

The machine-readable result is
`results/narrow-ml/benchmark-headroom/headroom-gate-status.json`; the concise
generated report is
`results/narrow-ml/benchmark-headroom/headroom-gate-report.md`. Generated
results are intentionally gitignored, while the configuration, runner,
evaluator, and this decision are versionable source artifacts.

## Application gate under the locked 200-tree fit

The candidate application independently fails the incremental-ML gate:

| Metric, relative to additive WLS | Ranger | Elastic net |
|---|---:|---:|
| Held-out MSE ratio | 1.0116 | 1.0027 |
| Equal-family score-variance ratio | 1.0091 | 1.0017 |
| Fraction of 22 coordinates improved (legacy audit) | 0.091 | 0.227 |
| Maximum family variance ratio | 1.0121 | 1.0040 |

The locked requirements were held-out MSE at most 0.98, score variance at most
0.95, and no family variance ratio above 1.05. The family-safety requirement
passes, but neither flexible learner supplies an incremental benefit. The
coordinate-win fraction is reported only for continuity with the initial memo;
it is not a scientific gate because it depends on the chosen within-family
basis.

The six demographic variables are substantively fixed, but most were recorded
after the conjoint block. A future application should reconstruct strictly
pre-block features from the raw survey and perform any learned preprocessing
inside the training folds. This timing qualification makes the lack of an ML
gain no less decisive here, but it limits broader claims about the application
screen.

## What was implemented

- [`benchmark_functions.R`](../analysis/narrow_ml/benchmark_functions.R):
  candidate geometry, exact target contrasts, calibrated DGP, symmetric
  elastic-net encoding, respondent-safe fitting, clustered scores, oracle, and
  family confidence regions.
- [`run_candidate_geometry_benchmark.R`](../analysis/narrow_ml/run_candidate_geometry_benchmark.R):
  deterministic shards, atomic checkpoints, resume, failure ledger, exact key
  audit, input/code/config hashes, and environment manifest.
- [`evaluate_headroom_gate.R`](../analysis/narrow_ml/evaluate_headroom_gate.R):
  exact-key and fingerprint validation plus replication-level paired bootstrap.
- [`evaluate_benchmark_gates.R`](../analysis/narrow_ml/evaluate_benchmark_gates.R):
  the unexecuted downstream gate, including per-cell learner benefit, oracle
  gain recovered, family-region coverage, invariant bias, null-family size,
  additive safety, and the application criteria.
- [`test-narrow-ml-benchmark.R`](../tests/testthat/test-narrow-ml-benchmark.R):
  lock, null, target shift, no-repeat assignment, respondent isolation, and
  level-order symmetry tests.

The local test suite passes 201 tests. `ranger` is now recorded in the
environment lockfile, and the application was rerun with the locked 200 trees.

## Publication decision

The evidence closes the proposed separate paper under its own gates:

- **New DDML estimator paper:** stop for lack of novelty.
- **ML-improves-factorial-inference paper:** stop for lack of available
  precision headroom and failure in the real application.
- **Negative benchmark by itself:** not currently enough for a full methods
  paper; the result is specific to a design with very high unexplained
  respondent heterogeneity.
- **Software/tutorial integration:** still reasonable. A design-aware
  model-assisted contrast interface can be useful, but its practical default
  should be simple cross-fitted adjustment, with flexible ML optional and no
  claim of systematic superiority.

The result should not be reversed by increasing nonlinear signal after seeing
the gate. A new paper attempt requires a new, substantively motivated setting
with rich strictly pre-treatment covariates and demonstrable oracle headroom,
locked before comparing learners.

