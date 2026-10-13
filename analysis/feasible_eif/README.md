# Gate D: feasible covariance-adaptive scoring

This directory freezes, implements, tests, and evaluates the feasible estimator
from Gate D of `docs/factorial-eif-theory-feasibility.md`.

The primary estimator uses outer respondent cross-fitting.  Within each outer
training set, it obtains inner-cross-fitted residual vectors, estimates both
compound-symmetric and unstructured working covariance matrices, and constructs
exactly calibrated blends with ordinary taskwise scoring.  A respondent-level
one-standard-error rule selects the covariance class and blend while always
retaining ordinary scoring as the safest candidate.

## Files

- `gate-d-production.yml`: frozen 2,000-replication production design.
- `gate-d-smoke.yml`: 40-replication implementation check inheriting production.
- `feasible_estimator.R`: covariance estimation, calibrated blends, scoring,
  cluster sandwich inference, and the safe selector.
- `simulation_functions.R`: frozen factorial geometry, data generators, mean
  nuisance, and cross-fitted replication engine.
- `run_gate_d_simulation.R`: deterministic parallel simulation runner.
- `summarize_gate_d.R`: paired Monte Carlo bootstrap and pass/fail evaluation.
- `test_feasible_estimator.R`: calibration, invariance, scale, support, and
  end-to-end correctness tests.
- `run_candidate_gate_d.R`: non-gating candidate-recruitment engineering and
  precision screen.
- `check_gate_e1_theory.R`: exhaustive finite-support checks for the exact
  carryover-bias, sharp sensitivity, efficiency-vulnerability, blend, blind-
  direction, and assignment-misspecification identities.
- `gate-e2-production.yml`: frozen version-3, 3,000-replication-per-scenario
  Gate E2 design.
- `e2_simulation_functions.R`: local-departure, assignment-error, rescue,
  Hausman, and structured-sensitivity simulation machinery.
- `run_gate_e2_simulation.R`: deterministic Gate E2 simulation runner with
  configuration and source-integrity checks.
- `summarize_gate_e2.R`: frozen hard and practical component evaluation.
- `test_gate_e2.R`: Gate E2 leakage, support, decomposition, rescue, and
  end-to-end regression tests.
- `diagnose_gate_e2_failure.R`: non-gating higher-order and practical-risk
  diagnosis of the immutable production result.
- `check_gate_e3_theory.R`: deterministic checks of the scalar Gaussian
  reduction and the efficient score for a prespecified carryover subspace.
- `check_gate_f_theory.R`: deterministic enumeration of Hamming-spectral
  factorial schedules, order-block efficient scoring, carryover rank, and the
  fixed-target estimand warning.

## Reproduce

From the repository root:

```sh
Rscript analysis/feasible_eif/test_feasible_estimator.R
Rscript analysis/feasible_eif/run_gate_d_simulation.R \
  --config=gate-d-production.yml --cores=8
Rscript analysis/feasible_eif/summarize_gate_d.R --mode=production
Rscript analysis/feasible_eif/run_candidate_gate_d.R
Rscript analysis/feasible_eif/check_gate_e1_theory.R
Rscript analysis/feasible_eif/test_gate_e2.R
Rscript analysis/feasible_eif/run_gate_e2_simulation.R \
  --config=gate-e2-production.yml --cores=8
Rscript analysis/feasible_eif/summarize_gate_e2.R --mode=production \
  --input=results/feasible-eif/gate-e2-production/gate_e2_simulation_results.rds
Rscript analysis/feasible_eif/diagnose_gate_e2_failure.R
Rscript analysis/feasible_eif/check_gate_e3_theory.R
Rscript analysis/feasible_eif/check_gate_f_theory.R
```

Production simulation results are in `results/feasible-eif/production/`.
Candidate supporting results are in `results/feasible-eif/candidate/`.

## Decision

Gate D passed all frozen components.  This establishes feasibility under the
simulated known-assignment, mean-no-carryover designs.  It does not establish
mean no-carryover in an application, validate the candidate platform's joint
randomizer, or replace the need for a fully unrestricted cluster estimator and
assumption-sensitivity analysis in Gate E.

Gate E1 is now complete.  Its protocol and decision are in
`docs/gate-e1-protocol.md` and `docs/gate-e1-theory.md`; the machine-readable
algebra-check result is `results/feasible-eif/gate_e1_theory_checks.yml`.

Gate E2 is also complete. Its formal decision is a hard fail: 15 of 17 hard
components passed, but feasible local AMSE and paired sample-size scaling did
not. The known-mean controls support the first-order E1 theory and identify a
finite-sample nuisance-estimation remainder. See `docs/gate-e2-decision.md`.
Because the hard gate failed, the practical components were recorded only as
descriptive evidence and were not formally adjudicated.

Gate E3 is complete and failed its stand-alone novelty rule. The exact scalar
limit is already the Gaussian adaptation problem solved by Armstrong, Kline,
and Sun (2025). An efficient score robust to a prespecified carryover basis is
computationally useful, but follows by augmenting the existing mean-design
calibration. See `docs/gate-e3-protocol.md` and `docs/gate-e3-theory.md`.

Gate F is complete. Its mathematical component passed: Hamming-radial binary
factorial schedules diagonalize by interaction order, and the exact
carryover-bias map has one visible temporal direction per active Walsh
feature. Its stand-alone novelty component failed because factorial
changeover efficiency and the underlying hypercube spectral result are
classical. Changing a dependent schedule can also change Gate E1's
task-marginal target under carryover, so design claims must hold a direct or
structured target fixed. See `docs/gate-f-protocol.md` and
`docs/gate-f-theory.md`.
