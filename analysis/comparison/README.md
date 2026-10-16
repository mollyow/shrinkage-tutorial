# Comparison v1 engine (Gate C1)

This directory implements the shared engine and the known-truth simulations
for `factorial_ml_comparison_v1` ([plan](../../docs/plan.md)). Gate C1 covers
everything the simulations need; strict endpoint readers and study-specific
target integration belong to Gate C2. No code here reads a fresh endpoint.

## Layout

| File | Role |
|---|---|
| `load.R` | Loads the engine and the local `orderridge` package source |
| `R/00_utils.R` | Seed rule, hashed sub-seeds, balanced respondent folds, atomic writes |
| `R/contracts.R` | Protocol configs and phase authorization from the gate records |
| `R/bases.R` | Target-orthonormal Helmert bases and family coordinates |
| `R/geometry_rating.R` | Candidate rating geometry, cell surfaces, exact truth, simulator |
| `R/geometry_choice.R` | Four-factor paired-choice geometry, utility, exact truth, simulator |
| `R/features.R` | Feature maps, fold-internal preprocessing, model matrices |
| `R/learners.R` | The frozen learner library and the oracle |
| `R/integration.R` | Exact rating integration; shared-rest, level-balanced choice Monte Carlo |
| `R/score.R` | Cross-fitted score, sandwich covariance, intervals, Wald tests |
| `R/run_dataset.R` | One dataset: simulate, cross-fit every method, score, summarize |
| `R/metrics.R` | Across-replication risk, ratios, headroom, gain recovered, coverage |
| `R/draw_count.R` | Integration-error table for the fixed 128-draw choice integration |
| `R/registry.R` | Expected-job registries and the seed-collision check |
| `R/schemas.R` | Output schemas, completeness checks, parquet writer |
| `R/report.R` | Markdown reports |
| `run_gate_c1.R` | Runs every C1 check and writes the C1 record |
| `run_simulations.R` | Runs calibration, smoke, or production when authorized |
| `evaluate_gate_d.R` | Multiplier selection and the Gate D decision |
| `tests/` | Gate C1 tests (exactness, invariance, reproducibility, leakage, schemas) |

The outcome-free candidate context artifact is built by
[`analysis/candidate/build_comparison_context_v4.R`](../candidate/build_comparison_context_v4.R).

## Commands

From the repository root:

```sh
Rscript analysis/candidate/build_comparison_context_v4.R
Rscript analysis/comparison/tests/testthat.R
Rscript analysis/comparison/run_gate_c1.R --workers=2

# After Gate C1 passes:
Rscript analysis/comparison/run_simulations.R --phase=calibration --stage=1 --workers=6
Rscript analysis/comparison/evaluate_gate_d.R --stage=1
Rscript analysis/comparison/run_simulations.R --phase=calibration --stage=2 --workers=6
Rscript analysis/comparison/evaluate_gate_d.R --stage=2
```

`run_simulations.R` refuses a phase until the gates before it have passed
(C1 for calibration, Gate D for smoke, C2 as well for production), using the
records in `results/comparison-v1/gate-c1/`, `gate-d/`, and `gate-c2/`. It
resumes from per-dataset checkpoints. Bulk outputs are not versioned; the gate
records are. Run phases from a committed worktree; every output records the
commit.

## Compute

In the passing C1 synthetic render on one Apple M1 core, a full rating dataset
took 27 minutes and a full choice dataset 81 minutes at 128 draws; the forest
was 23 and 62 of those (see `results/comparison-v1/gate-c1/synthetic-report.md`).
An earlier render on the same machine was about three times slower. At the
faster rate, Gate D calibration needs about 170 core-hours and the
11,000-dataset production schedule about 9,000, which should run on a
cluster.

## Implementation choices

The protocol left these open. Each was fixed before any calibration run and
without reference to any result.

1. **Paired-choice nonlinear heterogeneity.** `NL(p, x) = sum_k g_k(p) psi_k(x)`,
   where `g_k` are linear combinations of the 15 orthonormal main-effect
   functions and `psi_k` are four standardized threshold features mirroring
   the rating geometry's Boolean context features: `xor(b1, b2)`,
   `1{z1 > 0} 1{z2 > 0}`, `b3 1{z3 > 0}`, and `1{|z4| > q_0.75}`. `z4`'s
   magnitude below the threshold and `b4` are noise covariates. Exact truth is a
   16-point enumeration.
2. **Paired-choice coefficients.** Seed 260903011 draws, in order, the 15 main,
   80 pair (all six factor pairs, Helmert tensor bases), and 60 heterogeneity
   coefficients, scaled to component RMS 0.35, 0.15, and 1.
3. **Coordinates** use `theta = E_mu[psi(A, B) m]` with Helmert bases, as in the
   application audits. Every rotation-invariant family metric is unchanged.
4. **Rating prognostic levels are cumulative**: cell 4 contains the linear and
   nonlinear prognostic components.
5. **Intercepts** are calibrated under the assignment law, so the target-shift
   cell 6 has exactly cell 5's response surface.
6. **Simulated respondents** are drawn with probability equal to their
   normalized weight and therefore carry equal weight.
7. **Integration.** Rating is exact (1,440 profiles x 8 tasks per covariate
   context). Choice uses unbiased family-cell-stratified Monte Carlo: for each
   respondent and family, the draws of the rest (other focal factors, opponent,
   task, side) are shared across cells, and each rest component is
   level-balanced across draws. The oracle uses the same integration as every
   other method.
8. **Learners.** Forest `mtry = max(1, floor(fraction x p))`. Boosting stops
   early on weighted validation RMSE, is selected on projected weighted MSE,
   and is refit with the rounded mean best round count. Elastic net uses the
   outer-training lambda path for its inner folds. Order ridge selects
   penalties by the package's closed-form weighted inner-fold MSE.
9. **Invariance scope.** Fixed-penalty order ridge and tuned common ridge are
   exactly invariant (1e-10). Tuned unrestricted and monotone ridge reproduce
   the invariant objective to the optimizer tolerance (relative 1e-5).
10. **Missing indicators** are not added because simulated covariates are never
    missing.
