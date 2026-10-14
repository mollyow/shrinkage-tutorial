# Comparison v1, amendment 1: staged Gate C

**Protocol:** `factorial_ml_comparison_v1`, amendment 1  
**Amendment date:** 2026-10-13  
**Approved:** 2026-10-14  
**Status:** in force; Gate C1 implementation authorized; fresh outcomes prohibited  
**Machine-readable file:** [`comparison-v1-amendment-1.yml`](../config/comparison-v1-amendment-1.yml)  
**Lock manifest:** [`comparison-v1-amendment-1-lock.yml`](../config/comparison-v1-amendment-1-lock.yml)

Gates C–F here are the comparison benchmark's gates from the
[plan](negative-comparative-paper-software-plan.md#5-stage-gates). They are
unrelated to the earlier covariance-adaptive Gates C–G, such as
[`gate-d-decision.md`](gate-d-decision.md).

## Decision

Gate C is split into two stages. **Gate C1** freezes the shared engine and
everything the known-truth simulations need. **Gate C2** adds the
application-only machinery: strict endpoint readers, study-specific target
integration, and application report templates. Once C1 passes, the
positive-control calibration may run and Gate D may be evaluated before C2 is
built.

The reason is cost. Gate D is the benchmark's main scientific risk and depends
only on synthetic data. If no feasible learner can recover half of a
calibrated oracle gain, the benchmark cannot support a negative conclusion and
the project takes the software/tutorial route regardless of what the
applications would show. Building four strict endpoint readers before learning
that is avoidable work.

This amendment changes the order of work only. It changes none of the v1
protected choices: studies, endpoints, samples, estimands, covariates, folds,
seeds, learners, tuning budgets, target-integration rule, inference,
aggregation, margins, simulation cells, the positive-control rule, or the
failure and reporting rules. Fresh endpoints remain closed until full Gate C
(C1 and C2) passes, exactly as in v1. It nevertheless follows the v1
before-outcome amendment procedure: this dated changelog, a new lock manifest,
and no edits to any superseded file.

## Gate C split

| v1 Gate C requirement | Stage |
|---|---|
| Shared score, learner adapters, tuning, folds, metrics, aggregation, bootstrap | C1 |
| Simulation data generators and simulation target integration | C1 |
| Outcome-free version-4 candidate context artifact | C1 |
| Exact small-design score and integration agreement, including the paired-choice enumeration fixture | C1 |
| Scoped coding and rotation invariance tests | C1 |
| Fixed-coding reproducibility for elastic net, forest, and boosting | C1 |
| Outer and inner respondent-leakage tests | C1 |
| Seed-collision check | C1 |
| Simulation draw-count selection | C1 |
| Job, prediction, respondent-score, tuning, and failure schemas | C1 |
| `renv.lock` with `xgboost` 3.2.1.1, `glmnet` 4.1-8, and `ranger` 0.18.0 | C1 |
| Synthetic simulation-report render | C1 |
| Strict one-endpoint-at-a-time readers and missingness-only first access | C2 |
| Study-specific target integration and the Graham–Svolik nonproduct enumeration fixture | C2 |
| Application draw-count selection | C2 |
| Emitted Graham–Svolik inner folds | C2 |
| Synthetic endpoint fixtures, endpoint manifest schema, application report templates and render | C2 |
| Complete expected jobs and estimands; code and contract hash enforcement | Each stage for its own jobs |

Each stage ends with a decision record and an implementation manifest. The C2
manifest extends the C1 manifest and must confirm that every C1-hashed file is
unchanged or carries a recorded deviation.

## Amended sequence

1. Build and pass **Gate C1**; hash its manifest.
2. Run the positive-control **calibration** (seed phase offset 0) and evaluate
   **Gate D**.
3. If Gate D fails, stop. The v1 benchmark cannot support a negative
   conclusion, and the project takes the software/tutorial route. The
   multiplier grid is not extended (v1 rule). Any later benchmark would need a
   new protocol and could not be reported as v1.
4. If Gate D passes, hash the calibration results (v1 rule) and run the
   100-replication **smoke** run (seed phase offset 10,000,000) for numerics
   and runtime only.
5. Build and pass **Gate C2**. Full Gate C has now passed.
6. Run the **production simulations** and evaluate Gate E.
7. Run the **applications**, starting with the missingness-only first access.
8. Evaluate Gates A–F and choose the publication route mechanically.

Steps 6 onward are unchanged from v1.

## Safeguards

**C1 files are frozen once C1 passes.** If the smoke run or the C2 build
exposes a bug in a C1-hashed file, the fix is recorded as a dated deviation,
the C1 manifest is reissued, and every simulation phase already run on the old
files is rerun with the same seeds before production. If a rerun calibration
selects a different multiplier, both results are reported and the rerun
governs.

**C2 is written after calibration results are known.** It may therefore
implement only choices already fixed by v1 and the study contracts. Any C2
implementation detail those documents do not fix must be listed in the C2
decision record.

**Learners touch only simulated data before full Gate C.** The plan's statement
that no learner may be fit before Gate C passes is narrowed to fresh endpoints.
Learners may be fit to simulated data after C1. No learner may be fit to, and
no value may be read from, any fresh endpoint until C1 and C2 have both passed.

## Clarifications needed to run Gate D

v1 leaves three Gate D details open. They are fixed here before any
calibration result exists.

1. **Draw counts.** C1 applies the v1 draw-count rule to the simulation
   families and locks a simulation count. C2 applies the same rule to the
   application families and locks an application count. The grid
   `{16, 32, 64, 128}`, the one-percent RMS criterion, and failure at 128 are
   unchanged.
2. **Where recovery is measured.** Additive adjustment and the oracle are fit at
   all seven calibration multipliers. Elastic net, forest, and boosting are fit
   only at the selected multiplier, on the same 100 calibration datasets. Gain
   recovered uses replication-mean family-balanced risks, as defined in the
   simulation contract.
3. **Both outcome types must pass.** Gate D passes only if the rating and the
   paired-choice positive controls each meet both conditions: oracle/additive
   risk at most 0.75, and at least one of elastic net, forest, or boosting
   recovering at least 50% of the oracle gain. The recovering learner may
   differ between the two geometries.

## Clauses superseded

| File (unchanged, still hashed) | Clause | Amended reading |
|---|---|---|
| `config/comparison-v1.yml` | `next_authorized_step_after_this_contract.simulations_authorized: false` | Calibration authorized after C1; smoke authorized after Gate D passes |
| `config/comparison-simulations-v1.yml` | `authorized_next_action.run_calibration` and `run_smoke: false` | As above; `run_production` stays false until full Gate C |
| `config/comparison-v1-lock.yml` | `authorized_next_action: implement_gate_c_without_reading_outcomes_or_running_simulation_calibration` | Implement C1 without reading outcomes |
| `docs/cross-study-comparison-contract.md` | "No calibration run, simulation smoke run … is authorized by this contract." | Calibration and smoke are authorized by this amendment on the conditions above |
| `docs/negative-comparative-paper-software-plan.md` §9–10 | Gate C precedes calibration; no learner fit before Gate C | Sequence above; learner restriction scoped to fresh endpoints |

Everything else in v1 stands, including the outcome-access barrier and the
Graham–Svolik deviation disclosure.
