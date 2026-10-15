# Plan: a prospective benchmark of machine-learning adjustment in conjoint experiments

**Protocol:** `factorial_ml_comparison_v1`  
**Updated:** 2026-10-15  
**Where we are:** Gate C1 passed on 2026-10-15. Next is the Gate D calibration.

## The question

In a conjoint experiment the attributes are randomized with known
probabilities. That means an attribute effect, or an interaction between two
attributes, can be estimated without bias with no outcome model at all. Adding
an outcome model doesn't change what is estimated; it only reduces noise. So
the question is about precision alone:

> When does a flexible machine-learning outcome model (elastic net, random
> forest, boosting) give materially more precise estimates of prespecified
> interaction effects than simple additive regression adjustment?

Earlier exploratory work on the candidate-recruitment study suggested that
simple adjustment helps and flexible ML adds almost nothing. The benchmark
tests that claim prospectively on four fresh public studies, with every
analysis choice fixed before their outcomes are read. It makes no claim that ML
is useless in general and proposes no new estimator.

## How the answer is computed

**Estimator.** Every method enters the same cross-fitted score. For each
respondent and each coordinate of a fixed interaction family, the score is the
method's held-out prediction, integrated over the target profile distribution,
plus the known randomization weight times the held-out residual. No propensity
is learned. Inference uses a respondent-clustered sandwich covariance, with
simultaneous intervals and a Wald test for each family.

**Methods compared.**

1. Unadjusted (prediction zero)
2. Task and design context only
3. Additive weighted least squares: the denominator for every comparison
4. Prespecified two-way least squares
5. Order-specific ridge through order three (the `orderridge` package)
6. Elastic net (`glmnet` 4.1-8)
7. Random forest (`ranger` 0.18.0)
8. Boosted trees (`xgboost` 3.2.1.1)

Common and monotone order ridge are secondary ablations. In simulations an
oracle uses the true conditional mean; nothing in an application is ever
called an oracle.

**Metrics.** For a family of dimension `d`, simulation risk is
`||estimate - truth||^2 / d`, and application precision is
`trace(covariance) / d`. Both are unchanged by rotating the family's
coordinates. Each method is compared with additive adjustment as a ratio.
Families are averaged equally within a study on the log scale, and the four
studies equally across studies.

**Interpretation.** For a ratio relative to additive adjustment:

| Classification | Rule |
|---|---|
| Material benefit | Estimate at most 0.90 and 95% upper bound below 0.95 |
| Small benefit | Estimate in (0.90, 0.95) without meeting the first rule |
| Practical equivalence | Whole 95% interval inside [0.95, 1.05] |
| Material harm | Estimate at least 1.10 and 95% lower bound above 1.05 |
| Unresolved | Anything else |

"No added value" means practical equivalence, or an interval that rules out
the meaningful gain. It never means only failing to reject equality.

## Studies

Each study counts for one quarter of the headline result.

| Study | Outcome analyzed | Design | Interaction families |
|---|---|---|---|
| Asylum seekers (Bansak, Hainmueller, Hangartner) | 2016 forced choice; the 2016 rating and both 2022 outcomes are internal replications | 15 countries, 18,021 respondents in 2016, 5 paired tasks, 9 attributes | reason x vulnerability; religion x origin; occupation x language |
| Climate agreements (Bechtel, Scheve) | Profile rating only; paired choice is excluded because the opponent-assignment mechanism is not certified | 4 countries, 8,500 respondents, 4 paired tasks, 6 attributes | cost x burden distribution; sanctions x monitoring; participation x emissions coverage |
| Income-tax plans (Ballard-Rosa, Martin, Scheve) | Forced choice; total effects of the six randomized bracket rates, with the displayed revenue treated as a mediator | 2,000 respondents in the displayed-revenue arm, 8 paired tasks | lowest x highest; upper-middle x highest; the two highest brackets |
| Democratic candidate choice (Graham, Svolik) | Forced choice in the coarsened democracy contest; the nonproduct assignment case | 1,682 respondents, 11 paired tasks, 64 supported cells | democracy x focal party; democracy x contest relation; democracy x focal gender |

Candidate recruitment (Eggers et al.) is shown as the developmental
application. It is excluded from every confirmatory summary because it shaped
the design, and it supplies the rating geometry for the simulations. Immigrant
admission and the European Defense Union study are reserves.

The study configs under `config/datasets/` are authoritative for provenance,
assignment laws, support, targets, covariates, and families. The outcome-blind
design audits under `analysis/<study>/` produced the folds, family coordinates,
and projectors in each `derived/` folder.

The first read of each outcome is a coding and missingness check. If any
expected task is missing or malformed, fitting stops until a separate
missing-outcome rule is written.

## Simulations

Simulations with known truth answer two questions the applications cannot:
whether the score's inference is valid, and whether the benchmark could detect
an ML benefit if one existed.

- **Rating geometry** (cells 1–7), built on the candidate-recruitment design:
  600 respondents, 8 tasks, 1,440 profiles. Cells add, in turn, prognostic
  covariates, linear and nonlinear heterogeneity, and a target shift.
  Integration is exact.
- **Paired-choice geometry** (cells 8–10), a four-factor version of the asylum
  design: 1,000 respondents, 5 tasks, 420 profiles. Cells are additive, weakly
  nonlinear, and the positive control. Integration is Monte Carlo with 128
  draws per respondent and family.

**Positive control.** Cells 7 and 10 make respondents' preferences depend
nonlinearly on their covariates. A multiplier sets the strength. Calibration
runs multipliers 0.5, 0.75, 1, 1.5, 2, 3, and 4 and, separately for each
geometry, picks the one whose oracle/additive risk is closest to 0.70 among
those at or below 0.75. That gives a setting where a better outcome model is
clearly worth about 30%. Rating multipliers 3 and 4 push the mean rating
outside [0, 1], so they cannot be generated and are never selected.

The schedule is 100 calibration datasets per candidate, 100 smoke datasets per
cell, and 500 production datasets per cell (2,000 for the four headline cells),
for 11,000 production datasets. Seeds for every phase are fixed in
`config/comparison-simulations-v1.yml`.

## Gates

| Gate | Passes when | Status |
|---|---|---|
| A. Scope | The paper can separate fixed-contrast precision from randomization tests, heterogeneity discovery, structural preference models, and generic ML | Judged when the introduction is drafted |
| B. Data | Four fresh eligible studies, at least three with rich covariates | Passed (September 2026) |
| C1. Engine and simulations | Exactness (1e-10), coding and rotation invariance, reproducibility, no fold leakage, seed collisions, schemas, frozen learner versions, a full-size synthetic run | Passed (2026-10-15) |
| D. Sensitivity | In both geometries, oracle/additive risk at most 0.75, and elastic net, forest, or boosting recovers at least 50% of the oracle gain | Next |
| C2. Applications | Strict outcome readers, study-specific integration and draw counts, synthetic endpoint fixtures, application report templates | After D |
| E. Inference | In the headline simulation cells: coverage 93–97%, absolute standardized bias below 0.10, ratio interval width at most 0.06, failures below 1%, and at most 5% excess risk (10% per cell) under additive truth | After production |
| F. Paper | Gates A–E pass, plus a precise result ruling out a 5% gain or a prespecified diagnostic that separates settings with and without recoverable headroom | After applications |

Gain recovered is `(R_additive - R_learner) / (R_additive - R_oracle)`. It is
computed only when the oracle gain is at least 5%, and it is measured on the
same 100 calibration datasets as the selection. If Gate D fails, the benchmark
cannot support a negative conclusion. The project then releases the software
with a tutorial or software paper, and the multiplier grid is not extended.

## Sequence

1. Pass C1.
2. Calibrate the positive controls and evaluate Gate D.
3. Run the smoke simulations, for numerics and runtime only.
4. Build and pass C2.
5. Take the outcome-access snapshot (below).
6. Run the production simulations and evaluate Gate E.
7. Run the applications, starting with the missingness check.
8. Choose the publication route from Gates A–F.

## Rules

**Outcome barrier.** No outcome from the four fresh studies is read until C1,
D, and C2 have passed and the outcome-access snapshot exists. The snapshot is
a commit, tagged `comparison-v1-outcome-access` and pushed to GitHub, that
adds a manifest of SHA-256 hashes for the engine and reader code, both
protocol configs, the study configs, the `derived/` design artifacts,
`renv.lock`, and the gate results. The C2 readers refuse to run unless the
checkout is that tag with a clean worktree. This snapshot is the
preregistration.

**Changes before outcome access** are made in place and recorded in the
changelog below with a date and a reason. Git history keeps every earlier
version.

**After outcome access** the primary analysis is fixed. A necessary bug fix is
disclosed, applied to every method and dataset it affects, and reported next to
the original result. Any other change is exploratory.

**Simulation runs** start from a committed worktree, and every output records
the commit. If engine code changes after C1 passes, C1 is rerun, and any phase
already run on the old code is rerun with the same seeds.

## Compute

In the passing C1 run on one Apple M1 core, a full rating dataset took 32
minutes and a full choice dataset 83 minutes; the forest was 27 and 59 of
those. An earlier run on the same machine was about three times slower, so
treat these as best-case figures. At these rates Gate D needs about 170
core-hours (10 for the oracle/additive selection, 160 for the three learners
at the selected multipliers). Production needs about 10,000 core-hours, and up
to three times that on a busy machine, so production runs on a cluster.

## Software

The `orderridge` package keeps its workflow: `factorial_basis()`,
`amie_contrast()`, then `fit_order_ridge()` or `estimate_amie()`. Version 0.1
adds one public diagnostic, `diagnose_factorial_support()`. It reports
profile-level support, model rank, and contrast-level estimability, and gives
null witnesses for contrasts that cannot be estimated. It is not yet
implemented. The package vignette, `vignettes/orderridge.Rmd`, is the
tutorial.

Learner definitions and tuning stay in `analysis/comparison/`, not in the
package. Before release: `R CMD check` on release, old-release, and devel R;
examples that run from a clean library; size guards on complete-profile
construction; package claims framed as implementation and diagnostics, not
estimator novelty; and an archival DOI.

## Publication

*Political Analysis* if Gates A–F pass. Otherwise *The R Journal* or the
*Journal of Statistical Software*, with the package as the main contribution.

## Changelog

- **2026-09-03.** Protocol v1 fixed: studies, outcomes, learners, tuning,
  integration, inference, aggregation, margins, simulation cells, and the
  positive-control rule.
- **2026-10-14.** Gate C split into C1 and C2, so Gate D can be evaluated before
  the outcome readers are built. Gate D details fixed: both geometries must
  pass, and the recovering learners are fit only at the selected multiplier,
  on the same calibration datasets.
- **2026-10-15.** Simulation draw count fixed at 128 for every paired-choice
  function. The v1 one-percent rule was written for the applications and still
  governs them in C2; it no longer applies to the simulations. The draws are
  unbiased and their noise sits inside each respondent's score, so they do not
  affect validity. At 128 draws the worst-case error is 2.24% of the score
  standard deviation (multiplier 4), which adds 0.05% to the score variance.
  The noise falls on the oracle and the flexible learners, not on the additive
  denominator, so it works slightly against passing Gate D. Keeping
  multipliers 3 and 4 selectable can only make the choice positive control
  easier to reach. The full table is in
  `results/comparison-v1/gate-c1/draw-count-selection.yml`. No calibration
  result existed when this was decided.
- **2026-10-15.** Per-amendment lock manifests replaced by the single
  outcome-access snapshot. The history documents and the code from earlier,
  closed routes were removed: the order-adaptive shrinkage methods gate,
  covariance-adaptive inference, the theorem searches, and the old order-ridge
  fit of the candidate application (the developmental display will use the
  benchmark engine). They remain in git history up to commit `bfdedc5`.
- **2026-10-15.** C1 fixes: the test that Monte Carlo integration is unbiased
  used a function the integration handles exactly, so it failed on rounding
  noise. It now includes interactions that produce real Monte Carlo error. The
  C1 record also no longer writes a duplicate `passed` key. Unused engine
  helpers and the `targets` dependency were removed, and C1 was rerun on the
  result.
