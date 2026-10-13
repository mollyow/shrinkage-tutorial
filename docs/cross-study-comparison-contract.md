# Frozen cross-study comparison contract

**Protocol:** `factorial_ml_comparison_v1`  
**Freeze date:** 2026-09-03  
**Status:** scientific choices frozen; Gate C implementation pending; fresh outcomes prohibited

## Decision

The prospective benchmark will compare outcome learners only for fixed,
target-weighted factorial interaction families. It asks whether a flexible
conditional-mean model improves the precision of a known-randomization,
cross-fitted causal score relative to strong additive adjustment.

This contract completes the scientific part of the pre-outcome lock. It does
**not** authorize reading the four fresh endpoints. Outcome access remains
blocked until the common engine, strict endpoint readers, synthetic tests,
report schemas, dependency lock, and an implementation checksum manifest pass
Gate C.

The authoritative machine-readable files are
[`comparison-v1.yml`](../config/comparison-v1.yml) and
[`comparison-simulations-v1.yml`](../config/comparison-simulations-v1.yml).

## What counts as headline evidence

Each fresh study receives exactly one quarter of the cross-study headline
weight:

| Study family | Headline endpoint | Targeted analysis |
|---|---|---|
| Asylum seekers | 2016 `pref` | Forced choice under the documented product profile and opponent law |
| Climate agreements | `rating_cj` | Profile rating under mean-no-opponent and mean-no-carryover restrictions |
| Income-tax plans | `chose_plan` | Forced-choice total effects of the six randomized rates, averaging the natural revenue display mechanism |
| Graham-Svolik | `c_win` | Forced choice for the coarsened, supported democracy-contest design |

Asylum's 2016 rating and both 2022 outcomes are prespecified internal
replications, not extra studies. Candidate recruitment is displayed as the
developmental application and is excluded from every confirmatory aggregate.
Immigrant admission and European Defense Union remain reserves.

## Endpoint rules

All learners operate on the closed unit interval. Binary outcomes stay at
zero/one, the asylum rating is transformed by `(rate - 1) / 6`, and the climate
rating by `(rating_cj - 1) / 9`. All primary learning uses squared error, so a
binary endpoint's held-out loss is its Brier score. Binary log loss is
secondary. Predictions are projected to `[0,1]` before loss calculation,
target integration, and residual scoring; the frequency and magnitude of
projection are reported.

For asylum and tax choices, both profiles are oriented in turn as the focal
profile. A valid task has two finite binary indicators summing to one. For the
two rating outcomes, both profile ratings must be present and inside the
documented integer range. Graham-Svolik uses only the locked `candNum == 1`
nonvalence row and requires a finite binary `c_win`.

The first endpoint access is a coding-and-missingness check only. If any
expected task is missing or malformed, fitting stops. A separate versioned
missing-outcome protocol would then be required; the pipeline may not silently
average over the observed tasks, substitute another outcome, or promote a
secondary endpoint. This fail-closed rule avoids inventing a causal
missingness assumption after seeing response availability.

This is deliberately stricter than the provisional task-retention language in
some Gate-B study contracts and supersedes that language. The common contract
also supersedes any earlier statement that endpoint access became permissible
as soon as the comparison protocol was frozen: Gate C must still pass first.

## Samples, targets, and folds

The study-specific Gate-B contracts remain authoritative for provenance,
assignment, support, populations, baseline-covariate eligibility, targets, and
interaction families. The explicit common rules above govern endpoint
completeness and access. In particular:

- Asylum uses equal weight for each of 15 countries and the released weight
  within country. The waves are fit separately, and the 2022 randomized frame
  is integrated with probability `1/4`.
- Climate uses an equal four-country mixture and released weights within
  country. Its paired-choice endpoint remains excluded because the opponent
  law is not certified.
- Tax uses the 2,000-person displayed-revenue arm. The focal and opponent tax
  schedules are independent uniform draws, while `taxrev` is neither a factor
  nor a learner feature.
- Graham-Svolik uses the released 1,682-person population, the exact
  nonproduct assignment probabilities for its four coarsened factors, and a
  uniform target over the 64 supported cells. No unsupported complete-profile
  interpretation is made.

The primary partition is the already frozen five-fold respondent split with
seed `20260901`. The same outer and inner respondent labels are used by every
method. Graham-Svolik's missing inner labels must be emitted before endpoint
access using the same outcome-free algorithm as the other studies: seed
`20260901 + 104729`, multiplier 257, and reversed codes. Four additional
respondent partitions are diagnostic only and can never replace the primary
partition.

Numerical imputation, missingness indicators, and scaling are learned inside
training folds. Categorical levels come from the outcome-blind design locks,
with an explicit missing level. There is no outcome-based feature screening,
category pooling, target encoding, or use of post-treatment variables.

Task and neutral profile position enter as categorical design context: `task`
and `prof` for asylum, `conjoint` and derived neutral row index for climate,
`table` and `plan` for tax, and `matchNum` for Graham-Svolik. These columns have
penalty factor zero in penalized linear fits and enter tree learners as ordinary
predictors. This is an explicit common-protocol exception to earlier
never-feature lists for those fields only. Country is a mandatory target
stratum for asylum and climate and is not duplicated in the baseline block.
`cj_order`, `taxrev`, identifiers, weights, outcomes, mediators, and
post-treatment variables remain excluded. The machine-readable contract gives
the complete resolved feature map for each study and learner.

## Learners

Every fitted nuisance model enters the same cross-fitted randomization score.
The primary library is:

1. an unadjusted score with nuisance prediction zero;
2. task/design-context-only weighted least squares;
3. additive weighted least squares, the primary denominator;
4. a prespecified two-way weighted least-squares expansion;
5. unrestricted order-specific ridge through focal-profile order three;
6. elastic net;
7. a regression forest; and
8. boosted regression trees.

The two-way matrix contains focal-factor pairs, eligible opponent-factor
pairs, focal-by-opponent terms, and profile-factor-by-baseline-covariate terms.
It excludes baseline-covariate pairs and outcome-selected terms. Climate omits
opponent features because its eligible rating analysis relies on mean no
opponent.

Elastic net uses `glmnet` 4.1-8, five alpha values from zero through one, 60
lambda values, and weighted inner-fold MSE. The forest uses `ranger` 0.18.0,
an 18-point grid over `mtry`, terminal-node size, and sampling fraction, 300
trees for tuning, and 1,000 for the outer-training refit. Boosting uses
`xgboost` 3.2.1.1 with an eight-point depth, learning-rate, and child-weight
grid and inner-fold early stopping. All stochastic fits use one thread and
hashed seeds; parallelism occurs across jobs. The forest consumes the raw
columns in the same resolved feature map; the linear, ridge, elastic-net, and
boosting fits use their explicitly declared matrices.

Common and monotone order ridge are secondary ablations. The oracle exists
only in known-truth simulations. No application result may be called an
oracle.

Exact coding invariance is required for the estimand and target projectors,
coordinate transformations, scores given identical nuisance predictions,
rotation-invariant family summaries, least-squares fitted functions, and the
order-ridge fits. It is not claimed for elastic net, forests, or boosting:
those algorithms are representation-sensitive by construction. They instead
must be deterministically reproducible under the one frozen coding, seed,
package version, and one-thread setting.

## Estimation and target integration

For each fixed orthonormal family coordinate, the respondent score is

```text
target-integrated held-out prediction
+ known-randomization Riesz weight * held-out residual.
```

There is no learned treatment propensity. Task and side contributions are
averaged to one respondent score, and inference uses a respondent-clustered
sandwich covariance with the locked population weights.

Graham-Svolik uses exact 64-cell integration for every observed respondent-task
context. The larger product designs use family-cell-stratified Monte Carlo:
the focal family cell is enumerated exactly, while other focal attributes,
opponents, task, side, and frame are drawn from the declared target. Draws are
respondent-specific, independent of outcomes and training, and common across
methods. This makes the plugin calculation unbiased over the recorded
integration randomization and places its numerical variation inside the
respondent score.

Gate C chooses the draw count from `{16, 32, 64, 128}` using only synthetic
known-function tests. It selects the smallest count giving RMS integration
error below one percent of the exact respondent-score standard deviation for
every family. Failure at 128 fails Gate C. The selected count is hashed before
outcomes are unlocked.

## Metrics and aggregation

For a family of rank `d`, simulation risk is

```text
||estimated contrast - true contrast||^2 / d,
```

and application precision is `trace(covariance) / d`. These summaries are
unchanged by orthogonal rotations of the registered family coordinates.

Ratios use additive adjustment as the denominator. Families are averaged
equally within a study on the log-ratio scale; the four headline studies are
then averaged equally and exponentiated. Coordinate win counts are never a
headline result. The paper reports the numerator and denominator alongside
every ratio.

Comparison intervals use 4,000 paired respondent-cluster bootstrap draws from
the saved cross-fitted scores and losses, without learner refitting. Study
strata are preserved and the same bootstrap draw is used for all learners.
The four studies are fixed: they are not resampled, and the portfolio interval
does not claim inference to a superpopulation of conjoint studies. Effect
inference uses the respondent sandwich, pointwise normal intervals, 50,000-draw
Gaussian max-|t| intervals within each fixed family, and whole-family Wald
tests. There are no bands for a fitted response surface.

The interpretation margins remain:

| Classification | Locked rule for a method/additive ratio |
|---|---|
| Material benefit | estimate at most 0.90 and 95% upper endpoint below 0.95 |
| Small benefit | estimate between 0.90 and 0.95 without meeting the material rule |
| Practical equivalence | entire 95% interval inside `[0.95, 1.05]` |
| Material harm | estimate at least 1.10 and 95% lower endpoint above 1.05 |
| Unresolved | anything else |

Failure to reject equality is not equivalence.

## Positive controls and production simulations

The simulation registry contains seven repeated-rating cells on the
developmental candidate geometry and three paired-choice cells on a synthetic
four-factor subset of the audited asylum geometry. It includes additive
negative controls, prognostic nonlinearity without treatment heterogeneity,
linear and nonlinear treatment heterogeneity, mild target shift, and one
high-headroom positive control for each outcome type.

Only one scalar—the multiplier on the already defined nonlinear
treatment-heterogeneity component—is calibrated. On a frozen seven-value grid,
the rule chooses the passing multiplier closest to an oracle/additive risk
ratio of 0.70. If no value produces a ratio at most 0.75, Gate D fails; the grid
cannot be expanded. At least one of elastic net, forest, or boosting must
recover 50% of the available oracle gain.

Calibration and the 100-replication smoke run are not substantive evidence.
Production uses fresh seeds: 500 replications in all ten cells, with four
headline cells extended to 2,000, for 11,000 production datasets.
Calibration, smoke, and production receive disjoint phase offsets of zero,
10,000,000, and 20,000,000 before cell, replication, and purpose offsets are
added. Calibration candidates deliberately reuse common random numbers across
the seven multipliers; this is recorded as intentional pairing rather than a
seed collision.

The rating geometry uses the candidate study only developmentally. Gate C must
rebuild a design-only empirical distribution of its six binary respondent
covariates from an eight-column allowlist containing ID, weight, and those six
variables. The outcome is forbidden, the version-3 prepared bundle is stale,
and the new artifact must be hashed before calibration.

## Required implementation freeze

Before any fresh endpoint is read, Gate C must add and hash:

- strict one-endpoint-at-a-time readers;
- the common score, learner, integration, and aggregation engine;
- exact paired-choice and nonproduct enumeration fixtures;
- the scoped coding/rotation tests above, fixed-coding reproducibility tests for
  representation-sensitive learners, and respondent-leakage tests;
- synthetic endpoint and report fixtures;
- the outcome-free, version-4 candidate context-distribution artifact;
- complete job, prediction, respondent-score, tuning, and failure schemas;
- `renv.lock`, including the frozen boosting implementation; and
- an implementation manifest covering every source, contract, fold, contrast,
  and reporting file.

The immediate next authorized activity is implementation and testing on
synthetic and design-only inputs. No calibration run, simulation smoke run,
production simulation, or application analysis is authorized by this
contract.

## Deviations and exclusions that must remain visible

Candidate recruitment is developmental because it shaped the benchmark. Its
rowwise profile probability is known, but independence across its eight tasks
is not. Graham-Svolik's one-row response-display deviation remains disclosed;
that row was never summarized or used to choose an estimand, learner, or
comparison. Climate choice, tax revenue-controlled effects, Graham-Svolik's
complete-profile effects, and all data-selected interactions remain outside
v1.
