# Negative comparative paper and software plan

**Plan date:** 2026-08-31  
**Route:** prospective comparison paper plus a small diagnostic extension to
the existing R package.  
**Working title:** **Before Machine Learning: A Prospective Benchmark for
Causal Conjoint Inference**

## Decision in one paragraph

The paper will ask one question:

> When does flexible outcome learning materially improve inference for
> prespecified, target-weighted factorial contrasts beyond strong simple
> adjustment?

It will not narrate the sequence of failed theorem searches and will not claim
that machine learning is useless. Its contribution will be a prospective,
estimand-level benchmark organized around three checks:

1. **Identification:** is the requested contrast supported by the assignment
   design?
2. **Headroom:** how much improvement is available even to an oracle outcome
   model?
3. **Recovery:** how much of that opportunity can a feasible learner recover
   without harming coverage or stability?

The paper proceeds as a full *Political Analysis* submission only if it passes
the data, positive-control, inference, precision, and reproducibility gates
below. Otherwise, the project releases the package and a software/tutorial
paper without trying to turn imprecise null results into a methods claim.

## 1. Contribution and novelty boundary

### Primary estimand

Let $m$ denote the response surface and let $C$ contain a fixed finite set of
direct target-profile contrasts under a prespecified distribution $\mu$:

\[
\theta=C^\top m_\mu.
\]

The primary targets are complete, prespecified factorial interaction families,
not interactions selected after examining the outcome. For families of
different dimensions, the benchmark uses family-balanced risk,

\[
R_\theta
=
\frac1F\sum_{f=1}^F\frac1{d_f}
\|\widehat\theta_f-\theta_f\|_2^2.
\]

This prevents a large multilevel interaction family from dominating a
binary-by-binary family merely because it has more coordinates.

### The actual gap

The paper does **not** introduce DML, cross-fitting, AMCEs or AMIEs, a new
learner, selective machine learning, or a new interaction-discovery procedure.
Those areas are already occupied. In particular:

- Ham, Imai, and Janson use arbitrary ML statistics inside a randomization test
  for whether a conjoint factor matters in any way
  ([Political Analysis, 2024](https://doi.org/10.1017/pan.2023.41)).
- Robinson and Duch study ML for heterogeneous conjoint effects
  ([The Journal of Politics, 2024](https://doi.org/10.1086/727597)).
- Acharya, Hainmueller, and Xu use structural deep learning and DML for
  preference parameters
  ([2026 working paper](https://arxiv.org/abs/2604.10845)).
- Wang and coauthors benchmark methods for discovering subgroup and context
  interactions and propose a black-box procedure
  ([2025 working paper](https://arxiv.org/abs/2510.02123)).
- Generic learner selection for doubly robust functionals already has an oracle
  theory
  ([Cui and Tchetgen Tchetgen, 2024](https://doi.org/10.1093/biomet/asad055)).

The narrower available contribution is a prospective comparison of
**incremental precision for fixed reduced-form causal contrasts**, evaluated
using influence-score variance, risk, coverage, and equivalence margins rather
than interaction discovery or predictive accuracy alone.

### Defensible claims

If the gates pass, the paper can claim that it provides:

1. a fair target-weighted comparison of design-only, simple adjusted,
   regularized, and flexible estimators using identical respondent folds;
2. a practical separation of identification failure, absent oracle headroom,
   finite-sample learner cost, and restriction sensitivity;
3. known-truth positive controls proving that the benchmark can detect a real
   ML benefit;
4. a prospective multi-dataset assessment of whether flexible prediction
   improves inference for fixed conjoint contrasts; and
5. tested software for invariant contrasts, support diagnostics, structured
   ridge estimation, and conservative cross-fitted inference.

### Claims to exclude

The paper will not claim:

- that ML generally has no value in factorial experiments;
- a new DML or efficient-influence-function estimator;
- a new monotone or multi-penalty ridge method;
- valid inference for data-selected interactions or an entire fitted surface;
- recovery of individual structural preferences;
- that prediction error is interchangeable with causal-contrast precision;
- that ML repairs a contrast unsupported by the randomization; or
- that a covariance-efficient repeated-task estimator is robust without its
  no-carryover restriction.

## 2. The organizing diagnostic

Each analysis follows the same sequence:

```text
Is the target contrast identified by the design?
                    |
                    v
How much could a true outcome model improve over additive adjustment?
                    |
                    v
How much can a feasible learner recover at the available sample size?
                    |
                    v
Does the gain survive clustering, target shift, and split sensitivity?
```

For known-truth simulations, define

\[
H=\frac{R_{\mathrm{oracle}}}{R_{\mathrm{additive}}}
\]

and, only when the oracle gain is at least five percent,

\[
G_j
=
\frac{R_{\mathrm{additive}}-R_j}
{R_{\mathrm{additive}}-R_{\mathrm{oracle}}}.
\]

Here $H$ is the residual risk after oracle adjustment, while $G_j$ is the
fraction of available gain recovered by learner $j$. The paper will report the
underlying risks as well as these summaries.

Oracle headroom is known only in simulations or a fully specified
semi-synthetic experiment. In a real application, the paper will not label the
best observed learner an oracle. It will instead report held-out prediction,
cross-fitted influence-score variance, confidence-interval width, and split
stability relative to the prespecified additive baseline.

## 3. What existing evidence can and cannot do

The completed gates motivate the design but are not to be pooled into a new
confirmatory headline result:

- In the order-ridge pilot, monotone/unrestricted target-risk was $0.9874$,
  with a 95% upper bound of $1.0067$.
- In the candidate-calibrated headroom study, oracle/additive family-balanced
  MSE was $0.9674$ with interval $[0.9572,0.9781]$.
- In the candidate application, ranger/additive score variance was $1.0091$
  and elastic-net/additive was $1.0017$.
- The repeated-task work found oracle covariance headroom under no carryover,
  but feasible covariance learning changed point risk by essentially zero and
  gains could reverse under carryover departures.
- The support gates show constructively that missing target support is an
  identification problem, not a learner-performance problem.

These results are development evidence. The new benchmark needs a fresh
protocol, scenario registry, seeds, learner definitions, and reporting rules.
No old failed gate may be relabeled as prospectively preregistered.

## 4. Prospective evidence plan

### 4.1 Dataset benchmark

The live design-only shortlist and its unresolved blockers are recorded in the
[application dataset registry](application-dataset-registry.md) and the
machine-readable `config/application-registry.yml`. Candidate recruitment is a
developmental application: it has already influenced this project's estimands,
learners, and simulations, so it is displayed transparently but does not count
toward the confirmatory data gate.

Before inspecting any new learner comparison, perform a design-only audit of a
shortlist of public datasets. The first such audit, for immigrant admission,
is complete but amber because structural support conflicts across sources and
the exact within-pair opponent law is not generator-certified; see the
[study-specific decision](immigrant-design-contract-decision.md). The second,
for the asylum-seeker family, is [Gate-B green](asylum-design-contract-decision.md):
its prospective uniform law, samples, targets, covariates, folds, and complete
families are locked without opening its outcomes. The third audit, for
[climate agreements](climate-design-contract-decision.md), is Gate-B green for
profile ratings under declared mean-no-opponent and mean-no-carryover
restrictions but amber for paired choice because its within-pair opponent law
is undocumented. The fourth audit, for
[income-tax plans](income-tax-design-contract-decision.md), is green for a
narrow paired-choice total-effect surface over the six randomized bracket
rates. Its noisy revenue display is averaged as a downstream mediator and is
not treated as a seventh factor. The fifth audit, for
[Graham-Svolik candidate choice](graham-svolik-design-contract-decision.md), is
green for a narrow coarsened democracy-contest analysis. It certifies the
democracy, party, contest-relation, and candidate-gender marginals while
explicitly excluding the uncertified complete-profile law. A dataset is
eligible only if it has:

- a documented randomized attribute mechanism;
- respondent or independent-cluster identifiers;
- enough information to reconstruct the focal and, where relevant, competing
  profiles;
- a prespecifiable target distribution and contrast family;
- public data and a reproducible license/provenance path; and
- documented timing for any respondent covariates used by learners.

Selection is based on these design criteria, never on whether ML wins.

The full-paper route requires **four fresh eligible study families beyond
candidate recruitment**. Four now contribute eligible analyses: the Bansak-
Hainmueller-Hangartner asylum family with both outcomes, the Bechtel-Scheve
climate family with rating only, the Ballard-Rosa-Martin-Scheve tax-plan study
with forced choice, and the Graham-Svolik candidate study with a coarsened
nonproduct forced-choice analysis. Each counts once, not once per outcome or
arm. Gate B is therefore passed. Hainmueller-Hopkins immigrant admission stays
amber as an optional reserve, and European Defense Union remains the leading
thin-covariate calibration case.

Together, the four fresh studies must span at least three substantive domains,
include a multinational study, include both rating and paired-choice outcomes,
include product and restricted assignment, and contain at least three studies
with five or more documented pre-conjoint or immutable respondent covariates.
Countries, waves, outcomes, and experimental arms nested inside one study do
not count as independent studies. Headline summaries weight study families
equally.

Bundled empirical datasets from projects such as `sconjoint` are shortlist
leads, not analysis inputs. Package releases and development branches expose
different moderator frames. Every retained study must be rebuilt from its
original archive with a pinned version and checksum. The comparison concerns
reduced-form contrast precision, not any source project's structural
preference estimands.

At least one retained application should be a rating design and at least one a
paired-choice design. For paired choices, the protocol must define the focal
profile contrast while averaging over the randomized competing profile. A
small exact-enumeration test must agree with the implemented target integration
before the dataset becomes eligible.

Outcome-specific eligibility is permitted, but an uncertified outcome is
excluded rather than inheriting the study's status. Climate therefore enters
only through ratings unless its exact opponent law is recovered.

For every dataset:

- lock one to four substantively motivated interaction families before learner
  comparison;
- preserve all respondent rows in one fold;
- use the same folds, target integration, and survey weights for every method;
- declare post-treatment or ambiguously timed covariates unusable in the
  primary ML comparison; and
- report unsupported targets as unsupported rather than extrapolating.

### 4.2 Known-truth simulation

The main simulation mirrors the eligible application designs rather than
building a learner-favorable abstract example.

#### Rating-design cells

Use the 1,440-profile candidate geometry, repeated respondent tasks, and the
four locked interaction families. Include:

1. additive outcome surface with no useful respondent predictors;
2. additive surface with linearly prognostic respondent covariates;
3. linear profile-by-covariate effect modification;
4. nonlinear prognostic structure without treatment heterogeneity;
5. candidate-calibrated nonlinear treatment heterogeneity;
6. the same calibrated truth under mild target shift,
   $\max_a\mu(a)/q(a)\le1.5$; and
7. a frozen high-headroom nonlinear stress test calibrated before production
   so oracle/additive risk is approximately $0.70$.

The stress test is a positive control, not a claim about typical conjoint data.

#### Paired-choice cells

Use a compact version of the European Defense geometry and known joint
randomization of focal and competing profiles. Include:

1. an additive random-utility truth;
2. a low-headroom nonlinear truth; and
3. a high-headroom nonlinear positive control.

#### Replication schedule

- Run 100 replications per cell only for timing, numerical stability, and
  positive-control calibration. Do not use this run for substantive results.
- Run 500 fresh replications in all ten frozen cells.
- Extend four prespecified headline cells to 2,000 total replications: rating
  negative control, rating high-headroom control, choice negative control, and
  choice high-headroom control.
- Use common data and folds for all methods within a replication.

This entails 11,000 production datasets. A timing run will set the worker and
checkpoint plan but may not change scenarios, estimands, or thresholds.

### 4.3 Learner library

The primary comparison contains a deliberately finite library:

| Role | Method |
|---|---|
| Design-only reference | Known-randomization unadjusted score |
| Minimal adjustment | Task-position-only outcome model |
| Primary baseline | Additive weighted linear or generalized-linear model |
| Conventional expansion | Prespecified all-two-way model |
| Structured regularization | Unrestricted order-specific ridge |
| Sparse regularization | Elastic net |
| Tree ensemble | Regression/probability forest |
| Boosting | One frozen boosted-tree implementation |
| Simulation ceiling | True conditional-mean oracle |

Common and monotone order ridge are prespecified secondary ablations. They are
not proposed methods and do not drive the paper's publication decision.

All fitted outcome models enter the same known-randomization cross-fitted score.
Tuning occurs inside the outer-training respondents. Prediction clipping,
survey weighting, treatment of missing covariates, factor encoding, target
integration, and tuning budgets are frozen before production. Held-out outcome
loss is secondary; family-balanced influence-score risk and coverage are
primary.

### 4.4 Metrics

For simulations report:

- family-balanced contrast MSE and paired risk ratios;
- oracle headroom and gain recovered;
- bias in Monte Carlo standard-deviation units;
- pointwise and simultaneous coverage for the fixed contrasts;
- interval width and fixed-family test size/power;
- target-profile and assignment-law prediction loss;
- runtime, peak memory, numerical failures, and tuning stability; and
- sensitivity to the respondent-fold partition.

For applications report:

- held-out outcome loss;
- cross-fitted respondent-score variance and interval-width ratios relative to
  additive adjustment;
- point-estimate and family-test stability;
- support, target-weight leverage, and effective-sample diagnostics;
- runtime and failures; and
- repeated-split sensitivity as a diagnostic, while retaining one locked split
  for the reported estimator and interval.

Do not use coordinate win counts as a headline metric. Aggregate coordinates
within each complete family first, then weight families and studies equally.

### 4.5 Interpretation rules

For a risk or influence-variance ratio relative to additive adjustment:

| Classification | Prespecified rule |
|---|---|
| Material benefit | Estimate $\le0.90$ and 95% upper endpoint $<0.95$ |
| Small benefit | Estimate in $(0.90,0.95)$ without satisfying the first rule |
| Practical equivalence | Entire 95% interval inside $[0.95,1.05]$ |
| Material harm | Estimate $\ge1.10$ and 95% lower endpoint $>1.05$ |
| Unresolved | All other cases |

Thus “no added value” means practical equivalence or a confidence interval
that rules out the prespecified meaningful gain. It never means merely failing
to reject equality.

## 5. Stage gates

### Gate A: scope and prior art

Pass only if the introduction can distinguish fixed-contrast precision from
CRT testing, heterogeneous-effect discovery, structural preference learning,
and generic selective ML. Failure produces a software/tutorial paper.

### Gate B: data eligibility

Pass the full-paper route only if four fresh public study families beyond the
developmental candidate application satisfy the design, timing, provenance,
sample-size, and estimand rules in the application registry. At least three
must satisfy the rich-covariate tier. Fewer studies support a research note or
software paper, not the planned cross-study claim.

**Current status: passed.** The eligible families are asylum seekers, climate
agreements (rating only), income-tax plans (six-rate total effects), and
Graham-Svolik candidate choice (coarsened nonvalence democracy-contest effects).

### Gate C: implementation correctness

Required:

- coding, level-permutation, and basis-rotation discrepancies below $10^{-10}$;
- exact agreement with direct small-design calculations;
- no respondent leakage in outer or inner folds;
- paired-choice target integration verified by enumeration;
- no silent failed fits or dropped estimands; and
- assignment, target, fold, and learner contracts recorded by hash.

### Gate D: benchmark sensitivity

Before production, the positive controls must have:

- oracle/additive risk at most $0.75$; and
- at least one feasible flexible learner recovering at least 50% of the
  available oracle gain.

If this fails, the benchmark cannot support a negative conclusion because it
has not shown that it can detect a benefit.

### Gate E: inference and precision

Across headline simulation cells:

- 95% coverage must lie between 93% and 97%;
- absolute standardized bias must be below $0.10$;
- headline risk-ratio confidence-interval width must be at most $0.06$;
- numerical failure rate must be below 1%, with every failure reported; and
- under additive/no-headroom truth, aggregate excess risk must be at most 5%
  and no cell may exceed 10%.

### Gate F: full-paper contribution

Whether ML wins is not itself a gate. Proceed to a full comparison paper only
if Gates A--E pass and the prospective results yield at least one of:

1. a precise cross-study result that rules out a five-percent incremental
   precision gain for the flexible library in the studied class; or
2. a reproducible, prespecified diagnostic separation between settings where
   flexible learning does and does not recover material headroom.

If neither occurs, release the correct benchmark and software as a tutorial or
software paper without a general empirical conclusion.

## 6. Paper structure

1. **Why prediction gains need not imply inferential gains.** Motivate fixed
   factorial contrasts and define meaningful incremental value.
2. **Identification, headroom, and recovery.** Present the diagnostic sequence,
   target distribution, cross-fitted score, and family-balanced metrics.
3. **Prospective benchmark protocol.** Explain dataset selection, learner
   parity, positive controls, equivalence margins, and reproducibility.
4. **Known-truth simulations.** Show calibration, headroom, recovery, coverage,
   safety, and sample-complexity patterns.
5. **Cross-study conjoint benchmark.** Report results from all locked eligible
   datasets, including null, beneficial, harmful, and unresolved findings.
6. **Practical workflow and software.** Convert the evidence into a short
   decision guide.
7. **Limits.** Restrict conclusions to the estimands, designs, datasets, and
   learner library actually studied.

The repeated-task covariance/carryover results may appear as a short boundary
appendix showing a different source of apparent headroom. The theorem-search
history, broad augmentation optimization, and full Hamming-slice development
do not belong in the paper. Exact support examples may appear in the diagnostic
section or package vignette only.

## 7. Publication route

The primary target is *Political Analysis*, but only after the four-dataset and
full-paper gates pass. Its research-article format requires an original
contribution to methods used in political research and currently allows 6,000
words
([official instructions](https://www.cambridge.org/core/journals/political-analysis/information/author-instructions)).

If the main contribution remains implementation and practical guidance, the
fallback is *The R Journal* or *Journal of Statistical Software*. *The R
Journal* explicitly expects package papers to explain interface, performance,
comparisons, and limitations rather than reproduce a package manual
([package-paper guidance](https://journal.r-project.org/R_package_guidelines.html)).

Do not plan a JOSS submission yet. That route is reconsidered only after a
feature-complete public release and evidence of sustained use.

## 8. Software plan

### 8.1 Scope

Keep the existing `orderridge` package and its current stable workflow:

```text
factorial_basis()
    -> amie_contrast()
    -> fit_order_ridge() / estimate_amie()
```

Add one required public diagnostic in version 0.1:

```text
factorial_basis()
    -> amie_contrast()
    -> diagnose_factorial_support()
    -> fit_order_ridge() / estimate_amie()
```

Do not put a generic ML framework or a public method-comparison wrapper in the
package. Learner definitions and tuning are part of the paper protocol and
belong under `analysis/comparison/`.

### 8.2 Required version-0.1 interface

```r
diagnose_factorial_support(
  basis,
  contrasts,
  assignment_prob,
  model_order = basis$max_order,
  feasible = NULL,
  observed_counts = NULL,
  tail_bound = NULL,
  tol = basis$tol
)
```

It returns a `factorial_support_diagnostic` object containing:

- profile-level target weights, assignment probabilities, feasibility,
  observed counts, and zero type;
- model dimension, rank, nullity, smallest retained singular value, and
  conditioning;
- contrast-level ordinary overlap, exact-model estimability, row-space
  residual, variance factor, leakage, alias radius, and optional worst-case
  bias;
- minimum-variance representers for estimable contrasts;
- constructive null witnesses for nonestimable contrasts; and
- explicit exact-order and approximate-tail assumptions.

Only `print()` and `as.data.frame()` methods are required. The calculation uses
a weighted SVD and adds no hard package dependency.

`estimate_amie()` must remain conservative: it should not silently extrapolate
over zero assignment probability. The diagnostic explains the failure and the
extra assumption that would be needed.

### 8.3 Optional version 0.2

Only after the paper gate, consider:

```r
suggest_augmentation(
  diagnostic,
  candidates,
  budget,
  objective = c("estimability", "variance", "robust"),
  method = c("exact", "greedy"),
  max_subsets = 100000
)
```

Exact enumeration is allowed only below the search guard. Greedy output must
say `exact = FALSE`, state that it has no generic approximation guarantee, and
never be called optimal. Structural zeros are excluded from candidates. This
function does not block version 0.1 or the paper.

### 8.4 Package tests

Retain all current tests and add `test-support-diagnostics.R` covering:

1. full-support analytic agreement;
2. identification that depends on an exact low-order restriction;
3. a valid null witness for a nonidentified contrast;
4. direct variance and alias-radius agreement;
5. parity, Hamming-slice, and forbidden-pair fixtures;
6. nonuniform assignment;
7. baseline, level, row, and basis reparameterizations;
8. structural, design, and realized zeros;
9. rank-tolerance boundaries; and
10. clear failures for misaligned probabilities or contrasts.

Keep the large deterministic gate enumerations as integration checks rather
than routine unit tests.

### 8.5 Documentation and dependencies

Write two vignettes:

1. **A safe factorial-analysis workflow**: target, fixed contrasts, support
   audit, common/unrestricted/monotone ridge, and cross-fitted inference.
2. **What flexible prediction cannot repair**: restricted support, exact
   low-order identification, null witnesses, and approximate-tail bias.

Keep `stats`, `utils`, and `nloptr` as the only hard computational dependencies.
Keep `glmnet`, `ranger`, boosting, `targets`, `yaml`, and plotting tools in the
paper environment or `Suggests`. Do not add CVXR, a mixed-integer solver,
tidymodels, or a generic optimal-design dependency.

### 8.6 Release criteria

Before version 0.1:

- all package tests and deterministic gate scripts pass;
- `R CMD check` passes on release, old-release, and development R;
- examples and vignettes run from a clean library;
- complete-profile construction has explicit size guards;
- package claims are revised from estimator novelty to implementation,
  comparison, and diagnostics;
- the paper pipeline records dependency versions separately; and
- the release and reproduction archive receive immutable version identifiers
  and an archival DOI.

The package is not yet ready for a JOSS submission: it first needs a complete
public release and evidence of use. If the full empirical paper fails but the
software becomes the main output, target *The R Journal* or *Journal of
Statistical Software* after version 0.1.

## 9. Execution order

1. **Freeze this scope.** Write a one-page claim matrix and do not add another
   estimator or theorem gate.
2. **Lock dataset contracts without learner outcomes.** Promote four fresh
   study families from the provisional registry only after their assignment,
   covariate-timing, sample, support, and contrast checks pass.
3. **Implement support diagnostics.** Add the version-0.1 function, tests, and
   vignettes; rerun all deterministic gates.
4. **Build `analysis/comparison/`.** Unify data contracts, learner adapters,
   respondent folds, direct contrasts, influence scores, metrics, hashes,
   checkpoints, and failure ledgers.
5. **Calibrate only the positive controls and compute budget.** Freeze their
   parameters before production.
6. **Run the 100-replication smoke test.** Check numerics and runtime only.
7. **Run the 500-replication screen and four 2,000-replication headline
   extensions.** Do not alter cells after seeing results.
8. **Run all locked applications.** Report every eligible dataset and every
   prespecified method.
9. **Evaluate Gates A--F.** Choose full paper versus software/tutorial route
   mechanically.
10. **Write and release.** Render the paper, execute fast and archival
    reproduction modes, run package checks, and archive code, configurations,
    hashes, and retained results.

## 10. Immediate next deliverable

The [design-only application registry](application-dataset-registry.md) has
four eligible fresh families and Gate B is formally passed. The scientific
comparison choices are now frozen in the
[`factorial_ml_comparison_v1` contract](cross-study-comparison-contract.md) and
its machine-readable
[`application`](../config/comparison-v1.yml) and
[`simulation`](../config/comparison-simulations-v1.yml) configurations. These
files lock endpoints, missingness handling, estimands, folds, learners, tuning
budgets, equivalence margins, target integration, positive controls, and
reporting requirements.

The next deliverable is Gate C implementation: strict endpoint readers, the
shared score and learner engine, exact small-design tests, synthetic report
fixtures, output schemas, dependency lock, and a complete implementation hash
manifest. The protocol freeze does not authorize endpoint access. No fresh
outcome may be read and no learner may be fit until Gate C passes and those
artifacts are hashed. The one-row Graham-Svolik inspection deviation remains
disclosed and was never used for a comparison or design decision.
