# Narrow paper assessment: design-aware orthogonal learning

**Assessment date:** 2026-08-27  
**Initial verdict:** yellow for a software-and-benchmark paper; red for a
new-DDML or new-theory paper.  
**Executed verdict (2026-08-27):** red for the separate ML paper. The
candidate-geometry oracle-headroom gate failed, and the full learner benchmark
was stopped according to the prespecified sequence. See the
[executed gate decision](narrow-ml-gate-decision.md).

## Recommendation in one paragraph

There was a potentially useful narrower paper, but it would have needed to ask **when
model-assisted adjustment improves causal factorial inference**, not claim to
invent DDML for conjoint experiments. The viable contribution is a reduced-
form, design-aware implementation that combines a separate assignment and
target law, restricted factorial supports, coding-invariant multilevel
factorial geometry, respondent-isolated learners, and clustered inference for
prespecified factorial functionals. A first application screen shows that
ordinary adjustment is valuable but flexible ML contributes essentially
nothing beyond a strong additive model. A small simulation shows that ML can
help under sufficiently nonlinear heterogeneity. That is enough to justify one
bounded benchmark stage, but not enough to justify a full paper. That stage has
now been run: even the known conditional mean reduced family-balanced MSE by
only about 3.3% relative to additive adjustment (ratio 0.967, bootstrap 95%
interval 0.957--0.978), so the 15% ML-improvement target is unattainable in the
candidate-calibrated design.

Suggested working title:

> **When Does Machine Learning Improve Conjoint Inference? Design-Aware
> Model-Assisted Estimation for Factorial Experiments**

Use “orthogonal learning” in the technical sections. “Model-assisted” is more
accurate in the title because assignment is known: the randomization correction
provides validity, while ML is used principally for precision.

## 1. Novelty audit

### Claims that are no longer available

The following claims are red:

- the first DDML procedure for conjoint experiments;
- a new augmented score for AMCEs, AMIEs, or factorial coefficients;
- a new all-pairs cross-fitting construction for learned U-statistics;
- a new squared-mean/order-energy U-statistic;
- a new generic bootstrap for a degenerate U-statistic; or
- a generic comparison of ML methods for conjoint data.

The decisive direct competitor is Acharya, Hainmueller, and Xu's
`sconjoint`, which already uses respondent-level cross-fitting, flexible first
stages, orthogonal scores, and respondent-clustered inference for conjoint
quantities ([paper](https://arxiv.org/abs/2604.10845),
[package](https://yiqingxu.org/packages/sconjoint/)). Other occupied pieces
include:

- weighted or restricted conjoint designs and clustered AMCE/ACIE estimation
  in [`cjoint`](https://web.mit.edu/teppei/www/research/conjoint.pdf);
- target distributions and partial-profile estimands in
  [`factorEx`](https://naokiegami.com/paper/conjoint_profile_with_appendix.pdf);
- baseline-invariant causal interactions and regularized factorial ANOVA in
  [`FindIt`](https://imai.fas.harvard.edu/research/files/int.pdf);
- heterogeneous high-dimensional factorial treatments in
  [`FactorHet`](https://imai.fas.harvard.edu/research/files/factorhet.pdf);
- BART-based conjoint heterogeneity in
  [`cjbart`](https://doi.org/10.1086/727597); and
- ML-based, design-valid factor tests in
  [`CRTConjoint`](https://arxiv.org/abs/2201.08343).

The general regression-adjustment theory is also occupied. Zhao and Ding cover
multiarm and factorial covariate adjustment
([JRSS B](https://academic.oup.com/jrsssb/article/85/1/1/7008637)), and a
recent paper develops conditionally unbiased ML-assisted cross-fitting for
randomized experiments ([Lu et al.](https://arxiv.org/abs/2508.15664)).

### Why the proposed global-test theorem also fails its gate

The strongest attempted theorem was a pairwise-cross-fitted test of

$$
Q_k=\|\beta_k\|_2^2,
$$

valid both away from and at the zero-interaction boundary. That route initially
looked promising because ordinary first-order inference degenerates at
\(Q_k=0\).

The latest revision of Escanciano and Terschuur already partitions **all
observation pairs**, trains nuisances outside each evaluation block, and uses
each pair once for DML U-statistics; it explicitly defers the degenerate case
([arXiv:2206.05235](https://arxiv.org/abs/2206.05235)). Chen and Lei give a
closely related two-sample construction
([Machine Learning 2025](https://doi.org/10.1007/s10994-024-06719-4)). The
remaining null problem has several close solutions:

- classical wild bootstrap for degenerate kernels
  ([Leucht and Neumann 2013](https://doi.org/10.1016/j.jmva.2013.03.003));
- boundary inference for nonnegative quadratic dissimilarities
  ([Hudson 2023](https://arxiv.org/abs/2306.07492));
- adaptive inference for treatment-effect variance with cross-fitted ML
  ([Sanchez-Becerra 2023](https://arxiv.org/abs/2306.03363)); and
- a dimension-agnostic, studentized cross-U test for a squared vector mean
  ([Kim and Ramdas](https://doi.org/10.3150/23-BEJ1613)).

For the factorial score, the missing random-kernel transfer argument appears to
be a short perturbation lemma under learner stability, not a paper-level
theorem. Without stability, a general result is likely false because a
bootstrap conditional on the fitted learners need not reproduce their
algorithmic randomness. The full-theory route therefore fails unless the
project is expanded into a much broader uniform nonregular-inference problem,
which would no longer be specifically about factorial experiments.

## 2. The remaining software gap

No located package combines all of the following in one reduced-form system:

1. A known assignment law \(q(a\mid x,t)\) and a distinct target law
   \(\mu(a)\), with an explicit \(\mu\ll q\) support check.
2. Product, nonproduct, and restricted profile supports, including rank and
   identification diagnostics for multilevel factorial orders.
3. Canonical \(\mu\)-weighted factorial spaces invariant to baselines, level
   permutations, and rotations of a contrast basis.
4. A learner-agnostic outcome-regression interface that isolates complete
   respondents in every training split.
5. Known-randomization correction and respondent-clustered joint inference for
   fixed factorial contrasts.
6. Efficient counterfactual integration that exploits factorial structure
   instead of treating every complete profile as an unrelated treatment arm.
7. An honest benchmark showing when flexible learning improves on simple
   adjustment—and when it does not.

This is an integration and usability contribution. Each ingredient has close
precedent. The safe verbs are “unify,” “implement,” “benchmark,” and
“specialize,” not “invent” or “first.”

## 3. Cheap empirical screens

### Candidate-recruitment application

The reproducible screen uses the locked five respondent folds, the 22
prespecified interaction coordinates, the six clearly pre-treatment binary
demographics, the uniform 1,440-profile target, eight task positions, and
published respondent weights. Every nuisance fit excludes complete held-out
respondents. The plug-in term integrates over all target profiles and task
positions for each demographic stratum.

| Nuisance adjustment | Held-out RMSE | Geometric-mean SE ratio to raw score |
|---|---:|---:|
| None/raw randomization score | 0.4678 | 1.0000 |
| Task-position model | 0.3329 | 0.7012 |
| Additive profile + task + demographics | 0.3117 | 0.6539 |
| Linear two-way/profile-by-demographic model | 0.3125 | 0.6558 |
| Elastic net over the expanded model | 0.3121 | 0.6547 |
| Regression forest | 0.3137 | 0.6579 |

The important comparison is not ML versus the raw Horvitz--Thompson score.
Basic outcome centering and additive adjustment already create most of the
gain. Relative to the additive model, the elastic net has:

- held-out RMSE ratio **1.0014**;
- geometric-mean standard-error ratio **1.0013**;
- geometric-mean variance ratio **1.0027**; and
- lower variance for only **5 of 22** coordinates.

Thus the application strongly supports model-assisted adjustment, but it does
not support a claim that flexible ML materially improves inference. The script
is [`analysis/narrow_ml/run_candidate_precision_screen.R`](../analysis/narrow_ml/run_candidate_precision_screen.R).

A separate sensitivity diagnostic using two additional respondent indices and
factor-swap integration reached the same conclusion: elastic-net/additive
held-out MSE was 0.9980, and the coordinate-variance ratio was 0.9984. Those
two indices require a formal pre-treatment timing audit before any confirmatory
use, so the six-demographic screen remains the primary feasibility result.

### Small simulation screen

The deliberately small screen uses 600 respondents, six tasks, five binary
attributes, ten two-way coefficients, three respondent folds, and 30
replications per scenario. It is diagnostic evidence, not a production
simulation.

| Truth | Learner | Coefficient RMSE | Coverage | SE ratio to rich linear model |
|---|---|---:|---:|---:|
| Simple | Rich linear | 0.00978 | 0.923 | 1.000 |
| Simple | Forest | 0.00987 | 0.937 | 1.000 |
| Linear heterogeneity | Rich linear | 0.00928 | 0.967 | 1.000 |
| Linear heterogeneity | Forest | 0.00942 | 0.980 | 1.050 |
| Nonlinear heterogeneity | Rich linear | 0.01528 | 0.963 | 1.000 |
| Nonlinear heterogeneity | Forest | 0.01192 | 0.967 | 0.774 |

Flexible learning can help when the conditional response surface is genuinely
nonlinear: in that deliberately favorable cell the forest reduced coefficient
RMSE by about 22% and reported SEs by about 23%. It did not help under simple or
linear truth. The screen is reproducible with
[`analysis/narrow_ml/run_simulation_screen.R`](../analysis/narrow_ml/run_simulation_screen.R).

## 4. What the paper could contribute

The paper should have three claims, in descending order of importance:

1. **Design-aware implementation.** A unified interface turns arbitrary
   held-out outcome predictions into valid estimates of prespecified causal
   factorial functionals while respecting assignment, target, support, and
   respondent clustering.
2. **Practical benchmark and guidance.** Predictive flexibility helps only when
   it reduces the variance of the relevant influence scores beyond a strong
   simple adjustment. Held-out prediction error alone is not the paper's
   causal performance metric.
3. **Omnibus order diagnostics as an application of existing theory.** An
   honest cross-U statistic can test whether a complete target-weighted
   factorial order is zero. Present this as a useful specialization, not a new
   U-statistic or new null theory.

The paper should not center individual preference recovery, latent preference
distributions, or structural choice parameters; `sconjoint`, `cjbart`, and
`FactorHet` are stronger and more natural tools for those questions. This
project's comparative advantage is reduced-form causal functionals under an
explicit randomized design and target law, including rating outcomes.

## 5. Minimal implementation, only if the benchmark stage continues

Refactor the existing `estimate_amie()` machinery into one engine:

```r
estimate_factorial_contrasts(
  x, y, cluster, basis, contrasts, assignment_prob,
  learner,
  context_cols = NULL,
  integration = c("target_grid", "observed_complement"),
  weights = NULL,
  outer_folds = 5L,
  fold_id = NULL,
  covariance = c("full", "diagonal", "none"),
  seed = 1L
)
```

Use a dependency-free learner contract:

```r
list(
  fit = function(x, y, weights, cluster, seed, ...),
  predict = function(object, newdata, ...)
)
```

Return held-out predictions, respondent score matrices, covariance, folds,
fits, target-integration diagnostics, importance-weight leverage, and support
rank. Retain `estimate_amie()` as a backward-compatible wrapper around the
ridge adapter. Put optional `glmnet` and tree learners in `Suggests`; the core
score engine should add no learner dependency.

For pairwise contrasts under a product target, `observed_complement` can swap
only the contrasted factors and average over their target levels while holding
the remaining randomized attributes and respondent context fixed. Production
code must verify the required factorization of the contrast weights rather
than silently assume it.

## 6. Decisive next benchmark

Before package refactoring, run one fuller but still bounded simulation using
the actual seven-attribute, 1,440-profile candidate design:

- 600 respondents and eight tasks;
- three respondent-level outer folds and respondent-safe inner folds;
- the same 22 locked interaction coordinates;
- additive truth, moderate profile-by-covariate heterogeneity, strong
  learnable nonlinear heterogeneity, and target/assignment mismatch with
  \(\max \mu/q\le2\);
- respondent random intercepts and arbitrary within-respondent task noise; and
- raw randomization, additive adjustment, elastic net, a forest or boosting
  learner, plug-in ML, and oracle adjustment.

Measure coefficient RMSE, empirical variance, bias in standard-error units,
95% coverage, interval width, family-test size and power, held-out prediction
error, Riesz leverage, failures, and runtime.

Precommit these gates:

1. In heterogeneous cells, ML/additive coefficient-MSE ratio at most **0.85**,
   with a bootstrap 95% upper bound below **0.95**.
2. Coverage between **93% and 97%** and absolute bias below **0.1 reported
   standard errors**.
3. Under additive truth, aggregate ML/additive excess variance at most **5%**,
   with no cell above **10%**.
4. In an application, ML/additive held-out MSE at most **0.98**, score-variance
   ratio at most **0.95**, at least **70%** of prespecified coordinates
   improved, and no family worsened by more than **5%**.

The candidate application currently fails every incremental-ML application
gate. A second application can be added only if chosen for design reasons
before inspecting method-specific results—for example, a public forced-choice
conjoint with documented restrictions and rich pre-treatment covariates. It
must not be selected because ML happens to win.

## 7. Publication decision

| Route | Current decision | What would change it |
|---|---|---|
| New DDML estimator paper | Stop | Nothing found in the present formulation. |
| New boundary/global-test theory | Stop | A genuinely uniform theorem not reducible to existing DML-U, cross-U, or boundary results. |
| Political-methodology software/benchmark paper | Conditional | Full realistic benchmark passes, a prespecified second application is informative, and the unified interface changes practice. |
| JSS/R Journal software article | Plausible | Production package, extensive correctness tests, documentation, and comparator coverage; large ML gains are helpful but not essential. |
| Tutorial/application note | Plausible now | Present design-aware adjustment honestly as a synthesis of established methods. |

The recommended investment is one bounded simulation-and-comparator sprint,
not a manuscript. If that gate fails, integrate simple model-assisted factorial
contrasts into the existing package and stop pursuing a separate ML paper.
