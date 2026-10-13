# Post-audit contribution statement

## Recommended claim

Repeated conjoint and factorial experiments typically collect several outcomes
from each respondent. Analysts can use their covariance to estimate fixed
factorial contrasts more precisely, but the gain relies on restrictions about
how one task's randomized profiles affect another task's outcome. We provide a
target-preserving, coding-invariant implementation that makes this tradeoff
explicit. It reports an unrestricted taskwise estimate, the efficient estimate
under no carryover, and intermediate estimates valid for prespecified carryover
features, with respondent-clustered cross-fitted inference and sensitivity
summaries for a fixed contrast family.

The contribution is a **factorial/conjoint synthesis and implementation**. It
brings together established factorial estimands, conditional-moment efficiency,
restricted crossover estimation, and approximate-misspecification sensitivity
without letting the causal target change as the working carryover model changes.
It also documents an important practical negative result: in the designs
studied here, flexible covariance learning does not materially improve point
estimation beyond simpler methods, despite nonzero oracle efficiency headroom.

## What is genuinely useful

- The target distribution and contrast family are fixed before choosing a
  carryover model.
- The `U0` loading makes any target change visible when lag features are added.
- Factorial bases and reported contrasts are invariant to baseline coding and
  level labels.
- Support and rank diagnostics show which target contrasts and carryover
  directions the realized randomization can identify.
- The efficiency ladder quantifies the precision price of protection against a
  prespecified carryover space.
- Cross-fitting is isolated by respondent, and inference is for a fixed finite
  contrast set rather than a selected response surface.
- The software keeps unrestricted taskwise inference available as the
  assumption-light default.

## What is not a contribution

The paper does not invent the EIF, covariance-optimal GLS/GEE weighting,
cross-fitting, factorial AMCEs or AMIEs, coding invariance, restricted crossover
WLS, minimax approximate-moment estimation, bias-aware intervals, or adaptation
between a restricted-efficient and unrestricted-robust estimator.

Target-profile weighting is already explicit in de la Cuesta, Egami, and Imai's
conjoint framework. A broad claim to bring DML to conjoint data is also occupied
by Acharya, Hainmueller, and Xu (2026), although their structural preference
target differs from our design-based factorial contrasts.

In particular, Jiang and Ding's 2026 crossover framework already fixes linear
causal targets over treatment sequences and obtains covariance-weighted
restricted estimators under carryover assumptions. Standard conditional-moment
theory supplies the covariate-conditional semiparametric version, while
Armstrong--Kolesar and Bonhomme--Weidner supply the finite-radius robustness
problem. Our structured score is their combination in factorial coordinates.

## Publication verdict

**Do not proceed as a stand-alone full methods paper claiming a new estimator.**
The most defensible outputs are:

1. an applied-methods/tutorial paper centered on practical analysis choices for
   repeated conjoint experiments;
2. a small, well-tested software paper; or
3. a substantively led application in which task history or feature
   complementarity is important and this framework supplies the analysis.

The negative ML result can be part of any of these, phrased narrowly: flexible
covariance ML added essentially no point-estimation value in the designs and
sample sizes examined. It should not be stated as a theorem that ML has no
value for factorial experiments.

## Remaining route back to a full methods paper

The finite-cluster theorem for the **complete overlapping cross-fit estimator**
has now been investigated and fails as a route back. The exact two-fold MSE has
one crossed term beyond marginal-fold risk. That term is zero for the intended
Gaussian covariance learner; the faithful fixed-operator factorial calculation
also makes complete and marginal risk exactly proportional. In generic regular
examples the term can move the optimizer, but the resulting risk improvement
is third order, and higher-order GMM/DML theory already covers the required
overlap expansion. The full result and executable check are in the
[`finite-cluster theorem-search decision`](finite-g-theorem-search-gate.md).

The sole identified escape is a growing-dimensional repeated-factorial
occupancy result that simultaneously gives:

- sharp identification of a fixed pure-profile contrast under partially
  specified carryover;
- polynomial-time computation without enumerating complete schedules; and
- uniform, respondent-clustered inference when the number of factorial features
  grows and finite schedule cells are sparse.

That is a different many-profile research project and needs its own literature
and feasibility gate. It should be attempted only if a small precommitted
screen first finds a consequential gain in a sparse-support regime. More
learners, simulations, or applications do not satisfy it.
