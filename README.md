# Order-adaptive shrinkage for causal factorial response surfaces

This repository now contains a clean, stage-gated implementation of an R
package and paper project for target-weighted, coding-invariant shrinkage of
dense factorial response surfaces. The working paper title is **“Order-Adaptive
Shrinkage for Causal Factorial Response Surfaces.”**

The project is deliberately marked **prototype / methods gate not passed**. A
fresh 200-replication optimizer-confirmation screen missed all three
prespecified benefit thresholds, so the current route is a comparison/software
paper rather than a claim for a generally performance-improving new estimator.
See `docs/gate-decision.md`. No old simulation figure or result is evidence for
the proposed estimator, and only a complete production run can reverse the
formal gate status.

A separate feasibility audit considered replacing or extending the shrinkage
approach with “DDML for factorial experiments.” That broad methods claim also
fails its novelty gate: the natural construction is an established augmented
score followed by a classical mean-norm U-statistic. A follow-up screen finds a
conditional design-aware software/benchmark route: simple adjustment is useful,
but flexible ML adds essentially nothing in the candidate application. No new
package estimator has been added; the isolated diagnostic scripts are under
`analysis/narrow_ml/`. See the
[`initial feasibility audit`](docs/orthogonal-ml-feasibility.md) and the
[`narrow-paper assessment`](docs/narrow-orthogonal-ml-paper.md).
The candidate-geometry oracle gate has now failed decisively, so the separate
ML paper is stopped; see the
[`executed gate decision`](docs/narrow-ml-gate-decision.md).

A different, narrower route studies efficient inference for repeated factorial
experiments by learning within-respondent outcome covariance under a
task-specific mean no-carryover restriction. Its oracle-headroom Gate C,
feasible-estimation Gate D, and Gate E1 theory exercise passed. Gate E2 then
hard-failed two of seventeen computational checks: a finite-sample
nuisance-estimation remainder prevented the feasible adaptive estimator from
realizing its first-order risk calculation at 250--500 respondent clusters.
The efficient-influence-function and sensitivity theory survived its controls,
but automatic feasible adaptation is not ready to recommend. Unrestricted
taskwise AIPW remains primary. See the [`Gate D decision`](docs/gate-d-decision.md),
[`Gate E1 theory decision`](docs/gate-e1-theory.md), and
[`Gate E2 decision`](docs/gate-e2-decision.md).

Gate E3 then tested whether an impossibility theorem could rescue a stand-alone
adaptation paper. It failed the novelty gate: one contrast reduces exactly to
the established restricted-versus-unrestricted Gaussian adaptation problem,
for which optimal imperfect adaptation already exists. A prespecified
carryover basis does produce a useful intermediate efficient score, but by
applying the existing calibration result to an augmented mean design. See the
[`Gate E3 theory decision`](docs/gate-e3-theory.md).

Gate F tested a factorial schedule-design theorem. The mathematics passed:
Hamming-distance randomization separates exactly by interaction order, makes
the efficient score computable through small temporal blocks, and exposes one
carryover-sensitive direction per active factorial feature. The stand-alone
novelty gate failed because the spectral identity and factorial crossover
efficiency design are classical. It also revealed an important estimand
warning: changing the dependent assignment law can change the task-marginal
causal target under carryover. See the
[`Gate F theory decision`](docs/gate-f-theory.md).

Gate G then derived and implemented the missing finite-sample cross-fit
remainder. The mathematics and fixed-operator inference worked: 35 exact
checks passed, oracle variance ratios were 0.952--1.018, and 95% coverage was
0.951--0.956. Count correction also reduced covariance-estimation error by
11--23% and repaired conservative plug-in intervals. But the 5,000-dataset
screen formally failed its scaling and local-AMSE gates, while corrected
covariance and finite-(G) weights changed point-estimation risk by essentially
zero. Honest split GLS recovered modest variance only in the restricted
finite-support model. The result therefore supports diagnostics and a careful
negative claim about added ML value, not a new adaptive-estimator paper. See
the [`Gate G decision`](docs/gate-g-decision.md).

The focused audit of the combined target-weighted EIF, carryover frontier, and
structured score is also complete, and the full-methods novelty gate fails.
The structured score is restricted covariance-weighted least squares on an
augmented `[B,Z]` mean space; its finite-radius version is established
approximate-moment minimax estimation. Most decisively, Jiang and Ding's August
2026 crossover paper already covers fixed sequence-level causal targets,
carryover restrictions, rank diagnostics, feasible covariance-weighted
restricted WLS, and robust inference. The remaining factorial, target-weighted,
and cross-fitted implementation is useful applied methodology, but no current
theorem survives as a new stand-alone methods contribution. See the
[`integrated novelty audit`](docs/integrated-novelty-audit.md) and the
[`post-audit contribution statement`](docs/post-audit-contribution-statement.md).
The audit also rules out a broad claim to introduce DML to conjoint analysis:
target-profile weighting is established, and a 2026 paper already uses DML with
flexible learning for repeated conjoint choices, albeit for a different
structural preference target.

The follow-up finite-cluster theorem search is now complete and also fails the
stand-alone novelty gate. In the intended Gaussian covariance-learning model,
the only crossed term distinguishing the complete two-fold estimator from a
marginal fold is exactly zero by odd/even symmetry. In the faithful fixed-
operator factorial enumeration, the complete and marginal quadratic risks are
exactly proportional, so their optimizers coincide. A generic non-even learned-
weight counterexample exists, but reoptimization improves risk only at third
order and the required overlap expansion is already covered by higher-order
DML/GMM theory. See the
[`finite-cluster theorem-search decision`](docs/finite-g-theorem-search-gate.md)
and its [`exact executable calculation`](analysis/theorem_gate/README.md).

The subsequent weak-overlap theorem gate is also complete. Product assignment
admits two useful factorial identities: an elementary-symmetric variance formula
under exact low degree and an exact Boolean-lattice formula for leakage from
omitted supersets under approximate low degree. The frozen population screen
shows that a known order ellipsoid can create substantial oracle headroom under
severe target/assignment shift. The stand-alone novelty gate nevertheless
fails: after an assignment-orthonormal basis transformation, the estimator,
modulus, and phase transition are classical minimax linear/optimal-recovery
results. The sparse-random-occupancy simulation was stopped under the
prespecified sequential rule. See the
[`weak-overlap gate decision`](docs/weak-overlap-gate-decision.md) and its
[`executable calculation`](analysis/weak_overlap/README.md).

The next restricted-support alias-geometry gate is complete and also fails the
stand-alone novelty test. Exact low degree can identify some full-target
factorial contrasts even when assignment excludes profiles, and the project now
has exact rank, minimum-variance, alias-witness, and approximate-tail sensitivity
calculations for parity fractions, Hamming slices, unions of slices, and a
forbidden pair. But arbitrary-support estimability is classical linear-model
and design-ideal theory, while the strongest union-of-slices rank formula is an
established affine Hilbert function. The remaining diagnostics are useful for
software and practical guidance; ML cannot remove the positive off-support
alias radius without a quantitative extrapolation assumption. See the
[`restricted-support gate decision`](docs/restricted-support-gate-decision.md)
and its
[`exact enumeration`](analysis/restricted_support/README.md).

The final minimal support-augmentation gate is also complete. Exact repair of
a fixed contrast reduces to sparse representation in the unidentified quotient;
full-model repair is matroid basis extension; and variance-plus-ellipsoidal-alias
optimization is ordinary correlated-observation $c$-optimal design. Exact
enumeration verifies the factorial special cases but also gives a direct
non-supermodularity counterexample and a frozen case in which forward greedy is
48.9% worse than exhaustive robust selection. This is a hard novelty failure,
so the project now stops opening adjacent theorem gates. The remaining viable
route is a practical support/overlap diagnostic or an explicit negative
comparative paper. See the
[`support-augmentation gate decision`](docs/support-augmentation-gate-decision.md)
and its
[`deterministic calculation`](analysis/support_augmentation/README.md).

The active publication route is now a prospective negative comparison rather
than another estimator search. It asks when flexible outcome learning improves
inference for fixed target-weighted factorial contrasts, using an
identification--headroom--recovery audit, known-truth positive controls,
equivalence margins, and at least four design-eligible public conjoints. The
software plan keeps the existing invariant ridge and AMIE workflow and adds one
support diagnostic; the ML learner library remains in the paper reproduction
pipeline. See the
[`negative comparative paper and software plan`](docs/negative-comparative-paper-software-plan.md).
The outcome-blind application audit now has four eligible fresh study
families: asylum seekers with both outcomes, climate agreements with rating
only, income-tax plans with the narrow six-rate total-effect choice analysis,
and Graham-Svolik with a narrow nonvalence democracy-contest choice analysis.
Climate paired choice remains excluded because zero identical pairs—versus
11.806 expected under independent profile draws—strongly disfavors the iid
implementation while leaving the exact opponent mechanism uncertified.
For tax plans, the noisy displayed-revenue category is locked as a downstream
mediator and not a seventh factor. Graham-Svolik supplies the required
nonproduct design through exact coarsened democracy and party block marginals;
its complete candidate-profile surface remains excluded because two documented
assignment mechanisms conflict with the release. Immigrant admission remains
amber but is no longer required. Portfolio Gate B is passed; no new aggregate
outcome, effect, or learner result has been opened. See the
[`application registry`](docs/application-dataset-registry.md),
[`asylum decision`](docs/asylum-design-contract-decision.md), and
[`climate decision`](docs/climate-design-contract-decision.md), and
[`income-tax decision`](docs/income-tax-design-contract-decision.md), and the
[`Graham-Svolik decision`](docs/graham-svolik-design-contract-decision.md).
The common scientific choices are now frozen in the
[`cross-study comparison contract`](docs/cross-study-comparison-contract.md).
Fresh endpoints remain closed while the shared Gate-C implementation and its
hash manifest are built and tested on synthetic and design-only inputs.
[`Amendment 1`](docs/comparison-v1-amendment-1-staged-gate-c.md) stages that
work: Gate C1 builds the shared engine and simulations, so the positive-control
Gate D can be evaluated before Gate C2 builds the endpoint readers. It changes
the order of work only, not any frozen scientific choice.

## What is implemented

- `factorial_basis()`: target-weighted factorial order subspaces, with exact
  support/rank diagnostics and transformations for new profiles.
- `fit_order_ridge()`: common, unrestricted order-specific, or monotonically
  ordered ridge penalties; weighted loss; `p > n`; and respondent-isolated CV.
- `balanced_order_sure()`: the balanced orthogonal SURE/PAVA special case.
  Adaptive fits use it automatically as an optimizer initializer when exact
  empirical balance is detected.
- `estimate_amie()`: respondent-level cross-fitting, known-randomization Riesz
  correction, clustered covariance, and simultaneous intervals for fixed
  target-profile contrasts.
- Exact finite-profile simulation risk, preregistered gate calculations, and
  fast/production configuration files.
- A data-boundary-safe candidate-recruitment analysis: code reads only the
  cleaned public replication file, while the large source archive remains
  ignored.

## Install and verify

```r
install.packages(c("nloptr", "testthat", "yaml", "jsonlite", "targets"))
system("R CMD INSTALL .")
testthat::test_local(".")
```

For the locked reproduction environment, install `renv` and run
`renv::restore()` from the repository root. The lockfile records R 4.4.2 and
all package dependencies used by the package, simulations, paper, and
application scripts.

Run the fast reproduction mode with:

```sh
Rscript simulation/run_simulations.R \
  --config config/simulation-fast.yml \
  --output results/simulations/fast/results.csv \
  --workers 1
Rscript simulation/summarize_gates.R \
  results/simulations/fast/results.csv \
  results/simulations/fast/gate-status.json
```

The production configuration uses 250 replications per screening cell and is
intentionally separate. Headline cells must be extended to 1,000 replications
before a methods-paper verdict. Because the proof-of-concept gate failed, the
production pipeline and candidate application are implemented but are not run
by default.

## Reproducibility boundary

The untracked `shrinkage/` directory is a 2019 historical archive. The
untracked `Candidate recruitment/` directory contains a larger substantive
project and raw/private files. Both are ignored by the new package. The
outcome-bearing candidate application scripts read only:

`Candidate recruitment/replication/PSRM_replication/data/CleanedData.csv`

Separate outcome-blind design audits use strict column allowlists against
public confirmatory archives. Those audits cannot load the locked response
fields.

See `docs/historical-material.md` and `docs/prior-art-matrix.md` for the evidence
and novelty boundaries.

## Historical tutorial

The original [tutorial](tutorial.md) remains available for provenance, but its
reference-coded estimator, terminology, and simulations are superseded. In
particular, the new project does not claim to invent multi-penalty ridge,
monotone shrinkage, PAVA, AMIEs, cross-fitting, or Bayesian ANOVA pooling.
