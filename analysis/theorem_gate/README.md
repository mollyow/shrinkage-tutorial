# Exact two-fold theorem search

This directory contains a standalone finite-sample calculation motivated by
Gate G. It does not source or modify the frozen Gate E2/G scripts.

`two_fold_exact_weight_search.R` studies the smallest model that retains the
shared-nuisance remainder:

- two repeated tasks;
- independently randomized binary profiles;
- two equal cross-fitting folds;
- task-specific saturated cell means trained on the opposite fold;
- a target averaging the two task-specific factorial contrasts; and
- Gaussian respondent errors with correlated task outcomes.

For every supported assignment in both folds, the script constructs the exact
outcome loading of both a single fold estimate and the complete two-fold
estimate. Gaussian outcomes are integrated analytically. Exhaustive assignment
enumeration then produces two quadratic risk functions over the calibrated
mixing matrices

\[
A=\begin{pmatrix}1/2&a_{12}\\a_{21}&1/2\end{pmatrix}.
\]

The calculation distinguishes five operators:

1. ordinary taskwise scoring;
2. the asymptotic covariance-optimal operator;
3. Gate G's closed expected marginal correction based on
   \(\Sigma+\Gamma_n\);
4. the exact marginal-fold optimum conditional on supported training cells;
5. the exact optimum for the complete overlapping two-fold estimator.

The principal theorem-search finding is an exact proportionality in this
minimal model: after averaging over the supported assignment distribution,
the complete-estimator quadratic risk matrix is one-half the marginal-fold
risk matrix. Consequently, the exact marginal and complete optimal fixed
mixing weights coincide. The executable check is imposed directly on the
quadratic Gram tensors, not inferred from rounded optimizer output.

This statement concerns supported-assignment risk for a fixed,
outcome-independent operator. It does not propagate uncertainty from learning
the operator with the outcomes and does not assert the same proportionality
for three or more folds, whose nuisance-training samples overlap.

The companion theoretical audit supplies that missing learned-weight
calculation. It shows that the complete two-fold MSE adds a single crossed
moment to marginal risk; the moment is exactly zero for Gaussian residual-
covariance learning, while in regular counterexamples its reoptimization gain
is only third order. The resulting hard novelty decision is documented in
[`docs/finite-g-theorem-search-gate.md`](../../docs/finite-g-theorem-search-gate.md).

Run the default exact calculation with:

```sh
Rscript analysis/theorem_gate/two_fold_exact_weight_search.R
```

The default fold size is four, giving 196 supported assignments per fold and
38,416 ordered assignment pairs. Optional arguments are:

```text
--fold-size=2|3|4|5
--rho=0,0.35,0.70,0.90
--output=path/to/results.csv
--quiet
```

The script checks supported-state counts, affine loading reconstruction, exact
centering, equality between the estimator and its outcome-loading
representation, fold-label invariance, first-order conditions, and both risk
minimization inequalities. Its calculations are exact over the enumerated
assignment distribution up to floating-point summation error; there is no
outcome or assignment Monte Carlo error.

To reproduce the fold-size 2, 3, and 4 identity grid and write the aggregate
method-risk table, run:

```sh
Rscript analysis/theorem_gate/run_two_fold_size_grid.R
```
