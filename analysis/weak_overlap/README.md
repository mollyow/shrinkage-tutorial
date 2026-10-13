# Weak-overlap theorem gate

This directory contains the deterministic calculation used for the
prespecified weak-overlap theorem gate.  The governing protocol and decision
are:

- [`docs/weak-overlap-gate-protocol.md`](../../docs/weak-overlap-gate-protocol.md)
- [`docs/weak-overlap-gate-decision.md`](../../docs/weak-overlap-gate-decision.md)

## Run

From the repository root:

```sh
Rscript analysis/weak_overlap/weak_overlap_gate.R
```

The script has no package dependencies beyond base R.  It writes:

- `weak_overlap_validation.csv`: 291 direct-matrix identity checks for
  $J\leq 8$, including unequal assignment shifts and order-one and order-two
  targets;
- `weak_overlap_risk_grid.csv`: the 1,872 estimator rows in the frozen
  population-information risk screen; and
- `weak_overlap_gate_summary.csv`: the eight gate counts and risk-ratio
  summaries quoted in the decision memo.

The run fails if any checked identity has absolute error at least $10^{-10}$.
The current maximum is $7.11\times 10^{-15}$.

## What is calculated

For binary factors under a uniform target and product assignment, the script
computes:

1. the target-Walsh Gram matrix and target-to-assignment basis transform;
2. the unrestricted Riesz variance;
3. the exact-degree population-information variance;
4. omitted-interaction contamination under degree truncation;
5. the optimal linear estimator over the frozen order-weighted ellipsoid; and
6. worst-case MSE and conservative normal, bias-aware half-lengths.

Equal assignment shifts permit a permutation-symmetric reduction from
$2^J$ coefficients to $2J$ groups.  The implementation verifies that reduced
solve against the complete $2^J$-dimensional solve for every $J\leq 8$.

## Boundary of the evidence

This is a Gaussian-sequence/population-Gram calculation.  It is exact for an
exactly orthogonal fixed design and is an information calculation for iid
random profiles.  It does **not** establish empirical-Gram concentration,
feasibility with sparsely occupied profiles, or a new minimax theorem.  The
prespecified novelty rule failed before a random-design simulation was
authorized, so the latter was deliberately not run.
