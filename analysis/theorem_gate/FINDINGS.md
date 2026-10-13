# Findings from the exact two-fold theorem search

## Question

Does minimizing the expected risk of a single directed cross-fitting fold use
different fixed mixing weights from minimizing the risk of the complete
two-fold estimator, once the nuisance estimator trained on the opposite fold
is included exactly?

## Exact result in the minimal model

No. In the supported-assignment, two-task binary model implemented here, the
complete-estimator quadratic Gram tensor equals one-half of the marginal-fold
Gram tensor. Therefore, for every positive-definite task-error covariance
matrix, the complete risk is exactly one-half of the marginal risk for every
fixed calibrated mixing operator, and the two risk minimizers coincide.

This was checked by exhaustive assignment enumeration at fold sizes 2, 3,
and 4. The largest absolute tensor discrepancy was
`4.1106007e-14`, attributable to floating-point summation:

| Respondents per fold | Supported states per fold | Ordered fold pairs | Maximum tensor discrepancy |
|---:|---:|---:|---:|
| 2 | 4 | 16 | 0 |
| 3 | 36 | 1,296 | `1.3322676e-15` |
| 4 | 196 | 38,416 | `4.1106007e-14` |

The result is imposed as a deterministic assertion on the full quadratic
Gram tensors. It is stronger than observing equal rounded optimizer output.

## What remains finite-sample

The exact supported optimum can differ from Gate G's closed marginal formula,
which uses unconditional assignment-count moments rather than conditioning on
support. At error correlation 0.70, the closed formula's excess complete risk
relative to the exact supported optimum was 10.80%, 0.68%, and 0.06% for fold
sizes 2, 3, and 4. Thus the tiny-sample discrepancy in this experiment comes
from the assignment-support convention, not from optimizing a marginal fold
instead of the complete two-fold average.

## Scope boundary

The calculation integrates Gaussian outcomes analytically but treats the
mixing operator as fixed and independent of those outcomes. It does **not**
cover:

- operator estimation from the same outcomes used in the final estimator;
- uncertainty and bias induced by outcome-adaptive covariance weights;
- three or more folds, where nuisance-training samples overlap;
- unsupported assignments, richer factorial bases, or non-saturated nuisance
  fits.

Consequently, this executable result alone does not establish that feasible
outcome-learned GLS weights are harmless. The companion theoretical audit does
analyze those terms: the exact complete-estimator MSE adds one crossed moment,
which vanishes for the intended Gaussian residual-covariance learner by
odd/even symmetry. Generic regular counterexamples exist, but the gain from
reoptimizing the crossed term is only third order. See
[`docs/finite-g-theorem-search-gate.md`](../../docs/finite-g-theorem-search-gate.md).
