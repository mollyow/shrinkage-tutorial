# Restricted-support alias-geometry gate

This directory contains the exact-enumeration calculation for the
[`frozen protocol`](../../docs/restricted-support-gate-protocol.md).  The
pass/fail interpretation is in the
[`gate decision`](../../docs/restricted-support-gate-decision.md).

Run from the repository root:

```sh
Rscript analysis/restricted_support/restricted_support_gate.R
```

The script uses base R and enumerates $J=3,\ldots,10$ binary factorials through
degree four.  It evaluates parity fractions, single and multiple Hamming
slices, a forbidden-pair support, and deterministic generic-support controls.
It writes rank, coefficient-estimability, variance-factor, and formula-check
CSVs in this directory, plus `restricted_support_gate_summary.csv` containing
the decision-memo counts.

The current run contains 234 canonical closed-form rank checks with zero
discrepancy.  Across all 742 checks, the maximum numerical error is
$2.84\times10^{-13}$.

The calculation is a fixed-allocation Gaussian cell-mean experiment.  It does
not establish finite-sample performance under iid sparse profile occupancy.
