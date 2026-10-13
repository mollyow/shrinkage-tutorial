# Minimal support-augmentation gate

This directory contains the deterministic calculation for the
[`frozen gate protocol`](../../docs/support-augmentation-gate-protocol.md).
The completed gate **fails as a stand-alone methods theorem**; see the
[`decision memo`](../../docs/support-augmentation-gate-decision.md).

Run from the repository root:

```sh
Rscript analysis/support_augmentation/support_augmentation_gate.R
```

The base-R script verifies quotient-space estimability, whole Hamming-layer
augmentation, parity de-aliasing, the Sherman--Morrison one-profile variance
update, a one-point Hamming-slice optimum, and forbidden-pair reductions to
antipodal and Hadamard designs. It also records a small factorial
counterexample to blanket supermodularity of the $c$-variance objective and
compares greedy with exhaustive robust augmentation in a frozen three-factor
example.

The generated files are:

- `support_augmentation_validation.csv`: seven numerical identity checks;
- `support_augmentation_parity.csv`: target-specific parity de-aliasing cases;
- `support_augmentation_layers.csv`: 120 whole-layer cases;
- `support_augmentation_special_cases.csv`: parity balance, one-point
  Hamming-slice, and forbidden-pair examples;
- `support_augmentation_risk.csv`: the supermodularity counterexample and 16
  greedy-versus-exhaustive robust comparisons; and
- `support_augmentation_gate_summary.csv`: machine-readable headline results.

All seven identities pass with maximum residual
$5.1\times10^{-15}$. The worst frozen greedy/exhaustive risk ratio is
$1.488636$.

This is exact finite-design enumeration.  It is not a Monte Carlo performance
study; the protocol's hard novelty stop was reached before simulation.
