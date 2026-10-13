# Covariance-headroom theory gate

This directory implements Gate C from
`docs/factorial-eif-theory-feasibility.md`.  It asks a narrow question:
if the within-respondent residual covariance were known, how much could an
exactly calibrated repeated-factorial score improve on ordinary taskwise AIPW?

It does **not** establish that an estimated covariance-adaptive procedure is
efficient, that its intervals cover, that mean no-carryover holds, or that the
candidate survey's joint task-assignment law has been recovered.

## Files

- `oracle-gate.yml` freezes the Gate-C specification and inherited thresholds.
- `calibration_solver.R` contains the full finite-support calibration solver and
  the scalable product/SRS-without-replacement task solver.
- `test_calibration_solver.R` checks exact enumeration, calibration, analytic
  special cases, rotation equivariance, wrong-nuisance unbiasedness, and support
  failure.
- `run_candidate_oracle_gate.R` cross-fits the locked candidate data, estimates
  the residual moment, runs a respondent bootstrap, and writes the gate result.

## Reproduce

From the repository root:

```sh
Rscript analysis/covariance_headroom/test_calibration_solver.R
Rscript analysis/covariance_headroom/run_candidate_oracle_gate.R
```

Outputs are written to `results/covariance-headroom/`.  The primary bootstrap
uses 5,000 complete-respondent draws and is deterministic under the frozen seed.

## Interpretation

The primary residualizer is respondent-cross-fitted, task-specific additive
WLS.  The primary covariance is an unstructured 8 by 8 residual second moment
weighted by squared respondent survey weights.  Compound symmetry is required
as a robustness check.  Product-uniform task assignment and ordered uniform
sampling without replacement are both required because the observed data have
no repeated profiles within respondent, while the original platform transition
law is unavailable.

The frozen pass rule is an efficient/ordinary oracle variance ratio no larger
than 0.75 and a conditional-bootstrap 97.5th percentile below 0.85.  This
threshold came from the theory memo before this implementation; the freeze is a
stage-gate record, not a prospective preregistration, because this dataset had
already been examined in the broader project.
