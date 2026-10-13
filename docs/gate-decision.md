# Stage 3 proof-of-concept gate

**Decision date:** 2026-08-26  
**Current route:** comparison/software paper; the full new-methods claim is paused.

## Decision

The corrected proof-of-concept screen does not justify proceeding to the full
methods-paper simulation or the substantive application.  This is a stage-gate
decision, not a claim that monotone order shrinkage can never help.

The final optimizer-confirmation pilot used:

- the two core geometries (eight binary factors and six three-level factors),
  each through order three;
- $n/p\in\{0.5,1\}$, signal-to-noise ratio 2, dense effects, and steep or mild
  declining per-degree-of-freedom order scales;
- 25 independently regenerated datasets per cell (eight cells and 200 paired
  replications total);
- common respondent/row folds for ordinary, unrestricted order-specific, and
  monotone order-specific ridge; and
- exact target-profile risk, with every profile weighted by the prespecified
  target measure.

The aggregate metric is the ratio of geometric mean paired risks,

$$
\exp\left\{\frac{1}{C}\sum_{c=1}^C
  \frac{1}{R_c}\sum_{r=1}^{R_c}
  \log\left(\frac{R_{M,cr}}{R_{U,cr}}\right)\right\},
$$

so each scenario cell receives equal weight even when production replication
counts differ.

| Check | Required | Observed | Result |
|---|---:|---:|---|
| Geometric mean monotone/unrestricted risk | $\leq0.95$ | 0.9874 | Fail |
| Monte Carlo 95% upper bound | $<0.98$ | 1.0067 | Fail |
| Primary cells won | $\geq75\%$ | 62.5% (5/8) | Fail |

The pilot was intentionally not marked production-eligible.  It contains no
flat or nonmonotone stress cells, so it cannot adjudicate the robustness
criteria.  There were no adaptive-fit errors after correcting the YAML-to-NLopt
numeric type boundary.

## Cell results

| Geometry | $n/p$ | Decline | Geometric risk ratio | Replication win rate |
|---|---:|---|---:|---:|
| 8 binary | 0.5 | mild | 0.9302 | 0.64 |
| 6 three-level | 0.5 | mild | 0.9728 | 0.64 |
| 8 binary | 1 | mild | 0.9581 | 0.48 |
| 6 three-level | 1 | mild | 0.9846 | 0.80 |
| 8 binary | 0.5 | steep | 1.0331 | 0.68 |
| 6 three-level | 0.5 | steep | 0.9961 | 0.40 |
| 8 binary | 1 | steep | 1.0247 | 0.48 |
| 6 three-level | 1 | steep | 1.0039 | 0.40 |

These results supersede the earlier coarse-tuner smoke results for deciding
whether to launch production.  All pre-2026 simulation outputs remain
non-evidentiary.

## Consequence

Under the prespecified sequence, the repository should now retain and develop:

1. the target-weighted, coding-invariant factorial geometry;
2. the common, unrestricted, and monotone solver implementation;
3. the honest cross-fitted AMIE implementation;
4. the benchmark and gate machinery; and
5. the candidate-recruitment analysis specification and locked estimands.

It should not currently claim that monotonicity is a generally new or
performance-improving shrinkage estimator.  A comparison/software paper can
instead ask when an order restriction helps, demonstrate the coding-invariant
implementation, and report the negative benchmark honestly.

A 1,000-replication confirmation may still be run if a publication decision
requires distinguishing a very small benefit from no benefit.  It cannot
rescue the original methods claim unless the prespecified 0.95, 0.98, and 75%
thresholds are all met on an eligible production run.

## Reproduction

```sh
Rscript simulation/run_simulations.R \
  --config config/simulation-optimizer-check.yml \
  --output results/simulations/optimizer-check/results.csv \
  --workers 3

Rscript simulation/summarize_gates.R \
  results/simulations/optimizer-check/results.csv \
  results/simulations/optimizer-check/gate-status.json \
  config/simulation-optimizer-check.yml
```

Generated pilot result files remain ignored.  They should be archived with
checksums only if this diagnostic evidence is included in a public release.
