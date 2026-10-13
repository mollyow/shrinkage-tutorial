# Gate E2 decision: carryover sensitivity and feasible adaptation

**Formal decision: HARD FAIL. Do not claim that the current feasible adaptive
procedure realizes the first-order risk bound at the frozen sample sizes.**

**Subsequent Gate E3 clarification:** the generic Gate E2 blend is a fixed
linear minimax rule for a prespecified sensitivity ball. It is not the
optimally adaptive nonlinear rule when the carryover bound is unknown.
Armstrong, Kline, and Sun (2025) already solve that Gaussian adaptation
problem. See `docs/gate-e3-theory.md`; the immutable Gate E2 decision is
unchanged.

This decision does **not** reject the Gate E1 efficient-influence-function,
efficiency-bound, exact-bias, or sensitivity geometry. It rejects the stronger
finite-sample claim made by the current feasible implementation and therefore
prevents a formal practical-value pass.

## Production provenance

The valid version-3 production run used the frozen 14-scenario design and 3,000
replications per scenario: 42,000 respondent-cluster simulations in total. All
42,000 completed. The source and configuration hashes matched exactly, every
planned replication ID and seed was present, and no required result or
diagnostic field was missing.

Two earlier partial runs were stopped and marked non-evidentiary after
predecision audits found an inner-selector leakage problem and then an
assignment-support edge case. Version 3 uses leave-two-fold covariance fitting,
assignment-independent outer folds, a prespecified empty-cell rule, and the
correct first-order variance for the saturated-mean assignment rescue. The
final 119 correctness tests include the two exact support-event seeds that had
failed in version 2. In valid production, only 2 of 15,000 outer folds in the
small-\(G\) scenario used the recorded one-cell fallback.

## Frozen component decision

Fifteen of seventeen hard components passed.

| Hard component | Result | Main evidence |
|---|---|---:|
| Configuration/source integrity | Pass | exact hash match |
| Exact carryover-bias formula | Pass | max standardized error 0.0605; max MC \(z=3.315\) |
| Local first-order AMSE | **Fail** | ratio range 1.0088–1.1098; required 0.90–1.10 |
| \(G=250/1000\) local scaling | **Fail** | max bias discrepancy 0.0208; variance discrepancy 0.3629; required both \(\le .15\) |
| Least-favourable risk shape | Pass | required reversal across \(r=.5,1,2,3\) |
| Ordinary primary inference | Pass | coverage 0.9450–0.9563 |
| Selected null/blind inference | Pass | coverage 0.9477–0.9497 |
| Oracle bias-aware coverage | Pass | minimum 0.970 |
| Hausman null/blind size | Pass | rejection 0.0460–0.0567 |
| Hausman power calibration | Pass | maximum absolute error 0.0118 |
| Hausman directional power shape | Pass | \(r=1\): 0.180; \(r=2\): at least 0.501 |
| Assignment-law exact decomposition | Pass | max standardized error 0.0383 |
| Combined assignment/carryover decomposition | Pass | max standardized error 0.0328 |
| Assignment-law rescue methods | Pass | frozen bias and coverage checks |
| Exact calibration | Pass | max residual \(1.39\times10^{-16}\) |
| Diagnostic completion | Pass | complete selector, tuning, departure, and Hausman records |
| Numerical completion | Pass | 0 failures in 42,000 replications |

The three AMSE rows outside the frozen interval were:

| Scenario | Method | Empirical/theoretical MSE |
|---|---|---:|
| least favourable, \(r=.5, G=500\) | \(\tau=.75\) | 1.1098 |
| least favourable, \(r=1, G=250\) | \(\tau=.50\) | 1.1073 |
| least favourable, \(r=1, G=250\) | \(\tau=.75\) | 1.1050 |

## Why the hard gate failed

The failure is a finite-sample nuisance-estimation remainder, not an error in
the first-order E1 formulas or in the frozen summarizer.

For the fully adaptive \(H^*\), feasible estimation produced the following
\(G\)-scaled variances in the paired \(r=1\) cells:

| | \(G=250\) | \(G=1{,}000\) | First-order limit |
|---|---:|---:|---:|
| Feasible saturated-mean \(H^*\) | 0.04175 | 0.03143 | 0.02845 |
| Excess over the limit | 0.01331 | 0.00298 | — |

The excess fell by a factor of 4.46 when \(G\) increased fourfold, the pattern
expected from an \(O(1/G)\) second-order term in scaled variance. The discrepancy
also increased monotonically with restriction strength: 0.027, 0.063, 0.134,
0.253, and 0.363 for \(H_0,\tau=.25,.50,.75,H^*\), respectively.

The frozen known-restricted-mean \(H^*\) control removes nuisance estimation
while preserving the local carryover bias. Its AMSE/theory ratios were
0.982–1.014, and its paired scaling discrepancy was only 0.0087. A post-gate
diagnostic using the corresponding known-mean fixed blends gave ratios
0.959–1.035. These controls strongly support the E1 first-order limit, but they
cannot reverse the formal failure of the feasible estimator.

The consequence is precise: the current procedure has valid clustered
inference in the frozen designs, but its first-order risk approximation is not
accurate enough at \(G=250\)–500 to support the claimed finite-sample efficiency
calculation.

## Practical evidence, not a formal practical decision

Because hard correctness failed, the protocol did not adjudicate practical
value. The following results are descriptive and must not be called a Gate E2
practical pass.

Four of five practical components met their thresholds:

- The oracle full-ball minimax blend passed every least-favourable risk check;
  its worst paired-bootstrap upper endpoint was 0.941.
- The selected adaptive estimator's null variance was 0.2468 of ordinary, with
  upper endpoint 0.2621.
- Feasible bias-aware primary coverage was at least 0.971.
- Structured lag intervals covered at 0.9637 and 0.9687 and had median lengths
  0.570 and 0.775 of the unrestricted-ball intervals.

The feasible generic blend failed its uniform regret criterion:

| Departure | Risk relative to the better of ordinary/selected | 95% paired bootstrap interval |
|---|---:|---:|
| Blind, \(r=2\) | 2.745 | [2.595, 2.912] |
| Lag, compound symmetry, \(r=1\) | 1.673 | [1.598, 1.754] |
| Lag, AR(1), \(r=1\) | 1.145 | [1.105, 1.185] |

This is not a failed ML optimization. The learned blend nearly equaled the
oracle full-ball rule: approximately 0.20 in the blind cell and 0.50 in both
lag cells. The problem is the objective. A full-ball rule must price the
least-favourable direction even when actual carryover is target-blind or only
weakly aligned. A direction-aware oracle instead used blends about 1.00, 1.00,
and 0.85 in these three cells.

The defensible interpretation is therefore not “ML has no value.” ML and
cross-fitting successfully learn covariance, nuisance functions, and the
specified robust blend. They do not reveal an otherwise unidentified local
carryover direction, so they cannot erase the robustness–efficiency tradeoff
without scientific structure.

## Paper decision and next gate

The current implementation is **not ready for a stand-alone full methods paper
that recommends automatic feasible adaptation**. Ordinary unrestricted
taskwise AIPW remains the confirmatory applied estimator.

A full paper remains plausible only if it adds both of the following:

1. **Higher-order feasibility.** Derive the finite-\(G\) nuisance-estimation
   term for cross-fitted factorial scores and construct either an exact
   calibration, bias/variance correction, or count-aware estimator. Prove a
   finite-sample or rate bound and test it on fresh, preregistered simulations.
2. **A factorial robustness–efficiency theorem.** Establish a LAN/constrained-
   risk or minimax-regret lower bound showing why no regular data-driven rule
   can uniformly mimic the better of restricted and unrestricted estimation
   across null, blind, and least-favourable local departures. Then give a
   constructive structured alternative based on a prespecified carryover
   subspace \(Z\), its sensitivity matrix \(K_Z\), valid intervals, and a
   feasible oracle-risk result.

That would support a paper framed around **efficient factorial inference and
the limits of automatic adaptation**, with scientifically structured carryover
as the constructive solution. Without those results, the appropriate output is
a theory/diagnostic note or a section in a broader factorial-inference paper,
not a new general-purpose ML estimator.

## Reproducibility

- Protocol: `docs/gate-e2-protocol.md`
- Frozen configuration: `analysis/feasible_eif/gate-e2-production.yml`
- Formal machine decision: `results/feasible-eif/gate-e2-production/gate_e2_decision.yml`
- Component table: `results/feasible-eif/gate-e2-production/gate_e2_components.csv`
- Post-gate diagnosis: `results/feasible-eif/gate-e2-production/gate_e2_postgate_diagnosis.yml`
- Diagnostic script: `analysis/feasible_eif/diagnose_gate_e2_failure.R`

The formal Gate E2 decision is immutable. Any higher-order estimator or
structured adaptation proposal must receive a newly versioned protocol and
fresh simulation seeds.
