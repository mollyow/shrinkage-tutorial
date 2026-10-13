# Gate G decision: finite-sample nuisance-remainder correction

**Formal decision: FAIL the proof-of-concept production-advance gate. Retain
the exact finite-sample accounting as supporting theory; do not claim that
learned covariance or a finite-$G$ operator materially improves the point
estimator.**

Gate G is a clean failure, not an incomplete run. The exact theory passed, all
5,000 fresh datasets and 300,000 method-coordinate rows were completed, and
source/configuration hashes matched. Two prespecified performance components
failed, so the project does not advance to a larger Gate G production study.

## What was tested

Gate G addressed the specific Gate E2 failure: a cross-fitted saturated mean
is shared by every respondent in a validation fold, creating a second-order
variance term that the rowwise score sandwich does not explicitly propagate.
The gate implemented:

1. the exact outcome loading of a fixed-operator $K$-fold estimator;
2. assignment-conditional covariance using that loading;
3. exact count correction of out-of-fold saturated residual moments;
4. a marginal finite-$G$ task-mixing operator;
5. conditional saturated GLS as an oracle finite-support benchmark; and
6. honest two-way feasible GLS with exact saturated-residual degrees of
   freedom.

The protocol, theory, configuration, and executable decision are in
`docs/gate-g-protocol.md`, `docs/gate-g-theory.md`,
`analysis/feasible_eif/gate-g-screen.yml`, and
`analysis/feasible_eif/summarize_gate_g.R`.

## Integrity and support

- 5 scenarios × 1,000 replications: complete.
- Expected/observed result rows: 300,000 / 300,000.
- Numerical failures: 0.
- Maximum calibration residual: $1.01\times10^{-14}$.
- Source and configuration hashes: matched.
- Deterministic algebra and support checks: 35 / 35 passed.
- Outer-training empty cells: 3 / 25,000 fold fits.
- Auxiliary residual-df fallbacks: 41 / 25,000 fold fits.
- Honest split ordinary-score fallbacks: 2 / 19,998 recorded split fits.

An initial production launch was stopped after its first cell because an
auxiliary inner-fold support case had not been given an executable fallback.
That incomplete launch is preserved under
`results/feasible-eif/gate-g-screen-aborted-support-bug`. Before a decision,
the already-derived assignment-only residual-df fallback was implemented,
the 15 affected seeds and the singleton case were retested, and the whole
screen was restarted from the original data seeds. No scenario, threshold, or
replication count changed.

## Prespecified gate results

| Component | Result | Prespecified rule | Decision |
|---|---:|---:|---|
| Exact oracle variance ratio | 0.952–1.018 | 0.95–1.05 | Pass |
| Exact oracle 95% coverage | 0.951–0.956 | 0.93–0.97 | Pass |
| Corrected scaling discrepancy | 0.199 | at most 0.10 | **Fail** |
| Local AMSE ratio | 0.972–1.084 | 0.95–1.05 | **Fail** |
| Covariance Frobenius-MSE reduction | 11.3%–22.6% | at least 10% | Pass |
| Count-corrected feasible safety | all count/raw ≤ 1.001 | at most 1.05 and geometric mean improves | Pass |
| Numerics and respondent isolation | zero failures; isolated | required | Pass |

The failed local AMSE rows were the $G=1000$ oracle $H^*$, 0.50 blend,
and 0.75 blend, with ratios 1.075, 1.074, and 1.084. The post-decision
bootstrap does not provide a discretionary rescue: the $H^*$ interval was
[1.013, 1.139], and the signed corrected scaling difference had interval
[0.040, 0.358] in first-order-variance units. Those diagnostics cannot and do
not change the frozen decision.

## What the gate nevertheless establishes

### 1. The Gate E2 mechanism is understood

For fixed fold-specific weights, the complete cross-fitted estimator is
exactly linear in all cluster outcomes. Propagating those loadings gives the
right finite-sample conditional variance. This is not merely an asymptotic
rate statement: the no-carryover empirical/propagated variance ratios and
coverage passed in all three sample-size cells.

### 2. The residual-moment correction works

The raw out-of-fold residual moment distorts task covariance because estimated
cell means affect diagonal and off-diagonal moments differently. The count
correction reduced covariance Frobenius MSE by 22.6% at $G=250$ and 11.3% at
$G=500$. Its practical benefit was interval calibration: raw plug-in
coverage was 0.980, 0.971, and 0.966 at $G=250,500,1000$, while corrected
plug-in coverage was 0.956, 0.951, and 0.956.

### 3. Better covariance estimation does not materially improve the point fit

The feasible count-corrected and raw $H^*$ estimators were virtually
identical. Their geometric-mean variance ratios to the first-order oracle were
1.19914 and 1.19954, an improvement of about 0.03%. The oracle finite-$G$
operator likewise changed null-cell empirical variance by less than 0.1%
relative to asymptotic $H^*$.

This distinction matters. Gate G finds value in finite-sample **risk and
standard-error accounting**, not in a new practically superior point
estimator.

### 4. Finite-support GLS recovers some nuisance cost, but is not general ML

Honest split, df-corrected GLS reduced null-cell variance relative to
cross-fitted oracle $H^*$ by approximately 13.8%, 8.6%, and 4.9% at
$G=250,500,1000$. Full oracle GLS did better. But this is conditional
Gauss–Markov estimation of an estimable saturated finite-support mean. It is
support-limited, relies on the restricted no-carryover mean model, and is not a
new DDML construction.

Under the least-favourable local departure, the efficiency gain was not
uniformly useful. Oracle $H^*$'s MSE was essentially equal to ordinary
taskwise scoring at $G=250$ and 5.7% worse at $G=1000$. Honest split GLS
was 6.7% better at $G=250$ but 2.8% worse at $G=1000$.

## Paper consequence

Gate G does not support a stand-alone paper claiming a new adaptive or ML
estimator for factorial experiments. Generic influence propagation,
degrees-of-freedom correction, and conditional GLS are established ideas. The
factorial closed forms and exact repeated-cluster implementation are useful,
but they are supporting results.

The honest claim is narrower:

- within-respondent covariance can create large *oracle* headroom under a
  strong no-carryover restriction;
- learning that covariance more accurately improves plug-in uncertainty
  quantification but adds essentially no point-estimation value here;
- empirical-design GLS can recover part of the nuisance cost when the complete
  finite support is estimable; and
- the same covariance adaptation can lose its advantage under small
  carryover departures.

Accordingly, unrestricted taskwise AIPW remains the primary general estimator.
The exact loading variance, count/df covariance corrections, and finite-support
GLS belong in a supplement or software/diagnostic section. A full methods
paper should proceed only if a focused prior-art audit finds novelty in the
*combined* target-weighted EIF, carryover robustness frontier, and structured
robust score—not through another performance simulation of covariance ML.

## Reproducible evidence

- Formal decision: `results/feasible-eif/gate-g-screen/gate_g_decision.yml`
- Component table: `results/feasible-eif/gate-g-screen/gate_g_components.csv`
- Main result bundle:
  `results/feasible-eif/gate-g-screen/gate_g_simulation_results.rds`
- Post-decision uncertainty:
  `results/feasible-eif/gate-g-screen/gate_g_posthoc_monte_carlo_uncertainty.csv`
- Method performance:
  `results/feasible-eif/gate-g-screen/gate_g_method_performance.csv`
- Support diagnostics:
  `results/feasible-eif/gate-g-screen/gate_g_support_summary.csv`
