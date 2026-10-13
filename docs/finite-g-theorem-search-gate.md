# Finite-cluster theorem search: global versus marginal cross-fit weighting

**Audit date:** 2026-08-28  
**Decision:** **FAIL as a route back to a stand-alone methods paper.**  
**Scope of the failure:** fixed profile support, a fixed number of tasks and
folds, and regular covariance or nuisance learning.  
**Preserved result:** exact two-fold variance identities, a reproducible
finite-support enumerator, and a precise statement of the only possible
nonregular escape route.

## Question

Gate G optimized a marginal train/evaluate fold after adding the variance of a
cross-fitted mean learner. The proposed new theorem would instead optimize the
variance or MSE of the complete cross-fitted estimator, allowing for the fact
that observations used to evaluate one fold are reused to train other folds'
nuisance functions and covariance weights.

The novelty test asks three separate questions:

1. Does the complete-estimator optimum differ from the marginal-fold optimum?
2. Does that difference survive for the intended Gaussian covariance learner?
3. If it survives somewhere, is deriving and estimating it a new, consequential
   paper-level result rather than standard higher-order GMM/DML theory?

The answers are, respectively: **sometimes in the abstract, no in the intended
Gaussian case, and no in the regular fixed-support regime.**

## 1. Exact two-fold identity for learned weights

Let independent fold datasets be \(\mathcal D_1\) and \(\mathcal D_2\). From a
generic fold define a centered evaluation vector \(Z\) and a calibrated learned
weight \(H=H_\lambda(\mathcal D)\). The honest two-way estimator has error

\[
\widehat\theta_\lambda-\theta
=
\frac12\{H_2^\mathsf T Z_1+H_1^\mathsf T Z_2\}.
\]

Write

\[
\Sigma_Z=E(ZZ^\mathsf T),\qquad
Q_H=E(HH^\mathsf T),\qquad
K=E(ZH^\mathsf T).
\]

Direct expansion gives the exact unconditional MSE

\[
\boxed{
E(\widehat\theta_\lambda-\theta)^2
=
\frac12\left\{
\operatorname{tr}(\Sigma_ZQ_H)+\operatorname{tr}(K^2)
\right\}.
}
\]

The first trace is one-half of the marginal train/evaluate risk. The second is
the covariance between the two reversed directions. It is the only place in
this minimal model where optimizing the complete estimator can differ from
optimizing a marginal fold.

This identity is useful, but it is a symmetrized split-sample or order-two
kernel calculation—not a factorial efficiency principle.

## 2. Why the crossed term vanishes for Gaussian covariance learning

In the intended covariance-learning model, conditional on assignments:

- the evaluation influence \(Z(\varepsilon)\) is linear, hence odd, in the
  centered Gaussian error;
- a residual covariance statistic \(S(\varepsilon)\) is quadratic, hence even;
  and
- the learned weight is any measurable function \(H=h_\lambda(S)\), including
  inversion, shrinkage, and deterministic positive-definite projection.

Therefore

\[
E\{Z(\varepsilon)H(\varepsilon)^\mathsf T\}=0
\]

by sign symmetry. For the ordinary Gaussian sample mean and covariance this
also follows from their independence. Hence \(K=0\) and

\[
E(\widehat\theta_\lambda-\theta)^2
=
\frac12\operatorname{tr}(\Sigma_ZQ_{H,\lambda}).
\]

Thus

\[
\boxed{
\arg\min_\lambda R_{\mathrm{complete}}(\lambda)
=
\arg\min_\lambda R_{\mathrm{marginal}}(\lambda)
}
\]

exactly in the Gaussian covariance-learning case. There is no missing global
weighting objective to derive there.

For the Gaussian linear model \(Y=b\theta+\varepsilon\), with calibrated
\(b^\mathsf TH=1\), the stronger decomposition is

\[
E(\widehat\theta-\theta)^2
=
\frac{1}{2n\,b^\mathsf T\Sigma^{-1}b}
+
\frac1{2n}E\{(H-h^*)^\mathsf T\Sigma(H-h^*)\},
\qquad
h^*=\frac{\Sigma^{-1}b}{b^\mathsf T\Sigma^{-1}b}.
\]

Random covariance learning can add finite-sample cost, but it cannot beat the
oracle or create a different complete-estimator optimum in this model.

## 3. Exact factorial enumeration

The executable check uses the smallest faithful Gate G design:

- two repeated tasks;
- independently randomized binary profiles;
- two equal respondent folds;
- task-specific saturated cell means learned on the opposite fold;
- a scalar target averaging the two task-specific factorial contrasts;
- supported training assignments; and
- correlated Gaussian task errors integrated analytically.

For every supported assignment pair, the code constructs the exact outcome
loading of both one directed fold estimate and the complete two-fold average.
It then integrates the Gaussian outcomes and forms the complete quadratic risk
over

\[
M=\begin{pmatrix}1/2&a_{12}\\a_{21}&1/2\end{pmatrix}.
\]

The result is stronger than equality of rounded optimizers:

\[
\boxed{Q_{\mathrm{complete}}=\tfrac12Q_{\mathrm{marginal}}.}
\]

| Respondents per fold | Supported states per fold | Ordered fold pairs | Maximum Gram discrepancy |
|---:|---:|---:|---:|
| 2 | 4 | 16 | 0 |
| 3 | 36 | 1,296 | \(1.33\times10^{-15}\) |
| 4 | 196 | 38,416 | \(4.11\times10^{-14}\) |

Consequently, the exact supported marginal and complete fixed-operator
minimizers coincide for every positive-definite two-task covariance matrix.
The complete risk is exactly one-half of the marginal risk.

Gate G's closed \(\Sigma+\Gamma_n\) formula can still differ slightly from the
exact supported optimum because it uses unconditional binomial count moments
instead of conditioning both folds on support. At error correlation 0.70, its
excess complete risk was 10.80%, 0.68%, and 0.06% for fold sizes 2, 3, and 4.
This discrepancy rapidly disappears and comes from the support convention—not
from marginal rather than global optimization.

## 4. Closed-form balanced specialization

A simpler balanced-slope version makes the rate transparent. Let

\[
M=\begin{pmatrix}\omega_1&a\\b&\omega_2\end{pmatrix},
\qquad
\Sigma=\begin{pmatrix}\sigma_{11}&\sigma_{12}\\
\sigma_{12}&\sigma_{22}\end{pmatrix},
\qquad G=2n.
\]

The exact two-fold variance is

\[
V_G(M)
=
\frac1G\operatorname{tr}(M^\mathsf T\Sigma M)
+
\frac{2}{G^2}(\sigma_{11}a^2+\sigma_{22}b^2).
\]

The complete and marginal finite-sample optima are the same:

\[
a_G=-\frac{\sigma_{12}\omega_2}
{\sigma_{11}(1+2/G)},
\qquad
b_G=-\frac{\sigma_{12}\omega_1}
{\sigma_{22}(1+2/G)}.
\]

Relative to the first-order optimizer \((a_\infty,b_\infty)\), the absolute
risk improvement is

\[
V_G(M_\infty)-V_G(M_G)
=
\frac{4}{G^3(1+2/G)}
\left(
\frac{\sigma_{12}^2\omega_2^2}{\sigma_{11}}
+
\frac{\sigma_{12}^2\omega_1^2}{\sigma_{22}}
\right).
\]

It is third order. With unit marginal variances, correlation 0.80, and equal
task weights, the relative risk reductions are approximately 0.255% at
\(G=50\), 0.067% at \(G=100\), 0.011% at \(G=250\), and 0.0007% at
\(G=1000\). This analytically explains Gate G's negligible point-risk change.

## 5. A generic counterexample exists, but it does not rescue the paper

The crossed trace need not always be zero. For example, let a fold contain a
target outcome, a known-zero control, and a learner statistic correlated with
the control error. Consider a calibrated complementary-fold control weight

\[
H_t=(1,\ 1-t+t\bar U)^\mathsf T,\qquad 0\le t\le1,
\]

with unit variances and \(\operatorname{Cov}(E_2,U)=c\). Then

\[
R_{\mathrm{marginal}}(t)
=
\frac{1+(1-t)^2}{2n}+\frac{t^2}{2n^2},
\]

whereas

\[
R_{\mathrm{complete}}(t)
=
\frac{1+(1-t)^2}{2n}
+\frac{t^2(1+c^2)}{2n^2}.
\]

The optima are genuinely different:

\[
t_{\mathrm{marginal}}=\frac{n}{n+1},
\qquad
t_{\mathrm{complete}}=\frac{n}{n+1+c^2}.
\]

But the optimizer difference is \(O(n^{-1})\), and the gain from reoptimizing
rather than merely evaluating the missing term is \(O(n^{-3})\). This is a
generic higher-order split-sample effect. It is not specific to factorial
experiments, vanishes for the intended even Gaussian covariance learner, and
is too small under the project's observed regime.

## 6. Reducibility to the closest literature

| Piece of the proposed theorem | Closest primary result | Verdict |
|---|---|---|
| Learn grouped-data weights by minimizing target-estimator variance | Young and Shah (2024) define the sandwich variance as a loss, learn weights with cross-fitting/DML2, prove convergence of the empirical minimizer, and obtain asymptotic minimum variance within a weight class. | Direct overlap at first order |
| Cross-fitted nuisance-product remainder | Newey and Robins (2018) give fast, explicit quadratic split-sample remainder results for average linear functionals. | Established structure |
| Higher-order effect of estimating an optimal weighting matrix | Newey and Smith (2004) derive higher-order GMM expansions that include estimated-weight contributions to bias and variance. | Established principle |
| Dependence created by reusing data across several splits | Fava (2025) proves a general split-sample estimator CLT accounting for reuse across splits. | Direct overlap for inference |
| Fold-dependent second-order DML bias and MSE | Velez (2026) gives explicit DML2 linear and quadratic first-step corrections, allows \(2\le K_n\le n\), and derives \(K\)-dependent second-order bias and MSE. | Decisive overlap |
| Two-fold learned covariance in multivariate estimating equations | Zhou and Zhang (2026) cross-fit an estimated covariance function and prove oracle-equivalent asymptotic inference. | Strong adjacent overlap |
| Fixed-support carryover restrictions and covariance-weighted estimation | Jiang and Ding (2026) provide the general crossover restricted-WLS formulation. | Direct domain overlap |

Young and Shah are especially close. Their scalar population sandwich loss is
the asymptotic variance after normalizing the estimating equation. Our
calibration constraint performs that normalization directly, leaving
\(E(H^\mathsf T\Sigma H)\) as the same objective in factorial coordinates.
Adding the cross-fitted mean-error covariance is a higher-order modification of
that loss, not a new weighting principle.

Velez closes the apparent overlap-theory gap. The paper's DML2 expansion
contains explicit linear and quadratic first-step corrections carrying the
fold dependence and supplies a second-order MSE approximation. Relabeling the
nuisance as the mean and covariance/weight learners produces the same type of
terms needed here.

## 7. Gate results

| Gate | Required result | Outcome |
|---|---|---|
| Fixed-operator separation | Complete and marginal optima differ in the faithful two-fold factorial model. | **Fail:** their full Gram matrices are proportional. |
| Intended learner separation | Gaussian covariance learning produces a nonzero crossed term. | **Fail:** odd/even symmetry forces it to zero. |
| Mathematical existence | Some regular learned-weight example has different optima. | Pass, but only through a generic second-order term. |
| Consequentiality | Reoptimization yields meaningful finite-sample improvement. | **Fail:** gain is third order; Gate G found about 0.03%. |
| Prior-art survival | The overlap and learned-weight expansion is not reducible to existing theory. | **Fail:** Young--Shah plus Newey--Smith, Newey--Robins, Fava, and Velez cover the hard steps. |

## Decision

**Stop the fixed-support finite-\(G\) theorem route.** It cannot support a
stand-alone full methods paper. Preserve the exact loading, count correction,
and enumeration as diagnostic or supplementary material.

The only plausible theoretical escape is a genuinely different triangular-
array problem with a growing number of profile cells:

\[
M_G/G\longrightarrow\kappa\in(0,\infty),
\]

so that empty cells persist, the saturated nuisance does not converge to a
deterministic function, and standard smooth fixed-dimensional expansions do
not apply. A credible result would need an occupancy-aware limiting risk,
every overlap and fallback term, its calibrated minimizer, and uniform
clustered inference.

That would be a many-profile occupancy paper, not a general factorial-DDML
paper. It should not be pursued until a small precommitted screen shows a
material gain—at least roughly 5% in a meaningful sparse-support region—over
ordinary, sandwich-loss, and conditional-GLS alternatives. The existing 0.03%
gain provides no reason to invest in it yet.

## Reproduction

Run:

```sh
Rscript analysis/theorem_gate/run_two_fold_size_grid.R
```

The implementation and exact results are in:

- `analysis/theorem_gate/two_fold_exact_weight_search.R`;
- `analysis/theorem_gate/run_two_fold_size_grid.R`;
- `analysis/theorem_gate/two_fold_exact_weight_size_grid.csv`;
- `analysis/theorem_gate/FINDINGS.md`; and
- `analysis/theorem_gate/README.md`.

The calculation does not modify or reuse Gate E2/G result files.

## Primary sources

- Young and Shah (2024), [Sandwich boosting for accurate estimation in
  partially linear models for grouped
  data](https://doi.org/10.1093/jrsssb/qkae032).
- Newey and Smith (2004), [Higher order properties of GMM and generalized
  empirical likelihood
  estimators](https://doi.org/10.1111/j.1468-0262.2004.00482.x).
- Newey and Robins (2018), [Cross-fitting and fast remainder rates for
  semiparametric estimation](https://arxiv.org/abs/1801.09138).
- Fava (2025), [Training and testing with multiple splits: a central limit
  theorem for split-sample estimators](https://arxiv.org/abs/2511.04957).
- Velez (2026), [Debiased machine learning with many cross-fitting
  folds](https://arxiv.org/abs/2411.01864).
- Zhou and Zhang (2026), [Hypothesis testing for penalized estimating equations
  with cross-fitted covariance
  calibration](https://arxiv.org/abs/2604.05055).
- Jiang and Ding (2026), [Principled analysis of crossover designs: causal
  effects, efficient estimation, and robust
  inference](https://arxiv.org/abs/2511.09215).
