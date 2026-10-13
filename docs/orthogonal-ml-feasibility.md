# Orthogonal ML for factorial experiments: feasibility and novelty audit

**Status:** stage-gate memo, 2026-08-26  
**Decision:** do not yet build or describe a new DDML estimator.

## Executive verdict

There is a useful analysis here, but the obvious methodological claim does not
survive an adversarial novelty check.

- **Red — standalone new estimator or test.** The proposed construction is a
  known-randomization augmented/Riesz score for a vector of factorial
  coefficients, followed by the standard unbiased U-statistic for the squared
  norm of a mean. Its off-null influence function, null degeneracy, and
  multiplier-bootstrap problem are established statistical machinery.
- **Red — “DDML for factorial experiments” as a broad first claim.** Generic
  DML already covers randomized multi-treatment contrasts, and recent work
  combines flexible learners and debiasing in conjoint or combinatorial
  experiments.
- **Yellow — design-specific software and theory.** A contribution may remain
  in a careful interface for target-versus-assignment weighting, categorical
  functional-ANOVA geometry, restricted profile supports, repeated respondent
  clusters, and high-dimensional order-wise tests. That would need genuinely
  new cluster/cross-fit theory or decisive empirical gains; it is not yet a
  methods-paper contribution.

The planned novelty gate therefore fails. Consistent with the stage-gated
plan, no package estimator was added at this gate. A later, isolated precision
screen tests the narrower software/benchmark possibility; its results and
updated decision are in
[`narrow-orthogonal-ml-paper.md`](narrow-orthogonal-ml-paper.md). That route has
now also failed its oracle-headroom gate; see the
[`executed gate decision`](narrow-ml-gate-decision.md). The derivation
below identifies exactly what that narrower project must improve on.

## 1. Two different interaction-energy estimands

Let \(A\) be a randomized profile, \(X\) pre-treatment respondent information,
and

$$
m(a,x)=\mathbb E(Y\mid A=a,X=x).
$$

Fix a target profile distribution \(\mu\), and let
\(\Phi_k(a)\in\mathbb R^{d_k}\) be a target-orthonormal basis for the order-
\(k\) factorial subspace. Define the respondent-conditional order coefficients

$$
b_k(x)=\mathbb E_{A\sim\mu}\{\Phi_k(A)m(A,x)\}.
$$

There are two natural, non-equivalent targets.

### Population-average surface energy

$$
\beta_k=\mathbb E_X\{b_k(X)\},
\qquad
Q_k^{\mathrm M}=\|\beta_k\|_2^2.
$$

This is the squared order-\(k\) component of the **marginal response surface**
\(\bar m(a)=\mathbb E_X\{m(a,X)\}\). It asks whether an interaction survives
averaging over respondents.

### Conditional or heterogeneous surface energy

$$
Q_k^{\mathrm C}=\mathbb E_X\{\|b_k(X)\|_2^2\}.
$$

This asks whether order-\(k\) interactions exist within respondents or
covariate strata, even if they cancel in the population average. The two obey

$$
Q_k^{\mathrm C}
=Q_k^{\mathrm M}+\operatorname{tr}\{\operatorname{Var}(b_k(X))\}.
$$

This distinction is substantive, not cosmetic. Opposing conditional
interactions can make \(Q_k^{\mathrm M}=0\) while
\(Q_k^{\mathrm C}>0\). A paper must therefore say “population-average
interaction energy” when it targets \(Q_k^{\mathrm M}\); it cannot call that
quantity total interaction strength.

For a product target distribution, these quantities are factorial versions of
functional-ANOVA/Sobol component energies. Functional ANOVA for dependent
inputs is established, and a recent paper gives an exact construction for
categorical and restricted-support inputs
([Hooker 2007](https://doi.org/10.1198/106186007X237892),
[Chastaing, Gamboa, and Prieur 2012](https://doi.org/10.1214/12-EJS749),
[Ferrere et al. 2026](https://arxiv.org/abs/2603.02673)).

## 2. What orthogonal learning gives for the marginal target

Suppose profile assignment has known probability \(q(a\mid x)\), while the
estimand averages profiles under \(\mu(a)\). On an evaluation respondent that
was not used to train a response-surface learner \(\widehat m\), the basic
vector score is

$$
Z_{ik}(\widehat m)
=
\mathbb E_{A\sim\mu}\{\Phi_k(A)\widehat m(A,X_i)\}
+
\frac{\mu(A_i)}{q(A_i\mid X_i)}\Phi_k(A_i)
\{Y_i-\widehat m(A_i,X_i)\}.
$$

For repeated tasks, this becomes a respondent-level average of task scores;
arbitrary dependence is then allowed within a respondent, while respondents
remain the independent units. Conditional on the training sample,

$$
\mathbb E\{Z_{ik}(\widehat m)\mid\text{training}\}=\beta_k
$$

for **any** independently trained learner, assuming the randomization
probabilities and target weights are correct. Thus ML can improve precision,
but identification and bias correction come from the known experiment.

This is a vector of ordinary augmented inverse-probability/Riesz scores. The
factorial basis makes the target interpretable; it does not create a new DML
principle. Related work already applies flexible learners and debiasing to
conjoint, combinatorial, and multiple-treatment experiments
([Acharya, Hainmueller, and Xu 2026](https://arxiv.org/abs/2604.10845),
[Ye et al. 2025](https://doi.org/10.1287/mnsc.2024.04625),
[Xiang et al. 2025](https://arxiv.org/abs/2505.12617)).

## 3. The proposed energy estimator is a classical mean-norm statistic

Let \(I_v\) be a validation fold containing \(g_v\) independent respondents,
all scored with the same learner trained outside that fold. Then

$$
\widehat Q_{k,v}
=
\frac{1}{g_v(g_v-1)}
\sum_{\substack{i,j\in I_v\\i\ne j}}
Z_{ik}^{\mathsf T}Z_{jk}
$$

is exactly conditionally unbiased for \(Q_k^{\mathrm M}\). It is also exactly

$$
\widehat Q_{k,v}
=\|\bar Z_{k,v}\|_2^2
-\frac{\operatorname{tr}(S_{k,v})}{g_v},
$$

where \(S_{k,v}\) is the usual sample covariance matrix. In words: square the
estimated coefficient vector and subtract its diagonal noise bias.

That statistic is the standard unbiased estimator of a high-dimensional mean
norm, equivalently a linear-kernel MMD U-statistic
([Chen and Qin 2010](https://doi.org/10.1214/09-AOS716),
[Gretton et al. 2012](https://www.jmlr.org/papers/v13/gretton12a.html),
[Li 2023](https://doi.org/10.1016/j.jmva.2023.105183)). In a balanced,
homoskedastic factorial experiment it reduces to the classical order-specific
ANOVA sum of squares with its noise degrees of freedom removed. Testing
\(Q_k^{\mathrm M}=0\) is then the usual omnibus null that all order-\(k\)
coefficients vanish.

An algebra check using arbitrary simulated score vectors reproduced the two
forms above to machine precision. Both algebra checks are preserved in
[`analysis/check_orthogonal_ml_derivations.R`](../analysis/check_orthogonal_ml_derivations.R).

## 4. A cross-fitting trap

It is safe to form respondent pairs **within** a validation fold because both
scores use a nuisance learner trained independently of both respondents. It is
not generally safe to pool every pair from ordinary K-fold cross-fitting.
For respondents in different validation folds, each respondent can appear in
the training data used to score the other. That destroys the simple
conditional-unbiasedness argument.

A complete finite example confirms the problem. There are four one-task
respondents, \(A_i\in\{-1,+1\}\), \(Y_i=A_i\), two folds of two respondents,
and a deliberately unstable learner that predicts the mean treatment in its
training fold. Exhaustive enumeration of all \(2^4=16\) assignments gives:

| Construction | Exact expectation | Truth |
|---|---:|---:|
| Average of the two within-fold pair statistics | 1.000000 | 1 |
| U-statistic using all six respondent pairs | 1.166667 | 1 |
| One representative cross-fold pair | 1.250000 | 1 |

So “all-pairs cross-fitted U-statistic is exactly unbiased for any learner” is
false. Honest options are an independent training/evaluation split, foldwise
U-statistics with only within-fold pairs, or extra stability and asymptotic
arguments for the learned nuisance fits.

## 5. Inference is also known—and difficult at zero

Away from zero, the first-order influence function for
\(Q_k^{\mathrm M}=\beta_k^{\mathsf T}\beta_k\) is

$$
2\beta_k^{\mathsf T}(Z_{ik}-\beta_k).
$$

At \(\beta_k=0\), this influence function is identically zero. For fixed
\(d_k\), the scaled U-statistic has a weighted centered chi-square limit,

$$
G\widehat Q_k
\ \Longrightarrow\
\sum_{r=1}^{d_k}\lambda_r(\chi^2_{1,r}-1),
$$

where \(\lambda_r\) are eigenvalues of the score covariance. With growing
\(d_k\), normal approximations require spectral or effective-rank conditions.
Wild and multiplier bootstrap methods for degenerate U-statistics are already
available
([Leucht and Neumann 2013](https://doi.org/10.1016/j.jmva.2013.03.003),
[Chwialkowski, Sejdinovic, and Gretton 2014](https://papers.nips.cc/paper/5452-a-wild-bootstrap-for-degenerate-kernel-tests)).

More generally, efficient estimation of squared conditional means/Sobol
functionals and the loss of ordinary first-order inference at the zero
boundary are established
([Da Veiga and Gamboa 2013](https://doi.org/10.1080/10485252.2013.784762),
[Verdinelli and Wasserman 2024](https://www.jmlr.org/papers/v25/22-0801.html),
[Klein et al. 2024](https://arxiv.org/abs/2407.15468),
[Hudson 2023](https://arxiv.org/abs/2306.07492),
[Sanchez-Becerra 2023](https://arxiv.org/abs/2306.03363)).

For fixed \(d_k\), a joint Wald or score test of the regular vector null
\(\beta_k=0\) is simpler than constructing confidence intervals for the
boundary parameter \(Q_k^{\mathrm M}\). The energy statistic becomes most
compelling when the order space is high-dimensional, when its magnitude is
itself substantively interesting, or when a direction-free test has useful
power.

The conditional energy \(Q_k^{\mathrm C}\) is a different and harder target.
Its ordinary one-step expansion leaves a second-order remainder of the form

$$
-\mathbb E\{\|\widehat b_k(X)-b_k(X)\|_2^2\},
$$

so root-\(G\) inference generally requires approximately an
\(o_p(G^{-1/4})\) nuisance rate. Known randomization removes confounding; it
does not remove this quadratic-functional difficulty.

## 6. Proposition-level novelty matrix

| Proposed claim | Verdict | Why |
|---|---|---|
| A DDML score for factorial coefficients or fixed AMIEs | Red | A direct specialization of known-randomization AIPW/Riesz scores. |
| Exact unbiasedness of an independently evaluated coefficient score for any learner | Red | Useful design fact, but immediate from randomization and augmentation. |
| An unbiased estimator of order energy based on cross-respondent inner products | Red | The standard mean-norm/linear-kernel U-statistic. |
| Off-null asymptotic normality and a degenerate null limit | Red | Standard quadratic-functional and U-statistic theory. |
| A generic multiplier or wild bootstrap at the null | Red | Existing degenerate-kernel bootstrap machinery applies. |
| Target-weighted categorical factorial geometry | Yellow | Valuable interface and interpretation, but close categorical/restricted-support functional ANOVA now exists. |
| Repeated-respondent cluster inference with overlapping cross-fit learners and growing order dimension | Yellow | Potentially nontrivial only with precise stability, dimension, and local-power results. |
| Unified software for assignment/target mismatch, restricted profiles, clustering, and ML adjustment | Yellow | Potential applied-methods or software contribution if it changes empirical performance or practice. |

## 7. The only route worth testing next

If the project continues, it should be framed as **design-aware orthogonal
learning for causal factorial estimands**, not as the invention of DDML or a
new interaction-energy statistic.

The smallest honest prototype would do three things:

1. Return target-weighted order coefficient vectors and covariance matrices
   from respondent-isolated, learner-agnostic augmented scores.
2. Report marginal order energy as a derived summary, allowing its unbiased
   estimate to be negative rather than truncating it at zero.
3. Compare a single honest split with within-fold U-statistics; never silently
   construct all cross-fold pairs.

Only then run a small precision benchmark. Compare unadjusted randomization
scores, linear/ridge adjustment, random forests or boosting, and an oracle
nuisance model under nonlinear respondent heterogeneity, unequal assignment,
restricted support, and repeated tasks. Judge ML on variance, RMSE, coverage,
power, and computation—not merely predictive accuracy.

Proceed to a paper only if at least one design-specific result is both new and
material, for example:

- a valid cluster multiplier procedure with learned, overlapping cross-fit
  nuisances and growing \(d_k\), including local-power guarantees; or
- stable, consequential precision/power gains over unadjusted and simple
  regression adjustment in realistic conjoint designs.

If neither appears, fold these ideas into the existing package as an
estimand-aware analysis option and present them as established methods brought
together for factorial experiments. That is potentially useful software, but
not a new general statistical method.

## Gate record

| Gate | Result |
|---|---|
| Distinct estimand identified | Pass, but two estimands must be separated. |
| New orthogonal score | Fail. |
| New quadratic estimator | Fail. |
| New boundary inference idea | Fail. |
| Exact arbitrary-learner all-pairs cross-fit claim | Fail; finite counterexample. |
| Plausible narrower applied/software route | Conditional pass. |
| Authorization to build a new estimator prototype | **No—stop after the feasibility memo.** |
