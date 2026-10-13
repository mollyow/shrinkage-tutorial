# Gate G theory: the finite-sample cross-fit remainder

**Subsequent gate decision:** the 5,000-dataset proof-of-concept screen failed
its corrected-scaling and local-AMSE thresholds. The exact fixed-operator
variance, coverage, covariance correction, numerics, and leakage components
passed. See `docs/gate-g-decision.md`. The theory below survives as a
supporting result; the project does not advance to a larger Gate G run.

## Result in one sentence

The apparent Gate E2 variance failure is a finite-sample consequence of
reusing each fitted saturated mean across a validation fold: it is invisible
to the ordinary rowwise sandwich, but it has an exact outcome-loading
representation and can therefore be propagated without approximation for a
fixed score operator.

## Setup

For respondent $i$, let $Y_i\in\mathbb R^T$ collect the repeated outcomes,
$B_i=B(A_i)$ be the task-by-parameter design for the restricted saturated
own-profile mean, and

\[
Y_i=B_i\gamma+\varepsilon_i,
\qquad
E(\varepsilon_i\mid A_i)=0,
\qquad
\operatorname{Var}(\varepsilon_i\mid A_i)=\Sigma_i.
\]

The target is $\theta=L\gamma$. A calibrated residual weight $H_i$ obeys
the population restriction $E(B_i^\top H_i)=L^\top$. If fold $k$'s
training estimator is $\widehat\gamma_k$, the held-out score is

\[
q_{H,i}
=
L\widehat\gamma_k
+H_i^\top(Y_i-B_i\widehat\gamma_k).
\]

Its exact error decomposition is

\[
q_{H,i}-\theta
=
H_i^\top\varepsilon_i
+C_i(\widehat\gamma_k-\gamma),
\qquad
C_i=L-H_i^\top B_i.
\]

The second term is a product of validation-sample calibration error and
training-sample mean error. It is second order for the point estimator, but
all observations in a validation fold share the same fitted mean. Treating
their scores as independent therefore misses its finite-sample variance.

## Exact $K$-fold loading

For saturated cell means, write the fold-$k$ training error as

\[
\widehat\gamma_k-\gamma
=
\sum_{j\notin I_k}W_{kj}\varepsilon_j,
\]

where $W_{kj}$ is determined only by assignments, folds, and cell counts.
Let $D_k=\sum_{i\in I_k}C_i$. Then

\[
\widehat\theta_H-\theta
=
\frac1G\sum_{j=1}^G
\left[
H_j^\top+
\sum_{k:j\notin I_k}D_kW_{kj}
\right]\varepsilon_j.
\]

Defining the bracketed loading as $\Lambda_j^\top$ gives the exact
assignment-conditional covariance

\[
\operatorname{Var}(\widehat\theta_H\mid A,\mathcal I)
=
\frac1{G^2}\sum_{j=1}^G
\Lambda_j^\top\Sigma_j\Lambda_j.
\]

This formula retains every overlap between fold training samples. It is exact
when the fold-specific operators are fixed independently of outcomes. For a
learned operator, the same calculation is a frozen-operator diagnostic; a
second influence term for operator learning is needed for exact unconditional
variance.

## Count inflation in out-of-fold residual covariance

An out-of-fold saturated residual is the respondent error minus an
independently trained cell-mean error. Conditional on assignments and folds,
its task-$t$, task-$u$ second moment is

\[
E(\widetilde\varepsilon_{it}\widetilde\varepsilon_{iu}\mid A,\mathcal I)
=F_{i,tu}\Sigma_{tu},
\]

where $F_{i,tu}$ is a known function of the two training cell counts and
their joint count. Averaging those factors over validation observations gives
an elementwise inflation matrix $F$. Consequently,

\[
E\{\widehat M_{\mathrm{raw}}\mid A,\mathcal I\}=F\circ\Sigma,
\qquad
\widehat M_{\mathrm{corrected}}=
\widehat M_{\mathrm{raw}}\oslash F
\]

is exactly centered at $\Sigma$ before positive-definite or structured
covariance projection. This corrects a particularly important distortion:
task-specific saturated means inflate diagonal residual moments much more
than off-diagonal moments and hence attenuate estimated within-respondent
correlation.

## Closed expected product-design correction

There is also a compact expected-risk specialization. Let one fold have
$n=(K-1)G/K$ training respondents, let $p=1/M$, and set

\[
d_n=nE\left\{\frac{1(N>0)}N\right\},
\qquad
s_n=1-(1-p)^n,
\qquad
N\sim\operatorname{Binomial}(n,p).
\]

Under iid product-uniform assignment and a schedule-constant task covariance,
define

\[
(\Gamma_n)_{tt}=\Sigma_{tt}\frac{d_n}{n},
\qquad
(\Gamma_n)_{tu}=\Sigma_{tu}\frac{s_n^2}{n},\quad t\ne u.
\]

For a centered unit-variance factorial feature and task mixing matrix $M_H$,
the marginal mean-estimation remainder is

\[
V_{\mathrm{nuis}}(M_H)
=
\sum_t(\Gamma_n)_{tt}
\left\{
\frac{\omega_t^2}{M}
-\frac{2\omega_t(M_H)_{tt}}{M}
\right\}
+\operatorname{tr}(M_H^\top\Gamma_nM_H).
\]

Product-design calibration fixes $(M_H)_{tt}=\omega_t$. The affine part is
therefore constant, and the exact minimizer of this marginal-fold objective is
the original task-mixing solution with

\[
\Sigma_{\mathrm{eff}}=\Sigma+\Gamma_n.
\]

This is not the exact minimizer of the whole overlapping $K$-fold estimator:
cross-fold covariances add a smaller term. It is exact for the supported-cell
marginal objective, while the loading formula above remains the full finite-
sample reference. The frozen task-wide fallback for an empty cell also adds a
small term; Gate G retains it in the loading computation and records support
failures rather than hiding them.

## Conditional finite-support benchmark

When the saturated mean dimension is estimable from the realized assignments
and $\Sigma_i$ is known, define

\[
\mathcal I_A=\sum_i B_i^\top\Sigma_i^{-1}B_i.
\]

Then

\[
\widehat\theta_{\mathrm{GLS}}
=
L\mathcal I_A^+
\sum_iB_i^\top\Sigma_i^{-1}Y_i,
\qquad
\operatorname{Var}(\widehat\theta_{\mathrm{GLS}}\mid A)
=L\mathcal I_A^+L^\top.
\]

This estimator enforces factorial calibration on the realized sample and
removes the cross-fit mean-estimation remainder within the restricted
finite-support model. It is conditional BLUE/GLS, not a new general ML
estimator. Same-data feasible GLS is only diagnostic unless covariance-weight
learning is separated from the outcomes or its influence is propagated.

Gate G implements that separation with a supported two-way split. A saturated
residual covariance is estimated on one half and used for GLS on the other;
the roles are then reversed and the two target estimates are averaged. For
task-specific residual projections $P_t$, the covariance moment uses

\[
E(e_t^\top e_u\mid A)
=
\Sigma_{tu}\operatorname{tr}\{(I-P_t)(I-P_u)\}.
\]

This makes each half-estimator conditionally centered under the restricted
mean model even though its weights are learned. Its full variance still
contains covariance-learning uncertainty, so the frozen-weight loading is not
used as an exact interval formula in the screen.

## What is and is not new

The exact loading identity, residual degrees-of-freedom correction, and
conditional GLS are not by themselves stand-alone methodological novelties.
The potentially paper-enabling piece is their target-weighted factorial
specialization and their integration with the earlier efficient influence
function and carryover-robustness results. The fresh Gate G screen determines
whether that integration also solves the empirical finite-sample problem.

## Status

- Deterministic identities: **pass (35 checks)**.
- Fresh proof-of-concept simulation: **formal fail** (two of seven performance
  components failed; all integrity prerequisites passed).
- Practical point-estimation gain from $\Sigma+\Gamma_n$: expected to be
  small and empirically below 0.1%; the primary value is correct finite-sample
  risk and plug-in interval calibration.
- Stand-alone novelty: **not established**.
