# Efficient influence functions for repeated factorial experiments

**Status:** theory-feasibility memo, 2026-08-27  
**Scope:** superpopulation inference with independent respondent clusters and a
known randomized assignment law  
**Preliminary verdict:** the ordinary target-weighted EIF is routine; a
covariance-aware EIF under the standard no-carryover mean restriction is a
plausible paper direction and changes the empirical question materially.

**Subsequent gate status:** Gates C, D, and E1 passed, but Gate E2 hard-failed
two finite-sample checks. The first-order EIF and sensitivity geometry remain
supported by known-mean controls; the current feasible adaptive procedure does
not yet realize that theory reliably at 250--500 respondent clusters. See
`docs/gate-e2-decision.md`.

Gate E3 then reduced one contrast exactly to the established Gaussian
restricted-versus-unrestricted adaptation problem and therefore failed as a
stand-alone novelty route. It also derived a useful efficient score for a
prespecified carryover basis by augmenting the mean design. See
`docs/gate-e3-theory.md`.

Gate F subsequently derived an exact Walsh--Krawtchouk decomposition for
Hamming-radial binary factorial schedules, including the order-specific
efficiency and carryover-rank operators. The result is mathematically useful
but failed as a stand-alone novelty route because factorial changeover design
and the hypercube spectral identity are classical. It also showed why schedule
comparisons must hold the causal target fixed under carryover. See
`docs/gate-f-theory.md`.

## Executive verdict

There are two different statistical models, and they lead to different
answers.

1. If the conditional mean of every task outcome is unrestricted as a
   function of the respondent's complete assignment vector, the EIF is the
   ordinary stochastic-intervention/multi-arm AIPW score applied to a whole
   respondent. This derivation is useful for correctness but is not a new
   theorem.
2. If we impose the scientifically standard conjoint restriction that a
   task's conditional mean depends on its own randomized profile, but not on
   profiles assigned in other tasks, the ordinary average of taskwise AIPW
   scores is generally **not efficient**. Correlated outcomes create additional
   information. The efficient residual weights solve a covariance-weighted
   calibration problem under the exact joint randomization law.

The second result is the viable route. It is especially relevant here because
the candidate-recruitment ratings have high within-respondent correlation. The
previous ML headroom experiment allowed flexible prediction from baseline
covariates but did not exploit other task assignments as randomized control
variates. Its negative verdict therefore does not answer this new efficiency
question.

The generic covariance-weighting principle is related to optimal GEE and
conditional-moment theory, so the EIF formula alone is not yet a defensible
headline novelty. A full paper would need factorial target geometry, exact
joint-design calibration, a robust cross-fitted estimator, and meaningful
gains in repeated factorial designs.

## 1. Statistical setup

Let respondents $i=1,\ldots,G$ be iid. For a fixed number of tasks $T$, one
respondent contributes

$$
O=(X,\mathbf A,\mathbf Y),
\qquad
\mathbf A=(A_1,\ldots,A_T),
\qquad
\mathbf Y=(Y_1,\ldots,Y_T)^\mathsf T.
$$

The complete within-respondent assignment has known probability

$$
q(\mathbf a\mid x)=P(\mathbf A=\mathbf a\mid X=x).
$$

This may be a product law, a without-replacement law, or another prespecified
randomization. It is not necessary for tasks to be independently assigned.
Variable bounded task counts can be incorporated into $X$; fixed $T$ keeps
the notation readable.

Let $\mathcal S$ be the target profile support and let $\mu$ be a
prespecified target law on it. For a $d$-dimensional factorial family, let
$\Phi(a)\in\mathbb R^d$ be any target-orthonormal basis:

$$
\sum_{a\in\mathcal S}\mu(a)\Phi(a)=0,
\qquad
\sum_{a\in\mathcal S}\mu(a)\Phi(a)\Phi(a)^\mathsf T=I_d.
$$

The direct profile load is

$$
\ell(a)=\mu(a)\Phi(a).
$$

Everything below also applies to named AMCE or AMIE contrasts after replacing
$\ell(a)$ by their prespecified signed profile weights.

The memo concerns the coefficient vector of the population-average response
surface, not the squared interaction energy and not a distribution of
respondent-specific effects.

## 2. Unrestricted cluster model

First allow

$$
m_t(\mathbf a,x)=E(Y_t\mid\mathbf A=\mathbf a,X=x)
$$

to depend arbitrarily on the complete assignment vector. Consider the general
linear target

$$
\theta
=
E_X\{g_m(X)\},
\qquad
g_m(x)
=
\sum_{\mathbf a}\sum_{t=1}^T
\kappa_t(\mathbf a,x)m_t(\mathbf a,x),
$$

where each $\kappa_t(\mathbf a,x)\in\mathbb R^d$ is fixed by the estimand.
Assume $\kappa_t(\mathbf a,x)=0$ whenever
$q(\mathbf a\mid x)=0$, and define

$$
r_t(\mathbf a,x)
=
\frac{\kappa_t(\mathbf a,x)}{q(\mathbf a\mid x)}.
$$

### Proposition 1: EIF in the unrestricted model

In the nonparametric model for $P_X$ and
$P_{\mathbf Y\mid\mathbf A,X}$, with $q$ known, the efficient influence
function is

$$
\phi_{\mathrm{unr}}(O)
=
g_m(X)-\theta
+
\sum_{t=1}^T
r_t(\mathbf A,X)
\{Y_t-m_t(\mathbf A,X)\}.
\tag{1}
$$

Write $R(\mathbf a,x)$ for the $d\times T$ matrix whose $t$th column is
$r_t(\mathbf a,x)$, and write

$$
\Sigma(\mathbf a,x)
=
\operatorname{Var}(\mathbf Y\mid\mathbf A=\mathbf a,X=x).
$$

The respondent-level efficiency bound is

$$
\Omega_{\mathrm{unr}}
=
\operatorname{Var}\{g_m(X)\}
+
E\!\left[
R(\mathbf A,X)
\Sigma(\mathbf A,X)
R(\mathbf A,X)^\mathsf T
\right].
\tag{2}
$$

Thus every regular estimator based on $G$ iid respondents has asymptotic
covariance at least $\Omega_{\mathrm{unr}}/G$.

### Proof sketch

A regular observed-data score decomposes into a score for $P_X$, a score for
the known assignment law, and a score for the conditional outcome law. The
assignment score is zero. Differentiating $P_X$ produces
$g_m(X)-\theta$. Differentiating each conditional mean produces the residual
term in (1). The two terms are orthogonal because the residual has conditional
mean zero given $(\mathbf A,X)$. Equation (2) follows by iterated
expectations. The resulting element is in the tangent space and therefore is
the canonical gradient.

This is a vector version of the usual EIF for a fixed stochastic intervention
or multivalued treatment. Factorial notation changes its interpretation, not
its semiparametric content.

### Standard task-marginal target

Suppose task $t$'s target changes the marginal distribution of $A_t$ to
$\mu$, while averaging the other randomized tasks under their design
conditional law. With task weights $\omega_t$,

$$
\kappa_t(\mathbf a,x)
=
\omega_t\ell(a_t)
q(\mathbf a_{-t}\mid A_t=a_t,X=x).
$$

Then

$$
r_t(\mathbf a,x)
=
\omega_t
\frac{\ell(a_t)}{q_t(a_t\mid x)}.
\tag{3}
$$

The familiar marginal task weight therefore remains correct even when the
complete assignments are dependent. For efficiency in the unrestricted
model, however, the residual in (1) is centered by
$E(Y_t\mid\mathbf A,X)$, not necessarily by
$E(Y_t\mid A_t,X)$.

## 3. Exact variance gap for a working outcome regression

For any fixed working regression $\eta_t(\mathbf a,x)$, define the augmented
score by replacing $m$ with $\eta$ in both the plug-in and residual terms
of (1). Let $e_t=\eta_t-m_t$. The working score remains unbiased, but its
covariance obeys the exact identity

$$
\Omega(\eta)-\Omega_{\mathrm{unr}}
=
E\!\left[
\operatorname{Var}_q\left{
\sum_{t=1}^T r_t(\mathbf A,X)e_t(\mathbf A,X)
\,\middle|\,X
\right\}
\right]
\succeq 0.
\tag{4}
$$

The difference between the working and efficient scores is a mean-zero
function of $(\mathbf A,X)$, and it is orthogonal to the canonical gradient.
This proves (4).

Equation (4) also shows why ordinary outcome-prediction MSE is not generally
the correct nuisance-selection loss. The relevant loss weights prediction
error by the Riesz loads, subtracts its conditional assignment mean, and is
matrix-valued for a family of factorial contrasts. The identity is useful, but
it is a direct projection argument and not sufficient novelty by itself.

## 4. No-carryover conditional-mean model

Now impose structure that is natural for repeated factorial experiments. Let

$$
E(\mathbf Y\mid\mathbf A=\mathbf a,X=x)
=
B(\mathbf a,x)\gamma(x),
\tag{5}
$$

where $B(\mathbf a,x)$ is known and $\gamma(x)$ is unrestricted.

Two important versions are:

- **Task-specific, no carryover:** $\gamma(x)$ contains one response-surface
  cell mean for every task and profile. Row $t$ of $B$ selects only the
  cell indexed by $A_t$.
- **Stable response surface:** all tasks share the same profile response
  surface. Every row of $B$ selects from the same profile-mean vector.
  Task-position intercepts may be appended as nuisance columns with zero load
  in the factorial contrast.

The target is

$$
\theta=E\{L(X)\gamma(X)\},
\tag{6}
$$

where $L(X)$ is a known $d\times p$ matrix. For a common surface on a
finite support,

$$
L=\Phi^\mathsf T D_\mu,
\qquad
D_\mu=\operatorname{diag}\{\mu(a):a\in\mathcal S\}.
$$

For the average of $T$ task-specific surfaces, $L$ contains $T$ copies
of this block, each multiplied by $1/T$.

Let

$$
J(x)
=
E_q\!\left[
B(\mathbf A,x)^\mathsf T
\Sigma(\mathbf A,x)^{-1}
B(\mathbf A,x)
\,\middle|\,X=x
\right].
\tag{7}
$$

Assume $\Sigma$ is positive definite and the columns of $L(x)^\mathsf T$
belong to the range of $J(x)$. A generalized inverse handles unidentified
nuisance directions.

### Proposition 2: covariance-aware efficient influence function

Define the $T\times d$ residual-weight matrix

$$
H^*(\mathbf a,x)
=
\Sigma(\mathbf a,x)^{-1}
B(\mathbf a,x)
J(x)^+
L(x)^\mathsf T.
\tag{8}
$$

Under model (5), the efficient influence function is

$$
\phi_{\mathrm{nc}}(O)
=
L(X)\gamma(X)-\theta
+
H^*(\mathbf A,X)^\mathsf T
\{\mathbf Y-B(\mathbf A,X)\gamma(X)\}.
\tag{9}
$$

Its efficiency bound is

$$
\Omega_{\mathrm{nc}}
=
\operatorname{Var}\{L(X)\gamma(X)\}
+
E\{L(X)J(X)^+L(X)^\mathsf T\}.
\tag{10}
$$

### Why (8) is the canonical weight

For any regular influence function with residual weight
$H(\mathbf A,X)$, pathwise differentiation of (6) imposes the calibration
condition

$$
E_q\{B(\mathbf A,X)^\mathsf T H(\mathbf A,X)\mid X\}
=
L(X)^\mathsf T.
\tag{11}
$$

Among all weights satisfying (11), the canonical weight uniquely minimizes

$$
E_q\left[
\operatorname{tr}
\{H(\mathbf A,X)^\mathsf T
\Sigma(\mathbf A,X)
H(\mathbf A,X)\}
\,\middle|\,X
\right].
\tag{12}
$$

A Lagrange-multiplier calculation gives (8), and substituting (8) into (12)
gives the second term of (10). This is the conditional-mean analogue of GLS or
an optimally weighted GEE.

The usual taskwise AIPW weights satisfy (11), so they remain valid. They are
efficient only when they also solve (12). In the task-specific no-carryover
model, an equivalent first-order condition is that every component of
$\Sigma(\mathbf A,X)H_0(\mathbf A,X)$ lies in the corresponding taskwise
mean-direction space. With correlated residuals and independently varying
task assignments, that condition generally fails.

### Robustness of a working covariance

Any estimated weight $\widehat H$ that exactly satisfies (11) yields the
score

$$
L(X)\widehat\gamma(X)
+
\widehat H(\mathbf A,X)^\mathsf T
\{\mathbf Y-B(\mathbf A,X)\widehat\gamma(X)\}.
\tag{13}
$$

Conditional on independently trained nuisances, (13) has expectation
$\theta$ for every working $\widehat\gamma$ and every positive-definite
working covariance used to construct $\widehat H$. Correct covariance is
needed for efficiency, not for centering. This makes foldwise exact calibration
under the known design the key computational requirement.

There is also an exact decomposition of the two possible efficiency losses.
For any calibrated fixed $H$, let $e(X)=\eta(X)-\gamma(X)$ and let
$\Omega(H,\eta)$ denote the covariance of its score. Then

$$
\begin{aligned}
\Omega(H,\eta)-\Omega_{\mathrm{nc}}
= {}& E\!\left[
\{H-H^*\}^\mathsf T\Sigma\{H-H^*\}
\right] \\
&+E\!\left[
d_e(\mathbf A,X)d_e(\mathbf A,X)^\mathsf T
\right],
\tag{14}
\end{aligned}
$$

where

$$
d_e(\mathbf A,X)
=
\{L(X)-H(\mathbf A,X)^\mathsf T B(\mathbf A,X)\}e(X).
$$

Both terms in (14) are positive semidefinite. The first is loss from using
nonoptimal covariance weights; the second is loss from outcome-regression
error for the chosen weights. Their cross terms vanish by calibration,
conditional residual centering, and the projection defining $H^*$. This
separates **within-respondent covariance headroom** from the baseline-covariate
prediction headroom studied in the earlier ML gate.

## 5. A two-task special case

Let $A_1,A_2$ be independent and uniform on $\{-1,+1\}$, and suppose

$$
E(Y_t\mid\mathbf A)=\alpha_t+\tau_t A_t,
\qquad
\operatorname{Var}(\mathbf Y\mid\mathbf A)
=
\sigma^2
\begin{pmatrix}
1&\rho\\
\rho&1
\end{pmatrix}.
$$

The target is $\theta=(\tau_1+\tau_2)/2$. The ordinary valid residual weights
are

$$
H_0(\mathbf A)
=
\frac12
\begin{pmatrix}A_1\\A_2\end{pmatrix}.
$$

The efficient weights are

$$
H^*(\mathbf A)
=
\frac12
\begin{pmatrix}
A_1-\rho A_2\\
A_2-\rho A_1
\end{pmatrix}.
\tag{15}
$$

Both satisfy the same factorial calibration conditions. Their residual
variance bounds are

$$
V_0=\frac{\sigma^2}{2},
\qquad
V^*=\frac{\sigma^2(1-\rho^2)}{2}.
$$

Thus the efficient/ordinary variance ratio is $1-\rho^2$. At
$\rho=0.8$, the variance ratio is $0.36$. Other task assignments enter
(15) only as randomized control variates; they do not alter the estimand.

More generally, suppose the tasks are independently assigned from the target
law, the same centered target-orthonormal contrast family is used at every
task, and the residual covariance $\Sigma$ is constant across assignments.
For task weights $\omega_t$, the covariance multipliers for the efficient and
ordinary residual terms are

$$
\kappa^*
=
\sum_{t=1}^T
\frac{\omega_t^2}{(\Sigma^{-1})_{tt}},
\qquad
\kappa_0
=
\sum_{t=1}^T \omega_t^2\Sigma_{tt}.
\tag{16}
$$

Because $1/(\Sigma^{-1})_{tt}$ is the linear conditional residual variance of
task $t$ given the other task residuals, $\kappa^*\leq\kappa_0$. Equality
holds under working independence and can fail strictly under correlation.

With $T$ independently randomized balanced tasks and
compound-symmetric residual covariance

$$
\Sigma=eI_T+u\mathbf 1\mathbf 1^\mathsf T,
$$

the ordinary and efficient residual bounds for an equal average of
task-specific balanced contrasts are

$$
V_0=\frac{e+u}{T},
\qquad
V^*=\frac{e(e+Tu)}{T\{e+(T-1)u\}}.
\tag{17}
$$

The gain can therefore be large when a respondent random intercept dominates
task-specific noise. Formula (17) is an analytic diagnostic, not evidence
that the same gain is attained in the candidate data.

## 6. Factorial invariance and support

If a target-orthonormal family basis is changed to
$\Phi^\dagger=\Phi Q$ for an orthogonal matrix $Q$, then

$$
\theta^\dagger=Q^\mathsf T\theta,
\qquad
H^{*\dagger}=H^*Q,
\qquad
\Omega^\dagger=Q^\mathsf T\Omega Q.
$$

The fitted family, its covariance operator, trace per degree of freedom, and
eigenvalues are therefore invariant. Named direct AMCEs or AMIEs do not change
at all under contrast recoding.

The correct identification condition in the restricted model is stronger and
more informative than a rowwise `mu/q` check:

$$
\operatorname{col}\{L(x)^\mathsf T\}
\subseteq
\operatorname{col}\{J(x)\}
\quad\text{almost surely}.
\tag{18}
$$

Equation (18) detects whether the exact repeated-task assignment supplies
information for the requested target contrast. It accommodates nonproduct and
without-replacement designs. A finite bound additionally requires
$E\{LJ^+L^\mathsf T\}<\infty$.

### Model boundary

Proposition 2 is an efficiency statement for the **conditional-mean** model
(5). It does not claim the bound for a stronger model in which complete
potential-outcome distributions obey no carryover. Such a smaller model can,
in principle, permit nonlinear use of other task outcomes and has a different
coarsening/operator problem. Conversely, an estimator regular under arbitrary
conditional-mean carryover generally cannot attain the smaller bound in (10).
The extra precision is therefore a real robustness--efficiency tradeoff, not a
free consequence of clustering.

## 7. What is and is not new

### Established

- Equation (1) is the ordinary EIF for a fixed stochastic intervention or a
  collection of multivalued treatment means.
- The projection identity (4) is standard augmentation geometry.
- Inverse-covariance weighting under conditional mean restrictions is the
  foundation of optimal GEE and semiparametric conditional-moment theory.
- Adaptive selection by held-out influence-curve variance is already part of
  adaptive prespecification.

Closest work includes:

- [Narita, Yasui, and Yata](https://arxiv.org/abs/1809.03084), whose
  off-policy efficiency theory already accommodates signed policy weights and
  therefore covers the one-task factorial contrast after a linear
  reparameterization;
- [Stephens et al. (2014)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4142698/),
  who study locally efficient marginal treatment-effect estimation with
  correlated outcomes;
- [Park and Kang (2022)](https://arxiv.org/abs/2110.07740), who use conditional
  treatment probabilities and outcome covariance to improve multilevel AIPW,
  while explicitly leaving the full semiparametric optimum unresolved;
- [Hristache and Patilea](https://arxiv.org/abs/1111.6428), whose seemingly
  unrelated conditional-moment theory is the broad theoretical umbrella for
  different taskwise conditioning sets;
- [Cheng, Bell, and Qian](https://arxiv.org/abs/2311.16529), who derive
  efficiency results for repeated micro-randomization under serial-dependence
  working assumptions and identify general cross-decision correlation as a
  difficult extension;
- [Xia et al. (2026)](https://arxiv.org/abs/2510.08972), who develop
  covariance-aware semiparametric efficiency theory for stepped-wedge designs;
  and
- [Jiang and Ding (2026)](https://arxiv.org/abs/2511.09215), who give a
  design-based framework for efficient crossover analysis.

### Plausible factorial contribution

A paper may still be defensible if it develops all of the following as one
result rather than presenting (8) as an isolated new idea:

1. The canonical gradient and bound for a vector of target-$\mu$,
   coding-invariant factorial contrasts under the exact repeated-task
   randomization law.
2. A calibrated estimator that remains valid under a misspecified working
   covariance and uses respondent-level cross-fitting for flexible mean and
   covariance nuisances.
3. Rank, support, and Riesz-leverage diagnostics for nonproduct, restricted,
   and without-replacement profile designs.
4. Computation that avoids enumerating the exponentially large joint task
   assignment space by exploiting factorial and assignment structure.
5. Theory and evidence showing when ordinary taskwise AIPW is efficient and
   when covariance-aware weighting yields consequential gains.

The clean paper title would be **“Efficient Inference for Repeated Factorial
Experiments.”** ML would be an implementation tool for nuisance estimation,
not the claimed invention.

## 8. Theory and empirical gates

### Gate A: literature separation

Read the full proofs in Hristache--Patilea, Stephens et al., Park--Kang,
Cheng--Bell--Qian, Xia et al., and Jiang--Ding. Continue only if the
target-$\mu$ vector result, exact factorial
support condition (18), or scalable calibrated solver is not obtained by a
direct relabeling of one of those results.

### Gate B: formal theorem

Complete a tangent-space proof of Proposition 2, including positive-
semidefinite covariance and a nonredundant paired-choice encoding, variable
task counts, nuisance task-position effects, and the nesting

$$
\Omega_{\text{stable surface}}
\preceq
\Omega_{\text{task-specific no carryover}}
\preceq
\Omega_{\text{unrestricted cluster}}.
$$

At least one strictness characterization or scalable factorial computation
result must be nontrivial.

### Gate C: oracle headroom

Before fitting flexible nuisances, estimate the covariance-aware oracle bound
in the candidate geometry. Proceed if the efficient/ordinary family-balanced
variance ratio is at most 0.75, with an upper confidence bound below
0.85. This is a new gate: the earlier baseline-covariate ML headroom ratio
of 0.967 does not test it.

**Executed 2026-08-27: PASS.** With respondent-cross-fitted task-specific
additive residualization, the efficient/ordinary oracle ratio was 0.2227 under
an unstructured residual covariance and documented product assignment; its
conditional respondent-bootstrap 95% interval was [0.2059, 0.2375]. The worst
result across the required unstructured/compound-symmetry and
product/uniform-without-replacement combinations was 0.2315 with upper endpoint
0.2489. Exact calibration error was below (9\times10^{-17}). See
`docs/covariance-headroom-gate-decision.md`. This passes only the oracle
headroom gate, not feasible estimation, coverage, no-carryover, or assignment-
mechanism verification.

### Gate D: feasible estimator

With prespecified working covariance classes, require:

- selected/ordinary variance at most 0.90, upper bound below 0.98;
- at least half of the oracle covariance gain recovered;
- no more than 5% aggregate harm when within-respondent correlation is weak;
- 95% interval coverage between roughly 93% and 97%; and
- numerical invariance under baseline swaps, level permutations, and basis
  rotations.

**Executed 2026-08-27: PASS.** The frozen production design used 2,000
replications in each of nine prespecified scenarios. Strong-cell
selected/ordinary variance ratios ranged from 0.291 to 0.686; the worst paired
bootstrap upper endpoint was 0.700 and the minimum oracle gain recovered was
0.945. The weak-correlation aggregate ratio was 0.996, pooled coverage ranged
from 94.48% to 95.38%, maximum RMS standardized bias was 0.036, sandwich
variance calibration ranged from 0.948 to 1.055, and all 18,000 replications
completed. Calibration and coding-invariance tests also passed. See
`docs/gate-d-decision.md`. This advances the project to Gate E; it does not
verify mean no-carryover or an application's joint assignment law.

### Gate E: assumption sensitivity

Because the extra efficiency comes from the no-carryover/stability restriction,
report a fully robust unrestricted estimator beside the efficient estimator.
Add diagnostics or sensitivity analyses for profile-order effects, carryover,
and task-by-profile interactions. If plausible violations materially change
the estimates, the efficient estimator cannot be the default.

**Gate E1 executed 2026-08-27: PASS.** The exact adaptive-score bias under
carryover is the projection of the conditional-mean departure onto the
difference between adaptive and ordinary Riesz weights.  A sharp,
coding-equivariant bias ellipsoid follows from measuring distance to the
no-carryover mean space.  Under the residual-standardized metric, its leverage
operator is exactly $T$ times the restricted efficiency gain.  This yields a
closed-form oracle sensitivity-indexed blend and shows why a failure to reject
the paired Hausman diagnostic cannot validate no carryover.  All 49
finite-support identities passed at numerical precision.  This is an internal
theory-feasibility pass, not a separate publication-novelty claim.  See
`docs/gate-e1-theory.md`.
Proceed to the bounded Gate E2 computation with unrestricted inference primary.

## 9. Immediate next step

The next bounded exercise should be a finite-support prototype, not another
large ML benchmark:

1. implement the calibration/QP solution for fixed $T$ and a supplied
   covariance matrix;
2. verify exact unbiasedness and the bound by enumeration;
3. reproduce ordinary AIPW when residuals are independent;
4. run an oracle covariance screen on the candidate data or a
   candidate-calibrated repeated-rating design; and
5. finish the close-paper comparison before building a general package.

If the covariance-aware headroom gate fails, or if Jiang--Ding/optimal-GEE
theory already implies the complete result without new factorial work, this
route should stop at a technical note or software specialization.
