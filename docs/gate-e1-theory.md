# Gate E1 decision: carryover sensitivity theory

**Decision: PASS. Proceed to Gate E2, with unrestricted inference primary.**

**Subsequent update:** Gate E2 hard-failed its feasible local-AMSE and
sample-size-scaling checks. Known-mean controls supported the first-order
results derived here, but the current feasible implementation retained a
material finite-sample nuisance-estimation remainder. See
`docs/gate-e2-decision.md`. Unrestricted taskwise AIPW remains primary.

Gate E3 subsequently showed that the scalar adaptation problem is exactly the
established restricted-versus-unrestricted Gaussian experiment. Equation
(E1.24) is minimax within the fixed linear blend class, not over all measurable
adaptive rules. See `docs/gate-e3-theory.md`.

## Plain-language result

The covariance-adaptive estimator gains precision by using the profiles shown
in other tasks as randomized control variables.  If those other profiles also
change the current task's mean response, the same maneuver can create bias.

Gate E1 gives an exact answer to “how much bias?” and a sensitivity scale for
“how much carryover would erase the precision gain?”  The central result is a
duality: under the natural residual-standardized measure of carryover, the
directions providing the largest efficiency gain are exactly the directions
most vulnerable to a no-carryover violation.

This does not invalidate the estimator.  It means that ordinary taskwise AIPW
must be reported as the assumption-light analysis, while covariance-adaptive
inference is a more efficient analysis conditional on a transparent mean
restriction.

## 1. Target under arbitrary carryover

For one respondent, observe

$$
O=(X,\mathbf A,\mathbf Y),
\qquad
\mathbf A=(A_1,\ldots,A_T),
$$

with known joint assignment law $q(\mathbf a\mid x)$.  Let

$$
m(\mathbf a,x)=E(\mathbf Y\mid\mathbf A=\mathbf a,X=x)
$$

be unrestricted.  In the task-specific no-carryover model,

$$
m(\mathbf a,x)=B(\mathbf a,x)\gamma(x),
\tag{E1.1}
$$

where row $t$ of $B$ selects only the surface cell associated with $A_t$.

With consistency, randomized complete schedules, and positivity on every
schedule used by the target, $m_t(\mathbf a,x)$ identifies the mean potential
outcome $E\{Y_t(\mathbf a)\mid X=x\}$.  Equation (E1.2) below is therefore a
causal stochastic-schedule contrast.  When $q$ is dependent across tasks it is
$q$-specific; it should not be described as a pure direct effect holding all
other task assignments fixed.

The main derivation uses the unweighted respondent law.  Section 11 gives the
exact extension for a normalized population-tilting or survey weight; weighted
application formulas must not reuse the unweighted sensitivity operator
unchanged.

Let $H_0(\mathbf a,x)$ be the ordinary $T\times d$ taskwise Riesz weight.  For
the standard task-marginal factorial target, row $t$ is the signed target load
for $A_t$ divided by its marginal assignment probability.  Define the target
under arbitrary carryover as

$$
\theta_U
=
E\{H_0(\mathbf A,X)^\mathsf Tm(\mathbf A,X)\}.
\tag{E1.2}
$$

Equivalently, task $t$'s complete-assignment mean is averaged over
$q(\mathbf A_{-t}\mid A_t,X)$ and then contrasted over $A_t$.  Under a product
design this is the familiar intervention on the current profile while the
other tasks retain their randomized marginal distributions.  Under a
dependent design, the conditional distribution in that statement is part of
the estimand and must be reported.

The semiparametrically efficient estimator in the unrestricted cluster model
uses $H_0$ with a regression for the complete conditional mean $m(\mathbf
A,X)$ and integrates that regression under $q$.  The simpler own-task
augmentation used below is also exactly centered for (E1.2), even when that
working regression is wrong.  At a no-carryover reference law the two have the
same influence function when the own-task regression is consistent.

If (E1.1) holds, ordinary calibration gives

$$
E_q\{B(\mathbf A,X)^\mathsf T H_0(\mathbf A,X)\mid X\}
=L(X)^\mathsf T,
$$

and (E1.2) reduces to $E\{L(X)\gamma(X)\}$.

For interpretation when $B$ spans every task-specific own-profile surface,
define

$$
\bar m_t(a,x)
=E_q\{m_t(\mathbf A,x)\mid A_t=a,X=x\},
\qquad
\delta_t(\mathbf A,x)
=m_t(\mathbf A,x)-\bar m_t(A_t,x).
\tag{E1.3}
$$

Then $E_q(\delta_t\mid A_t,X)=0$.  The vector $\delta$ is the part of the
conditional mean that cannot be represented by the task's own profile alone.

## 2. Exact carryover bias

Let $H(\mathbf a,x)$ be any covariance-adaptive weight satisfying the same
restricted calibration,

$$
E_q(B^\mathsf T H\mid X)=L^\mathsf T.
\tag{E1.4}
$$

For any independently trained own-task working mean $\eta(X)$, its uncentered
score is

$$
Q_H(O;\eta)
=L(X)\eta(X)
+H(\mathbf A,X)^\mathsf T
\{\mathbf Y-B(\mathbf A,X)\eta(X)\}.
\tag{E1.5}
$$

Put $D=H-H_0$.

### Proposition E1: exact bias operator

Under the true joint assignment law,

$$
\boxed{
E\{Q_H(O;\eta)\}-\theta_U
=E\{D(\mathbf A,X)^\mathsf Tm(\mathbf A,X)\}
=E(D^\mathsf T\delta).
}
\tag{E1.6}
$$

The identity holds for every working $\eta$; correct outcome regression is not
needed.  Consequently:

- ordinary taskwise scoring has $D=0$ and remains centered under arbitrary
  conditional-mean carryover;
- no carryover is sufficient but not necessary for adaptive-score validity;
- only the component of carryover seen by $D$ creates bias; and
- adding any no-carryover surface $Bv(X)$ to $m$ leaves the bias unchanged.

The last point means that carryover is naturally an equivalence class modulo
the no-carryover mean space, rather than a particular dummy-variable
coefficient.

**Proof.** Calibration makes the plug-in cancel:

$$
E(Q_H\mid X)
=L\eta+E_q\{H^\mathsf T(m-B\eta)\mid X\}
=E_q(H^\mathsf Tm\mid X).
$$

Subtract (E1.2).  Because $E_q(B^\mathsf TD\mid X)=0$, every $Bv(X)$ component
drops out.  In the task-specific decomposition (E1.3), the own-task marginal
surface also drops out, giving the final equality in (E1.6). $\square$

An immediate empirical consequence for fixed weights is

$$
E(\widehat\theta_H-\widehat\theta_0)=E(D^\mathsf T\delta).
\tag{E1.7}
$$

Thus the paired estimator difference estimates the bias relevant to this
particular factorial target.  It does not estimate the magnitude of all
carryover.

For a learned weight trained on an independent sample $\mathcal T$, the exact
statement is conditional:

$$
E(\widehat\theta_H-\widehat\theta_0\mid\mathcal T)
=E(D_{\mathcal T}^\mathsf T\delta\mid\mathcal T).
\tag{E1.7a}
$$

The unconditional identity additionally averages over training randomness.

## 3. Sharp sensitivity geometry

Choose a positive-definite task metric $M(\mathbf A,X)$ and define the
per-task distance from the no-carryover mean space by

$$
\varepsilon_M^2(m)
=
\inf_{v(\cdot)}
\frac1T E\left[
\{m-Bv(X)\}^\mathsf T
M
\{m-Bv(X)\}
\right].
\tag{E1.8}
$$

Two useful choices are:

- $M=I$, which measures RMS carryover in outcome units; and
- $M=\Sigma^{-1}$, where
  $\Sigma(\mathbf A,X)=\operatorname{Var}(\mathbf Y\mid\mathbf A,X)$, which
  measures carryover in residual-standard-deviation units.

Define the $d\times d$ carryover-leverage operator

$$
K_M(H)
=T E\{D^\mathsf TM^{-1}D\}.
\tag{E1.9}
$$

Write $b_H=E(D^\mathsf T\delta)$ for the exact bias vector.

### Proposition E2: sharp bias ellipsoid

For every reported scalar direction $a\in\mathbb R^d$,

$$
\boxed{
|a^\mathsf Tb_H|
\le
\varepsilon_M
\sqrt{a^\mathsf T K_M(H)a}.
}
\tag{E1.10}
$$

The bound is sharp over the $L_2(M)$ departure class.  One least-favourable
departure is

$$
c_a(\mathbf A,X)
=
\frac{
\varepsilon_M T M^{-1}D a
}{
\sqrt{a^\mathsf T K_M(H)a}
}.
\tag{E1.11}
$$

If $a^\mathsf T K_Ma=0$, the directional bias is identically zero and no
least-favourable departure needs to be defined.

It is $M$-orthogonal to the no-carryover mean space because
$E_q(B^\mathsf TD\mid X)=0$.  The image of the abstract $L_2(M)$
mean-departure ball is

$$
\mathcal B(\varepsilon_M)
=
\left\{
b\in\operatorname{range}(K_M):
b^\mathsf T K_M^+b\le\varepsilon_M^2
\right\}.
\tag{E1.12}
$$

Equation (E1.12) is coding equivariant.  An invertible recoding of nuisance
mean parameters leaves $H$, $D$, $K_M$, and the fitted target unchanged.  An
invertible reparameterization $\theta^\dagger=R\theta$ gives
$K_M^\dagger=RK_MR^\mathsf T$ and $b_H^\dagger=Rb_H$; the ellipsoid's gauge is
unchanged.  Eigenvalue summaries have their simplest invariance interpretation
under orthogonal target-basis changes.

**Proof.** Subtract the best $Bv$ from $m$ without changing (E1.6), then apply
Cauchy--Schwarz in the $M$ inner product.  Equation (E1.11) attains equality
and satisfies the normal equations for the projection onto the no-carryover
space. $\square$

For bounded or discrete outcomes, a large least-favourable mean departure may
not correspond to a valid outcome law.  Equation (E1.12) is then an outer bound
unless the relevant departure is embedded in a feasible distributional
submodel.  It is exact locally under a sufficiently rich differentiable
outcome model.  A secondary outcome-scale analysis can instead impose a
pointwise bound on carryover.

## 4. Efficiency gain equals local vulnerability

At a no-carryover reference law, set $M=\Sigma^{-1}$.  Let $H^*$ minimize
residual variance among weights satisfying (E1.4), and let
$\Omega_*$ and $\Omega_0$ be the respondent-level covariance matrices of the
restricted-efficient and ordinary influence functions.  Projection
orthogonality gives

$$
\Omega_0-\Omega_*
=E\{(H_0-H^*)^\mathsf T\Sigma(H_0-H^*)\}
\succeq0.
\tag{E1.13}
$$

Combining (E1.9) and (E1.13) gives the central Gate E1 identity:

$$
\boxed{
K_{\Sigma^{-1}}(H^*)
=T(\Omega_0-\Omega_*).
}
\tag{E1.14}
$$

This is the **efficiency-gain/local-vulnerability duality**.  A factorial
direction with no efficiency gain also has no first-order vulnerability from
using $H^*$ instead of $H_0$.  A direction with a large gain has proportionally
large worst-case carryover sensitivity in the residual-standardized metric.

The projection principle is generic, so (E1.14) should be presented as a
target-weighted repeated-factorial specialization, not as a wholly new theory
of misspecification.

## 5. Local misspecification and MSE

Let $G$ be the number of independent respondents and consider the triangular
mean array

$$
m_G(\mathbf A,X)
=B(\mathbf A,X)\gamma(X)+G^{-1/2}c(\mathbf A,X).
\tag{E1.15}
$$

The unrestricted target $\theta_{U,G}$ is still defined by (E1.2).  Let

$$
\beta=E\{(H^*-H_0)^\mathsf Tc\}.
\tag{E1.16}
$$

For the displayed normal limit, assume this array is embedded in a
differentiable-in-quadratic-mean local submodel, or impose the corresponding
Lindeberg and covariance-convergence conditions directly.  Under the
regularity conditions in Section 10,

$$
\sqrt G
\begin{pmatrix}
\widehat\theta_*-\theta_{U,G}\\
\widehat\theta_0-\theta_{U,G}
\end{pmatrix}
\rightsquigarrow
N\left[
\begin{pmatrix}\beta\\0\end{pmatrix},
\begin{pmatrix}
\Omega_* & \Omega_*\\
\Omega_* & \Omega_0
\end{pmatrix}
\right].
\tag{E1.17}
$$

In particular,

$$
\sqrt G(\widehat\theta_*-\widehat\theta_0)
\rightsquigarrow
N\{\beta,\Omega_0-\Omega_*\}.
\tag{E1.18}
$$

At the reference law, the two influence functions share their $X$ component
and differ only by $(H^*-H_0)^\mathsf T\varepsilon$.  The normal equations for
$H^*$ make that difference orthogonal to the efficient influence function,
which gives the cross block $\Omega_*$ in (E1.17).  The local mean shift in
(E1.16) follows from Proposition E1.  A joint triangular-array CLT, or Le Cam's
third lemma in the differentiable submodel, then gives (E1.17).

For a scalar direction $a$, the efficient estimator has smaller local
asymptotic MSE exactly when

$$
(a^\mathsf T\beta)^2
<a^\mathsf T(\Omega_0-\Omega_*)a.
\tag{E1.19}
$$

If the local departure satisfies $\varepsilon_{\Sigma^{-1}}(c)\le\rho$,
the sharp worst-case AMSE of the efficient estimator in direction $a$ is

$$
a^\mathsf T\Omega_*a
+\rho^2T\,a^\mathsf T(\Omega_0-\Omega_*)a.
\tag{E1.20}
$$

Therefore, in every direction with positive headroom:

$$
\boxed{
\rho<1/\sqrt T
\quad\Longleftrightarrow\quad
\text{restricted efficiency wins in worst-case local MSE}.
}
\tag{E1.21}
$$

For a fixed, nonshrinking carryover violation with nonzero bias, the adaptive
estimator is inconsistent and its squared bias eventually dominates.  The
robustness--efficiency comparison is intrinsically local unless scientific
knowledge supplies a fixed finite-sample bound.

## 6. A sensitivity-indexed blend

Consider the calibration-preserving blend

$$
H_\tau=(1-\tau)H_0+\tau H^*,
\qquad 0\le\tau\le1.
\tag{E1.22}
$$

Projection geometry gives

$$
\Omega_\tau
=\Omega_*+(1-\tau)^2(\Omega_0-\Omega_*),
\qquad
K_\tau
=\tau^2T(\Omega_0-\Omega_*).
\tag{E1.23}
$$

For the unrestricted residual-standardized local ball of radius $\rho$ and the
exact projection $H^*$, the same blend minimizes worst-case directional AMSE
within the fixed linear family $\{H_\tau:0\le\tau\le1\}$ for every factorial
direction with positive headroom:

$$
\boxed{
\tau_{\mathrm{mm}}(\rho)
=\frac{1}{1+T\rho^2}.
}
\tag{E1.24}
$$

Null-headroom directions tie for every $\tau$.  Equation (E1.24) does not
generally extend unchanged to a structured carryover class, a multivariate
loss, or a feasible weight that is not the exact projection.

This sensitivity-indexed rule is more defensible than selecting $\tau$ using
estimated variance alone when carryover is a concern.  It is a factorial
closed-form specialization of
established minimax/local-misspecification ideas, especially
[Bonhomme and Weidner](https://doi.org/10.3982/QE1930) and
[Armstrong and Kolesár](https://doi.org/10.3982/QE1609).

For a feasible limiting weight $H$ that is not $H^*$, let
$\phi_D=\phi_H-\phi_0$, $V_D=\operatorname{Var}(\phi_D)$, and
$C_{0D}=\operatorname{Cov}(\phi_0,\phi_D)$.  The blend
$H_0+\tau(H-H_0)$ has reference variance

$$
\Omega_0+\tau(C_{0D}+C_{0D}^\mathsf T)+\tau^2V_D
\tag{E1.24a}
$$

and sensitivity operator $\tau^2K_M(H)$.  Its scalar minimax weight must be
computed from these quantities directly; equation (E1.24) is valid only when
projection orthogonality reduces (E1.24a) to (E1.23).

Specifically, for direction $a$, put
$c_{0D,a}=a^\mathsf TC_{0D}a$, $v_{D,a}=a^\mathsf TV_Da$, and
$k_a=a^\mathsf TK_M(H)a$.  For a radius-$\rho$ ball, the generic clipped
solution is

$$
\tau_{H,a}(\rho)
=\Pi_{[0,1]}
\left\{
\frac{-c_{0D,a}}{v_{D,a}+\rho^2k_a}
\right\}.
\tag{E1.24b}
$$

If the denominator in (E1.24b) is zero, positive-semidefinite covariance and
sensitivity imply $c_{0D,a}=0$ as well, and every $\tau$ ties in that
direction.

## 7. Reporting sensitivity and tipping values

For a fixed scalar contrast $a^\mathsf T\theta$, define

$$
S_a=\sqrt{a^\mathsf T K_M(H)a}.
$$

If an analyst regards $\varepsilon_M\le\bar\varepsilon$ as plausible, a simple
bias-aware interval is

$$
a^\mathsf T\widehat\theta_H
\ \pm\ z_{1-\alpha/2}\widehat{\operatorname{se}}
\ \pm\ \bar\varepsilon S_a.
\tag{E1.25}
$$

The two plus/minus terms mean that the ordinary sampling interval is widened by
the worst-case bias magnitude.  Sharper bias-aware critical values can follow
the approximate-moment framework of Armstrong and Kolesár.

Two transparent tipping values are

$$
\varepsilon_{\mathrm{zero}}
=\frac{|a^\mathsf T\widehat\theta_H|}{S_a},
\qquad
\varepsilon_{\mathrm{sig}}
=\frac{
(|a^\mathsf T\widehat\theta_H|
-z_{1-\alpha/2}\widehat{\operatorname{se}})_+
}{S_a}.
\tag{E1.26}
$$

They are, respectively, the smallest permitted violation that can explain the
point estimate and the smallest that can erase conventional significance.
When $S_a=0$, use zero if the corresponding numerator is zero and infinity
otherwise.
For a fixed family, (E1.12) supplies a rotation-invariant ellipsoidal
sensitivity calculation rather than a collection of coding-dependent dummy
coefficients.

For an estimated family vector $\widehat\theta$ in the range of $K_M$, the
smallest radius capable of explaining the complete point estimate is

$$
\varepsilon_{\mathrm{family},0}
=\sqrt{\widehat\theta^\mathsf T K_M^+\widehat\theta}.
\tag{E1.26a}
$$

It is infinite if $\widehat\theta$ has a component outside
$\operatorname{range}(K_M)$.  Let $\widehat V$ be the sampling covariance of
$\widehat\theta$.  The squared family-rejection tipping value is obtained from
the convex trust-region problem

$$
\min_b\ b^\mathsf T K_M^+b
\quad\text{subject to}\quad
b\in\operatorname{range}(K_M),\qquad
(\widehat\theta-b)\in\operatorname{range}(\widehat V),\qquad
(\widehat\theta-b)^\mathsf T\widehat V^+
(\widehat\theta-b)
\le\chi^2_{r,1-\alpha}.
\tag{E1.26b}
$$

The family tipping radius is the square root of the optimum in (E1.26b); it is
infinite if the constraint set is empty.

Coordinate intervals in (E1.25) are pointwise.  Simultaneous family intervals
must use a simultaneous sampling critical value before adding the corresponding
coordinatewise worst-case bias margins.

The primary scale should be residual-standardized RMS carryover.  A secondary
scale should express carryover directly in rating-outcome points so substantive
readers can judge whether the tipping value is plausible.

## 8. What the paired diagnostic can establish

Under exact no carryover, a Hausman-type statistic based on (E1.18) is

$$
G(\widehat\theta_*-\widehat\theta_0)^\mathsf T
\widehat\Omega_D^+
(\widehat\theta_*-\widehat\theta_0),
\tag{E1.27}
$$

with an asymptotic chi-squared reference distribution using
$\operatorname{rank}(\Omega_D)$.  This statement requires fixed target
dimension, nonzero eigenvalues separated from zero, and a learned or selected
score converging to a deterministic limit.  In implementation, $\Omega_D$ must be
estimated directly from the paired respondent influence-score differences;
it should not be obtained by subtracting two estimated covariance matrices.

This is only a **targeted bias diagnostic**:

- it detects the projection $E(D^\mathsf T\delta)$;
- it has blind directions whenever nonzero $\delta$ is orthogonal to $D$;
- failure to reject does not establish global no carryover; and
- using the diagnostic as a preliminary test and then reporting ordinary
  restricted-model intervals creates nonuniform post-selection inference.

The limitation is quantitatively important.  In a scalar least-favourable
direction, the carryover value that exactly erases the efficient estimator's
MSE advantage gives the Hausman statistic noncentrality one.  A two-sided 5%
normal test then has only about 17% power.  The damaging violation can therefore
be difficult to detect, matching the nearly-true-model warning developed by
[Lumley](https://arxiv.org/abs/1707.05924).

This is why the unrestricted estimate and interval must remain available
regardless of the diagnostic result.  Hausman comparison and pretesting are
established ideas; see [Hausman](https://doi.org/10.2307/1913827),
[DiTraglia](https://doi.org/10.1016/j.jeconom.2016.07.006), and the crossover
warning in [Freeman](https://doi.org/10.1002/sim.4780081202).

## 9. Structured, sequential carryover

The unrestricted ball permits dependence of a task mean on both earlier and
future assigned profiles.  Under a dependent schedule law, conditional
dependence on a future assignment need not itself represent causal
anticipation; it can proxy for correlated past assignments.  A scientifically
tighter, explicitly sequential sensitivity class can specify

$$
c(\mathbf A,X)=Z(\mathbf A,X)\alpha,
$$

where $Z$ includes, for example, only lag-one profile features or previous
task-by-current-profile interactions and $\alpha$ is finite dimensional.  For
each $X$, first define

$$
\Pi_Z(X)
=
\{E_q(B^\mathsf TMB\mid X)\}^+
E_q(B^\mathsf TMZ\mid X),
\qquad
Z_\perp=Z-B\Pi_Z(X).
\tag{E1.28a}
$$

Then define

$$
Q_Z=\frac1T E(Z_\perp^\mathsf TMZ_\perp),
\qquad
R_Z=E(D^\mathsf TZ_\perp),
\qquad
K_Z=R_ZQ_Z^+R_Z^\mathsf T.
\tag{E1.28}
$$

Require $\ker(Q_Z)\subseteq\ker(R_Z)$ so unsupported coefficient directions
cannot alter the target bias.  Equations (E1.10)--(E1.12) and the tipping
formulas use $K_Z$ for this class.  The efficiency duality generally does not:
$K_Z$ need not equal $T(\Omega_0-\Omega_*)$.

Put $A_a=a^\mathsf T(\Omega_0-\Omega_*)a$ and
$B_a=a^\mathsf TK_Za$.  When both are positive, the structured-class
break-even radius and direction-specific minimax blend are

$$
\rho_{\mathrm{crit},a}=\sqrt{A_a/B_a},
\qquad
\tau_{\mathrm{mm},a}
=\frac{A_a}{A_a+\rho^2B_a}.
\tag{E1.28b}
$$

The unrestricted-ball formulas return because $B_a=TA_a$.  In the surrounding
$M=\Sigma^{-1}$, exact-$H^*$ setting, a structured class is a subspace of the
unrestricted class and $K_Z\preceq K_M$; using (E1.24) is conservative for
that smaller class.  Gate E2 should report both
the unrestricted worst-case class and prespecified lag-only classes.  This
separates prespecified temporal pathways from unrestricted complete-schedule
mean dependence.  If $\alpha$ is allowed to vary freely with $X$, the
finite-dimensional formula must be replaced by its conditional operator
analogue.

## 10. Learned weights and cross-fitting

The feasible estimator has a conventional asymptotic-linear limit under the
following sufficient conditions:

1. respondent clusters are independent and have uniformly bounded
   $2+\zeta$ moments;
2. the number of outer folds is fixed and all learning occurs outside each
   evaluation fold;
3. the true assignment law has adequate overlap, relevant covariance
   eigenvalues are bounded away from zero, and every learned weight satisfies
   exact calibration under that law;
4. the complete learned score converges in $L_2$ to its limiting score;
5. the learned weight converges to a deterministic limit; and
6. a finite-library selector has a separated population minimizer, or all
   asymptotic minimizers produce the same limiting score.

One sufficient route to condition 4 is ordinary $L_2$ consistency of the mean
and weight nuisances together with uniformly bounded weights and regressions.
Alternatively, use suitable $L_4$, higher-moment, and uniform-integrability
conditions for their products.  Exact calibration removes
the usual product-of-errors **bias** rate; it does not remove the requirement
that the complete learned score converge in $L_2$.  If selection remains
unstable among genuinely different limiting weights, outer cross-fitting alone
does not prevent a mixture or otherwise nonstandard limit.  In that case the
selected estimator cannot use a conventional single-normal approximation.

Under a fixed working $H$ that is not exactly $H^*$, equations (E1.6)--(E1.12)
still hold.  For the paired diagnostic and joint limit, estimate the complete
joint covariance directly rather than substituting
$\Omega_0-\Omega_H$, because projection equality is special to $H^*$.

## 11. Population tilting and assignment-law misspecification

For a known respondent population-tilting or survey weight $S$, put
$R=S/E(S)$ under the sampling law and regard $R$ as part of $X$.  Replace the
target and exact bias by

$$
\theta_U=E(RH_0^\mathsf Tm),
\qquad
b_H=E(RD^\mathsf T\delta).
\tag{E1.29a}
$$

The uncentered weighted score is $RQ_H$; with empirical Hájek normalization
its influence function is $R(Q_H-\theta_U)$.  The normalization term is common
to the ordinary and adaptive estimators and does not change their variance
gap.

If the departure norm in (E1.8) remains defined under the unweighted sampling
law, put $D_H=H-H_0$ and $D_*=H^*-H_0$.  The correct weighted sensitivity
operator and efficiency gap are

$$
K_M^{(R)}(H)
=T E(R^2D_H^\mathsf TM^{-1}D_H),
\qquad
\Omega_0^{(R)}-\Omega_*^{(R)}
=E\{R^2D_*^\mathsf T\Sigma D_*\}.
\tag{E1.29b}
$$

Thus the duality survives as
$K_{\Sigma^{-1}}^{(R)}(H^*)=T\{\Omega_0^{(R)}-\Omega_*^{(R)}\}$, but with
squared respondent weights.  These are the operators that must be used for the
survey-weighted candidate analysis.

### Assignment-law misspecification

Exact calibration is with respect to a particular joint law.  If a weight is
constructed under $\widetilde q$ but the true law is $q$, let

$$
C_q(X)=E_q\{H(\mathbf A,X)^\mathsf TB(\mathbf A,X)\mid X\}.
$$

Under no carryover, the exact conditional bias of (E1.5) is

$$
\{L(X)-C_q(X)\}\{\eta(X)-\gamma(X)\}.
\tag{E1.29}
$$

Thus a correct mean nuisance can rescue a wrong calibration law under no
carryover, but arbitrary nuisance robustness is lost.  With carryover there is
also the direct term $E_q(H^\mathsf T\delta\mid X)$.  This justifies parallel
product-versus-without-replacement analyses when the platform transition law
is uncertain, but it does not verify either law.

For the survey-weighted score, the corresponding population bias is

$$
E\left[R\{L-C_q\}(\eta-\gamma)
+R E_q(H^\mathsf T\delta\mid X)\right].
\tag{E1.29c}
$$

## 12. Numerical verification

The exhaustive finite-support calculation in
`analysis/feasible_eif/check_gate_e1_theory.R` verifies 49 identities and
failure modes, including:

- exact bias with a deliberately wrong mean nuisance;
- invariance modulo the no-carryover surface;
- the sharp directional bound and vector bias ellipsoid;
- equation (E1.14);
- the squared-weight population-tilting extension;
- blend bias, variance, and minimax identities;
- the generic feasible-weight blend with its variance cross term;
- nuisance recoding and target-coordinate equivariance;
- structured lag-only and saturated carryover subspaces;
- singular bias ranges and family point-zero tipping values;
- a nonzero carryover direction invisible to the paired diagnostic;
- the analytic two-task bias $-\rho\kappa$; and
- the assignment-law error formula (E1.29).

All 49 checks passed at maximum numerical error below $2\times10^{-15}$.

## 13. Novelty assessment and Gate E1 verdict

The generic ingredients are established: specification comparison,
local-misspecification asymptotics, approximate moments, and minimax blending
are covered by prior work including Hausman, Armstrong--Kolesár,
Bonhomme--Weidner, Lumley, and DiTraglia.  Most importantly,
[Armstrong, Kline, and Sun](https://doi.org/10.3982/ECTA21991) directly study
adaptation between restricted-efficient and unrestricted-robust estimators,
bias bounds, shrinkage, and the weakness of pretesting.  A carryover pretest or
generic adaptive blend is therefore not a new contribution.

The sensitivity interpretation is also adjacent to influence and moment maps
in [Andrews, Gentzkow, and Shapiro](https://doi.org/10.1093/qje/qjx023) and
[Ichimura and Newey](https://doi.org/10.3982/QE826).

The covariance-weighting side also sits within optimal conditional moments and
correlated-outcome estimation; see
[Chamberlain](https://doi.org/10.1016/0304-4076(87)90015-7),
[Liang and Zeger](https://doi.org/10.1093/biomet/73.1.13), and
[Stephens, Tchetgen Tchetgen, and De Gruttola](https://doi.org/10.1515/ijb-2013-0031).
Modern causal analysis of carryover in crossover trials includes
[Shi and Ye](https://doi.org/10.1093/biomtc/ujae023).

The defensible factorial contribution is narrower:

1. the exact bias operator for a target-distribution-weighted factorial family
   under a known repeated-task randomization law;
2. the quotient geometry separating own-profile response surfaces from genuine
   cross-task mean departures;
3. the coding-equivariant sharp bias ellipsoid;
4. the unrestricted-ball identity $K=T(\Omega_0-\Omega_*)$ and its
   scalar-direction sensitivity-indexed blend; and
5. exact-calibration and cross-fitting conditions that make this geometry
   computationally usable without estimating the full joint response surface.

This clears Gate E1 only as an internal theory-feasibility result.  It
strengthens a full paper when combined with the EIF, scalable factorial
calibration, Gate D's feasible estimator, and Gate E2 evidence.  It is not, by
itself, a publication-novelty pass or a claim to have invented local
robustness, adaptation, or Hausman testing.

## 14. Consequence for Gate E2

Gate E2 should now be a bounded computational exercise.  It should compare:

- unrestricted-model-valid ordinary taskwise AIPW, plus a full-joint nuisance
  implementation where feasible;
- the selected covariance-adaptive estimator;
- fixed $\tau$ blends; and
- the oracle sensitivity-indexed blend (E1.24) and a generic feasible blend
  computed from (E1.24a).

It should use residual-standardized departures calibrated by
$\sqrt G\,\varepsilon_{\Sigma^{-1}}$, including least-favourable, lag-only,
blind-direction, and benign carryover patterns.  It must verify bias, MSE,
ordinary and bias-aware coverage, diagnostic power, and assignment-law
misspecification.  The ordinary estimator remains the confirmatory default
unless a scientifically credible carryover bound lies below the relevant
tipping value.
