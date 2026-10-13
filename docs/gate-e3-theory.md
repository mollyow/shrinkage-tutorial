# Gate E3 decision: Gaussian adaptation and structured carryover

**Decision: FAIL as a stand-alone-paper novelty gate.**

The exercise succeeds mathematically but answers the novelty question
negatively. For one factorial contrast, the local problem is exactly the
restricted-versus-unrestricted Gaussian experiment studied by Armstrong,
Kline, and Sun. Perfect oracle selection is impossible, but useful automatic
adaptation is possible and has already been optimized. The proposed broad
claim that ML or another data-driven rule cannot add value would therefore be
incorrect.

A prespecified carryover basis does yield a useful intermediate estimator. It
is efficient and exactly centered in a mean model that permits those carryover
paths. But its derivation is the existing covariance-calibration EIF applied to
the augmented mean design `[B, Z]`, not new statistical decision theory.

## Plain-language result

For a fixed contrast, the restricted and unrestricted estimates reveal only
one noisy measurement of the restricted estimate's carryover bias. Local bias
shrinks at exactly the same rate as sampling noise, so more respondents do not
make that standardized bias consistently learnable.

That prevents perfect adaptation: no rule can always behave exactly like the
restricted estimate when it is safe and exactly like the unrestricted estimate
when it is dangerous. It does **not** prevent partial adaptation. Existing
soft-threshold and minimax-regret rules make the best attainable compromise.

Scientific structure can do something different. If lag-one carryover is
plausible and prespecified, we can allow lag-one carryover in the mean model and
derive its efficient score. This estimator is:

- exactly robust to arbitrary carryover within the stated lag subspace;
- more efficient than ordinary taskwise scoring when the subspace does not
  exhaust every possible complete-schedule mean; and
- less efficient than the no-carryover score, reflecting the price of the
  additional protection.

## 1. Exact scalar Gaussian reduction

Fix a scalar factorial direction `a`. At a no-carryover reference law, write

\[
v_* = a^\mathsf T\Omega_*a,
\qquad
\Delta = a^\mathsf T(\Omega_0-\Omega_*)a>0,
\qquad
b=a^\mathsf T\beta.
\]

Here `*` denotes the restricted-efficient score and `0` denotes ordinary
unrestricted-valid taskwise scoring. Gate E1 gives

\[
\sqrt G
\begin{pmatrix}
\widehat\theta_R-\theta_{U,G}\\
\widehat\theta_U-\theta_{U,G}
\end{pmatrix}
\rightsquigarrow
N\!\left[
\begin{pmatrix}b\\0\end{pmatrix},
\begin{pmatrix}
v_* & v_*\\
v_* & v_*+\Delta
\end{pmatrix}
\right].
\tag{E3.1}
\]

Define the Hausman difference

\[
W=\sqrt G(\widehat\theta_R-\widehat\theta_U).
\]

The invertible transformation from
`(restricted, unrestricted)` to `(restricted, difference)` turns (E3.1) into

\[
X\sim N(h+b,v_*),
\qquad
W\sim N(b,\Delta),
\qquad
X\perp W,
\tag{E3.2}
\]

where `h` is the local target. Projection efficiency is exactly what produces
the independence.

Every translation-equivariant estimator can be written

\[
\widehat h_g=X-\sqrt\Delta\,g(Z),
\qquad
Z=W/\sqrt\Delta\sim N(u,1),
\qquad
u=b/\sqrt\Delta.
\tag{E3.3}
\]

Its local squared-error risk is

\[
\boxed{
R_g(u)=v_*+\Delta E_u\{g(Z)-u\}^2.
}
\tag{E3.4}
\]

Thus factorial adaptation is exactly the problem of estimating one normal
mean `u` from one observation. The restricted estimator uses `g=0`; the
unrestricted estimator uses `g(z)=z`. Their risks are

\[
R_R(u)=v_*+\Delta u^2,
\qquad
R_U(u)=v_*+\Delta.
\tag{E3.5}
\]

The better-endpoint oracle has risk

\[
R_{\mathrm{end}}(u)=v_*+\Delta\min(u^2,1).
\tag{E3.6}
\]

This reduction is exact in the Gate E1 limit, not an analogy.

### Where the factorial design enters

For the unrestricted residual-standardized carryover ball,

\[
K_{\Sigma^{-1}}(H^*)=T(\Omega_0-\Omega_*).
\]

The sharp bias bound therefore implies

\[
|b|\le \rho\sqrt{T\Delta},
\qquad
|u|\le \rho\sqrt T.
\tag{E3.7}
\]

For a structured class with directional operator
`B_a=a^\mathsf TK_Za`, the bound becomes

\[
|u|\le
\rho\sqrt{B_a/\Delta}.
\tag{E3.8}
\]

The randomization, target distribution, and carryover geometry determine
`v_*`, `Delta`, and the permitted set for `u`. Once those quantities are
known, the scalar decision problem contains no further factorial structure.

## 2. Fixed-radius linear minimax rule

Consider the linear correction `g(z)=lambda z`. It corresponds to

\[
\widehat h_\lambda
=X-\lambda W
=(1-\lambda)\widehat\theta_R
+\lambda\widehat\theta_U.
\]

For `|u|<=s`, its worst-case risk beyond `v_*` is

\[
\Delta\{\lambda^2+(1-\lambda)^2s^2\}.
\tag{E3.9}
\]

Direct differentiation gives

\[
\lambda_{\mathrm{lin}}(s)=\frac{s^2}{1+s^2}.
\tag{E3.10}
\]

Writing the Gate E1 weight on the restricted score as
`tau=1-lambda`, the unrestricted ball has `s^2=T rho^2` and hence

\[
\tau=\frac{1}{1+T\rho^2}.
\tag{E3.11}
\]

This recovers the Gate E1 formula exactly. Gate E3 clarifies its scope: it is
optimal among fixed linear blends. The exact minimax estimator of a bounded
normal mean is generally nonlinear.

## 3. What is and is not impossible

### Proposition E3.1: no exact endpoint-oracle adaptation

No translation-equivariant estimator can attain (E3.6) for every `u`.

**Proof.** To attain the endpoint oracle at `u=0`, equation (E3.4) requires

\[
E_0\{g(Z)^2\}=0.
\]

Therefore `g=0` almost surely under `N(0,1)`. All normal location laws are
mutually absolutely continuous, so `g=0` almost surely under every `N(u,1)`.
The rule then has excess risk `u^2`, which exceeds the unrestricted rule's
excess risk one whenever `|u|>1`. \(\square\)

One quantitative version follows immediately. If
`E_0[g(Z)^2]<=epsilon`, the normal likelihood ratio and Cauchy--Schwarz give

\[
|E_u g(Z)|
\le e^{u^2/2}\sqrt\epsilon.
\]

Jensen's inequality then yields

\[
E_u\{g(Z)-u\}^2
\ge
\left(
|u|-e^{u^2/2}\sqrt\epsilon
\right)_+^2.
\tag{E3.12}
\]

This displays the cost away from zero of forcing risk extremely close to the
restricted bound at zero. It is a simple Gaussian constrained-risk inequality,
not a new factorial lower bound.

### The broader impossibility claim is false

Equation (E3.12) rules out **perfect** oracle performance. It does not say that
automatic adaptation has no value. Armstrong, Kline, and Sun study essentially
(E3.2)--(E3.4), minimize worst-case proportional regret relative to an oracle,
and derive both an optimally adaptive rule and a near-optimal soft-threshold
rule. Their local bias is not consistently estimable, but it is still
informative enough to support imperfect risk adaptation.

Consequently, Gate E2's fixed full-ball blend should not be described as the
best possible automatic procedure. It is the best fixed linear blend for its
prespecified ball. An estimator that uses the realized Hausman difference can
adapt nonlinearly, although its inference and worst-case risk must be evaluated
using the existing adaptation theory rather than naive post-selection standard
errors.

There is an important regularity fork. If the estimator must be regular and
asymptotically normal throughout the unrestricted local-carryover model, its
information bound is `v_*+Delta`; it cannot also attain `v_*` at `b=0`.
Optimally adaptive threshold rules obtain null efficiency gains by being
nonregular in the local bias parameter. Their limiting distribution changes
with `b`, which is itself not consistently estimable. Consequently, ordinary
DDML-style normal standard errors do not automatically provide uniformly valid
inference for the adaptive point estimate.

## 4. Constructive structured-carryover score

Let `Z(A,X)` be a prespecified `T` by `k` carryover design, such as lag-one
factorial features. Enlarge the conditional-mean model to

\[
m(\mathbf A,X)
=C(\mathbf A,X)\xi(X),
\qquad
C=[B,Z],
\qquad
\xi=(\gamma^\mathsf T,\alpha^\mathsf T)^\mathsf T.
\tag{E3.13}
\]

This is an exact structural restriction: `alpha(X)` may have arbitrary
magnitude, but carryover outside the span of `Z` is excluded.

Preserve the unrestricted task-marginal target defined by `H_0`. Put

\[
L_Z(X)
=E_q\{H_0(\mathbf A,X)^\mathsf TZ(\mathbf A,X)\mid X\},
\qquad
\widetilde L(X)=[L(X),L_Z(X)].
\tag{E3.14}
\]

The structured target is then

\[
\theta_U
=E\{\widetilde L(X)\xi(X)\}.
\tag{E3.15}
\]

The `L_Z` block must not be silently set to zero. It is zero for some centered
lag features under product assignment, but not in general.

Define

\[
J_Z(X)
=E_q\{C^\mathsf T\Sigma^{-1}C\mid X\}.
\tag{E3.16}
\]

When the columns of `tilde L(X)^T` lie in the range of `J_Z(X)`, set

\[
\boxed{
H_Z(\mathbf A,X)
=\Sigma(\mathbf A,X)^{-1}C(\mathbf A,X)
J_Z(X)^+\widetilde L(X)^\mathsf T.
}
\tag{E3.17}
\]

### Proposition E3.2: structured efficient influence function

In model (E3.13), the efficient influence function is

\[
\phi_Z(O)
=\widetilde L(X)\xi(X)-\theta_U
+H_Z(\mathbf A,X)^\mathsf T
\{\mathbf Y-C(\mathbf A,X)\xi(X)\}.
\tag{E3.18}
\]

It is calibrated by

\[
E_q(C^\mathsf TH_Z\mid X)=\widetilde L^\mathsf T.
\tag{E3.19}
\]

It uniquely minimizes conditional residual variance among all weights
satisfying (E3.19), modulo null directions.

**Proof.** Apply the covariance-weighted calibration projection from
Proposition 2 of the feasibility memo with `B` replaced by `C` and `L` replaced
by `tilde L`. Equation (E3.19) follows from the generalized-inverse range
condition. Pathwise differentiation gives the same calibration constraint, and
projection in the `Sigma` inner product gives efficiency. \(\square\)

### Exact robustness and nested efficiency

Because `H_0` itself satisfies

\[
E_q(C^\mathsf TH_0\mid X)=\widetilde L^\mathsf T,
\]

it is feasible for the structured problem. At a no-carryover reference law,
the feasible calibration sets are nested. Therefore

\[
\boxed{
\Omega_*\preceq\Omega_Z\preceq\Omega_0.
}
\tag{E3.20}
\]

More precisely,

\[
\Omega_Z-\Omega_*
=E\{(H_Z-H^*)^\mathsf T\Sigma(H_Z-H^*)\},
\tag{E3.21}
\]

and

\[
\Omega_0-\Omega_Z
=E\{(H_0-H_Z)^\mathsf T\Sigma(H_0-H_Z)\}.
\tag{E3.22}
\]

Thus:

- `Z` empty returns the no-carryover efficient score `H*`;
- adding carryover columns can only increase the efficiency bound;
- a saturated carryover design that spans every complete-schedule mean forces
  the ordinary unrestricted score; and
- intermediate scientifically meaningful designs can retain part of the
  efficiency gain while remaining exactly valid for their allowed carryover.

For departures outside `span[B,Z]`, put `D_Z=H_Z-H_0` and measure distance from
that enlarged mean space. The same projection argument gives

\[
K_{\Sigma^{-1}}^{\perp[B,Z]}(H_Z)
=T(\Omega_0-\Omega_Z),
\tag{E3.23}
\]

so the remaining efficiency gain again equals vulnerability to omitted
carryover directions.

For a fixed or independently trained score `H` and the local structured class
`c=Z alpha` with radius `rho`, let `K_Z(H)` denote the Gate E1 bias operator
for that allowed class. A conservative scalar interval is

\[
a^\mathsf T\widehat\theta_H
\ \pm\ \frac1{\sqrt G}
\left{
z_{1-\alpha/2}\sqrt{a^\mathsf T\widehat\Omega_Ha}
+\rho\sqrt{a^\mathsf T\widehat K_Z(H)a}
\right}.
\tag{E3.23a}
\]

The second term is zero for `H_Z` when the declared `Z` model is exact. A
fixed finite contrast family can instead combine a simultaneous clustered
sampling critical value with the corresponding structured bias ellipsoid.
Both `Z` and `rho` must be fixed independently of the analyzed outcomes unless
selection is explicitly incorporated into the inference.

Standard cross-fitted clustered inference is valid under the exact structured
model when the learned nuisance functions and weights satisfy the required
rates. Gate E2 warns that its first-order risk approximation can still have a
material finite-sample remainder when saturated means are estimated with only
250--500 clusters.

## 5. Finite-radius direct structured score

If the analyst specifies a finite sensitivity radius rather than assuming all
carryover lies exactly in `Z`, the score itself can be optimized. This is
stronger than blending `H*` and `H_0` after they have been constructed.

For fixed `X=x`, enumerate supported schedules `s` and stack their weights into
`mathsf H`. Residualize `Z` off `B` in the chosen carryover metric to obtain
`Z_perp`, and define

\[
Q_Z(x)=\frac1T E_q(Z_\perp^\mathsf TMZ_\perp\mid X=x).
\]

Put

\[
V_x=\operatorname{blockdiag}_s\{q_s\Sigma_s\},
\quad
A_{B,x}=[q_sB_s^\mathsf T]_s,
\quad
A_{Z,x}=[q_sZ_{\perp,s}^\mathsf T]_s.
\]

The exact calibration constraint is
\(A_{B,x}\mathsf H=L(x)^\mathsf T\). For radius `rho`, minimize the matrix risk envelope

\[
\mathcal R_\rho(\mathsf H)
=\mathsf H^\mathsf TV_x\mathsf H
+\rho^2
\{A_{Z,x}(\mathsf H-\mathsf H_0)\}^\mathsf T
Q_Z(x)^+
\{A_{Z,x}(\mathsf H-\mathsf H_0)\}.
\tag{E3.24}
\]

Let

\[
P_{\rho,x}
=V_x+\rho^2A_{Z,x}^\mathsf TQ_Z(x)^+A_{Z,x},
\]

and

\[
G_{\rho,x}
=\rho^2A_{Z,x}^\mathsf TQ_Z(x)^+A_{Z,x}\mathsf H_0.
\]

Completing the square subject to calibration gives

\[
\boxed{
\begin{aligned}
\mathsf H_\rho
={}&P_{\rho,x}^{-1}G_{\rho,x}\\
&+P_{\rho,x}^{-1}A_{B,x}^\mathsf T
\{A_{B,x}P_{\rho,x}^{-1}A_{B,x}^\mathsf T\}^+
\{L^\mathsf T-A_{B,x}P_{\rho,x}^{-1}G_{\rho,x}\}.
\end{aligned}
}
\tag{E3.25}
\]

Positive assignment probabilities and positive-definite residual covariances
make `P_rho,x` positive definite; the usual range condition handles redundant
calibration columns. For every other calibrated score,

\[
\mathcal R_\rho(\mathsf H)-\mathcal R_\rho(\mathsf H_\rho)
=(\mathsf H-\mathsf H_\rho)^\mathsf T
P_{\rho,x}
(\mathsf H-\mathsf H_\rho)
\succeq0.
\tag{E3.26}
\]

At `rho=0`, this is `H*`. As `rho` diverges, it approaches the exact
`Z`-robust score `H_Z`, subject to the support and kernel conditions. Because
`H_0` and all endpoint blends are feasible competitors, `H_rho` weakly
dominates them under the declared structured worst-case criterion.

This is computationally useful, but it is the quadratic
approximate-moment/robust-calibration solution specialized to factorial
schedules. It does not change the Gate E3 novelty verdict.

## 6. Enumerated factorial check

The deterministic checker uses two independently randomized binary profiles,
two tasks, and residual covariance

\[
\Sigma=
\begin{pmatrix}1&.8\\.8&1\end{pmatrix}.
\]

The target is the average high-minus-low contrast across tasks. The structured
carryover basis allows task 2's mean to depend on task 1's assigned profile:

\[
Z(\mathbf A)=(0,A_1)^\mathsf T.
\]

Exact enumeration gives:

| Score | Residual variance | Bias per unit lag coefficient |
|---|---:|---:|
| No-carryover efficient `H*` | 0.72 | -0.80 |
| Lag-robust efficient `H_Z` | 1.36 | 0 |
| Ordinary taskwise `H_0` | 2.00 | 0 |

The structured score preserves half of the available variance improvement:
ordinary-to-restricted headroom is 1.28, while ordinary-to-structured headroom
is 0.64. It is invariant to scaling or redundantly coding the lag basis.

All 32 deterministic checks passed, including Gaussian independence, fixed-
radius minimax algebra, exact augmented calibration, bias annihilation,
variance nesting, projection identities, the direct structured-calibration
path, and carryover-basis recoding. The checker is
`analysis/feasible_eif/check_gate_e3_theory.R`.

## 7. Novelty audit

The overlap is direct rather than merely thematic.

| Proposed Gate E3 result | Closest established result | Verdict |
|---|---|---|
| One-contrast Gaussian reduction | Armstrong, Kline, and Sun's restricted/unrestricted normal experiment | Exact specialization |
| Unknown local bias cannot be consistently learned | Same paper; earlier pretest and local-misspecification literature | Established |
| Minimax regret and automatic adaptation | Optimally adaptive and soft-threshold rules in the same paper | Established |
| No exact oracle adaptation | Normal-mean admissibility and constrained-risk theory | Established |
| Known-radius minimax estimation | Bounded-normal-mean and approximate-moment theory | Established |
| Bias-aware intervals over convex carryover classes | Armstrong and Kolesar's approximate-moment sensitivity analysis | Established |
| Efficient score under `[B,Z]` | Existing covariance-weighted calibration applied to an enlarged linear mean model | Direct extension |
| Factorial map from carryover to local bias | Gate E1's target-weighted bias operator and quotient geometry | Useful factorial specialization |

Primary references are:

- [Armstrong, Kline, and Sun (2025), *Adapting to
  Misspecification*](https://doi.org/10.3982/ECTA21991).
- [Armstrong and Kolesar (2021), *Sensitivity Analysis Using Approximate Moment
  Condition Models*](https://doi.org/10.3982/QE1609).
- [Brown and Low (1996), *A Constrained Risk Inequality With Applications to
  Nonparametric Functional Estimation*](https://doi.org/10.1214/aos/1032181166).
- [Casella and Strawderman (1981), *Estimating a Bounded Normal
  Mean*](https://doi.org/10.1214/aos/1176345527).
- [Bonhomme and Weidner (2022), *Minimizing Sensitivity to Model
  Misspecification*](https://doi.org/10.3982/QE1930).

Armstrong, Kline, and Sun also treat vector-valued bias and multiple restricted
models in their supplement. Moving from one factorial contrast to a contrast
family therefore does not by itself restore novelty.

## 8. Gate decision and consequences

Gate E3 fails both novelty conditions in its protocol:

1. the adaptation lower bound loses its factorial structure after the exact
   Gaussian reduction; and
2. the structured score is obtained by applying the existing calibration EIF
   to an augmented design matrix.

The following claims must not be made:

- that generic ML has no value for factorial adaptation;
- that no useful data-driven adaptation is possible;
- that the fixed full-ball blend is optimally adaptive when the sensitivity
  radius is unknown; or
- that structured `Z` minimax estimation is new general decision theory.

The following results remain useful:

- the factorial design supplies an exact, coding-invariant map from carryover
  functions to the Gaussian bias parameter;
- the no-carryover and structured-carryover efficient scores can be computed by
  exact design calibration;
- structured mean models create a transparent efficiency ladder between `H*`
  and `H_0`; and
- ordinary unrestricted taskwise AIPW remains the assumption-light primary
  estimator.

A subsequent stand-alone-paper gate would need a result that does not disappear
inside the Gaussian reduction. The two plausible targets are:

1. a finite-`G` correction and rate theorem for the nuisance-estimation
   remainder exposed by Gate E2; or
2. a factorial schedule-design or rank theorem characterizing visible, blind,
   and protected carryover directions and optimizing the
   efficiency--robustness frontier over the randomization law.

Absent one of those, the appropriate product is a broader repeated-factorial
inference paper or software/tutorial contribution, not a new paper on
automatic adaptation.
