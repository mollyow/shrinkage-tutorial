# Weak-overlap theorem gate: decision

**Decision date:** 2026-08-31  
**Status:** **FAIL as a stand-alone methods theorem; retain the factorial
identities as supporting propositions and design diagnostics.**  
**Protocol:** [`weak-overlap-gate-protocol.md`](weak-overlap-gate-protocol.md)  
**Executable calculation:**
[`analysis/weak_overlap/`](../analysis/weak_overlap/README.md)

## Bottom line

Weak overlap creates a clean factorial calculation, and structural assumptions
can greatly reduce worst-case risk relative to full-profile weighting.  The
calculation does **not**, however, produce the new theorem needed for a full
methods paper.  Once the response surface is expressed in an
assignment-orthonormal basis, the candidate estimator, risk, modulus, and
regular/irregular phase transition are the classical solution to estimating a
linear functional over a convex ellipsoid.

The useful factorial residue is narrower:

- an exact elementary-symmetric formula showing which interaction orders drive
  the variance of a target factorial contrast under product shift; and
- a Boolean-lattice formula showing exactly how omitted supersets of the target
  contrast leak into a degree-truncated estimate.

Those are worth preserving as propositions.  They do not by themselves support
a claim to a new minimax or DML method.

## 1. Algebraic reduction

Let $A_j\in\{-1,1\}$, let the target distribution be uniform, and let the
assignment distribution be product with

\[
E_g(A_j)=\delta_j,
\qquad
v_j=1-\delta_j^2.
\]

The target-Walsh characters are
$\phi_T(A)=\prod_{j\in T}A_j$.  The assignment-orthonormal characters are

\[
\psi_T(A)=\prod_{j\in T}\frac{A_j-\delta_j}{\sqrt{v_j}}.
\]

If

\[
m(A)=\sum_T\beta_T\phi_T(A)=\sum_T\gamma_T\psi_T(A),
\]

then the triangular change of basis is

\[
\gamma_T
=
\sqrt{v_T}
\sum_{U\supseteq T}\beta_U\delta_{U\setminus T},
\]

and its inverse gives the target coefficient

\[
\boxed{
\beta_S
=
\frac{1}{\sqrt{v_S}}
\sum_{V\subseteq S^c}
(-1)^{|V|}\frac{\delta_V}{\sqrt{v_V}}
\gamma_{S\cup V}.
}
\]

Here $v_T=\prod_{j\in T}v_j$ and
$\delta_T=\prod_{j\in T}\delta_j$.  The full unbiased rule therefore has
variance factor

\[
\prod_{j=1}^J v_j^{-1}.
\]

With a fixed nonzero shift, this grows exponentially in the number of factors.
It is the exact second moment of the full target-to-assignment density ratio.

## 2. What exact low degree buys

Suppose the response surface has exact degree at most $q$.  For a target set
$S$, write $s=|S|$ and $d=q-s$.  In the Gaussian sequence experiment, or an
exactly orthogonal fixed design, the minimum variance is

\[
\boxed{
\frac{\sigma^2}{n v_S}
\sum_{\substack{V\subseteq S^c\\|V|\le d}}
\frac{\delta_V^2}{v_V}.
}
\]

For equal shifts $\delta_j=\delta$ and $v=1-\delta^2$, this becomes

\[
\frac{\sigma^2}{n v^s}
\sum_{k=0}^{d}
{J-s\choose k}
\left(\frac{\delta^2}{v}\right)^k.
\]

It is polynomial in $J$ for fixed excess degree $d$, whereas the saturated
variance is exponential.  For iid random profiles this is a population Fisher
information calculation.  Achievability additionally needs the relevant
empirical Gram matrix to concentrate and enough realized support.

## 3. What approximate low degree costs

The degree-$q$ estimator is generally biased if higher-order terms are merely
small rather than zero.  An omitted coefficient $\beta_{S\cup W}$ with
$h=|W|>d$ enters with exact multiplier

\[
\boxed{
(-1)^d {h-1\choose d}\delta_W.
}
\]

No omitted coefficient that is not a superset of $S$ contaminates the estimate
under product assignment.  This clean supersets-only statement need not survive
nonproduct or structurally restricted assignment.

For a tail ellipsoid

\[
\sum_{h>d}a_{s+h}^2
\sum_{|W|=h}\beta_{S\cup W}^2\le R^2,
\]

the exact worst-case squared bias of truncation is

\[
\boxed{
R^2\sum_{h>d}
\frac{{h-1\choose d}^2}{a_{s+h}^2}
\sum_{|W|=h}\delta_W^2.
}
\]

This is the main factorial-specific diagnostic.  It shows why “approximately
low order” cannot be invoked verbally: the combinatorial number of omitted
interactions may overwhelm coefficient-wise decay.

For example, with equal shifts and a plain tail-radius bound, the amplification
contains

\[
\sum_{h>d}{J-s\choose h}{h-1\choose d}^2\delta^{2h},
\]

which grows exponentially in $J$ up to polynomial factors for fixed $d$ and
fixed $|\delta|>0$.  A fixed-degree estimator therefore needs an increasingly
strong aggregate-tail restriction to remain uniformly honest as the factorial
dimension grows.

## 4. The oracle rule is generic minimax linear estimation

Let $G$ be the target-Walsh Gram matrix under assignment, let $e_S$ select the
target coefficient, and place the response surface in

\[
\beta^\top W\beta\le R^2.
\]

For a linear coefficient load $q$, worst-case risk in the population Gaussian
experiment is

\[
\frac{\sigma^2}{n}q^\top G^{-1}q
+R^2(q-e_S)^\top W^{-1}(q-e_S).
\]

The optimizer is

\[
\boxed{
q^*=
\left\{\frac{\sigma^2}{n}G^{-1}+R^2W^{-1}\right\}^{-1}
R^2W^{-1}e_S.
}
\]

This is generalized ridge/optimal recovery.  Factorial order weights specify
the ellipsoid, but do not alter the underlying theorem.

### Smallest weak-overlap subproblem

The issue is already complete with two factors.  Let

\[
m(a_1,a_2)=
\beta_0+\beta_1a_1+\beta_2a_2+\gamma a_1a_2,
\qquad |\gamma|\le B_n,
\]

and target $\theta=\beta_1$.  In assignment-orthonormal Gaussian coordinates,

\[
Z_1=\sqrt{v_1}(\beta_1+\delta_2\gamma)
    +\frac{\sigma}{\sqrt n}\varepsilon_1,
\qquad
Z_{12}=\sqrt{v_1v_2}\gamma
    +\frac{\sigma}{\sqrt n}\varepsilon_{12}.
\]

The minimax-linear estimator partially corrects the interaction coordinate,
with shrinkage

\[
w^*=
\frac{v_1v_2B_n^2}{v_1v_2B_n^2+\sigma^2/n},
\]

and risk

\[
R^*_{\mathrm{lin}}
=
\frac{\sigma^2}{nv_1}
+
\frac{\delta_2^2B_n^2(\sigma^2/n)}
     {v_1v_2B_n^2+\sigma^2/n}.
\]

A two-point lower bound gives the same rate, though not necessarily the same
constant, over all estimators:

\[
R^*\asymp
\frac{sigma^2}{nv_1}
+\delta_2^2
\min\left\{B_n^2,\frac{\sigma^2}{nv_1v_2}\right\}.
\]

For the lower bound, first vary the unrestricted focal coefficient to obtain
the Gaussian-location term $\sigma^2/(nv_1)$.  Then choose two values of
$\gamma$ separated by

\[
\min\left\{B_n,\frac{\sigma}{\sqrt{nv_1v_2}}\right\},
\]

and offset $\beta_1$ so that the two distributions have the same $Z_1$ mean.
Only the $Z_{12}$ means differ, by at most a constant multiple of their noise
standard deviation, while the target values differ by the second term's square
root.  Le Cam's two-point argument supplies that lower bound.  The maximum of
the two lower bounds is within a factor two of their sum, and the displayed
linear rule supplies the matching upper bound.

Thus an ML nuisance learner cannot remove the information problem in this
submodel.  It can improve neither the rate nor the overlap boundary without a
stronger structural assumption; nonlinear rules may still improve constants.

If $v_{2,n}\asymp n^{-a}$, $B_n\asymp n^{-b}$, and focal overlap $v_1$ is
bounded below, then

\[
R^*\asymp n^{-1}+n^{-\max(2b,1-a)}.
\]

Consequently, root-$n$ MSE requires either stable background overlap or an
interaction radius of order $n^{-1/2}$; consistency requires either
$n v_{2,n}\to\infty$ or $B_n\to0$.  If $B_n$ stays bounded away from zero and
$n v_{2,n}=O(1)$, consistent estimation is impossible.  A centered regular
root-$n$ limit is stronger than merely attaining root-$n$ worst-case MSE with
restriction-induced bias.

These lower-bound statements remain informative under iid random assignment:
random realized support cannot be more informative than the corresponding
population experiment, and indistinguishable rare-stratum alternatives remain
indistinguishable when their expected counts stay bounded.  The matching
population upper bound does not automatically transfer.  That direction needs
empirical-Gram concentration, occupancy, and a feasible estimator analysis.

## 5. Deterministic verification and risk screen

The executable calculation produced 291 direct-matrix checks.  All passed the
prespecified $10^{-10}$ tolerance:

| Identity | Maximum absolute error |
|---|---:|
| Basis cross-product | $4.44\times10^{-16}$ |
| Exact-degree variance | $1.78\times10^{-15}$ |
| Full Riesz variance | $7.11\times10^{-15}$ |
| Symmetry-reduced minimax solve | $4.55\times10^{-15}$ |
| Truncation contamination | $2.66\times10^{-15}$ |

The frozen grid contains 240 design/ellipsoid cells and 1,872 method rows.
Defining weak overlap as a full density-ratio second moment of at least 10:

- 84 cells met the weak-overlap definition;
- in 77 of 84, the ellipsoid-oracle linear rule had at least 10% lower
  worst-case MSE than both the full Riesz rule and the best prespecified
  degree truncation;
- all 84 oracle bias-aware intervals were shorter than the radius-one trivial
  interval; and
- 33 of the 77 apparent wins had more profiles than observations and were
  explicitly marked optimistic.

The strongest grid result was $J=12$, $n=16{,}000$, $\delta=.7$, and
$\kappa=1.5$.  The density-ratio second moment was 3,229.7, leaving a full-Riesz
effective sample size of only 5.0.  Oracle minimax risk was 0.032 versus 0.202
for full Riesz and 4.61 for the best degree-at-most-five truncation; its
bias-aware half-length was 0.384.

This is evidence that a declared structural class can contain substantial
decision-theoretic information.  It is not a feasible-method result: the
ellipsoid and noise level are known, the calculation uses the population Gram
matrix, and sparse realized occupancy is not simulated.

## 6. Reducibility and novelty audit

| Nearby area | What is already available | Consequence for this gate |
|---|---|---|
| [Classical Gaussian linear-functional theory](https://doi.org/10.1137/1129002) and [optimal recovery](https://doi.org/10.1214/aos/1176325367) | Minimax linear functionals over convex classes, moduli, and near-minimax affine rules | The two-factor estimator, risk, and phase transition are direct instances. |
| [Minimax retargeted means](https://arxiv.org/abs/1901.10296) | Minimax linear target-population estimation over RKHS classes, dual to regularized regression | A target-Walsh ellipsoid is an RKHS specialization. |
| [Optimal affine inference](https://doi.org/10.3982/ECTA14434) | Bias-aware, finite-sample optimal intervals for linear functionals over convex regression classes | The proposed honest interval is covered once the class is fixed. |
| [Linear regression under distribution shift](https://proceedings.mlr.press/v139/lei21a.html) | Minimax linear rules and constant-factor comparisons with unrestricted estimators | The population source-to-target regression problem is directly adjacent. |
| [RKHS regression under covariate shift](https://arxiv.org/abs/2205.02986) | Minimax rates with bounded or finite-second-moment density ratios | Weak target/assignment overlap plus an order kernel is not a new category. |
| [High-dimensional treatments](https://arxiv.org/abs/2602.21423) | Impossibility without structure, sparse pseudo-outcome regression, approximate sparsity, and minimax lower bounds | A broad “ML solves high-dimensional factorial positivity” claim is occupied. |
| [Factored-action OPE](https://arxiv.org/abs/2307.07014) and [minimax value intervals](https://proceedings.neurips.cc/paper/2020/hash/1cd138d0499a68f4bb72bee04bbec2d7-Abstract.html) | Exact factorizations and bias/coverage tradeoffs under insufficient policy overlap | The same curse-of-dimensionality/structural-bias tradeoff appears in an adjacent field. |
| [Synthetic Combinations](https://arxiv.org/abs/2303.14226) | Low-rank unit structure plus sparse Boolean-Fourier outcomes for combinatorial interventions | Stronger Fourier structure is already used to extrapolate across unobserved combinations. |

The audit does not show that every displayed factorial identity has appeared in
exactly this notation.  It shows that the central minimax theorem is obtained by
substitution into established theory.  Rebranding the coordinates as factorial
effects is not enough for a stand-alone theoretical contribution.

## 7. Gate decision

| Rule | Result | Reason |
|---|---|---|
| 1. Nontrivial approximate regime | **Not established** | The oracle grid is encouraging, but fixed-degree honest truncation suffers combinatorial leakage and no iid random-design growing-$J$ theorem was proved. |
| 2. Intrinsic factorial result | **Fail** | The lower bound and phase transition reduce to the generic Gaussian linear-functional modulus. |
| 3. Computational relevance | **Stopped** | The population screen shows headroom, but feasibility under sparse random occupancy was not run after the hard novelty failure. |
| 4. Robustness | **Not established** | The positive screen assumes the prespecified, known geometric ellipsoid and a population Gram matrix. |
| 5. Novelty | **Fail** | The estimator and honest inference are classical minimax linear/optimal-affine procedures after a factorial basis change. |

Rule 5 was a prespecified hard stop.  The random-design simulation was therefore
not launched: even a favorable result could not reverse the stand-alone
novelty decision.

## 8. What to retain and what would reopen the gate

Retain the exact-degree overlap formula and the omitted-superset contamination
formula in a broader paper, tutorial, or design note.  They give researchers a
simple answer to two practical questions: which orders must be estimated for a
given contrast, and how dangerous is approximate rather than exact truncation?

Reopen a full methods-paper gate only around an ingredient that does not reduce
to the sequence problem above.  The three plausible candidates are:

1. **Nonproduct or structural-zero assignment:** characterize new factorial
   alias geometry when the supersets-only triangular reduction fails.
2. **Finite-experiment equivalence under growing sparse occupancy:** prove
   sharp conditions under which the population sequence bound is achievable
   from random profiles, and an impossibility result when it is not.
3. **Honest adaptation to unknown interaction decay:** determine whether one
   can adapt across substantively meaningful order classes without paying the
   standard impossibility cost for honest inference.

Each would require a new frozen gate.  The present result should not be expanded
into a stand-alone weak-overlap paper.
