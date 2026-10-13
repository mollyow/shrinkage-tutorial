# Minimal support-augmentation gate

**Frozen:** 2026-08-31, before retaining augmentation calculations.  
**Purpose:** determine whether choosing a small number of additional feasible
factorial profiles for prespecified causal contrasts yields a new theorem, or
only standard design augmentation, foldover, $c$-optimality, and row selection.

**Decision:** **FAIL as a stand-alone theorem**; see
[`support-augmentation-gate-decision.md`](support-augmentation-gate-decision.md).

## Scientific question

An existing experiment assigns profiles in $\Omega_0$ and targets one or more
uniform-distribution factorial contrasts.  Some additional profiles are
scientifically feasible but were not included; others are structurally
impossible.  Given a budget, which feasible profiles should be added to:

1. make the target contrasts identifiable under an exact low-degree model;
2. minimize sampling variance; and
3. reduce sensitivity to omitted higher-order interactions?

The gate must keep two kinds of zeros separate:

- **design zeros** can be repaired by augmentation;
- **structural zeros** cannot be repaired if all feasible profiles are already
  present, and continue to require extrapolation assumptions.

## Model and exact-identification problem

Let $x_L(a)$ contain target-Walsh characters through degree $q$.  Let
$X_0$ be their evaluation matrix on the existing support, let
$\mathcal A$ be the set of feasible candidate profiles, and let the columns of
$C$ define a fixed finite set of target contrasts.

For a selected augmentation $S\subseteq\mathcal A$, exact identification
requires

\[
\operatorname{col}(C)
\subseteq
\operatorname{row}
\begin{pmatrix}X_0\\X_S\end{pmatrix}.
\]

The minimum-cardinality problem is

\[
\min_{S\subseteq\mathcal A}|S|
\quad\text{subject to the displayed containment.}
\]

The gate must derive its quotient-space form.  If $P_0$ projects onto the
orthogonal complement of $\operatorname{row}(X_0)$, augmentation selects a
small dictionary of projected rows $P_0x_L(a)$ whose span contains $P_0C$.
Calling this a factorial problem does not make generic sparse representation or
row completion new.

## Precision and robustness objective

Let $n_a$ be the fixed existing cell allocation plus added integer counts, with
a fixed augmentation budget.  For one target vector $c$, signed cell weights
$\zeta$ must satisfy

\[
X_L^\top\zeta=c.
\]

With homoskedastic noise and an omitted-interaction ellipsoid
$\|W^{1/2}\beta_H\|_2\le B$, the prespecified robust linear-risk objective is

\[
\boxed{
\min_{n,\zeta}
\left{
\sigma^2\sum_a\frac{\zeta_a^2}{n_a}
+B^2\|W^{-1/2}X_H^\top\zeta\|_2^2
\right}
}
\]

subject to feasibility, the exact low-order moment constraint, integer counts,
and the augmentation budget.  Terms with $n_a=0$ require $\zeta_a=0$.

Special cases must be labelled correctly:

- $B=0$: exact-model $c$-optimal design augmentation;
- fixed support and allocation: minimax linear estimation over an ellipsoid;
- support-cardinality constraint: row/sensor selection;
- adding the complementary regular fraction: foldover design.

## Canonical analytic cases

The gate must cover:

1. **Parity fraction:** augmentation from the complementary half-fraction,
   including both exact de-aliasing and approximate-tail sensitivity.
2. **Hamming-layer design:** augmentation by whole layers and by individual
   profiles.  The whole-layer problem must be mapped to the known affine
   Hilbert-function rank criterion.
3. **Forbidden pair:** if every scientifically feasible profile is already in
   the design, record augmentation as impossible rather than proposing an
   infeasible profile.
4. **Irregular support:** small exact enumerations used to test whether greedy
   rank or variance gains have a factorial guarantee.

## Required analytic results

Before a paper-level claim:

1. derive the quotient-space/sparse-representation characterization for fixed
   target contrasts;
2. distinguish minimum augmentation for full model rank from minimum
   augmentation for the requested contrasts;
3. derive the one-profile generalized-inverse or Sherman--Morrison variance
   update where regularity permits;
4. state the robust alias-variance optimization and its dual or lower bound;
5. prove the optimal whole-layer augmentation rule for unions of Hamming
   slices, including infeasibility under structural constraints;
6. determine whether variance or alias-radius improvement is submodular,
   weakly submodular, or admits an explicit counterexample; and
7. identify which results survive integer allocation and iid random occupancy.

Every closed-form claim used in the decision must agree with exhaustive small
factorial enumeration to absolute tolerance $10^{-10}$.

## Deterministic screen

Use $J=3,\ldots,8$ and $q\le3$ where exhaustive enumeration is feasible.
Record:

- initial rank and target estimability;
- minimum number of individual profiles or whole layers required;
- rank gain and target-specific identification by budget;
- exact-model variance factor;
- unweighted full-tail alias radius;
- robust risk for a small frozen grid of $B/\sigma$ values; and
- greedy versus exhaustive optimum for small irregular supports.

Large Monte Carlo simulation is not authorized until the novelty gate passes.

## Pass rules

The stand-alone route passes only if all conditions hold.

1. **Factorial theorem:** there is a sharp augmentation result beyond row-space
   completion, sparse dictionary representation, or a direct corollary of a
   known factorial defining relation.
2. **Optimization contribution:** the joint variance-plus-alias problem yields
   a new exact solution or approximation guarantee under a substantively
   meaningful factorial feasibility class.
3. **Causal relevance:** the result changes design advice for prespecified
   target-distribution contrasts while respecting structural impossibilities.
4. **Approximate-model honesty:** the method controls, rather than hides,
   off-support sensitivity.
5. **Novelty:** the result survives comparison with foldover and follow-up
   designs, exact and approximate $c$-optimal design, robust design under model
   misspecification, and sensor/row-selection theory.

Failure of rule 1 or rule 5 is a hard stop.  A useful algorithm without a new
guarantee is software functionality, not a full methods theorem.

## Mandatory novelty audit

The decision must cover primary work on:

- full and partial foldover or follow-up fractional factorial designs;
- exact design augmentation and exchange algorithms;
- Elfving/$c$-optimal design with singular information matrices;
- compound or minimax design under model misspecification;
- sensor, feature, and row subset selection;
- submodularity or weak submodularity of experimental-design criteria;
- augmentation of symmetric/Hamming-layer supports; and
- causal design under restricted treatment combinations.

## Sequential stopping and pivot

If the theorem reduces to established design optimization, stop before a large
simulation.  Preserve a practical `suggest_augmentation()` diagnostic if it
adds software value.

After a failure here, the project should stop opening adjacent theorem gates.
The default route becomes a software/diagnostic or negative comparative paper
using the accumulated invariant estimation, overlap, and support results.
