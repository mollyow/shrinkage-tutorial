# Restricted-support alias-geometry gate

**Frozen:** 2026-08-31, before retaining theorem or numerical results.  
**Purpose:** test whether structural restrictions on randomized factorial
profiles create a genuinely new estimability theorem for target-distribution
factorial contrasts, rather than a relabelling of classical
$c$-estimability, fractional-factorial aliasing, or optimal recovery.

## Question

Let the scientific target distribution put mass on the full factorial profile
space while the assignment distribution is supported only on a constrained
set $\Omega$.  Can a fixed target-Walsh coefficient be identified and
efficiently estimated under a low-order response model even though ordinary
target-to-assignment weighting does not exist?  If so, does factorial geometry
yield a theorem not already implied by the row space of a restricted design
matrix?

## Model

Let $A\in\{-1,1\}^J$, with uniform scientific target $\mu$.  Write

\[
m(A)=\sum_{T\subseteq[J]}\beta_T\phi_T(A),
\qquad
\phi_T(A)=\prod_{j\in T}A_j,
\]

and target $\theta_S=\beta_S$.  Assignment has known probabilities
$g(a)>0$ for $a\in\Omega$ and zero probability otherwise.

The primary exact model is $\beta_T=0$ for $|T|>q$.  Let $X_{\Omega,q}$ be
the evaluation matrix whose columns are all $\phi_T$ with $|T|\le q$.
Calculations use a homoskedastic Gaussian cell-mean experiment with fixed
allocation proportional to $g$.  This isolates aliasing from random cell-count
noise; any positive result must later be checked under iid random occupancy.

The approximate model allows higher-order coefficients in a prespecified
target-Walsh ellipsoid.  Exact identification and restriction-dependent
partial identification must not be conflated.

## Canonical support families

The gate must cover at least:

1. **Parity half-fraction:**
   $\Omega=\{a:\prod_j a_j=1\}$.
2. **Single Hamming slice:**
   $\Omega_k=\{a:\#\{j:a_j=1\}=k\}$.
3. **Unions of Hamming slices:** at least two separated layers and enough
   layers to test degree-$q$ interpolation.
4. **Forbidden pair:** all profiles except those with $A_1=A_2=1$.
5. **Generic matched-size support:** uniformly sampled subsets of the cube,
   used only as a rank benchmark rather than a theorem opportunity.

## Required results

Before any performance claim:

1. state and verify the necessary-and-sufficient row-space condition for a
   target contrast to be estimable under exact degree $q$;
2. derive the minimum-variance representer and variance when it exists;
3. give a constructive indistinguishable-pair lower bound when it does not;
4. derive the exact high-order leakage vector and ellipsoid worst-case bias;
5. characterize the alias classes for the parity fraction;
6. characterize rank and estimable target functionals on a Hamming slice or a
   union of slices, including all boundary qualifications; and
7. identify precisely which statements survive nonuniform probabilities and
   iid sparse occupancy.

All formulas used in the decision must agree with direct singular-value or
Moore--Penrose calculations for every enumerated case to absolute tolerance
$10^{-10}$.

## Deterministic screen

Enumerate $J=3,\ldots,10$ and $q=0,\ldots,\min(4,J)$ where feasible.  For each
canonical support, record:

- number of supported profiles;
- model dimension and restricted-design rank;
- nullity;
- which individual Walsh coefficients are estimable;
- minimum variance factors for estimable coefficients;
- alias or leakage coefficients for prespecified order-one and order-two
  targets; and
- agreement with any claimed closed form.

Random generic supports use deterministic seeds and at least three supports per
size.  They are controls for the fact that rank is often generic; they do not
establish substantive novelty.

## Pass rules

The route passes only if all five conditions hold.

1. **Beyond row-space restatement.**  At least one sharp result characterizes
   factorial target estimability or information using the structure of a
   scientifically relevant constrained support, without simply saying to rank
   or invert $X_{\Omega,q}$.
2. **Beyond classical aliasing.**  The result is not a defining-relation alias
   table, a direct application of $c$-estimability/Gauss--Markov theory, a
   design-ideal computation, or a standard harmonic-polynomial fact on a
   slice.
3. **Causal relevance.**  The result identifies a target-distribution contrast
   that ordinary overlap cannot identify and supplies a usable design or
   inference implication.
4. **Approximate-model honesty.**  The claimed gain remains meaningful under a
   transparent approximate restriction, or the theorem is explicitly limited
   to exact models with an unavoidable sensitivity bound.
5. **Novelty.**  A primary-source audit leaves a theorem-sized contribution
   after accounting for fractional-factorial design, constrained experimental
   design, algebraic statistics, Boolean/slice harmonic analysis, and causal
   partial-identification or overlap theory.

Failure of rule 2 or rule 5 is a hard stop for a stand-alone methods paper.
Failure of rule 4 prevents general causal-inference claims even if an exact
design theorem survives.

## Mandatory novelty audit

The decision must map the candidate result to:

- estimability of linear parametric functions and the Gauss--Markov theorem;
- regular and nonregular fractional-factorial defining relations and aliasing;
- algebraic-statistical design ideals and circuits;
- harmonic multilinear polynomials and low-degree functions on the slice;
- optimal design for a specified contrast;
- constrained randomization and restricted treatment combinations; and
- partial identification or minimax inference under support violations.

## Sequential stopping rule

This gate is analytic first.  If the exact support theorem is classical after
the mandatory audit, retain the rank/leakage diagnostic but do not run a large
Monte Carlo experiment.  A favorable oracle variance calculation cannot
reverse a failed novelty gate.

## Pivot rule

On failure, preserve a general support/rank diagnostic for the software or
tutorial paper.  Reopen a theorem gate only for an ingredient that changes the
problem, such as optimal selection of constrained profile support for a fixed
target contrast, partial identification with scientifically calibrated
off-support smoothness, or fusion of several experiments whose support union
restores factorial estimability.
