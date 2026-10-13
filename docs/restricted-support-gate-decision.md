# Restricted-support alias-geometry gate: decision

**Decision date:** 2026-08-31  
**Status:** **FAIL as a stand-alone methods theorem; retain the exact
estimability and alias-radius diagnostics.**  
**Protocol:**
[`restricted-support-gate-protocol.md`](restricted-support-gate-protocol.md)  
**Executable calculation:**
[`analysis/restricted_support/`](../analysis/restricted_support/README.md)

## Bottom line

Structural profile restrictions do create sharp and useful factorial geometry.
Under an exact low-degree response model, a target contrast can sometimes be
identified even though its target distribution includes profiles that can never
be assigned.  Under an approximate model, however, every proper support leaves
an irreducible off-support alias unless the approximation error itself shrinks
or is included in the interval.

This does not yield a new stand-alone theorem.  The arbitrary-support result is
ordinary estimability of a linear function; parity and forbidden-profile
relations are classical fractional-factorial aliasing; and the strongest
structured result—the rank of a low-degree model on a union of Hamming
slices—is an already-known affine Hilbert function.

The gate therefore separates two claims:

- **Useful:** coding-invariant software can diagnose exactly which causal
  factorial contrasts survive a restricted profile design and quantify their
  off-support sensitivity.
- **Not defensible:** a claim to invent identification, ML extrapolation, or a
  new alias-geometry theorem for arbitrary fractions or unions of slices.

## 1. Arbitrary-support estimability

Let $X_L=X_{\Omega,q}$ be the evaluation matrix on the supported profiles
$\Omega$, with columns $\phi_T$ for $|T|\le q$.  Let $e_S$ select the target
uniform-cube coefficient $\beta_S$.

The following statements are equivalent:

1. $\beta_S$ is identified under the exact degree-$q$ model;
2. $e_S\in\operatorname{row}(X_L)$;
3. every $d\in\ker(X_L)$ satisfies $d_S=0$; and
4. every degree-$q$ multilinear polynomial that vanishes on $\Omega$ has zero
   $\phi_S$ coefficient.

This gives a constructive impossibility result.  If the condition fails, choose
$d\in\ker(X_L)$ with $d_S\ne0$.  Response surfaces with coefficient vectors
$\beta$ and $\beta+td$ agree at every assignable profile but have different
target contrasts.  No estimator or ML learner can distinguish them.

For known probabilities $g$ on $\Omega$, put

\[
G=X_L^\top D_gX_L,
\qquad
D_g=\operatorname{diag}\{g(a):a\in\Omega\}.
\]

When the contrast is estimable, its minimum-$L_2(g)$ representer is

\[
r_S^*(a)=x_L(a)^\top G^+e_S,
\]

and

\[
E_g\{r_S^*(A)\phi_T(A)\}=1\{T=S\},
\qquad
\|r_S^*\|_g^2=e_S^\top G^+e_S.
\]

In the homoskedastic Gaussian cell-mean experiment, the exact-model efficiency
bound is

\[
\frac{\sigma^2}{n}e_S^\top G^+e_S.
\]

These are the standard $c$-estimability and generalized Gauss--Markov formulas
for a singular linear model.

## 2. Exact off-support leakage and partial identification

Partition the complete Walsh basis into the modeled columns $X_L$ and omitted
higher-order columns $X_H$.  Any signed cell weights $\zeta$ that recover the
target under the exact model satisfy

\[
X_L^\top\zeta=e_S.
\]

Their higher-order leakage is

\[
\ell(\zeta)=X_H^\top\zeta.
\]

If the omitted coefficients obey

\[
\|W^{1/2}\beta_H\|_2\le B,
\]

the exact worst-case bias is

\[
B\|W^{-1/2}\ell(\zeta)\|_2.
\]

Define the irreducible alias radius

\[
\boxed{
\rho_S(\Omega)
=
\min_{\zeta:X_L^\top\zeta=e_S}
\|W^{-1/2}X_H^\top\zeta\|_2.
}
\]

Its dual form is

\[
\rho_S(\Omega)
=
\sup_{\substack{X_Ld+X_Hz=0\\
                 \|W^{1/2}z\|_2\le1}}
|e_S^\top d|.
\]

The two surfaces $\pm B(d,z)$ in the dual construction have identical values
on $\Omega$.  Consequently, every estimator has a squared-error lower bound of
at least constant order $B^2\rho_S(\Omega)^2$, in addition to sampling noise.

### A sharp no-free-extrapolation corollary

If $\Omega$ omits at least one cube profile and $X_H$ contains every omitted
Walsh order, then

\[
\boxed{\rho_S(\Omega)>0.}
\]

If the radius were zero, the complete Walsh transform of $\zeta$ would equal
$e_S$.  Fourier inversion would force the unique weights

\[
\zeta(a)=2^{-J}\phi_S(a)
\]

on every cube profile, which is impossible for weights supported on a proper
subset.

Thus exact low order can restore point identification, but approximate low
order cannot restore ordinary regular inference for free.  Root-$n$ centered
inference needs at least

\[
B_n\rho_{S,J}=o(n^{-1/2}),
\]

or a confidence interval must include the worst-case bias radius.

For uniform assignment on $m=|\Omega|$ profiles and an unweighted omitted tail,
Parseval gives the computable identity

\[
\boxed{
\rho_S(\Omega)^2
=
\frac{2^J}{m}e_S^\top G^+e_S-1.
}
\]

This connects exact-model variance inflation and approximate-model aliasing:
a support that requires large exact-model weights necessarily permits greater
off-support misspecification.

## 3. Canonical support geometries

### Parity half-fraction

Let

\[
\Omega=\{a:\phi_{[J]}(a)=1\}.
\]

On this support, $\phi_T=\phi_{T^c}$.  For the degree-$q$ model,

\[
\operatorname{rank}(X_L)
=
p_q-rac12
\sum_{r=J-q}^{q}{J\choose r},
\qquad
p_q=\sum_{r=0}^q{J\choose r},
\]

where the displayed sum is zero when $J-q>q$.  An individual coefficient
$\beta_T$ is estimable exactly when

\[
|T^c|>q.
\]

Hence, if $2q<J$, deleting half of the cube leaves the entire degree-$q$ model
identified, with variance factor one under uniform assignment on the fraction.
But the exact estimator for $\beta_T$ has leakage one onto
$\beta_{T^c}$.  The apparent miracle disappears immediately if that complementary
higher-order interaction may be nonzero.

### One Hamming slice

Let

\[
\Omega_k=\{a:\#\{j:a_j=1\}=k\}.
\]

Then

\[
\boxed{
\operatorname{rank}(X_L)
=
{J\choose \min(q,k,J-k)}.
}
\]

The vanishing relation

\[
\sum_{j=1}^J A_j-(2k-J)=0
\]

and its products generate cross-order aliases.  For $q\ge1$ and a noncentral
slice, no individual uniform-cube Walsh coefficient in the model is
identified.  On the balanced slice, the intercept is individually identified
when $q=1$; for $q\ge2$, no individual coefficient is.  Harmonic contrasts can
still be identified.

For example, in the additive model the identifiable main-effect functionals
are $u^\top\beta$ with $u^\top\mathbf 1=0$.  If
$t=2k-J$ and $\bar\beta=J^{-1}\sum_j\beta_j$, the identifiable symmetric
quantity is

\[
\beta_0+t\bar\beta,
\]

not $\beta_0$ and the average main effect separately.

### Union of Hamming slices

Let $H\subseteq\{0,\ldots,J\}$ and

\[
\Omega_H=\bigcup_{k\in H}\Omega_k.
\]

For $q\le J/2$, define

\[
h_d={J\choose d}-{J\choose d-1},
\qquad
L_d=\#\{k\in H:d\le k\le J-d\}.
\]

The exact restriction rank is

\[
\boxed{
\operatorname{rank}(X_L)
=
\sum_{d=0}^q h_d\min\{q-d+1,L_d\}.
}
\]

Therefore the complete degree-$q$ surface is identifiable exactly when

\[
L_d\ge q-d+1,
\qquad d=0,\ldots,q.
\]

In particular, $q+1$ suitably central slices identify the whole degree-$q$
model.  The calculation verified this in all 39 enumerated central-layer
cases.  This is useful for design, but it is an affine Hilbert-function theorem
already present in the Boolean-layer literature.

### One forbidden pair

If profiles with $A_1=A_2=1$ are forbidden, then

\[
1+A_1+A_2+A_1A_2=0
\]

on the support.  Multiplying this relation by characters in the remaining
attributes gives

\[
\boxed{
\operatorname{rank}(X_L)
=
p_q-sum_{r=0}^{q-2}{J-2\choose r}.
}
\]

An individual coefficient $\beta_T$ is estimable exactly when

\[
|T\setminus\{1,2\}|>q-2.
\]

Thus the additive model remains fully identified, while the degree-two model
aliases the intercept, the two implicated main effects, and their interaction.
Effects sufficiently far from the forbidden coordinates remain estimable.

## 4. Exact computational verification

The executable gate enumerated $J=3,\ldots,10$ and degrees through four.
It produced:

- 855 support/rank rows;
- 690 order-specific estimability rows;
- 742 formula checks;
- 234 canonical closed-form rank checks with zero discrepancy; and
- maximum numerical validation error $2.84\times10^{-13}$.

The main controls behaved as expected:

- all 39 central $q+1$-slice supports had full degree-$q$ rank;
- 616 of 621 random matched-size supports had maximal possible rank, showing
  that the canonical deficiencies are structured rather than consequences of
  support size alone; and
- every proper support with an estimable checked contrast had a positive
  unweighted alias radius, ranging from 0.408 to 6.44 in the enumerated cases.

Examples make the geometry concrete:

- with $J=8$ and $q=3$, the parity half-fraction used 128 of 256 profiles and
  retained full rank 93;
- with $J=8$ and $q=2$, excluding the $A_1=A_2=1$ profiles reduced rank from
  37 to 36; and
- with $J=10$ and $q=4$, the five central Hamming slices used 912 of 1,024
  profiles and had full rank 386.

These are exact fixed-allocation calculations.  Under iid assignment, the
rank statements remain identification statements, but variance achievability
also requires adequate realized support and information-matrix concentration.

## 5. Novelty audit

| Candidate contribution | Prior result | Decision |
|---|---|---|
| Arbitrary-support null-space characterization | [Pistone and Wynn](https://doi.org/10.1093/biomet/83.3.653) represent a fraction by its vanishing ideal and obtain identifiable model terms from the quotient. | Classical design-ideal formulation. |
| Coding-independent aliasing | [Pistone and Rogantin](https://doi.org/10.1016/j.jspi.2007.05.009) treat generic fractions embedded in full factorials; [Aoki](https://arxiv.org/abs/1810.08417) gives coding-independent contrast representations. | A causal target label does not create a new alias theorem. |
| Single-slice basis and rank | [Filmus](https://arxiv.org/abs/1406.0142) gives an explicit orthogonal basis on a slice; [Filmus and Mossel](https://arxiv.org/abs/1507.02713) establish the unique bounded-degree harmonic representation. | The harmonic decomposition is established. |
| Union-of-slices rank | [Bernasconi and Egidi](https://doi.org/10.1006/inco.1999.2798) determine the Boolean symmetric-set Hilbert function; [Venkitesh](https://arxiv.org/abs/2111.06993) determines it for arbitrary unions of layers in uniform grids. | The strongest closed-form rank theorem is occupied. |
| Minimum-variance target contrast | Singular $c$-optimal design and generalized Gauss--Markov theory use the same range condition and generalized inverse; see [Fan and Chaloner](https://doi.org/10.1016/S0378-3758(01)00289-0). | Classical linear-design result. |
| Bias-aware inference under approximate restrictions | [Optimal affine inference](https://doi.org/10.3982/ECTA14434) treats linear functionals over convex regression classes using the same bias modulus. | The alias radius is a useful specialization, not a new inference theory. |

The Boolean-layer result is decisive.  The rank of the degree-$q$ restricted
design is precisely the affine Hilbert function already determined for
arbitrary symmetric subsets of the cube.  The harmonic formula above is an
equivalent representation, not a new result.

## 6. Gate decision

| Rule | Result | Reason |
|---|---|---|
| 1. Beyond row-space restatement | **Mathematical pass** | Explicit parity, slice-union, and forbidden-pair formulas replace brute-force rank checks. |
| 2. Beyond classical aliasing | **Fail** | The formulas are defining relations, design ideals, or affine Hilbert functions. |
| 3. Causal relevance | **Conditional pass** | Exact low degree can identify full-target contrasts without ordinary overlap, but the conclusion is entirely restriction-dependent. |
| 4. Approximate-model honesty | **Negative result only** | Every proper support has positive full-tail alias radius; ordinary root-$n$ inference requires shrinking misspecification or bias-aware intervals. |
| 5. Novelty | **Fail** | Arbitrary fractions and unions of Hamming slices are directly covered by established design and algebraic-combinatorial results. |

Rules 2 and 5 were prespecified hard stops.  No Monte Carlo performance study
was launched because estimation accuracy cannot reverse the theorem-level
novelty failure.

## 7. What survives

The following material is worth retaining in software or a broader practical
paper:

1. a target-contrast support diagnostic based on the row-space condition;
2. a constructive alias witness when a requested AMIE is not identified;
3. the minimum-variance exact-model representer when it is identified;
4. the approximate-tail alias radius and its bias-aware interval contribution;
5. interpretable templates for parity, fixed-cardinality, and forbidden-pair
   designs.

The next plausible research gate is not “more general support algebra.”  It is
**minimal support augmentation for prespecified causal contrasts**: given
scientifically unavoidable forbidden combinations and an existing design,
which small set of additional feasible profiles most reduces variance and
worst-case alias radius?  A stand-alone route would require a new approximation
guarantee or factorial-specific structural theorem beyond standard
$c$-optimal design and sensor-selection results.
