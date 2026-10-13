# Minimal support-augmentation gate: decision

**Decision date:** 2026-08-31  
**Status:** **FAIL as a stand-alone methods theorem; retain augmentation as a
design diagnostic and software module.**  
**Protocol:**
[`support-augmentation-gate-protocol.md`](support-augmentation-gate-protocol.md)  
**Executable calculation:**
[`analysis/support_augmentation/`](../analysis/support_augmentation/README.md)

## Bottom line

Adding a small number of feasible factorial profiles can repair identification,
lower variance, and reduce sensitivity to omitted interactions.  Those are
useful design tasks, but the proposed general method is not new:

- exact repair for a fixed contrast is sparse exact representation in the
  quotient left unidentified by the original design;
- exact repair of the entire model is linear-matroid basis extension;
- optimal allocation is augmented $c$- or $L$-optimal design; and
- variance plus worst-case ellipsoidal alias bias is the same $c$-optimal
  problem with a structured correlated-error covariance matrix.

The factorial geometry gives clean special cases for parity fractions, unions
of Hamming slices, and forbidden profiles.  However, these reduce to familiar
aliasing, interpolation, signed cubature, orthogonal arrays, and Hadamard
designs.  The frozen screen also produces a factorial counterexample to a
blanket supermodularity claim, and the natural greedy robust augmentation is
48.9% worse than exhaustive selection in one small prespecified case.

This fails the two hard gates: there is no surviving general factorial theorem
beyond known reductions, and the literature already contains the relevant
follow-up, robust-design, and exact $c$-optimal-design machinery.  In accordance
with the protocol, no large Monte Carlo study was run.

## 1. Exact augmentation is a quotient-space problem

Let $X_0$ be the degree-$q$ factorial design matrix on the existing support,
let $X_F$ contain all scientifically feasible candidate rows, and let the
columns of $C$ be the fixed target contrasts.  Choose a matrix $Z$ whose
columns span $\ker(X_0)$, and define

\[
D=X_FZ,
\qquad
T=C^\top Z.
\]

For a selected candidate set $S$,

\[
\boxed{
C^\top\beta\text{ is estimable after adding }S
\iff
\operatorname{row}(T)\subseteq\operatorname{row}(D_S).
}
\]

This removes every direction already learned by the original experiment.  The
remaining design problem is to select projected candidate rows whose span
contains the requested projected contrasts.  Equivalently,

\[
\boxed{
k^\star=\min_{L:T=LD}\|L\|_{2,0},
}
\]

where $\|L\|_{2,0}$ counts the candidate rows used by at least one contrast.
For one contrast, this is the sparsest exact representation

\[
k^\star=\min_{\lambda:D^\top\lambda=t}\|\lambda\|_0.
\]

Two cases must therefore remain separate:

- **Complete low-order surface.**  If feasible candidates can restore full
  rank, exactly
  $p_q-\operatorname{rank}(X_0)$ profiles are necessary and sufficient.  This
  is the basis-extension property of a linear matroid.
- **Selected causal contrasts.**  Fewer profiles may suffice, but finding the
  smallest set is a target-specific sparse-representation problem.  Generic
  sparsest representation is computationally hard; factorial coordinates do
  not remove that difficulty for arbitrary feasible-profile sets.

The quotient characterization is useful implementation guidance, but it is a
re-expression of classical estimability rather than a new identification
theorem.

## 2. Precision augmentation is classical $c$-optimal design

With existing information $M_0$ and $n_a$ additional observations at feasible
profile $a$,

\[
M(n)=M_0+\sum_{a\in F}n_ax_L(a)x_L(a)^\top.
\]

For one target vector $c$, continuous allocation solves

\[
\min_{n_a\ge0,\ \sum_an_a=N}c^\top M(n)^+c.
\]

This is the ordinary augmented $c$-optimal-design problem.  Its support-point
condition is

\[
\{x_L(a)^\top M(n)^+c\}^2
\begin{cases}
=\eta,&n_a>0,\\
\le\eta,&n_a=0.
\end{cases}
\]

When $M$ is nonsingular, the exact value of adding one observation at $x$ is
the Sherman--Morrison update

\[
c^\top M^{-1}c-c^\top(M+xx^\top)^{-1}c
=
\frac{(c^\top M^{-1}x)^2}{1+x^\top M^{-1}x}.
\]

The executable gate reproduces this identity to numerical error below
$1.2\times10^{-16}$.

## 3. Robust alias control also collapses to $c$-optimal design

Let $X_L$ contain the modeled low-order factorial columns and $X_H$ the omitted
columns.  An exact low-order linear estimator with signed cell weights $\zeta$
must satisfy

\[
X_L^\top\zeta=c.
\]

If cell means have variances $\sigma^2/n_a$ and omitted coefficients obey
$\|W^{1/2}\beta_H\|_2\le B$, its exact worst-case mean-squared error is

\[
R(\zeta,n)
=
\sigma^2\sum_a\frac{\zeta_a^2}{n_a}
+B^2\|W^{-1/2}X_H^\top\zeta\|_2^2.
\]

For fixed support and allocation, set

\[
Q=\sigma^2D_n^{-1}+B^2X_HW^{-1}X_H^\top.
\]

Then

\[
\zeta^*=Q^{-1}X_L(X_L^\top Q^{-1}X_L)^+c,
\qquad
R^*=c^\top(X_L^\top Q^{-1}X_L)^+c.
\]

Thus the robust objective is exactly $c$-optimal design with correlated
observations having covariance $Q$.  The causal contrast supplies $c$, and the
factorial tail supplies a structured covariance kernel, but neither changes
the optimization class.

The collapse is even stronger for the complete, unweighted Walsh tail.  By
Parseval, whenever $X_L^\top\zeta=c$,

\[
\|X_H^\top\zeta\|_2^2
=2^J\|\zeta\|_2^2-\|c\|_2^2.
\]

Under equal allocation of a fixed total $n$ over a support $S$,

\[
R^*(S)
=
\left(\frac{\sigma^2|S|}{n}+2^JB^2\right)
c^\top(X_{L,S}^\top X_{L,S})^+c
-B^2\|c\|_2^2.
\]

At a fixed augmentation budget, variance-optimal and full-tail-alias-optimal
supports therefore have exactly the same ordering.  Order-weighted tails break
that scalar identity, but return the standard correlated-observation robust
design problem above.

## 4. What factorial structure does provide

### Parity half-fraction

For

\[
\Omega_0=\{a:\phi_{[J]}(a)=1\},
\]

the degree-$q$ model has deficiency

\[
\Delta_+
=
\frac12\sum_{r=J-q}^{q}{J\choose r},
\]

where the sum is zero when $2q<J$.  The missing quotient coordinates are the
familiar complementary aliases $\phi_T=\phi_{T^c}$ on the fraction.  If the
opposite half is feasible, restoring the whole surface requires exactly
$\Delta_+$ suitably chosen profiles.  Restoring a selected contrast is the
sparse-representation problem in the corresponding quotient Walsh matrix.

When $2q<J$, the fraction is already orthogonal for all modeled columns.  Under
uniform allocation it has exact-model variance factor one, so adding one
profile can actually worsen exact-model variance by diluting the orthogonal
allocation even while lowering the alias radius.  Balance, not mere distance
from the old profiles, determines whether an added set preserves that
variance.

### Unions of Hamming slices

Let $H\subseteq\{0,\ldots,J\}$ be the observed Hamming layers.  For
$q\le J/2$, define

\[
h_d={J\choose d}-{J\choose d-1},
\qquad
L_d=\#\{k\in H:d\le k\le J-d\}.
\]

The full-model deficiency is

\[
\boxed{
\Delta_H
=
\sum_{d=0}^q h_d\{q-d+1-L_d\}_+.
}
\]

If arbitrary candidate profiles span the missing quotient, exactly $\Delta_H$
are needed to restore the complete model.  If augmentation must add whole
layers, put $r_d=\{q-d+1-L_d\}_+$.  Added layers must cover each nested depth
class at least $r_d$ times.  With all missing layers feasible and equal cost per
layer, the minimum number is

\[
\max_{0\le d\le q}r_d,
\]

and selecting the deepest available layers first is optimal.  This is a nested
cover corollary of the established slice-rank formula.  Unequal costs per layer
give a weighted nested-cover problem rather than a new universal rule.

For an additive model on one slice, any off-slice profile restores full rank,
but not all such profiles provide good precision.  In the checked
$J=6$, layer-$1$ example for a specified main effect, the best single addition
lies in layer $5$ with the target factor high; the farthest layer $6$ is not
optimal.  A geometric "add the farthest profile" rule is therefore false even
in this simple case.

### One omitted or forbidden pair

Suppose profiles with $A_1=A_2=1$ are absent.  The null directions contributed
by degree $q$ are generated by

\[
(1+A_1+A_2+A_1A_2)\phi_R,
\qquad |R|\le q-2.
\]

On a candidate profile with $A_1=A_2=1$, the quotient design is precisely the
degree-$(q-2)$ Walsh design on the remaining $J-2$ factors.  Recovering the
whole degree-$q$ model therefore requires

\[
\sum_{r=0}^{q-2}{J-2\choose r}
\]

suitably chosen omitted profiles.  Recovering a particular implicated contrast
is a signed-cubature problem:

\[
\sum_iw_i\phi_U(z_i)=1\{U=\varnothing\},
\qquad |U|\le q-2.
\]

For $q=3$, two profiles suffice exactly when their remaining coordinates are
antipodal.  For $q=4$, at least $J-1$ profiles are needed, with equality tied
to a Hadamard design of order $J-1$.  These facts are useful diagnostics but
belong to classical cubature and orthogonal-array geometry.

Most importantly, if $A_1=A_2=1$ is scientifically impossible rather than
merely omitted from the original randomization, none of these profiles is a
feasible candidate.  Design augmentation then cannot repair the implicated
full-target contrast.

## 5. Deterministic screen

The screen uses exact finite-cube enumeration rather than Monte Carlo.

| Check | Result |
|---|---:|
| Hamming-slice rank identity, maximum residual | $0$ |
| Parity target estimability, maximum residual | $5.1\times10^{-15}$ |
| Sherman--Morrison update, maximum residual | $1.1\times10^{-16}$ |
| Parity antipodal-pair closed form, maximum residual | $5.6\times10^{-16}$ |
| One-point Hamming-slice formula and optimum, maximum residual | $3.6\times10^{-15}$ |
| Degree-3 forbidden-pair antipodal minimum, maximum residual | $3.2\times10^{-15}$ |
| Degree-4 forbidden-pair Hadamard minimum, maximum residual | $4.5\times10^{-16}$ |
| Whole-layer cases, $J=3,\ldots,8$, $q\le3$ | $120/120$ agree with exhaustive search |
| Structurally infeasible whole-layer cases | $9$ correctly flagged |
| Frozen robust greedy comparisons | $16$ |
| Worst greedy/exhaustive robust-risk ratio | $1.4886$ |

The robust counterexample uses three factors, an additive target model, the
four base profiles

\[
\{---,+--,-+-,--+\},
\]

and the four candidate profiles

\[
\{++-,+-+,-++,+++\}.
\]

For the first main effect, alias radius $B=4$, and a two-profile budget, exact
enumeration selects $\{++-,+-+\}$ and attains risk $5.5$.  Forward greedy
selects $\{+++, +-+\}$ and attains risk $8.1875$, a ratio of
$1.488636$.  This does not rule out every specialized approximation theorem,
but it does rule out treating the obvious greedy procedure as automatically
near-optimal.

The same design gives a direct failure of blanket supermodularity for the
$c$-variance cost.  For nested candidate sets $A=\{++-\}$ and
$B=\{++-,+-+\}$, adding $-++$ changes the cost by $0$ at $A$ but by
$-1/96$ at $B$.  Its marginal value increases rather than decreases.

## 6. Prior-art audit

| Component | Closest established area | Gate assessment |
|---|---|---|
| Add runs to an existing factorial design | Follow-up and foldover design | Established |
| Minimum profiles for full rank | Linear-matroid basis extension | Established |
| Minimum profiles for selected contrasts | Sparse exact representation | Generic hard problem |
| Minimize contrast variance | Exact or approximate $c$-optimal design | Established |
| Variance plus ellipsoidal omitted-term bias | Minimax design under model misspecification | Established |
| Select a budgeted profile subset | Row/sensor selection and experimental-design subset selection | Established |
| Restricted Hamming regions | Symmetry/orbit and Hamming-layer optimal design | Established |
| Forbidden-profile quotient | Signed cubature, orthogonal arrays, Hadamard designs | Established |

The most directly relevant primary sources are:

- Dykstra's general augmentation algorithm and Meyer, Steinberg, and Box's
  factorial follow-up design already formulate adding runs to an existing
  design: [Dykstra (1971)](https://doi.org/10.1080/00401706.1971.10488830) and
  [Meyer, Steinberg, and Box (1996)](https://doi.org/10.1080/00401706.1996.10484538).
- Partial and optimal foldover plans explicitly target de-aliasing factorial
  effects: [Li and Lin (2003)](https://doi.org/10.1198/004017003188618779).
- Finite-domain exact $c$-optimality and modern exact subset selection already
  cover the core variance objective:
  [Harman and Jurík (2008)](https://doi.org/10.1016/j.csda.2008.06.023),
  [Černý and Hladík (2012)](https://doi.org/10.1007/s10589-010-9377-8), and
  [Watson and Pan (2023)](https://doi.org/10.1007/s11222-023-10280-w).
- The generic minimum-support representation problem is already known to be
  computationally hard:
  [Natarajan (1995)](https://doi.org/10.1137/S0097539792240406).
- Bias--variance design under omitted terms goes back at least to
  [Box and Draper (1959)](https://doi.org/10.1080/01621459.1959.10501525) and
  [Wiens (1992)](https://doi.org/10.1016/0378-3758(92)90142-F); it has also
  been developed specifically for factorial designs contaminated by omitted
  interactions, for example
  [Wilmut and Zhou (2011)](https://doi.org/10.1016/j.jspi.2010.07.002).
- Budgeted observation selection and approximation guarantees form a large
  adjacent literature; representative examples are
  [Krause, Singh, and Guestrin (2008)](https://www.jmlr.org/papers/v9/krause08a.html)
  and
  [Chamon and Ribeiro (2017)](https://proceedings.neurips.cc/paper/2017/hash/0d3180d672e08b4c5312dcdafdf6ef36-Abstract.html).
- Optimal design on restricted Hamming regions is already treated using
  permutation symmetry and orbit reductions by
  [Freise, Holling, and Schwabe (2020)](https://doi.org/10.1016/j.jspi.2019.04.005)
  and
  [Freise and Schwabe (2019)](https://doi.org/10.1007/s00362-018-01063-x).

## 7. Gate decision

| Pass rule | Result | Reason |
|---|---|---|
| Factorial theorem | **Fail** | The clean formulas are quotient-space, aliasing, slice interpolation, or cubature specializations. |
| Optimization contribution | **Fail** | The general exact and robust problems are sparse representation, $c$-optimal design, and row selection. |
| Causal relevance | Pass | Prespecified target-distribution contrasts and structural feasibility make the diagnostic scientifically meaningful. |
| Approximate-model honesty | Pass | The robust objective exposes rather than hides off-support alias sensitivity. |
| Novelty | **Fail** | Follow-up/foldover, robust design, and exact subset-selection literatures already occupy the central claim. |

Rules 1 and 5 were prespecified hard stops.  The stand-alone augmentation paper
therefore stops here.

## 8. What to retain

The work is still worth keeping as a practical component of a broader paper or
package.  A future `suggest_augmentation()` interface could:

1. accept existing support, feasible candidates, target contrasts, costs, and
   an optional omitted-interaction bound;
2. separate design zeros from structural zeros;
3. report exact estimability and quotient rank;
4. find exact small-budget solutions by enumeration or mixed-integer methods;
5. compute variance and worst-case alias sensitivity; and
6. label heuristic solutions honestly, without a generic optimality guarantee.

That would be useful software.  It should not be presented as a new general
augmentation estimator or theorem.  The larger project should now stop opening
adjacent theorem gates and choose between a practical software/diagnostic paper
and a transparent negative comparative paper based on the accumulated results.
