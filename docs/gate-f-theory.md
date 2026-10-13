# Gate F decision: spectral design of repeated factorial schedules

**Decision: FAIL as a stand-alone novelty gate.**

**Mathematical/actionability component: PASS.**  Binary factorial structure
reduces an exponentially large schedule calculation to one small temporal
problem per interaction order, and the exact carryover operator has a simple
rank characterization.

**Stand-alone novelty component: FAIL.**  The diagonalization is classical
Hamming-scheme mathematics, while factorial changeover structure,
effect-specific carryover adjustment, canonical efficiencies, and schedule
tradeoffs are established in the crossover-design literature.  The result is
worth retaining as a theorem and software feature in a broader repeated-
factorial-inference paper, but it is not a defensible new headline by itself.

## Plain-language result

Suppose each conjoint profile consists of binary attributes.  A designer can
control how different the next profile is from the current one—for example,
by changing exactly one attribute, half the attributes, or drawing a fresh
profile independently.

That choice affects different interaction orders differently.  Main effects,
two-way interactions, and three-way interactions each acquire one temporal
correlation number.  Once those numbers are known, the efficient-inference and
carryover calculations require only `T` by `T` matrices; no enumeration of all
profile sequences is needed.

The same calculation also gives a sharp limitation.  For each factorial
contrast, covariance adaptation is vulnerable to exactly one aggregate
temporal direction of additive carryover.  Every other factorial feature and
most temporal carryover patterns are invisible to its bias.  The order has
carryover vulnerability if and only if it has covariance-efficiency headroom.

This is useful.  It is not new enough on its own.  More importantly, schedule
comparisons are estimand-dependent: when carryover exists, changing the
assignment schedule can change the task-marginal causal contrast.  A schedule
that looks simultaneously more efficient and less vulnerable for that
schedule-specific target need not improve inference for a fixed pure direct
effect.

## 1. Hamming-radial factorial schedules

Let a profile be

\[
A_t\in\{-1,1\}^F
\]

and let the target profile distribution be uniform.  For every subset
`S` of the attributes, define the Walsh factorial character

\[
\chi_S(a)=\prod_{j\in S}a_j,
\qquad k=|S|.
\]

The characters are an orthonormal factorial basis.  Order `k` contains
\(\binom Fk\) characters.

Consider a stationary Markov schedule.  At a transition, first choose a
Hamming distance `h` with probability \(\pi_h\), and then choose uniformly
which `h` attributes to flip.  Define

\[
K_k(h)
=
\sum_j(-1)^j
\binom kj\binom{F-k}{h-j}
\]

and the normalized order-specific eigenvalue

\[
\rho_k
=
\sum_{h=0}^F
\pi_h\frac{K_k(h)}{\binom Fh}.
\tag{F.1}
\]

### Proposition F1: factorial spectral decomposition

For any two factorial characters,

\[
E\{\chi_S(A_t)\chi_U(A_s)\}
=
1\{S=U\}\rho_{|S|}^{|t-s|}.
\tag{F.2}
\]

Thus every feature of order `k` has the same task-correlation matrix

\[
R_k(t,s)=\rho_k^{|t-s|},
\tag{F.3}
\]

and different Walsh features remain orthogonal across tasks.

**Proof.**  Conditional on changing `h` coordinates, let `j` be the number
of changed coordinates that lie in `S`.  The character changes sign by
\((-1)^j\).  Counting the possible intersections gives
\(K_k(h)/\binom Fh\), so each character is an eigenfunction of the
transition operator.  Stationarity, iteration of the Markov operator, and
Walsh orthogonality give (F.2). \(\square\)

The spectral identity is classical.  Its benefit here is computational and
interpretive: the eigenvalues are indexed by factorial interaction order.

## 2. Efficient scoring reduces by interaction order

Fix one feature `S` of order `k`.  Let

\[
x_S=
(\chi_S(A_1),\ldots,\chi_S(A_T))^\mathsf T,
\]

let \(\omega\in\mathbb R^T\) contain the task weights, and suppose the
within-respondent residual covariance is the positive-definite matrix
\(\Sigma\), constant across schedules.

A linear residual weight can be written \(H_S=Cx_S\).  Calibration for all
task-specific direct coefficients requires

\[
\operatorname{diag}(CR_k)=\omega.
\tag{F.4}
\]

Define

\[
J_k=\Sigma^{-1}\circ R_k,
\qquad
v_k=J_k^{-1}\omega,
\qquad
C_k=\Sigma^{-1}\operatorname{diag}(v_k).
\tag{F.5}
\]

Then the covariance-efficient and ordinary taskwise weights are

\[
H^*_S=C_kx_S,
\qquad
H_{0,S}=\operatorname{diag}(\omega)x_S.
\tag{F.6}
\]

Their residual variance terms are

\[
V_{*,k}=\omega^\mathsf TJ_k^{-1}\omega,
\qquad
V_{0,k}=\omega^\mathsf T(\Sigma\circ R_k)\omega,
\tag{F.7}
\]

with headroom

\[
\Delta_k=V_{0,k}-V_{*,k}\ge0.
\tag{F.8}
\]

The full problem therefore requires at most `F` inversions of `T` by `T`
matrices.  Evaluating one proposed radial schedule costs
\(O(FT^3)\), rather than enumerating \(2^{FT}\) binary schedules.

## 3. Exact carryover visibility and rank

Put

\[
P_k=
\{C_k-\operatorname{diag}(\omega)\}R_k.
\tag{F.9}
\]

Calibration implies \(\operatorname{diag}(P_k)=0\).  Consider additive
cross-task factorial carryover

\[
c_t(\mathbf A)
=
\sum_{r\ne t}\sum_U
\Gamma_{tr,U}\chi_U(A_r).
\tag{F.10}
\]

### Proposition F2: one visible temporal direction per factorial feature

The exact bias of the efficient score relative to ordinary taskwise scoring
for target feature `S` is

\[
\beta_S
=
\sum_{t\ne r}(P_k)_{tr}\Gamma_{tr,S}.
\tag{F.11}
\]

Every carryover feature \(U\ne S\) is target-blind.  Moreover,

\[
\Delta_k
=
\operatorname{tr}\!left[
\Sigma
\{C_k-\operatorname{diag}(\omega)\}
R_k
\{C_k-\operatorname{diag}(\omega)\}^\mathsf T
\right],
\tag{F.12}
\]

so, when \(R_k\) is nonsingular,

\[
P_k\ne0
\quad\Longleftrightarrow\quad
\Delta_k>0.
\tag{F.13}
\]

The additive cross-task carryover space has \(T(T-1)\) coordinates per
Walsh feature.  For an active feature, the map in (F.11) has rank one and
kernel dimension

\[
T(T-1)-1.
\tag{F.14}
\]

Across all nonintercept factorial features, the rank is

\[
\sum_{k=1}^F
\binom Fk1\{\Delta_k>0\}.
\tag{F.15}
\]

Under Gate E1's unrestricted residual-standardized departure ball, the
carryover-leverage operator on order `k` is

\[
K_k=T\Delta_k I_{\binom Fk}.
\tag{F.16}
\]

Hence efficiency headroom and worst-direction carryover sensitivity are the
same spectral object.  At singular endpoints \(|\rho_k|=1\), cross-task
features are deterministic signed copies of own-task features; the relevant
quotient collapses and both \(P_k\) and \(\Delta_k\) are zero.

## 4. Product and sampling without replacement

The previous Markov schedule is not the only useful benchmark.  Let the
uniform target contain `M` profiles and draw `T<=M` profiles as an ordered
simple random sample without replacement.  For every centered target-
orthonormal profile contrast,

\[
R(t,s)=
\begin{cases}
1,&t=s,\\
-1/(M-1),&t\ne s.
\end{cases}
\tag{F.17}
\]

Product assignment instead gives \(R=I_T\).  Neither design distinguishes
factorial orders.

For equal task weights and

\[
\Sigma=eI_T+u\mathbf1\mathbf1^\mathsf T,
\qquad e>0,\quad u>0,
\]

write `r` for the common off-diagonal assignment correlation.  Direct
calculation gives

\[
V_0(r)
=
\frac{e+u\{1+(T-1)r\}}{T},
\tag{F.18}
\]

\[
V_*(r)
=
\frac{e(e+Tu)}
{T\{e+(T-1)u(1-r)\}},
\tag{F.19}
\]

and

\[
\Delta(r)
=
\frac{(T-1)u^2\{1+(T-1)r\}(1-r)}
{T\{e+(T-1)u(1-r)\}}.
\tag{F.20}
\]

All three quantities increase with `r` on the feasible interval
\([-1/(T-1),0]\).  Consequently, sampling without replacement has smaller
variance and smaller Gate E1 full-ball leverage than product assignment for
the schedule-specific task-marginal target.  At `T=M`, both variances equal
\(e/T\), and there is no covariance-adaptation headroom.

This result is mathematically exact but often practically tiny in a large
conjoint universe.  With `M=1440`, `T=8`, `e=.02`, and `u=.08`, the
without-replacement/product variance ratio is about `0.9961` for ordinary
scoring and `0.9993` for the restricted efficiency bound.

## 5. Order-specific schedule tradeoffs

For two tasks with residual correlation `c` and unit marginal residual
variance, equations (F.7)--(F.8) simplify to

\[
V_{0,k}
=
\frac{1+c\rho_k}{2},
\qquad
V_{*,k}
=
\frac{1-c^2}{2(1-c\rho_k)},
\tag{F.21}
\]

\[
\Delta_k
=
\frac{c^2(1-\rho_k^2)}{2(1-c\rho_k)}.
\tag{F.22}
\]

When `c>0` and \(\rho_k\in[-1,0]\), making \(\rho_k\) more negative lowers
all three quantities for the schedule-specific target.

The schedule spectrum makes the tradeoffs concrete.

- A fresh product draw has \(\rho_k=0\) for every nonconstant order.
- A uniform transition to any different profile has
  \(\rho_k=-1/(2^F-1)\) for every nonconstant order.  This is the unique
  radial transition that minimizes the largest order eigenvalue, but its gain
  vanishes exponentially in `F`.
- Changing exactly one attribute gives \(\rho_k=1-2k/F\).
- Independently flipping each attribute with probability `p` gives
  \(\rho_k=(1-2p)^k\).
- Changing exactly half the attributes when `F` is even leaves odd orders
  temporally uncorrelated, anticorrelates order two, and positively correlates
  order four.

For `F=4`, the fixed-distance spectrum is

| order | `h=0` | `h=1` | `h=2` | `h=3` | `h=4` |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 1/2 | 0 | -1/2 | -1 |
| 2 | 1 | 0 | -1/3 | 0 | 1 |
| 3 | 1 | -1/2 | 0 | 1/2 | -1 |
| 4 | 1 | -1 | 1 | -1 | 1 |

No one fixed distance is best for every order.  A schedule designer can
precompute the normalized Krawtchouk matrix `Lambda`, write

\[
(\rho_1,\ldots,\rho_F)^\mathsf T=\Lambda\pi,
\]

and optimize the `F+1` probabilities `pi` over the simplex.  With two tasks,
weighted sums or maxima of \(V_{*,k}\) are convex functions of `pi` for
positive residual correlation; linear constraints such as
\(|\rho_k|\le s_k\) can protect prespecified orders.  For more tasks, the
same problem remains low-dimensional, although the objective need not be
globally convex.

## 6. The estimand warning

Equations (F.18)--(F.22) compare the Gate E1 target defined by ordinary
taskwise scoring under each assignment law.  Under carryover, that target
averages the other tasks according to the schedule's conditional law.  It is
therefore generally a different causal contrast after `q` changes.

The distinction is visible in the smallest lag model.  With two tasks, let
\(x_t=\chi_S(A_t)\), \(E(x_1x_2)=\rho\), residual correlation `c`, and allow
the second outcome mean to contain an arbitrary coefficient on the lag feature
`x_1`.  If the target is the average pure current-profile coefficient and the
lag coefficient has target load zero, its exact efficient bound is

\[
\boxed{
V_{Z,\mathrm{direct}}(\rho)
=
\frac{2-\rho^2-c^2}{4(1-\rho^2)}.
}
\tag{F.23}
\]

For \(|c|<1\), this is uniquely minimized at \(\rho=0\), product assignment,
and diverges as \(|\rho|\to1\).  By contrast, preserving the ordinary
schedule-specific target assigns the lag coefficient load \(\rho/2\), which
changes with the schedule and yields a different bound.

Therefore the paper must not claim that dependent scheduling simultaneously
improves efficiency and carryover robustness without stating which causal
target is fixed.  The correct design problem is: fix the direct or structured
target first, then optimize a score that calibrates that same target under
every candidate schedule.

## 7. Novelty audit

The broad design claim is already occupied.

- [Fletcher and John (1985)](https://doi.org/10.1111/j.2517-6161.1985.tb01338.x)
  define factorial structure in changeover designs and construct useful
  generalized cyclic designs.
- [Fletcher (1987)](https://doi.org/10.1093/biomet/74.3.649) develops
  canonical efficiency factors and an algorithm for selecting factorial
  changeover designs.
- [Fletcher, Lewis, and Matthews (1990)](https://doi.org/10.1002/sim.4780091002)
  explicitly construct crossover designs with known efficiencies for direct
  factorial main effects and interactions, tailored to the investigator's
  priorities.
- [Lewis and Russell (1998)](https://doi.org/10.1111/1467-9876.00116) construct
  multifactor designs balancing direct and carryover effects.
- [Laska and Meisner (1985)](https://doi.org/10.1080/01621459.1985.10478172)
  optimize allocations over treatment sequences with arbitrary within-subject
  covariance and carryover.
- [Karlin, Lindqvist, and Yao (1993)](https://doi.org/10.1002/rsa.3240040102)
  give spectral representations for Markov chains on hypercubes.
- Modern causal work also optimizes temporal designs under carryover or
  habituation; see [Basse, Ding, and Toulis
  (2023)](https://doi.org/10.1093/biomet/asac024) and [Bojinov, Simchi-Levi,
  and Zhao (2023)](https://doi.org/10.1287/mnsc.2022.4583).

The precise conjunction of target-weighted causal EIFs, the Gate E1
sensitivity operator, and Krawtchouk schedule blocks may be a useful modern
presentation.  But Proposition F1 is a standard spectral fact, Proposition F2
specializes the existing Gate E1 projection identity, and order-specific
factorial crossover efficiency is old.  The novelty threshold is therefore
not met.

## 8. What survives and what comes next

Retain the following in a broader paper:

1. the `T` by `T` order-block computation;
2. the exact rank and target-blind carryover characterization;
3. the fixed-target warning in equation (F.23); and
4. a small schedule-diagnostic function, not a claim to a new design theory.

Do not launch a large schedule simulation or add a second application on the
strength of Gate F.  The remaining obstacle to a full methods paper is still
inferential: the feasible covariance-adaptive estimator did not realize its
first-order bound at the sample sizes in Gate E2.  The next defensible gate is
a finite-sample nuisance-remainder exercise that derives the extra variance,
constructs a count-aware or corrected score, and tests it on fresh simulations.
Without that result, the collected EIF, sensitivity, structured-carryover, and
spectral-design results support a focused theory/diagnostic paper rather than
a new general-purpose ML method.
