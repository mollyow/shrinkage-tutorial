# Focused novelty audit: efficient and carryover-robust inference for repeated factorial experiments

**Audit date:** 2026-08-28  
**Formal gate decision:** **FAIL as a stand-alone full methods paper.**  
**Practical decision:** retain the results as an applied-methods synthesis,
software contribution, or component of a substantively led paper. Do not run
another estimator-performance simulation in an attempt to reverse this gate.

## Bottom line

The combined project is coherent and useful. It fixes a real problem in repeated
conjoint and factorial experiments: covariance-based precision gains require
restrictions on how a respondent's task outcomes depend on the profiles shown in
other tasks, and those restrictions can create carryover bias. The project also
keeps the target factorial contrast fixed while moving between a no-carryover
model, a prespecified structured-carryover model, and an unrestricted analysis.

That conjunction is not, however, enough to support the planned claim of a new
general method. Each mathematical step is obtained by a direct specialization
of existing theory, and a very recent paper by Jiang and Ding now covers much of
the domain-specific middle ground: fixed linear causal estimands over treatment
sequences, carryover restrictions, rank/identification conditions, optimal
covariance weighting, feasible restricted WLS, and robust design-based
inference for general crossover designs.

A broad "DML for conjoint" claim is independently unavailable. Acharya,
Hainmueller, and Xu (2026) already combine flexible deep learning,
respondent-level repeated-choice structure, and double/debiased inference in a
conjoint application. Their structural preference parameters differ from the
design-based target-profile contrasts studied here, so that paper is not an
exact substitute for the present estimand. It nevertheless occupies the broad
methodological framing.

The narrow residue is an implementation for high-dimensional factorial profiles
with target-distribution-weighted AMIEs, coding-invariant bases, respondent-level
cross-fitting, and explicit sensitivity summaries. That is valuable applied
methodology. It is not a nontrivial new efficiency, robustness, or decision
theory result under the project's own novelty rule.

## Exact candidate that was audited

For a fixed target distribution over profiles and a fixed finite vector of
factorial contrasts, the project combines four objects:

1. an unrestricted taskwise influence score `H0` that remains valid when a task
   mean can depend on the complete randomized schedule;
2. the no-carryover efficient score

   \[
   H^*=\Sigma^{-1}B
   \{E_q(B^\mathsf T\Sigma^{-1}B\mid X)\}^{+}L^\mathsf T;
   \]

3. a structured-carryover efficient score based on `C = [B,Z]`,

   \[
   H_Z=\Sigma^{-1}C
   \{E_q(C^\mathsf T\Sigma^{-1}C\mid X)\}^{+}
   [L,U_0]^\mathsf T,
   \qquad
   U_0=E_q(H_0^\mathsf TZ\mid X);
   \]

4. a finite-radius score `H_rho` that minimizes residual variance plus squared
   worst-case bias over an ellipsoidal structured-carryover neighborhood.

The intended implementation adds cluster-level cross-fitting, joint sandwich
inference for a fixed contrast family, support/rank diagnostics, and factorial
computation that does not depend on arbitrary dummy coding.

## Reducibility check

The decisive check is not whether this exact notation has appeared before. It
is whether the claimed result survives after the problem is written in the
standard notation of the nearest literature.

### The target-preserving load is a pullback of a fixed estimand

Let the schedule-specific conditional mean obey

\[
m(\mathbf A,X)=[B(\mathbf A),Z(\mathbf A)]\xi(X).
\]

If the fixed unrestricted target is represented by `H0`, then on this smaller
mean space

\[
E_q\{H_0^\mathsf Tm\mid X\}
=
[E_q(H_0^\mathsf TB\mid X),E_q(H_0^\mathsf TZ\mid X)]\xi(X)
= [L,U_0]\xi(X).
\]

Thus `U0` is important—it prevents a silent change of estimand—but it is the
coordinate representation of an already-fixed linear functional on the
restricted mean space. In Riesz/conditional-moment language it is a pullback,
not a new efficiency principle.

### The structured score is restricted GLS

Once the mean space is enlarged from `B` to `[B,Z]`, covariance-weighted
projection gives `H_Z`. This is the standard efficient conditional-moment/GLS
score with an augmented regressor matrix. The variance ordering

\[
\Omega_*\preceq\Omega_Z\preceq\Omega_0
\]

is the usual information ordering for nested mean models.

There is also a direct mapping to Jiang and Ding's crossover framework. Their
fixed causal quantity `theta(W)` is our fixed target represented by `H0`; their
linear restriction matrix has null space equal to the allowed schedule-mean
space `col[B,Z]`; their restricted covariance-weighted least-squares BLUE has
the same normal equations as `H_Z`; and their rank condition is the finite-cell
version of the range condition for
`E_q(C' Sigma^{-1} C | X)`. In those coordinates, `U0` is simply the loading of
the fixed `W` functional on the `Z` coefficients.

### The finite-radius path is approximate-moment minimax estimation

For supported schedules, the project solves

\[
\min_{A_BH=L^\mathsf T}
H^\mathsf TVH+
\rho^2
\{A_Z(H-H_0)\}^\mathsf TQ_Z^+
\{A_Z(H-H_0)\}.
\]

This is the ellipsoidal approximate-moment problem studied by Armstrong and
Kolesar and the local minimum-MSE problem studied by Bonhomme and Weidner. The
matrix

\[
P_\rho=V+\rho^2A_Z^\mathsf TQ_Z^+A_Z
\]

and the path from `H*` at radius zero to `H_Z` at infinite radius are its
factorial KKT specialization. The bias ellipsoid, tipping radius, bias-aware
interval, and restricted-versus-unrestricted adaptation problem are likewise
established.

## Prior-art matrix

| Candidate claim | Closest primary result | Exact relationship | Verdict |
|---|---|---|---|
| Target-weighted AMCE/AMIE or factorial contrast | Hainmueller, Hopkins, and Yamamoto (2014); de la Cuesta, Egami, and Imai (2022); Egami and Imai (2019) | Marginal factorial estimands, the essential role of the target profile distribution, causal interactions, and coding/baseline invariance are established. | Established |
| EIF for a fixed linear target under known randomization | Stochastic-policy/AIPW and Riesz-representer theory; Narita, Yasui, and Yata (2019) | A factorial contrast is a signed multivalued-policy value after linear reparameterization. | Direct specialization |
| Covariance-optimal score under a conditional mean restriction | Chamberlain (1987); Newey (1990); Stephens, Tchetgen Tchetgen, and De Gruttola (2014) | Inverse-covariance projection under conditional moments and correlated outcomes gives the displayed score. | Direct specialization |
| Orthogonal/Riesz nuisance correction and clustered cross-fitting | Chernozhukov et al. (2018); Chernozhukov, Newey, and Singh (2022); Chiang et al. (2022); Park and Kang (2022) | Generic DML, learned Riesz correction, cluster-isolated cross-fitting, and cluster-robust covariance are already available. | Established implementation principle |
| ML and DML for repeated conjoint choices | Acharya, Hainmueller, and Xu (2026) | The paper uses a flexible structural preference model and DML rather than the present design-based target-profile contrasts, but it rules out a broad claim to introduce DML to conjoint data. | Strong adjacent overlap |
| Repeated-randomization semiparametric efficiency | Cheng, Bell, and Qian (2024); Xia et al. (2026) | Both derive covariance-aware efficient scores under repeated or staggered randomized treatment structures. | Strong adjacent overlap |
| Fixed target plus no-carryover/structured carryover restrictions | Jiang and Ding (2026, v3) | Their `theta(W)`, restriction matrix, rank condition, restricted WLS/BLUE, feasible covariance weighting, and robust inference cover the same finite-support problem at a more general crossover level. | Direct domain overlap |
| `H_Z` under `[B,Z]` | Conditional-moment efficiency plus Jiang-Ding restricted WLS | Add the allowed carryover columns and pull the fixed target into the augmented coordinates. | Direct extension |
| Nested efficiency ladder | Conditional-moment projection geometry | Adding nuisance mean directions weakly raises the efficiency bound. | Standard corollary |
| `H_rho` and sensitivity operator | Armstrong and Kolesar (2021); Bonhomme and Weidner (2022) | Variance plus ellipsoidal worst-case bias, with a ridge path between restricted and robust estimators. | Exact specialization |
| Automatic adaptation or pretest | Armstrong, Kline, and Sun (2025) | Restricted-efficient versus unrestricted-robust Gaussian adaptation is treated directly. | Established |
| Conjoint carryover diagnostic using ML | Ham, Imai, and Janson (2024) | Conditional randomization tests already allow ML statistics for carryover and other regularity assumptions. | Established testing route |
| Carryover-aware schedule design | Fletcher, Lewis, and Matthews (1990); Basse, Ding, and Toulis (2023); Bojinov, Simchi-Levi, and Zhao (2023); Jiang and Ding (2026) | Factorial direct/carryover effects, balance, path-dependent estimands, and efficiency under supported sequences are established. | Factorial specialization only |
| Exact cross-fit loading and finite-cell covariance correction | General influence propagation, residual degrees-of-freedom correction, and feasible GLS | Gate G verified the algebra and finite-sample behavior, but point-risk changes were negligible and the main performance gate failed. | Supporting result only |

## What the direct crossover competitor changes

Jiang and Ding's August 10, 2026 revision is especially consequential. It does
not merely mention carryover. It:

- defines arbitrary fixed vector causal estimands as linear functionals of
  potential outcomes over treatment sequences;
- separates the target sequence set from the sequences supported by the
  experiment;
- encodes no-carryover and other causal restrictions as linear coefficient
  restrictions;
- gives rank conditions for identification;
- derives the covariance-weighted BLUE by restricted WLS;
- proves first-order equivalence after estimating sequence-specific covariance
  matrices from the same data; and
- supplies robust design-based variance and tests of restrictions.

The current project adds factorial organization, target-profile distributions,
covariates, cross-fitting, and sensitivity displays. Those are useful extensions
for conjoint practice. But combining Jiang-Ding's finite-support result with
conditional-moment efficiency and approximate-moment sensitivity produces the
current theorem stack without a new hard step.

One distinction should be preserved accurately. Conditional on independently
trained nuisance functions, the proposed cross-fitted correction has exact
foldwise unbiasedness. Jiang and Ding instead establish first-order equivalence
for same-sample feasible covariance weighting in a fixed-cell regime. That is a
real technical difference, but exact sample splitting is a standard device and
Gate G found no consequential point-estimation improvement from the correction.
It is not enough to reverse the novelty verdict.

Likewise, the target distribution is not a new ingredient in conjoint analysis:
de la Cuesta, Egami, and Imai establish that the AMCE depends on the profile
distribution over which it is averaged and develop target-distribution-aware
design and estimation. The present project integrates that idea with repeated-
task covariance and carryover restrictions; it does not originate it.

## Claim boundary

The project **can** claim:

- a clear target-preserving translation of efficient and sensitivity-aware
  inference to repeated factorial/conjoint experiments;
- a coding-invariant implementation for fixed target-distribution-weighted
  contrasts;
- an explicit display of the price of robustness for prespecified carryover
  spaces;
- respondent-isolated nuisance fitting and inference; and
- careful negative evidence that flexible covariance ML may add little beyond
  simpler estimators in realistic conjoint sample sizes.

It **cannot** claim:

- a new general EIF or efficiency bound principle;
- a new structured-carryover estimator in the abstract;
- a new minimax robustness path or bias-aware interval;
- a new theory of adaptation between restricted and unrestricted estimators;
- a new general method for crossover identification or efficient estimation;
  or
- evidence that ML generally has no value in factorial experiments.

## Gate decision

The prespecified full-paper gate required at least one nontrivial result that
could not be obtained by relabeling existing conditional-moment, crossover, or
approximate-misspecification theory. None remains after the direct mapping
above. The conjunction itself is also weakened by Jiang and Ding's general
crossover framework.

**Therefore the stand-alone full methods route stops here.** The result should
not be sent out under a title or abstract implying invention of a new efficient,
ML, or robust estimator.

## Viable outputs

1. **Applied-methods synthesis or tutorial.** Present unrestricted taskwise
   AIPW as the assumption-light default, explain exactly what no carryover buys,
   provide the structured robustness ladder, and show when covariance learning
   is or is not worth the complexity.
2. **Software contribution.** Package the target-weighted factorial basis,
   support diagnostics, fixed-target score construction, clustered cross-fit
   inference, and sensitivity summaries, with the general methods credited to
   their sources.
3. **Substantively led conjoint paper.** Use the framework as analysis
   infrastructure for a question where complementarity or task-history effects
   are substantively important; the statistical machinery is supporting rather
   than the headline contribution.

The proposed follow-up theorem on **feasible finite-cluster optimality for the
complete overlapping cross-fit estimator** has now been searched directly and
does not reopen the full-methods route. The exact two-fold expansion contains
one crossed term, but that term is zero for the intended Gaussian covariance
learner by odd/even symmetry. In the faithful fixed-operator factorial
enumeration, the complete and marginal quadratic risks are exactly
proportional, so their minimizers coincide. A regular counterexample with
different optima exists, but the optimizer difference is only (O(n^{-1})),
the risk gain from reoptimization is (O(n^{-3})), and the underlying overlap
terms are already covered by higher-order GMM and DML expansions. See the
[`finite-cluster theorem-search decision`](finite-g-theorem-search-gate.md).

The only remaining theoretical escape identified by the search is a genuinely
different growing-profile occupancy problem, with the number of profile cells
of the same order as the number of clusters so empty cells persist. Such a
paper would need an occupancy-aware limiting risk, a feasible calibrated
minimizer, and uniform clustered inference. It should not be pursued unless a
small preregistered screen first shows a material gain (roughly 5% or more) in
a substantively meaningful sparse-support regime. Merely adding learners,
applications, or simulations would not be sufficient.

## Primary sources checked

- Chamberlain (1987), [Asymptotic efficiency in estimation with conditional
  moment restrictions](https://doi.org/10.1016/0304-4076(87)90015-7).
- Newey (1990), [Efficient instrumental variables estimation of nonlinear
  models](https://doi.org/10.1002/jae.3950050202).
- Liang and Zeger (1986), [Longitudinal data analysis using generalized linear
  models](https://doi.org/10.1093/biomet/73.1.13).
- Hainmueller, Hopkins, and Yamamoto (2014), [Causal inference in conjoint
  analysis](https://doi.org/10.1093/pan/mpt024).
- de la Cuesta, Egami, and Imai (2022), [Improving the external validity of
  conjoint analysis: the essential role of profile
  distribution](https://doi.org/10.1017/pan.2020.40).
- Chernozhukov et al. (2018), [Double/debiased machine learning for treatment
  and structural parameters](https://doi.org/10.1111/ectj.12097).
- Chernozhukov, Newey, and Singh (2022), [Automatic debiased machine learning of
  causal and structural effects](https://doi.org/10.3982/ECTA18515).
- Chiang et al. (2022), [Multiway cluster robust double/debiased machine
  learning](https://doi.org/10.1080/07350015.2021.1895815).
- Stephens, Tchetgen Tchetgen, and De Gruttola (2014), [Locally efficient
  estimation of marginal treatment effects when outcomes are
  correlated](https://doi.org/10.1515/ijb-2013-0031).
- Egami and Imai (2019), [Causal interaction in factorial
  experiments](https://doi.org/10.1080/01621459.2018.1476246).
- Fletcher, Lewis, and Matthews (1990), [Factorial designs for crossover
  clinical trials](https://doi.org/10.1002/sim.4780091002).
- Armstrong and Kolesar (2021), [Sensitivity analysis using approximate moment
  condition models](https://doi.org/10.3982/QE1609).
- Bonhomme and Weidner (2022), [Minimizing sensitivity to model
  misspecification](https://doi.org/10.3982/QE1930).
- Park and Kang (2022), [A more efficient, doubly robust, nonparametric
  estimator of treatment effects in multilevel
  studies](https://arxiv.org/abs/2110.07740).
- Basse, Ding, and Toulis (2023), [Minimax designs for causal effects in
  temporal experiments with treatment
  habituation](https://doi.org/10.1093/biomet/asac024).
- Bojinov, Simchi-Levi, and Zhao (2023), [Design and analysis of switchback
  experiments](https://doi.org/10.1287/mnsc.2022.4583).
- Madsen, Scheike, and Pipper (2023), [Unbiased and efficient estimation of
  causal treatment effects in crossover
  trials](https://doi.org/10.1002/bimj.202200170).
- Ham, Imai, and Janson (2024), [Using machine learning to test causal
  hypotheses in conjoint analysis](https://doi.org/10.1017/pan.2023.41).
- Shi and Ye (2024), [Behavioral carry-over effect and power consideration in
  crossover trials](https://doi.org/10.1093/biomtc/ujae023).
- Cheng, Bell, and Qian (2024), [Efficient and globally robust causal excursion
  effect estimation](https://arxiv.org/abs/2311.16529).
- Armstrong, Kline, and Sun (2025), [Adapting to
  misspecification](https://arxiv.org/abs/2305.14265).
- Missault et al. (2025), [Robust and efficient multiple-unit switchback
  experimentation](https://arxiv.org/abs/2506.12654).
- Xia et al. (2026), [Robust and efficient semiparametric inference for the
  stepped wedge design](https://arxiv.org/abs/2510.08972).
- Acharya, Hainmueller, and Xu (2026), [Learning preferences from conjoint data:
  a hybrid structural deep learning
  approach](https://arxiv.org/abs/2604.10845).
- Jiang and Ding (2026, version 3), [Principled analysis of crossover designs:
  causal effects, efficient estimation, and robust
  inference](https://arxiv.org/abs/2511.09215).

This was a focused, claim-driven primary-source audit, not a registered
systematic review. The decisive sources were read at the level needed to map
the candidate estimand, restrictions, score, and optimization problem—not only
by title similarity.
