# Weak-overlap theorem gate: target factorial contrasts under product shift

**Frozen:** 2026-08-31, before retaining numerical gate output.  
**Purpose:** falsify, rather than presume, a stand-alone theorem about learning
target-weighted factorial contrasts when the randomized profile distribution
and scientific target distribution differ increasingly with the number of
attributes.

## Question

Can a fixed low-order factorial contrast remain honestly estimable when the
full target-to-assignment density ratio is unstable, if the response surface is
only *approximately* low order?  A paper-level result must do more than apply
generic minimax-linear, RKHS, covariate-shift, or high-dimensional-regression
theory in a Walsh basis.

## Smallest model

Let $A=(A_1,\ldots,A_J)\in\{-1,1\}^J$.  The scientific target is the uniform
product law \(\mu_J\).  The assignment law is the product distribution

\[
g_{J,\delta}(A)=\prod_{j=1}^J
\left\{\frac{1+\delta_j}{2}\right\}^{1(A_j=1)}
\left\{\frac{1-\delta_j}{2}\right\}^{1(A_j=-1)},
\qquad |\delta_j|<1.
\]

Write the target-Walsh expansion as

\[
m(A)=\sum_{T\subseteq[J]}\beta_T\phi_T(A),
\qquad \phi_T(A)=\prod_{j\in T}A_j.
\]

For a fixed set \(S\), the target is

\[
\theta_S=E_{\mu_J}\{m(A)\phi_S(A)\}=\beta_S.
\]

The primary screen uses \(|S|=1\), equal shift \(\delta_j=\delta\), Gaussian
homoskedastic errors, and known assignment.  Unequal shifts and higher-order
targets are deterministic checks, not separate opportunities to tune the
gate.

## Structural classes

The calculations must distinguish assumptions that are often blurred.

1. **Exact degree:** \(\beta_T=0\) for \(|T|>q\).
2. **Approximate degree:** a target-basis ellipsoid

   \[
   \mathcal F_J(R,W)=
   \left\{m:\sum_T w_{|T|}\beta_T^2\le R^2\right\},
   \]

   with weights fixed before results are examined.  The headline family uses
   \(w_r=\kappa^{2(r-|S|)_+}\), \(\kappa\in\{1.5,2,4\}\), normalized so the
   target coefficient has radius \(R\).
3. **Tail-radius sensitivity:** \(\sum_{|T|>q}\beta_T^2\le B_J^2\).  This is
   used to state exactly how fast an unrestricted tail would have to vanish;
   it is not silently treated as exact truncation.

The gate must say when bounded outcomes, sparsity, heredity, or a Bayesian
prior would be an additional assumption rather than a consequence of order
decay.

## Required analytic results

Before a performance claim, derive and verify:

1. the target Riesz representer and its unrestricted variance under product
   shift;
2. the change of basis between target-Walsh and assignment-orthonormal
   characters;
3. the minimum variance for \(\theta_S\) under an exact degree-\(q\) model;
4. the contamination coefficients and worst-case bias of the corresponding
   degree-\(q\) estimator under approximate degree;
5. the minimax-linear estimator and risk over the ellipsoid in the population
   Gaussian regression experiment;
6. at least one lower-bound or modulus argument showing whether the displayed
   rate is intrinsic rather than an artifact of the chosen estimator; and
7. which claims remain valid with sparse realized profile occupancy rather
   than a population-Gram or full-cell Gaussian approximation.

Every closed form must agree with direct matrix calculation for all
\(J\le 8\), several unequal shift vectors, and targets of orders one and two to
absolute tolerance \(10^{-10}\).

## Numerical screen

The deterministic risk grid is fixed at:

- \(J\in\{4,6,8,10,12\}\) where direct computation is feasible;
- \(n\in\{250,1000,4000,16000\}\);
- equal \(\delta\in\{0,0.2,0.5,0.7\}\);
- \(\kappa\in\{1.5,2,4\}\), \(R=1\), and \(\sigma=1\);
- truncation orders \(q=|S|,\ldots,\min(J,5)\), plus unrestricted and
  ellipsoid-minimax linear rules.

Report variance, maximum squared bias, worst-case MSE, and a conservative 95%
bias-aware half-length.  The zero estimator and the full unbiased Riesz
estimator are mandatory anchors.  Population-Gram results are explicitly
labelled optimistic when \(2^J>n\).  Any apparent pass in such a cell must be
rechecked in an iid random-design simulation with common datasets.

## Pass rules

The route passes only if **all** of the following survive.

1. **Nontrivial approximate regime.**  There is a sequence with genuinely
   unstable full-profile weights in which an honest interval shrinks under an
   approximate, not exact, order restriction without requiring an
   exponentially vanishing tail radius or a target shift so small that full
   overlap is effectively restored.
2. **Intrinsic result.**  A matching lower bound or sharp modulus identifies a
   factorial phase transition, not merely the generic solution of a linear
   inverse problem over an ellipsoid.
3. **Computational relevance.**  The feasible rule avoids enumeration and, in
   supported random-design cells, reduces worst-case MSE by at least 10%
   relative to both the full Riesz rule and the best prespecified truncation,
   while producing a shorter-than-trivial honest interval.
4. **Robustness.**  The conclusion is not confined to an exactly matched
   Gaussian prior, exact low degree, or a single tuned shift/decay cell.
5. **Novelty.**  Exact mapping to the nearest high-dimensional-treatment,
   minimax-retargeting, distribution-shift regression, off-policy evaluation,
   and honest minimax-inference results leaves at least one nontrivial theorem.

Failure of either rule 1 or rule 5 ends the stand-alone methods route even if
the numerical estimator looks good.

## Mandatory reducibility audit

The decision must map the candidate result to, at minimum:

- minimax and augmented minimax linear estimation of retargeted means;
- optimal affine inference over convex function classes;
- near-optimal linear regression and RKHS regression under covariate shift;
- high-dimensional binary-vector treatment estimation under exact and
  approximate sparsity;
- factored-action off-policy evaluation; and
- Fourier/low-rank recovery for combinatorial interventions.

The words “factorial,” “ANOVA,” or “target distribution” do not establish
novelty if the estimator is obtained by substituting a product kernel or Walsh
basis into one of those results.

## Pivot rule

If approximate order decay cannot overcome weak overlap honestly, retain the
closed-form calculation as a negative design diagnostic.  The only possible
follow-up would require a substantively justified stronger class—such as exact
degree, sparse influential coordinates, monotonicity/shape restrictions, or a
common-study fusion structure—and would receive a new gate rather than a
weakened interpretation of this one.
