# Gate E3 protocol: limits of automatic adaptation

## Question

Gate E3 asks whether Gate E2's robustness--efficiency tradeoff can support a
stand-alone factorial-methods paper through two theoretical results:

1. a lower bound showing why a data-driven rule cannot uniformly reproduce the
   better of unrestricted and restricted-efficient inference; and
2. a constructive estimator that is efficient while remaining exactly robust
   to a prespecified structured carryover class.

This is a bounded theory and novelty exercise. It does not permit another large
simulation to substitute for an absent contribution.

## Required derivations

The exercise must produce all of the following.

1. Reduce one fixed factorial contrast's Gate E1 local experiment to a
   canonical Gaussian decision problem.
2. Prove at least one explicit no-oracle or constrained-risk statement and
   identify the strongest existing result that already covers it.
3. Derive the optimal fixed linear rule for a known sensitivity radius and
   show its relation to the Gate E1 blend.
4. For a prespecified carryover basis `Z`, derive the efficient weight that is
   exactly centered over the enlarged mean model `[B, Z]`, its efficiency
   bound, and its relationship to the no-carryover and unrestricted scores.
5. Check the scalar reduction and structured calibration on an enumerated
   finite factorial design.
6. Give a blunt novelty verdict that separates useful factorial translation
   from new statistical decision theory.

## Pass rule

Gate E3 passes as a stand-alone-paper theory gate only if both conditions hold:

- the adaptation lower bound uses randomized factorial structure in a way not
  reducible to established restricted-versus-unrestricted Gaussian adaptation;
  and
- the structured estimator or interval has a nontrivial optimality property
  beyond applying existing covariance-weighted calibration or approximate-
  moment theory to an augmented mean design.

Exact algebra, useful software, or an appealing application cannot compensate
for failure of the novelty rule.

## Pivot rule

If the Gaussian reduction is already covered by prior work, preserve it as a
warning against claiming that generic ML solves the problem. If augmenting the
mean design yields a useful but generic structured estimator, retain it as a
possible component of a broader repeated-factorial-inference paper. Do not
market either result as a new general adaptation method.

