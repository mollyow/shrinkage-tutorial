# Comparison/software paper route

> **Superseded for the active route.** The current prospective negative-paper
> and software plan is
> [`docs/negative-comparative-paper-software-plan.md`](../docs/negative-comparative-paper-software-plan.md).
> This older outline remains as the narrower order-ridge comparison history.

## Working title

**Coding-Invariant Shrinkage for Causal Factorial Response Surfaces**

“Order-Adaptive Shrinkage for Causal Factorial Response Surfaces” remains the
methods-route title only if a later eligible production run reverses the Stage
3 gate.

## Central contribution

The paper should no longer argue that the monotone restriction is a generally
superior new estimator. Its defensible contribution is a transparent,
target-distribution-weighted implementation that makes three established ridge
strategies comparable on the same causal factorial geometry:

1. one common penalty;
2. unrestricted order-specific penalties; and
3. monotone order-specific penalties.

The accompanying inferential contribution is an implementation and
specialization of a known-randomization, cross-fitted correction for a fixed set
of prespecified AMIEs. Cross-fitting and AMIEs themselves are established.

## Paper structure

1. **The coding problem.** Show how penalizing standardized reference-coded
   coefficients changes fitted functions under innocuous reparameterizations.
2. **Target-weighted factorial geometry.** Define $\mu$, $\mathcal V_k$, and
   $\mathcal H_k$; establish the invariance result; report support diagnostics.
3. **Three ridge strategies on one geometry.** Present common, unrestricted,
   and monotone order penalties as structured generalized ridge. Treat the
   balanced PAVA result as a fast established special case.
4. **When does the order restriction help?** Report the proof-of-concept
   negative result, then benchmark hierarchy strength, information, factor
   arity, and within-order heterogeneity without universal-improvement claims.
5. **Inference for fixed causal contrasts.** State the exact held-out
   augmentation identity, fixed-dimensional cluster CLT assumptions, and
   coverage evidence if a later inference simulation is run.
6. **Candidate-recruitment illustration.** Run only after an explicit decision
   to continue the comparison route. Use the locked 22-coordinate/four-family
   AMIE specification and treat profile rankings as descriptive.
7. **Practical guidance.** Explain when to use common, unrestricted, monotone,
   or truncated fits; foreground support, coding, tuning stability, and the cost
   of high-order surfaces.

## Next evidence in order

1. Obtain an independent proof audit of coding invariance, the PAVA
   specialization, foldwise unbiasedness, and the fixed-$q$ cluster CLT.
2. Decide whether a 1,000-replication focused confirmation is worth the compute.
   It estimates the size of a small effect; it does not replace the production
   gate.
3. If continuing, add competitors in compatible experiments and frame the
   exercise as a benchmark rather than a horse race designed to vindicate
   monotonicity.
4. Run clustered AMIE coverage simulations before reporting application
   intervals.
5. Only then execute the locked application checkpoint and full fit.

## Claims to avoid

- “First” or “novel” multi-penalty/monotone ridge.
- General dominance over unrestricted order ridge.
- Inference for selected interactions or the full fitted surface.
- Heredity: the order restriction encodes declining average signal by order.
- “Optimal” institutional design based on a ranked fitted profile.
