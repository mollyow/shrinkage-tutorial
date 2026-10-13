# Gate E1 protocol: carryover sensitivity theory

## Question

Gate E1 asks whether the efficiency result can be accompanied by a principled
analysis of violations of the task-specific mean no-carryover restriction.  It
is a theory gate, not a performance simulation.

The exercise must distinguish the assumption-light task-marginal factorial
target from the smaller no-carryover model.  It must not treat failure to reject
a carryover diagnostic as evidence that no carryover exists.

## Required outputs

Gate E1 requires all of the following:

1. An exact finite-sample expectation formula for the covariance-adaptive
   score under arbitrary conditional-mean carryover.
2. A local-misspecification limit comparing the restricted-efficient and
   unrestricted estimators.
3. An interpretable norm on departures from the no-carryover mean space and a
   sharp worst-case bias bound for fixed factorial contrasts.
4. Coding-invariant coordinate and family sensitivity summaries, including a
   critical violation value.
5. A precise account of what a paired robust-versus-efficient diagnostic can
   and cannot test.
6. Conditions under which learned covariance weights and a finite-library
   selector retain a conventional cross-fitted limit.
7. Exhaustive finite-support checks of the algebra, including a nonzero blind
   carryover direction.

## Novelty rule

Hausman comparisons, local misspecification, minimax interpolation, and
bias-aware moment inference are prior methods.  Gate E1 passes only if their
specialization to repeated target-weighted factorial experiments produces a
useful additional result, such as an exact factorial carryover-bias operator,
a sharp coding-invariant sensitivity geometry, or a computationally useful
efficiency-versus-vulnerability identity.

## Decision rule

Proceed to Gate E2 only if the theory:

- preserves a clearly defined causal task-marginal target under carryover;
- produces a sensitivity scale that can be fixed before seeing outcomes;
- yields falsifiable numerical identities;
- gives a valid unrestricted estimator that can remain primary; and
- supports a bounded simulation design rather than an open-ended robustness
  exercise.

If these conditions fail, retain the unrestricted estimator and describe the
covariance-adaptive result only as an oracle calculation under no carryover.

