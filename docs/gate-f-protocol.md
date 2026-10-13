# Gate F protocol: factorial schedule geometry

## Question

Gate F asks whether the repeated-factorial project has a design result that is
both scientifically useful and sufficiently distinct from generic crossover
design.  The candidate result uses the product structure of a binary factorial
profile space to connect randomized profile transitions, interaction order,
efficient scoring, and vulnerability to mean carryover.

This is a theory-and-enumeration gate.  A simulation win is neither required
nor sufficient.

## Prespecified exercise

Work under a transparent reference case:

- `F` binary randomized attributes and the uniform target distribution;
- `T` repeated tasks per independent respondent;
- stationary schedules whose transition probabilities depend only on Hamming
  distance, together with independent product assignment and uniform sampling
  without replacement as benchmarks;
- a task covariance matrix `Sigma` that is constant across schedules; and
- fixed task weights `omega` for each reported factorial contrast.

The exercise must do all of the following.

1. Diagonalize a Hamming-distance schedule in the Walsh factorial basis and
   identify the schedule eigenvalue for every interaction order.
2. Reduce covariance-efficient scoring for every order to a `T` by `T`
   calibration problem and verify the reduction by finite enumeration.
3. Derive the exact carryover-bias operator, including its rank and null space,
   for additive cross-task factorial carryover.
4. Compare product and without-replacement schedules analytically in at least
   one nontrivial covariance class.
5. Produce an actionable order-specific scheduling rule or optimization
   problem whose size grows polynomially in `F` and `T`, rather than with the
   `2^(F T)` schedule space.
6. Audit the result against classical factorial changeover designs, modern
   temporal causal design, and the classical Hamming/Krawtchouk spectral
   identity.

## Correctness requirements

Deterministic checks must cover:

- Walsh orthogonality and the Krawtchouk eigenvalue formula;
- cross-order and cross-feature orthogonality at multiple time lags;
- product, no-immediate-repeat, antipodal, and whole-sequence
  without-replacement special cases;
- agreement between the reduced score and direct finite-support calibration;
- the efficiency-gap/projection-norm identity;
- the carryover operator's stated rank and kernel dimensions; and
- the closed-form two-task and compound-symmetry comparisons.

## Decision rule

Gate F has two separate components.

The **mathematical/actionability component** passes only if the factorial
schedule problem reduces exactly by interaction order, exposes a nontrivial
carryover null space, and yields a practical schedule comparison or optimizer.

The **stand-alone novelty component** passes only if the central theorem is not
a direct combination of:

- classical factorial-structure/canonical-efficiency results for changeover
  designs;
- a standard Krawtchouk diagonalization of a radial Markov chain on the
  hypercube; and
- the covariance-calibration and sensitivity identities already derived in
  Gates E1--E3.

A previously unstated notation, a faster calculation alone, or a conjoint
interpretation alone does not pass the novelty component.

## Pivot rule

If the mathematics passes but stand-alone novelty fails, retain the spectral
schedule result as one theorem and a software module in a broader paper on
efficient and carryover-aware inference for repeated factorial experiments.
Do not market it as a new general theory of crossover design.  The next gate
must then solve a remaining inferential problem, not merely add more schedule
examples.
