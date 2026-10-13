# Income-tax-plan design-contract decision

**Decision date:** 2026-09-01  
**Gate result:** green for a narrow paired-choice analysis of total effects of
the six randomized bracket rates  
**Outcome status:** still prohibited

## Decision

The Ballard-Rosa, Martin, and Scheve income-tax conjoint can be the third fresh
study in the negative comparative benchmark, but only under a narrower
estimand than a seven-attribute conjoint surface.

The treatment is the complete six-rate tax schedule. The displayed revenue
category is generated after those rates are assigned: the authors calculate a
revenue amount, add a normal perturbation, and bin it. It is therefore a
stochastic downstream mediator, not a seventh independently randomized
factor. The eligible estimand averages over that natural display mechanism.
Controlled revenue effects, mediation effects, and factorial interactions
involving revenue are not identified by the locked design.

This decision used archive structure, treatment fields, source documentation,
and baseline covariates only. The audit's `haven::read_dta()` call is restricted
to an explicit allowlist. It does not load a choice, ideal-rate response,
derived preference score, attention indicator, or learner result.

## Archive and sample reconstruction

The immutable lock is Harvard Dataverse version 1.0, released April 5, 2016
under CC0 1.0. The release contains one processed 36,000-row Stata file and one
analysis do-file. It omits the raw survey export, the original randomizer,
recode code referenced by the do-file, the IRS revenue calculator, and its
random seed.

The processed file contains:

- 2,250 unique respondents;
- eight tasks and two explicitly labelled plans per respondent;
- 18,000 paired tasks and 36,000 profile rows; and
- complete, respondent-invariant survey weights.

The author code defines the main analysis as `saw_revenue == 1`. This gives
2,000 respondents, 16,000 tasks, and 32,000 profile rows. The remaining 250
respondents completed an otherwise identical version without seeing the
revenue category. That group is not the separate 500-person rating pilot
mentioned in article footnote 15. Neither the pilot observations nor a rating
field are in the archive.

All 2,000 main-arm respondents are retained. In particular, the design sample
is not filtered using the post-conjoint attention check or completion speed.
Five outer folds and five independent inner labels are assigned by respondent,
so all 16 rows for one person remain together.

## Assignment and support

The six randomized factors have 4, 4, 4, 4, 5, and 6 levels, respectively.
Their unrestricted product contains 7,680 rate schedules. The source describes
pairs of randomly generated plans, says each bracket rate was randomly
assigned for each pair, invokes conditional independent randomization, and
states that the six rates were independently varied.

The prospective assignment lock is therefore

\[
q(r)=\frac{1}{7{,}680}, \qquad
g(r_A,r_B)=q(r_A)q(r_B)=\frac{1}{58{,}982{,}400}.
\]

This is a source-supported reconstruction, not a claim that the missing
generator was inspected. Realized design checks are consistent with it:

- 7,554 schedules are observed in the main arm, compared with an iid-uniform
  occupancy expectation of 7,560.96;
- all 15 two-factor Cartesian products are complete;
- every one-way distribution remains close to uniform by task and side;
- cross-side factor tables show no material departure from independence;
- zero tasks contain identical complete schedules, but 2.083 are expected and
  the iid probability of zero is 0.124, so this is not anomalous; and
- 32 equal-schedule pairs recur across different tasks within respondents,
  compared with 29.17 expected under independent row draws. Thus the archive
  shows repeats rather than a no-repeat rule.

These diagnostics check for contradictions; they do not estimate the locked
assignment law from the realized sample. Across-task dependence is left
unrestricted for respondent-clustered covariance. Because the exact joint
history law is unavailable, the contract prospectively imposes the article's
mean no-carryover condition: conditional expected current choice does not
depend on schedules shown in earlier tasks after conditioning on the current
task, focal and opponent schedules, side, and baseline variables.

## Why revenue is averaged rather than controlled

For a schedule `r`, the eligible response surface can be written informally as

\[
\bar m(r)=
E\{Y(r,Z_r;B,Z_B)\},
\]

where the expectation averages over the weighted respondent population, task
and side, an opponent schedule `B` drawn from `q`, and the two naturally
generated noisy revenue displays. The experiment randomizes `r`, so its total
effect is identified without reconstructing the conditional law of `Z_r`
given `r`.

The distinction matters operationally. `taxrev` may be retained by the design
audit to verify the mechanism, but it is on the learner's never-feature list.
Conditioning a primary nuisance regression on realized revenue would change
the surface and would require integrating against an unreleased
`q(taxrev | rates)` law. The data confirm the stochastic mechanism: hundreds
of repeated rate schedules appear with more than one revenue category.

The labelled `taxrate5` field is authoritative: code 5 means 45 percent and is
well represented. A convenience numeric field in the processed archive maps
that code incorrectly; it is outside the read allowlist.

## Target, covariates, and contrasts

The respondent target is the displayed-revenue main-study population, using
the released survey weight normalized within its 2,000 respondents. The
profile target is uniform over 7,680 schedules, with an independent uniform
opponent, equal averaging over eight task positions and both profile sides,
and natural averaging over the noisy revenue mechanism.

The primary learner receives only the seven variables the online appendix
says YouGov used to construct the matched sample before the experiment:

- age;
- female;
- race;
- education;
- seven-category party identification plus its `not sure` category;
- ideology; and
- political interest.

Ideology has 182 missing respondents and political interest has 46; all other
locked features are complete. Missing categories are explicit, while any
numeric centering, scaling, or imputation is learned inside training folds.

The article states that subgroup-analysis survey covariates were collected
post-experiment. The contract therefore conservatively excludes the behavioral
games, income, tax attitudes, racial resentment, knowledge, numeracy, ideal
rates, and quality-control fields even where archive ordering might suggest an
earlier page.

Three whole uniform-target interaction families are fixed:

| Family | Cells | Dimension |
|---|---:|---:|
| Lowest bracket x highest bracket | 24 | 15 |
| $85,000-$175,000 x highest bracket | 24 | 15 |
| $175,000-$375,000 x highest bracket | 30 | 20 |

Every coordinate is identified and must remain in the reported family. The
audit emits deterministic target-orthonormal coordinates and function-space
projectors so the estimands do not depend on a reference category, a level
permutation, or a nonsingular rotation of coordinates.

## Scope of the pass

This study counts once toward portfolio Gate B through the forced-choice total
schedule analysis. It does not make any of the following eligible:

- a seven-factor surface containing revenue;
- direct or controlled effects that hold revenue fixed;
- revenue mediation effects;
- a headline analysis of the 250-person no-revenue arm; or
- the unavailable rating pilot.

This decision initially left the portfolio at three eligible fresh families.
The subsequent Graham-Svolik audit passed for a narrow coarsened nonproduct
analysis, so the four-study Gate B is now satisfied. The cross-study benchmark
must still be frozen before any new outcome comparison is run.
