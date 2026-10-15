# Climate-agreement conjoint: outcome-blind Gate-B audit

This directory freezes the design of Bechtel and Scheve's four-country climate-
agreement conjoint before any new response, interaction, or learner result is
opened. The audit reads only identifiers, weights, randomized attributes,
design context, and prospectively classified respondent features. It never
loads `choice_cj`, `rating_cj`, or the outcome-derived `incon` flag.

## Decision

The audit has a deliberately split result:

- **Profile marginal and support: green under the locked reconstruction.** The
  source says fully randomized but supplies no generator or explicit
  probabilities. The prospectively declared uniform product contains 2,880
  profiles, all of which occur, and its one-way and full-profile diagnostics
  are compatible. All 15 two-factor supports are complete overall and within
  each country.
- **Rating analysis: conditionally green.** It may enter the benchmark only
  under prospectively declared mean-no-opponent and mean-no-carryover
  restrictions: once the current focal profile, task, and pre-treatment
  respondent variables are fixed, its mean rating depends on neither the
  paired opponent nor profiles from earlier tasks.
- **Paired choice: amber and excluded.** The archive does not identify the
  exact opponent law.

The reason for the last distinction is not cosmetic. None of the 34,000 tasks
contains two identical complete profiles. Independent uniform draws predict
11.806 such pairs, and the probability of observing zero is
`7.4477e-6`. Individual-attribute match frequencies look ordinary and complete
profiles repeat across different tasks, so the evidence points specifically to
an undocumented within-pair rejection rule. The small probability assumes iid
task pairs; with arbitrary dependence among four tasks, zero duplicates does
not logically reject a per-task `q x q` marginal. The audit's conclusion is
therefore the narrower—and sufficient—one: the exact opponent law is not known.

The machine-readable contract is
[`config/datasets/climate-agreements.yml`](../../config/datasets/climate-agreements.yml).
The benchmark plan is [`docs/plan.md`](../../docs/plan.md).

## Run

From the repository root:

```sh
Rscript analysis/climate/audit_design.R
Rscript analysis/climate/tests/testthat.R
```

The audit uses `haven::read_dta(col_select=...)` as a strict read barrier. It
fails closed on a checksum change, an unexpected generated file, a forbidden
column, a sample/support/rank mismatch, respondent leakage across folds, or a
failed invariance test. Raw archive files and the generated
`analysis/climate/derived/` bundle are gitignored.

## Locked design

The analysis universe has 8,500 respondents: 2,000 each in France, Germany,
and the United Kingdom, and 2,500 in the United States. Each respondent rates
eight profiles in four paired tasks, for 68,000 profile rows. The six factors
are household cost, burden distribution, participating countries, emissions
coverage, sanctions, and monitoring. The display order is randomized once per
respondent and fixed across tasks, but its 696 observed archive codes are
opaque and are averaged over rather than modeled as sparse fixed effects.

The primary target gives each country one quarter of the population mass,
normalizes the released survey weights within country, and uses the uniform
profile distribution. The six primary learner features are country, age,
female, country-specific education, country-specific income, and the climate-
cooperation support item documented before the conjoint. Other attitudes,
knowledge questions, willingness to pay, environmentalism, ideology, and the
attention check remain excluded from the primary learner.

## Registered whole families

| Family | Cells | Interaction dimension |
|---|---:|---:|
| Household cost x burden distribution | 20 | 12 |
| Sanctions x monitoring | 16 | 9 |
| Participating countries x emissions coverage | 9 | 4 |

Every cell occurs in every country and task position. The emitted bases and
projectors are target-orthonormal and invariant to baselines, level order, and
nonsingular coordinate rotations.

Paired choice becomes eligible only if the original generator is recovered or
the authors confirm the no-identical-profile rule and exact conditional
opponent probabilities. Otherwise it stays out of the benchmark.
