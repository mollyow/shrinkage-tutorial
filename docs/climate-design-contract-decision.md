# Climate-agreement conjoint: Gate-B design-contract decision

**Decision date:** 2026-09-01  
**Overall result:** split—rating conditionally green; paired choice amber  
**Outcome status:** `choice_cj`, `rating_cj`, and outcome-derived `incon` remain unopened

## Decision

The Bechtel-Scheve climate-agreement study contributes one eligible analysis to
the prospective benchmark: target-weighted profile-rating contrasts under
declared **mean-no-opponent** and **mean-no-carryover** restrictions. Its
paired-choice outcome is not eligible.

This is not a compromise label for an otherwise clean independent-pair design.
The archive contains zero identical complete-profile pairs among 34,000 tasks.
Independent uniform draws from 2,880 profiles predict 11.806, and their exact
probability of producing zero is about `7.4477e-6`. Attribute-by-attribute
matches are ordinary, while 67 repeated complete-profile rows occur across
different tasks within respondents. The most plausible explanation is an
undocumented rule that rejected an identical opponent within a task.

The exact conditional opponent law is therefore unknown. The zero count
strongly disfavors globally iid task-pair draws, but it does not logically
reject a per-task `q x q` marginal when the four tasks within a respondent may
be dependent. Indeed, under the most conservative perfect dependence of the
four duplicate indicators, the respondent-cluster upper probability of zero
is about 0.0522. The audit therefore does not write `q x q` into the contract
as known merely because the article describes a fully randomized conjoint.

The executable record consists of the
[machine-readable contract](../config/datasets/climate-agreements.yml),
[design audit](../analysis/climate/audit_design.R), and
[tests](../analysis/climate/tests/testthat/test-design-audit.R).

## Why the rating route is narrower

Write `R_i(a,b)` for respondent `i`'s potential rating of focal profile `a`
when it is paired with opponent profile `b`. The rating analysis prospectively
assumes

\[
E\{R_i(a,b)\mid X_i,\text{task}\}
=
E\{R_i(a,b')\mid X_i,\text{task}\}
\quad\text{for every }b,b'.
\]

This is the mean-no-opponent restriction. It says only that the opponent does
not change the focal profile's **conditional mean rating**. It does not require
the two displayed profiles to be independent.

Write `H_it` for the profiles shown before task `t`. The companion mean-no-
carryover restriction is

\[
E\{R_{it}(a,b,h)\mid X_i,t\}
=
E\{R_{it}(a,b,h')\mid X_i,t\}
\quad\text{for every }h,h'.
\]

It allows arbitrary statistical dependence among a respondent's rows for
variance estimation while ruling out a mean effect of prior displayed profiles
on the current rating. Both restrictions are identifying assumptions, not
facts guaranteed by randomization. The benchmark must state them, and later
sensitivity analyses should assess plausible opponent and carryover effects.

Under both restrictions, the prospectively locked current-row profile-law
reconstruction is sufficient for the fixed target-weighted rating contrasts
despite uncertified within-pair and across-task joint laws. Without them, a
rating contrast may compare focal profiles under different opponent or
profile-history environments. Forced choice necessarily depends on both
current profiles, so these restrictions do not rescue the paired-choice
endpoint.

## Provenance lock

The primary sources are the published
[article](https://doi.org/10.1073/pnas.1306374110), its supplementary
information, and the authors' [Harvard Dataverse archive](https://doi.org/10.7910/DVN/UGZ2BY).
The deposit is version 1.0, released under CC0 1.0. It contains the article,
supplement, original Stata data, README, and author analysis code. It does not
contain the survey-platform generator, a standalone full questionnaire, an
assignment-probability file, or a profile-side field.

The original Stata binary is 3,001,271 bytes with MD5
`85dc26aeda5a99e881ad777f03d8410c`. Dataverse's 6,188,201-byte figure refers to
its ingested tabular representation, not a conflicting original file. All five
archive files and the downloaded metadata record are byte- and checksum-locked
in the contract.

The audit uses `haven::read_dta(col_select=...)`. It never loads a full data
frame and deletes responses afterward. Its hard-forbidden fields are:

- `choice_cj`, the paired choice;
- `rating_cj`, the 1–10 profile rating; and
- `incon`, a derived choice/rating consistency indicator that would leak both
  responses.

Ambiguously timed and outcome-adjacent attitudes, knowledge items, and the
attention check are also outside the executable read allowlist. The audit does
not need their values to decide that they are unusable.

## Reconstructed sample and design

The design-only file contains:

- 8,500 respondents;
- 34,000 paired tasks;
- 68,000 profile rows;
- four tasks and eight rows for every respondent; and
- two adjacent source rows for every respondent-task.

France, Germany, and the United Kingdom each contribute 2,000 respondents;
the United States contributes 2,500. The released respondent weights are
positive, finite, constant over a respondent's eight rows, and sum to the
country sample size.

There is no archived side variable. The audit derives `profile_index = 1,2`
from immutable source-row order within respondent-task and treats it as a
neutral technical index. It is not called left/right or Agreement 1/2.

The six randomized factors are:

1. household cost: 0.5%, 1%, 1.5%, 2%, or 2.5% of GDP;
2. distribution of costs: rich countries only, proportional to current
   emissions, proportional to historical emissions, or rich countries paying
   more than poor countries;
3. participating countries: 20, 80, or 160 of 192;
4. emissions **coverage**: 40%, 60%, or 80% of current global emissions;
5. sanctions: none, low, medium, or high; and
6. monitoring: own government, independent commission, United Nations, or
   Greenpeace.

The cost and sanction amounts were displayed in local currency. The archived
Stata labels retain only one country's amounts, so the contract uses the
country-invariant policy tiers documented in the article rather than treating
those legacy labels as universal.

The full profile product has `5 x 4 x 3 x 3 x 4 x 4 = 2,880` cells. Every cell
occurs, with counts from 9 to 44. All 15 two-factor Cartesian products are
complete overall and separately in every country. The source says fully
randomized but gives neither a generator nor an explicit probability table, so
the contract prospectively reconstructs a uniform independent within-profile
law rather than calling it source-certified. One-way checks by country, task,
and neutral profile index are compatible with this reconstruction; the maximum
absolute Pearson residual is small. A pooled 2,880-cell Pearson test is also
compatible. These are diagnostics, not proof of the generator.

The dimension-row order was randomized once for each respondent and held fixed
over four tasks. The archive contains 696 opaque `cj_order` codes but no map to
the underlying permutations. The primary analysis averages over this
randomized display context rather than fitting 696 sparse nuisance indicators.

## Target and covariates

The primary target is:

- an equal one-quarter mixture of the four countries;
- released survey weights normalized within country;
- the uniform product distribution over 2,880 profiles; and
- equal averaging over the four tasks and randomized display-order context.

The effective respondent sample sizes under the released weights are 1,997.9
for France, 1,977.1 for Germany, 1,843.3 for the United Kingdom, and 2,307.7 for
the United States.

Six primary learner features are locked:

- country;
- age;
- female;
- the education variable belonging to the respondent's country;
- country-prefixed household-income category; and
- high versus low support for global climate cooperation, which the article
  documents as preceding the conjoint.

There is one missing education value, 301 missing raw-income values, and one
missing pre-conjoint support value. Missing categories and any numeric
imputation/scaling are learned without outcomes and, where applicable, inside
outer training folds. The complete derived `inc_high_group` and `educ_high`
variables are not primary because missing raw values were silently placed in
their low categories.

Ideology, reciprocity, willingness to pay, environmentalism, knowledge items,
and the attention check are excluded from primary learners because their exact
timing is unresolved or they are outcome-adjacent. The primary sample does not
condition on the post-survey attention indicator.

## Locked interaction families

The uniform target gives complete, identified interaction spaces:

| Whole family | Pair cells | Dimension | Minimum country-cell count |
|---|---:|---:|---:|
| Household cost x burden distribution | 20 | 12 | 752 |
| Sanctions x monitoring | 16 | 9 | 938 |
| Participating countries x emissions coverage | 9 | 4 | 1,721 |

All 25 coordinates are retained. No coordinate may be dropped after response
inspection. The audit emits deterministic uniform-target bases, direct
contrast weights, and function-space projectors. Tests verify orthonormality,
annihilation of intercept and main effects, projector idempotence, baseline
changes, random coordinate rotations, and level permutations below numerical
tolerance.

Five outer folds and independent five-valued inner labels are fixed at the
respondent level and balanced within country. Every one of a respondent's
eight rows remains in the same fold.

## What is and is not eligible

This study counts once toward the four-study portfolio, through its rating
analysis under both mean restrictions only. It does not count once per
outcome. The paired-choice response is excluded and cannot inherit the rating
decision.

Paired choice becomes eligible only if either:

1. the original randomizer/generator is recovered; or
2. the authors confirm the no-identical-profile rule and exact conditional
   opponent probabilities.

A prospective uniform-ordered-distinct law,
`g(a,b)=1/[2880 x 2879]` for `a != b`, is recorded only as a possible
sensitivity assumption. The realized absence of duplicates does not prove
that exact law.

The operational unblocker is therefore: **obtain the original
randomizer/generator or author confirmation of the no-identical-profile rule
and exact conditional opponent probabilities; otherwise paired choice stays
out.**

## Gate conclusion

The audit passes for a narrow rating analysis and fails for forced choice. No
response value or learner result was used to reach either conclusion. With
asylum seekers already eligible for both outcomes, this decision initially
made climate the second eligible fresh family. The subsequent income-tax audit
raised the count to three. The later narrow Graham-Svolik democracy-contest
contract supplies the fourth eligible study and the required nonproduct case.
Portfolio Gate B is now passed; the climate endpoint decision itself is
unchanged.
