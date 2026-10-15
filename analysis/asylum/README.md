# Asylum-seeker conjoint: outcome-blind Gate-B audit

This directory freezes the design contract for the 2016 and 2022 waves of
Bansak, Hainmueller, and Hangartner's asylum-seeker conjoint before any new
learner comparison is run.

The audit is intentionally narrower than a replication of the article. It
reads identifiers, the nine randomized attributes, respondent design fields,
survey weights, and prospectively classified nuisance variables. It never
loads `pref`, `rate`, or `ratebin`, and it fits no outcome model.

## Decision

The study passes its Gate-B design audit **green**:

- the prospective plan documents unrestricted uniform randomization;
- both wave-specific Cartesian supports are complete;
- all five tasks and both explicit profile sides are reconstructible;
- the three registered whole interaction families have their full target-
  weighted ranks, in every country;
- primary nuisance features are limited to immutable demographics and three
  questions documented before the conjoint;
- the 2022 randomized frame is integrated at probability `1/4`; and
- respondent-level outer and inner fold labels are locked without outcomes.

The machine-readable contract is
[`config/datasets/asylum-seekers.yml`](../../config/datasets/asylum-seekers.yml).
The benchmark plan is [`docs/plan.md`](../../docs/plan.md).

## Run

From the repository root:

```sh
Rscript analysis/asylum/audit_design.R
Rscript analysis/asylum/tests/testthat.R
```

The audit uses `data.table::fread(select=...)` as a strict column read barrier.
It fails closed if a locked source checksum changes, if an unexpected generated
file is present, or if any sample, support, rank, fold, or invariance check
fails.

Raw archive files live under the gitignored `Candidate recruitment/` tree. The
generated `analysis/asylum/derived/` bundle is also gitignored; rerun the audit
to reproduce it locally.

## Locked design facts

| Wave | Raw respondents | Design-complete | Primary-weight eligible | Profile support |
|---|---:|---:|---:|---:|
| 2016 | 18,030 | 18,021 | 17,874 | 136,080 |
| 2022 | 14,976 | 14,966 | 14,846 | 170,100 |

Nine respondents in 2016 and ten in 2022 have all nine profile attributes
missing on all ten rows. There are no partially observed profile blocks. The
design-complete universe is therefore fixed without consulting an outcome.
The primary population analysis additionally requires a finite positive
released survey weight; the full design-complete sample is the unweighted
sensitivity.

The only support change is the 2022 `War` level for migration reason. Separate
waves are primary. A common-support rating comparison removes `War` from the
focal-profile target; a forced-choice comparison must remove it from both the
focal and opponent profile distributions.

## Registered whole families

| Family | 2016 dimension | 2022 dimension | Common-support dimension |
|---|---:|---:|---:|
| Migration reason × vulnerability | 12 | 16 | 12 |
| Religion × origin | 12 | 12 | 12 |
| Occupation × language | 10 | 10 | 10 |

The emitted bases are deterministic and orthonormal under the uniform target.
Tests cover random contrast rotations, level permutations, annihilation of
reference-coded main-effect spaces, and projector idempotence.

## Primary nuisance features

The primary learner may use country, gender, age, home-country birth,
education, and the three pre-conjoint questions about asylum/immigration
levels. In 2022 it must also condition on the randomized frame and integrate
over frames at exactly `1/4` each. Employment, income, ideology, feeling
thermometers, nationalism, cosmopolitanism, and EU/NATO/Russia measures remain
outside the primary learner because they were elicited after the conjoint or
are outcome-adjacent.

The released 2016 instrument is incomplete. Its item ordering is strongly
supported by the prospective plan's statement that the earlier questionnaire
was reused, but this is recorded as a provenance limitation rather than
silently treated as page-level verification.
