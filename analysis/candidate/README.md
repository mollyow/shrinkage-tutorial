# Candidate-recruitment application

Candidate recruitment (Eggers, Fowler, Howell, and Offer-Westort 2025) is the
benchmark's **developmental application**. It is shown in the paper but
excluded from every confirmatory summary, because it helped shape the
estimands, learners, and simulations. It also supplies the rating geometry for
the simulations. See [`docs/plan.md`](../../docs/plan.md).

| File | Role |
|---|---|
| `estimands.yml` | Data source, factor levels, target, folds, and the four interaction families |
| `build_comparison_context_v4.R` | Builds the outcome-free covariate context the rating simulations sample from (reads eight allowlisted columns, never the outcome) |
| `prepare_candidate_data.R` | Validates the cleaned data and writes folds and family contrast matrices to `derived/` |
| `contract_helpers.R` | Contract and survey-weight checks shared by the scripts and tests |
| `tests/` | Fast contract tests: `Rscript analysis/candidate/tests/testthat.R` |

```sh
Rscript analysis/candidate/build_comparison_context_v4.R
Rscript analysis/candidate/prepare_candidate_data.R
```

## Data

The only permitted source is
`Candidate recruitment/replication/PSRM_replication/data/CleanedData.csv`
(MD5 `02069b7dac945e6ace2a7d236768bb4f`). The scripts never read the raw survey
directory, and the preparation script has no `--input` option.

The replication deposit is
[Harvard Dataverse DOI 10.7910/DVN/BQAHGZ](https://doi.org/10.7910/DVN/BQAHGZ),
version 1, declared CC0 1.0; the article
([DOI 10.1017/psrm.2025.10076](https://doi.org/10.1017/psrm.2025.10076)) is
CC BY 4.0. `Candidate recruitment/` is gitignored, so a fresh clone needs:

1. Download `PSRM.zip` version 1 (65,168,930 bytes; MD5
   `0fb95eb8c2f57bb363c956a7eff8feb6`).
2. Extract it and run the authors' `master.do` (or `cleaning.do`) in Stata 16
   or later, which exports `data/CleanedData.csv`.
3. Place the extracted directory at
   `Candidate recruitment/replication/PSRM_replication/` and confirm the CSV's
   MD5 above.

Preparation fails unless it finds 24,000 rows from 3,000 respondents with eight
tasks each, no missing core values, finite positive survey weights constant
within respondent, an outcome in `[0, 1]`, the seven attributes at their locked
levels, and all 1,440 profiles of the `5 x 4 x 3 x 3 x 2 x 2 x 2` grid.

## Assignment

Each profile's attributes were randomized independently with equal
probability, so a row's marginal profile probability is `1/1440`. Independence
across a respondent's tasks is not established: none of the 3,000 respondents
saw a repeated profile among eight tasks, which is nearly impossible under
independent draws. The analysis therefore uses `1/1440` only as a rowwise
marginal and clusters by respondent, and the rating simulations sample eight
profiles without replacement within respondent.

## Interaction families

The target is uniform over all 1,440 profiles.

| Family | Attributes | Dimension |
|---|---|---:|
| Authority x legislative threshold | `authority`, `majority` | 1 |
| Authority x staff support | `authority`, `staff` | 1 |
| Fundraising burden x probability of winning | `fundraising`, `winning` | 12 |
| Fundraising burden x salary | `fundraising`, `pay` | 8 |

Each family is reduced to an orthonormal basis of its interaction subspace,
giving 22 coordinates in total. Respondent characteristics are not randomized,
so the application makes no claim about moderators.
