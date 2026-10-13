# Candidate-recruitment application

This directory defines the locked candidate-recruitment application of
order-adaptive ridge. In the current cross-study ML benchmark it is a
**developmental application**, not independent confirmatory evidence, because
it helped determine estimands, learners, and simulation calibration. It
separates three operations that
should not be conflated:

1. `estimands.yml` locks the data source, factor levels, target population,
   folds, comparison models, and four interaction families before fitting.
2. `prepare_candidate_data.R` detects and validates the published cleaned data,
   creates respondent-level folds, and constructs the locked contrast matrices.
3. `run_candidate_analysis.R` evaluates six models out of sample, fits the final
   response surfaces, ranks profiles with the monotone model, and performs
   cross-fitted inference for the locked families.

No family, coordinate, cell, sign, or coefficient is selected after looking at
the outcome.

## Locked data contract

The only permitted source is the published replication file
`Candidate recruitment/replication/PSRM_replication/data/CleanedData.csv`
(MD5 `02069b7dac945e6ace2a7d236768bb4f`). The scripts never read the raw YouGov
directory and the preparation script intentionally has no `--input` option.

The public source is Eggers, Fowler, Howell, and Offer-Westort (2025),
“Opportunities to govern: how to increase the supply of moderate and qualified
candidates,” *Political Science Research and Methods*,
[article DOI 10.1017/psrm.2025.10076](https://doi.org/10.1017/psrm.2025.10076).
Its replication deposit is
[Harvard Dataverse DOI 10.7910/DVN/BQAHGZ](https://doi.org/10.7910/DVN/BQAHGZ),
version 1. The Dataverse record declares CC0 1.0 and supplies one unrestricted
`PSRM.zip` file (65,168,930 bytes; MD5
`0fb95eb8c2f57bb363c956a7eff8feb6`). The local archive has that exact size and
checksum. The article itself is CC BY 4.0; that article license and the data
record's CC0 declaration should not be conflated.

`Candidate recruitment/` is intentionally gitignored, so a fresh clone needs a
manual, provenance-preserving data step:

1. Download `PSRM.zip` version 1 from the Dataverse DOI and verify its byte size
   and MD5 above.
2. Extract it. The public archive expands as `PSRM/` and includes
   `data/UCHI0012_OUTPUT.sav`, not the generated `CleanedData.csv`.
3. In Stata 16 or later, run the authors' `master.do` (or its cleaning stage,
   `cleaning.do`) from the extracted replication directory. The latter exports
   `data/CleanedData.csv`.
4. Place/rename the extracted directory as
   `Candidate recruitment/replication/PSRM_replication/`. Confirm that the
   generated CSV has MD5 `02069b7dac945e6ace2a7d236768bb4f` before running
   this application.

The public archive contains third-party survey inputs without separate terms
inside the zip. This project cites the Dataverse record's CC0 declaration and
does not make a broader claim about undocumented upstream rights.

Preparation must verify all of the following before producing an analysis
bundle:

- 24,000 rows from 3,000 respondents, with eight task positions per respondent;
- a unique respondent-task row and no missing core values;
- finite positive survey weights that are invariant across all eight rows for a
  respondent, and an outcome in `[0, 1]`;
- exactly seven randomized attributes with their locked levels; and
- all 1,440 profiles in the complete `5 x 4 x 3 x 3 x 2 x 2 x 2` grid.

The source columns are detected with case- and punctuation-insensitive aliases,
but ambiguous matches fail. Numeric treatment levels must agree with the lock
within `1e-6`. Human-readable labels are locked from the replication cleaning
rules and published survey wording; the detected numeric values and labels are
written to `candidate_level_lock.csv` for review.

The rowwise assignment model is not inferred from realized cell frequencies. The
published design and preregistration state that the seven attribute levels were
independently randomized and that each level was presented with equal
probability within a profile; the replication randomization-inference code
separately redraws supports of size `5, 4, 3, 3, 2, 2, 2`. This implies the
locked rowwise marginal profile probability `1/1440`. There is no production
survey-platform randomizer export
in the deposit, so `estimands.yml` records and explicitly acknowledges that
remaining audit limitation. Preparation and fitting both fail if that
acknowledgment or the assignment contract is removed or changed.

A later joint-assignment audit found that none of the 3,000 respondents saw a
duplicate complete profile among their eight tasks. Zero repeats is essentially
incompatible with eight independent uniform draws from 1,440 profiles, so the
available evidence should not be described as establishing independence across
tasks. The analysis uses `1/1440` only as the marginal probability for a row's
profile and clusters arbitrary within-respondent dependence. The original
platform logs would be required to identify the joint task mechanism. The
candidate-geometry benchmark consequently samples eight profiles without
replacement within respondent and treats iid-within-respondent assignment only
as a possible sensitivity analysis.

`estimands.yml` version 4 makes this terminology correction without changing
the rowwise score, target, or respondent-clustered covariance. Prepared bundles
and analysis manifests created under version 3 are stale by checksum and must
be rebuilt before this developmental application is reused; no existing output
is silently relabeled.

## Prespecified estimands

The target distribution is uniform over all 1,440 profiles. Four complete
two-factor interaction families are locked:

| Family | Attributes | Full-rank dimension |
|---|---|---:|
| Authority x legislative threshold | `authority`, `majority` | 1 |
| Authority x staff support | `authority`, `staff` | 1 |
| Fundraising burden x probability of winning | `fundraising`, `winning` | 12 |
| Fundraising burden x salary | `fundraising`, `pay` | 8 |

For levels `(a, b)`, the interpretable functional-ANOVA cell surface is

```text
E_other[m(a,b,other)]
- E_{B,other}[m(a,B,other)]
- E_{A,other}[m(A,b,other)]
+ E_all[m].
```

The preparation script constructs these as direct weights over target-profile
means. It retains all 43 redundant cell labels for interpretation, then reduces
each family to a deterministic orthonormal basis for its complete interaction
subspace. The resulting 22 full-rank coordinates are the quantities passed to
`estimate_amie()` for covariance estimation, simultaneous intervals, and family
Wald tests. Cell surfaces are reconstructed linearly from those coordinates.

For each binary-by-binary family, the output also reports the classic DID on its
usual scale:

```text
mu11 - mu10 - mu01 + mu00
```

where each cell mean is averaged uniformly over the other five attributes. This
is the DID itself, not DID divided by four.

## Run

From the repository root:

```sh
Rscript analysis/candidate/prepare_candidate_data.R
Rscript analysis/candidate/run_candidate_analysis.R
```

Preparation writes to `analysis/candidate/derived/` and fitting writes to
`analysis/candidate/results/` by default. For an isolated or temporary run:

```sh
Rscript analysis/candidate/prepare_candidate_data.R \
  --output-dir /tmp/candidate-prepared

Rscript analysis/candidate/run_candidate_analysis.R \
  --prepared /tmp/candidate-prepared/candidate_prepared.rds \
  --output-dir /tmp/candidate-results
```

To verify only the data lock and planned package API before a long fit:

```sh
Rscript analysis/candidate/run_candidate_analysis.R \
  --prepared /tmp/candidate-prepared/candidate_prepared.rds \
  --api-check-only
```

The fitting script loads and checks only the package source in this repository;
it never falls back to functions in the global environment or to a stale
installed package. It checks for `factorial_basis()`, `fit_order_ridge()`,
`predict()`, `estimate_amie()`, and `rank_profiles()` before fitting. If the
local source or `pkgload` is unavailable, it stops. A deterministic checksum of
`DESCRIPTION`, `NAMESPACE`, and every `R/*.R` source file is recorded in the
analysis manifest and `package_source_checksums.csv`.

## Evaluation design

The same deterministic five respondent-level folds are used throughout, so all
eight tasks from a respondent stay together. The held-out comparison includes:

- additive weighted least squares;
- all-two-way weighted least squares;
- saturated profile-plus-task-position weighted least squares;
- ordinary ridge, represented as a common penalty across factorial orders;
- unrestricted order-specific ridge; and
- monotone order-specific ridge.

All three ridge models use `fit_order_ridge()` with the same target-weighted
factorial basis, survey-weighted squared-error loss, unpenalized task-position
matrix, and respondent-level inner fold assignment within each outer fold. They
also use the same locked tuning controls: a 15-point common-penalty grid,
80 optimizer iterations, `1e-5` relative tolerance, and log-penalty bounds of
`[-10, 8]`. Each structure selects the minimum-CV solution; there is no one-SE
rule or estimator-specific fold advantage. The saved inner-fold assignments and
complete tuning histories make these equality conditions auditable.

The saturated benchmark is fit as profile and task-position fixed effects. A
small Schur-complement system avoids materializing a dense
`24,000 x 1,448` indicator matrix. If a target profile is absent from an outer
training fold, its saturated held-out prediction is explicitly `NA`, never a
zero or an imputed grand mean. Metrics therefore report both observation and
survey-weight coverage, and `saturated_training_support.csv` records every such
case by fold.

Task-position indicators are unpenalized. Row-level RMSE, MAE, R-squared, and
calibration use survey weights. Profile-level metrics first aggregate held-out
rows with survey weights and then weight the 1,440 target profiles uniformly.
Every model-by-fold table is then expanded to all 1,440 target profiles. A
profile with no held-out rows remains an explicit missing row. Primary
full-target metrics are `NA` unless all 1,440 profiles are jointly supported by
all six models; explicitly named `common_support_*` columns report conditional
metrics and `target_profile_coverage` reports their target mass. Nothing is
silently dropped or renormalized under a full-target label. Pooled out-of-fold
metrics use the same check. This makes the empirical-sample prediction question
distinct from the locked uniform-profile target question.

Final profile predictions average the nuisance task-position contribution over
the eight task positions. Only the monotone order-ridge fit produces a ranking;
it is therefore based on the fitted randomized-attribute surface, not on an
arbitrary task position.

Comparison tuning and AMIE nuisance tuning are deliberately separate. The
former shares one respondent-level inner fold assignment across the three ridge
structures in each comparison fit. `estimate_amie()` creates its own
respondent-level inner folds within each outer-training sample, using the locked
inference cross-fit seed. The analysis exports those AMIE outer and inner folds,
tuning evaluations, and diagnostics separately so they cannot be mistaken for
the comparison folds.

The package is asked for pointwise AMIE intervals only. The application then
uses the single locked `simultaneous_seed` to construct one max-|t| band across
all 22 prespecified coordinates, family-coordinate bands, and interpretable
cell-surface bands. Fixed family and cell seed offsets are themselves locked in
`estimands.yml`; there is no second package-generated “global” band with an
untracked seed.

## Checkpoints, targets, and fast checks

Each completed outer comparison fold is saved under `results/checkpoints/`.
Resume is automatic only when the estimand lock, cleaned input, local package
checksum, application-script checksums, seed, and model set all match. A stale
checkpoint causes a hard failure rather than being mixed into a new analysis.
Final full-data fitting and AMIE inference remain distinct post-comparison
stages; if interrupted there, retain the outer checkpoints and restart the
script. No expensive fit is run by the contract tests.

The root targets workflow is opt-in through `CANDIDATE_STAGE`:

```sh
CANDIDATE_STAGE=check Rscript -e \
  'targets::tar_make(names = "candidate_application_checkpoint")'

CANDIDATE_STAGE=full Rscript -e \
  'targets::tar_make(names = "candidate_analysis_outputs")'
```

`off` is the default. `check` runs fast application tests, preparation, and the
API/contract checkpoint. `full` launches fitting only when the complete
production methods gate in `gate_status` has passed; a fast, pilot, incomplete,
or failed gate stops before outcome fitting. The standalone fast test command
is `Rscript analysis/candidate/tests/testthat.R`.

## Outputs

Preparation writes an RDS analysis bundle plus auditable CSV files for the
column map, exact level lock, validations, respondent folds, target grid,
profile counts, coordinate and cell metadata, direct contrast weights, cell
reconstruction maps, and classic DID weights.

Fitting writes tidy CSV files for:

- held-out row and profile predictions and their metrics;
- final predictions and the monotone-order-ridge profile ranking;
- penalties and convergence diagnostics for all three ridge structures;
- shared respondent-level inner folds and complete ridge tuning evaluations;
- saturated-model training support and explicit missing-cell status;
- separate AMIE outer/inner folds, nuisance-fit diagnostics, and tuning history;
- full-rank AMIE coordinates with pointwise and simultaneous intervals;
- reconstructed functional-ANOVA cells with simultaneous intervals;
- classic binary DIDs and four family-level Wald tests; and
- respondent-level cross-fitted scores when exposed by the package API.

The fitted comparison models and AMIE object are saved as RDS files. Outer-fold
checkpoints support safe resumption. Both stages write checksum/seed manifests,
and the fitting stage records `sessionInfo()`.

## Interpretation boundary

Random assignment supports causal contrasts of candidate-profile attributes for
the study population under the stated target weighting and assignment model.
The application does not treat respondent characteristics as randomized, does
not search respondent moderators, and does not turn profile rankings into claims
about which real-world candidates will run or win. Survey-weighted generalization
and the uniform-profile estimand are separate choices and should be described as
such in the paper.
