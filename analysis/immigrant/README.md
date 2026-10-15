# Immigrant-admission outcome-blind design audit

This directory freezes the design contract for Hainmueller and Hopkins's
immigrant-admission conjoint before any new outcome analysis. It is a Gate B
audit for the negative comparative paper, not an application result.

The audit deliberately reads only respondent/task identifiers, survey weights,
the nine randomized attributes, and a locked set of pre-conjoint variables.
It never imports `Chosen_Immigrant`, `Support_Admission`, or any other outcome,
and it fits no model.

## Decision

The study is **amber**, not yet a full Gate B pass.

- The design-complete universe is 1,407 respondents, 7,035 paired tasks, and
  14,070 profile rows. The previously recorded 1,396-respondent sample was
  outcome-complete and is not an outcome-blind enrollment rule.
- The released support contains 936,960 admissible profiles. Its conditional-
  uniform assignment law is strongly supported by the data and author code,
  but the archive does not contain the original profile generator.
- The paired-choice target additionally needs a within-pair opponent law.
  Independent `q x q` draws are a plausible prospective assumption, not a
  generator-certified fact in this archive.
- The article says persecution was paired only with Iraq, Sudan, and Somalia.
  The released data contain 500 China-persecution profiles and the released
  author code explicitly includes China. This changes the support from 23 to
  24 origin-reason cells and the full profile count from 897,920 to 936,960.
- A minor non-gating typo says job experience has five values in the article's
  prose; its table and the released data both show the four levels locked here.
- Education x profession and language x education pass identification. Origin
  x reason is identified on the released support, but remains on hold until the
  article/archive discrepancy is resolved prospectively.

The machine-readable lock is
[`config/datasets/immigrant-admission.yml`](../../config/datasets/immigrant-admission.yml).

## Run

From the repository root:

```sh
Rscript analysis/immigrant/audit_design.R
Rscript analysis/immigrant/tests/testthat.R
```

The default output directory is `analysis/immigrant/derived/`, which is
gitignored. To run in an isolated directory:

```sh
Rscript analysis/immigrant/audit_design.R \
  --output-dir /tmp/immigrant-design-audit
```

The output bundle includes archive checksums, the exact level lock, support
cells, assignment diagnostics, the four profile-probability strata, covariate
missingness and encodings, family ranks, invariant family projectors, direct
target-weighted contrast coordinates, a validation table, and a checksummed
manifest.

## Locked target and interaction geometry

The proposed primary target is the released row-marginal assignment law
`mu = q`, not a uniform distribution on either the full Cartesian grid or the
admissible cells. The published design documents randomization within and
across comparisons, but does not by itself certify independent `q x q` profile
draws within a pair; that must be confirmed or adopted as an explicit
prospective assumption.
Survey weights separately define the respondent-population target. All
inference and cross-fitting would cluster by `CaseID` and allow arbitrary
dependence among a respondent's five tasks.

For each supported pair, the audit constructs the complete cell-function
subspace orthogonal under `mu` to the intercept and both main-effect spaces.
The dimensions are:

| Family | Allowed cells | Interaction dimensions | Status |
|---|---:|---:|---|
| Education x profession | 61 | 44 | Pass, conditional on assignment-law resolution |
| Origin x application reason | 24 | 12 | Hold because of article/archive conflict |
| Language x education | 28 | 18 | Pass, conditional on assignment-law resolution |

These are support-aware function contrasts, not coefficients from a chosen
dummy coding. The family projector is the invariant object. Coordinates are a
deterministic representation obtained by scanning the locked cell order; their
labels should not be treated as intrinsically substantive. The audit rebuilds
the additive space with unrelated contrast codings and verifies that both the
projector and canonical representation agree below `1e-10`. Direct contrast
weights sum to zero, annihilate the additive space, and have an identity
target-weighted Gram matrix to numerical precision.

## What would turn amber into green

Either obtain the original generator or author confirmation, or make an
explicit prospective decision that the released rows and released author code
are authoritative and that profiles within a pair are integrated under `q x q`.
Under the latter route, disclose the published discrepancy, use the 24-cell
deployed support, retain `mu = q`, and label within-pair independence as an
uncertified identifying assumption. No outcome may be read before that choice
is recorded.
