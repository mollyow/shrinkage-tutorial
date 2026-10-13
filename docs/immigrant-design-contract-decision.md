# Immigrant-admission design-contract decision

**Gate:** B, outcome-blind application enrollment  
**Decision date:** 2026-09-01  
**Decision:** **AMBER — technically usable, but do not open outcomes yet**

## Bottom line

The Hainmueller-Hopkins immigrant-admission conjoint is a strong restricted-
support case for the negative comparative paper. The released design supports
three complete, coding-invariant interaction families with 74 released-support
coordinates, of which the 12 origin-reason coordinates remain on hold. The
sample structure, profile levels, survey weights, restrictions,
and pre-conjoint covariates can all be frozen without looking at outcome values.

It is not yet a green Gate B study because two parts of the assignment law are
uncertified. First, the published article and released archive disagree about
one structural restriction. The article says that persecution was available
only for Iraq, Sudan, and Somalia. The released data contain 500
China-persecution profiles, and the released estimation code explicitly
averages persecution across China, Sudan, Somalia, and Iraq. Second, published
randomization language does not by itself establish the exact within-pair
opponent law needed for paired-choice estimands. The archive has no original
randomizer with which to settle either point.

The audit therefore stops before outcomes. It can move to green by either:

1. obtaining the generator or author confirmation; or
2. prospectively declaring the released rows and released author code to be the
   authoritative deployed design, disclosing that this differs from the
   article's description, and adopting a `q x q` within-pair opponent law as an
   explicit uncertified assumption.

The second route is defensible and is the recommended fallback. It must be
recorded before any new outcome, effect, or learner-performance result is read.

## What was audited

The source is Hainmueller and Hopkins, “The Hidden American Immigration
Consensus” ([article DOI](https://doi.org/10.1111/ajps.12138)), using version
2.1 of the [Harvard Dataverse archive](https://doi.org/10.7910/DVN/25505),
which declares CC0 1.0. The local data and two code files exactly match the
archive checksums. The archive contains only the data, estimation code, and
plotting code; it contains neither the survey-platform generator nor a complete
questionnaire. Wave-1 question wording is documented in the
[online appendix](https://www.mit.edu/~jhainm/Paper/webappendix_2012_08_30DJH.pdf).

The executable audit uses a column-selection barrier inside `haven::read_dta()`.
It loads only identifiers, weights, randomized attributes, and locked
pre-conjoint variables. It does not load `Chosen_Immigrant`,
`Support_Admission`, or any other outcome, and it fits no model.

## Design universe

The archive has 1,714 respondents represented by ten rows each. For 307
respondents all nine profile attributes are missing. The remaining design-
complete universe is exactly:

- 1,407 respondents;
- five paired tasks per respondent;
- 7,035 respondent-tasks; and
- 14,070 profile rows.

Every respondent-task contains two contiguous source rows. There is no released
profile-side variable, so the contract derives a neutral `profile_index` of 1
or 2 from immutable source-row order. It must not be interpreted as left versus
right. Survey weights are positive, finite, and constant across all ten rows
for each respondent.

This corrects the earlier registry entry. The 1,396-respondent/13,960-row figure
conditions on outcome completeness and cannot define outcome-blind enrollment.
Outcome-specific missing-pair handling will be frozen only after this gate is
green; it will not alter the 1,407-respondent design universe.

## Reconstructed support and assignment law

There are nine randomized attributes. Two pairs have structural zeros.

One smaller documentation mismatch is resolved rather than gating: the article
prose calls job experience a five-value attribute, while its Table 1 and the
released labeled data agree on four values. The contract follows the table and
data.

For education $E$ and profession $J$, education levels 1–4 can occur with
the seven professions $\{1,2,3,4,6,7,9\}$, while education levels 5–7 can
occur with all eleven professions. This gives 61 of 77 cells.

For origin $O$ and application reason $R$, the released design has reasons
1–2 for origins 1–6 and reasons 1–3 for origins 7–10. This gives 24 of 30 cells.
The released [author code](<../Candidate recruitment/data/hainmueller_hopkins/rep_estimates.do>)
states the same four-origin rule. The article instead implies 23 cells by
excluding China-persecution.

The released data closely match the following conditional-uniform row law:

\[
q(a)=
\frac{1}{7J(e)\,2\,10R(o)\,4\,4\,5\,4},
\]

where $J(e)=7$ for education 1–4 and 11 for education 5–7, and
$R(o)=2$ for origins 1–6 and 3 for origins 7–10. The four profile-probability
strata are therefore:

| Education | Origin | Profile probability |
|---|---|---:|
| Levels 1–4 | Origins 1–6 | $1/627{,}200$ |
| Levels 5–7 | Origins 1–6 | $1/985{,}600$ |
| Levels 1–4 | Origins 7–10 | $1/940{,}800$ |
| Levels 5–7 | Origins 7–10 | $1/1{,}478{,}400$ |

The released rule yields
$61\times24\times2\times4\times4\times5\times4=936{,}960$ admissible
profiles. The article's three-origin rule yields 897,920.

Realized frequencies cannot prove a randomization law. The contract labels this
law “strongly supported, not generator-certified.” The published design
documents randomization within and across comparisons, but that does not by
itself certify independent `q x q` draws of the two profiles. The opponent law
for the paired-choice target must therefore be confirmed or adopted as an
explicit prospective assumption. The contract makes no iid claim across a
respondent's tasks. Future folds and covariance must be clustered by `CaseID`
and allow arbitrary across-task dependence. `contest_no` is a task-position
control; randomized attribute display order is integrated out because it was
fixed within respondent but is not released.

## Target

The proposed primary target is the deployed row-marginal assignment law,
$\mu=q$, over the 936,960 released-support profiles. This choice:

- never extrapolates to a structural zero;
- avoids profile importance weighting in the Riesz score; and
- makes the target-weighted interaction geometry correspond to the actual
  distribution of randomized profiles.

Survey weights separately target the respondent population. Tasks and neutral
profile sides are averaged uniformly. For the paired-choice outcome, the other
profile would be integrated over $q$ under the prospective `q x q` assumption;
this is not represented as generator-certified evidence.

A prespecified sensitivity target is uniform over the same 936,960 admissible
profiles. Its largest $\mu/q$ ratio is only about 1.578. Uniform weighting over
the unrestricted Cartesian grid is prohibited because it assigns mass to
profiles that could not be fielded.

## Interaction-family identification

For a factor pair $U,V$, let $\Omega_{UV}$ be its allowed cells, let $p$
be their target probabilities, and let $D_p=\operatorname{diag}(p)$. Let
$A$ span the intercept and both main-effect spaces on that support. The audit
constructs a complete basis $H$ satisfying

\[
H^\mathsf{T}D_pH=I,
\qquad
A^\mathsf{T}D_pH=0.
\]

For pair-cell means $g$, the coordinates are

\[
\theta=H^\mathsf{T}D_pg=C^\mathsf{T}g,
\qquad C=D_pH.
\]

Thus the estimand is the entire supported interaction increment—every
pair-cell contrast that annihilates additive functions—not a chosen collection
of reference-coded coefficients. The generated bases pass weighted Gram,
additive-orthogonality, zero-sum, and rank checks below $10^{-10}$.
The invariant estimand is the family subspace and its weighted projector.
Individual orthonormal coordinates are a locked computational representation,
constructed deterministically from that projector and the canonical cell order;
they are not given standalone substantive labels. The audit rebuilds the
additive space with Helmert and sum contrasts and verifies both projector and
canonical-basis agreement below $10^{-10}$.

| Family | Cartesian cells | Allowed cells | Additive rank | Supported interaction rank | Decision |
|---|---:|---:|---:|---:|---|
| Education x profession | 77 | 61 | 17 | **44** | Pass, conditional on assignment-law resolution |
| Origin x application reason | 30 | 24 | 12 | **12** | Hold because its support is disputed |
| Language x education | 28 | 28 | 10 | **18** | Pass, conditional on assignment-law resolution |

Every allowed pair cell occurs in the released sample. Minimum counts are 154,
432, and 450, respectively. Language x education is a full 4-by-7 family, but
its target marginal mean averages profession under the distribution permitted
at each education level; it must not be described as holding profession to one
common distribution.

## Covariate lock

The primary nuisance-feature allowlist is age, education, race/ethnicity,
gender, household income, state, panel tenure, the Wave-1 immigration-attitude
item, and two Wave-1 indices: self-monitoring and ethnocentrism. The article
states that these Wave-1 constructs preceded a roughly three-week washout and
the Wave-2 conjoint.

All source covariates are constant within respondent. The seven stable panel
variables have no missing values. The strict three-item self-monitoring index
is missing for 11 respondents, and the Wave-1 immigration attitude is missing
for 13. The author's Stata `egen rowmean` ethnocentrism recipe is missing for
251 respondents; a stricter version requiring all three outgroup thermometers
is missing for 271. The latter distinction is now explicit rather than silently
conflated.

All preprocessing must occur inside each outer training fold, with missingness
indicators and fold-local imputation. Complete-case deletion is prohibited.
Age, panel tenure, self-monitoring, and ethnocentrism are numeric and are
fold-standardized; missing numeric values use a training-fold median plus an
indicator. Education, income, and the Wave-1 attitude are ordered variables but
are treated categorically in the primary benchmark. Race/ethnicity, gender,
and state are unordered. All categorical variables use a symmetric full
one-hot expansion over locked levels, including an explicit missing level and
without outcome-based target encoding or level pooling. An ordinal numeric
encoding may be a labeled sensitivity analysis, not an adaptive primary choice.
The level dictionary is the set of nonmissing codes in the outcome-blind
1,407-respondent design universe, matched to the checksum-locked DTA labels,
plus `__MISSING__`; every fold retains the same columns even if a training fold
lacks a rare level.
Party identification and ideology remain secondary because their exact survey
timing is undocumented. `W2_Q16` is excluded as post-exposure. Wave-2
occupation/employment items are excluded from the primary learner. Survey
weights are weights, not features; respondent IDs are cluster keys, not
features.

## Gate decision and next action

This audit passes sample reconstruction, support rank, target definition,
covariate timing for the primary set, and coding-invariant family construction.
The final assignment certification remains open for two reasons: the article
and released implementation conflict about one support cell, and the exact
within-pair opponent law is not generator-certified.

Therefore:

- do not read outcomes yet;
- first seek the original generator or a short author clarification;
- if neither is available, record the released-implementation-authoritative
  fallback and the `q x q` within-pair assumption in a dated amendment;
- retain the 24-cell support and $\mu=q$, disclose the mismatch, and then turn
  this study green under that explicit assumption; and
- before fitting, implement the supported-subspace contrast constructor in the
  package. The current reference-cell `amie_contrast()` is not sufficient for
  structural zeros, and the registry's planned `diagnose_factorial_support()`
  still needs to be added.

## Reproduction

Run:

```sh
Rscript analysis/immigrant/audit_design.R
Rscript analysis/immigrant/tests/testthat.R
```

The machine-readable contract is
[`config/datasets/immigrant-admission.yml`](../config/datasets/immigrant-admission.yml),
the audit is
[`analysis/immigrant/audit_design.R`](../analysis/immigrant/audit_design.R),
and its generated aggregate outputs are written to the gitignored
`analysis/immigrant/derived/` directory.
