# Asylum-seeker conjoint design-contract decision

**Decision date:** 2026-09-01  
**Gate:** B, fresh-application eligibility  
**Result:** **GREEN**  
**Outcome status:** no `pref`, `rate`, or `ratebin` values were loaded or modeled

## Bottom line

The Bansak-Hainmueller-Hangartner asylum-seeker study is eligible as the first
fresh confirmatory study family in the prospective comparative paper. It has a
source-certified product assignment law, explicit pair sides, large samples in
15 countries, two outcomes reserved for later analysis, a usable primary
covariate tier, and three completely supported interaction families.

This is a design decision, not evidence that machine learning helps. No new
effect, predictive comparison, influence-score variance, confidence interval,
or learner result was examined.

Primary sources are the [Nature article](https://doi.org/10.1038/s41586-023-06417-6),
the [CC0 Dataverse V2.0 archive](https://doi.org/10.7910/DVN/FTL1MM), and the
[prospective 2022 plan](https://osf.io/jd8n3/) with its frozen
[questionnaire appendix](https://osf.io/zbhca). The machine-readable contract
is [`config/datasets/asylum-seekers.yml`](../config/datasets/asylum-seekers.yml),
and the executable audit is
[`analysis/asylum/audit_design.R`](../analysis/asylum/audit_design.R).

## Why the assignment gate passes

The prospective plan describes the nine attributes as using non-restricted
uniform randomization. The published article and supplement describe levels as
fully randomized within and between respondents. These sources are sufficient
to lock independent uniform profile draws even though the archive does not
contain a survey-platform generator export.

The wave-specific laws are therefore

\[
q_{2016}(a)=1/136{,}080,
\qquad
q_{2022}(a)=1/170{,}100,
\]

with ordered profile pairs drawn from \(q_w\times q_w\). There are no
structural restrictions. Every one of the 36 attribute-pair projections has
complete Cartesian support in both waves. Realized marginal standardized
residuals are all below 2.65; that is a discrepancy screen, not the basis for
the randomization claim.

The only design change is a fifth, `War`, value for migration reason in 2022.
It is not silently pooled with 2016.

The four 2022 opening frames—general, Mediterranean, Afghanistan, and
Ukraine—were randomized before the conjoint at probability `1/4`. Frame is a
design context, not an ordinary respondent characteristic. The primary 2022
nuisance fit conditions on it and the target integrates over the four frames
with exactly equal weight.

## Outcome-blind sample reconstruction

| Quantity | 2016 | 2022 |
|---|---:|---:|
| Archive respondents | 18,030 | 14,976 |
| Archive respondent-tasks | 90,150 | 74,880 |
| Archive profile rows | 180,300 | 149,760 |
| All-profile-attribute-missing respondents | 9 | 10 |
| Design-complete respondents | 18,021 | 14,966 |
| Design-complete tasks | 90,105 | 74,830 |
| Design-complete profiles | 180,210 | 149,660 |
| Primary-weight eligible respondents | 17,874 | 14,846 |
| Primary-weight eligible profiles | 178,740 | 148,460 |

Every archive respondent has five tasks and two profiles per task. `prof`
explicitly encodes task and side (`A1`, `A2`, …, `E1`, `E2`), and all keys are
unique. The nine and ten incomplete respondents have all nine attributes
missing on all ten rows; there are no partial profile blocks. This makes the
design-enrollment rule unambiguous and outcome-free.

The smaller profile counts used in the published weighted analysis are not the
design universe. They are exactly the number of design-complete respondents
with nonmissing primary weights times ten. We therefore freeze two samples:

1. the design-complete universe for the unweighted sensitivity; and
2. the finite-positive-primary-weight subset for the population-weighted
   primary analysis.

Respondent identifiers do not overlap across waves. The waves are repeated
cross-sections, not a respondent panel, and will never be clustered or linked
across years.

## Target distributions

For each wave, the profile target is its uniform randomized product law. The
population target gives each of the 15 countries weight `1/15`; within country,
the released age-gender-education entropy-balancing weights are normalized to
sum to `1/15`. Those weights were trimmed at six by the source authors. The
observed-country mixture, the full unweighted design universe, and the
ideology-calibrated alternative weight are sensitivities.

The primary analyses estimate waves separately: 2016 is the primary wave and
2022 is a nested temporal replication. For a common-support comparison:

- a rating estimand conditions the focal profile's reason on the four shared
  values; and
- a forced-choice estimand conditions both focal and opponent profiles on
  non-`War` reasons.

Deleting only focal `War` rows for forced choice would leave the 2022 opponent
distribution different from the 2016 choice environment.

## Covariate lock

The primary learner receives eight raw variables:

- country, gender, and age;
- home-country birth and harmonized education; and
- the three questions on desired asylum/immigration levels in the home country
  and Europe.

The frozen 2022 questionnaire places the last three before the five conjoint
pairs. Home-country birth and completed education are logically fixed even
though elicited later. The 2022 frame precedes these attitude questions, so
frame/design history must remain in the nuisance conditioning set.

Employment, income, ideology, feeling thermometers, cosmopolitanism,
nationalism, EU/NATO/Russia attitudes, empathy, political interest, and
knowledge are excluded from the primary learner because the questionnaire
places them after the conjoint or makes them outcome-adjacent. Employment and
income may appear only in a labeled timing sensitivity. Survey weights are
weights, never features; respondent IDs are cluster keys, never features.

The original 2016 questionnaire is not in the public archive. The prospective
plan says the 2022 questionnaire reused the 2015/2016 instrument apart from
listed changes, so the ordering of the three direct attitudes is strongly
supported rather than independently page-certified. This is a disclosed
provenance limitation. A stricter sensitivity drops those three attitudes in
2016 and uses only immutable demographics.

All preprocessing is learned within outer training folds. Categories use the
global outcome-blind level universe solely to keep columns aligned. Age uses a
foldwise median and IQR. Ages 16 and 17 are retained because voting eligibility
differs across countries; the single age above 110 is flagged and imputed
without removing the respondent.

## Locked whole interaction families

Each family is the saturated pair-cell function space projected orthogonally
away from its intercept and two main-effect spaces under the declared uniform
target.

| Family | 2016 rank | 2022 rank | Smallest country-cell count |
|---|---:|---:|---:|
| Migration reason × vulnerability | 12 | 16 | 334 |
| Religion × origin | 12 | 12 | 405 |
| Occupation × language | 10 | 10 | 481 |

The 2016 minimum counts for these families are 535, 513, and 598,
respectively. Thus every registered cell occurs in every country, not merely
in the pooled data.

The audit emits target-orthonormal bases and function-space projectors. Their
observed ranks are 34 coordinates in 2016 and 38 in 2022; the common-support
registry has 34. Maximum numerical errors for orthonormality, removal of main
effects, and projector idempotence are below `7e-16`. Tests also verify
invariance to reference coding, random contrast rotations, and level
permutations.

## Fold lock and leakage protection

Five outer folds and five global inner-fold labels are assigned once per
respondent using outcome-free hashes, stratified by wave, country, and the
2022 frame. Counts differ by at most one within every stratum. For each outer
training sample, all five inner labels remain represented in every stratum.
All ten rows for a respondent inherit the same outer and inner labels.

The same folds must be used for every estimator and both later outcomes. A
learner may not obtain a favorable new split.

## Limitations that do not reverse the decision

- The public archive lacks platform randomizer code and attribute-row display
  order. The prospective assignment law is explicit, and row display order is
  unnecessary for the registered targets.
- The 2016 original questionnaire is absent. This motivates the demographic-
  only timing sensitivity described above.
- The PAP contains a duplicated Afghanistan-frame wording under one Ukraine
  bullet, while the appended questionnaire and published supplement give the
  correct Ukraine text. This does not change the released frame assignment.
- The PAP spells `Handicapped` incorrectly in one place; the archive, analysis
  code, and supplement agree on the released label.
- Equal weighting of countries is a prospective scientific choice, not a
  population-size-weighted Europe estimand. Both interpretations will be
  labeled explicitly.

None of these issues creates a support ambiguity like the immigrant-admission
archive's origin-by-reason conflict.

## Gate consequence

This study now counts as **one** fresh eligible study family, not two studies
and not thirty country-wave datasets. Both the rating and paired-choice
outcomes remain unopened for the new benchmark. The full comparative-paper
data gate still requires three additional eligible study families, including a
restricted-support design.

The next outcome-blind audit at the time of this decision was the climate-
agreement conjoint. That audit is now complete with a
[split decision](climate-design-contract-decision.md): rating is conditionally
eligible, while paired choice remains amber. The current next audit is the
income-tax-plan design.
