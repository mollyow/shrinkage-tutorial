# Graham–Svolik candidate-choice experiment: narrow Gate-B audit

This directory freezes a deliberately narrow, outcome-blind contract for the
Graham and Svolik candidate-choice experiment. The result is **green for three
coarsened democracy interaction families under mean no carryover**, and
**failed for a complete candidate-profile surface**.

The executable audit reads only an explicit allowlist of randomized design
fields from the original experiment CSV and a separate allowlist of pre-choice
respondent variables. It does not load a choice, win indicator, derived
response, learner result, or effect estimate.

The machine-readable contract is
[`config/datasets/graham-svolik.yml`](../../config/datasets/graham-svolik.yml).
The benchmark plan is [`docs/plan.md`](../../docs/plan.md).

## Run

From the repository root:

```sh
Rscript analysis/graham_svolik/audit_design.R
Rscript analysis/graham_svolik/tests/testthat.R
```

The script fails closed on a changed original-file checksum, an unexpected
generated file, any mismatch in the selected-column barrier, sample or block
counts, incomplete target support, rank loss, projector dependence on coding,
or respondent leakage across folds. It writes a derived audit bundle and a
checksum manifest to `analysis/graham_svolik/derived/`.

The derived bundle is built from Graham and Svolik's replication archive
(Harvard Dataverse, version 1.1, <https://doi.org/10.7910/DVN/EEARKA>), which
is released under CC BY 4.0. The article is Graham and Svolik, "Democracy in
America? Partisanship, Polarization, and the Robustness of Support for
Democracy in the United States," *American Political Science Review*
(<https://doi.org/10.1017/S0003055420000052>).

## Narrow design

The original experiment CSV contains 1,682 complete respondent IDs, 13 tasks
per respondent, and two mirrored candidate rows per task. The audit uses the
archive's `candNum == 1` orientation. In an undemocratic-versus-generic task,
this makes the undemocratic candidate focal; in a generic-versus-generic task,
it uses the source-designated generic focal candidate. The opponent is generic
in every task retained by the narrow analysis.

Two randomized negative-valence tasks are removed using only the democracy
treatment type. This leaves 18,502 tasks: exactly 11 per respondent. Every
respondent contributes four pooled-generic tasks and one task for each of seven
named undemocratic behaviors.

The coarsened current-task assignment law is

```text
democracy:       generic 4/11; each named undemocratic behavior 1/11
focal party:     Democrat 1/2; Republican 1/2
contest relation: same party 6/13; cross party 7/13
focal gender:    female 3/8; male 5/8
```

The source describes the underlying attribute blocks as independently
randomized, so the locked current-task probability is the product of those
four components. The primary target is uniform over all
`8 × 2 × 2 × 2 = 64` supported cells. Policy positions, opponent traits,
left/right placement, scenario number, and other randomized current-task
fields are standardized over their outcome-blind empirical distribution.

Assignments are blocked without replacement across a respondent's tasks, and
the actual presentation sequence was not released. Identification therefore
uses the prospectively declared mean no-carryover restriction: conditional on
baseline variables, both current candidate profiles, and current task context,
the mean current choice is unaffected by earlier displayed profiles. All
splitting and covariance calculations remain respondent-clustered.

## Locked families

| Whole family | Cells | Interaction dimension | Smallest observed cell |
|---|---:|---:|---:|
| Democracy behavior × focal party | 16 | 7 | 830 |
| Democracy behavior × same/cross-party contest | 16 | 7 | 750 |
| Democracy behavior × focal candidate gender | 16 | 7 | 606 |

The emitted contrast bases are target-orthonormal. The audit verifies that the
function-space projectors are unchanged by reference-baseline swaps, random
level permutations, and nonsingular random contrast rotations. These are
fixed whole families; no coordinate may be selected or discarded after the
endpoint is opened.

## Endpoint rule

The endpoint stays unopened until the common four-study learner, fold,
equivalence-margin, and reporting protocol is frozen. After that freeze, the
primary endpoint is `c_win` on the selected `candNum == 1` nonvalence row. A
task is retained for that endpoint only when `c_win` is observed. Its
missingness and the four-component treatment distribution must be reported
before any learner is fitted. `c_vote` is held out and is not silently promoted
to a second primary endpoint.

## Learner features and folds

The 28 locked pre-choice raw features are age, respondent gender, race,
education, household-income category, state-tax wording block, party ID,
ideology, Trump approval, sixteen policy-position ratings, ANES knowledge
total, authoritarianism total, and voting-duty orientation. Imputation,
missingness indicators, numeric scaling, and categorical encoding are learned
inside each outer training fold. The survey weight is used for the target
population and loss, never as a predictor.

The audit writes a deterministic, balanced five-fold assignment based only on
respondent ID. All 11 retained tasks for one respondent remain in one fold.

## Why the full-profile analysis is excluded

The complete candidate generator is not released, and two documented rules
conflict with the archive. The profession probabilities printed in the
supplement sum to 0.90 and omit `Police officer`, although that category occurs
in the release. The supplement states that experience is age minus a random
integer from 20 through 30; the released differences are 23 through 33 with an
undocumented endpoint pattern. Age, experience, and profession therefore do
not enter the causal-factor target. They and other randomized fields are only
audited and empirically standardized as current-task context.

## Counts and protocol disclosure

The supplement reports 1,692 Wave-2 completers, the article reports 1,691
respondents and 21,604 choices, and the experiment CSV contains 1,682 complete
IDs. The covariate CSV contains those 1,682 linkable IDs plus nine rows whose
experiment ID is missing. The enrolled cohort is every released complete
experimental ID; no response-based repair or exclusion is attempted.

During initial archive inspection, before this executable barrier existed, a
shell `head` command inadvertently displayed one full experiment row and its
four binary response fields. That row was quarantined and was not summarized,
used to select the study or estimands, supplied to a learner, or used in an
effect calculation. No additional response row was opened. The contract and
every audit manifest retain this disclosure; the project must not claim that
zero response values were ever visible historically.

Once this audit passes, the next action is to freeze the common cross-study
comparison protocol and all reporting code. Opening `c_win` before that freeze
would violate the contract.
