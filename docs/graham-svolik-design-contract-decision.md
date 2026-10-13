# Graham-Svolik design-contract decision

**Decision date:** 2026-09-01  
**Decision:** Gate B green for a narrow, coarsened democracy-by-contest analysis  
**Not eligible:** a complete candidate-profile response surface

## Bottom line

The Graham-Svolik candidate experiment can fill the fourth fresh application
slot, but only after narrowing the estimand to the part of the randomization
that is documented exactly.

The eligible analysis uses all 1,682 respondents in the released experimental
frame and their 11 nonvalence tasks. It compares one generic focal candidate
with each of seven focal undemocratic behaviors, crossed in turn with focal
party, same- versus cross-party competition, and focal candidate gender. The
three complete interaction families each have seven degrees of freedom.

This is a useful nonproduct design. Candidate attributes are allocated in
respondent-level blocks, the democracy treatment constrains the opponent to be
generic, candidates in a pair share policy areas, and treatment schedules are
drawn without replacement across tasks. The design is therefore materially
different from the product-randomized applications already in the portfolio.

The pass does **not** extend to age, experience, profession, or the complete
candidate profile. The released generator is absent, and the documentation for
experience and profession conflicts with the released assignments.

## Sources and provenance

The audit uses the [APSR article](https://doi.org/10.1017/S0003055420000052),
the [official Harvard Dataverse archive](https://doi.org/10.7910/DVN/EEARKA),
and Appendix C.3 and Figure C.5 of the archived supplement.

The current Dataverse release is version 1.1, released on February 1, 2024,
under CC BY 4.0. Version 1.0 has the same 131 file signatures and citation
metadata; the recorded change is the license. The canonical experiment input
is the original uploaded CSV, not Dataverse's ingested tabular derivative:

| Input | Bytes | MD5 |
|---|---:|---|
| `data_experiment.csv` | 75,655,739 | `90f79baea84d1334c7c2da073ceacd67` |
| `data_covariates.csv` | 618,033 | `77c08ca5fa0782cc87ea87a4da232e71` |
| `weights.csv` | 92,369 | `044e136b1ca52e08122e9edf409d2930` |
| `final_appendix.pdf` | 6,306,444 | `4353be8391c771fba1a9518aba09fa62` |
| `codebook.xlsx` | 32,468 | `cad34a0d60d54d537f40fadf3c098f45` |

Dataverse reports the checksum of the original CSV while its default file API
returns an ingested `.tab` file with a different byte stream. Reproduction must
request `?format=original` and verify both original byte size and MD5.

No QSF, JavaScript, or original assignment generator is released. The narrow
pass instead relies on the prospective randomization description and exact
block identities reproduced in the released design columns.

## Released design universe

The original experiment CSV contains exactly:

- 1,682 respondent IDs;
- 13 retained main-design tasks per respondent;
- two mirrored candidate rows per task; and
- 43,732 profile rows in total.

`(id, matchNum, candNum)` is unique. Within each respondent-task, the second
row exchanges every focal and opponent assignment field from the first row.
Every assignment field required by the narrow contract is complete.

The source survey contained 16 tasks. The authors state that scenarios 14-16
used different procedures and exclude them from their main analysis. The
released experiment table likewise contains only scenario IDs 1-13. Among
these 13 tasks, every respondent has exactly:

- four generic-versus-generic tasks;
- one task for each of seven undemocratic behaviors against a generic
  candidate;
- one task for each of two negative-valence behaviors against a generic
  candidate; and
- three Democratic-Democratic, seven cross-party, and three
  Republican-Republican contests.

The primary sample excludes the two valence tasks using randomized treatment
fields alone. It therefore contains 1,682 respondents, 18,502 paired tasks,
and exactly 11 tasks per respondent. This is not an outcome or attention-check
filter.

The count trail is imperfect but not fatal. The supplement reports 1,692
Wave-2 completers, the article reports 1,691 respondents and 21,604 choices,
and the released experiment contains 1,682 complete IDs. The separate
covariate file has the same 1,682 linkable IDs plus nine rows with missing IDs.
The primary cohort is defined transparently as every complete respondent in
the released randomized experiment; no outcome-dependent attempt is made to
recover the other records.

## Exact narrow assignment law

Use the archive's existing `candNum == 1` orientation. In an undemocratic task,
candidate 1 is the undemocratic focal candidate; in a generic-generic task it
is the source-designated generic focal candidate. The opponent is therefore
generic in every retained task. Random left-right display remains a nuisance
context and is averaged over.

Let (D) be an eight-level focal democracy factor: pooled generic conduct
(G), followed by the seven named undemocratic behaviors (U_1,\ldots,U_7).
In the 11-task core,

\[
q_D(G)=\frac{4}{11},\qquad q_D(U_j)=\frac{1}{11}.
\]

Let (P_f) and (P_o) be focal and opponent party. The ordered current-task
party law is

\[
q(DD)=q(RR)=\frac{3}{13},\qquad
q(DR)=q(RD)=\frac{7}{26}.
\]

Equivalently, focal party is uniform and independent of contest relation, with

\[
q(P_f=D)=q(P_f=R)=\frac12,qquad
q(\text{same})=\frac{6}{13},\quad
q(\text{cross})=\frac{7}{13}.
\]

The candidate-gender block contains 20 men and 12 women over all 32 candidate
slots, so its current-focal marginal is (5/8) male and (3/8) female. The
source states that the attribute blocks were randomized independently. The
narrow observed law therefore factorizes as

\[
q(D,P_f,R,S)=q_D(D)q_P(P_f)q_R(R)q_S(S),
\]

where (R) is contest relation and (S) is focal gender.

The primary target is uniform over the supported
(8\times2\times2\times2=64) cells of these four coarsened components. All
remaining randomized candidate and task features are averaged over their
outcome-blind empirical distribution. This makes the needed Riesz ratio depend
only on the certified coarsened law; it does not require invented probabilities
for profession or experience.

Because assignments are blocked without replacement across tasks and actual
display sequence is not released, identification requires a prospectively
declared mean no-carryover restriction: conditional on baseline variables,
the current focal and opponent profiles, and current task context, the mean
current choice is unaffected by profiles assigned in earlier displays.
Within-respondent outcomes may otherwise be arbitrarily dependent. All folds,
tuning, influence-score aggregation, and covariance calculations remain at the
respondent level.

## Locked interaction families

The target gives equal weight to levels within the coarsened factors. Each
family retains every target-orthogonal interaction coordinate:

| Family | Cells | Dimension | Smallest observed cell |
|---|---:|---:|---:|
| Democracy behavior x focal party | 16 | 7 | 830 |
| Democracy behavior x same/cross-party contest | 16 | 7 | 750 |
| Democracy behavior x focal candidate gender | 16 | 7 | 606 |

The audit verifies full support, rank seven, and invariance of the function-
space projector to level permutations and full-rank contrast rotations. No
coordinate may be removed after outcomes are opened.

The substantive interpretation is deliberately limited. These are effects of
coarsened, randomized contest configurations averaged over other profile
features. They are not effects of freely setting an arbitrary complete
candidate profile, and the ranked candidate configurations should not be
described as optimal political candidates.

## Baseline learner features

The two-wave design gives unusually strong timing documentation. Wave 1 was
separated from the candidate choices specifically to collect demographics,
policy views, party identification, ideology, and democracy attitudes first.
At the beginning of Wave 2, political knowledge, voting-duty, and
authoritarian-child-trait items preceded the candidate task.

The primary learner receives the following 28 raw pre-choice features:

- age, respondent gender, race, education, household-income category, and the
  state-tax wording/design block;
- seven-point party identification, ideology, Trump approval;
- sixteen pre-choice ratings of the four positions in each of the education,
  immigration, marijuana, and tax policy areas; and
- ANES knowledge total, authoritarianism total, and voting-duty orientation.

Numeric imputation, missing indicators, scaling, and categorical encoding are
learned inside each outer training fold. Survey weight is used in the target
loss and target population, never as a learner feature. State fixed effects
and item-level ratings of the exact democracy treatments are sensitivity tiers,
not primary features.

Profile-derived proximity, rank, distance, respondent-party congruence, all
outcome fields, final-choice debrief items, and duplicate engineered codings
are excluded from the primary learner.

## Why the complete-profile route fails

The full randomization is not sufficiently certified for a complete candidate
surface:

1. Figure C.5 lists profession probabilities that sum to 0.90 and omits
   `Police officer`, even though that category appears in the codebook and
   about 10% of released profiles.
2. The supplement says experience is age minus a random integer from 20 to 30.
   The release instead has age-minus-experience values 23 through 33, with
   approximately half mass at the endpoints, consistent with a different
   undocumented generator.
3. The original generator and actual task display order are absent.

Age, experience, and profession may enter an outcome learner as observed
nuisance context, but they cannot be primary causal factors or receive claimed
known profile probabilities. Policy attributes also remain empirically
standardized nuisance features in version one.

## Outcome barrier and protocol deviation

The executable audit reads the original experiment CSV with a strict design-
column allowlist and reads baseline variables from the separate covariate CSV.
It never broad-loads the experiment and drops response fields afterward.
`c_win`, `o_win`, `c_vote`, `o_vote`, and all derived response fields are
hard-forbidden.

During the initial archive inspection, a shell `head` command inadvertently
printed one complete data row and therefore displayed four binary response
fields. That record was immediately quarantined: it was not summarized, used
to choose the study, used to set an estimand, or supplied to any learner or
effect calculation. No additional response row was opened. This is a real but
minor protocol deviation and must remain in the audit manifest and eventual
paper. It does not reveal a learner comparison, an effect estimate, or an
aggregate outcome pattern.

The paired-choice endpoint itself remains locked until the complete four-study
benchmark protocol, folds, learners, equivalence margins, and reporting code
are frozen. Before it is opened, the endpoint rule is fixed: use the archived
candidate-preference indicator for the selected focal candidate and retain a
core task only when that task's preference response is observed. Report all
endpoint-missing tasks and compare their randomized treatment distribution
with the full design universe; do not substitute the turnout-intention outcome
or add an attention/completion-speed restriction.

## Gate decision

**Narrow Gate B: green.** The certified coarsened law, full family support,
sample size, pre-choice covariates, and respondent identifiers are sufficient
for the three locked democracy interaction families under mean no carryover.

**Complete-profile Gate B: failed.** Do not estimate or advertise an arbitrary
candidate factorial surface under claimed known probabilities.

With asylum seekers, climate ratings, and income-tax choices already green,
this narrow nonproduct application supplies the fourth fresh study family and
the missing restricted/nonproduct design. Portfolio Gate B is therefore
passed. The cross-study comparison protocol was subsequently frozen on
2026-09-03. The next step is its outcome-blind Gate C implementation; no new
outcome may be opened and no benchmark learner may be fit before that gate
passes.
