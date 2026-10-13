# Design-only application dataset registry

**Registry date:** 2026-09-01  
**Status:** Gate B passed and the common scientific comparison protocol is frozen; Gate C implementation remains pending  
**Selection rule:** studies are screened on design, support, covariate timing,
sample structure, and reproducibility—not on whether a flexible learner wins.

## Decision

The empirical paper should display candidate recruitment, but it should not
count that study as fresh evidence. It has already shaped the estimands,
learner library, simulations, and earlier feasibility gates. The defensible
design is therefore:

1. **one developmental study:** candidate recruitment; and
2. **four fresh confirmatory study families:** asylum seekers, climate
   agreements, income-tax plans, and Graham-Svolik democratic candidate
   choice.

The European Defense Union study is the leading thin-covariate calibration
case. Immigrant admission remains an amber restricted-support reserve. No new
learner comparison, aggregate outcome association, effect estimate, or
interaction result was used to make this ranking.

The portfolio Gate B is passed. Asylum seekers are eligible for both locked
outcomes. Climate agreements count through their rating analysis only. Income-
tax plans now count through a narrow paired-choice analysis of total effects of
the six randomized rates; the naturally induced revenue display is averaged
over rather than treated as a seventh factor. Graham-Svolik counts through a
narrow paired-choice analysis of the coarsened nonvalence democracy-contest
randomization, not through a complete candidate-profile surface. Its block law
supplies the required nonproduct case. The next stage is Gate C implementation
under the now-frozen
[common comparison protocol](cross-study-comparison-contract.md). No new
endpoint may be opened during that work.

## Eligibility rule

A fresh headline study must have all of the following:

- an original or author-endorsed public archive with a persistent identifier,
  an explicit reuse license, and immutable file checksums;
- respondent, task, and—when paired—profile-position identifiers or a
  deterministic neutral reconstruction, along with every randomized attribute
  and the declared outcome columns;
- enough documentation to reconstruct assignment probabilities, restrictions,
  structural zeros, and the supported target distribution;
- at least 1,000 independent respondent clusters and 5,000 respondent-tasks;
- a primary covariate set limited to documented pre-conjoint variables or
  logically immutable demographics;
- at least one substantively motivated, fixed, target-weighted factor-pair
  interaction family that is identified on the declared support;
- respondent-level outer and inner folds; and
- a primary sample that does not condition on a post-assignment attention
  check or another post-treatment variable.

Eligibility is outcome- and estimand-specific. A study may enter with one
prespecified outcome while another remains excluded. Such a study counts only
once, and only the eligible outcome enters the benchmark.

The four fresh studies, considered together, must include at least three
substantive domains, one multinational study, both rating and paired-choice
outcomes, both product and restricted assignment, and at least three studies
with five or more usable pre-conjoint covariates. Countries, waves, outcomes,
and experimental arms nested inside one study are not counted as independent
studies. Headline cross-study summaries weight study families equally.

Published articles necessarily report substantive outcomes; this audit is not
literature-blind. The protection is narrower and enforceable: enrollment,
covariates, targets, contrasts, and analysis code are frozen before any **new**
learner-performance or effect analysis is run.

## Recommended portfolio

| Role | Study | Design-only facts | Main value | Status before outcomes |
|---|---|---|---|---|
| Developmental exhibit | Eggers et al., candidate recruitment | 3,000 respondents; 8 single-profile ratings; 7 attributes; all 1,440 profiles observed | Unique rating design and an extensively audited application | **Eligible for display after a terminology correction; excluded from Gate B and confirmatory aggregate** |
| Confirmatory anchor | Bansak, Hainmueller, and Hangartner, asylum seekers | 18,021/14,966 design-complete respondents in 2016/2022; 15 countries; 5 paired tasks; 9 attributes; forced choice and 1–7 rating | Very large multinational test with ample nuisance-learning information | **Gate B green; outcomes remain unopened for the benchmark** |
| Confirmatory product-design case | Bechtel and Scheve, climate agreements | 8,500 respondents in 4 countries; 4 paired tasks; 68,000 profile rows; all 2,880 profiles; 6 attributes | A second multinational domain and a sharp test of outcome-specific assignment requirements | **Gate B green for rating/profile contrasts under declared mean-no-opponent and mean-no-carryover restrictions; paired choice amber and excluded** |
| Confirmatory high-task case | Ballard-Rosa, Martin, and Scheve, income-tax plans | Main arm: 2,000 respondents; 8 paired tasks; 32,000 profile rows; 6 randomized bracket rates; choice | High-task product design and a useful downstream-mediator boundary case | **Gate B green for six-rate total schedule effects under the locked q x q reconstruction and mean no-carryover; outcomes remain unopened** |
| Confirmatory nonproduct case | Graham and Svolik, democratic candidate choice | Released core: 1,682 respondents; 11 paired tasks; exact blocked democracy and party marginals; rich pre-choice covariates | A demanding nonproduct law and whole democracy-by-contest families | **Gate B green for the narrow coarsened nonvalence analysis under mean no-carryover; complete-profile surface excluded** |
| Restricted-support reserve | Hainmueller and Hopkins, immigrant admission | Outcome-blind design universe: 1,407 respondents, 5 paired tasks, 14,070 profile rows; 9 attributes; choice and rating | Restricted support, a rich locked pre-conjoint covariate set, and 74 released-support coordinates (12 on hold) | **Gate B amber: assignment law not fully certified; not needed for the four-study gate** |
| Thin-X calibration | Nicoli, Burgoon, and van der Duin, European Defense Union | Archive defense arm: 3,839 respondents; 3 paired tasks; 23,034 profile rows; 6 attributes; choice and support/rating measures | Prospectively weak covariate headroom and a recent low-task study | **Provisional secondary** |

### 0. Candidate recruitment: developmental exhibit

Sources: [article](https://doi.org/10.1017/psrm.2025.10076),
[CC0 replication archive](https://doi.org/10.7910/DVN/BQAHGZ), and the local
[application contract](../analysis/candidate/README.md).

- The marginal profile law is uniform on the
  `5 x 4 x 3 x 3 x 2 x 2 x 2` grid.
- The seven attributes are documented as independently randomized within a
  profile. Independence **across tasks is not established**: no respondent
  received a duplicate complete profile among eight tasks, which is
  inconsistent with ordinary iid sampling from 1,440 profiles.
- Cluster inference can allow arbitrary within-respondent dependence, but the
  registry must distinguish the known rowwise marginal law from the unknown
  joint task mechanism.
- `estimands.yml` version 4 now labels `1/1440` as a rowwise marginal and makes
  the unknown across-task mechanism explicit. Version-3 prepared and analysis
  manifests are stale by checksum and must be rebuilt before reuse.
- The four already locked families remain authority x legislative threshold,
  authority x staff support, fundraising burden x probability of winning, and
  fundraising burden x salary.

This study stays in the paper because it motivated the question and gives a
single-profile rating example. The terminology correction does not change the
rowwise score or respondent-clustered covariance, but it must be made before
reuse. All confirmatory summaries must also be reported without this study.

### 1. Asylum-seeker profiles: confirmatory anchor

Sources: [article](https://doi.org/10.1038/s41586-023-06417-6) and
[CC0 replication archive](https://doi.org/10.7910/DVN/FTL1MM), plus the
[prospective plan](https://osf.io/jd8n3/).

- The outcome-blind design universes contain 18,021 respondents and 180,210
  profile rows in 2016, and 14,966 respondents and 149,660 profile rows in
  2022. Nine and ten additional respondents, respectively, have wholly missing
  ten-row profile blocks.
- The plan explicitly specifies non-restricted uniform randomization. The full
  product supports contain 136,080 profiles in 2016 and 170,100 in 2022. The
  only support change is the added 2022 `War` migration-reason level.
- The primary target uses separate wave-specific profile laws, equal weight for
  each of 15 countries with released weights normalized within country, and a
  `1/4` mixture over the randomized 2022 frames.
- Common-support rating comparisons remove `War` from the focal target;
  forced-choice comparisons remove it from both focal and opponent targets.
- The locked primary learner features are country, gender, age, home-country
  birth, education, and three asylum/immigration questions documented before
  the conjoint. Employment, income, ideology, feeling thermometers, and later
  attitude batteries are not primary features.
- The complete family ranks are 12/16 for migration reason x vulnerability in
  2016/2022, 12 for religion x origin, and 10 for occupation x language. Every
  family cell occurs in every country.
- Five respondent-level outer folds and five inner labels are frozen without
  outcomes. The waves have no overlapping respondent IDs and are repeated
  cross-sections, not a panel.

The executable [contract](../config/datasets/asylum-seekers.yml),
[audit](../analysis/asylum/audit_design.R), and
[decision](asylum-design-contract-decision.md) reproduce these facts while
prohibiting `pref`, `rate`, and `ratebin` from being loaded.

**Current decision:** Gate B green. The absent platform generator is not a
blocker because the prospective assignment law is explicit. The missing
original 2016 questionnaire remains a disclosed limitation and motivates a
demographic-only timing sensitivity. This family counts once toward the
four-study portfolio, regardless of its countries, waves, or two outcomes.

### 2. Immigrant admission: restricted-support case

Sources: [article](https://doi.org/10.1111/ajps.12138) and
[CC0 replication archive](https://doi.org/10.7910/DVN/25505).

- The outcome-blind design universe has 1,407 respondents, five paired tasks,
  7,035 tasks, and 14,070 profile rows. Another 307 archive respondents have
  the entire profile block missing. The former registry count of 1,396 was
  outcome-complete and is not an admissible design-enrollment rule.
- The nine attributes have two restrictions. Education x profession contains
  61 of 77 cells. The released origin x reason support contains 24 of 30 cells.
- The released rows closely match a conditional-uniform law with education and
  origin drawn uniformly, profession uniform on the set allowed at each
  education, reason uniform on the set allowed at each origin, and all other
  attributes uniform. This yields 936,960 admissible profiles. The primary
  proposed target is this released row-marginal assignment distribution,
  `mu = q`. Published randomization does not by itself certify the within-pair
  `q x q` opponent law, so this must be confirmed or adopted prospectively as
  an explicit assumption; across-task dependence remains unrestricted.
- There is a material source conflict. The article says persecution was paired
  only with Iraq, Sudan, and Somalia, implying 23 origin-reason cells and
  897,920 profiles. The released data contain 500 China-persecution profiles,
  and the released author code explicitly uses China as a fourth eligible
  origin. The three-file archive has no original randomizer.
- A strict primary nuisance allowlist is now locked to stable demographics and
  documented Wave-1 constructs. Party ID and ideology are secondary because
  exact timing is unresolved; the Wave-2 attitude and occupation variables are
  excluded. Missingness indicators and imputation must be learned inside each
  outer fold.
- Support-aware weighted-basis calculations identify 44 education x profession
  coordinates, 12 released-support origin x reason coordinates, and 18 language
  x education coordinates. All allowed cells occur in the released sample, and
  numerical invariance checks pass.

The machine-readable
[design contract](../config/datasets/immigrant-admission.yml), executable
[audit](../analysis/immigrant/audit_design.R), and full
[decision](immigrant-design-contract-decision.md) were created without loading
outcomes or fitting models.

**Current decision:** Gate B is amber. Education x profession and language x
education pass identification. Origin x reason remains on hold. Turn the study
green only after obtaining generator/author confirmation or prospectively
declaring the released data and author code authoritative, disclosing the
article conflict, and adopting the uncertified within-pair `q x q` law.
Implement the package's supported-subspace contrast constructor before outcome
fitting.

### 3. Climate agreements: product-randomization case

Sources: [article](https://doi.org/10.1073/pnas.1306374110) and the authors'
[CC0 replication archive](https://doi.org/10.7910/DVN/UGZ2BY).

- The outcome-blind universe contains 8,500 respondents—2,000 each in France,
  Germany, and the United Kingdom and 2,500 in the United States—34,000 paired
  tasks, and 68,000 profile rows. Every respondent has four tasks and two
  adjacent source rows per task.
- Six attributes vary within profiles: household cost, burden distribution,
  participating countries, emissions coverage, sanctions, and monitoring.
  The contract prospectively reconstructs the row marginal as the uniform
  product `q(a)=1/2880`. All 2,880 profiles and all 15 two-factor Cartesian
  supports are observed, overall and by country. Marginal checks by country,
  task, and profile index and a pooled full-profile test are compatible with
  this reconstruction, but no generator or explicit probability table exists.
- The exact paired law is uncertified. Zero tasks contain identical complete
  profiles, versus 11.806 expected under independent uniform task-pair draws;
  the iid probability of zero is `7.4477e-6`. With arbitrary dependence among
  four tasks, however, a respondent-cluster upper probability is about 0.0522,
  so the zero count does not logically reject per-task `q x q` by itself.
  Individual-factor match rates look ordinary, profiles repeat across
  different tasks, and the likely within-pair rejection rule is not source-
  certified because the generator is absent.
- For ratings, the contract prospectively assumes conditional mean no-opponent
  and mean no-carryover: fixing the current focal profile, respondent
  covariates, and task, changing either the opponent or profiles shown in prior
  tasks does not change the focal profile's conditional mean rating. These are
  identifying restrictions, not randomized guarantees.
- The primary target is an equal one-quarter country mixture, with released
  weights normalized within country, and the uniform profile marginal.
- Primary learner features are country, age, female, country-specific
  education, country-specific income, and the documented pre-conjoint climate-
  cooperation support item. Ambiguously timed attitudes, reciprocity,
  knowledge items, and the attention check remain excluded.
- The locked families are household cost x burden distribution (dimension
  12), sanctions x monitoring (9), and participation x emissions coverage (4).
  Every family cell occurs in every country and task position.
- `choice_cj`, `rating_cj`, and outcome-derived `incon` remained unopened. The
  original DTA is 3,001,271 bytes with MD5
  `85dc26aeda5a99e881ad777f03d8410c`; Dataverse's larger number refers to its
  ingested tabular copy.

The executable [contract](../config/datasets/climate-agreements.yml),
[audit](../analysis/climate/audit_design.R), and
[decision](climate-design-contract-decision.md) reproduce these facts and
freeze respondent-level folds and invariant interaction projectors.

**Current decision:** Gate B is green for the rating/profile analysis under
the prospectively declared mean-no-opponent and mean-no-carryover restrictions.
Paired choice is amber and excluded. Obtain the original randomizer/generator or author
confirmation of the no-identical-profile rule and exact conditional opponent
probabilities; otherwise paired choice stays out.

### 4. Income-tax plans: high-task structured case

Sources: [article](https://doi.org/10.1086/687324) and
[CC0 replication archive](https://doi.org/10.7910/DVN/NGRGS5), plus the
[executable contract](../config/datasets/income-tax-plans.yml) and
[audit decision](income-tax-design-contract-decision.md).

- The release contains 2,250 respondents. The author-defined primary arm is
  `saw_revenue == 1`: 2,000 respondents, 16,000 paired tasks, and 32,000
  profile rows. The other 250 respondents saw an otherwise identical version
  without the revenue display. The distinct 500-person rating pilot mentioned
  in the article is not archived.
- The six rate factors form an unrestricted
  `4 x 4 x 4 x 4 x 5 x 6 = 7,680` schedule grid. The primary arm observes
  7,554 schedules, close to the iid-uniform occupancy expectation of 7,560.96,
  and every two-factor cell is observed.
- The sources support a prospective uniform product reconstruction,
  `q(r)=1/7,680`, and an independent ordered opponent draw within a task,
  `q(r_A)q(r_B)=1/58,982,400`. The original generator is absent, so this is
  source-supported rather than generator-certified. Unlike the climate data,
  the realized pair and repeat diagnostics do not contradict it: zero exact
  within-task schedule matches has probability 0.124 under the lock, and 33
  across-task repeat pairs are almost exactly the 32.81 iid expectation.
- Revenue is computed from the six rates, perturbed by Gaussian noise, and
  binned before display. It is a stochastic downstream mediator, not a seventh
  randomized factor. The eligible surface therefore averages the natural
  revenue mechanism; primary nuisance fits may not condition on the realized
  revenue category.
- The target uses survey weights normalized within the main arm, uniform rate
  schedules and opponents, and equal averaging over eight tasks and two sides.
  Mean no-carryover is declared prospectively. Arbitrary dependence within a
  respondent is allowed for covariance.
- Seven conservative baseline features are locked: age, female, race,
  education, party identification, ideology, and political interest. These are
  the variables used by YouGov to construct the matched sample. Post-conjoint
  policy attitudes, experimental-game measures with conflicting timing
  evidence, attention indicators, and the realized revenue category are not
  learner features.

The complete locked families are lowest-bracket x highest-bracket rate
(dimension 15), $85,000-$175,000 x highest-bracket rate (15), and
$175,000-$375,000 x highest-bracket rate (20). All coordinates are retained.

**Current decision:** Gate B green for paired-choice total effects of the six
randomized rate schedules only. Seven-factor surfaces, direct or controlled
revenue effects, the small no-revenue arm as a headline analysis, and pilot
ratings remain ineligible. No choice value was loaded to make this decision.

### 5. Democratic candidate choice: nonproduct confirmatory case

Sources: [article](https://doi.org/10.1017/S0003055420000052),
[CC BY 4.0 replication archive](https://doi.org/10.7910/DVN/EEARKA), and the
[design-contract decision](graham-svolik-design-contract-decision.md).

- The canonical original experiment CSV has 1,682 IDs, 13 retained tasks per
  ID, two mirrored candidate rows per task, and 43,732 rows. The audit locks the
  original CSV byte size and MD5 rather than confusing them with Dataverse's
  ingested `.tab` derivative.
- The core excludes the two randomized negative-valence tasks and selects the
  archive's candidate-1 orientation. Every respondent then supplies four
  generic controls and one task for each of seven undemocratic behaviors, for
  18,502 paired choices. The opponent is generic in every retained task.
- The exact focal democracy law is `q(G)=4/11` and `q(U_j)=1/11`. The exact
  ordered party law is `q(DD)=q(RR)=3/13` and
  `q(DR)=q(RD)=7/26`; equivalently, focal party is uniform and the contest is
  same-party with probability `6/13`. Focal gender has the documented marginal
  `5/8` male and `3/8` female. Attribute blocks are independently randomized.
- The target is uniform over the supported democracy x focal-party x contest-
  relation x focal-gender cells. Other profile and task variables are averaged
  over their outcome-blind empirical distribution. This makes the analysis a
  coarsened contest experiment, not a claim about arbitrary complete profiles.
- Actual display sequence is absent and assignments are without replacement,
  so the contract declares conditional mean no carryover. Respondent-level
  folds and covariance otherwise allow arbitrary within-respondent dependence.
- The rich 28-feature learner set consists of documented pre-choice
  demographics, party/ideology variables, sixteen policy ratings, knowledge,
  authoritarianism, and voting-duty orientation. All preprocessing is learned
  within training folds.
- The locked complete families are democracy behavior x focal party,
  democracy behavior x same/cross-party contest, and democracy behavior x
  focal candidate gender. Each has dimension seven and full support; their
  smallest observed cells are 830, 750, and 606.

The complete-profile route is excluded. The published profession probabilities
omit Police officer and sum to 0.90, while the published age-experience
generator conflicts with the released offsets. No original generator or task
sequence is available. Age, experience, profession, and policy context may be
learner inputs or empirical nuisance context, but not primary target factors
with claimed known probabilities.

The source count trail is also disclosed: 1,692 Wave-2 completers in the
supplement, 1,691 article respondents, and 1,682 complete IDs in the released
experiment. The analysis enrolls every released design-complete ID and does not
use an attention check or outcome filter.

One raw data row was inadvertently printed during initial archive inspection,
exposing four binary response fields. It was quarantined and never summarized
or used in enrollment, target, learner, or effect decisions. No learner result,
effect estimate, or aggregate outcome pattern was opened. The deviation stays
in the permanent manifest.

The executable [contract](../config/datasets/graham-svolik.yml),
[audit](../analysis/graham_svolik/audit_design.R), and full
[decision](graham-svolik-design-contract-decision.md) freeze this narrow scope.

**Current decision:** Gate B green for the coarsened nonvalence paired-choice
analysis only. It supplies the fourth fresh study and the required nonproduct
assignment. Outcomes remain locked until the common benchmark is frozen.

### 6. European Defense Union: thin-X calibration

Sources: [article](https://doi.org/10.1093/isq/sqaf044) and
[CC0 replication archive](https://doi.org/10.7910/DVN/HTAPIS).

- The public archive has 7,678 respondents across two framing arms. The defense
  arm has 3,839 respondents and 23,034 profile rows, not exactly the nominal
  3,750 reported in the study description; this discrepancy must be resolved.
- Respondents complete three paired tasks over six independently randomized
  attributes, yielding a 192-profile product grid.
- Clear primary covariates appear limited to country, age, education, and
  income. This makes the study useful precisely as a setting with little
  obvious ML headroom.
- The authors' attention-check restriction must not define the primary sample.
  The full randomized arm is primary; attention-passing respondents may form a
  labeled sensitivity analysis.

Provisional complete families: governance x opt-outs, program scope x joint
procurement, and financing x force size.

**Pass condition:** reconcile the article and archive sample counts, lock one
framing arm or an explicit mixture target, recover the exact rating/support
scale, and document covariate timing. This study does not satisfy the rich-X
requirement and cannot rescue a failure among the first four by itself.

## Reserves and exclusions

| Study | Why it is attractive | Current disposition |
|---|---|---|
| [Saha and Weeks](https://doi.org/10.1007/s11109-020-09636-z); [archive](https://doi.org/10.7910/DVN/KVTPVX) | Substantively natural gender-by-ambition interactions; original CC0 archive | About 1,191 respondents x 3 tasks falls below the 5,000-task headline threshold, and moderator schemas differ across `sconjoint` versions. Software/example reserve only. |
| [Teele, Kalla, and Rosenbluth](https://doi.org/10.1017/S0003055418000217); [archive](https://doi.org/10.7910/DVN/FVCGHC) | Candidate gender and social-role interactions across voter and legislator samples | Sparse learner covariates, mixed samples, possible profile restrictions, and custom redistribution terms. Reserve pending a separate contract. |
| [Hainmueller, Hopkins, and Yamamoto](https://doi.org/10.1093/pan/mpt024); [public mirror](https://doi.org/10.7910/DVN/ARHZU4) | Familiar conjoint geometry and both outcomes | Only 311 respondents and little covariate headroom. Use only as a small software smoke test. |
| Local campaign-finance conjoint | Two single-profile tasks and complete observed support | Excluded until a public DOI, license, original randomization record, and sample-count provenance are found. |

Bundled or transformed `sconjoint` objects are discovery aids, not analysis
inputs. Their release and development branches expose different moderator
frames. Every admitted study must be rebuilt from its original archive with a
pinned file version and checksum.

## Gate-B verdict and next audit

**Verdict: four study families have eligible locked analyses; portfolio Gate B
passed.** Asylum seekers are eligible for both outcomes. Climate agreements are
eligible for rating only and still count once, not once per outcome. Income-tax
plans are eligible for the narrow six-rate total-effect choice analysis.
Graham-Svolik is eligible for the narrow coarsened nonvalence choice analysis
and supplies the restricted/nonproduct assignment case.

The common estimands, endpoint rules, respondent folds, learner library,
tuning budgets, positive controls, equivalence margins, and reporting
requirements are now frozen in
[`comparison-v1.yml`](../config/comparison-v1.yml) and
[`comparison-simulations-v1.yml`](../config/comparison-simulations-v1.yml).
The next step is to implement the shared engine, strict endpoint readers,
synthetic Gate-C tests, output schemas, dependency lock, and implementation
manifest. Only after those artifacts pass and are hashed may the locked
outcomes be read. Immigrant admission remains an optional reserve.

Seeking the climate opponent generator and resolving the defense-study counts
remain worthwhile extensions, but neither is required for the four-study
benchmark.
