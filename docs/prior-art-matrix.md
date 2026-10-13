# Prior-art matrix and novelty boundary

**Status:** working research memo, last checked 2026-08-26. This is a targeted scan of the closest statistical literatures, not a proof that no paper uses the exact same construction. Any priority claim should be checked again immediately before submission.

The separate [orthogonal-ML feasibility audit](orthogonal-ml-feasibility.md)
records the go/no-go decision for a proposed DDML extension. Its broad new-
estimator claim fails: the natural factorial coefficient score and interaction-
energy U-statistic are established machinery. A narrower design-aware
software/benchmark route remains conditional.

## Bottom line

The estimator currently demonstrated in the repository---ridge regression with one penalty for each polynomial or interaction order, constrained by

$$
\lambda_1 \leq \lambda_2 \leq \cdots \leq \lambda_K,
$$

and tuned by cross-validation---is **not, on its own, a defensible general methodological novelty**. Multiple-penalty ridge, group-adaptive ridge, monotone shrinkage, PAVA estimation of ordered prior variances, and CV/ML/REML selection of multiple quadratic penalties are all established.

The potentially publishable contribution is narrower:

> A design-weighted, coding-invariant, order-adaptive ridge estimator for causal response surfaces in randomized multilevel factorial and repeated conjoint experiments, paired with an honest cross-fitted correction for prespecified average marginal interaction effects (AMIEs).

The balanced-design PAVA calculation is a useful specialization and computational result, but it is not by itself a novelty claim. The inference result is also an application of orthogonal-score and cross-fitting logic, not a claim to have invented cross-fitting.

## Stage 1 gate outcome

The invariant formulation is clean enough to implement, but the gate passes
only in a narrow sense. A ridge fit on standardized reference-coded dummy
coefficients is not equivalent under baseline changes, level permutations, or
general contrast transformations. The target-weighted spaces $\mathcal H_k$
and their function norms supply information that those raw groups omit.

Once the canonical basis or its exact Gram penalty matrices are supplied,
however, the numerical problem is an ordinary generalized/multiple-penalty
ridge problem and can in principle be solved by existing generalized-ridge
machinery. Therefore the repository may claim a factorial geometry,
estimand-aware interface, and inferential integration—not a fundamentally new
ridge solver. This qualified outcome supports a software/comparison project;
it does not independently support a full new-estimator claim.

## Closest prior art

| Area | Primary source | What is already established | Exact overlap with this project | Remaining distinction, if developed |
|---|---|---|---|---|
| Generalized ridge | [Hoerl and Kennard (1970)](https://doi.org/10.1080/00401706.1970.10488634) | Generalized ridge allows differential shrinkage rather than one common ridge constant. | The proposed estimator is a structured generalized-ridge problem. | Factorial-order subspaces and causal targets are application structure, not a new ridge family. |
| Multiple quadratic penalties | [Wood (2000)](https://doi.org/10.1111/1467-9868.00240) | A general objective with several positive-semidefinite quadratic penalties and data-driven smoothing-parameter estimation by GCV. | One penalty matrix per interaction order is a special case. | Coding-invariant construction of the penalty matrices from the randomization measure could be new in this application. |
| Multiple-penalty ML/REML | [Wood (2011)](https://doi.org/10.1111/j.1467-9868.2010.00749.x) | Stable ML and REML estimation of multiple smoothing parameters in penalized generalized models. | Constrained REML for order penalties builds on this machinery. | An order-restricted factorial implementation may be useful, but REML itself is not novel. |
| CV for multiple ridge penalties | [van de Wiel, van Nee, and Rauschenberger (2021)](https://doi.org/10.1080/10618600.2021.1904962) | Fast cross-validation for block-specific ridge penalties, with software in `multiridge`. | This is the closest computational comparator to unrestricted order-specific ridge. | The monotone constraint and design-weighted factorial subspaces are absent, but merely adding the constraint is incremental. |
| Cost of tuning many penalties | [Feng and Simon (2020)](https://doi.org/10.5705/ss.202017.0310) | Oracle inequalities for split validation and CV, including penalized regression with multiple penalty parameters. | Supports, rather than originates, multi-parameter tuning. | A paper-specific finite-grid cluster-CV bound would be a specialization, not the headline result. |
| Group-adaptive empirical-Bayes ridge | [van de Wiel et al. (2016)](https://doi.org/10.1002/sim.6732) | GRridge estimates group-specific penalties from co-data and explicitly permits enforced monotonicity for ranked groups. | Ranked groups can be interaction orders; decreasing prior variance is equivalent to increasing ridge penalties. | The proposed groups are canonical factorial subspaces under a known design, rather than externally supplied feature groups. This is the closest conceptual antecedent. |
| Flexible structured co-data ridge | [van Nee, Wessels, and van de Wiel (2021)](https://doi.org/10.1002/sim.9162) | Empirical-Bayes learning of multiple, overlapping, and hierarchically structured group penalties with group-level shrinkage. | Demonstrates that structured group metadata and adaptive ridge penalties are established. | Does not target causal factorial estimands or coding invariance across factorial parameterizations. |
| Ordered smoothers | [Kneip (1994)](https://doi.org/10.1214/aos/1176325498); [Beran (2000)](https://doi.org/10.1080/01621459.2000.10473911) | Ordered linear smoothers and monotone shrinkage over an ordered orthogonal basis, selected by risk criteria. | Higher orders are an exogenous ordering of basis components. | The proposed method orders whole factorial-degree subspaces and must handle clustered randomized designs. |
| Adaptive monotone shrinkage | [Ma, Foster, and Stine (2014)](https://www.auai.org/uai2014/proceedings/individuals/140.pdf) | Ordered Gaussian prior variances, empirical-Bayes/PAVA computation, SURE equivalence, and oracle/robustness results in an orthonormal regression setting. | The balanced factorial calculation is a grouped version of this result. | General target-weighted order subspaces, unbalanced designs, and AMIE inference remain distinct. Do not claim PAVA or monotone prior variances as new. |
| Separate main-effect and interaction ridge | [van de Wiel, Amestoy, and Hoogland (2024)](https://doi.org/10.1515/em-2023-0039) | Their `ridge2` comparator uses one tuned penalty for main effects and another for two-way interactions. Their proposed Bayint prior also links local interaction shrinkage to parent main effects. | `ridge2` is the two-order version without the monotonic constraint. | This project can cover more than two orders and encodes effect hierarchy, not local heredity. |
| Batch-specific ANOVA shrinkage | [Gelman (2005)](https://doi.org/10.1214/009053604000001048) | Hierarchical ANOVA treats batches of effects through separate variance components and partial pooling. | Interaction orders can be treated as batches with order-specific variances. | Known randomization-weighted order projections and coding-invariant penalties sharpen the estimand and parameterization. |
| Structured Bayesian ANOVA | [Volfovsky and Hoff (2014)](https://doi.org/10.1214/13-AOAS685) | Hierarchical array priors borrow lower-order structure to estimate interaction arrays. | Both seek improved estimation of factorial interactions through structured shrinkage. | Their factor-specific covariance borrowing is not a monotone common penalty by order and is not the proposed design-based inference procedure. |
| Sparse interaction heredity | [Bien, Taylor, and Tibshirani (2013)](https://doi.org/10.1214/13-AOS1096); [Lim and Hastie (2015)](https://doi.org/10.1080/10618600.2014.938812); [Griffin and Brown (2017)](https://doi.org/10.1214/15-BA990) | Strong/weak heredity links the presence or shrinkage of an interaction to its parent main effects. | These are necessary competitors when the data-generating process is sparse or hereditary. | The proposed monotone order penalty encodes **effect hierarchy**, not heredity. The two concepts must not be conflated. |
| Higher-order interaction shrinkage | [Ferrari and Dunson (2021)](https://doi.org/10.1080/01621459.2020.1745813) | Bayesian latent-factor modeling can induce structured shrinkage and accommodate higher-order interactions. | Competes in moderate-dimensional higher-order settings. | Different latent model, inferential target, and computational regime. |
| Causal factorial interaction | [Egami and Imai (2019)](https://doi.org/10.1080/01621459.2018.1476246) | Defines baseline-invariant AMIEs, estimates them through weighted zero-sum ANOVA, and directly regularizes them in a penalized ANOVA framework. | Establishes both the factorial causal estimand and regularized factorial-ANOVA setting. | Dense monotone shrinkage of canonical order subspaces, followed by an honest AMIE bias correction, is the plausible wedge. |
| Repeated conjoint estimands | [Hainmueller, Hopkins, and Yamamoto (2014)](https://doi.org/10.1093/pan/mpt024) | Randomized conjoint designs identify marginal component effects, with respondent-clustered uncertainty. | Supplies the core political-methodology setting and clustering unit. | The contribution must concern response-surface regularization and interaction estimation, not basic conjoint identification. |
| Coding dependence of penalized interactions | [Martini et al. (2019)](https://doi.org/10.1534/g3.118.200961) | Penalized polynomial/interaction fits generally change under translations or other recodings when lower- and higher-order raw coefficients are penalized. | The repository's reference-coded, column-standardized penalty is exposed to this problem. | Penalizing norms of canonical design-weighted function subspaces, rather than raw coefficients, directly addresses it. This is the strongest methodological differentiator. |
| Cross-fitted debiasing | [Chernozhukov et al. (2018)](https://doi.org/10.1111/ectj.12097) | Orthogonal scores and cross-fitting remove regularization bias for low-dimensional targets estimated with flexible nuisance learners. | The proposed AMIE estimator follows this general architecture. | Known factorial randomization gives an especially simple Riesz representer and exact foldwise unbiasedness for any held-out nuisance fit. Do not claim cross-fitting itself as new. |
| Name collision | [Kawaguchi et al. (2022)](https://doi.org/10.6339/21-JDS1030) | “Hierarchical ridge regression” already names a two-level meta-feature model. | The current repository title is ambiguous. | Prefer “order-adaptive ridge,” “effect-hierarchy ridge,” or a design-specific title. |

## Precise novelty boundary

### Established and therefore not claimable

- Ridge regression with one penalty per coefficient group or effect order.
- Cross-validation, marginal likelihood, empirical Bayes, GCV, or REML for choosing several quadratic penalties.
- The inequality interpretation $\lambda_1\leq\cdots\leq\lambda_K$ as decreasing Gaussian prior variances.
- Monotone shrinkage, isotonic pooling, or PAVA computation in an ordered orthogonal problem.
- The substantive heuristic that higher-order factorial effects are often smaller.
- Regularized estimation of factorial interactions or AMIEs.
- Cross-fitting and one-step/orthogonal-score bias correction in general.
- Heredity restrictions. The proposed estimator does not impose heredity.

### Potentially defensible contribution

Subject to a renewed search and full proofs, the paper may claim to **develop and study** the following combination:

1. A canonical decomposition by factorial order defined as orthogonal complements of nested function spaces in $L_2(\mu)$, where $\mu$ is a prespecified randomization or target measure.
2. A monotone order penalty applied to the function norms of these subspaces, making fitted response surfaces invariant to reference categories and full-rank contrast reparameterizations.
3. An implementation for randomized multilevel factorial and repeated conjoint designs, including unequal assignment probabilities and respondent clustering.
4. Cross-fitted, randomization-based correction of a fixed, prespecified set of AMIEs, so that shrinkage is used for efficiency while the reported causal contrasts are not left with plug-in ridge bias.
5. A balanced-design specialization that exposes the risk calculation and weighted-PAVA solution, clearly presented as an adaptation of existing monotone-shrinkage theory.

The safe wording is “we develop,” “we specialize,” or “we combine.” Avoid “the first” unless a later systematic review supports the exact claim.

## Effect hierarchy is not heredity

- **Effect hierarchy:** average signal per degree of freedom tends to decline with interaction order. The constraint $\lambda_1\leq\cdots\leq\lambda_K$ encodes this belief.
- **Effect heredity:** an interaction involving factors $j$ and $k$ should be present or weakly shrunk only when one or both parent main effects are important. Bayint, `hierNet`, and `glinternet` encode versions of this idea.

An order-adaptive ridge may retain a large $j$-by-$k$ interaction even when both corresponding main effects are zero, provided the data support it. That is intentional and should be described as robustness to anti-hereditary interactions, not as a heredity method.

## Competitors and required ablations

The minimum empirical comparison should include:

1. Saturated or estimable OLS with respondent-clustered inference.
2. Truncated ANOVA models at prespecified maximum orders.
3. Ordinary ridge with one penalty.
4. Unrestricted order-specific multi-penalty ridge. This is the indispensable ablation for the monotonicity constraint.
5. The proposed monotone order-adaptive ridge.
6. GRridge or `ecpc`, where the feature grouping can be supplied comparably.
7. `FindIt`/regularized causal ANOVA for factorial AMIEs.
8. `glinternet` or `hierNet` for sparse heredity.
9. Bayint for feasible two-way, moderate-dimensional settings.
10. A direct Horvitz--Thompson/ANOVA AMIE estimator and the cross-fitted corrected estimator, to isolate efficiency and coverage from prediction.

Report target-weighted response-surface MSE, order-specific function error, AMIE bias and interval coverage, optimal-profile or policy regret if substantively meaningful, and runtime. Simulations must include monotone, flat, reversed, sparse/hereditary, and anti-hereditary signal patterns.

## Go/no-go conditions for a methods-paper claim

- Fitted functions and AMIEs must agree numerically after every baseline swap, level permutation, and random full-rank contrast transformation.
- The required order spaces and AMIEs must be identified on the actual randomization support; prohibited profiles cannot be recovered by shrinkage.
- Cross-fitted intervals must attain approximately nominal respondent-clustered coverage in simulations modeled on the intended applications.
- The monotone constraint must offer a stable risk or efficiency advantage over unrestricted multi-penalty ridge in at least the stated hierarchy regime, without catastrophic behavior under plausible violations.
- The paper must show that the design-weighted construction changes something material relative to reference-coded ridge: invariance, interpretation, inference, or all three.

If these checks fail, the appropriate product is a tutorial, software note, or applied comparison rather than a new general method.
