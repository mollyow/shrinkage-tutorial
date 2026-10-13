# Income-tax-plan outcome-blind design audit

This directory implements the Gate-B contract for Ballard-Rosa, Martin, and
Scheve's income-tax conjoint. It admits only the N=2,000 displayed-revenue arm
and only total effects of the six randomized bracket rates. The noisy revenue
display is a downstream mediator, not a seventh factor.

The locked contract is
[`config/datasets/income-tax-plans.yml`](../../config/datasets/income-tax-plans.yml),
and the substantive decision is
[`docs/income-tax-design-contract-decision.md`](../../docs/income-tax-design-contract-decision.md).

Run the audit from the repository root with:

```sh
Rscript analysis/tax/audit_design.R
```

Run its tests with:

```sh
Rscript analysis/tax/tests/testthat.R
```

The audit reads the Stata file with a strict `col_select` allowlist, verifies
archive hashes, reconstructs the respondent-task-plan keys and main arm, checks
the uniform schedule and opponent laws, audits the natural revenue mechanism,
locks three complete interaction spaces, assigns respondent-level folds, and
writes a manifest that records `outcome_blind: true` and
`outcomes_loaded: false`.

No generated result in `analysis/tax/derived/` is evidence about an outcome.
Choice fields, ideal-rate responses, derived preference scores, attention
variables, and post-conjoint attitudes are never loaded by this audit.
