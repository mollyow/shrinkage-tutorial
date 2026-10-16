# Gate C1 synthetic simulation render

Every simulation method on one full-size test-phase dataset per geometry, using the disjoint test seed stream. Rendered from schema-validated outputs.

## rating geometry, cell 5 (rating_nonlinear_heterogeneity), one test-phase dataset

Synthetic render for Gate C1. One dataset is not evidence about any method; the table shows that every method ran at full size and how long it took on one core.

| method | job_seconds | ok_fraction |
|---|---|---|
| additive | 2.0 | 1.00 |
| common_order_ridge | 17.0 | 1.00 |
| elastic_net | 66.2 | 1.00 |
| monotone_order_ridge | 41.8 | 1.00 |
| oracle | 1.6 | 1.00 |
| ranger | 1369.9 | 1.00 |
| task_only | 1.6 | 1.00 |
| two_way | 40.8 | 1.00 |
| unadjusted | 0.6 | 1.00 |
| unrestricted_order_ridge | 43.1 | 1.00 |
| xgboost | 30.5 | 1.00 |

## choice geometry, cell 9 (choice_low_headroom_nonlinear), one test-phase dataset

Synthetic render for Gate C1. One dataset is not evidence about any method; the table shows that every method ran at full size and how long it took on one core.

| method | job_seconds | ok_fraction |
|---|---|---|
| additive | 15.5 | 1.00 |
| common_order_ridge | 32.6 | 1.00 |
| elastic_net | 190.8 | 1.00 |
| monotone_order_ridge | 62.1 | 1.00 |
| oracle | 7.7 | 1.00 |
| ranger | 3722.6 | 1.00 |
| task_only | 13.3 | 1.00 |
| two_way | 425.2 | 1.00 |
| unadjusted | 7.8 | 1.00 |
| unrestricted_order_ridge | 67.0 | 1.00 |
| xgboost | 314.7 | 1.00 |

## Session

```yaml
created_utc: '2026-10-16T00:12:59Z'
git_head: 4cb082eb348eaeaa36ebf79e0cb18fcbf4d94e37
worktree_dirty: yes
r_version: R version 4.4.2 (2024-10-31)
platform: aarch64-apple-darwin20
blas: /Library/Frameworks/R.framework/Versions/4.4-arm64/Resources/lib/libRblas.0.dylib
package_versions:
  glmnet: 4.1.8
  ranger: 0.18.0
  xgboost: 3.2.1.1
  Matrix: 1.7.1
  nloptr: 2.2.1
  digest: 0.6.39
  yaml: 2.3.12
  nanoparquet: 0.4.3
  orderridge: 0.0.0.9000
purpose: gate_c1_synthetic_render

```

