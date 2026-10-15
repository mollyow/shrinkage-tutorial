# orderridge and a prospective benchmark of ML adjustment in conjoint experiments

This repository holds two things:

1. **`orderridge`**, an R package for target-weighted, coding-invariant
   factorial bases, order-specific ridge penalties, and cross-fitted inference
   for attribute effects and interactions under known randomization.
2. **A prospective benchmark** that asks when a flexible machine-learning
   outcome model gives materially more precise estimates of prespecified
   interaction effects than simple additive adjustment. It covers four fresh
   public conjoint studies plus known-truth simulations.

The plan, current status, and rules are in [`docs/plan.md`](docs/plan.md).

## Layout

| Path | Contents |
|---|---|
| `R/`, `tests/`, `man/` | The `orderridge` package |
| `analysis/comparison/` | Benchmark engine, simulations, and gate runners |
| `analysis/candidate/` | Developmental candidate-recruitment application and the simulation context artifact |
| `analysis/asylum/`, `climate/`, `tax/`, `graham_svolik/`, `immigrant/` | Outcome-blind design audits, one per study |
| `config/comparison-v1.yml`, `config/comparison-simulations-v1.yml` | The benchmark protocol |
| `config/datasets/`, `config/application-registry.yml` | Study contracts and the study registry |
| `results/comparison-v1/` | Gate records; bulk outputs stay local |
| `tutorial.md` | The original 2019 shrinkage tutorial, kept for reference |

## Install and test

The lockfile records R 4.4.2 and every dependency:

```r
install.packages("renv")
renv::restore()
system("R CMD INSTALL .")
testthat::test_local(".")
```

The benchmark engine has its own tests:

```sh
Rscript analysis/comparison/tests/testthat.R
```

## Run the benchmark

From the repository root:

```sh
Rscript analysis/comparison/run_gate_c1.R --workers=2
Rscript analysis/comparison/run_simulations.R --phase=calibration --stage=1 --workers=6
Rscript analysis/comparison/evaluate_gate_d.R --stage=1
Rscript analysis/comparison/run_simulations.R --phase=calibration --stage=2 --workers=6
Rscript analysis/comparison/evaluate_gate_d.R --stage=2
```

Each phase refuses to run until the gates before it have passed. See
[`analysis/comparison/README.md`](analysis/comparison/README.md).

## Data boundary

The untracked `Candidate recruitment/` and `shrinkage/` directories are local
source archives and are not part of the package. The candidate scripts read
only `Candidate recruitment/replication/PSRM_replication/data/CleanedData.csv`.
The design audits read public archives through strict column allowlists and
never load the outcome columns. No outcome from the four fresh studies has been
read for the benchmark.
