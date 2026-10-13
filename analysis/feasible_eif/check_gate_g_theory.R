#!/usr/bin/env Rscript

# Deterministic checks for Gate G's exact cross-fit loading representation and
# count-corrected saturated residual covariance.

find_project_root <- function(start = getwd()) {
  here <- normalizePath(start, winslash = "/", mustWork = TRUE)
  repeat {
    if (file.exists(file.path(here, "DESCRIPTION"))) return(here)
    parent <- dirname(here)
    if (identical(parent, here)) {
      stop("Could not find the project root.", call. = FALSE)
    }
    here <- parent
  }
}

project_root <- find_project_root()
source(file.path(
  project_root, "analysis", "covariance_headroom", "calibration_solver.R"
))
source(file.path(
  project_root, "analysis", "feasible_eif", "feasible_estimator.R"
))
source(file.path(
  project_root, "analysis", "feasible_eif", "simulation_functions.R"
))
source(file.path(
  project_root, "analysis", "feasible_eif", "sensitivity_functions.R"
))
source(file.path(
  project_root, "analysis", "feasible_eif", "e2_simulation_functions.R"
))
source(file.path(
  project_root, "analysis", "feasible_eif", "gate_g_corrections.R"
))
source(file.path(
  project_root, "analysis", "feasible_eif", "gate_g_simulation_functions.R"
))

check_count <- 0L

check_close <- function(label, value, target, tolerance = 1e-10) {
  error <- max(abs(as.numeric(value) - as.numeric(target)))
  if (!is.finite(error) || error > tolerance) {
    stop(sprintf("FAIL %-65s %.6g", label, error), call. = FALSE)
  }
  check_count <<- check_count + 1L
  cat(sprintf("PASS %-65s %.3e\n", label, error))
  invisible(error)
}

check_true <- function(label, value) {
  if (!isTRUE(value)) stop(sprintf("FAIL %s", label), call. = FALSE)
  check_count <<- check_count + 1L
  cat(sprintf("PASS %s\n", label))
  invisible(value)
}

config <- read_gate_e2_config(file.path(
  project_root, "analysis", "feasible_eif", "gate-e2-production.yml"
))
geometry <- gate_d_geometry(config)
scenario <- list(
  id = "gate_g_algebra",
  class = "null",
  respondents = 320L,
  tasks = 8L,
  assignment = "product",
  true_assignment = "product",
  analysis_assignment = "product",
  covariance = "compound_symmetry",
  rho = 0.8,
  residual_rho = 0.8,
  carryover = "none"
)
set.seed(2026082801L)
data <- draw_gate_d_data(scenario, config, geometry)
n <- data$respondents
tasks <- data$tasks
task_weights <- rep(1 / tasks, tasks)
sigma_true <- scenario_covariance(scenario, tasks)
outer_fold <- balanced_supported_cluster_folds(
  data$profile_index, 5L, 2026082802L, geometry$profile_count
)
oracle <- solve_task_mixing(
  sigma_true, task_weights, "product"
)
operators <- rep(list(oracle), length(unique(outer_fold)))

# -------------------------------------------------------------------------
# 1. Exact cross-fitted linear representation and covariance propagation.
# -------------------------------------------------------------------------

linearized <- gate_g_crossfit_linearization(
  data$profile_index, data$outcome, outer_fold, operators,
  geometry, task_weights
)
check_close(
  "cross-fitted point estimate equals its propagated linear form",
  linearized$estimate,
  linearized$linear_estimate
)
check_close(
  "recorded linear-representation error is numerical zero",
  linearized$representation_error,
  0
)
check_true(
  "every outer mean-training set has complete saturated support",
  all(linearized$fold_diagnostics$empty_training_cells == 0L)
)

conditional_mean <- vapply(
  seq_len(ncol(geometry$riesz_lookup)),
  function(coordinate) {
    sum(linearized$linear_load[, , coordinate] * data$mean)
  },
  numeric(1)
)
check_close(
  "conditional mean of the cross-fitted score is the factorial target",
  conditional_mean,
  geometry$truth,
  tolerance = 1e-9
)

conditional_covariance <- gate_g_conditional_covariance(
  linearized$linear_load, sigma_true
)
check_true(
  "propagated conditional covariance is positive definite",
  min(eigen(conditional_covariance, symmetric = TRUE)$values) > 0
)

first_order_variance <- gate_e2_mixing_variance(
  oracle$mixing, sigma_true, diag(tasks)
) / n
check_true(
  "finite-sample mean propagation exceeds the first-order oracle variance",
  all(diag(conditional_covariance) > first_order_variance)
)

# Perturbing held-out outcomes changes their point contribution but cannot
# alter any fixed operator.  The linear loading depends only on assignments,
# folds, and the prespecified operator.
perturbed_outcome <- data$outcome
perturbed_outcome[outer_fold == 1L, ] <-
  perturbed_outcome[outer_fold == 1L, ] + 1000
linearized_perturbed <- gate_g_crossfit_linearization(
  data$profile_index, perturbed_outcome, outer_fold, operators,
  geometry, task_weights
)
check_close(
  "fixed-operator linear load is outcome invariant",
  linearized_perturbed$linear_load,
  linearized$linear_load
)

# -------------------------------------------------------------------------
# 2. Exact count inflation of saturated OOF residual moments.
# -------------------------------------------------------------------------

inner_fold <- balanced_supported_cluster_folds(
  data$profile_index, 4L, 2026082803L, geometry$profile_count
)
oof <- gate_g_saturated_oof_residual(
  data$profile_index, data$outcome, inner_fold, geometry, task_weights
)
check_close("supported OOF construction has no empty cells",
            oof$empty_cell_count, 0)
inflation <- gate_g_saturated_residual_inflation(
  data$profile_index, inner_fold, geometry$profile_count
)
check_true(
  "count inflation is symmetric and elementwise above one",
  max(abs(inflation - t(inflation))) < 1e-12 && all(inflation > 1)
)
check_true(
  "diagonal cell-mean inflation exceeds average off-diagonal inflation",
  mean(diag(inflation)) >
    mean(inflation[row(inflation) != col(inflation)])
)

# Compute E(r_i r_i' | assignments) independently by propagating every OOF
# residual's coefficients on all respondent error vectors.
direct_expected_raw <- matrix(0, tasks, tasks)
for (respondent in seq_len(n)) {
  fold <- inner_fold[[respondent]]
  train <- inner_fold != fold
  coefficient <- vector("list", n)
  coefficient[[respondent]] <- diag(tasks)
  for (training_respondent in which(train)) {
    diagonal <- numeric(tasks)
    for (tt in seq_len(tasks)) {
      count <- sum(
        train & data$profile_index[, tt] ==
          data$profile_index[respondent, tt]
      )
      diagonal[[tt]] <- -as.numeric(
        data$profile_index[training_respondent, tt] ==
          data$profile_index[respondent, tt]
      ) / count
    }
    coefficient[[training_respondent]] <- diag(diagonal)
  }
  covariance_i <- matrix(0, tasks, tasks)
  for (value in coefficient) {
    if (!is.null(value)) {
      covariance_i <- covariance_i + value %*% sigma_true %*% t(value)
    }
  }
  direct_expected_raw <- direct_expected_raw + covariance_i / n
}
check_close(
  "count formula equals direct OOF residual covariance propagation",
  inflation * sigma_true,
  direct_expected_raw,
  tolerance = 1e-11
)
check_close(
  "elementwise count correction exactly recovers true covariance in expectation",
  direct_expected_raw / inflation,
  sigma_true,
  tolerance = 1e-11
)

corrected <- gate_g_count_corrected_covariances(
  oof$residual, data$profile_index, inner_fold, geometry$profile_count
)
check_close(
  "reported inflation equals the independently computed matrix",
  corrected$inflation,
  inflation
)
check_close(
  "corrected raw moment is raw moment divided by exact count inflation",
  corrected$corrected_second_moment,
  corrected$raw_second_moment / inflation
)

permutation <- sample(seq_len(geometry$profile_count))
permuted_profile <- matrix(
  permutation[data$profile_index], n, tasks
)
permuted_inflation <- gate_g_saturated_residual_inflation(
  permuted_profile, inner_fold, geometry$profile_count
)
check_close(
  "count correction is invariant to profile-level relabeling",
  permuted_inflation,
  inflation
)

task_permutation <- sample(seq_len(tasks))
task_inflation <- gate_g_saturated_residual_inflation(
  data$profile_index[, task_permutation], inner_fold,
  geometry$profile_count
)
check_close(
  "count correction is equivariant to task relabeling",
  task_inflation,
  inflation[task_permutation, task_permutation]
)

# -------------------------------------------------------------------------
# 3. Expected finite-G mean-remainder objective under product assignment.
# -------------------------------------------------------------------------

training_count <- 400L
count_moment <- gate_g_expected_mean_error_covariance(
  sigma_true, training_count, geometry$profile_count
)
check_close(
  "reciprocal count moment agrees with direct binomial summation",
  count_moment$reciprocal_count_moment,
  sum(stats::dbinom(
    seq_len(training_count), training_count,
    1 / geometry$profile_count
  ) / seq_len(training_count))
)
check_true(
  "expected mean-error covariance is positive definite",
  min(eigen(count_moment$covariance, symmetric = TRUE)$values) > 0
)
expected_oracle_risk <- gate_g_expected_crossfit_scaled_variance(
  oracle$mixing, sigma_true, training_count, geometry$profile_count,
  task_weights
)
check_close(
  "CS/product count formula reproduces the independent 20k-MC prediction",
  expected_oracle_risk$total,
  0.03373712,
  tolerance = 1e-8
)
finite_g <- gate_g_count_aware_operator(
  sigma_true, training_count, geometry$profile_count, task_weights
)
check_close(
  "finite-G operator preserves factorial calibration",
  finite_g$calibration,
  task_weights
)
finite_g_risk <- gate_g_expected_crossfit_scaled_variance(
  finite_g$mixing, sigma_true, training_count, geometry$profile_count,
  task_weights
)
check_true(
  "finite-G count-aware operator minimizes risk relative to asymptotic H-star",
  finite_g_risk$total <= expected_oracle_risk$total + 1e-12
)
large_training_operator <- gate_g_count_aware_operator(
  sigma_true, 100000L, geometry$profile_count, task_weights
)
check_true(
  "finite-G operator converges to asymptotic H-star as training grows",
  max(abs(large_training_operator$mixing - oracle$mixing)) < 1e-5
)

# -------------------------------------------------------------------------
# 4. Full-sample oracle GLS is the finite-support conditional benchmark.
# -------------------------------------------------------------------------

gls <- gate_g_full_oracle_gls(
  data$profile_index, data$outcome, geometry, task_weights, sigma_true
)
check_true(
  "full oracle GLS identifies every saturated task-profile mean",
  gls$information_rank == tasks * geometry$profile_count
)
check_close(
  "full oracle GLS is conditionally centered for the factorial target",
  vapply(seq_len(ncol(geometry$riesz_lookup)), function(coordinate) {
    sum(gls$linear_load[, , coordinate] * data$mean)
  }, numeric(1)),
  geometry$truth,
  tolerance = 1e-9
)
check_close(
  "full oracle GLS estimate equals its exact outcome loading",
  gls$estimate,
  vapply(seq_len(ncol(geometry$riesz_lookup)), function(coordinate) {
    sum(gls$linear_load[, , coordinate] * data$outcome)
  }, numeric(1))
)
check_true(
  "finite-support GLS covariance is positive definite",
  min(eigen(gls$covariance, symmetric = TRUE)$values) > 0
)

# -------------------------------------------------------------------------
# 5. Honest two-way feasible GLS and saturated residual degrees of freedom.
# -------------------------------------------------------------------------

split_fold <- balanced_supported_cluster_folds(
  data$profile_index, 2L, 2026082804L, geometry$profile_count
)
covariance_half <- split_fold == 1L
df_fit <- gate_g_saturated_residual_df_covariances(
  data$profile_index[covariance_half, , drop = FALSE],
  data$outcome[covariance_half, , drop = FALSE],
  geometry, task_weights
)
check_close(
  "saturated same-task residual degrees of freedom equal n minus M",
  diag(df_fit$degrees_of_freedom),
  sum(covariance_half) - geometry$profile_count
)
profiles_half <- data$profile_index[covariance_half, , drop = FALSE]
n_half <- nrow(profiles_half)
one_hot <- function(values) {
  model.matrix(~ factor(values, levels = seq_len(geometry$profile_count)) - 1)
}
X1 <- one_hot(profiles_half[, 1])
X2 <- one_hot(profiles_half[, 2])
P1 <- X1 %*% solve(crossprod(X1), t(X1))
P2 <- X2 %*% solve(crossprod(X2), t(X2))
direct_df_12 <- sum(diag((diag(n_half) - P1) %*% (diag(n_half) - P2)))
check_close(
  "joint residual df count formula equals direct projection trace",
  df_fit$degrees_of_freedom[1, 2],
  direct_df_12,
  tolerance = 1e-9
)
split_df <- gate_g_split_gls(
  data$profile_index, data$outcome, split_fold,
  geometry, task_weights, "df_corrected"
)
check_close(
  "honest split GLS equals its realized outcome loading",
  split_df$estimate,
  vapply(seq_len(ncol(geometry$riesz_lookup)), function(coordinate) {
    sum(split_df$linear_load[, , coordinate] * data$outcome)
  }, numeric(1))
)
check_close(
  "honest split GLS is calibrated for the restricted target",
  vapply(seq_len(ncol(geometry$riesz_lookup)), function(coordinate) {
    sum(split_df$linear_load[, , coordinate] * data$mean)
  }, numeric(1)),
  geometry$truth,
  tolerance = 1e-9
)

rare_profile <- matrix(
  rep(rep(seq_len(geometry$profile_count), each = 2L), tasks),
  ncol = tasks
)
rare_rows <- which(rare_profile[, 1] == 1L)
rare_profile[rare_rows[[2L]], 1] <- 2L
set.seed(2026082805L)
rare_outcome <- matrix(
  stats::rnorm(nrow(rare_profile) * tasks), nrow(rare_profile), tasks
)
rare_covariance_fit <- gate_g_inner_covariance_fit(
  rare_profile, rare_outcome, geometry, task_weights,
  4L, 2026082806L
)
check_true(
  "singleton auxiliary cell invokes exact residual-df covariance fallback",
  identical(
    rare_covariance_fit$strategy,
    "full_residual_df_fallback"
  )
)
check_true(
  "residual-df covariance fallback remains positive definite",
  min(eigen(
    rare_covariance_fit$corrected$compound_symmetry$covariance,
    symmetric = TRUE
  )$values) > 0
)

# -------------------------------------------------------------------------
# 6. Gate-G configuration replaces, rather than merges, frozen scenarios.
# -------------------------------------------------------------------------

gate_g_config <- read_gate_g_config(file.path(
  project_root, "analysis", "feasible_eif", "gate-g-screen.yml"
))
expected_gate_g_ids <- c(
  "gate_g_null_g250", "gate_g_null_g500", "gate_g_null_g1000",
  "gate_g_lf_r1_g250", "gate_g_lf_r1_g1000"
)
check_close(
  "Gate-G reader loads exactly the preregistered replication count",
  gate_g_config$replications_per_scenario,
  1000
)
check_true(
  "Gate-G reader replaces the frozen E2 scenario registry",
  identical(
    vapply(gate_g_config$simulation$scenarios, `[[`, character(1), "id"),
    expected_gate_g_ids
  )
)
rare_support_replication <- run_gate_g_replication(
  gate_g_config$simulation$scenarios[[1]], 697L,
  gate_g_config, gate_d_geometry(gate_g_config)
)
check_true(
  "rare unsupported half-split uses the declared ordinary-score fallback",
  any(rare_support_replication$split_diagnostics$strategy ==
        "ordinary_support_fallback") &&
    is.null(rare_support_replication$error)
)

cat(sprintf("All %d Gate-G theory checks passed.\n", check_count))
