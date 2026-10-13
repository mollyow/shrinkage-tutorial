#!/usr/bin/env Rscript

script_path <- function() {
  flag <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (!length(flag)) {
    return(normalizePath("analysis/feasible_eif/test_feasible_estimator.R"))
  }
  normalizePath(sub("^--file=", "", flag[[1L]]), mustWork = TRUE)
}
script <- script_path()
analysis_dir <- dirname(script)
repo_root <- normalizePath(file.path(analysis_dir, "..", ".."), mustWork = TRUE)
correctness_dir <- file.path(repo_root, "results", "feasible-eif")
dir.create(correctness_dir, recursive = TRUE, showWarnings = FALSE)
correctness_path <- file.path(correctness_dir, "correctness-tests.yml")
if (file.exists(correctness_path)) unlink(correctness_path)
source(file.path(repo_root, "analysis", "covariance_headroom",
                 "calibration_solver.R"), local = TRUE)
source(file.path(analysis_dir, "feasible_estimator.R"), local = TRUE)
source(file.path(analysis_dir, "simulation_functions.R"), local = TRUE)

check_close <- function(observed, expected, tolerance = 1e-10, label = "check") {
  error <- max(abs(observed - expected))
  if (!is.finite(error) || error > tolerance) {
    stop(sprintf("%s failed: %.6g > %.6g", label, error, tolerance),
         call. = FALSE)
  }
  cat(sprintf("PASS %-50s %.3e\n", label, error))
  invisible(error)
}

config <- read_gate_d_config(file.path(analysis_dir, "gate-d-smoke.yml"))
geometry <- gate_d_geometry(config)
sigma <- matrix(0.6, 4, 4)
diag(sigma) <- 1
residual <- matrix(stats::rnorm(800), 200, 4) %*% chol(sigma)
covariance_set <- estimate_working_covariances(residual)

for (mechanism in c("product", "uniform_without_replacement")) {
  library <- make_operator_library(
    covariance_set, rep(0.25, 4), mechanism,
    if (mechanism == "uniform_without_replacement") 16 else NULL,
    c(0.25, 0.5, 0.75, 1)
  )
  calibration <- vapply(library, `[[`, numeric(1), "calibration_residual")
  check_close(calibration, rep(0, length(calibration)),
              label = paste("all blends calibrate under", mechanism))
}

# Exhaustive expectation with a deliberately wrong nuisance surface.
M <- 3L
T <- 2L
schedules <- as.matrix(expand.grid(rep(list(seq_len(M)), T)))
q_joint <- rep(1 / nrow(schedules), nrow(schedules))
riesz <- matrix(c(-sqrt(3 / 2), 0, sqrt(3 / 2)), ncol = 1)
invisible(validate_centered_riesz_lookup(riesz))
truth_mean <- matrix(c(-0.3, 0.2, 0.8, 0.5, -0.1, 0.4), M, T)
wrong_mean <- matrix(c(0.7, -0.2, 0.1, -0.5, 0.9, 0.2), M, T)
target <- mean(vapply(seq_len(T), function(tt) {
  mean(riesz[, 1] * truth_mean[, tt])
}, numeric(1)))
wrong_plugin <- mean(vapply(seq_len(T), function(tt) {
  mean(riesz[, 1] * wrong_mean[, tt])
}, numeric(1)))
wrong_covariance <- matrix(c(1.7, -0.35, -0.35, 0.8), 2, 2)
covariance_set_wrong <- list(
  compound_symmetry = stabilize_working_covariance(wrong_covariance),
  unstructured = stabilize_working_covariance(wrong_covariance)
)
library_wrong <- make_operator_library(
  covariance_set_wrong, c(0.5, 0.5), "product", blend_grid = c(0.5, 1)
)
for (candidate in names(library_wrong)) {
  expected_score <- 0
  for (s in seq_len(nrow(schedules))) {
    schedule <- schedules[s, ]
    e <- vapply(seq_len(T), function(tt) {
      truth_mean[schedule[[tt]], tt] - wrong_mean[schedule[[tt]], tt]
    }, numeric(1))
    F <- riesz[schedule, , drop = FALSE]
    expected_score <- expected_score + q_joint[[s]] *
      (wrong_plugin + drop(crossprod(
        apply_task_mixing(library_wrong[[candidate]], F), e
      )))
  }
  check_close(expected_score, target,
              label = paste("wrong-nuisance expectation", candidate))
}

# Orthogonal coordinate rotations preserve the score and selector loss.
set.seed(20260827)
n <- 60L
profile <- matrix(sample.int(16, n * 4, replace = TRUE), n, 4)
e <- matrix(rnorm(n * 4), n, 4)
operator <- make_operator_library(
  covariance_set, rep(0.25, 4), "product", blend_grid = c(0.5, 1)
)$compound_symmetry_tau_0.5
base_score <- score_factorial_clusters(
  e, profile, geometry$riesz_lookup, operator, geometry$truth
)
rotation_raw <- qr.Q(qr(matrix(rnorm(16), 4, 4)))
rotated_lookup <- geometry$riesz_lookup %*% rotation_raw
rotated_score <- score_factorial_clusters(
  e, profile, rotated_lookup, operator,
  drop(crossprod(rotation_raw, geometry$truth))
)
check_close(rotated_score, base_score %*% rotation_raw,
            label = "orthogonal score rotation")
check_close(weighted_score_summary(rotated_score)$loss,
            weighted_score_summary(base_score)$loss,
            label = "rotation-invariant selector loss")

# A named level permutation changes only lookup indices, not scores.
permutation <- sample.int(16)
inverse_permutation <- match(seq_len(16), permutation)
permuted_lookup <- geometry$riesz_lookup[permutation, , drop = FALSE]
permuted_profile <- matrix(inverse_permutation[profile], n, 4)
permuted_score <- score_factorial_clusters(
  e, permuted_profile, permuted_lookup, operator, geometry$truth
)
check_close(permuted_score, base_score, label = "profile-level permutation")

# Reference-level changes in a saturated nuisance regression preserve fitted
# profile means and therefore preserve direct target integration.
set.seed(17)
f <- factor(rep(c("a", "b", "c"), each = 20), levels = c("a", "b", "c"))
y <- c(rnorm(20, -0.2), rnorm(20, 0.4), rnorm(20, 0.9))
fit_a <- lm(y ~ f)
f_c <- stats::relevel(f, ref = "c")
fit_c <- lm(y ~ f_c)
target_levels <- data.frame(f = factor(c("a", "b", "c"),
                                        levels = c("a", "b", "c")))
pred_a <- predict(fit_a, target_levels)
target_levels_c <- data.frame(f_c = factor(c("a", "b", "c"),
                                            levels = c("a", "b", "c")))
pred_c <- predict(fit_c, target_levels_c)
check_close(unname(pred_a), unname(pred_c), label = "reference-baseline swap")

# Jointly permuting task labels, outcomes, assignments, weights, and covariance
# preserves the cluster score.
task_permutation <- c(3, 1, 4, 2)
permuted_operator <- operator
permuted_operator$mixing <- operator$mixing[task_permutation, task_permutation]
task_score <- score_factorial_clusters(e, profile, geometry$riesz_lookup, operator)
task_score_permuted <- score_factorial_clusters(
  e[, task_permutation], profile[, task_permutation], geometry$riesz_lookup,
  permuted_operator
)
check_close(task_score_permuted, task_score, label = "joint task permutation")

# Outcome/covariance scale and survey-weight normalization invariance.
tiny <- estimate_working_covariances(1e-6 * residual)
large <- estimate_working_covariances(1e6 * residual)
tiny_operator <- make_operator_library(
  tiny, rep(0.25, 4), "product", blend_grid = 1
)$compound_symmetry_tau_1
large_operator <- make_operator_library(
  large, rep(0.25, 4), "product", blend_grid = 1
)$compound_symmetry_tau_1
check_close(tiny_operator$mixing, large_operator$mixing,
            label = "learned-operator outcome-scale invariance")
weights <- runif(n, 0.3, 2)
summary_one <- weighted_score_summary(base_score, weights)
summary_scaled <- weighted_score_summary(base_score, 17 * weights)
check_close(summary_one$estimate, summary_scaled$estimate,
            label = "survey-weight scale point invariance")
check_close(summary_one$covariance, summary_scaled$covariance,
            label = "survey-weight scale covariance invariance")

# Full smoke end-to-end and cluster-fold isolation.
scenario <- config$simulation$scenarios[[1L]]
replication <- run_gate_d_replication(scenario, 1L, config, geometry)
if (!is.null(replication$error) || any(!is.finite(replication$results$estimate)) ||
    any(!is.finite(replication$results$standard_error))) {
  stop("The end-to-end feasible estimator smoke test failed.", call. = FALSE)
}
if (nrow(replication$selector) != config$estimator$outer_folds ||
    anyDuplicated(replication$selector$outer_fold)) {
  stop("Outer respondent folds were not evaluated exactly once.", call. = FALSE)
}
if (max(replication$results$calibration_residual) > 1e-10) {
  stop("End-to-end calibration exceeds tolerance.", call. = FALSE)
}
cat("PASS end-to-end outer-fold isolation and calibration\n")
if (!requireNamespace("yaml", quietly = TRUE)) {
  stop("Package `yaml` is required to record the correctness result.", call. = FALSE)
}
yaml::write_yaml(list(
  passed = TRUE,
  calibration_tolerance = 1e-10,
  invariance_tolerance = 1e-10,
  completed_at_utc = format(Sys.time(), tz = "UTC", usetz = TRUE),
  solver_md5 = unname(tools::md5sum(file.path(
    repo_root, "analysis", "covariance_headroom", "calibration_solver.R"
  ))),
  feasible_estimator_md5 = unname(tools::md5sum(file.path(
    analysis_dir, "feasible_estimator.R"
  )))
), correctness_path)
cat("All feasible-estimator tests passed.\n")
