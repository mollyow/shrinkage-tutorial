#!/usr/bin/env Rscript

script_path <- function() {
  flag <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (!length(flag)) return(normalizePath("analysis/covariance_headroom/test_calibration_solver.R"))
  normalizePath(sub("^--file=", "", flag[[1L]]), mustWork = TRUE)
}
source(file.path(dirname(script_path()), "calibration_solver.R"), local = TRUE)

check_close <- function(observed, expected, tolerance = 1e-10, label = "check") {
  error <- max(abs(observed - expected))
  if (!is.finite(error) || error > tolerance) {
    stop(sprintf("%s failed: maximum error %.6g exceeds %.6g.",
                 label, error, tolerance), call. = FALSE)
  }
  cat(sprintf("PASS %-46s max error %.3e\n", label, error))
  invisible(error)
}

enumerated_design <- function(schedules, profile_count) {
  task_count <- ncol(schedules)
  lapply(seq_len(nrow(schedules)), function(i) {
    out <- matrix(0, task_count, task_count * profile_count)
    for (tt in seq_len(task_count)) {
      out[tt, (tt - 1L) * profile_count + schedules[i, tt]] <- 1
    }
    out
  })
}

target_matrix <- function(direct_weights, task_weights) {
  direct_weights <- as.matrix(direct_weights)
  # direct_weights is M x d; the finite solver expects d x (T*M).
  do.call(cbind, lapply(task_weights, function(w) w * t(direct_weights)))
}

compare_enumerated_to_mixing <- function(
    profile_count, task_count, covariance, direct_weights,
    mechanism = c("product", "uniform_without_replacement")) {
  mechanism <- match.arg(mechanism)
  schedules <- as.matrix(expand.grid(rep(list(seq_len(profile_count)), task_count)))
  if (identical(mechanism, "uniform_without_replacement")) {
    schedules <- schedules[apply(schedules, 1L, function(x) !anyDuplicated(x)),,
                           drop = FALSE]
  }
  probability <- rep(1 / nrow(schedules), nrow(schedules))
  B <- enumerated_design(schedules, profile_count)
  task_weights <- rep(1 / task_count, task_count)
  L <- target_matrix(direct_weights, task_weights)
  finite <- solve_finite_support_calibration(B, covariance, probability, L)
  mixing <- solve_task_mixing(
    covariance, task_weights, mechanism,
    support_size = if (identical(mechanism, "uniform_without_replacement")) {
      profile_count
    } else NULL
  )
  q <- 1 / profile_count
  riesz <- as.matrix(direct_weights) / q
  expected <- lapply(seq_len(nrow(schedules)), function(i) {
    apply_task_mixing(mixing, riesz[schedules[i, ], , drop = FALSE])
  })
  check_close(
    unlist(finite$weights), unlist(expected), 2e-10,
    sprintf("finite enumeration vs %s solver", mechanism)
  )
  gram <- crossprod(riesz) / profile_count
  expected_bound <- mixing$optimal_multiplier * gram
  check_close(finite$bound, expected_bound, 2e-10,
              sprintf("enumerated vs %s bound", mechanism))
  invisible(list(finite = finite, mixing = mixing))
}

# T=1 reduces to the ordinary direct-weight/randomization-probability score.
ell_binary <- matrix(c(-1, 1), ncol = 1)
B_one <- list(matrix(c(1, 0), 1, 2), matrix(c(0, 1), 1, 2))
one <- solve_finite_support_calibration(
  B_one, matrix(2.5, 1, 1), c(0.5, 0.5), t(ell_binary)
)
check_close(vapply(one$weights, `[`, numeric(1), 1, 1), c(-2, 2),
            label = "T=1 ordinary Riesz score")

# Product and SRSWOR closed forms agree with exhaustive finite-state solutions.
sigma_two <- matrix(c(1.0, 0.55, 0.55, 1.4), 2, 2)
compare_enumerated_to_mixing(2, 2, sigma_two, ell_binary, "product")
ell_three <- cbind(c(-1, 0, 1), c(1, -2, 1))
compare_enumerated_to_mixing(3, 2, sigma_two, ell_three,
                             "uniform_without_replacement")
compare_enumerated_to_mixing(2, 2, sigma_two, ell_binary,
                             "uniform_without_replacement")

# Diagonal covariance cannot improve on ordinary taskwise scoring.
diagonal <- solve_task_mixing(diag(c(0.8, 1.2, 2.0)), c(0.2, 0.3, 0.5),
                              "product")
check_close(diagonal$mixing, diag(c(0.2, 0.3, 0.5)),
            label = "diagonal covariance returns ordinary score")
check_close(diagonal$variance_ratio, 1,
            label = "diagonal covariance has no headroom")

# In the balanced two-task unit-variance example the exact gain is 1-rho^2.
rho <- 0.63
correlated <- solve_task_mixing(matrix(c(1, rho, rho, 1), 2, 2),
                                c(0.5, 0.5), "product")
check_close(correlated$variance_ratio, 1 - rho^2,
            label = "two-task analytic variance ratio")

# Scaling a covariance matrix cannot change its optimal mixing weights or ratio.
tiny_scale <- solve_task_mixing(1e-12 * matrix(c(1, rho, rho, 1), 2, 2),
                                c(0.5, 0.5), "product")
large_scale <- solve_task_mixing(1e12 * matrix(c(1, rho, rho, 1), 2, 2),
                                 c(0.5, 0.5), "product")
check_close(tiny_scale$mixing, correlated$mixing,
            label = "small covariance-scale invariance")
check_close(large_scale$mixing, correlated$mixing,
            label = "large covariance-scale invariance")

# The closed form is intentionally restricted to centered profile functions.
validate_centered_riesz_lookup(cbind(c(-1, 1), c(1, -1)))
noncentered_error <- tryCatch({
  validate_centered_riesz_lookup(matrix(c(0, 1), ncol = 1))
  NULL
}, error = identity)
if (is.null(noncentered_error) ||
    !grepl("target-centered", conditionMessage(noncentered_error))) {
  stop("A noncentered Riesz lookup was not rejected.", call. = FALSE)
}
cat("PASS noncentered scalable Riesz lookup fails closed\n")

# Target rotations transform scores and bounds by congruence.
angle <- 0.37
rotation <- matrix(c(cos(angle), -sin(angle), sin(angle), cos(angle)), 2, 2)
schedules <- as.matrix(expand.grid(rep(list(1:3), 2)))
B_rotation <- enumerated_design(schedules, 3)
L_rotation <- target_matrix(ell_three, c(0.5, 0.5))
base <- solve_finite_support_calibration(
  B_rotation, sigma_two, rep(1 / nrow(schedules), nrow(schedules)), L_rotation
)
rotated <- solve_finite_support_calibration(
  B_rotation, sigma_two, rep(1 / nrow(schedules), nrow(schedules)),
  rotation %*% L_rotation
)
expected_rotated_weights <- lapply(base$weights, function(x) x %*% t(rotation))
check_close(unlist(rotated$weights), unlist(expected_rotated_weights),
            label = "orthogonal target rotation equivariance")
check_close(rotated$bound, rotation %*% base$bound %*% t(rotation),
            label = "rotated bound congruence")

# Calibration gives exact expectation for an arbitrary, deliberately wrong
# independently trained nuisance vector.
set.seed(20260827)
truth <- rnorm(ncol(B_rotation[[1L]]))
nuisance <- rnorm(length(truth))
plugin <- drop(L_rotation %*% nuisance)
expected_score <- plugin
for (s in seq_along(B_rotation)) {
  expected_score <- expected_score +
    drop(crossprod(base$weights[[s]], B_rotation[[s]] %*% (truth - nuisance))) /
      length(B_rotation)
}
check_close(expected_score, drop(L_rotation %*% truth),
            label = "exact unbiasedness with wrong nuisance")

# Nonidentified target directions fail closed.
support_error <- tryCatch({
  solve_finite_support_calibration(
    list(matrix(c(1, 0), 1, 2)), matrix(1, 1, 1), 1,
    matrix(c(0, 1), 1, 2)
  )
  NULL
}, error = identity)
if (is.null(support_error) || !grepl("not identified", conditionMessage(support_error))) {
  stop("The rank-deficient-support test did not fail as required.", call. = FALSE)
}
cat("PASS rank-deficient support fails closed\n")

# The candidate-scale closed form is tiny and allocates no joint schedule array.
candidate_scale <- solve_task_mixing(
  matrix(0.08, 8, 8) + diag(0.02, 8), rep(1 / 8, 8),
  "uniform_without_replacement", support_size = 1440
)
check_close(candidate_scale$calibration, rep(1 / 8, 8),
            label = "candidate-scale exact calibration")
if (object.size(candidate_scale) > 1e6) {
  stop("Candidate-scale task solver unexpectedly exceeds one megabyte.", call. = FALSE)
}
cat(sprintf("PASS candidate-scale solver object size %d bytes\n",
            as.integer(object.size(candidate_scale))))

cat("All calibration-solver tests passed.\n")
