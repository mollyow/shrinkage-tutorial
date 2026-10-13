#!/usr/bin/env Rscript

# Exhaustive finite-support checks for the Gate E1 carryover-sensitivity
# identities.  This is an algebra test, not a Monte Carlo experiment.

script_path <- function() {
  flag <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (!length(flag)) {
    return(normalizePath("analysis/feasible_eif/check_gate_e1_theory.R"))
  }
  normalizePath(sub("^--file=", "", flag[[1L]]), mustWork = TRUE)
}

source(
  file.path(dirname(script_path()), "..", "covariance_headroom",
            "calibration_solver.R"),
  local = TRUE
)

tolerance <- 2e-10
errors <- numeric()

check_close <- function(observed, expected, label, tol = tolerance) {
  error <- max(abs(observed - expected))
  if (!is.finite(error) || error > tol) {
    stop(sprintf("%s failed: error %.6g exceeds %.6g.", label, error, tol),
         call. = FALSE)
  }
  errors[[label]] <<- error
  cat(sprintf("PASS %-49s %.3e\n", label, error))
  invisible(error)
}

check_true <- function(value, label) {
  if (!isTRUE(value)) stop(sprintf("%s failed.", label), call. = FALSE)
  errors[[label]] <<- 0
  cat(sprintf("PASS %s\n", label))
  invisible(TRUE)
}

weighted_sum <- function(items, probability) {
  Reduce(`+`, Map(function(item, p) p * item, items, probability))
}

weighted_cross_variance <- function(weights, covariance, probability) {
  weighted_sum(Map(function(h, s) crossprod(h, s %*% h),
                   weights, covariance), probability)
}

weighted_bias <- function(weight_difference, departure, probability) {
  drop(weighted_sum(Map(function(d, c) crossprod(d, c),
                        weight_difference, departure), probability))
}

states <- as.matrix(expand.grid(a1 = c(-1, 1), a2 = c(-1, 1)))
state_probability <- rep(1 / nrow(states), nrow(states))
task_count <- 2L

# Task-specific intercept/effect surfaces.  The two target coordinates are the
# average and difference of the task-specific factorial effects.
B <- lapply(seq_len(nrow(states)), function(s) {
  a <- states[s, ]
  rbind(c(1, a[[1L]], 0, 0), c(0, 0, 1, a[[2L]]))
})
L <- rbind(
  average = c(0, 0.5, 0, 0.5),
  difference = c(0, 0.5, 0, -0.5)
)
Sigma <- matrix(c(1.0, 0.63, 0.63, 1.35), 2, 2)
covariance <- rep(list(Sigma), nrow(states))

efficient <- solve_finite_support_calibration(
  B, covariance, state_probability, L
)
H_star <- efficient$weights
H_zero <- lapply(seq_len(nrow(states)), function(s) {
  a <- states[s, ]
  rbind(c(a[[1L]] / 2, a[[1L]] / 2),
        c(a[[2L]] / 2, -a[[2L]] / 2))
})
D <- Map(`-`, H_star, H_zero)

calibrate <- function(H) {
  weighted_sum(Map(function(b, h) crossprod(b, h), B, H),
               state_probability)
}
check_close(calibrate(H_zero), t(L), "ordinary exact calibration")
check_close(calibrate(H_star), t(L), "efficient exact calibration")
check_close(calibrate(D), matrix(0, ncol(L), nrow(L)),
            "difference has zero calibration")

# Exact score-bias identity with both a wrong own-task nuisance and genuine
# cross-task mean carryover.
gamma <- c(0.20, 0.45, -0.10, 0.72)
eta <- c(-0.15, 0.05, 0.31, -0.20)
kappa <- 0.28
carryover <- lapply(seq_len(nrow(states)), function(s) {
  a <- states[s, ]
  c(kappa * a[[2L]] + 0.11 * a[[1L]] * a[[2L]],
    kappa * a[[1L]] - 0.07 * a[[1L]] * a[[2L]])
})
mean_truth <- Map(function(b, c) drop(b %*% gamma) + c, B, carryover)

expected_score <- function(H, nuisance) {
  plugin <- drop(L %*% nuisance)
  plugin + drop(weighted_sum(Map(function(h, m, b) {
    crossprod(h, m - drop(b %*% nuisance))
  }, H, mean_truth, B), state_probability))
}

robust_target <- drop(weighted_sum(Map(function(h, m) crossprod(h, m),
                                       H_zero, mean_truth),
                                   state_probability))
bias <- weighted_bias(D, carryover, state_probability)
check_close(expected_score(H_zero, eta), robust_target,
            "ordinary score remains centered")
check_close(expected_score(H_star, eta) - robust_target, bias,
            "exact adaptive carryover bias")

# The bias depends only on the equivalence class modulo the no-carryover mean
# space: adding Bv cannot change it.
set.seed(20260827)
own_task_shift <- rnorm(ncol(B[[1L]]))
shifted_carryover <- Map(function(c, b) c + drop(b %*% own_task_shift),
                         carryover, B)
check_close(weighted_bias(D, shifted_carryover, state_probability), bias,
            "quotient invariance modulo no-carryover means")

# Sharp residual-standardized sensitivity ellipsoid.  M=Sigma^{-1}; epsilon is
# the per-task RMS Mahalanobis distance from the no-carryover mean space.
M <- solve(Sigma)
information_M <- weighted_sum(lapply(B, function(b) crossprod(b, M %*% b)),
                              state_probability)
rhs_M <- drop(weighted_sum(Map(function(b, c) crossprod(b, M %*% c),
                               B, carryover), state_probability))
projection_coefficient <- drop(solve(information_M, rhs_M))
carryover_perp <- Map(function(c, b) c - drop(b %*% projection_coefficient),
                      carryover, B)
orthogonality <- weighted_sum(Map(function(b, c) crossprod(b, M %*% c),
                                  B, carryover_perp), state_probability)
check_close(orthogonality, rep(0, ncol(B[[1L]])),
            "metric projection orthogonality")

epsilon_squared <- drop(weighted_sum(lapply(carryover_perp, function(c) {
  crossprod(c, M %*% c)
}), state_probability)) / task_count
K <- task_count * weighted_sum(lapply(D, function(d) {
  crossprod(d, Sigma %*% d)
}), state_probability)
K_inverse <- symmetric_pseudoinverse(K)$inverse
ellipsoid_ratio <- drop(crossprod(bias, K_inverse %*% bias)) / epsilon_squared
check_true(ellipsoid_ratio <= 1 + 1e-10,
           "arbitrary carryover lies inside sharp bias ellipsoid")

# A least-favourable departure attains the directional support bound and the
# boundary of the vector ellipsoid.
direction <- c(0.6, -0.8)
sensitivity_radius <- 0.37
direction_scale <- sqrt(drop(crossprod(direction, K %*% direction)))
sharp_carryover <- lapply(D, function(d) {
  drop(sensitivity_radius * task_count * Sigma %*% d %*% direction /
         direction_scale)
})
sharp_norm <- drop(weighted_sum(lapply(sharp_carryover, function(c) {
  crossprod(c, M %*% c)
}), state_probability)) / task_count
sharp_bias <- weighted_bias(D, sharp_carryover, state_probability)
check_close(sqrt(sharp_norm), sensitivity_radius,
            "least-favourable departure radius")
check_close(drop(crossprod(direction, sharp_bias)),
            sensitivity_radius * direction_scale,
            "sharp directional bias bound")
check_close(drop(crossprod(sharp_bias, K_inverse %*% sharp_bias)),
            sensitivity_radius^2,
            "sharp vector bias-ellipsoid boundary")

# Efficiency gain and carryover vulnerability are the same quadratic form.
V_zero <- weighted_cross_variance(H_zero, covariance, state_probability)
V_star <- weighted_cross_variance(H_star, covariance, state_probability)
variance_gap <- V_zero - V_star
projection_cross_term <- weighted_sum(Map(function(d, h, s) {
  crossprod(d, s %*% h)
}, D, H_star, covariance), state_probability)
check_close(projection_cross_term, matrix(0, nrow(L), nrow(L)),
            "efficient projection orthogonality")
check_close(K / task_count, variance_gap,
            "efficiency-gain vulnerability duality")
cross_covariance <- weighted_sum(Map(function(hs, h0, s) {
  crossprod(hs, s %*% h0)
}, H_star, H_zero, covariance), state_probability)
check_close(cross_covariance, V_star,
            "efficient-ordinary cross covariance")

# A normalized respondent population tilt enters bias linearly and the
# sensitivity/variance operators quadratically.
x_probability <- c(0.4, 0.6)
population_tilt <- c(0.65, 1.25)
population_tilt <- population_tilt /
  sum(x_probability * population_tilt)
weighted_second_moment <- sum(x_probability * population_tilt^2)
K_population <- weighted_second_moment * K
variance_gap_population <- weighted_second_moment * variance_gap
check_close(K_population / task_count, variance_gap_population,
            "survey-tilted efficiency-vulnerability duality")
check_close(sum(x_probability * population_tilt) * bias, bias,
            "survey tilt enters exact bias linearly")

# Every calibration-preserving blend inherits exact linear bias, quadratic
# vulnerability, and the projection variance formula.
tau <- 0.42
H_tau <- Map(function(h0, hs) h0 + tau * (hs - h0), H_zero, H_star)
D_tau <- Map(`-`, H_tau, H_zero)
K_tau <- task_count * weighted_sum(lapply(D_tau, function(d) {
  crossprod(d, Sigma %*% d)
}), state_probability)
V_tau <- weighted_cross_variance(H_tau, covariance, state_probability)
check_close(weighted_bias(D_tau, carryover, state_probability), tau * bias,
            "blend bias scales linearly")
check_close(K_tau, tau^2 * K, "blend vulnerability scales quadratically")
check_close(V_tau, V_star + (1 - tau)^2 * variance_gap,
            "blend variance projection identity")

rho_local <- 0.44
tau_formula <- 1 / (1 + task_count * rho_local^2)
tau_numeric <- optimize(
  function(x) (1 - x)^2 + task_count * rho_local^2 * x^2,
  interval = c(0, 1)
)$minimum
check_close(tau_numeric, tau_formula, "universal minimax blend", tol = 2e-5)

# A calibrated but nonprojected feasible weight has a variance cross term.  Its
# sensitivity-indexed blend must use the generic quadratic formula rather than
# the oracle closed form.
set.seed(20260828)
raw_null <- lapply(seq_len(nrow(states)), function(s) {
  matrix(rnorm(task_count * nrow(L)), task_count, nrow(L))
})
raw_calibration <- weighted_sum(Map(function(b, h) crossprod(b, h),
                                    B, raw_null), state_probability)
calibration_projection <- solve_finite_support_calibration(
  B, covariance, state_probability, t(raw_calibration)
)$weights
null_perturbation <- Map(`-`, raw_null, calibration_projection)
H_generic <- Map(function(hs, g) hs + 0.18 * g, H_star, null_perturbation)
D_generic <- Map(`-`, H_generic, H_zero)
check_close(calibrate(H_generic), t(L),
            "generic feasible weight remains calibrated")
generic_direction <- c(1, 0)
V_D_generic <- weighted_cross_variance(D_generic, covariance,
                                       state_probability)
C_0D_generic <- weighted_sum(Map(function(h0, d, s) {
  crossprod(h0, s %*% d)
}, H_zero, D_generic, covariance), state_probability)
K_generic <- task_count * V_D_generic
generic_cross <- drop(crossprod(
  generic_direction, C_0D_generic %*% generic_direction
))
generic_difference_variance <- drop(crossprod(
  generic_direction, V_D_generic %*% generic_direction
))
generic_vulnerability <- drop(crossprod(
  generic_direction, K_generic %*% generic_direction
))
rho_generic <- 0.41
tau_generic <- max(0, min(1, -generic_cross /
  (generic_difference_variance + rho_generic^2 * generic_vulnerability)))
tau_generic_numeric <- optimize(
  function(x) 2 * x * generic_cross +
    x^2 * (generic_difference_variance +
             rho_generic^2 * generic_vulnerability),
  interval = c(0, 1)
)$minimum
check_close(tau_generic_numeric, tau_generic,
            "generic feasible minimax blend with cross term", tol = 2e-5)

# Orthogonal changes of the reported factorial family rotate the bias and
# sensitivity operator but cannot change their substantive content.
angle <- 0.39
Q <- matrix(c(cos(angle), -sin(angle), sin(angle), cos(angle)), 2, 2)
D_rotated <- lapply(D, function(d) d %*% Q)
K_rotated <- task_count * weighted_sum(lapply(D_rotated, function(d) {
  crossprod(d, Sigma %*% d)
}), state_probability)
bias_rotated <- weighted_bias(D_rotated, carryover, state_probability)
check_close(K_rotated, t(Q) %*% K %*% Q,
            "factorial rotation congruence")
check_close(bias_rotated, drop(t(Q) %*% bias),
            "factorial rotation of bias")

# An arbitrary invertible recoding of the no-carryover mean parameters leaves
# the score weights unchanged.  An arbitrary invertible output transformation
# acts equivariantly on the complete bias ellipsoid.
C_parameter <- matrix(c(1.0, 0.2, -0.1, 0.0,
                        0.1, 1.1, 0.0, 0.2,
                        0.0, -0.1, 0.9, 0.1,
                        0.2, 0.0, 0.1, 1.2), 4, 4, byrow = TRUE)
B_recoded <- lapply(B, function(b) b %*% C_parameter)
L_recoded <- L %*% C_parameter
recoded <- solve_finite_support_calibration(
  B_recoded, covariance, state_probability, L_recoded
)
check_close(unlist(recoded$weights), unlist(H_star),
            "invariant nuisance-parameter recoding")
check_close(recoded$bound, V_star,
            "recoded target has identical bound")

R_output <- matrix(c(1.2, -0.3, 0.25, 0.85), 2, 2, byrow = TRUE)
output_recoded <- solve_finite_support_calibration(
  B, covariance, state_probability, R_output %*% L
)
H_output_expected <- lapply(H_star, function(h) h %*% t(R_output))
D_output <- lapply(D, function(d) d %*% t(R_output))
K_output <- task_count * weighted_sum(lapply(D_output, function(d) {
  crossprod(d, Sigma %*% d)
}), state_probability)
bias_output <- weighted_bias(D_output, carryover, state_probability)
check_close(unlist(output_recoded$weights), unlist(H_output_expected),
            "general target-coordinate equivariance")
check_close(K_output, R_output %*% K %*% t(R_output),
            "general bias-operator congruence")
check_close(bias_output, drop(R_output %*% bias),
            "general bias-vector equivariance")
check_close(
  drop(crossprod(bias_output, solve(K_output, bias_output))),
  drop(crossprod(bias, solve(K, bias))),
  "bias-ellipsoid gauge invariance"
)

state_permutation <- c(4L, 2L, 1L, 3L)
state_reordered <- solve_finite_support_calibration(
  B[state_permutation], covariance[state_permutation],
  state_probability[state_permutation], L
)
check_close(unlist(state_reordered$weights),
            unlist(H_star[state_permutation]),
            "joint-state ordering invariance")

# A nonzero carryover direction can be invisible to the paired Hausman
# contrast.  Construct one in the metric-orthogonal carryover space and remove
# its projection onto the finite-dimensional bias representer.
random_departure <- lapply(seq_len(nrow(states)), function(s) rnorm(task_count))
rhs_random <- drop(weighted_sum(Map(function(b, c) crossprod(b, M %*% c),
                                    B, random_departure), state_probability))
coef_random <- drop(solve(information_M, rhs_random))
random_perp <- Map(function(c, b) c - drop(b %*% coef_random),
                   random_departure, B)
random_bias <- weighted_bias(D, random_perp, state_probability)
represented <- lapply(D, function(d) {
  drop(task_count * Sigma %*% d %*% K_inverse %*% random_bias)
})
blind_departure <- Map(`-`, random_perp, represented)
blind_bias <- weighted_bias(D, blind_departure, state_probability)
blind_norm <- drop(weighted_sum(lapply(blind_departure, function(c) {
  crossprod(c, M %*% c)
}), state_probability)) / task_count
check_close(blind_bias, rep(0, nrow(L)),
            "Hausman contrast has carryover blind directions")
check_true(blind_norm > 1e-5, "blind carryover direction is nonzero")

# Structured lag-only sensitivity.  Task 2 may depend on task 1's profile;
# task 1 cannot depend on a future assignment.  Residualization is in the same
# M metric used to define the departure radius.
residualize_departure_design <- function(Z) {
  cross_BZ <- weighted_sum(Map(function(b, z) crossprod(b, M %*% z),
                               B, Z), state_probability)
  projection <- solve(information_M, cross_BZ)
  Z_perp <- Map(function(z, b) z - b %*% projection, Z, B)
  Q_Z <- weighted_sum(lapply(Z_perp, function(z) crossprod(z, M %*% z)),
                      state_probability) / task_count
  R_Z <- weighted_sum(Map(function(d, z) crossprod(d, z), D, Z_perp),
                      state_probability)
  Q_inverse <- symmetric_pseudoinverse(Q_Z)$inverse
  list(
    Z_perp = Z_perp,
    Q = Q_Z,
    R = R_Z,
    K = R_Z %*% Q_inverse %*% t(R_Z),
    range_residual = max(abs(R_Z - R_Z %*% Q_inverse %*% Q_Z))
  )
}

Z_lag <- lapply(seq_len(nrow(states)), function(s) {
  matrix(c(0, states[s, 1L]), ncol = 1)
})
lag_geometry <- residualize_departure_design(Z_lag)
check_close(lag_geometry$range_residual, 0,
            "structured carryover range condition")
check_true(min(eigen(K - lag_geometry$K, symmetric = TRUE)$values) > -1e-10,
           "lag-only vulnerability is bounded by unrestricted vulnerability")
alpha_lag <- 0.31
lag_departure <- lapply(lag_geometry$Z_perp,
                        function(z) drop(z * alpha_lag))
check_close(weighted_bias(D, lag_departure, state_probability),
            drop(lag_geometry$R * alpha_lag),
            "structured carryover exact bias")
lag_radius_squared <- drop(alpha_lag^2 * lag_geometry$Q)
lag_bias <- weighted_bias(D, lag_departure, state_probability)
lag_gauge <- drop(crossprod(
  lag_bias, symmetric_pseudoinverse(lag_geometry$K)$inverse %*% lag_bias
))
check_close(lag_gauge, lag_radius_squared,
            "structured carryover bias boundary")

# A saturated departure design recovers the unrestricted sensitivity operator,
# including its factor-T scaling.
saturated_dimension <- nrow(states) * task_count
Z_saturated <- lapply(seq_len(nrow(states)), function(s) {
  z <- matrix(0, task_count, saturated_dimension)
  columns <- (s - 1L) * task_count + seq_len(task_count)
  z[cbind(seq_len(task_count), columns)] <- 1
  z
})
saturated_geometry <- residualize_departure_design(Z_saturated)
check_close(saturated_geometry$range_residual, 0,
            "saturated carryover range condition")
check_close(saturated_geometry$K, K,
            "saturated subspace recovers unrestricted operator")

structured_direction <- c(1, 0)
A_direction <- drop(crossprod(
  structured_direction, variance_gap %*% structured_direction
))
B_direction <- drop(crossprod(
  structured_direction, lag_geometry$K %*% structured_direction
))
check_true(A_direction > 0 && B_direction > 0,
           "structured blend direction is informative")
rho_structured <- 0.52
tau_structured <- A_direction /
  (A_direction + rho_structured^2 * B_direction)
tau_structured_numeric <- optimize(
  function(x) (1 - x)^2 * A_direction +
    rho_structured^2 * x^2 * B_direction,
  interval = c(0, 1)
)$minimum
check_close(tau_structured_numeric, tau_structured,
            "direction-specific structured minimax blend", tol = 2e-5)
check_true(abs(tau_structured -
                 1 / (1 + task_count * rho_structured^2)) > 1e-4,
           "structured blend differs from unrestricted formula")

# Closed-form two-task check from the memo.
rho <- 0.71
Sigma_cs <- matrix(c(1, rho, rho, 1), 2, 2)
scalar <- solve_finite_support_calibration(
  B, Sigma_cs, state_probability, L[1, , drop = FALSE]
)
H_scalar_zero <- lapply(seq_len(nrow(states)), function(s) {
  matrix(states[s, ] / 2, ncol = 1)
})
H_scalar_expected <- lapply(seq_len(nrow(states)), function(s) {
  a <- states[s, ]
  matrix(c((a[[1L]] - rho * a[[2L]]) / 2,
           (a[[2L]] - rho * a[[1L]]) / 2), ncol = 1)
})
check_close(unlist(scalar$weights), unlist(H_scalar_expected),
            "two-task efficient-weight formula")
cross_departure <- lapply(seq_len(nrow(states)), function(s) {
  kappa * rev(states[s, ])
})
scalar_difference <- Map(`-`, scalar$weights, H_scalar_zero)
check_close(weighted_bias(scalar_difference, cross_departure,
                          state_probability),
            -rho * kappa, "two-task analytic carryover bias")

# A rank-one target family makes the range qualification in the family tipping
# formula observable: in-range vectors have finite gauge, while an orthogonal
# component cannot be explained by any finite carryover radius.
D_rank_one <- lapply(scalar_difference, function(d) cbind(d, 2 * d))
K_rank_one <- task_count * weighted_sum(lapply(D_rank_one, function(d) {
  crossprod(d, Sigma_cs %*% d)
}), state_probability)
K_rank_inverse <- symmetric_pseudoinverse(K_rank_one)$inverse
theta_in_range <- c(0.4, 0.8)
theta_outside <- c(0.8, -0.4)
check_close(
  K_rank_one %*% K_rank_inverse %*% theta_in_range,
  theta_in_range,
  "family tipping finite on bias-operator range"
)
outside_projection <- drop(
  theta_outside - K_rank_one %*% K_rank_inverse %*% theta_outside
)
check_true(sqrt(sum(outside_projection^2)) > 0.5,
           "family tipping infinite outside bias-operator range")

# If the analyst calibrates under the wrong joint assignment law, exact
# nuisance robustness is lost.  The direct enumeration matches the algebraic
# calibration-error formula.
true_probability <- c(0.35, 0.15, 0.15, 0.35)
C_true <- weighted_sum(Map(function(b, h) crossprod(b, h), B, H_star),
                       true_probability)
wrong_law_score <- drop(L %*% eta) + drop(weighted_sum(
  Map(function(h, b) crossprod(h, drop(b %*% (gamma - eta))), H_star, B),
  true_probability
))
wrong_law_bias_formula <- drop((L - t(C_true)) %*% (eta - gamma))
check_close(wrong_law_score - drop(L %*% gamma), wrong_law_bias_formula,
            "assignment-law misspecification bias")
check_close(
  weighted_sum(Map(function(b, h) crossprod(b, h), B, H_zero),
               true_probability),
  t(L),
  "ordinary calibration survives same-marginal joint-law error"
)
wrong_law_carryover_score <- drop(L %*% eta) + drop(weighted_sum(
  Map(function(h, m, b) {
    crossprod(h, m - drop(b %*% eta))
  }, H_star, mean_truth, B), true_probability
))
true_robust_target <- drop(weighted_sum(
  Map(function(h, m) crossprod(h, m), H_zero, mean_truth),
  true_probability
))
wrong_law_carryover_formula <- wrong_law_bias_formula + weighted_bias(
  Map(`-`, H_star, H_zero), carryover, true_probability
)
check_close(wrong_law_carryover_score - true_robust_target,
            wrong_law_carryover_formula,
            "joint assignment and carryover bias decomposition")

result <- list(
  gate = "E1_carryover_sensitivity_theory",
  internal_theory_feasibility_passed = TRUE,
  standalone_publication_novelty_passed = FALSE,
  unrestricted_inference_remains_primary = TRUE,
  finite_support_checks_passed = TRUE,
  check_count = length(errors),
  maximum_numerical_error = max(errors),
  arbitrary_departure_ellipsoid_ratio = ellipsoid_ratio,
  blind_departure_per_task_norm = sqrt(blind_norm),
  lag_to_unrestricted_max_eigenvalue_ratio =
    max(eigen(lag_geometry$K, symmetric = TRUE)$values) /
    max(eigen(K, symmetric = TRUE)$values),
  variance_gap_minimum_eigenvalue = min(eigen(variance_gap, symmetric = TRUE)$values),
  decision = "proceed_to_gate_E2_computational_sensitivity",
  generated_at_utc = format(Sys.time(), tz = "UTC", usetz = TRUE)
)
output_dir <- file.path(dirname(script_path()), "..", "..", "results",
                        "feasible-eif")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
yaml::write_yaml(result, file.path(output_dir, "gate_e1_theory_checks.yml"))

cat(sprintf("All %d Gate E1 finite-support checks passed.\n", length(errors)))
