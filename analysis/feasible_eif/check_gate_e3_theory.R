# Algebra checks for Gate E3's Gaussian reduction and structured-carryover
# calibration. These checks are deterministic and do not alter the frozen Gate
# E2 implementation or decision.

find_project_root <- function(start = getwd()) {
  here <- normalizePath(start, winslash = "/", mustWork = TRUE)
  repeat {
    marker <- file.path(here, "DESCRIPTION")
    if (file.exists(marker)) return(here)
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

check_count <- 0L

check_close <- function(label, value, target, tolerance = 1e-10) {
  error <- max(abs(as.numeric(value) - as.numeric(target)))
  if (!is.finite(error) || error > tolerance) {
    stop(sprintf("FAIL %-55s %.6g", label, error), call. = FALSE)
  }
  check_count <<- check_count + 1L
  cat(sprintf("PASS %-55s %.3e\n", label, error))
  invisible(error)
}

check_true <- function(label, value) {
  if (!isTRUE(value)) stop(sprintf("FAIL %s", label), call. = FALSE)
  check_count <<- check_count + 1L
  cat(sprintf("PASS %s\n", label))
  invisible(value)
}

# -------------------------------------------------------------------------
# 1. The scalar restricted/unrestricted limit is one bounded-normal-mean
#    problem after an invertible transformation.
# -------------------------------------------------------------------------

restricted_variance <- 0.40
headroom <- 0.75
local_target <- -0.30
local_bias <- 0.55

joint_covariance <- matrix(c(
  restricted_variance,
  restricted_variance,
  restricted_variance,
  restricted_variance + headroom
), 2L, 2L, byrow = TRUE)

# Input order is (restricted, unrestricted); output is
# (restricted, restricted - unrestricted).
transform <- matrix(c(1, 0, 1, -1), 2L, 2L, byrow = TRUE)
transformed_covariance <- transform %*% joint_covariance %*% t(transform)
transformed_mean <- drop(
  transform %*% c(local_target + local_bias, local_target)
)

check_close(
  "Gaussian transform gives independent restricted and diagnostic terms",
  transformed_covariance,
  diag(c(restricted_variance, headroom))
)
check_close(
  "diagnostic mean is exactly the local restricted-score bias",
  transformed_mean,
  c(local_target + local_bias, local_bias)
)

# Every translation-equivariant estimator can be written as
# R - sqrt(headroom) * g(Z), with Z ~ N(mu, 1). For a fixed linear
# correction g(z) = lambda z, verify the exact risk and minimax coefficient.
linear_excess_risk <- function(lambda, mu) {
  lambda^2 + (1 - lambda)^2 * mu^2
}

for (radius in c(0.25, 1, 3)) {
  lambda <- radius^2 / (1 + radius^2)
  tau <- 1 - lambda
  derivative <- 2 * lambda - 2 * (1 - lambda) * radius^2
  grid <- seq(0, 1, length.out = 10001L)
  grid_objective <- linear_excess_risk(grid, radius)
  check_close(
    sprintf("linear minimax first-order condition radius %.2f", radius),
    derivative,
    0
  )
  check_true(
    sprintf("linear minimax grid optimum radius %.2f", radius),
    abs(grid[[which.min(grid_objective)]] - lambda) <= 1.1e-4
  )
  check_close(
    sprintf("Gate E1 blend mapping radius %.2f", radius),
    tau,
    1 / (1 + radius^2)
  )
}

# Exact oracle mimicry is impossible. Attaining the restricted risk at mu=0
# forces g=0 almost everywhere under a positive normal density; that rule has
# excess risk mu^2 and loses to the unrestricted rule's excess risk 1 when
# |mu| > 1.
check_true(
  "zero-correction rule loses to unrestricted beyond break-even",
  linear_excess_risk(0, 1.5) > linear_excess_risk(1, 1.5)
)

# -------------------------------------------------------------------------
# 2. A two-task binary factorial example with lag-one carryover.
# -------------------------------------------------------------------------

schedule <- expand.grid(a1 = c(-1, 1), a2 = c(-1, 1))
state_count <- nrow(schedule)
probability <- rep(1 / state_count, state_count)
task_count <- 2L

# Four task-specific cell means: task 1 low/high, task 2 low/high.
B <- lapply(seq_len(state_count), function(s) {
  out <- matrix(0, task_count, 4L)
  out[1L, if (schedule$a1[[s]] == -1) 1L else 2L] <- 1
  out[2L, if (schedule$a2[[s]] == -1) 3L else 4L] <- 1
  out
})

# A scientifically structured violation: task 2 can depend linearly on the
# profile assigned in task 1.
Z <- lapply(seq_len(state_count), function(s) {
  matrix(c(0, schedule$a1[[s]]), task_count, 1L)
})

covariance <- matrix(c(1, 0.8, 0.8, 1), task_count, task_count)
target_map <- matrix(c(-0.5, 0.5, -0.5, 0.5), 1L)

# Ordinary taskwise Riesz weights for the average of the two task-specific
# high-minus-low contrasts.
ordinary <- lapply(seq_len(state_count), function(s) {
  matrix(c(schedule$a1[[s]], schedule$a2[[s]]), task_count, 1L)
})

weighted_sum <- function(values) {
  Reduce(`+`, Map(function(weight, value) weight * value,
                  probability, values))
}

ordinary_B_calibration <- weighted_sum(Map(crossprod, B, ordinary))
lag_target_map <- weighted_sum(Map(crossprod, ordinary, Z))
check_close(
  "ordinary score calibrates the task-specific factorial target",
  ordinary_B_calibration,
  t(target_map)
)
check_close(
  "lag feature has zero task-marginal target load in product design",
  lag_target_map,
  0
)

restricted_fit <- solve_finite_support_calibration(
  B = B,
  covariance = covariance,
  probability = probability,
  L = target_map
)

augmented_B <- Map(cbind, B, Z)
augmented_target_map <- cbind(target_map, lag_target_map)
structured_fit <- solve_finite_support_calibration(
  B = augmented_B,
  covariance = covariance,
  probability = probability,
  L = augmented_target_map
)

score_variance <- function(weights) {
  weighted_sum(lapply(weights, function(weight) {
    crossprod(weight, covariance %*% weight)
  }))
}

carryover_bias <- function(weights) {
  weighted_sum(Map(function(weight, ordinary_weight, lag_design) {
    crossprod(weight - ordinary_weight, lag_design)
  }, weights, ordinary, Z))
}

restricted_value <- score_variance(restricted_fit$weights)
structured_value <- score_variance(structured_fit$weights)
ordinary_value <- score_variance(ordinary)

check_close("no-carryover efficient variance in toy design",
            restricted_value, 0.72)
check_close("lag-robust efficient variance in toy design",
            structured_value, 1.36)
check_close("ordinary variance in toy design", ordinary_value, 2)
check_true(
  "nested mean models give nested efficiency bounds",
  restricted_value < structured_value && structured_value < ordinary_value
)
check_close(
  "no-carryover efficient score is biased for lag carryover",
  carryover_bias(restricted_fit$weights),
  -0.8
)
check_close(
  "augmented efficient score annihilates prespecified lag carryover",
  carryover_bias(structured_fit$weights),
  0
)

gamma <- c(0.10, 0.45, -0.20, 0.30)
alpha <- 0.37
mean_surface <- Map(function(own_design, lag_design) {
  drop(own_design %*% gamma + lag_design %*% alpha)
}, B, Z)
expected_score <- function(weights) {
  weighted_sum(Map(function(weight, mean_value) {
    crossprod(weight, mean_value)
  }, weights, mean_surface))
}
target_value <- expected_score(ordinary)
check_close(
  "structured score is centered for every mean in span[B,Z]",
  expected_score(structured_fit$weights),
  target_value
)
check_close(
  "restricted-score bias equals alpha times lag bias operator",
  expected_score(restricted_fit$weights) - target_value,
  alpha * carryover_bias(restricted_fit$weights)
)

projection_gap <- function(left, right) {
  weighted_sum(Map(function(left_weight, right_weight) {
    difference <- left_weight - right_weight
    crossprod(difference, covariance %*% difference)
  }, left, right))
}

check_close(
  "ordinary-to-structured variance gap is a projection norm",
  ordinary_value - structured_value,
  projection_gap(ordinary, structured_fit$weights)
)
check_close(
  "structured-to-restricted variance gap is a projection norm",
  structured_value - restricted_value,
  projection_gap(structured_fit$weights, restricted_fit$weights)
)
check_close(
  "structured efficiency-vulnerability identity",
  task_count * projection_gap(ordinary, structured_fit$weights),
  task_count * (ordinary_value - structured_value)
)

# A finite-radius structured rule can optimize the calibrated score directly,
# rather than merely blending the no-carryover and ordinary endpoints.
precision <- solve_spd(covariance)
information_B <- weighted_sum(lapply(B, function(value) {
  crossprod(value, precision %*% value)
}))
cross_information_BZ <- weighted_sum(Map(function(own_design, lag_design) {
  crossprod(own_design, precision %*% lag_design)
}, B, Z))
projection_BZ <- symmetric_pseudoinverse(information_B)$inverse %*%
  cross_information_BZ
Z_perp <- Map(function(own_design, lag_design) {
  lag_design - own_design %*% projection_BZ
}, B, Z)
Q_Z <- weighted_sum(lapply(Z_perp, function(value) {
  crossprod(value, precision %*% value)
})) / task_count
Q_Z_inverse <- symmetric_pseudoinverse(Q_Z)$inverse

block_diagonal <- function(blocks) {
  row_count <- sum(vapply(blocks, nrow, integer(1)))
  column_count <- sum(vapply(blocks, ncol, integer(1)))
  out <- matrix(0, row_count, column_count)
  row_offset <- 0L
  column_offset <- 0L
  for (block in blocks) {
    rows <- row_offset + seq_len(nrow(block))
    columns <- column_offset + seq_len(ncol(block))
    out[rows, columns] <- block
    row_offset <- row_offset + nrow(block)
    column_offset <- column_offset + ncol(block)
  }
  out
}

V_stack <- block_diagonal(lapply(probability, function(weight) {
  weight * covariance
}))
A_B <- do.call(cbind, Map(function(weight, value) {
  weight * t(value)
}, probability, B))
A_Z <- do.call(cbind, Map(function(weight, value) {
  weight * t(value)
}, probability, Z_perp))
ordinary_stack <- do.call(rbind, ordinary)

solve_direct_structured <- function(radius) {
  penalty <- radius^2 * t(A_Z) %*% Q_Z_inverse %*% A_Z
  P <- V_stack + penalty
  P_inverse <- solve_spd(P)
  G <- penalty %*% ordinary_stack
  calibration_information <- A_B %*% P_inverse %*% t(A_B)
  calibration_inverse <- symmetric_pseudoinverse(
    calibration_information
  )$inverse
  out <- P_inverse %*% G +
    P_inverse %*% t(A_B) %*% calibration_inverse %*%
    (t(target_map) - A_B %*% P_inverse %*% G)
  list(weights = out, P = P)
}

stack_to_weights <- function(value) {
  lapply(seq_len(state_count), function(s) {
    rows <- (s - 1L) * task_count + seq_len(task_count)
    value[rows, , drop = FALSE]
  })
}

structured_objective <- function(weight_stack, radius) {
  difference <- weight_stack - ordinary_stack
  drop(crossprod(weight_stack, V_stack %*% weight_stack) +
         radius^2 * crossprod(
           A_Z %*% difference,
           Q_Z_inverse %*% (A_Z %*% difference)
         ))
}

direct_zero <- solve_direct_structured(0)
check_close(
  "direct structured path begins at no-carryover efficient score",
  direct_zero$weights,
  do.call(rbind, restricted_fit$weights)
)

direct_large <- solve_direct_structured(1e3)
check_close(
  "direct structured path converges to exact lag-robust score",
  direct_large$weights,
  do.call(rbind, structured_fit$weights),
  tolerance = 1e-5
)

radius <- 1
direct_middle <- solve_direct_structured(radius)
check_close(
  "direct structured path preserves factorial calibration",
  A_B %*% direct_middle$weights,
  t(target_map)
)
check_true(
  "direct structured objective is no worse than ordinary",
  structured_objective(direct_middle$weights, radius) <=
    structured_objective(ordinary_stack, radius) + 1e-10
)

blend_objectives <- vapply(seq(0, 1, length.out = 1001L), function(tau) {
  blend <- ordinary_stack + tau *
    (do.call(rbind, restricted_fit$weights) - ordinary_stack)
  structured_objective(blend, radius)
}, numeric(1))
check_true(
  "direct structured score weakly dominates all endpoint blends",
  structured_objective(direct_middle$weights, radius) <=
    min(blend_objectives) + 1e-10
)

# The structured model is a function space, not a coefficient coding. Scaling
# or redundantly recoding Z must not alter fitted weights.
scaled_Z <- lapply(Z, function(value) 2 * value)
scaled_fit <- solve_finite_support_calibration(
  B = Map(cbind, B, scaled_Z),
  covariance = covariance,
  probability = probability,
  L = cbind(target_map, 2 * lag_target_map)
)
check_close(
  "structured score is invariant to invertible lag recoding",
  unlist(scaled_fit$weights),
  unlist(structured_fit$weights)
)

redundant_Z <- lapply(Z, function(value) cbind(value, 2 * value))
redundant_fit <- solve_finite_support_calibration(
  B = Map(cbind, B, redundant_Z),
  covariance = covariance,
  probability = probability,
  L = cbind(target_map, lag_target_map, 2 * lag_target_map)
)
check_close(
  "structured score is invariant to redundant lag coordinates",
  unlist(redundant_fit$weights),
  unlist(structured_fit$weights),
  tolerance = 1e-9
)

cat(sprintf("All %d Gate-E3 algebra checks passed.\n", check_count))
