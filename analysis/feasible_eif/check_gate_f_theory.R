# Deterministic algebra checks for Gate F's factorial schedule geometry.  This
# file is deliberately separate from the frozen Gate E2 implementation.

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
check_count <- 0L

check_close <- function(label, value, target, tolerance = 1e-10) {
  error <- max(abs(as.numeric(value) - as.numeric(target)))
  if (!is.finite(error) || error > tolerance) {
    stop(sprintf("FAIL %-64s %.6g", label, error), call. = FALSE)
  }
  check_count <<- check_count + 1L
  cat(sprintf("PASS %-64s %.3e\n", label, error))
  invisible(error)
}

check_true <- function(label, value) {
  if (!isTRUE(value)) stop(sprintf("FAIL %s", label), call. = FALSE)
  check_count <<- check_count + 1L
  cat(sprintf("PASS %s\n", label))
  invisible(value)
}

safe_choose <- function(n, k) {
  if (length(k) != 1L || !is.finite(k) || k < 0 || k > n || k != round(k)) {
    return(0)
  }
  choose(n, k)
}

krawtchouk_eigenvalue <- function(factor_count, order, distance) {
  numerator <- sum(vapply(0:order, function(overlap) {
    (-1)^overlap * safe_choose(order, overlap) *
      safe_choose(factor_count - order, distance - overlap)
  }, numeric(1)))
  numerator / choose(factor_count, distance)
}

factorial_subsets <- function(factor_count, include_intercept = TRUE) {
  out <- if (include_intercept) list(integer(0)) else list()
  for (order in seq_len(factor_count)) {
    out <- c(out, combn(
      seq_len(factor_count), order, simplify = FALSE
    ))
  }
  out
}

profile_matrix <- function(factor_count) {
  as.matrix(expand.grid(rep(list(c(-1, 1)), factor_count)))
}

walsh_matrix <- function(profiles, subsets) {
  vapply(subsets, function(subset) {
    if (length(subset) == 0L) return(rep(1, nrow(profiles)))
    apply(profiles[, subset, drop = FALSE], 1L, prod)
  }, numeric(nrow(profiles)))
}

hamming_transition <- function(profiles, distance) {
  factor_count <- ncol(profiles)
  distances <- outer(
    seq_len(nrow(profiles)), seq_len(nrow(profiles)),
    Vectorize(function(left, right) {
      sum(profiles[left, ] != profiles[right, ])
    })
  )
  (distances == distance) / choose(factor_count, distance)
}

matrix_power <- function(value, exponent) {
  if (exponent == 0L) return(diag(nrow(value)))
  out <- value
  if (exponent == 1L) return(out)
  for (index in 2:exponent) out <- out %*% value
  out
}

ar_correlation <- function(rho, task_count) {
  rho^abs(outer(seq_len(task_count), seq_len(task_count), `-`))
}

compound_symmetry <- function(task_count, correlation) {
  out <- matrix(correlation, task_count, task_count)
  diag(out) <- 1
  out
}

solve_task_mixing <- function(covariance, feature_correlation, task_weights) {
  precision <- solve(covariance)
  information <- precision * feature_correlation
  multiplier <- solve(information, task_weights)
  mixing <- precision %*% diag(multiplier, nrow = length(multiplier))
  ordinary_mixing <- diag(task_weights, nrow = length(task_weights))
  efficient_variance <- drop(crossprod(task_weights, multiplier))
  ordinary_variance <- drop(crossprod(
    task_weights,
    (covariance * feature_correlation) %*% task_weights
  ))
  list(
    information = information,
    multiplier = multiplier,
    mixing = mixing,
    ordinary_mixing = ordinary_mixing,
    efficient_variance = efficient_variance,
    ordinary_variance = ordinary_variance,
    gap = ordinary_variance - efficient_variance
  )
}

# -------------------------------------------------------------------------
# 1. Walsh characters exactly diagonalize Hamming-radial transitions.
# -------------------------------------------------------------------------

factor_count <- 4L
profiles <- profile_matrix(factor_count)
profile_count <- nrow(profiles)
subsets <- factorial_subsets(factor_count)
walsh <- walsh_matrix(profiles, subsets)
orders <- vapply(subsets, length, integer(1))

check_close(
  "Walsh basis is orthonormal under the uniform profile target",
  crossprod(walsh) / profile_count,
  diag(ncol(walsh))
)

fixed_distance_transitions <- lapply(0:factor_count, function(distance) {
  hamming_transition(profiles, distance)
})

for (distance in 0:factor_count) {
  transition <- fixed_distance_transitions[[distance + 1L]]
  check_close(
    sprintf("distance-%d transition is row stochastic", distance),
    rowSums(transition),
    rep(1, profile_count)
  )
  for (column in seq_along(subsets)) {
    lambda <- krawtchouk_eigenvalue(
      factor_count, orders[[column]], distance
    )
    check_close(
      sprintf(
        "Krawtchouk eigenpair h=%d Walsh-column=%d", distance, column
      ),
      transition %*% walsh[, column],
      lambda * walsh[, column]
    )
  }
}

distance_probability <- c(0.08, 0.17, 0.40, 0.23, 0.12)
radial_transition <- Reduce(`+`, Map(
  `*`, fixed_distance_transitions, distance_probability
))
rho_by_column <- vapply(orders, function(order) {
  sum(distance_probability * vapply(0:factor_count, function(distance) {
    krawtchouk_eigenvalue(factor_count, order, distance)
  }, numeric(1)))
}, numeric(1))

for (lag in 0:4) {
  empirical_lag_moment <- crossprod(
    walsh,
    matrix_power(radial_transition, lag) %*% walsh
  ) / profile_count
  check_close(
    sprintf("radial Markov lag-%d factorial moment", lag),
    empirical_lag_moment,
    diag(rho_by_column^lag)
  )
}

product_distance_probability <- choose(
  factor_count, 0:factor_count
) / 2^factor_count
product_rho <- vapply(seq_len(factor_count), function(order) {
  sum(product_distance_probability * vapply(
    0:factor_count,
    function(distance) {
      krawtchouk_eigenvalue(factor_count, order, distance)
    },
    numeric(1)
  ))
}, numeric(1))
check_close(
  "independent product transitions have zero nonintercept eigenvalues",
  product_rho,
  rep(0, factor_count)
)

no_repeat_probability <- c(
  0,
  choose(factor_count, 1:factor_count) / (2^factor_count - 1)
)
no_repeat_rho <- vapply(seq_len(factor_count), function(order) {
  sum(no_repeat_probability * vapply(
    0:factor_count,
    function(distance) {
      krawtchouk_eigenvalue(factor_count, order, distance)
    },
    numeric(1)
  ))
}, numeric(1))
check_close(
  "no-immediate-repeat transition is order neutral",
  no_repeat_rho,
  rep(-1 / (2^factor_count - 1), factor_count)
)

antipodal_rho <- vapply(seq_len(factor_count), function(order) {
  krawtchouk_eigenvalue(factor_count, order, factor_count)
}, numeric(1))
check_close(
  "antipodal transition alternates odd and preserves even orders",
  antipodal_rho,
  (-1)^seq_len(factor_count)
)

expected_f4_table <- rbind(
  c(1, 0.5, 0, -0.5, -1),
  c(1, 0, -1 / 3, 0, 1),
  c(1, -0.5, 0, 0.5, -1),
  c(1, -1, 1, -1, 1)
)
computed_f4_table <- outer(
  seq_len(factor_count), 0:factor_count,
  Vectorize(function(order, distance) {
    krawtchouk_eigenvalue(factor_count, order, distance)
  })
)
check_close(
  "F=4 order-by-distance spectrum has the predicted tradeoffs",
  computed_f4_table,
  expected_f4_table
)

# -------------------------------------------------------------------------
# 2. The full finite-support score reduces to one T by T system per order.
# -------------------------------------------------------------------------

small_factor_count <- 2L
small_profiles <- profile_matrix(small_factor_count)
small_profile_count <- nrow(small_profiles)
small_transitions <- lapply(0:small_factor_count, function(distance) {
  hamming_transition(small_profiles, distance)
})
small_distance_probability <- c(0.20, 0.50, 0.30)
small_transition <- Reduce(`+`, Map(
  `*`, small_transitions, small_distance_probability
))
small_rho <- sum(small_distance_probability * vapply(
  0:small_factor_count,
  function(distance) {
    krawtchouk_eigenvalue(small_factor_count, 1L, distance)
  },
  numeric(1)
))

task_count <- 3L
sequences <- as.matrix(expand.grid(rep(
  list(seq_len(small_profile_count)), task_count
)))
sequence_probability <- apply(sequences, 1L, function(sequence) {
  probability <- 1 / small_profile_count
  for (task in 2:task_count) {
    probability <- probability * small_transition[
      sequence[[task - 1L]], sequence[[task]]
    ]
  }
  probability
})
check_close(
  "enumerated Markov schedule probabilities sum to one",
  sum(sequence_probability),
  1
)

feature <- small_profiles[, 1L]
sequence_feature <- t(apply(sequences, 1L, function(sequence) {
  feature[sequence]
}))
enumerated_feature_correlation <- crossprod(
  sequence_feature * sqrt(sequence_probability)
)
theoretical_feature_correlation <- ar_correlation(
  small_rho, task_count
)
check_close(
  "enumerated schedule correlation equals the spectral T by T block",
  enumerated_feature_correlation,
  theoretical_feature_correlation
)

covariance <- compound_symmetry(task_count, 0.55)
task_weights <- c(0.20, 0.30, 0.50)
reduced_fit <- solve_task_mixing(
  covariance, theoretical_feature_correlation, task_weights
)
reduced_weights <- t(apply(sequence_feature, 1L, function(value) {
  drop(reduced_fit$mixing %*% value)
}))

# Direct finite-support calibration gives
# h_s = Sigma^{-1} diag(x_s) J^{-1} omega.  Evaluate it separately for every
# supported schedule and compare with the reduced mixing representation.
precision <- solve(covariance)
direct_weights <- t(apply(sequence_feature, 1L, function(value) {
  drop(precision %*% diag(value) %*% reduced_fit$multiplier)
}))
check_close(
  "reduced mixing agrees with direct finite-support calibration",
  reduced_weights,
  direct_weights
)

enumerated_calibration <- colSums(
  sequence_probability * sequence_feature * reduced_weights
)
check_close(
  "reduced score exactly calibrates every task coefficient",
  enumerated_calibration,
  task_weights
)

enumerated_efficient_variance <- sum(sequence_probability * vapply(
  seq_len(nrow(sequences)),
  function(index) {
    drop(crossprod(
      reduced_weights[index, ],
      covariance %*% reduced_weights[index, ]
    ))
  },
  numeric(1)
))
check_close(
  "reduced variance equals the enumerated finite-support objective",
  enumerated_efficient_variance,
  reduced_fit$efficient_variance
)

# -------------------------------------------------------------------------
# 3. The carryover map sees one temporal direction per active Walsh feature.
# -------------------------------------------------------------------------

rank_factor_count <- 3L
rank_profiles <- profile_matrix(rank_factor_count)
rank_subsets <- factorial_subsets(
  rank_factor_count, include_intercept = FALSE
)
rank_orders <- vapply(rank_subsets, length, integer(1))
rank_feature_count <- length(rank_subsets)
rank_distance_probability <- c(0.12, 0.28, 0.37, 0.23)
rank_task_count <- 3L
rank_covariance <- compound_symmetry(rank_task_count, 0.60)
rank_task_weights <- rep(1 / rank_task_count, rank_task_count)

temporal_operators <- vector("list", rank_feature_count)
efficiency_gaps <- numeric(rank_feature_count)
for (feature_index in seq_len(rank_feature_count)) {
  order <- rank_orders[[feature_index]]
  rho <- sum(rank_distance_probability * vapply(
    0:rank_factor_count,
    function(distance) {
      krawtchouk_eigenvalue(rank_factor_count, order, distance)
    },
    numeric(1)
  ))
  feature_correlation <- ar_correlation(rho, rank_task_count)
  fit <- solve_task_mixing(
    rank_covariance, feature_correlation, rank_task_weights
  )
  mixing_difference <- fit$mixing - fit$ordinary_mixing
  temporal_operators[[feature_index]] <-
    mixing_difference %*% feature_correlation
  efficiency_gaps[[feature_index]] <- fit$gap

  check_close(
    sprintf("order-%d temporal bias operator has zero diagonal", order),
    diag(temporal_operators[[feature_index]]),
    rep(0, rank_task_count)
  )
  projection_norm <- sum(diag(
    t(mixing_difference) %*% rank_covariance %*%
      mixing_difference %*% feature_correlation
  ))
  check_close(
    sprintf("order-%d efficiency gap is the score projection norm", order),
    projection_norm,
    fit$gap
  )
  check_true(
    sprintf("order-%d activity agrees with positive efficiency gap", order),
    (sum(temporal_operators[[feature_index]]^2) > 1e-12) ==
      (fit$gap > 1e-12)
  )
}

off_diagonal_pairs <- which(
  row(diag(rank_task_count)) != col(diag(rank_task_count)),
  arr.ind = TRUE
)
domain_per_feature <- nrow(off_diagonal_pairs)
bias_operator <- matrix(
  0,
  rank_feature_count,
  rank_feature_count * domain_per_feature
)
for (target_feature in seq_len(rank_feature_count)) {
  for (source_feature in seq_len(rank_feature_count)) {
    columns <- (source_feature - 1L) * domain_per_feature +
      seq_len(domain_per_feature)
    if (target_feature == source_feature) {
      bias_operator[target_feature, columns] <- vapply(
        seq_len(domain_per_feature),
        function(index) {
          temporal_operators[[target_feature]][
            off_diagonal_pairs[index, 1L],
            off_diagonal_pairs[index, 2L]
          ]
        },
        numeric(1)
      )
    }
  }
}
active_feature_count <- sum(efficiency_gaps > 1e-12)
operator_rank <- qr(bias_operator, tol = 1e-10)$rank
check_close(
  "carryover-map rank is one per active Walsh feature",
  operator_rank,
  active_feature_count
)
check_close(
  "carryover-map kernel has the stated exact dimension",
  ncol(bias_operator) - operator_rank,
  rank_feature_count * domain_per_feature - active_feature_count
)
source_feature_by_column <- rep(
  seq_len(rank_feature_count), each = domain_per_feature
)
cross_feature_mask <- outer(
  seq_len(rank_feature_count), source_feature_by_column, `!=`
)
check_true(
  "different Walsh features are exactly carryover-blind",
  all(bias_operator[cross_feature_mask] == 0)
)

# -------------------------------------------------------------------------
# 4. Product versus whole-sequence sampling without replacement.
# -------------------------------------------------------------------------

cs_task_count <- 8L
cs_profile_count <- 16L
cs_correlation <- 0.80
cs_covariance <- compound_symmetry(cs_task_count, cs_correlation)
cs_weights <- rep(1 / cs_task_count, cs_task_count)
without_replacement_r <- -1 / (cs_profile_count - 1)
product_feature_correlation <- diag(cs_task_count)
without_replacement_feature_correlation <- compound_symmetry(
  cs_task_count, without_replacement_r
)

product_fit <- solve_task_mixing(
  cs_covariance, product_feature_correlation, cs_weights
)
without_replacement_fit <- solve_task_mixing(
  cs_covariance, without_replacement_feature_correlation, cs_weights
)

cs_closed_form <- function(assignment_correlation) {
  a_term <- 1 + (cs_task_count - 2) * cs_correlation
  c_term <- (1 - cs_correlation) *
    (1 + (cs_task_count - 1) * cs_correlation)
  ordinary <- (
    1 + (cs_task_count - 1) * cs_correlation * assignment_correlation
  ) / cs_task_count
  efficient <- c_term / (
    cs_task_count *
      (a_term - (cs_task_count - 1) * cs_correlation *
         assignment_correlation)
  )
  c(ordinary = ordinary, efficient = efficient, gap = ordinary - efficient)
}

check_close(
  "product compound-symmetry variances have the closed form",
  c(
    ordinary = product_fit$ordinary_variance,
    efficient = product_fit$efficient_variance,
    gap = product_fit$gap
  ),
  cs_closed_form(0)
)
check_close(
  "without-replacement compound-symmetry variances have the closed form",
  c(
    ordinary = without_replacement_fit$ordinary_variance,
    efficient = without_replacement_fit$efficient_variance,
    gap = without_replacement_fit$gap
  ),
  cs_closed_form(without_replacement_r)
)
check_true(
  "without replacement reduces ordinary variance under positive covariance",
  without_replacement_fit$ordinary_variance <
    product_fit$ordinary_variance
)
check_true(
  "without replacement reduces the restricted efficiency bound",
  without_replacement_fit$efficient_variance <
    product_fit$efficient_variance
)
check_true(
  "without replacement reduces unrestricted-ball vulnerability",
  without_replacement_fit$gap < product_fit$gap
)

assignment_grid <- seq(without_replacement_r, 0, length.out = 1001L)
closed_form_grid <- t(vapply(
  assignment_grid, cs_closed_form, numeric(3)
))
check_true(
  "ordinary, efficient, and vulnerability terms increase toward product",
  all(apply(closed_form_grid, 2L, function(value) {
    all(diff(value) >= -1e-12)
  }))
)

# -------------------------------------------------------------------------
# 5. Two-task closed forms make order-targeted schedule tradeoffs explicit.
# -------------------------------------------------------------------------

two_task_correlation <- 0.80
two_task_formula <- function(lambda) {
  ordinary <- (1 + two_task_correlation * lambda) / 2
  efficient <- (1 - two_task_correlation^2) /
    (2 * (1 - two_task_correlation * lambda))
  gap <- two_task_correlation^2 * (1 - lambda^2) /
    (2 * (1 - two_task_correlation * lambda))
  c(ordinary = ordinary, efficient = efficient, gap = gap)
}

for (lambda in c(-0.8, -0.25, 0, 0.35, 0.85)) {
  fit <- solve_task_mixing(
    compound_symmetry(2L, two_task_correlation),
    compound_symmetry(2L, lambda),
    c(0.5, 0.5)
  )
  check_close(
    sprintf("two-task variance formula lambda=%+.2f", lambda),
    c(
      ordinary = fit$ordinary_variance,
      efficient = fit$efficient_variance,
      gap = fit$gap
    ),
    two_task_formula(lambda)
  )
}

negative_lambda_grid <- seq(-0.99, 0, length.out = 1001L)
negative_formula_grid <- t(vapply(
  negative_lambda_grid, two_task_formula, numeric(3)
))
check_true(
  "more negative spectra weakly improve all two-task criteria",
  all(apply(negative_formula_grid, 2L, function(value) {
    all(diff(value) >= -1e-12)
  }))
)

best_distance_by_order <- apply(computed_f4_table, 1L, function(value) {
  which(value == min(value)) - 1L
})
check_true(
  "fixed Hamming distances target different factorial orders",
  identical(
    unname(best_distance_by_order),
    list(4L, 2L, 4L, c(1L, 3L))
  )
)

# A radial design is described by F+1 probabilities.  Its K order spectra are
# a linear Krawtchouk transform, so evaluating a proposed schedule needs only
# K small T by T systems.
spectral_map <- computed_f4_table
candidate_probability <- c(0.02, 0.18, 0.50, 0.28, 0.02)
candidate_spectrum <- drop(spectral_map %*% candidate_probability)
direct_candidate_spectrum <- vapply(seq_len(factor_count), function(order) {
  sum(candidate_probability * vapply(
    0:factor_count,
    function(distance) {
      krawtchouk_eigenvalue(factor_count, order, distance)
    },
    numeric(1)
  ))
}, numeric(1))
check_close(
  "schedule evaluation is the low-dimensional Krawtchouk transform",
  candidate_spectrum,
  direct_candidate_spectrum
)

# -------------------------------------------------------------------------
# 6. A schedule comparison must hold the causal target fixed.
# -------------------------------------------------------------------------

# Allow Y_2's mean to contain a lag feature x_1.  The augmented calibration
# target is (1/2, 1/2, lag_load).  Setting lag_load=lambda/2 preserves the
# ordinary score's q-specific task-marginal estimand.  Setting it to zero
# instead targets the pure current-profile/direct factorial effect for every
# schedule.
augmented_two_task_bound <- function(lambda, lag_load, sigma_squared = 1) {
  joint_features <- as.matrix(expand.grid(x1 = c(-1, 1), x2 = c(-1, 1)))
  probabilities <- (1 + lambda *
                        joint_features[, 1L] * joint_features[, 2L]) / 4
  covariance <- sigma_squared * compound_symmetry(
    2L, two_task_correlation
  )
  precision <- solve(covariance)
  information <- matrix(0, 3L, 3L)
  for (index in seq_len(nrow(joint_features))) {
    x1 <- joint_features[index, 1L]
    x2 <- joint_features[index, 2L]
    design <- matrix(c(
      x1, 0,
      0, x2,
      0, x1
    ), 2L, 3L)
    information <- information + probabilities[[index]] *
      crossprod(design, precision %*% design)
  }
  target <- c(0.5, 0.5, lag_load)
  drop(crossprod(target, solve(information, target)))
}

q_specific_bound_formula <- function(lambda) {
  (1 + two_task_correlation * lambda) / 2 -
    two_task_correlation^2 * (1 - lambda^2) / 4
}
fixed_direct_bound_formula <- function(lambda) {
  (2 - lambda^2 - two_task_correlation^2) /
    (4 * (1 - lambda^2))
}

for (lambda in c(-0.8, -0.25, 0, 0.35, 0.85)) {
  check_close(
    sprintf("q-specific lag-robust bound lambda=%+.2f", lambda),
    augmented_two_task_bound(lambda, lambda / 2),
    q_specific_bound_formula(lambda)
  )
  check_close(
    sprintf("fixed-direct lag-robust bound lambda=%+.2f", lambda),
    augmented_two_task_bound(lambda, 0),
    fixed_direct_bound_formula(lambda)
  )
}

fixed_target_grid <- seq(-0.95, 0.95, length.out = 1001L)
fixed_target_values <- vapply(
  fixed_target_grid, fixed_direct_bound_formula, numeric(1)
)
check_close(
  "fixed direct-effect target is uniquely optimized at product assignment",
  fixed_target_grid[[which.min(fixed_target_values)]],
  0
)
check_true(
  "changing assignment changes the q-specific lag target load",
  (-0.25 / 2) != (0.35 / 2)
)

cat(sprintf("All %d Gate-F algebra checks passed.\n", check_count))
