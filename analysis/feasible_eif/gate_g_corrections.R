# Gate G finite-sample corrections.  These functions are new and deliberately
# do not alter the frozen Gate E2 implementation.

gate_g_operator_mixing <- function(operator) {
  if (is.matrix(operator)) return(operator)
  feasible_assert(!is.null(operator$mixing),
                  "A Gate-G operator must supply a mixing matrix.")
  as.matrix(operator$mixing)
}

# Exact reciprocal-count moments for a uniformly assigned saturated cell.
# The formulas condition on a nonempty training cell.  Gate G's supported-fold
# construction makes that the relevant finite-sample regime; the probability
# of invoking the frozen task-wide empty-cell fallback is recorded separately.
gate_g_reciprocal_binomial_moment <- function(training_count, profile_count) {
  training_count <- as.integer(training_count)
  profile_count <- as.integer(profile_count)
  feasible_assert(training_count >= 1L && profile_count >= 2L,
                  "Positive training and profile counts are required.")
  probability <- 1 / profile_count
  counts <- seq_len(training_count)
  sum(stats::dbinom(counts, training_count, probability) / counts)
}

# For product-uniform assignment, Gamma is the exact marginal expected
# covariance of task-specific saturated training-cell mean errors when empty
# cells contribute zero error (equivalently, conditional on supported cells):
#   Gamma_tt = Sigma_tt E{1/N_t; N_t > 0}
#   Gamma_ts = Sigma_ts Pr(N_t > 0)^2 / n, t != s.
# It prices the leading O(G^-2) mean-estimation remainder in the unscaled
# cross-fit risk. The affine part is constant over calibrated operators, so
# solve_task_mixing(Sigma + Gamma) exactly minimizes the marginal-fold
# objective. Cross-fold overlap adds a smaller term; the propagated loading
# calculation remains the exact full K-fold benchmark.
gate_g_expected_mean_error_covariance <- function(
    covariance, training_count, profile_count) {
  covariance <- feasible_symmetric(as.matrix(covariance))
  training_count <- as.integer(training_count)
  profile_count <- as.integer(profile_count)
  feasible_assert(nrow(covariance) == ncol(covariance),
                  "The task covariance must be square.")
  reciprocal <- gate_g_reciprocal_binomial_moment(
    training_count, profile_count
  )
  support_probability <- 1 - (1 - 1 / profile_count)^training_count
  gamma <- covariance * support_probability^2 / training_count
  diag(gamma) <- diag(covariance) * reciprocal
  list(
    covariance = feasible_symmetric(gamma),
    reciprocal_count_moment = reciprocal,
    support_probability = support_probability,
    empty_cell_probability = 1 - support_probability,
    training_count = training_count,
    profile_count = profile_count
  )
}

gate_g_count_aware_operator <- function(
    covariance, training_count, profile_count, task_weights = NULL,
    tolerance = 1e-10) {
  covariance <- feasible_symmetric(as.matrix(covariance))
  count_moment <- gate_g_expected_mean_error_covariance(
    covariance, training_count, profile_count
  )
  operator <- solve_task_mixing(
    covariance + count_moment$covariance,
    task_weights = task_weights,
    mechanism = "product",
    tolerance = tolerance
  )
  operator$base_covariance <- covariance
  operator$mean_error_covariance <- count_moment$covariance
  operator$count_moment <- count_moment
  operator$finite_sample_objective <- TRUE
  class(operator) <- c("gate_g_count_aware_operator", class(operator))
  operator
}

gate_g_expected_crossfit_scaled_variance <- function(
    mixing, covariance, training_count, profile_count, task_weights) {
  mixing <- as.matrix(mixing)
  covariance <- feasible_symmetric(as.matrix(covariance))
  task_weights <- as.numeric(task_weights)
  count_moment <- gate_g_expected_mean_error_covariance(
    covariance, training_count, profile_count
  )
  gamma <- count_moment$covariance
  first_order <- mixing_quadratic(
    mixing, covariance, diag(nrow(covariance))
  )
  quadratic_remainder <- sum((t(mixing) %*% gamma) * t(mixing))
  calibration_constant <-
    -sum(diag(gamma) * task_weights^2 / profile_count)
  list(
    total = first_order + quadratic_remainder + calibration_constant,
    first_order = first_order,
    nuisance_remainder = quadratic_remainder + calibration_constant,
    mean_error_covariance = gamma,
    empty_cell_probability = count_moment$empty_cell_probability
  )
}

gate_g_target_load <- function(geometry, task_weights) {
  riesz_lookup <- as.matrix(geometry$riesz_lookup)
  target_probability <- as.numeric(geometry$target_probability)
  task_weights <- as.numeric(task_weights)
  tasks <- length(task_weights)
  profiles <- nrow(riesz_lookup)
  coordinates <- ncol(riesz_lookup)
  out <- array(
    0, c(tasks, profiles, coordinates),
    dimnames = list(
      paste0("task_", seq_len(tasks)),
      paste0("profile_", seq_len(profiles)),
      colnames(riesz_lookup)
    )
  )
  for (tt in seq_len(tasks)) {
    out[tt, , ] <- task_weights[[tt]] *
      riesz_lookup * target_probability
  }
  out
}

gate_g_profile_design <- function(profile, tasks, profile_count) {
  profile <- as.integer(profile)
  feasible_assert(length(profile) == tasks &&
                    all(profile >= 1L & profile <= profile_count),
                  "The profile vector is outside the declared support.")
  out <- matrix(0, tasks, tasks * profile_count)
  out[cbind(
    seq_len(tasks),
    (seq_len(tasks) - 1L) * profile_count + profile
  )] <- 1
  out
}

# For fixed fold-specific score operators and task-specific saturated cell
# means, the complete cross-fitted estimator is linear in every observed
# outcome.  This function returns those exact respondent-by-task loadings.
gate_g_crossfit_linearization <- function(
    profile_index, outcome, outer_fold, operators,
    geometry, task_weights, tolerance = 1e-10) {
  profile_index <- as.matrix(profile_index)
  outcome <- as.matrix(outcome)
  outer_fold <- as.integer(outer_fold)
  feasible_assert(identical(dim(profile_index), dim(outcome)),
                  "Profiles and outcomes must align.")
  n <- nrow(outcome)
  tasks <- ncol(outcome)
  profile_count <- geometry$profile_count
  coordinate_count <- ncol(geometry$riesz_lookup)
  fold_values <- sort(unique(outer_fold))
  feasible_assert(length(operators) == length(fold_values),
                  "One fixed operator is required for every outer fold.")
  target_load <- gate_g_target_load(geometry, task_weights)
  linear_load <- array(
    0, c(n, tasks, coordinate_count),
    dimnames = list(
      NULL, paste0("task_", seq_len(tasks)),
      colnames(geometry$riesz_lookup)
    )
  )
  score <- matrix(
    NA_real_, n, coordinate_count,
    dimnames = list(NULL, colnames(geometry$riesz_lookup))
  )
  fold_diagnostics <- vector("list", length(fold_values))

  for (fold_position in seq_along(fold_values)) {
    fold <- fold_values[[fold_position]]
    valid <- outer_fold == fold
    train <- !valid
    valid_ids <- which(valid)
    train_ids <- which(train)
    valid_count <- length(valid_ids)
    train_count <- length(train_ids)
    mixing <- gate_g_operator_mixing(operators[[fold_position]])
    feasible_assert(nrow(mixing) == tasks && ncol(mixing) == tasks,
                    "A fold mixing matrix has the wrong task dimension.")

    fit <- fit_task_saturated(
      profile_index, outcome, train, geometry, task_weights
    )
    residual <- outcome[valid, , drop = FALSE] -
      fit$prediction[valid, , drop = FALSE]
    score[valid, ] <- score_factorial_clusters(
      residual,
      profile_index[valid, , drop = FALSE],
      geometry$riesz_lookup,
      structure(list(mixing = mixing),
                class = c("feasible_task_operator", "factorial_task_mixing")),
      fit$plugin
    )

    # The direct held-out residual load is H_i / n, where
    # H_i = C * (Riesz feature by task).
    heldout_weight <- array(0, c(valid_count, tasks, coordinate_count))
    for (local_index in seq_len(valid_count)) {
      respondent <- valid_ids[[local_index]]
      assigned_riesz <- geometry$riesz_lookup[
        profile_index[respondent, ], , drop = FALSE
      ]
      H <- mixing %*% assigned_riesz
      heldout_weight[local_index, , ] <- H
      linear_load[respondent, , ] <-
        linear_load[respondent, , ] + H / n
    }

    # Fold-specific empirical calibration discrepancy, already scaled for the
    # overall n-respondent estimator.
    discrepancy <- (valid_count / n) * target_load
    for (local_index in seq_len(valid_count)) {
      respondent <- valid_ids[[local_index]]
      for (tt in seq_len(tasks)) {
        cell <- profile_index[respondent, tt]
        discrepancy[tt, cell, ] <- discrepancy[tt, cell, ] -
          heldout_weight[local_index, tt, ] / n
      }
    }

    # A nonempty saturated cell mean is the average of the corresponding
    # training outcomes.  The frozen empty-cell rule is the task-wide training
    # mean, which is also linear and is propagated explicitly.
    empty_cells <- 0L
    for (tt in seq_len(tasks)) {
      counts <- tabulate(
        profile_index[train, tt], nbins = profile_count
      )
      for (cell in seq_len(profile_count)) {
        if (counts[[cell]] > 0L) {
          rows <- train_ids[profile_index[train, tt] == cell]
          for (coordinate in seq_len(coordinate_count)) {
            linear_load[rows, tt, coordinate] <-
              linear_load[rows, tt, coordinate] +
              discrepancy[tt, cell, coordinate] / counts[[cell]]
          }
        } else {
          empty_cells <- empty_cells + 1L
          for (coordinate in seq_len(coordinate_count)) {
            linear_load[train_ids, tt, coordinate] <-
              linear_load[train_ids, tt, coordinate] +
              discrepancy[tt, cell, coordinate] / train_count
          }
        }
      }
    }
    fold_diagnostics[[fold_position]] <- data.frame(
      outer_fold = fold,
      validation_count = valid_count,
      training_count = train_count,
      empty_training_cells = empty_cells,
      maximum_empirical_calibration_discrepancy = max(abs(discrepancy)),
      stringsAsFactors = FALSE
    )
  }

  estimate <- colMeans(score)
  linear_estimate <- vapply(seq_len(coordinate_count), function(coordinate) {
    sum(linear_load[, , coordinate] * outcome)
  }, numeric(1))
  names(linear_estimate) <- colnames(geometry$riesz_lookup)
  representation_error <- max(abs(estimate - linear_estimate))
  feasible_assert(
    representation_error <= tolerance * max(1, max(abs(estimate))),
    sprintf("Gate-G linear representation failed (error %.6g).",
            representation_error)
  )

  list(
    estimate = estimate,
    score = score,
    linear_load = linear_load,
    linear_estimate = linear_estimate,
    representation_error = representation_error,
    fold_diagnostics = do.call(rbind, fold_diagnostics)
  )
}

gate_g_conditional_covariance <- function(linear_load, covariance) {
  covariance <- as.matrix(covariance)
  dimensions <- dim(linear_load)
  feasible_assert(length(dimensions) == 3L &&
                    dimensions[[2L]] == nrow(covariance) &&
                    nrow(covariance) == ncol(covariance),
                  "Linear loads and task covariance do not align.")
  out <- matrix(0, dimensions[[3L]], dimensions[[3L]])
  for (respondent in seq_len(dimensions[[1L]])) {
    loading <- linear_load[respondent, , , drop = FALSE]
    dim(loading) <- c(dimensions[[2L]], dimensions[[3L]])
    out <- out + crossprod(loading, covariance %*% loading)
  }
  feasible_symmetric(out)
}

# Exact conditional multiplicative inflation of the raw OOF saturated-residual
# second moment.  F[t,s] is defined by
# E(M_raw[t,s] | assignments) = F[t,s] * Sigma[t,s].
gate_g_saturated_residual_inflation <- function(
    profile_index, validation_fold, profile_count) {
  profile_index <- as.matrix(profile_index)
  validation_fold <- as.integer(validation_fold)
  n <- nrow(profile_index)
  tasks <- ncol(profile_index)
  feasible_assert(length(validation_fold) == n,
                  "Residual folds must align with respondents.")
  inflation_addition <- matrix(0, tasks, tasks)
  fold_values <- sort(unique(validation_fold))

  for (fold in fold_values) {
    valid <- validation_fold == fold
    train <- !valid
    valid_profile <- profile_index[valid, , drop = FALSE]
    train_profile <- profile_index[train, , drop = FALSE]
    for (tt in seq_len(tasks)) {
      count_t <- tabulate(train_profile[, tt], nbins = profile_count)
      feasible_assert(all(count_t[unique(valid_profile[, tt])] > 0L),
                      "A validation cell is absent from its mean-training set.")
      for (ss in seq_len(tasks)) {
        count_s <- tabulate(train_profile[, ss], nbins = profile_count)
        joint <- table(
          factor(train_profile[, tt], levels = seq_len(profile_count)),
          factor(train_profile[, ss], levels = seq_len(profile_count))
        )
        index <- cbind(valid_profile[, tt], valid_profile[, ss])
        numerator <- joint[index]
        denominator <- count_t[valid_profile[, tt]] *
          count_s[valid_profile[, ss]]
        feasible_assert(all(denominator > 0),
                        "Count inflation encountered an unsupported cell.")
        inflation_addition[tt, ss] <- inflation_addition[tt, ss] +
          sum(numerator / denominator) / n
      }
    }
  }
  feasible_symmetric(1 + inflation_addition)
}

gate_g_count_corrected_covariances <- function(
    residual, profile_index, validation_fold, profile_count,
    relative_floor = 1e-8) {
  residual <- as.matrix(residual)
  profile_index <- as.matrix(profile_index)
  feasible_assert(identical(dim(residual), dim(profile_index)) &&
                    all(is.finite(residual)),
                  "OOF residuals and profiles must be finite and aligned.")
  inflation <- gate_g_saturated_residual_inflation(
    profile_index, validation_fold, profile_count
  )
  raw <- feasible_symmetric(crossprod(residual) / nrow(residual))
  corrected_raw <- feasible_symmetric(raw / inflation)
  unstructured <- stabilize_working_covariance(
    corrected_raw, relative_floor
  )
  task_count <- ncol(residual)
  cs_raw <- matrix(
    if (task_count == 1L) 0 else
      mean(corrected_raw[row(corrected_raw) != col(corrected_raw)]),
    task_count, task_count
  )
  diag(cs_raw) <- mean(diag(corrected_raw))
  compound_symmetry <- stabilize_working_covariance(
    cs_raw, relative_floor
  )
  list(
    unstructured = unstructured,
    compound_symmetry = compound_symmetry,
    raw_second_moment = raw,
    corrected_second_moment = corrected_raw,
    inflation = inflation
  )
}

gate_g_saturated_oof_residual <- function(
    profile_index, outcome, validation_fold, geometry, task_weights) {
  profile_index <- as.matrix(profile_index)
  outcome <- as.matrix(outcome)
  validation_fold <- as.integer(validation_fold)
  feasible_assert(identical(dim(profile_index), dim(outcome)) &&
                    length(validation_fold) == nrow(outcome),
                  "OOF inputs do not align.")
  residual <- matrix(NA_real_, nrow(outcome), ncol(outcome))
  empty_cell_count <- 0L
  for (fold in sort(unique(validation_fold))) {
    valid <- validation_fold == fold
    fit <- fit_task_saturated(
      profile_index, outcome, !valid, geometry, task_weights
    )
    empty_cell_count <- empty_cell_count + fit$empty_cell_count
    residual[valid, ] <- outcome[valid, , drop = FALSE] -
      fit$prediction[valid, , drop = FALSE]
  }
  feasible_assert(all(is.finite(residual)),
                  "Saturated OOF residual construction is incomplete.")
  list(residual = residual, empty_cell_count = empty_cell_count)
}

# Residual covariance from a saturated fit on one complete supported sample.
# For task-specific projection matrices P_t,
#   E[e_t' e_u | A] = Sigma_tu tr{(I-P_t)(I-P_u)}.
# The trace is computed from cell and joint-cell counts without constructing
# n-by-n projection matrices.
gate_g_saturated_residual_df_covariances <- function(
    profile_index, outcome, geometry, task_weights,
    relative_floor = 1e-8) {
  profile_index <- as.matrix(profile_index)
  outcome <- as.matrix(outcome)
  feasible_assert(identical(dim(profile_index), dim(outcome)),
                  "Profiles and outcomes must align.")
  n <- nrow(outcome)
  tasks <- ncol(outcome)
  profile_count <- geometry$profile_count
  fit <- fit_task_saturated(
    profile_index, outcome, rep(TRUE, n), geometry, task_weights
  )
  # An empty task-profile cell has no residual row and does not enter the
  # observed-design projection. The rank/count formula below remains exact.
  residual <- outcome - fit$prediction
  cross_moment <- crossprod(residual)
  degrees_of_freedom <- matrix(NA_real_, tasks, tasks)
  for (tt in seq_len(tasks)) {
    count_t <- tabulate(profile_index[, tt], nbins = profile_count)
    rank_t <- sum(count_t > 0L)
    for (ss in seq_len(tasks)) {
      count_s <- tabulate(profile_index[, ss], nbins = profile_count)
      rank_s <- sum(count_s > 0L)
      joint <- table(
        factor(profile_index[, tt], levels = seq_len(profile_count)),
        factor(profile_index[, ss], levels = seq_len(profile_count))
      )
      denominator <- outer(count_t, count_s)
      overlap_trace <- sum(ifelse(
        denominator > 0,
        as.matrix(joint)^2 / denominator,
        0
      ))
      degrees_of_freedom[tt, ss] <-
        n - rank_t - rank_s + overlap_trace
    }
  }
  feasible_assert(all(is.finite(degrees_of_freedom)) &&
                    all(degrees_of_freedom > 0),
                  "Residual covariance degrees of freedom are not positive.")
  raw_second_moment <- feasible_symmetric(cross_moment / n)
  corrected_second_moment <- feasible_symmetric(
    cross_moment / degrees_of_freedom
  )
  raw <- estimate_working_covariances(
    residual, relative_floor = relative_floor
  )
  unstructured <- stabilize_working_covariance(
    corrected_second_moment, relative_floor
  )
  cs_raw <- matrix(
    if (tasks == 1L) 0 else mean(corrected_second_moment[
      row(corrected_second_moment) != col(corrected_second_moment)
    ]),
    tasks, tasks
  )
  diag(cs_raw) <- mean(diag(corrected_second_moment))
  compound_symmetry <- stabilize_working_covariance(
    cs_raw, relative_floor
  )
  list(
    raw = raw,
    corrected = list(
      unstructured = unstructured,
      compound_symmetry = compound_symmetry
    ),
    residual = residual,
    degrees_of_freedom = feasible_symmetric(degrees_of_freedom),
    raw_second_moment = raw_second_moment,
    corrected_second_moment = corrected_second_moment,
    empty_cell_count = fit$empty_cell_count
  )
}

gate_g_full_oracle_gls <- function(
    profile_index, outcome, geometry, task_weights, covariance,
    tolerance = 1e-10) {
  profile_index <- as.matrix(profile_index)
  outcome <- as.matrix(outcome)
  covariance <- as.matrix(covariance)
  n <- nrow(outcome)
  tasks <- ncol(outcome)
  profile_count <- geometry$profile_count
  coordinate_count <- ncol(geometry$riesz_lookup)
  parameter_count <- tasks * profile_count
  precision <- solve_spd(covariance, tolerance = tolerance)
  information <- matrix(0, parameter_count, parameter_count)
  rhs <- numeric(parameter_count)
  design_list <- vector("list", n)
  for (respondent in seq_len(n)) {
    design <- gate_g_profile_design(
      profile_index[respondent, ], tasks, profile_count
    )
    design_list[[respondent]] <- design
    information <- information + crossprod(design, precision %*% design)
    rhs <- rhs + drop(crossprod(
      design, precision %*% outcome[respondent, ]
    ))
  }
  inverse <- symmetric_pseudoinverse(information, tolerance)
  target_array <- gate_g_target_load(geometry, task_weights)
  target <- matrix(0, parameter_count, coordinate_count)
  for (tt in seq_len(tasks)) {
    rows <- (tt - 1L) * profile_count + seq_len(profile_count)
    target[rows, ] <- target_array[tt, , ]
  }
  estimability_residual <- max(abs(
    target - information %*% inverse$inverse %*% target
  ))
  feasible_assert(
    estimability_residual <= 100 * tolerance * max(1, max(abs(target))),
    sprintf("The finite-sample GLS target is not estimable (%.6g).",
            estimability_residual)
  )
  coefficient <- inverse$inverse %*% rhs
  estimate <- drop(crossprod(target, coefficient))
  loading <- array(0, c(n, tasks, coordinate_count))
  middle <- inverse$inverse %*% target
  for (respondent in seq_len(n)) {
    loading[respondent, , ] <-
      precision %*% design_list[[respondent]] %*% middle
  }
  conditional_covariance <- gate_g_conditional_covariance(
    loading, covariance
  )
  bound <- crossprod(target, inverse$inverse %*% target)
  feasible_assert(
    max(abs(conditional_covariance - bound)) <=
      500 * tolerance * max(1, max(abs(bound))),
    "Direct GLS covariance and information bound disagree."
  )
  list(
    estimate = estimate,
    covariance = conditional_covariance,
    linear_load = loading,
    information_rank = inverse$rank,
    estimability_residual = estimability_residual
  )
}
