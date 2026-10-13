#!/usr/bin/env Rscript

# Exact theorem search for the finite-sample two-fold nuisance remainder.
#
# This file is deliberately standalone. It does not source or modify the
# frozen Gate E2/G implementation. The model is the smallest saturated
# repeated-factorial problem that retains the Gate G mechanism:
#
#   * two tasks;
#   * one binary profile feature per task, assigned independently;
#   * task-specific saturated cell means, trained on the opposite fold;
#   * two equal folds;
#   * a scalar target averaging the two task-specific factorial contrasts;
#   * Gaussian respondent errors with a supplied 2 by 2 covariance matrix.
#
# Conditional on assignments, every estimator below is exactly linear in the
# Gaussian errors. We therefore integrate outcomes analytically. For small
# fold sizes, we enumerate every supported assignment in both folds and obtain
# the ex-ante marginal-fold and complete-estimator quadratic risks without
# Monte Carlo error.

theorem_assert <- function(ok, message) {
  if (!isTRUE(ok)) stop(message, call. = FALSE)
}

symmetric_part <- function(x) (x + t(x)) / 2

parse_arguments <- function(arguments) {
  out <- list(
    fold_size = 4L,
    rho = c(0, 0.35, 0.70, 0.90),
    output = NULL,
    quiet = FALSE
  )
  for (argument in arguments) {
    if (grepl("^--fold-size=", argument)) {
      out$fold_size <- as.integer(sub("^--fold-size=", "", argument))
    } else if (grepl("^--rho=", argument)) {
      values <- strsplit(sub("^--rho=", "", argument), ",", fixed = TRUE)[[1L]]
      out$rho <- as.numeric(values)
    } else if (grepl("^--output=", argument)) {
      out$output <- sub("^--output=", "", argument)
    } else if (identical(argument, "--quiet")) {
      out$quiet <- TRUE
    } else {
      stop(sprintf("Unknown argument: %s", argument), call. = FALSE)
    }
  }
  theorem_assert(
    length(out$fold_size) == 1L && is.finite(out$fold_size) &&
      out$fold_size == as.integer(out$fold_size) && out$fold_size >= 2L &&
      out$fold_size <= 5L,
    "`--fold-size` must be an integer from 2 through 5."
  )
  theorem_assert(
    length(out$rho) >= 1L && all(is.finite(out$rho)) &&
      all(abs(out$rho) < 1),
    "Every `--rho` value must be finite and strictly between -1 and 1."
  )
  out
}

enumerate_supported_assignments <- function(fold_size, task_count = 2L) {
  total_bits <- fold_size * task_count
  theorem_assert(total_bits <= 10L,
                 "The exact enumerator is intentionally limited to 10 bits.")
  integers <- 0:(2^total_bits - 1L)
  bit_matrix <- vapply(
    0:(total_bits - 1L),
    function(bit) bitwAnd(bitwShiftR(integers, bit), 1L),
    integer(length(integers))
  )
  assignments <- lapply(seq_len(nrow(bit_matrix)), function(index) {
    matrix(2 * bit_matrix[index, ] - 1,
           nrow = fold_size, ncol = task_count)
  })
  supported <- vapply(assignments, function(assignment) {
    all(vapply(seq_len(task_count), function(task) {
      length(unique(assignment[, task])) == 2L
    }, logical(1)))
  }, logical(1))
  assignments[supported]
}

cell_mean <- function(profile, outcome) {
  c(
    `-1` = mean(outcome[profile == -1]),
    `1` = mean(outcome[profile == 1])
  )
}

fold_estimate <- function(
    validation_profile, validation_outcome,
    training_profile, training_outcome,
    mixing, task_weights) {
  fold_size <- nrow(validation_profile)
  task_count <- ncol(validation_profile)
  predictions <- matrix(NA_real_, fold_size, task_count)
  plugin <- 0
  for (task in seq_len(task_count)) {
    means <- cell_mean(
      training_profile[, task], training_outcome[, task]
    )
    predictions[, task] <- means[as.character(validation_profile[, task])]
    plugin <- plugin + task_weights[[task]] *
      (means[["1"]] - means[["-1"]]) / 2
  }
  residual <- validation_outcome - predictions
  correction <- mean(rowSums(
    (residual %*% mixing) * validation_profile
  ))
  plugin + correction
}

complete_estimate <- function(
    fold_one_profile, fold_one_outcome,
    fold_two_profile, fold_two_outcome,
    mixing, task_weights) {
  0.5 * (
    fold_estimate(
      fold_one_profile, fold_one_outcome,
      fold_two_profile, fold_two_outcome,
      mixing, task_weights
    ) +
      fold_estimate(
        fold_two_profile, fold_two_outcome,
        fold_one_profile, fold_one_outcome,
        mixing, task_weights
      )
  )
}

# Outcome loadings for one fold estimate. Rows are ordered as validation
# respondents followed by training respondents. The expression is exact for
# supported saturated cells.
marginal_loading <- function(
    validation_profile, training_profile, mixing, task_weights) {
  fold_size <- nrow(validation_profile)
  task_count <- ncol(validation_profile)
  theorem_assert(
    identical(dim(validation_profile), dim(training_profile)) &&
      all(vapply(seq_len(task_count), function(task) {
        length(unique(training_profile[, task])) == 2L
      }, logical(1))),
    "Marginal loadings require aligned profiles and supported training cells."
  )
  loading <- matrix(0, 2L * fold_size, task_count)
  transformed_validation <- t(mixing %*% t(validation_profile))
  loading[seq_len(fold_size), ] <- transformed_validation / fold_size

  for (respondent in seq_len(fold_size)) {
    for (task in seq_len(task_count)) {
      level <- training_profile[respondent, task]
      count <- sum(training_profile[, task] == level)
      plugin_loading <- task_weights[[task]] * level / (2 * count)
      prediction_loading <- sum(
        transformed_validation[validation_profile[, task] == level, task]
      ) / (fold_size * count)
      loading[fold_size + respondent, task] <-
        plugin_loading - prediction_loading
    }
  }
  loading
}

# Complete two-fold estimator loadings, ordered as fold one then fold two.
complete_loading <- function(
    fold_one_profile, fold_two_profile, mixing, task_weights) {
  fold_size <- nrow(fold_one_profile)
  first <- marginal_loading(
    fold_one_profile, fold_two_profile, mixing, task_weights
  )
  second_raw <- marginal_loading(
    fold_two_profile, fold_one_profile, mixing, task_weights
  )
  second <- rbind(
    second_raw[fold_size + seq_len(fold_size), , drop = FALSE],
    second_raw[seq_len(fold_size), , drop = FALSE]
  )
  0.5 * (first + second)
}

mixing_basis_loadings <- function(
    loading_function, fold_one_profile, fold_two_profile, task_weights) {
  base <- diag(task_weights, nrow = 2L)
  direction_12 <- matrix(c(0, 0, 1, 0), 2L, 2L)
  direction_21 <- matrix(c(0, 1, 0, 0), 2L, 2L)
  load_zero <- loading_function(
    fold_one_profile, fold_two_profile, base, task_weights
  )
  load_12 <- loading_function(
    fold_one_profile, fold_two_profile, base + direction_12, task_weights
  ) - load_zero
  load_21 <- loading_function(
    fold_one_profile, fold_two_profile, base + direction_21, task_weights
  ) - load_zero
  array(
    c(load_zero, load_12, load_21),
    dim = c(nrow(load_zero), ncol(load_zero), 3L),
    dimnames = list(NULL, c("task_1", "task_2"),
                    c("fixed", "a12", "a21"))
  )
}

loading_gram_tensor <- function(loadings) {
  task_count <- dim(loadings)[[2L]]
  basis_count <- dim(loadings)[[3L]]
  out <- array(0, c(basis_count, basis_count, task_count, task_count))
  for (left_task in seq_len(task_count)) {
    left <- loadings[, left_task, , drop = FALSE]
    dim(left) <- c(dim(loadings)[[1L]], basis_count)
    for (right_task in seq_len(task_count)) {
      right <- loadings[, right_task, , drop = FALSE]
      dim(right) <- c(dim(loadings)[[1L]], basis_count)
      out[, , left_task, right_task] <- crossprod(left, right)
    }
  }
  out
}

quadratic_from_gram <- function(gram, covariance) {
  out <- matrix(0, dim(gram)[[1L]], dim(gram)[[2L]])
  for (left_task in seq_len(nrow(covariance))) {
    for (right_task in seq_len(ncol(covariance))) {
      out <- out + covariance[left_task, right_task] *
        gram[, , left_task, right_task]
    }
  }
  symmetric_part(out)
}

quadratic_optimum <- function(quadratic, tolerance = 1e-12) {
  quadratic <- symmetric_part(quadratic)
  curvature <- quadratic[-1L, -1L, drop = FALSE]
  linear <- quadratic[-1L, 1L]
  eigenvalues <- eigen(curvature, symmetric = TRUE, only.values = TRUE)$values
  theorem_assert(
    min(eigenvalues) > tolerance * max(1, max(abs(eigenvalues))),
    "The off-diagonal risk curvature is not positive definite."
  )
  optimum <- -solve(curvature, linear)
  names(optimum) <- c("a12", "a21")
  coefficient <- c(1, optimum)
  list(
    off_diagonal = optimum,
    risk = drop(crossprod(coefficient, quadratic %*% coefficient)),
    gradient = 2 * drop(linear + curvature %*% optimum),
    curvature = curvature
  )
}

quadratic_risk <- function(quadratic, mixing) {
  coefficient <- c(1, mixing[1, 2], mixing[2, 1])
  drop(crossprod(coefficient, quadratic %*% coefficient))
}

mixing_from_off_diagonal <- function(off_diagonal, task_weights) {
  matrix(
    c(task_weights[[1L]], off_diagonal[[2L]],
      off_diagonal[[1L]], task_weights[[2L]]),
    2L, 2L
  )
}

product_optimal_mixing <- function(covariance, task_weights) {
  precision <- solve(covariance)
  coefficient <- task_weights / diag(precision)
  precision %*% diag(coefficient, nrow = length(task_weights))
}

gate_g_marginal_formula_mixing <- function(
    covariance, fold_size, profile_count, task_weights) {
  probability <- 1 / profile_count
  counts <- seq_len(fold_size)
  reciprocal <- sum(
    dbinom(counts, fold_size, probability) / counts
  )
  support_probability <- 1 - (1 - probability)^fold_size
  gamma <- covariance * support_probability^2 / fold_size
  diag(gamma) <- diag(covariance) * reciprocal
  list(
    mixing = product_optimal_mixing(covariance + gamma, task_weights),
    gamma = gamma,
    support_probability = support_probability,
    reciprocal_count_moment = reciprocal
  )
}

accumulate_exact_grams <- function(states, task_weights, quiet = FALSE) {
  state_count <- length(states)
  pair_count <- state_count^2
  marginal_sum <- array(0, c(3L, 3L, 2L, 2L))
  complete_sum <- array(0, c(3L, 3L, 2L, 2L))
  pair_position <- 0L

  for (first_index in seq_len(state_count)) {
    for (second_index in seq_len(state_count)) {
      pair_position <- pair_position + 1L
      first <- states[[first_index]]
      second <- states[[second_index]]
      marginal_basis <- mixing_basis_loadings(
        marginal_loading, first, second, task_weights
      )
      complete_basis <- mixing_basis_loadings(
        complete_loading, first, second, task_weights
      )
      marginal_sum <- marginal_sum + loading_gram_tensor(marginal_basis)
      complete_sum <- complete_sum + loading_gram_tensor(complete_basis)
    }
    if (!quiet && state_count >= 500L && first_index %% 100L == 0L) {
      message(sprintf("Enumerated %d of %d first-fold states.",
                      first_index, state_count))
    }
  }
  list(
    marginal = marginal_sum / pair_count,
    complete = complete_sum / pair_count,
    state_count = state_count,
    pair_count = pair_count
  )
}

run_deterministic_checks <- function(states, task_weights, tolerance = 2e-11) {
  fold_size <- nrow(states[[1L]])
  expected_states <- (2^fold_size - 2)^2
  theorem_assert(length(states) == expected_states,
                 "Supported-state enumeration count is wrong.")
  first <- states[[min(2L, length(states))]]
  second <- states[[max(1L, length(states) - 1L)]]
  mixing <- matrix(c(0.5, -0.17, 0.23, 0.5), 2L, 2L)

  # The basis representation must reproduce an arbitrary calibrated matrix.
  basis <- mixing_basis_loadings(
    complete_loading, first, second, task_weights
  )
  reconstructed <- basis[, , 1L] +
    mixing[1, 2] * basis[, , 2L] +
    mixing[2, 1] * basis[, , 3L]
  direct <- complete_loading(first, second, mixing, task_weights)
  theorem_assert(max(abs(reconstructed - direct)) <= tolerance,
                 "Complete loading is not affine in the mixing weights.")

  # Exact linear representation and centering under a saturated mean.
  beta_zero <- c(0.4, -0.7)
  beta_one <- c(0.8, 0.3)
  mean_one <- sweep(first, 2L, beta_one, "*") +
    matrix(beta_zero, fold_size, 2L, byrow = TRUE)
  mean_two <- sweep(second, 2L, beta_one, "*") +
    matrix(beta_zero, fold_size, 2L, byrow = TRUE)
  error_one <- matrix(seq_len(2L * fold_size), fold_size, 2L) / 37
  error_two <- matrix(rev(seq_len(2L * fold_size)), fold_size, 2L) / 41
  target <- sum(task_weights * beta_one)
  estimate <- complete_estimate(
    first, mean_one + error_one,
    second, mean_two + error_two,
    mixing, task_weights
  )
  loading <- complete_loading(first, second, mixing, task_weights)
  represented <- target + sum(loading * rbind(error_one, error_two))
  theorem_assert(abs(estimate - represented) <= tolerance,
                 "Complete estimator and outcome loading disagree.")
  centered <- complete_estimate(
    first, mean_one, second, mean_two, mixing, task_weights
  )
  theorem_assert(abs(centered - target) <= tolerance,
                 "Supported complete estimator is not exactly centered.")

  # Swapping the folds must leave the complete loading unchanged after rows
  # are returned to the original respondent order.
  swapped_raw <- complete_loading(second, first, mixing, task_weights)
  swapped <- rbind(
    swapped_raw[fold_size + seq_len(fold_size), , drop = FALSE],
    swapped_raw[seq_len(fold_size), , drop = FALSE]
  )
  theorem_assert(max(abs(swapped - loading)) <= tolerance,
                 "Complete loadings are not invariant to fold-label swaps.")

  5L
}

summarize_rho <- function(
    rho, exact_grams, fold_size, task_weights,
    tolerance = 5e-11) {
  covariance <- matrix(c(1, rho, rho, 1), 2L, 2L)
  marginal_quadratic <- quadratic_from_gram(
    exact_grams$marginal, covariance
  )
  complete_quadratic <- quadratic_from_gram(
    exact_grams$complete, covariance
  )
  marginal_optimum <- quadratic_optimum(marginal_quadratic)
  complete_optimum <- quadratic_optimum(complete_quadratic)
  marginal_mixing <- mixing_from_off_diagonal(
    marginal_optimum$off_diagonal, task_weights
  )
  complete_mixing <- mixing_from_off_diagonal(
    complete_optimum$off_diagonal, task_weights
  )
  asymptotic_mixing <- product_optimal_mixing(covariance, task_weights)
  gate_g_formula <- gate_g_marginal_formula_mixing(
    covariance, fold_size, 2L, task_weights
  )
  ordinary_mixing <- diag(task_weights, nrow = 2L)

  theorem_assert(max(abs(marginal_optimum$gradient)) <= tolerance,
                 "Marginal optimum has a nonzero gradient.")
  theorem_assert(max(abs(complete_optimum$gradient)) <= tolerance,
                 "Complete optimum has a nonzero gradient.")
  theorem_assert(
    quadratic_risk(complete_quadratic, complete_mixing) <=
      quadratic_risk(complete_quadratic, marginal_mixing) + tolerance,
    "Complete optimum does not minimize complete-estimator risk."
  )
  theorem_assert(
    quadratic_risk(marginal_quadratic, marginal_mixing) <=
      quadratic_risk(marginal_quadratic, complete_mixing) + tolerance,
    "Marginal optimum does not minimize marginal-fold risk."
  )
  if (abs(rho) <= tolerance) {
    theorem_assert(
      max(abs(marginal_optimum$off_diagonal)) <= tolerance &&
        max(abs(complete_optimum$off_diagonal)) <= tolerance,
      "Independent task errors should give diagonal optimal mixing."
    )
  }

  methods <- list(
    ordinary = ordinary_mixing,
    asymptotic_hstar = asymptotic_mixing,
    gate_g_marginal_formula = gate_g_formula$mixing,
    exact_supported_marginal = marginal_mixing,
    exact_complete = complete_mixing
  )
  do.call(rbind, lapply(names(methods), function(method) {
    mixing <- methods[[method]]
    data.frame(
      fold_size = fold_size,
      supported_states_per_fold = exact_grams$state_count,
      supported_assignment_pairs = exact_grams$pair_count,
      rho = rho,
      method = method,
      a12 = mixing[1, 2],
      a21 = mixing[2, 1],
      marginal_risk = quadratic_risk(marginal_quadratic, mixing),
      complete_risk = quadratic_risk(complete_quadratic, mixing),
      complete_excess_percent = 100 * (
        quadratic_risk(complete_quadratic, mixing) /
          complete_optimum$risk - 1
      ),
      marginal_excess_percent = 100 * (
        quadratic_risk(marginal_quadratic, mixing) /
          marginal_optimum$risk - 1
      ),
      stringsAsFactors = FALSE
    )
  }))
}

run_search <- function(options) {
  task_weights <- c(0.5, 0.5)
  states <- enumerate_supported_assignments(options$fold_size, 2L)
  check_count <- run_deterministic_checks(states, task_weights)
  start_time <- proc.time()[["elapsed"]]
  exact_grams <- accumulate_exact_grams(
    states, task_weights, options$quiet
  )
  elapsed <- proc.time()[["elapsed"]] - start_time
  proportionality_residual <- max(abs(
    exact_grams$complete - exact_grams$marginal / 2
  ))
  theorem_assert(
    proportionality_residual <= 1e-10,
    sprintf(
      paste0("The complete two-fold risk is not one-half the marginal risk ",
             "(Gram residual %.6g)."),
      proportionality_residual
    )
  )
  results <- do.call(rbind, lapply(options$rho, function(rho) {
    summarize_rho(
      rho, exact_grams, options$fold_size, task_weights
    )
  }))
  results$complete_half_marginal_gram_residual <-
    proportionality_residual

  theorem_assert(all(is.finite(as.matrix(results[, c(
    "a12", "a21", "marginal_risk", "complete_risk",
    "complete_excess_percent", "marginal_excess_percent"
  )]))), "The exact search produced nonfinite results.")
  check_count <- check_count + 7L

  if (!is.null(options$output)) {
    output_directory <- dirname(options$output)
    if (!dir.exists(output_directory)) {
      dir.create(output_directory, recursive = TRUE, showWarnings = FALSE)
    }
    write.csv(results, options$output, row.names = FALSE)
  }

  if (!options$quiet) {
    print(results, row.names = FALSE, digits = 8)
    cat(sprintf(
      paste0("\nExact enumeration: %d supported states/fold, ",
             "%d ordered assignment pairs.\n"),
      exact_grams$state_count, exact_grams$pair_count
    ))
    cat(sprintf("Deterministic checks passed: %d.\n", check_count))
    cat(sprintf(
      "Maximum |complete Gram - marginal Gram / 2|: %.6g.\n",
      proportionality_residual
    ))
    cat(sprintf("Enumeration runtime: %.3f seconds.\n", elapsed))
  }

  invisible(list(
    results = results,
    grams = exact_grams,
    checks = check_count,
    complete_half_marginal_gram_residual = proportionality_residual,
    runtime_seconds = elapsed
  ))
}

if (sys.nframe() == 0L) {
  run_search(parse_arguments(commandArgs(trailingOnly = TRUE)))
}
