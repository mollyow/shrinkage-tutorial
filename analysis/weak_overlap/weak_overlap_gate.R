#!/usr/bin/env Rscript

# Deterministic weak-overlap theorem gate.
#
# The calculation uses the target-Walsh basis on {-1,1}^J, a product
# assignment distribution with E_g(A_j) = delta_j, and the population
# Gaussian regression experiment.  It verifies all closed forms against
# direct matrix algebra at small J and writes the frozen risk grid specified
# in docs/weak-overlap-gate-protocol.md.

gate_assert <- function(ok, message) {
  if (!isTRUE(ok)) stop(message, call. = FALSE)
}

script_directory <- function() {
  arguments <- commandArgs(trailingOnly = FALSE)
  file_argument <- grep("^--file=", arguments, value = TRUE)
  if (length(file_argument) == 1L) {
    return(dirname(normalizePath(
      sub("^--file=", "", file_argument), mustWork = TRUE
    )))
  }
  normalizePath("analysis/weak_overlap", mustWork = TRUE)
}

parse_arguments <- function(arguments) {
  out <- list(
    output = NULL,
    validation_output = NULL,
    summary_output = NULL,
    quiet = FALSE
  )
  for (argument in arguments) {
    if (grepl("^--output=", argument)) {
      out$output <- sub("^--output=", "", argument)
    } else if (grepl("^--validation-output=", argument)) {
      out$validation_output <- sub(
        "^--validation-output=", "", argument
      )
    } else if (grepl("^--summary-output=", argument)) {
      out$summary_output <- sub("^--summary-output=", "", argument)
    } else if (identical(argument, "--quiet")) {
      out$quiet <- TRUE
    } else {
      stop(sprintf("Unknown argument: %s", argument), call. = FALSE)
    }
  }
  out
}

profile_index <- function(bits) {
  as.integer(1L + sum(as.integer(bits) * 2L^(seq_along(bits) - 1L)))
}

subset_bits <- function(J) {
  ids <- 0:(2^J - 1L)
  bits <- vapply(
    0:(J - 1L),
    function(bit) bitwAnd(bitwShiftR(ids, bit), 1L),
    integer(length(ids))
  )
  storage.mode(bits) <- "integer"
  bits
}

target_gram <- function(bits, delta) {
  dimension <- nrow(bits)
  gram <- matrix(1, dimension, dimension)
  for (j in seq_along(delta)) {
    differs <- outer(bits[, j], bits[, j], FUN = "!=")
    gram[differs] <- gram[differs] * delta[[j]]
  }
  gram
}

basis_transform <- function(bits, delta) {
  # gamma = M beta, where gamma is the coefficient vector in the
  # assignment-orthonormal product basis and beta is the target-Walsh vector.
  dimension <- nrow(bits)
  variance <- 1 - delta^2
  transform <- matrix(0, dimension, dimension)
  for (u in seq_len(dimension)) {
    U <- bits[u, ] == 1L
    scale <- sqrt(prod(variance[U]))
    if (!any(U)) scale <- 1
    supersets <- rowSums(bits[, U, drop = FALSE]) == sum(U)
    candidates <- which(supersets)
    for (r in candidates) {
      remainder <- bits[r, ] == 1L & !U
      shift <- prod(delta[remainder])
      if (!any(remainder)) shift <- 1
      transform[u, r] <- scale * shift
    }
  }
  transform
}

elementary_sum <- function(values, maximum_order) {
  maximum_order <- min(maximum_order, length(values))
  coefficients <- c(1, rep(0, maximum_order))
  if (maximum_order == 0L || length(values) == 0L) return(1)
  for (value in values) {
    for (degree in rev(seq_len(maximum_order))) {
      coefficients[degree + 1L] <- coefficients[degree + 1L] +
        value * coefficients[degree]
    }
  }
  sum(coefficients)
}

exact_degree_variance_factor <- function(delta, S, q) {
  s <- length(S)
  gate_assert(q >= s, "The fitted degree must contain the target order.")
  variance <- 1 - delta^2
  complement <- setdiff(seq_along(delta), S)
  rho <- delta[complement]^2 / variance[complement]
  prod(1 / variance[S]) * elementary_sum(rho, q - s)
}

truncation_load <- function(bits, delta, S, q) {
  s <- length(S)
  k <- q - s
  in_S <- rowSums(bits[, S, drop = FALSE]) == s
  extra <- rowSums(bits[, -S, drop = FALSE])
  if (length(S) == ncol(bits)) extra <- rep(0L, nrow(bits))
  load <- numeric(nrow(bits))
  target <- rep(0L, ncol(bits))
  target[S] <- 1L
  load[profile_index(target)] <- 1
  contaminated <- which(in_S & extra > k)
  for (index in contaminated) {
    remainder <- which(bits[index, ] == 1L & !(seq_len(ncol(bits)) %in% S))
    load[index] <- (-1)^k * choose(length(remainder) - 1L, k) *
      prod(delta[remainder])
  }
  load
}

reduced_inverse_gram <- function(J, delta) {
  # Operator on vectors invariant to permutations of factors 2,...,J.
  # Groups are ordered by target-factor inclusion b=0,1, then by the number r
  # of included non-target factors.  Entries are group values, not sums.
  N <- J - 1L
  variance <- 1 - delta^2
  non_target <- matrix(0, N + 1L, N + 1L)
  for (r in 0:N) {
    for (t in 0:N) {
      lower <- max(0L, r + t - N)
      upper <- min(r, t)
      value <- 0
      for (h in lower:upper) {
        value <- value + choose(r, h) * choose(N - r, t - h) *
          (-delta)^(r + t - 2L * h)
      }
      non_target[r + 1L, t + 1L] <- value / variance^N
    }
  }
  target <- matrix(c(1, -delta, -delta, 1), 2L, 2L) / variance
  kronecker(target, non_target)
}

group_metadata <- function(J) {
  N <- J - 1L
  data.frame(
    target_included = rep(0:1, each = N + 1L),
    other_order = rep(0:N, times = 2L),
    multiplicity = rep(choose(N, 0:N), times = 2L)
  )
}

minimax_group_rule <- function(J, delta, n, kappa, R = 1, sigma = 1) {
  metadata <- group_metadata(J)
  inverse_gram <- reduced_inverse_gram(J, delta)
  order <- metadata$target_included + metadata$other_order
  weights <- kappa^(2 * pmax(order - 1L, 0L))
  inverse_weights <- 1 / weights
  target <- numeric(nrow(metadata))
  target[J + 1L] <- 1 # b=1, r=0
  noise <- sigma^2 / n
  system <- noise * inverse_gram + R^2 * diag(inverse_weights)
  rhs <- R^2 * inverse_weights * target
  load <- as.numeric(solve(system, rhs))
  represented <- as.numeric(inverse_gram %*% load)
  variance_mse <- noise * sum(
    metadata$multiplicity * load * represented
  )
  bias_mse <- R^2 * sum(
    metadata$multiplicity * (load - target)^2 * inverse_weights
  )
  list(
    load = load,
    variance = variance_mse,
    maximum_squared_bias = bias_mse,
    mse = variance_mse + bias_mse,
    maximum_bias = sqrt(bias_mse),
    standard_error = sqrt(variance_mse),
    half_length = sqrt(bias_mse) + qnorm(0.975) * sqrt(variance_mse)
  )
}

truncated_group_rule <- function(
    J, delta, q, n, kappa, R = 1, sigma = 1) {
  metadata <- group_metadata(J)
  s <- 1L
  k <- q - s
  load <- numeric(nrow(metadata))
  target <- numeric(nrow(metadata))
  target[J + 1L] <- 1
  load[J + 1L] <- 1
  for (r in 0:(J - 1L)) {
    index <- J + 1L + r
    if (r > k) {
      load[index] <- (-1)^k * choose(r - 1L, k) * delta^r
    } else if (r > 0L) {
      load[index] <- 0
    }
  }
  order <- metadata$target_included + metadata$other_order
  weights <- kappa^(2 * pmax(order - 1L, 0L))
  inverse_weights <- 1 / weights
  variance_factor <- exact_degree_variance_factor(
    rep(delta, J), S = 1L, q = q
  )
  variance_mse <- sigma^2 / n * variance_factor
  bias_mse <- R^2 * sum(
    metadata$multiplicity * (load - target)^2 * inverse_weights
  )
  unweighted_tail_amplification <- sqrt(sum(
    metadata$multiplicity * (load - target)^2 * (order > q)
  ))
  list(
    load = load,
    variance = variance_mse,
    maximum_squared_bias = bias_mse,
    mse = variance_mse + bias_mse,
    maximum_bias = sqrt(bias_mse),
    standard_error = sqrt(variance_mse),
    half_length = sqrt(bias_mse) + qnorm(0.975) * sqrt(variance_mse),
    unweighted_tail_amplification = unweighted_tail_amplification
  )
}

expand_group_values <- function(bits, values) {
  J <- ncol(bits)
  other_order <- if (J == 1L) rep(0L, nrow(bits)) else
    rowSums(bits[, -1L, drop = FALSE])
  index <- bits[, 1L] * J + other_order + 1L
  values[index]
}

run_validation <- function() {
  residuals <- list()
  add_residual <- function(check, value, J, target_order, q, shift) {
    residuals[[length(residuals) + 1L]] <<- data.frame(
      check = check,
      J = J,
      target_order = target_order,
      q = q,
      shift = shift,
      residual = value
    )
  }

  for (J in 2:8) {
    bits <- subset_bits(J)
    shift_cases <- list(
      equal_02 = rep(0.2, J),
      equal_05 = rep(0.5, J),
      unequal = seq(-0.35, 0.45, length.out = J)
    )
    for (case_name in names(shift_cases)) {
      delta <- shift_cases[[case_name]]
      gram <- target_gram(bits, delta)
      transform <- basis_transform(bits, delta)
      add_residual(
        "basis_crossproduct", max(abs(crossprod(transform) - gram)),
        J, 0L, NA_integer_, case_name
      )
      inverse_gram <- solve(gram)
      full_closed <- prod(1 / (1 - delta^2))
      add_residual(
        "full_riesz_variance", max(abs(diag(inverse_gram) - full_closed)),
        J, 0L, J, case_name
      )

      target_sets <- list(1L)
      if (J >= 2L) target_sets[[2L]] <- c(1L, 2L)
      orders <- rowSums(bits)
      for (S in target_sets) {
        s <- length(S)
        target_bits <- rep(0L, J)
        target_bits[S] <- 1L
        target_index <- profile_index(target_bits)
        for (q in s:min(J, s + 2L)) {
          low <- which(orders <= q)
          target_row <- match(target_index, low)
          low_inverse <- solve(gram[low, low, drop = FALSE])
          direct_variance <- low_inverse[target_row, target_row]
          closed_variance <- exact_degree_variance_factor(delta, S, q)
          add_residual(
            "exact_degree_variance", abs(direct_variance - closed_variance),
            J, s, q, case_name
          )
          direct_load <- as.numeric(
            low_inverse[target_row, , drop = FALSE] %*%
              gram[low, , drop = FALSE]
          )
          closed_load <- truncation_load(bits, delta, S, q)
          add_residual(
            "truncation_contamination",
            max(abs(direct_load - closed_load)),
            J, s, q, case_name
          )
        }
      }
    }

    # The symmetry-reduced minimax solve is checked against the complete
    # 2^J-dimensional normal equations.
    delta <- 0.5
    gram <- target_gram(bits, rep(delta, J))
    inverse_gram <- solve(gram)
    target <- numeric(nrow(bits))
    target[profile_index(c(1L, rep(0L, J - 1L)))] <- 1
    for (kappa in c(1.5, 2, 4)) {
      order <- rowSums(bits)
      inverse_weights <- kappa^(-2 * pmax(order - 1L, 0L))
      noise <- 1 / 1000
      full_load <- as.numeric(solve(
        noise * inverse_gram + diag(inverse_weights),
        inverse_weights * target
      ))
      reduced <- minimax_group_rule(J, delta, 1000, kappa)
      expanded <- expand_group_values(bits, reduced$load)
      add_residual(
        "minimax_symmetry_reduction", max(abs(full_load - expanded)),
        J, 1L, NA_integer_, sprintf("equal_05_kappa_%s", kappa)
      )
    }
  }

  output <- do.call(rbind, residuals)
  maxima <- aggregate(residual ~ check, output, max)
  gate_assert(
    max(output$residual) < 1e-10,
    sprintf(
      "A deterministic identity failed: maximum residual %.3g.",
      max(output$residual)
    )
  )
  list(detail = output, maxima = maxima)
}

method_row <- function(
    J, n, delta, kappa, method, q, fit, full_variance_factor) {
  trivial_half_length <- 1
  data.frame(
    J = J,
    profiles = 2^J,
    n = n,
    profiles_per_observation = 2^J / n,
    delta = delta,
    assignment_probability_min = ((1 - abs(delta)) / 2)^J,
    density_ratio_second_moment = (1 - delta^2)^(-J),
    effective_sample_size_riesz = n * (1 - delta^2)^J,
    kappa = kappa,
    method = method,
    q = if (is.null(q)) NA_integer_ else q,
    variance = fit$variance,
    maximum_squared_bias = fit$maximum_squared_bias,
    worst_case_mse = fit$mse,
    standard_error = fit$standard_error,
    maximum_bias = fit$maximum_bias,
    bias_aware_half_length_95 = fit$half_length,
    shorter_than_trivial_interval = fit$half_length < trivial_half_length,
    risk_ratio_to_zero = fit$mse,
    risk_ratio_to_full_riesz = fit$mse /
      (full_variance_factor / n),
    unweighted_tail_amplification = if (
      is.null(fit$unweighted_tail_amplification)
    ) NA_real_ else fit$unweighted_tail_amplification,
    tail_radius_for_root_n_bias = if (
      is.null(fit$unweighted_tail_amplification) ||
        fit$unweighted_tail_amplification == 0
    ) NA_real_ else 1 / sqrt(n) / fit$unweighted_tail_amplification
  )
}

run_grid <- function() {
  grid <- expand.grid(
    J = c(4L, 6L, 8L, 10L, 12L),
    n = c(250L, 1000L, 4000L, 16000L),
    delta = c(0, 0.2, 0.5, 0.7),
    kappa = c(1.5, 2, 4),
    KEEP.OUT.ATTRS = FALSE,
    stringsAsFactors = FALSE
  )
  rows <- vector("list", nrow(grid) * 9L)
  cursor <- 0L
  for (cell in seq_len(nrow(grid))) {
    J <- grid$J[[cell]]
    n <- grid$n[[cell]]
    delta <- grid$delta[[cell]]
    kappa <- grid$kappa[[cell]]
    full_factor <- (1 - delta^2)^(-J)

    zero <- list(
      variance = 0,
      maximum_squared_bias = 1,
      mse = 1,
      standard_error = 0,
      maximum_bias = 1,
      half_length = 1
    )
    cursor <- cursor + 1L
    rows[[cursor]] <- method_row(
      J, n, delta, kappa, "zero", NULL, zero, full_factor
    )

    full <- list(
      variance = full_factor / n,
      maximum_squared_bias = 0,
      mse = full_factor / n,
      standard_error = sqrt(full_factor / n),
      maximum_bias = 0,
      half_length = qnorm(0.975) * sqrt(full_factor / n)
    )
    cursor <- cursor + 1L
    rows[[cursor]] <- method_row(
      J, n, delta, kappa, "full_riesz", J, full, full_factor
    )

    for (q in 1:min(J, 5L)) {
      truncated <- truncated_group_rule(J, delta, q, n, kappa)
      cursor <- cursor + 1L
      rows[[cursor]] <- method_row(
        J, n, delta, kappa, "truncated", q, truncated, full_factor
      )
    }

    minimax <- minimax_group_rule(J, delta, n, kappa)
    cursor <- cursor + 1L
    rows[[cursor]] <- method_row(
      J, n, delta, kappa, "ellipsoid_minimax_linear", NULL,
      minimax, full_factor
    )
  }
  rows <- rows[seq_len(cursor)]
  do.call(rbind, rows)
}

summarize_grid <- function(results) {
  keys <- c("J", "n", "delta", "kappa")
  minimax <- results[
    results$method == "ellipsoid_minimax_linear", , drop = FALSE
  ]
  truncated <- results[results$method == "truncated", , drop = FALSE]
  best_truncation <- aggregate(
    worst_case_mse ~ J + n + delta + kappa,
    data = truncated,
    FUN = min
  )
  names(best_truncation)[[5L]] <- "best_truncation_mse"
  comparison <- merge(minimax, best_truncation, by = keys)
  comparison$ratio_to_best_truncation <-
    comparison$worst_case_mse / comparison$best_truncation_mse
  weak <- comparison$density_ratio_second_moment >= 10
  headroom <-
    comparison$ratio_to_best_truncation <= 0.9 &
    comparison$risk_ratio_to_full_riesz <= 0.9
  optimistic <- comparison$profiles > comparison$n

  gate_assert(nrow(comparison) == 240L, "The frozen grid must have 240 cells.")
  gate_assert(nrow(results) == 1872L, "The frozen grid must have 1,872 rows.")

  data.frame(
    metric = c(
      "design_cells",
      "method_rows",
      "weak_overlap_cells",
      "weak_10pct_better_than_full_and_best_truncation",
      "weak_shorter_than_trivial_interval",
      "weak_wins_with_profiles_greater_than_n",
      "minimum_ratio_to_best_truncation",
      "median_weak_ratio_to_best_truncation"
    ),
    value = c(
      nrow(comparison),
      nrow(results),
      sum(weak),
      sum(weak & headroom),
      sum(weak & comparison$shorter_than_trivial_interval),
      sum(weak & headroom & optimistic),
      min(comparison$ratio_to_best_truncation),
      median(comparison$ratio_to_best_truncation[weak])
    )
  )
}

main <- function() {
  options(stringsAsFactors = FALSE, digits = 15)
  arguments <- parse_arguments(commandArgs(trailingOnly = TRUE))
  directory <- script_directory()
  output_path <- if (is.null(arguments$output)) {
    file.path(directory, "weak_overlap_risk_grid.csv")
  } else arguments$output
  validation_path <- if (is.null(arguments$validation_output)) {
    file.path(directory, "weak_overlap_validation.csv")
  } else arguments$validation_output
  summary_path <- if (is.null(arguments$summary_output)) {
    file.path(directory, "weak_overlap_gate_summary.csv")
  } else arguments$summary_output

  started <- proc.time()[[3L]]
  validation <- run_validation()
  results <- run_grid()
  summary <- summarize_grid(results)
  write.csv(validation$detail, validation_path, row.names = FALSE)
  write.csv(results, output_path, row.names = FALSE)
  write.csv(summary, summary_path, row.names = FALSE)

  if (!arguments$quiet) {
    print(validation$maxima, row.names = FALSE, digits = 6)
    cat("\n")
    print(summary, row.names = FALSE, digits = 8)
    cat(sprintf(
      "\nWrote %d risk rows to %s.\n",
      nrow(results), normalizePath(output_path, mustWork = TRUE)
    ))
    cat(sprintf(
      "Wrote %d validation rows to %s.\n",
      nrow(validation$detail),
      normalizePath(validation_path, mustWork = TRUE)
    ))
    cat(sprintf(
      "Wrote %d gate metrics to %s.\n",
      nrow(summary), normalizePath(summary_path, mustWork = TRUE)
    ))
    cat(sprintf("Elapsed: %.2f seconds.\n", proc.time()[[3L]] - started))
  }
  invisible(list(
    results = results,
    validation = validation,
    summary = summary
  ))
}

if (sys.nframe() == 0L) main()
