#!/usr/bin/env Rscript

# Deterministic minimal support-augmentation gate.
#
# The script verifies the quotient-space characterization, exact Hamming-layer
# augmentation, one-run c-variance updates, parity de-aliasing, one-point slice
# augmentation, and forbidden-pair cubature cases. It also gives a factorial
# counterexample to blanket supermodularity and compares exhaustive with greedy
# robust augmentation in a frozen J=3 example.

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
  normalizePath("analysis/support_augmentation", mustWork = TRUE)
}

parse_arguments <- function(arguments) {
  out <- list(
    validation_output = NULL,
    parity_output = NULL,
    layers_output = NULL,
    risk_output = NULL,
    special_output = NULL,
    summary_output = NULL,
    quiet = FALSE
  )
  fields <- c(
    "validation", "parity", "layers", "risk", "special", "summary"
  )
  for (argument in arguments) {
    matched <- FALSE
    for (field in fields) {
      prefix <- sprintf("^--%s-output=", field)
      if (grepl(prefix, argument)) {
        out[[paste0(field, "_output")]] <- sub(prefix, "", argument)
        matched <- TRUE
        break
      }
    }
    if (!matched && identical(argument, "--quiet")) {
      out$quiet <- TRUE
      matched <- TRUE
    }
    if (!matched) stop(sprintf("Unknown argument: %s", argument), call. = FALSE)
  }
  out
}

binary_geometry <- function(J) {
  ids <- 0:(2^J - 1L)
  bits <- vapply(
    0:(J - 1L),
    function(bit) bitwAnd(bitwShiftR(ids, bit), 1L),
    integer(length(ids))
  )
  storage.mode(bits) <- "integer"
  profiles <- 2L * bits - 1L
  walsh <- matrix(1, nrow(bits), nrow(bits))
  for (j in seq_len(J)) {
    selected <- bits[, j] == 1L
    walsh[, selected] <- walsh[, selected, drop = FALSE] * profiles[, j]
  }
  list(
    bits = bits,
    profiles = profiles,
    walsh = walsh,
    order = rowSums(bits),
    layer = rowSums(bits)
  )
}

matrix_rank <- function(x, tolerance = 1e-10) {
  if (!length(x) || nrow(x) == 0L || ncol(x) == 0L) return(0L)
  values <- svd(x, nu = 0L, nv = 0L)$d
  sum(values > tolerance * max(1, values[[1L]]))
}

null_basis <- function(x, tolerance = 1e-10) {
  p <- ncol(x)
  if (nrow(x) == 0L) return(diag(p))
  decomposition <- eigen(crossprod(x), symmetric = TRUE)
  scale <- max(1, max(abs(decomposition$values)))
  null <- decomposition$values < tolerance * scale
  decomposition$vectors[, null, drop = FALSE]
}

rowspace_residual <- function(x, target) {
  basis <- null_basis(x)
  if (ncol(basis) == 0L) return(0)
  sqrt(sum(crossprod(basis, target)^2))
}

is_estimable <- function(x, target, tolerance = 1e-10) {
  rowspace_residual(x, target) < tolerance
}

quotient_estimable <- function(x0, candidates, selected, target) {
  basis <- null_basis(x0)
  if (ncol(basis) == 0L) return(TRUE)
  target_q <- as.numeric(crossprod(basis, target))
  if (sqrt(sum(target_q^2)) < 1e-10) return(TRUE)
  if (!length(selected)) return(FALSE)
  dictionary <- candidates[selected, , drop = FALSE] %*% basis
  matrix_rank(rbind(dictionary, target_q)) == matrix_rank(dictionary)
}

mask_members <- function(mask, size) {
  which(vapply(
    seq_len(size),
    function(index) bitwAnd(mask, bitwShiftL(1L, index - 1L)) != 0L,
    logical(1L)
  ))
}

mask_size <- function(mask, size) length(mask_members(mask, size))

minimum_target_augmentation <- function(x0, candidates, target) {
  if (is_estimable(x0, target)) {
    return(list(size = 0L, selected = integer(0)))
  }
  candidate_count <- nrow(candidates)
  for (budget in seq_len(candidate_count)) {
    combinations <- combn(candidate_count, budget)
    for (column in seq_len(ncol(combinations))) {
      selected <- combinations[, column]
      if (quotient_estimable(x0, candidates, selected, target)) {
        direct <- is_estimable(
          rbind(x0, candidates[selected, , drop = FALSE]), target
        )
        gate_assert(direct, "Quotient and direct estimability disagree.")
        return(list(size = budget, selected = selected))
      }
    }
  }
  list(size = Inf, selected = integer(0))
}

slice_union_rank <- function(J, q, layers) {
  maximum_degree <- min(q, floor(J / 2))
  total <- 0
  for (degree in 0:maximum_degree) {
    harmonic_dimension <- choose(J, degree) - if (degree == 0L) {
      0
    } else choose(J, degree - 1L)
    radial_dimension <- min(q, J - degree) - degree + 1L
    eligible <- sum(layers >= degree & layers <= J - degree)
    total <- total + harmonic_dimension * min(radial_dimension, eligible)
  }
  total
}

model_dimension <- function(J, q) sum(choose(J, 0:q))

minimum_layer_augmentation <- function(J, q, initial, candidates) {
  target_rank <- model_dimension(J, q)
  initial <- sort(unique(initial))
  candidates <- sort(unique(setdiff(candidates, initial)))
  if (slice_union_rank(J, q, initial) == target_rank) {
    return(list(size = 0L, selected = integer(0), feasible = TRUE))
  }
  if (slice_union_rank(J, q, c(initial, candidates)) < target_rank) {
    return(list(size = Inf, selected = integer(0), feasible = FALSE))
  }
  depth <- pmin(candidates, J - candidates)
  order <- order(-depth, candidates)
  sorted <- candidates[order]
  for (budget in seq_along(sorted)) {
    selected <- sorted[seq_len(budget)]
    if (slice_union_rank(J, q, c(initial, selected)) == target_rank) {
      return(list(size = budget, selected = selected, feasible = TRUE))
    }
  }
  stop("A feasible layer augmentation was not found.", call. = FALSE)
}

exhaustive_layer_augmentation <- function(J, q, initial, candidates) {
  target_rank <- model_dimension(J, q)
  initial <- sort(unique(initial))
  candidates <- sort(unique(setdiff(candidates, initial)))
  if (slice_union_rank(J, q, initial) == target_rank) return(0L)
  if (slice_union_rank(J, q, c(initial, candidates)) < target_rank) return(Inf)
  for (budget in seq_along(candidates)) {
    combinations <- combn(candidates, budget)
    for (column in seq_len(ncol(combinations))) {
      if (slice_union_rank(
        J, q, c(initial, combinations[, column])
      ) == target_rank) return(budget)
    }
  }
  Inf
}

layer_cases <- function(J) {
  all_layers <- 0:J
  central_depth <- floor(J / 2)
  central <- all_layers[pmin(all_layers, J - all_layers) == central_depth]
  list(
    empty_all = list(initial = integer(0), candidates = all_layers),
    extremes_all = list(initial = c(0L, J), candidates = all_layers),
    central_one_all = list(
      initial = central[[1L]], candidates = all_layers
    ),
    alternating_all = list(
      initial = all_layers[all_layers %% 2L == 0L],
      candidates = all_layers
    ),
    extremes_no_deepest = list(
      initial = c(0L, J),
      candidates = setdiff(all_layers, central)
    )
  )
}

run_layer_gate <- function() {
  rows <- list()
  cursor <- 0L
  maximum_rank_residual <- 0
  for (J in 3:8) {
    geometry <- binary_geometry(J)
    for (q in 0:min(3L, J)) {
      columns <- which(geometry$order <= q)
      for (case_name in names(layer_cases(J))) {
        case <- layer_cases(J)[[case_name]]
        greedy <- minimum_layer_augmentation(
          J, q, case$initial, case$candidates
        )
        exact <- exhaustive_layer_augmentation(
          J, q, case$initial, case$candidates
        )
        gate_assert(
          identical(greedy$size, exact) ||
            (is.infinite(greedy$size) && is.infinite(exact)),
          "Depth-greedy and exhaustive layer augmentation disagree."
        )
        final_layers <- sort(unique(c(case$initial, greedy$selected)))
        support <- which(geometry$layer %in% final_layers)
        direct_rank <- if (length(support)) {
          matrix_rank(geometry$walsh[support, columns, drop = FALSE])
        } else 0L
        formula_rank <- slice_union_rank(J, q, final_layers)
        maximum_rank_residual <- max(
          maximum_rank_residual, abs(direct_rank - formula_rank)
        )
        cursor <- cursor + 1L
        rows[[cursor]] <- data.frame(
          J = J,
          q = q,
          case = case_name,
          initial_layers = paste(case$initial, collapse = ":"),
          candidate_layers = paste(case$candidates, collapse = ":"),
          feasible = greedy$feasible,
          minimum_added_layers = greedy$size,
          selected_layers = paste(greedy$selected, collapse = ":"),
          final_rank = direct_rank,
          model_dimension = model_dimension(J, q)
        )
      }
    }
  }
  list(rows = do.call(rbind, rows), rank_residual = maximum_rank_residual)
}

run_parity_gate <- function() {
  cases <- data.frame(
    J = c(3L, 4L, 4L, 5L),
    q = c(2L, 2L, 3L, 3L)
  )
  rows <- list()
  validation <- numeric(nrow(cases))
  for (case_index in seq_len(nrow(cases))) {
    J <- cases$J[[case_index]]
    q <- cases$q[[case_index]]
    geometry <- binary_geometry(J)
    columns <- which(geometry$order <= q)
    design <- geometry$walsh[, columns, drop = FALSE]
    plus <- which(apply(geometry$profiles, 1L, prod) == 1)
    minus <- setdiff(seq_len(nrow(design)), plus)
    x0 <- design[plus, , drop = FALSE]
    candidates <- design[minus, , drop = FALSE]
    basis <- null_basis(x0)
    residual_squared <- if (ncol(basis)) rowSums(basis^2) else
      rep(0, ncol(x0))
    nonestimable <- which(residual_squared > 1e-10)
    target_index <- nonestimable[
      which.min(geometry$order[columns[nonestimable]])
    ]
    target <- diag(ncol(x0))[, target_index]
    solution <- minimum_target_augmentation(x0, candidates, target)
    augmented <- rbind(
      x0, candidates[solution$selected, , drop = FALSE]
    )
    validation[[case_index]] <- rowspace_residual(augmented, target)
    rows[[case_index]] <- data.frame(
      J = J,
      q = q,
      initial_profiles = length(plus),
      candidate_profiles = length(minus),
      initial_rank = matrix_rank(x0),
      nullity = ncol(x0) - matrix_rank(x0),
      target_order = geometry$order[columns[[target_index]]],
      minimum_added_profiles = solution$size,
      selected_candidate_indices = paste(solution$selected, collapse = ":")
    )
  }
  list(rows = do.call(rbind, rows), residual = max(validation))
}

c_variance <- function(information, target) {
  as.numeric(crossprod(target, solve(information, target)))
}

run_variance_checks <- function() {
  geometry <- binary_geometry(3L)
  columns <- which(geometry$order <= 1L)
  design <- geometry$walsh[, columns, drop = FALSE]
  base <- c(1L, 2L, 3L, 5L)
  candidates <- setdiff(seq_len(nrow(design)), base)
  information <- crossprod(design[base, , drop = FALSE])
  maximum_update_residual <- 0
  for (candidate in candidates) {
    x <- design[candidate, ]
    updated <- information + tcrossprod(x)
    inverse <- solve(information)
    for (target_index in seq_len(ncol(design))) {
      target <- diag(ncol(design))[, target_index]
      direct <- c_variance(updated, target)
      closed <- c_variance(information, target) -
        as.numeric(crossprod(target, inverse %*% x))^2 /
          (1 + as.numeric(crossprod(x, inverse %*% x)))
      maximum_update_residual <- max(
        maximum_update_residual, abs(direct - closed)
      )
    }
  }

  # Exact factorial counterexample to supermodularity of c-variance.
  target <- diag(ncol(design))[, 2L]
  cost <- function(selected) {
    extra <- if (length(selected)) {
      crossprod(design[candidates[selected], , drop = FALSE])
    } else matrix(0, ncol(design), ncol(design))
    c_variance(information + extra, target)
  }
  A <- 1L
  B <- c(1L, 2L)
  element <- 3L
  marginal_A <- cost(c(A, element)) - cost(A)
  marginal_B <- cost(c(B, element)) - cost(B)
  violation <- marginal_A - marginal_B
  gate_assert(
    violation > 1e-10,
    "The frozen factorial supermodularity counterexample disappeared."
  )
  list(
    update_residual = maximum_update_residual,
    supermodularity = data.frame(
      J = 3L,
      q = 1L,
      base_profiles = paste(base, collapse = ":"),
      candidate_profiles = paste(candidates, collapse = ":"),
      target_column = 2L,
      A = paste(A, collapse = ":"),
      B = paste(B, collapse = ":"),
      added_element = element,
      marginal_at_A = marginal_A,
      marginal_at_B = marginal_B,
      supermodularity_violation = violation
    )
  )
}

target_metrics <- function(design, target, total_profiles) {
  supported_profiles <- nrow(design)
  decomposition <- svd(
    design / sqrt(supported_profiles),
    nu = 0L,
    nv = min(dim(design))
  )
  scale <- if (length(decomposition$d)) decomposition$d[[1L]] else 1
  rank <- sum(decomposition$d > 1e-10 * max(1, scale))
  basis <- if (rank) {
    decomposition$v[, seq_len(rank), drop = FALSE]
  } else matrix(0, ncol(design), 0L)
  coordinates <- if (rank) {
    as.numeric(crossprod(basis, target))
  } else numeric(0)
  projection <- if (rank) {
    as.numeric(basis %*% coordinates)
  } else rep(0, length(target))
  residual <- sqrt(sum((target - projection)^2))
  if (residual >= 1e-10) {
    return(list(
      estimable = FALSE,
      residual = residual,
      variance_factor = NA_real_,
      information_quadratic = NA_real_,
      alias_squared = NA_real_
    ))
  }
  variance_factor <- sum(
    (coordinates / decomposition$d[seq_len(rank)])^2
  )
  information_quadratic <- variance_factor / supported_profiles
  list(
    estimable = TRUE,
    residual = residual,
    variance_factor = variance_factor,
    information_quadratic = information_quadratic,
    alias_squared = max(
      0,
      total_profiles * information_quadratic - 1
    )
  )
}

walsh_column_index <- function(members) {
  1L + if (length(members)) sum(2L^(members - 1L)) else 0L
}

run_special_case_checks <- function() {
  rows <- list()
  cursor <- 0L
  validation <- list()

  add_row <- function(
      case_id, family, J, q, target, base_profiles,
      candidate_profiles, budget, selected, layers, metrics,
      minimum_budget, estimable_sets_at_minimum, optimal_sets,
      comparison_variance_factor, structural_result) {
    cursor <<- cursor + 1L
    rows[[cursor]] <<- data.frame(
      case_id = case_id,
      family = family,
      J = J,
      q = q,
      target = target,
      base_profiles = base_profiles,
      candidate_profiles = candidate_profiles,
      budget = budget,
      selected_profile_ids_zero_based = paste(selected - 1L, collapse = ":"),
      selected_layers = paste(layers, collapse = ":"),
      estimable = metrics$estimable,
      minimum_budget = minimum_budget,
      estimable_sets_at_minimum = estimable_sets_at_minimum,
      optimal_sets = optimal_sets,
      variance_factor = metrics$variance_factor,
      alias_radius = if (metrics$estimable) {
        sqrt(metrics$alias_squared)
      } else NA_real_,
      information_quadratic = metrics$information_quadratic,
      comparison_variance_factor = comparison_variance_factor,
      structural_result = structural_result
    )
  }

  # A parity half-fraction: an antipodal pair has increasing marginal value.
  geometry <- binary_geometry(4L)
  columns <- which(geometry$order <= 1L)
  design <- geometry$walsh[, columns, drop = FALSE]
  base <- which(apply(geometry$profiles, 1L, prod) == 1L)
  candidates <- setdiff(seq_len(nrow(design)), base)
  first <- candidates[[1L]]
  antipode <- which(rowSums(sweep(
    geometry$profiles,
    2L,
    -geometry$profiles[first, ],
    FUN = "-"
  )^2) == 0)[[1L]]
  gate_assert(
    antipode %in% candidates,
    "The frozen parity antipode is outside the candidate half-fraction."
  )
  target <- diag(length(columns))[, match(1L, columns)]
  parity_base <- target_metrics(design[base, , drop = FALSE], target, 16L)
  parity_single <- target_metrics(
    design[c(base, first), , drop = FALSE], target, 16L
  )
  parity_pair <- target_metrics(
    design[c(base, first, antipode), , drop = FALSE], target, 16L
  )
  marginal_empty <- parity_base$information_quadratic -
    parity_single$information_quadratic
  marginal_after <- parity_single$information_quadratic -
    parity_pair$information_quadratic
  parity_residual <- max(abs(c(
    parity_base$variance_factor - 1,
    parity_base$alias_squared - 1,
    parity_single$variance_factor - 27 / 26,
    parity_single$alias_squared - 11 / 13,
    parity_pair$variance_factor - 1,
    parity_pair$alias_squared - 3 / 5,
    marginal_empty - 1 / 104,
    marginal_after - 1 / 65
  )))
  gate_assert(
    marginal_after > marginal_empty + 1e-10,
    "The parity increasing-marginal counterexample disappeared."
  )
  validation[[length(validation) + 1L]] <- data.frame(
    check = "parity_antipodal_pair_closed_form",
    residual = parity_residual
  )
  add_row(
    "parity_j4_q1_base", "parity", 4L, 1L, "intercept",
    length(base), length(candidates), 0L, integer(0), integer(0),
    parity_base, 0L, 1L, 1L, NA_real_,
    "Base parity half-fraction."
  )
  add_row(
    "parity_j4_q1_single", "parity", 4L, 1L, "intercept",
    length(base), length(candidates), 1L, first,
    geometry$layer[first], parity_single, 0L, 1L, 8L,
    parity_pair$variance_factor,
    "One candidate; its antipode has larger conditional marginal value."
  )
  add_row(
    "parity_j4_q1_antipodal_pair", "parity", 4L, 1L,
    "intercept", length(base), length(candidates), 2L,
    c(first, antipode), geometry$layer[c(first, antipode)],
    parity_pair, 0L, 1L, 4L, parity_single$variance_factor,
    "Antipodal balance restores variance factor one."
  )

  # One profile outside a Hamming slice identifies the additive model.  The
  # variance-optimal profile is not the most distant layer in this example.
  geometry <- binary_geometry(6L)
  columns <- which(geometry$order <= 1L)
  design <- geometry$walsh[, columns, drop = FALSE]
  base <- which(geometry$layer == 1L)
  candidates <- setdiff(seq_len(nrow(design)), base)
  target <- diag(length(columns))[, match(
    walsh_column_index(1L), columns
  )]
  m0 <- length(base)
  lambda <- 4 * m0 * 1L * 5L / (6L * 5L)
  u <- c(1 - 1 / 6, rep(-1 / 6, 5L))
  hamming_metrics <- lapply(candidates, function(candidate) {
    metrics <- target_metrics(
      design[c(base, candidate), , drop = FALSE], target, 64L
    )
    layer <- geometry$layer[[candidate]]
    layer_sum <- 2 * layer - 6
    difference <- 2 * (layer - 1)
    centered <- geometry$profiles[candidate, ] - layer_sum / 6
    formula_quadratic <- 1 / difference^2 +
      1 / (m0 * difference^2) +
      sum((u - centered / difference)^2) / lambda
    list(metrics = metrics, formula = formula_quadratic)
  })
  hamming_formula_residual <- max(vapply(
    hamming_metrics,
    function(result) abs(
      result$metrics$information_quadratic - result$formula
    ),
    numeric(1L)
  ))
  hamming_variance <- vapply(
    hamming_metrics,
    function(result) result$metrics$variance_factor,
    numeric(1L)
  )
  minimum_variance <- min(hamming_variance)
  optimal <- candidates[abs(hamming_variance - minimum_variance) < 1e-10]
  farthest <- candidates[geometry$layer[candidates] == 6L][[1L]]
  farthest_metrics <- hamming_metrics[[match(farthest, candidates)]]$metrics
  hamming_residual <- max(
    hamming_formula_residual,
    abs(minimum_variance - 49 / 32),
    abs(hamming_metrics[[match(optimal[[1L]], candidates)]]$metrics$
      alias_squared - 13),
    abs(farthest_metrics$variance_factor - 77 / 50)
  )
  gate_assert(
    length(optimal) == 5L &&
      all(geometry$layer[optimal] == 5L) &&
      all(geometry$profiles[optimal, 1L] == 1L),
    "The frozen one-point Hamming optimum changed."
  )
  validation[[length(validation) + 1L]] <- data.frame(
    check = "hamming_one_point_main_effect_formula_and_optimum",
    residual = hamming_residual
  )
  optimal_metrics <- hamming_metrics[[match(optimal[[1L]], candidates)]]$metrics
  add_row(
    "hamming_j6_k1_q1_optimal_single", "hamming_slice", 6L, 1L,
    "factor_1_main_effect", length(base), length(candidates), 1L,
    optimal[[1L]], geometry$layer[optimal[[1L]]], optimal_metrics,
    1L, length(candidates), length(optimal),
    farthest_metrics$variance_factor,
    "Five layer-5 profiles with A1=+1 tie for the one-profile optimum."
  )
  add_row(
    "hamming_j6_k1_q1_farthest_single", "hamming_slice", 6L, 1L,
    "factor_1_main_effect", length(base), length(candidates), 1L,
    farthest, geometry$layer[farthest], farthest_metrics,
    1L, length(candidates), 1L, minimum_variance,
    "The unique layer-6 profile is slightly worse than the layer-5 optimum."
  )

  # With A1=A2=+1 excluded, degree-three identification of beta_12 needs an
  # antipodal pair in the two remaining coordinates.
  geometry <- binary_geometry(4L)
  columns <- which(geometry$order <= 3L)
  design <- geometry$walsh[, columns, drop = FALSE]
  base <- which(!(
    geometry$profiles[, 1L] == 1L & geometry$profiles[, 2L] == 1L
  ))
  candidates <- setdiff(seq_len(nrow(design)), base)
  target <- diag(length(columns))[, match(
    walsh_column_index(c(1L, 2L)), columns
  )]
  singleton_estimable <- vapply(candidates, function(candidate) {
    target_metrics(
      design[c(base, candidate), , drop = FALSE], target, 16L
    )$estimable
  }, logical(1L))
  pairs <- combn(candidates, 2L)
  pair_metrics <- lapply(seq_len(ncol(pairs)), function(column) {
    target_metrics(
      design[c(base, pairs[, column]), , drop = FALSE], target, 16L
    )
  })
  pair_estimable <- vapply(
    pair_metrics, function(metrics) metrics$estimable, logical(1L)
  )
  pair_antipodal <- vapply(seq_len(ncol(pairs)), function(column) {
    selected <- pairs[, column]
    all(
      geometry$profiles[selected[[1L]], -(1:2)] ==
        -geometry$profiles[selected[[2L]], -(1:2)]
    )
  }, logical(1L))
  good_pairs <- which(pair_estimable)
  forbidden_q3_residual <- max(vapply(
    pair_metrics[good_pairs],
    function(metrics) max(
      abs(metrics$variance_factor - 7 / 4),
      abs(metrics$alias_squared - 1)
    ),
    numeric(1L)
  ))
  gate_assert(
    !any(singleton_estimable) &&
      identical(pair_estimable, pair_antipodal) &&
      length(good_pairs) == 2L,
    "Degree-three forbidden-pair augmentation is not antipodal."
  )
  validation[[length(validation) + 1L]] <- data.frame(
    check = "forbidden_pair_j4_q3_antipodal_minimum",
    residual = forbidden_q3_residual
  )
  selected_pair <- pairs[, good_pairs[[1L]]]
  selected_pair_metrics <- pair_metrics[[good_pairs[[1L]]]]
  add_row(
    "forbidden_pair_j4_q3_antipodal", "forbidden_pair", 4L, 3L,
    "factor_1_by_factor_2", length(base), length(candidates), 2L,
    selected_pair, geometry$layer[selected_pair], selected_pair_metrics,
    2L, length(good_pairs), length(good_pairs), NA_real_,
    "No singleton works; exactly the two outer-coordinate antipodal pairs work."
  )

  # At degree four with three remaining coordinates, beta_12 requires a
  # four-node strength-two cubature rule, equivalently a 4x4 Hadamard design.
  geometry <- binary_geometry(5L)
  columns <- which(geometry$order <= 4L)
  design <- geometry$walsh[, columns, drop = FALSE]
  base <- which(!(
    geometry$profiles[, 1L] == 1L & geometry$profiles[, 2L] == 1L
  ))
  candidates <- setdiff(seq_len(nrow(design)), base)
  target <- diag(length(columns))[, match(
    walsh_column_index(c(1L, 2L)), columns
  )]
  estimable_counts <- integer(4L)
  four_combinations <- NULL
  four_metrics <- NULL
  for (budget in 1:4) {
    combinations <- combn(candidates, budget)
    metrics <- lapply(seq_len(ncol(combinations)), function(column) {
      target_metrics(
        design[c(base, combinations[, column]), , drop = FALSE],
        target,
        32L
      )
    })
    estimable <- vapply(
      metrics, function(result) result$estimable, logical(1L)
    )
    estimable_counts[[budget]] <- sum(estimable)
    if (budget == 4L) {
      four_combinations <- combinations
      four_metrics <- metrics
    }
  }
  good_fours <- which(vapply(
    four_metrics, function(metrics) metrics$estimable, logical(1L)
  ))
  hadamard_residual <- max(vapply(good_fours, function(column) {
    selected <- four_combinations[, column]
    augmented_outer <- cbind(1, geometry$profiles[selected, 3:5])
    max(abs(crossprod(augmented_outer) - 4 * diag(4L)))
  }, numeric(1L)))
  forbidden_q4_residual <- max(
    hadamard_residual,
    vapply(
      four_metrics[good_fours],
      function(metrics) max(
        abs(metrics$variance_factor - 7 / 4),
        abs(metrics$alias_squared - 1)
      ),
      numeric(1L)
    )
  )
  gate_assert(
    identical(estimable_counts, c(0L, 0L, 0L, 2L)),
    "The four-profile Hadamard minimum changed."
  )
  validation[[length(validation) + 1L]] <- data.frame(
    check = "forbidden_pair_j5_q4_hadamard_minimum",
    residual = forbidden_q4_residual
  )
  selected_four <- four_combinations[, good_fours[[1L]]]
  selected_four_metrics <- four_metrics[[good_fours[[1L]]]]
  add_row(
    "forbidden_pair_j5_q4_hadamard", "forbidden_pair", 5L, 4L,
    "factor_1_by_factor_2", length(base), length(candidates), 4L,
    selected_four, geometry$layer[selected_four], selected_four_metrics,
    4L, length(good_fours), length(good_fours), NA_real_,
    "No set below four works; exactly two four-profile Hadamard sets work."
  )

  list(
    rows = do.call(rbind, rows),
    validation = do.call(rbind, validation)
  )
}

robust_risk <- function(x_low, x_high, support, target, B, sigma = 1) {
  low <- x_low[support, , drop = FALSE]
  high <- x_high[support, , drop = FALSE]
  criterion <- diag(sigma^2, length(support)) +
    B^2 * tcrossprod(high)
  criterion_inverse <- solve(criterion)
  information <- crossprod(low, criterion_inverse %*% low)
  coefficient <- solve(information, target)
  signed_weights <- criterion_inverse %*% low %*% coefficient
  variance <- sigma^2 * sum(signed_weights^2)
  maximum_squared_bias <- B^2 * sum(crossprod(high, signed_weights)^2)
  list(
    risk = variance + maximum_squared_bias,
    variance = variance,
    maximum_squared_bias = maximum_squared_bias
  )
}

combine_risk_output <- function(supermodularity, robust_rows) {
  super <- data.frame(
    record_type = "supermodularity_counterexample",
    B = NA_real_,
    budget = NA_integer_,
    exhaustive_selected = NA_character_,
    greedy_selected = NA_character_,
    exhaustive_risk = NA_real_,
    greedy_risk = NA_real_,
    greedy_risk_ratio = NA_real_,
    exhaustive_variance = NA_real_,
    exhaustive_maximum_squared_bias = NA_real_,
    base_profiles = supermodularity$base_profiles,
    candidate_profiles = supermodularity$candidate_profiles,
    target_column = supermodularity$target_column,
    A = supermodularity$A,
    comparison_B = supermodularity$B,
    added_element = supermodularity$added_element,
    marginal_at_A = supermodularity$marginal_at_A,
    marginal_at_B = supermodularity$marginal_at_B,
    supermodularity_violation = supermodularity$supermodularity_violation
  )
  robust <- data.frame(
    record_type = "robust_greedy_comparison",
    robust_rows,
    base_profiles = NA_character_,
    candidate_profiles = NA_character_,
    target_column = NA_integer_,
    A = NA_character_,
    comparison_B = NA_character_,
    added_element = NA_integer_,
    marginal_at_A = NA_real_,
    marginal_at_B = NA_real_,
    supermodularity_violation = NA_real_
  )
  rbind(super, robust)
}

run_gate <- function() {
  layer <- run_layer_gate()
  parity <- run_parity_gate()
  variance <- run_variance_checks()
  special <- run_special_case_checks()

  # The robust risk rows are constructed separately so the counterexample and
  # performance comparison can share one machine-readable output.
  geometry <- binary_geometry(3L)
  low_columns <- which(geometry$order <= 1L)
  high_columns <- which(geometry$order > 1L)
  x_low <- geometry$walsh[, low_columns, drop = FALSE]
  x_high <- geometry$walsh[, high_columns, drop = FALSE]
  base <- c(1L, 2L, 3L, 5L)
  candidates <- setdiff(seq_len(nrow(x_low)), base)
  target <- diag(ncol(x_low))[, 2L]
  robust_rows <- list()
  cursor <- 0L
  for (B in c(0, 0.25, 1, 4)) {
    risks <- lapply(0:(2^length(candidates) - 1L), function(mask) {
      selected <- mask_members(mask, length(candidates))
      robust_risk(
        x_low, x_high, c(base, candidates[selected]), target, B
      )
    })
    greedy <- integer(0)
    for (budget in seq_along(candidates)) {
      remaining <- setdiff(seq_along(candidates), greedy)
      additions <- vapply(remaining, function(index) {
        mask <- sum(2L^(c(greedy, index) - 1L))
        risks[[mask + 1L]]$risk
      }, numeric(1L))
      greedy <- c(greedy, remaining[[which.min(additions)]])
      masks <- Filter(
        function(mask) mask_size(mask, length(candidates)) == budget,
        0:(2^length(candidates) - 1L)
      )
      exact_values <- vapply(
        masks, function(mask) risks[[mask + 1L]]$risk, numeric(1L)
      )
      exact_mask <- masks[[which.min(exact_values)]]
      greedy_mask <- sum(2L^(greedy - 1L))
      exact <- risks[[exact_mask + 1L]]
      greedy_result <- risks[[greedy_mask + 1L]]
      cursor <- cursor + 1L
      robust_rows[[cursor]] <- data.frame(
        B = B,
        budget = budget,
        exhaustive_selected = paste(
          mask_members(exact_mask, length(candidates)), collapse = ":"
        ),
        greedy_selected = paste(greedy, collapse = ":"),
        exhaustive_risk = exact$risk,
        greedy_risk = greedy_result$risk,
        greedy_risk_ratio = greedy_result$risk / exact$risk,
        exhaustive_variance = exact$variance,
        exhaustive_maximum_squared_bias = exact$maximum_squared_bias
      )
    }
  }
  risk <- combine_risk_output(
    variance$supermodularity, do.call(rbind, robust_rows)
  )

  validation <- rbind(data.frame(
    check = c(
      "hamming_layer_rank_formula",
      "parity_augmented_target_estimability",
      "sherman_morrison_variance_update"
    ),
    residual = c(
      layer$rank_residual,
      parity$residual,
      variance$update_residual
    )
  ), special$validation)
  gate_assert(
    max(validation$residual) < 1e-10,
    sprintf(
      "An augmentation identity failed: maximum residual %.3g.",
      max(validation$residual)
    )
  )
  list(
    validation = validation,
    parity = parity$rows,
    layers = layer$rows,
    risk = risk,
    special = special$rows
  )
}

summarize_gate <- function(results) {
  robust <- results$risk[
    results$risk$record_type == "robust_greedy_comparison", , drop = FALSE
  ]
  finite_layers <- is.finite(results$layers$minimum_added_layers)
  data.frame(
    metric = c(
      "validation_checks",
      "maximum_validation_residual",
      "parity_cases",
      "minimum_parity_added_profiles",
      "maximum_parity_added_profiles",
      "hamming_layer_cases",
      "infeasible_hamming_layer_cases",
      "greedy_layer_matches_exhaustive_cases",
      "robust_greedy_comparisons",
      "maximum_robust_greedy_risk_ratio",
      "supermodularity_violation",
      "special_case_rows",
      "hamming_one_point_optimal_profiles",
      "forbidden_q3_minimum_added_profiles",
      "forbidden_q4_minimum_added_profiles"
    ),
    value = c(
      nrow(results$validation),
      max(results$validation$residual),
      nrow(results$parity),
      min(results$parity$minimum_added_profiles),
      max(results$parity$minimum_added_profiles),
      nrow(results$layers),
      sum(!finite_layers),
      nrow(results$layers),
      nrow(robust),
      max(robust$greedy_risk_ratio),
      results$risk$supermodularity_violation[[1L]],
      nrow(results$special),
      results$special$optimal_sets[
        results$special$case_id == "hamming_j6_k1_q1_optimal_single"
      ],
      results$special$minimum_budget[
        results$special$case_id == "forbidden_pair_j4_q3_antipodal"
      ],
      results$special$minimum_budget[
        results$special$case_id == "forbidden_pair_j5_q4_hadamard"
      ]
    )
  )
}

main <- function() {
  options(stringsAsFactors = FALSE, digits = 15)
  arguments <- parse_arguments(commandArgs(trailingOnly = TRUE))
  directory <- script_directory()
  paths <- list(
    validation = if (is.null(arguments$validation_output)) {
      file.path(directory, "support_augmentation_validation.csv")
    } else arguments$validation_output,
    parity = if (is.null(arguments$parity_output)) {
      file.path(directory, "support_augmentation_parity.csv")
    } else arguments$parity_output,
    layers = if (is.null(arguments$layers_output)) {
      file.path(directory, "support_augmentation_layers.csv")
    } else arguments$layers_output,
    risk = if (is.null(arguments$risk_output)) {
      file.path(directory, "support_augmentation_risk.csv")
    } else arguments$risk_output,
    special = if (is.null(arguments$special_output)) {
      file.path(directory, "support_augmentation_special_cases.csv")
    } else arguments$special_output,
    summary = if (is.null(arguments$summary_output)) {
      file.path(directory, "support_augmentation_gate_summary.csv")
    } else arguments$summary_output
  )

  started <- proc.time()[[3L]]
  results <- run_gate()
  summary <- summarize_gate(results)
  for (name in c("validation", "parity", "layers", "risk", "special")) {
    write.csv(results[[name]], paths[[name]], row.names = FALSE)
  }
  write.csv(summary, paths$summary, row.names = FALSE)

  if (!arguments$quiet) {
    print(results$validation, row.names = FALSE, digits = 6)
    cat("\n")
    print(summary, row.names = FALSE, digits = 8)
    cat(sprintf(
      "\nWrote augmentation outputs to %s.\n",
      normalizePath(directory, mustWork = TRUE)
    ))
    cat(sprintf("Elapsed: %.2f seconds.\n", proc.time()[[3L]] - started))
  }
  invisible(c(results, list(summary = summary)))
}

if (sys.nframe() == 0L) main()
