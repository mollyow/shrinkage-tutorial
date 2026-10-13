#!/usr/bin/env Rscript

# Deterministic restricted-support alias-geometry gate.
#
# This script enumerates binary factorial profiles and checks row-space,
# rank, variance, and leakage claims for the support families frozen in
# docs/restricted-support-gate-protocol.md.  It uses base R only.

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
  normalizePath("analysis/restricted_support", mustWork = TRUE)
}

parse_arguments <- function(arguments) {
  out <- list(
    rank_output = NULL,
    estimability_output = NULL,
    validation_output = NULL,
    summary_output = NULL,
    quiet = FALSE
  )
  for (argument in arguments) {
    if (grepl("^--rank-output=", argument)) {
      out$rank_output <- sub("^--rank-output=", "", argument)
    } else if (grepl("^--estimability-output=", argument)) {
      out$estimability_output <- sub(
        "^--estimability-output=", "", argument
      )
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

choose_sum <- function(n, maximum_order) {
  if (maximum_order < 0L) return(0)
  sum(choose(n, 0:min(n, maximum_order)))
}

support_specifications <- function(geometry, J, q) {
  layer <- geometry$layer
  profiles <- geometry$profiles
  central_k <- floor(J / 2)
  offcentral_k <- max(1L, floor(J / 3))
  union_two_k <- sort(unique(c(
    max(0L, floor(J / 3)),
    min(J, ceiling(2 * J / 3))
  )))
  layer_priority <- order(
    -pmin(0:J, J - (0:J)),
    abs((0:J) - J / 2),
    0:J
  )
  union_qplus_k <- sort((0:J)[layer_priority][
    seq_len(min(q + 1L, J + 1L))
  ])

  specifications <- list(
    list(
      name = "parity_plus",
      family = "parity",
      detail = "product=+1",
      index = which(apply(profiles, 1L, prod) == 1)
    ),
    list(
      name = "slice_central",
      family = "slice",
      detail = sprintf("k=%d", central_k),
      index = which(layer == central_k),
      layers = central_k
    ),
    list(
      name = "slice_offcentral",
      family = "slice",
      detail = sprintf("k=%d", offcentral_k),
      index = which(layer == offcentral_k),
      layers = offcentral_k
    ),
    list(
      name = "union_two",
      family = "slice_union",
      detail = sprintf("k=%s", paste(union_two_k, collapse = ":")),
      index = which(layer %in% union_two_k),
      layers = union_two_k
    ),
    list(
      name = "union_q_plus_1",
      family = "slice_union",
      detail = sprintf("k=%s", paste(union_qplus_k, collapse = ":")),
      index = which(layer %in% union_qplus_k),
      layers = union_qplus_k
    ),
    list(
      name = "forbidden_pair",
      family = "forbidden_pair",
      detail = "exclude A1=A2=+1",
      index = which(!(profiles[, 1L] == 1L & profiles[, 2L] == 1L))
    )
  )

  sizes <- sort(unique(vapply(
    specifications, function(specification) length(specification$index),
    integer(1L)
  )))
  for (size in sizes) {
    for (replicate in 1:3) {
      set.seed(100000L * J + 1000L * q + 10L * size + replicate)
      specifications[[length(specifications) + 1L]] <- list(
        name = sprintf("generic_n%d_r%d", size, replicate),
        family = "generic",
        detail = sprintf("size=%d;replicate=%d", size, replicate),
        index = sort(sample(seq_len(nrow(profiles)), size, replace = FALSE))
      )
    }
  }
  specifications
}

weighted_svd_analysis <- function(design) {
  supported_profiles <- nrow(design)
  weights <- rep(1 / supported_profiles, supported_profiles)
  weighted_design <- design * sqrt(weights)
  decomposition <- svd(
    weighted_design,
    nu = 0L,
    nv = min(dim(weighted_design))
  )
  scale <- if (length(decomposition$d)) max(decomposition$d) else 1
  rank <- sum(decomposition$d > 1e-10 * max(1, scale))
  if (rank == 0L) {
    range_basis <- matrix(0, ncol(design), 0L)
    inverse_information_diagonal <- rep(0, ncol(design))
  } else {
    range_basis <- decomposition$v[, seq_len(rank), drop = FALSE]
    inverse_information_diagonal <- rowSums(
      sweep(
        range_basis, 2L, decomposition$d[seq_len(rank)], FUN = "/"
      )^2
    )
  }
  residual_squared <- pmax(0, 1 - rowSums(range_basis^2))
  estimable <- residual_squared < 1e-10
  list(
    rank = rank,
    residual_squared = residual_squared,
    estimable = estimable,
    inverse_information_diagonal = inverse_information_diagonal,
    range_basis = range_basis,
    singular_values = decomposition$d,
    weights = weights
  )
}

rank_only <- function(design) {
  qr(design, tol = 1e-10, LAPACK = FALSE)$rank
}

parity_predicted_rank <- function(J, q) {
  model_dimension <- choose_sum(J, q)
  lower <- max(0L, J - q)
  if (lower > q) return(model_dimension)
  paired_columns <- sum(choose(J, lower:q)) / 2
  model_dimension - paired_columns
}

slice_predicted_rank <- function(J, q, k) {
  choose(J, min(q, k, J - k))
}

slice_union_predicted_rank <- function(J, q, layers) {
  maximum_harmonic_order <- min(q, floor(J / 2))
  total <- 0
  for (degree in 0:maximum_harmonic_order) {
    harmonic_dimension <- choose(J, degree) - if (degree == 0L) {
      0
    } else choose(J, degree - 1L)
    eligible_layers <- sum(layers >= degree & layers <= J - degree)
    total <- total + harmonic_dimension * min(
      q - degree + 1L,
      eligible_layers
    )
  }
  total
}

forbidden_pair_predicted_rank <- function(J, q) {
  choose_sum(J, q) - choose_sum(J - 2L, q - 2L)
}

predicted_rank <- function(specification, J, q, model_dimension) {
  if (identical(specification$family, "parity")) {
    return(parity_predicted_rank(J, q))
  }
  if (specification$family %in% c("slice", "slice_union")) {
    return(slice_union_predicted_rank(
      J, q, specification$layers
    ))
  }
  if (identical(specification$family, "forbidden_pair")) {
    return(forbidden_pair_predicted_rank(J, q))
  }
  NA_real_
}

predicted_estimability <- function(
    specification, subset_bits, subset_order, J, q) {
  model <- subset_order <= q
  bits <- subset_bits[model, , drop = FALSE]
  order <- subset_order[model]
  if (identical(specification$family, "parity")) {
    return((J - order) > q)
  }
  if (identical(specification$family, "forbidden_pair")) {
    outside_pair <- if (J == 2L) rep(0L, nrow(bits)) else
      rowSums(bits[, -(1:2), drop = FALSE])
    return(outside_pair > q - 2L)
  }
  if (identical(specification$family, "slice")) {
    k <- specification$layers[[1L]]
    if (q == 0L) return(rep(TRUE, length(order)))
    if (q == 1L && 2L * k == J) return(order == 0L)
    return(rep(FALSE, length(order)))
  }
  if (identical(specification$name, "union_q_plus_1")) {
    return(rep(TRUE, length(order)))
  }
  rep(NA, length(order))
}

subset_index <- function(bits) {
  1L + sum(bits * 2L^(seq_along(bits) - 1L))
}

parity_leakage_residual <- function(
    geometry, model_columns, design, analysis, q, J) {
  model_bits <- geometry$bits[model_columns, , drop = FALSE]
  model_order <- geometry$order[model_columns]
  candidate <- which(model_order <= min(2L, q) & (J - model_order) > q)
  if (!length(candidate)) return(NA_real_)
  target_local <- candidate[[1L]]
  target_global <- model_columns[[target_local]]
  c_vector <- numeric(length(model_columns))
  c_vector[[target_local]] <- 1
  information_inverse_c <- if (analysis$rank == 0L) c_vector * 0 else {
    basis <- analysis$range_basis
    singular <- analysis$singular_values[seq_len(analysis$rank)]
    as.numeric(basis %*% ((crossprod(basis, c_vector)) / singular^2))
  }
  representer <- as.numeric(design %*% information_inverse_c)
  support <- apply(geometry$profiles, 1L, prod) == 1
  load <- as.numeric(crossprod(
    geometry$walsh[support, , drop = FALSE],
    analysis$weights * representer
  ))
  expected <- numeric(length(load))
  expected[[target_global]] <- 1
  complement_bits <- 1L - geometry$bits[target_global, ]
  expected[[subset_index(complement_bits)]] <- 1
  max(abs(load - expected))
}

slice_relation_residual <- function(
    geometry, model_columns, support_index, q, k, J) {
  if (q == 0L) return(0)
  model_bits <- geometry$bits[model_columns, , drop = FALSE]
  lookup <- rep(NA_integer_, nrow(geometry$bits))
  lookup[model_columns] <- seq_along(model_columns)
  candidate_T <- which(geometry$order <= q - 1L)
  maximum <- 0
  constant <- 2L * k - J
  for (global_T in candidate_T) {
    T <- geometry$bits[global_T, ]
    relation <- numeric(length(model_columns))
    relation[[lookup[[global_T]]]] <-
      relation[[lookup[[global_T]]]] - constant
    for (j in seq_len(J)) {
      neighbor <- T
      neighbor[[j]] <- 1L - neighbor[[j]]
      neighbor_global <- subset_index(neighbor)
      neighbor_local <- lookup[[neighbor_global]]
      if (!is.na(neighbor_local)) {
        relation[[neighbor_local]] <- relation[[neighbor_local]] + 1
      }
    }
    residual <- max(abs(
      geometry$walsh[support_index, model_columns, drop = FALSE] %*%
        relation
    ))
    maximum <- max(maximum, residual)
  }
  maximum
}

forbidden_relation_residual <- function(
    geometry, model_columns, support_index, q, J) {
  if (q < 2L) return(0)
  lookup <- rep(NA_integer_, nrow(geometry$bits))
  lookup[model_columns] <- seq_along(model_columns)
  other_bits <- if (J == 2L) matrix(0L, 1L, 0L) else
    geometry$bits[geometry$order <= q - 2L, -(1:2), drop = FALSE]
  if (J > 2L) {
    other_bits <- unique(other_bits)
    other_bits <- other_bits[rowSums(other_bits) <= q - 2L, , drop = FALSE]
  }
  maximum <- 0
  for (row in seq_len(nrow(other_bits))) {
    base <- integer(J)
    if (J > 2L) base[-(1:2)] <- other_bits[row, ]
    relation <- numeric(length(model_columns))
    for (first in 0:1) {
      for (second in 0:1) {
        term <- base
        term[[1L]] <- first
        term[[2L]] <- second
        relation[[lookup[[subset_index(term)]]]] <-
          relation[[lookup[[subset_index(term)]]]] + 1
      }
    }
    residual <- max(abs(
      geometry$walsh[support_index, model_columns, drop = FALSE] %*%
        relation
    ))
    maximum <- max(maximum, residual)
  }
  maximum
}

alias_radius_residual <- function(
    geometry, model_columns, support_index, design, analysis, q) {
  candidates <- which(analysis$estimable)
  if (!length(candidates)) return(NA_real_)
  model_order <- geometry$order[model_columns]
  preferred <- candidates[model_order[candidates] <= 2L]
  target_local <- if (length(preferred)) preferred[[1L]] else candidates[[1L]]
  target <- numeric(length(model_columns))
  target[[target_local]] <- 1
  basis <- analysis$range_basis
  singular <- analysis$singular_values[seq_len(analysis$rank)]
  information_inverse_target <- as.numeric(
    basis %*% (crossprod(basis, target) / singular^2)
  )
  representer <- as.numeric(design %*% information_inverse_target)
  signed_weights <- analysis$weights * representer
  load <- as.numeric(crossprod(
    geometry$walsh[support_index, , drop = FALSE],
    signed_weights
  ))
  direct_squared <- sum(load[geometry$order > q]^2)
  variance_factor <- analysis$inverse_information_diagonal[[target_local]]
  parseval_squared <-
    nrow(geometry$profiles) * variance_factor / length(support_index) - 1
  abs(direct_squared - parseval_squared)
}

run_gate <- function() {
  rank_rows <- list()
  estimability_rows <- list()
  validation_rows <- list()
  rank_cursor <- 0L
  estimability_cursor <- 0L
  validation_cursor <- 0L

  for (J in 3:10) {
    geometry <- binary_geometry(J)
    for (q in 0:min(4L, J)) {
      model_columns <- which(geometry$order <= q)
      model_dimension <- length(model_columns)
      specifications <- support_specifications(geometry, J, q)
      for (specification in specifications) {
        support <- specification$index
        design <- geometry$walsh[
          support, model_columns, drop = FALSE
        ]
        is_generic <- identical(specification$family, "generic")
        if (is_generic) {
          observed_rank <- rank_only(design)
          analysis <- NULL
        } else {
          analysis <- weighted_svd_analysis(design)
          observed_rank <- analysis$rank
        }
        expected_rank <- predicted_rank(
          specification, J, q, model_dimension
        )
        rank_cursor <- rank_cursor + 1L
        rank_rows[[rank_cursor]] <- data.frame(
          J = J,
          q = q,
          support = specification$name,
          family = specification$family,
          detail = specification$detail,
          supported_profiles = length(support),
          model_dimension = model_dimension,
          rank = observed_rank,
          nullity = model_dimension - observed_rank,
          predicted_rank = expected_rank,
          rank_residual = if (is.na(expected_rank)) NA_real_ else
            abs(observed_rank - expected_rank)
        )

        if (!is_generic) {
          predicted <- predicted_estimability(
            specification,
            geometry$bits,
            geometry$order,
            J,
            q
          )
          model_order <- geometry$order[model_columns]
          for (order in 0:q) {
            selected <- model_order == order
            actual <- analysis$estimable[selected]
            expected <- predicted[selected]
            variance <- analysis$inverse_information_diagonal[selected &
              analysis$estimable]
            alias_radius <- if (length(variance)) {
              sqrt(pmax(
                0,
                nrow(geometry$profiles) * variance / length(support) - 1
              ))
            } else numeric(0)
            estimability_cursor <- estimability_cursor + 1L
            estimability_rows[[estimability_cursor]] <- data.frame(
              J = J,
              q = q,
              support = specification$name,
              family = specification$family,
              detail = specification$detail,
              coefficient_order = order,
              coefficient_count = sum(selected),
              estimable_count = sum(actual),
              predicted_estimable_count = if (all(is.na(expected))) {
                NA_real_
              } else sum(expected),
              maximum_rowspace_residual_squared = max(
                analysis$residual_squared[selected]
              ),
              minimum_variance_factor = if (length(variance)) {
                min(variance)
              } else NA_real_,
              maximum_variance_factor = if (length(variance)) {
                max(variance)
              } else NA_real_,
              minimum_unweighted_alias_radius = if (length(alias_radius)) {
                min(alias_radius)
              } else NA_real_,
              maximum_unweighted_alias_radius = if (length(alias_radius)) {
                max(alias_radius)
              } else NA_real_
            )
          }

          if (!all(is.na(predicted))) {
            validation_cursor <- validation_cursor + 1L
            validation_rows[[validation_cursor]] <- data.frame(
              J = J,
              q = q,
              support = specification$name,
              check = "individual_estimability_count",
              residual = max(abs(
                as.integer(analysis$estimable) - as.integer(predicted)
              ))
            )
          }
          if (!is.na(expected_rank)) {
            validation_cursor <- validation_cursor + 1L
            validation_rows[[validation_cursor]] <- data.frame(
              J = J,
              q = q,
              support = specification$name,
              check = "rank_formula",
              residual = abs(observed_rank - expected_rank)
            )
          }
          relation_residual <- NA_real_
          relation_check <- NULL
          if (identical(specification$family, "parity")) {
            relation_residual <- parity_leakage_residual(
              geometry, model_columns, design, analysis, q, J
            )
            relation_check <- "parity_alias_leakage"
          } else if (identical(specification$family, "slice")) {
            relation_residual <- slice_relation_residual(
              geometry,
              model_columns,
              support,
              q,
              specification$layers[[1L]],
              J
            )
            relation_check <- "slice_vanishing_relation"
          } else if (identical(specification$family, "forbidden_pair")) {
            relation_residual <- forbidden_relation_residual(
              geometry, model_columns, support, q, J
            )
            relation_check <- "forbidden_pair_relation"
          }
          if (!is.null(relation_check) && !is.na(relation_residual)) {
            validation_cursor <- validation_cursor + 1L
            validation_rows[[validation_cursor]] <- data.frame(
              J = J,
              q = q,
              support = specification$name,
              check = relation_check,
              residual = relation_residual
            )
          }
          radius_residual <- alias_radius_residual(
            geometry,
            model_columns,
            support,
            design,
            analysis,
            q
          )
          if (!is.na(radius_residual)) {
            validation_cursor <- validation_cursor + 1L
            validation_rows[[validation_cursor]] <- data.frame(
              J = J,
              q = q,
              support = specification$name,
              check = "parseval_alias_radius",
              residual = radius_residual
            )
          }
        }
      }
    }
  }

  rank_output <- do.call(rbind, rank_rows)
  estimability_output <- do.call(rbind, estimability_rows)
  validation_output <- do.call(rbind, validation_rows)
  worst <- validation_output[
    validation_output$residual == max(validation_output$residual), ,
    drop = FALSE
  ]
  gate_assert(
    max(validation_output$residual) < 1e-10,
    sprintf(
      paste0(
        "A restricted-support identity failed: maximum residual %.3g. ",
        "First worst case: J=%d, q=%d, support=%s, check=%s."
      ),
      max(validation_output$residual),
      worst$J[[1L]],
      worst$q[[1L]],
      worst$support[[1L]],
      worst$check[[1L]]
    )
  )
  list(
    rank = rank_output,
    estimability = estimability_output,
    validation = validation_output
  )
}

summarize_gate <- function(results) {
  canonical <- results$rank[results$rank$family != "generic", ]
  generic <- results$rank[results$rank$family == "generic", ]
  closed_form <- canonical[!is.na(canonical$predicted_rank), ]
  generic_maximal <- generic$rank == pmin(
    generic$model_dimension,
    generic$supported_profiles
  )
  qplus <- canonical[canonical$support == "union_q_plus_1", ]
  estimability <- merge(
    results$estimability,
    canonical[, c("J", "q", "support", "supported_profiles")],
    by = c("J", "q", "support")
  )
  proper <- estimability$supported_profiles < 2^estimability$J
  radii <- c(
    estimability$minimum_unweighted_alias_radius[proper],
    estimability$maximum_unweighted_alias_radius[proper]
  )
  radii <- radii[is.finite(radii)]

  data.frame(
    metric = c(
      "rank_rows",
      "estimability_rows",
      "validation_rows",
      "maximum_validation_residual",
      "closed_form_rank_checks",
      "maximum_closed_form_rank_residual",
      "canonical_full_rank_cases",
      "canonical_cases",
      "generic_maximal_rank_cases",
      "generic_cases",
      "central_q_plus_1_layer_full_rank_cases",
      "central_q_plus_1_layer_cases",
      "minimum_positive_proper_support_alias_radius",
      "maximum_proper_support_alias_radius"
    ),
    value = c(
      nrow(results$rank),
      nrow(results$estimability),
      nrow(results$validation),
      max(results$validation$residual),
      nrow(closed_form),
      max(closed_form$rank_residual),
      sum(canonical$rank == canonical$model_dimension),
      nrow(canonical),
      sum(generic_maximal),
      nrow(generic),
      sum(qplus$rank == qplus$model_dimension),
      nrow(qplus),
      min(radii),
      max(radii)
    )
  )
}

main <- function() {
  options(stringsAsFactors = FALSE, digits = 15)
  arguments <- parse_arguments(commandArgs(trailingOnly = TRUE))
  directory <- script_directory()
  rank_path <- if (is.null(arguments$rank_output)) {
    file.path(directory, "restricted_support_rank_grid.csv")
  } else arguments$rank_output
  estimability_path <- if (is.null(arguments$estimability_output)) {
    file.path(directory, "restricted_support_estimability.csv")
  } else arguments$estimability_output
  validation_path <- if (is.null(arguments$validation_output)) {
    file.path(directory, "restricted_support_validation.csv")
  } else arguments$validation_output
  summary_path <- if (is.null(arguments$summary_output)) {
    file.path(directory, "restricted_support_gate_summary.csv")
  } else arguments$summary_output

  started <- proc.time()[[3L]]
  results <- run_gate()
  summary <- summarize_gate(results)
  write.csv(results$rank, rank_path, row.names = FALSE)
  write.csv(results$estimability, estimability_path, row.names = FALSE)
  write.csv(results$validation, validation_path, row.names = FALSE)
  write.csv(summary, summary_path, row.names = FALSE)

  if (!arguments$quiet) {
    maxima <- aggregate(residual ~ check, results$validation, max)
    print(maxima, row.names = FALSE, digits = 6)
    cat("\n")
    print(summary, row.names = FALSE, digits = 8)
    cat(sprintf(
      "\nWrote %d rank rows to %s.\n",
      nrow(results$rank), normalizePath(rank_path, mustWork = TRUE)
    ))
    cat(sprintf(
      "Wrote %d estimability rows to %s.\n",
      nrow(results$estimability),
      normalizePath(estimability_path, mustWork = TRUE)
    ))
    cat(sprintf(
      "Wrote %d validation rows to %s.\n",
      nrow(results$validation),
      normalizePath(validation_path, mustWork = TRUE)
    ))
    cat(sprintf(
      "Wrote %d gate metrics to %s.\n",
      nrow(summary), normalizePath(summary_path, mustWork = TRUE)
    ))
    cat(sprintf("Elapsed: %.2f seconds.\n", proc.time()[[3L]] - started))
  }
  invisible(c(results, list(summary = summary)))
}

if (sys.nframe() == 0L) main()
