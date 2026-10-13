#' Target-weighted factorial function basis
#'
#' Construct a coding-invariant factorial basis from nominal factors.  Each
#' factor first receives contrasts that are exactly centered and orthonormal
#' under its target marginal distribution.  Tensor-product candidates are then
#' processed one interaction order at a time: the complete order-`k` block is
#' projected off the intercept and all lower orders and is orthonormalized under
#' the joint target distribution.  Consequently, order subspaces remain
#' well-defined even when the target distribution is not a product measure.
#'
#' @param profiles Data frame containing the profiles at which the returned
#'   basis should initially be evaluated.
#' @param factors Character names or integer positions of factor columns.  By
#'   default, all factor columns in `profiles` are used.
#' @param target Optional data frame giving the target support or full target
#'   profile grid.  Defaults to `profiles`.
#' @param target_weights Optional nonnegative weights, aligned with `target`
#'   when it is supplied and with `profiles` otherwise.
#' @param max_order Largest interaction order to retain.
#' @param tol Relative numerical rank tolerance.
#' @param factor_cols Alias for `factors`.
#' @param target_grid Alias for `target`.
#' @return An object of class `factorial_basis`.  Its `order` member has
#'   one entry per non-intercept design column.
#' @export
factorial_basis <- function(profiles, factors = NULL, target = NULL,
                            target_weights = NULL, max_order = NULL,
                            tol = 1e-10, factor_cols = NULL,
                            target_grid = NULL) {
  if (!is.null(factor_cols)) {
    if (!is.null(factors)) {
      stop("Supply only one of `factors` and `factor_cols`.", call. = FALSE)
    }
    factors <- factor_cols
  }
  if (!is.null(target_grid)) {
    if (!is.null(target)) {
      stop("Supply only one of `target` and `target_grid`.", call. = FALSE)
    }
    target <- target_grid
  }
  if (!is.data.frame(profiles) || !nrow(profiles)) {
    stop("`profiles` must be a nonempty data frame.", call. = FALSE)
  }
  if (!is.numeric(tol) || length(tol) != 1L || !is.finite(tol) || tol <= 0) {
    stop("`tol` must be one positive finite number.", call. = FALSE)
  }

  factors <- .fb_resolve_factors(profiles, factors)
  .fb_check_factor_columns(profiles, factors, "profiles")
  if (is.null(target)) target <- profiles
  if (!is.data.frame(target) || !nrow(target)) {
    stop("`target` must be a nonempty data frame.", call. = FALSE)
  }
  .fb_check_factor_columns(target, factors, "target")

  levels_list <- stats::setNames(lapply(factors, function(nm) {
    unique(c(levels(profiles[[nm]]), levels(target[[nm]])))
  }), factors)
  too_small <- factors[lengths(levels_list) < 2L]
  if (length(too_small)) {
    stop(sprintf(
      "Every factor must declare at least two levels; invalid: %s.",
      paste(too_small, collapse = ", ")
    ), call. = FALSE)
  }

  profiles_std <- .fb_standardize_profiles(
    profiles, factors, levels_list, context = "profiles"
  )
  target_std <- .fb_standardize_profiles(
    target, factors, levels_list, context = "target"
  )
  target_weights <- .fb_weights(target_weights, nrow(target_std))

  max_possible <- length(factors)
  if (is.null(max_order)) max_order <- max_possible
  if (!is.numeric(max_order) || length(max_order) != 1L ||
      !is.finite(max_order) || max_order != as.integer(max_order) ||
      max_order < 1L || max_order > max_possible) {
    stop(sprintf("`max_order` must be an integer from 1 to %d.", max_possible),
         call. = FALSE)
  }
  max_order <- as.integer(max_order)

  marginal_weights <- stats::setNames(lapply(factors, function(nm) {
    lev <- levels_list[[nm]]
    ans <- vapply(lev, function(value) {
      sum(target_weights[as.character(target_std[[nm]]) == value])
    }, numeric(1))
    names(ans) <- lev
    ans
  }), factors)
  unsupported <- lapply(marginal_weights, function(x) names(x)[x <= 0])
  unsupported <- unsupported[lengths(unsupported) > 0L]
  if (length(unsupported)) {
    detail <- paste(vapply(names(unsupported), function(nm) {
      sprintf("%s={%s}", nm, paste(unsupported[[nm]], collapse = ","))
    }, character(1)), collapse = "; ")
    stop(paste0(
      "The target distribution must give every declared factor level positive ",
      "marginal probability: ", detail, "."
    ), call. = FALSE)
  }

  contrasts <- stats::setNames(lapply(factors, function(nm) {
    .fb_weighted_contrasts(marginal_weights[[nm]], nm)
  }), factors)
  target_factor_matrices <- .fb_factor_matrices(
    target_std, factors, levels_list, contrasts
  )
  scope_sets <- lapply(seq_len(max_order), function(k) {
    utils::combn(seq_along(factors), k, simplify = FALSE)
  })

  n_target <- nrow(target_std)
  lower_target <- matrix(1, n_target, 1L,
                         dimnames = list(NULL, "(Intercept)"))
  order_specs <- vector("list", max_order)
  target_pieces <- vector("list", max_order)
  candidate_rows <- list()
  scope_rows <- list()
  column_rows <- list()
  order_vector <- integer()
  scope_vector <- character()
  column_counter <- 0L

  for (k in seq_len(max_order)) {
    raw_info <- .fb_raw_order(
      target_factor_matrices, scope_sets[[k]], factors, k
    )
    raw <- raw_info$matrix
    lower_coef <- crossprod(
      lower_target,
      sweep(raw, 1L, target_weights, `*`)
    )
    residual <- raw - lower_target %*% lower_coef
    weighted_residual <- sweep(residual, 1L, sqrt(target_weights), `*`)
    gram <- crossprod(weighted_residual)
    identity_error <- if (ncol(raw)) {
      max(abs(gram - diag(ncol(raw))))
    } else {
      0
    }
    preserve_scopes <- ncol(raw) > 0L &&
      identity_error <= max(tol, 100 * .Machine$double.eps)

    if (preserve_scopes) {
      raw_to_basis <- diag(ncol(raw))
      retained_rank <- ncol(raw)
    } else {
      decomp <- .fb_weighted_svd(weighted_residual, tol)
      retained_rank <- decomp$rank
      raw_to_basis <- decomp$transform
    }
    order_target <- residual %*% raw_to_basis
    if (retained_rank) {
      signs <- vapply(seq_len(retained_rank), function(j) {
        pivot <- which.max(abs(order_target[, j]))
        if (order_target[pivot, j] < 0) -1 else 1
      }, numeric(1))
      order_target <- sweep(order_target, 2L, signs, `*`)
      raw_to_basis <- sweep(raw_to_basis, 2L, signs, `*`)
    }

    raw_meta <- raw_info$metadata
    raw_meta$order <- k
    candidate_rows[[k]] <- raw_meta
    scope_meta <- raw_info$scope_metadata
    scope_meta$residual_rank <- vapply(seq_len(nrow(scope_meta)), function(j) {
      idx <- scope_meta$candidate_start[j]:scope_meta$candidate_end[j]
      .fb_rank(weighted_residual[, idx, drop = FALSE], tol)
    }, integer(1))
    scope_meta$preserved_in_final_basis <- preserve_scopes
    scope_rows[[k]] <- scope_meta

    if (retained_rank) {
      if (preserve_scopes) {
        final_scope <- raw_meta$scope
        final_names <- raw_meta$candidate
        source_candidate <- raw_meta$candidate
      } else {
        final_scope <- rep(sprintf("order%d:mixed", k), retained_rank)
        final_names <- sprintf("order%d.basis%d", k, seq_len(retained_rank))
        source_candidate <- rep(NA_character_, retained_rank)
      }
      colnames(order_target) <- final_names
      column_rows[[k]] <- data.frame(
        column = final_names,
        order = rep.int(k, retained_rank),
        scope = final_scope,
        source_candidate = source_candidate,
        mixed_within_order = rep.int(!preserve_scopes, retained_rank),
        stringsAsFactors = FALSE
      )
      order_vector <- c(order_vector, rep.int(k, retained_rank))
      scope_vector <- c(scope_vector, final_scope)
      column_counter <- column_counter + retained_rank
      lower_target <- cbind(lower_target, order_target)
    } else {
      column_rows[[k]] <- data.frame(
        column = character(), order = integer(), scope = character(),
        source_candidate = character(), mixed_within_order = logical(),
        stringsAsFactors = FALSE
      )
    }

    order_specs[[k]] <- list(
      order = k,
      scopes = raw_info$scopes,
      raw_column_names = raw_meta$candidate,
      lower_coef = unname(lower_coef),
      raw_to_basis = unname(raw_to_basis),
      rank = retained_rank,
      nominal_df = ncol(raw),
      preserve_scopes = preserve_scopes,
      column_names = if (retained_rank) final_names else character(),
      scope = if (retained_rank) final_scope else character()
    )
    target_pieces[[k]] <- order_target
  }

  column_metadata <- do.call(rbind, column_rows)
  rownames(column_metadata) <- NULL
  candidate_metadata <- do.call(rbind, candidate_rows)
  rownames(candidate_metadata) <- NULL
  scope_metadata <- do.call(rbind, scope_rows)
  rownames(scope_metadata) <- NULL
  target_design <- .fb_bind_pieces(target_pieces, n_target)
  if (ncol(target_design)) colnames(target_design) <- column_metadata$column
  rownames(target_design) <- rownames(target_std)

  nominal_df_by_order <- vapply(order_specs, `[[`, integer(1), "nominal_df")
  df_by_order <- tabulate(order_vector, nbins = max_order)
  names(nominal_df_by_order) <- names(df_by_order) <- seq_len(max_order)

  object <- list(
    call = match.call(),
    factors = factors,
    levels = levels_list,
    max_order = max_order,
    target = target_std,
    target_weights = target_weights,
    marginal_weights = marginal_weights,
    contrasts = contrasts,
    order_specs = order_specs,
    order = order_vector,
    scope = scope_vector,
    df_by_order = df_by_order,
    nominal_df_by_order = nominal_df_by_order,
    column_metadata = column_metadata,
    candidate_metadata = candidate_metadata,
    scope_metadata = scope_metadata,
    target_design = target_design,
    profiles = profiles_std,
    tol = tol
  )
  class(object) <- "factorial_basis"
  object$design <- factorial_design(object, profiles_std)

  weighted_target <- sweep(target_design, 1L, sqrt(target_weights), `*`)
  target_rank <- .fb_rank(weighted_target, tol)
  observed_rank <- .fb_rank(object$design, tol)
  target_gram <- crossprod(weighted_target)
  weighted_mean <- if (ncol(target_design)) {
    as.vector(crossprod(target_weights, target_design))
  } else {
    numeric()
  }
  full_grid_size <- prod(lengths(levels_list))
  positive_target <- target_std[target_weights > 0, factors, drop = FALSE]
  support_size <- nrow(unique(positive_target))
  factor_support <- data.frame(
    factor = factors,
    levels = lengths(levels_list),
    positive_levels = vapply(marginal_weights, function(x) sum(x > 0), integer(1)),
    minimum_marginal_weight = vapply(marginal_weights, min, numeric(1)),
    stringsAsFactors = FALSE
  )
  retained_columns <- length(order_vector)
  nominal_columns <- sum(nominal_df_by_order)
  object$rank <- target_rank
  object$target_gram <- target_gram
  object$diagnostics <- list(
    target_rows = n_target,
    support_size = support_size,
    full_grid_size = full_grid_size,
    full_joint_support = isTRUE(all.equal(support_size, full_grid_size)),
    factor_support = factor_support,
    nominal_columns = nominal_columns,
    retained_columns = retained_columns,
    dropped_df_by_order = nominal_df_by_order - df_by_order,
    target_rank = target_rank,
    observed_rank = observed_rank,
    rank_deficient = target_rank < nominal_columns,
    weighted_mean_error = if (length(weighted_mean)) max(abs(weighted_mean)) else 0,
    weighted_orthonormality_error = if (ncol(target_gram)) {
      max(abs(target_gram - diag(ncol(target_gram))))
    } else {
      0
    },
    mixed_orders = which(!vapply(order_specs, `[[`, logical(1), "preserve_scopes")),
    scope_rank = scope_metadata
  )
  object
}

#' Construct a design matrix from a factorial basis
#'
#' @param basis A `factorial_basis` object.
#' @param newdata Optional data frame.  Defaults to the profiles used to create
#'   the object.
#' @param include_intercept Include an unpenalized intercept column.
#' @return A numeric matrix carrying `order` and `scope` attributes.
#' @export
factorial_design <- function(basis, newdata = NULL, include_intercept = FALSE) {
  if (!inherits(basis, "factorial_basis")) {
    stop("`basis` must be a factorial_basis object.", call. = FALSE)
  }
  if (!is.logical(include_intercept) || length(include_intercept) != 1L ||
      is.na(include_intercept)) {
    stop("`include_intercept` must be TRUE or FALSE.", call. = FALSE)
  }
  if (is.null(newdata)) newdata <- basis$profiles
  standardized <- .fb_standardize_profiles(
    newdata, basis$factors, basis$levels, context = "newdata"
  )
  factor_matrices <- .fb_factor_matrices(
    standardized, basis$factors, basis$levels, basis$contrasts
  )
  n <- nrow(standardized)
  lower <- matrix(1, n, 1L, dimnames = list(NULL, "(Intercept)"))
  pieces <- vector("list", basis$max_order)
  for (k in seq_len(basis$max_order)) {
    spec <- basis$order_specs[[k]]
    raw <- .fb_raw_from_scopes(factor_matrices, spec$scopes)
    residual <- raw - lower %*% spec$lower_coef
    current <- residual %*% spec$raw_to_basis
    if (spec$rank) {
      colnames(current) <- spec$column_names
      lower <- cbind(lower, current)
    }
    pieces[[k]] <- current
  }
  design <- .fb_bind_pieces(pieces, n)
  if (ncol(design)) colnames(design) <- basis$column_metadata$column
  rownames(design) <- rownames(standardized)
  order <- basis$order
  scope <- basis$scope
  if (include_intercept) {
    design <- cbind("(Intercept)" = 1, design)
    order <- c(0L, order)
    scope <- c("(Intercept)", scope)
  }
  attr(design, "order") <- order
  attr(design, "scope") <- scope
  attr(design, "column_metadata") <- if (include_intercept) {
    rbind(data.frame(
      column = "(Intercept)", order = 0L, scope = "(Intercept)",
      source_candidate = NA_character_, mixed_within_order = FALSE,
      stringsAsFactors = FALSE
    ), basis$column_metadata)
  } else {
    basis$column_metadata
  }
  design
}

#' @export
predict.factorial_basis <- function(object, newdata = NULL,
                                    include_intercept = FALSE,
                                    type = "matrix", ...) {
  type <- match.arg(type, "matrix")
  factorial_design(object, newdata, include_intercept = include_intercept)
}

#' @export
transform.factorial_basis <- function(`_data`, newdata = NULL, profiles = NULL,
                                      include_intercept = FALSE, ...) {
  if (!is.null(profiles)) {
    if (!is.null(newdata)) {
      stop("Supply only one of `newdata` and `profiles`.", call. = FALSE)
    }
    newdata <- profiles
  }
  factorial_design(`_data`, newdata, include_intercept = include_intercept)
}

# Register the base transform generic without requiring a hand-edited
# NAMESPACE entry.  Roxygen will also discover the method from @export above.
.onLoad <- function(libname, pkgname) {
  registerS3method(
    "transform", "factorial_basis", transform.factorial_basis,
    envir = asNamespace(pkgname)
  )
}

#' @export
print.factorial_basis <- function(x, ...) {
  cat("Target-weighted factorial basis\n")
  cat("  factors:", paste(x$factors, collapse = ", "), "\n")
  cat("  maximum order:", x$max_order, "\n")
  cat("  df by order:", paste(x$df_by_order, collapse = ", "), "\n")
  cat("  target support:", x$diagnostics$support_size, "of",
      x$diagnostics$full_grid_size, "profiles\n")
  cat("  target rank:", x$rank, "\n")
  if (length(x$diagnostics$mixed_orders)) {
    cat("  joint-target mixed orders:",
        paste(x$diagnostics$mixed_orders, collapse = ", "), "\n")
  }
  invisible(x)
}

.fb_resolve_factors <- function(profiles, factors) {
  if (is.null(factors)) {
    factors <- names(profiles)[vapply(profiles, is.factor, logical(1))]
  } else if (is.numeric(factors)) {
    if (anyNA(factors) || any(factors != as.integer(factors)) ||
        any(factors < 1L | factors > ncol(profiles))) {
      stop("Numeric `factors` must be valid column positions.", call. = FALSE)
    }
    factors <- names(profiles)[as.integer(factors)]
  }
  if (!is.character(factors) || !length(factors) || anyNA(factors) ||
      any(!nzchar(factors)) || anyDuplicated(factors)) {
    stop("`factors` must identify one or more distinct factor columns.",
         call. = FALSE)
  }
  missing <- setdiff(factors, names(profiles))
  if (length(missing)) {
    stop(sprintf("Unknown factor columns: %s.", paste(missing, collapse = ", ")),
         call. = FALSE)
  }
  factors
}

.fb_check_factor_columns <- function(data, factors, context) {
  missing <- setdiff(factors, names(data))
  if (length(missing)) {
    stop(sprintf("`%s` is missing factor columns: %s.", context,
                 paste(missing, collapse = ", ")), call. = FALSE)
  }
  invalid <- factors[!vapply(data[factors], is.factor, logical(1))]
  if (length(invalid)) {
    stop(sprintf("Columns in `%s` must be factors: %s.", context,
                 paste(invalid, collapse = ", ")), call. = FALSE)
  }
  missing_values <- factors[vapply(data[factors], anyNA, logical(1))]
  if (length(missing_values)) {
    stop(sprintf("Factor columns in `%s` cannot contain missing values: %s.",
                 context, paste(missing_values, collapse = ", ")),
         call. = FALSE)
  }
  invisible(TRUE)
}

.fb_standardize_profiles <- function(data, factors, levels_list, context) {
  if (!is.data.frame(data)) {
    stop(sprintf("`%s` must be a data frame.", context), call. = FALSE)
  }
  .fb_check_factor_columns(data, factors, context)
  out <- data[factors]
  for (nm in factors) {
    values <- as.character(out[[nm]])
    bad <- setdiff(unique(values), levels_list[[nm]])
    if (length(bad)) {
      stop(sprintf("Unseen levels in `%s` column `%s`: %s.", context, nm,
                   paste(bad, collapse = ", ")), call. = FALSE)
    }
    out[[nm]] <- factor(values, levels = levels_list[[nm]])
  }
  rownames(out) <- rownames(data)
  out
}

.fb_weights <- function(weights, n) {
  if (is.null(weights)) weights <- rep(1, n)
  if (!is.numeric(weights) || length(weights) != n || anyNA(weights) ||
      any(!is.finite(weights)) || any(weights < 0) || sum(weights) <= 0) {
    stop(paste0(
      "`target_weights` must be a nonnegative finite numeric vector with ",
      "one entry per target row and positive sum."
    ), call. = FALSE)
  }
  as.numeric(weights) / sum(weights)
}

.fb_weighted_contrasts <- function(probability, factor_name) {
  level_names <- names(probability)
  probability <- as.numeric(probability)
  n_levels <- length(probability)
  root_probability <- sqrt(probability)
  centered_projector <- diag(n_levels) - tcrossprod(root_probability)
  eig <- eigen(centered_projector, symmetric = TRUE)
  q <- eig$vectors[, seq_len(n_levels - 1L), drop = FALSE]
  q <- q - root_probability %*% crossprod(root_probability, q)
  q <- qr.Q(qr(q, LAPACK = FALSE), complete = FALSE)
  contrast <- sweep(q, 1L, root_probability, `/`)
  for (j in seq_len(ncol(contrast))) {
    pivot <- which.max(abs(contrast[, j]))
    if (contrast[pivot, j] < 0) contrast[, j] <- -contrast[, j]
  }
  rownames(contrast) <- level_names
  colnames(contrast) <- paste0(factor_name, ".c", seq_len(ncol(contrast)))
  contrast
}

.fb_factor_matrices <- function(data, factors, levels_list, contrasts) {
  stats::setNames(lapply(factors, function(nm) {
    index <- match(as.character(data[[nm]]), levels_list[[nm]])
    out <- contrasts[[nm]][index, , drop = FALSE]
    rownames(out) <- rownames(data)
    out
  }), factors)
}

.fb_row_tensor <- function(matrices) {
  n <- nrow(matrices[[1L]])
  out <- matrix(1, n, 1L)
  out_names <- ""
  for (current in matrices) {
    p <- ncol(out)
    q <- ncol(current)
    out <- out[, rep(seq_len(p), each = q), drop = FALSE] *
      current[, rep(seq_len(q), times = p), drop = FALSE]
    current_names <- colnames(current)
    out_names <- if (identical(out_names, "")) {
      current_names
    } else {
      paste(rep(out_names, each = q), rep(current_names, times = p), sep = ":")
    }
  }
  colnames(out) <- out_names
  out
}

.fb_raw_order <- function(factor_matrices, scope_indices, factors, order) {
  pieces <- vector("list", length(scope_indices))
  metadata <- vector("list", length(scope_indices))
  scope_metadata <- vector("list", length(scope_indices))
  scopes <- vector("list", length(scope_indices))
  cursor <- 0L
  for (j in seq_along(scope_indices)) {
    idx <- scope_indices[[j]]
    scope <- paste(factors[idx], collapse = ":")
    piece <- .fb_row_tensor(factor_matrices[idx])
    width <- ncol(piece)
    start <- cursor + 1L
    end <- cursor + width
    pieces[[j]] <- piece
    metadata[[j]] <- data.frame(
      candidate = colnames(piece),
      scope = rep(scope, width),
      scope_column = seq_len(width),
      stringsAsFactors = FALSE
    )
    scope_metadata[[j]] <- data.frame(
      scope = scope,
      order = order,
      nominal_df = width,
      candidate_start = start,
      candidate_end = end,
      stringsAsFactors = FALSE
    )
    scopes[[j]] <- list(indices = idx, scope = scope, start = start, end = end)
    cursor <- end
  }
  matrix <- do.call(cbind, pieces)
  list(
    matrix = matrix,
    metadata = do.call(rbind, metadata),
    scope_metadata = do.call(rbind, scope_metadata),
    scopes = scopes
  )
}

.fb_raw_from_scopes <- function(factor_matrices, scopes) {
  pieces <- lapply(scopes, function(scope) {
    .fb_row_tensor(factor_matrices[scope$indices])
  })
  do.call(cbind, pieces)
}

.fb_weighted_svd <- function(weighted_matrix, tol) {
  p <- ncol(weighted_matrix)
  if (!p) return(list(rank = 0L, transform = matrix(numeric(), 0L, 0L)))
  decomposition <- svd(weighted_matrix, nu = 0L, nv = min(dim(weighted_matrix)))
  if (!length(decomposition$d) || decomposition$d[1L] == 0) {
    return(list(rank = 0L, transform = matrix(numeric(), p, 0L)))
  }
  keep <- which(decomposition$d > tol * max(1, decomposition$d[1L]))
  if (!length(keep)) {
    return(list(rank = 0L, transform = matrix(numeric(), p, 0L)))
  }
  transform <- sweep(
    decomposition$v[, keep, drop = FALSE], 2L, decomposition$d[keep], `/`
  )
  list(rank = length(keep), transform = transform)
}

.fb_rank <- function(x, tol) {
  if (!nrow(x) || !ncol(x)) return(0L)
  singular_values <- svd(x, nu = 0L, nv = 0L)$d
  if (!length(singular_values)) return(0L)
  as.integer(sum(singular_values > tol * max(1, singular_values[1L])))
}

.fb_bind_pieces <- function(pieces, n) {
  keep <- vapply(pieces, ncol, integer(1)) > 0L
  if (!any(keep)) return(matrix(numeric(), n, 0L))
  do.call(cbind, pieces[keep])
}
