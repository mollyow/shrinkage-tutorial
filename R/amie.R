.profile_key <- function(data, factors) {
  vals <- lapply(data[factors], function(v) {
    z <- as.character(v)
    z[is.na(z)] <- "<NA>"
    paste0(nchar(z), ":", z)
  })
  do.call(paste, c(vals, sep = "\u001f"))
}

.all_subsets <- function(x) {
  unlist(lapply(0:length(x), function(k) {
    if (k == 0L) return(list(character()))
    utils::combn(x, k, simplify = FALSE)
  }), recursive = FALSE)
}

#' Construct prespecified factorial-effect contrasts on the target support
#'
#' The returned matrix contains direct profile weights: if `l` is a column,
#' the estimand is `sum(l * m(target))`. Cell contrasts use a prespecified
#' inclusion-exclusion of conditional means; this coincides with the usual
#' functional-ANOVA interaction under a product target. A classic
#' difference-in-differences contrast is also available for two selected levels
#' of each of two factors.
#'
#' @param basis A `factorial_basis` object.
#' @param factors Factor names defining the interaction.
#' @param type `"cell"` for the interaction effect at every factor-level cell,
#'   or `"did"` for a classic two-by-two difference in differences.
#' @param levels For `type="did"`, a named list giving two levels (baseline,
#'   comparison) for each factor. For `type="cell"`, optional levels to retain.
#' @return Matrix with one row per target profile and one contrast per column.
#' @export
amie_contrast <- function(basis, factors, type = c("cell", "did"), levels = NULL) {
  if (!inherits(basis, "factorial_basis")) stop("`basis` must be a factorial_basis object.", call. = FALSE)
  type <- match.arg(type)
  factors <- as.character(factors)
  if (!length(factors) || anyDuplicated(factors) || !all(factors %in% basis$factors)) {
    stop("`factors` must be distinct factor names in `basis`.", call. = FALSE)
  }
  target <- basis$target
  mu <- .normalize_weights(basis$target_weights, nrow(target))
  conditional_weights <- function(fixed) {
    idx <- rep(TRUE, nrow(target))
    if (length(fixed)) {
      for (nm in names(fixed)) idx <- idx & as.character(target[[nm]]) == as.character(fixed[[nm]])
    }
    prob <- sum(mu[idx])
    if (prob <= 0) stop("A requested target cell has zero probability.", call. = FALSE)
    out <- numeric(nrow(target))
    out[idx] <- mu[idx] / prob
    out
  }

  if (identical(type, "did")) {
    if (length(factors) != 2L) stop("DID requires exactly two factors.", call. = FALSE)
    if (is.null(levels) || !all(factors %in% names(levels)) || any(lengths(levels[factors]) != 2L)) {
      stop("For DID, `levels` must give baseline and comparison levels for both factors.", call. = FALSE)
    }
    f1 <- factors[1L]
    f2 <- factors[2L]
    a <- as.character(levels[[f1]])
    b <- as.character(levels[[f2]])
    if (length(unique(a)) != 2L || length(unique(b)) != 2L ||
        !all(a %in% as.character(basis$levels[[f1]])) ||
        !all(b %in% as.character(basis$levels[[f2]]))) {
      stop("DID levels must be two distinct declared levels for each factor.", call. = FALSE)
    }
    l <- conditional_weights(stats::setNames(list(a[2L], b[2L]), c(f1, f2))) -
      conditional_weights(stats::setNames(list(a[2L], b[1L]), c(f1, f2))) -
      conditional_weights(stats::setNames(list(a[1L], b[2L]), c(f1, f2))) +
      conditional_weights(stats::setNames(list(a[1L], b[1L]), c(f1, f2)))
    out <- matrix(l, ncol = 1L)
    colnames(out) <- paste0(f1, "[", a[2L], "-", a[1L], "] x ",
                            f2, "[", b[2L], "-", b[1L], "] DID")
  } else {
    selected <- lapply(factors, function(nm) {
      available <- as.character(basis$levels[[nm]])
      requested <- if (is.null(levels) || is.null(levels[[nm]])) available else as.character(levels[[nm]])
      if (!all(requested %in% available)) stop(sprintf("Unknown level requested for `%s`.", nm), call. = FALSE)
      requested
    })
    names(selected) <- factors
    cells <- expand.grid(selected, KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE)
    out <- matrix(0, nrow(target), nrow(cells))
    subsets <- .all_subsets(factors)
    for (j in seq_len(nrow(cells))) {
      cell <- as.list(cells[j, , drop = FALSE])
      names(cell) <- factors
      for (subset in subsets) {
        sign <- (-1)^(length(factors) - length(subset))
        fixed <- if (length(subset)) cell[subset] else list()
        out[, j] <- out[, j] + sign * conditional_weights(fixed)
      }
    }
    colnames(out) <- apply(cells, 1L, function(row) {
      paste(paste0(factors, "=", row), collapse = " x ")
    })
  }
  rownames(out) <- .profile_key(target, basis$factors)
  attr(out, "factors") <- factors
  attr(out, "type") <- type
  attr(out, "definition") <- "direct target-profile weights"
  out
}

.assignment_at_rows <- function(assignment_prob, x, basis) {
  n <- nrow(x)
  target <- basis$target
  target_key <- .profile_key(target, basis$factors)
  if (anyDuplicated(target_key)) stop("Basis target contains duplicate profiles.", call. = FALSE)
  observed_key <- .profile_key(x, basis$factors)
  target_index <- match(observed_key, target_key)
  if (anyNA(target_index)) stop("Observed randomized profiles fall outside the target support.", call. = FALSE)
  q <- as.numeric(assignment_prob)
  if (length(q) == nrow(target)) {
    if (any(!is.finite(q)) || any(q <= 0)) {
      stop("Every target profile must have a finite positive assignment probability.", call. = FALSE)
    }
    if (abs(sum(q) - 1) > 1e-8) {
      stop("Target-profile `assignment_prob` must sum to one.", call. = FALSE)
    }
    q_target <- q
    q_obs <- q[target_index]
  } else if (length(q) == n) {
    q_target <- NULL
    q_obs <- q
  } else {
    stop("`assignment_prob` must have one value per target profile or observation.", call. = FALSE)
  }
  if (any(!is.finite(q_obs)) || any(q_obs <= 0)) stop("Assignment probabilities must be finite and positive.", call. = FALSE)
  list(observed = q_obs, target = q_target, target_index = target_index)
}

.with_seed <- function(seed, code) {
  old_seed_exists <- exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  if (old_seed_exists) old_seed <- get(".Random.seed", envir = .GlobalEnv)
  on.exit({
    if (old_seed_exists) assign(".Random.seed", old_seed, envir = .GlobalEnv)
    else if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) rm(".Random.seed", envir = .GlobalEnv)
  }, add = TRUE)
  set.seed(as.integer(seed))
  force(code)
}

.simultaneous_critical <- function(covariance, level, draws, seed) {
  se <- sqrt(pmax(diag(covariance), 0))
  keep <- se > sqrt(.Machine$double.eps)
  if (sum(keep) <= 1L) return(stats::qnorm(1 - (1 - level) / 2))
  corr <- covariance[keep, keep, drop = FALSE] / outer(se[keep], se[keep])
  corr <- (corr + t(corr)) / 2
  eig <- eigen(corr, symmetric = TRUE)
  eig$values[eig$values < 0] <- 0
  root <- eig$vectors %*% diag(sqrt(eig$values), nrow = length(eig$values))
  maxima <- .with_seed(seed, {
    z <- matrix(stats::rnorm(as.integer(draws) * nrow(root)), nrow = nrow(root))
    apply(abs(root %*% z), 2L, max)
  })
  unname(stats::quantile(maxima, probs = level, names = FALSE, type = 8))
}

#' Cross-fitted randomization inference for prespecified AMIEs
#'
#' `contrasts` is a matrix of direct target-profile weights. For a column `l`,
#' the target is `sum(l * m(target))`. Known assignment probabilities supply the
#' Riesz correction. Complete respondents are held out from nuisance fitting,
#' and covariance is computed from respondent-level scores.
#'
#' @param x Data frame of randomized profiles and nuisance regressors.
#' @param y Numeric outcome.
#' @param cluster Independent respondent identifier.
#' @param basis A `factorial_basis` object.
#' @param contrasts Matrix with one row per target profile.
#' @param assignment_prob Known probability for every target profile or row.
#'   If the two lengths coincide, target-profile alignment takes precedence.
#' @param weights Optional respondent survey weights, repeated on task rows.
#' @param outer_folds Number of respondent-level cross-fitting folds.
#' @param fold_id Optional precomputed outer fold labels.
#' @param fit_args Additional named arguments for `fit_order_ridge`. Any
#'   unpenalized nuisance terms must be additive profile-invariant regressors
#'   (for example, task-position fixed effects), so every target contrast
#'   annihilates them. Formula nuisance terms may not reference randomized
#'   factors; the same restriction is the caller's responsibility for a
#'   numeric nuisance matrix.
#' @param level Confidence level.
#' @param simultaneous Use a Gaussian max-|t| critical value across contrasts.
#' @param simultaneous_draws Number of Gaussian draws.
#' @param seed Random seed for folds and simultaneous critical values.
#' @return Object of class `order_amie`.
#' @export
estimate_amie <- function(x, y, cluster, basis, contrasts, assignment_prob,
                          weights = NULL, outer_folds = 5L, fold_id = NULL,
                          fit_args = list(), level = 0.95, simultaneous = TRUE,
                          simultaneous_draws = 20000L, seed = 1L) {
  x <- as.data.frame(x)
  y <- as.numeric(y)
  cluster_chr <- as.character(cluster)
  n <- nrow(x)
  if (length(level) != 1L || !is.finite(level) || level <= 0 || level >= 1) {
    stop("`level` must be one finite number strictly between zero and one.", call. = FALSE)
  }
  if (length(simultaneous_draws) != 1L || !is.finite(simultaneous_draws) ||
      simultaneous_draws != as.integer(simultaneous_draws) || simultaneous_draws < 1L) {
    stop("`simultaneous_draws` must be a positive integer.", call. = FALSE)
  }
  if (length(y) != n || length(cluster_chr) != n || anyNA(cluster_chr) || any(!is.finite(y))) {
    stop("`x`, finite `y`, and non-missing `cluster` must have equal length.", call. = FALSE)
  }
  contrasts <- as.matrix(contrasts)
  if (nrow(contrasts) != nrow(basis$target) || !ncol(contrasts) || any(!is.finite(contrasts))) {
    stop("`contrasts` must be finite with one row per target profile.", call. = FALSE)
  }
  if (is.null(colnames(contrasts))) colnames(contrasts) <- paste0("estimand_", seq_len(ncol(contrasts)))
  # AMIEs annihilate constants. Requiring this prevents an undefined nuisance
  # convention (such as task position) from entering L(f).
  if (any(abs(colSums(contrasts)) > 1e-8)) {
    stop("Each AMIE contrast must sum to zero over target-profile function values.", call. = FALSE)
  }
  if (inherits(fit_args$unpenalized, "formula")) {
    randomized_nuisance <- intersect(all.vars(fit_args$unpenalized), basis$factors)
    if (length(randomized_nuisance)) {
      stop(sprintf(
        paste0("Unpenalized AMIE nuisance terms must be profile-invariant and ",
               "may not reference randomized factors: %s."),
        paste(randomized_nuisance, collapse = ", ")
      ), call. = FALSE)
    }
  }
  ap <- .assignment_at_rows(assignment_prob, x, basis)
  l_obs <- contrasts[ap$target_index, , drop = FALSE]
  riesz_obs <- l_obs / ap$observed
  w_row <- if (is.null(weights)) rep(1, n) else as.numeric(weights)
  if (length(w_row) != n || any(!is.finite(w_row)) || any(w_row < 0) || !any(w_row > 0)) {
    stop("`weights` must be nonnegative, finite, and conformable.", call. = FALSE)
  }
  ids <- unique(cluster_chr)
  id_index <- match(cluster_chr, ids)
  cluster_weight <- vapply(seq_along(ids), function(i) mean(w_row[id_index == i]), numeric(1))
  within_weight_range <- vapply(seq_along(ids), function(i) diff(range(w_row[id_index == i])), numeric(1))
  if (any(within_weight_range > 1e-10)) {
    warning("Survey weights vary within respondent; respondent weights use the within-respondent mean.", call. = FALSE)
  }
  if (sum(cluster_weight > 0) < 2L) {
    stop("At least two positive-weight respondent clusters are required for covariance estimation.", call. = FALSE)
  }
  if (is.null(fold_id)) {
    fold_id <- cluster_folds(cluster_chr, nfolds = outer_folds, seed = seed)
  }
  if (length(fold_id) != n || anyNA(fold_id) || length(unique(fold_id)) < 2L) {
    stop("`fold_id` must supply at least two non-missing outer folds.", call. = FALSE)
  }
  leakage <- vapply(split(fold_id, cluster_chr), function(v) length(unique(v)), integer(1))
  if (any(leakage != 1L)) stop("Every respondent must be isolated in exactly one outer fold.", call. = FALSE)

  scores <- matrix(NA_real_, length(ids), ncol(contrasts), dimnames = list(ids, colnames(contrasts)))
  fits <- vector("list", length(unique(fold_id)))
  names(fits) <- as.character(sort(unique(fold_id)))
  z_target <- factorial_design(basis, basis$target, include_intercept = FALSE)
  fold_levels <- sort(unique(fold_id))
  numeric_unpenalized <- !is.null(fit_args$unpenalized) &&
    !inherits(fit_args$unpenalized, "formula")
  unpenalized_all <- NULL
  if (numeric_unpenalized) {
    unpenalized_all <- as.matrix(fit_args$unpenalized)
    if (nrow(unpenalized_all) != n || any(!is.finite(unpenalized_all))) {
      stop("Numeric `fit_args$unpenalized` must have one finite row per observation.", call. = FALSE)
    }
  }
  for (fold_position in seq_along(fold_levels)) {
    ff <- fold_levels[fold_position]
    train <- fold_id != ff
    valid <- !train
    fold_fit_args <- fit_args
    if (numeric_unpenalized) {
      fold_fit_args$unpenalized <- unpenalized_all[train, , drop = FALSE]
    }
    args <- c(list(
      x = x[train, , drop = FALSE], y = y[train], basis = basis,
      weights = w_row[train], cluster = cluster_chr[train], seed = seed + fold_position
    ), fold_fit_args)
    if (anyDuplicated(names(args)[nzchar(names(args))])) {
      dup <- unique(names(args)[duplicated(names(args)) & nzchar(names(args))])
      stop(sprintf("`fit_args` duplicates reserved arguments: %s", paste(dup, collapse = ", ")), call. = FALSE)
    }
    fit <- do.call(fit_order_ridge, args)
    fits[[as.character(ff)]] <- fit
    factorial_target_prediction <- as.vector(z_target %*% fit$coefficients)
    plug_in <- as.vector(crossprod(contrasts, factorial_target_prediction))
    prediction <- predict(
      fit, x[valid, , drop = FALSE],
      new_unpenalized = if (numeric_unpenalized) {
        unpenalized_all[valid, , drop = FALSE]
      } else NULL
    )
    correction_rows <- riesz_obs[valid, , drop = FALSE] * (y[valid] - prediction)
    valid_ids <- unique(cluster_chr[valid])
    for (id in valid_ids) {
      idx_local <- which(cluster_chr[valid] == id)
      scores[match(id, ids), ] <- plug_in + colMeans(correction_rows[idx_local, , drop = FALSE])
    }
  }
  if (anyNA(scores)) stop("Cross-fitting failed to produce a score for every respondent.", call. = FALSE)
  keep_cluster <- cluster_weight > 0
  scores_use <- scores[keep_cluster, , drop = FALSE]
  cw <- cluster_weight[keep_cluster]
  estimate <- as.vector(crossprod(cw / sum(cw), scores_use))
  centered <- sweep(scores_use, 2L, estimate, "-")
  g <- nrow(scores_use)
  covariance <- (g / (g - 1)) * crossprod(centered * cw) / sum(cw)^2
  dimnames(covariance) <- list(colnames(contrasts), colnames(contrasts))
  se <- sqrt(pmax(diag(covariance), 0))
  critical <- if (isTRUE(simultaneous)) {
    .simultaneous_critical(covariance, level, simultaneous_draws, seed + 100000L)
  } else stats::qnorm(1 - (1 - level) / 2)
  table <- data.frame(
    estimand = colnames(contrasts), estimate = estimate, std_error = se,
    conf_low = estimate - critical * se, conf_high = estimate + critical * se,
    level = level, simultaneous = isTRUE(simultaneous), critical_value = critical,
    row.names = NULL
  )
  object <- list(
    estimates = table, covariance = covariance, scores = scores,
    cluster_weights = stats::setNames(cluster_weight, ids), outer_fold_id = fold_id,
    contrasts = contrasts, fits = fits, assignment_prob = assignment_prob,
    call = match.call()
  )
  class(object) <- "order_amie"
  object
}

#' @export
print.order_amie <- function(x, ...) {
  cat("Cross-fitted factorial estimands\n")
  print(x$estimates, row.names = FALSE)
  invisible(x)
}
