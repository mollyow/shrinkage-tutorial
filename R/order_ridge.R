.normalize_weights <- function(weights, n) {
  if (is.null(weights)) weights <- rep(1, n)
  weights <- as.numeric(weights)
  if (length(weights) != n || any(!is.finite(weights)) || any(weights < 0) || !any(weights > 0)) {
    stop("`weights` must be a nonnegative finite vector with positive sum.", call. = FALSE)
  }
  weights / sum(weights)
}

.training_unpenalized <- function(x, unpenalized) {
  n <- nrow(x)
  if (is.null(unpenalized)) {
    u <- matrix(1, n, 1L, dimnames = list(NULL, "(Intercept)"))
    return(list(matrix = u, spec = list(type = "intercept", columns = colnames(u))))
  }
  if (inherits(unpenalized, "formula")) {
    mf <- stats::model.frame(unpenalized, data = x, na.action = stats::na.fail)
    trm <- stats::terms(mf)
    u <- stats::model.matrix(trm, mf)
    return(list(
      matrix = unname(u),
      spec = list(type = "formula", terms = trm, xlevels = stats::.getXlevels(trm, mf),
                  contrasts = attr(u, "contrasts"), columns = colnames(u))
    ))
  }
  u0 <- as.matrix(unpenalized)
  if (nrow(u0) != n || any(!is.finite(u0))) {
    stop("Matrix `unpenalized` must have one finite row per observation.", call. = FALSE)
  }
  if (is.null(colnames(u0))) colnames(u0) <- paste0("u", seq_len(ncol(u0)))
  u <- cbind("(Intercept)" = 1, u0)
  list(matrix = unname(u), spec = list(type = "matrix", columns = colnames(u)))
}

.prediction_unpenalized <- function(object, newdata, new_unpenalized = NULL) {
  spec <- object$unpenalized_spec
  n <- nrow(newdata)
  if (identical(spec$type, "intercept")) return(matrix(1, n, 1L))
  if (identical(spec$type, "formula")) {
    if (!is.null(new_unpenalized)) {
      u <- as.matrix(new_unpenalized)
      if (nrow(u) != n || ncol(u) != length(spec$columns) || any(!is.finite(u))) {
        stop("`new_unpenalized` must be finite and match the fitted formula matrix.", call. = FALSE)
      }
      return(unname(u))
    }
    mf <- stats::model.frame(spec$terms, data = newdata, xlev = spec$xlevels,
                             na.action = stats::na.fail)
    u <- stats::model.matrix(spec$terms, mf, contrasts.arg = spec$contrasts)
    missing_cols <- setdiff(spec$columns, colnames(u))
    if (length(missing_cols)) {
      z <- matrix(0, n, length(missing_cols), dimnames = list(NULL, missing_cols))
      u <- cbind(u, z)
    }
    return(unname(u[, spec$columns, drop = FALSE]))
  }
  if (is.null(new_unpenalized)) {
    stop("`new_unpenalized` is required because the fit used a numeric unpenalized matrix.", call. = FALSE)
  }
  u0 <- as.matrix(new_unpenalized)
  if (nrow(u0) != n || any(!is.finite(u0))) {
    stop("`new_unpenalized` must have one finite row per prediction row.", call. = FALSE)
  }
  u <- cbind(1, u0)
  if (ncol(u) != length(spec$columns)) {
    stop("`new_unpenalized` has the wrong number of columns.", call. = FALSE)
  }
  unname(u)
}

.prepare_ridge_system <- function(z, y, u, weights) {
  sw <- sqrt(weights)
  zw <- z * sw
  yw <- as.numeric(y) * sw
  uw <- u * sw
  qru <- qr(uw, tol = 1e-10, LAPACK = FALSE)
  rank_u <- qru$rank
  if (rank_u) {
    qu <- qr.Q(qru, complete = FALSE)[, seq_len(rank_u), drop = FALSE]
    zr <- zw - qu %*% crossprod(qu, zw)
    yr <- yw - as.vector(qu %*% crossprod(qu, yw))
  } else {
    zr <- zw
    yr <- yw
  }
  list(
    gram = crossprod(zr),
    score = as.vector(crossprod(zr, yr)),
    z = z,
    y = as.numeric(y),
    u = u,
    weights = weights,
    zw = zw,
    yw = yw,
    uw = uw,
    qru = qru,
    rank_u = rank_u
  )
}

.solve_spd <- function(a, b) {
  p <- length(b)
  if (!p) return(numeric())
  # Cholesky is both faster and more stable for strictly penalized systems.
  ch <- tryCatch(chol(a), error = function(e) NULL)
  if (!is.null(ch)) return(as.vector(backsolve(ch, forwardsolve(t(ch), b))))
  # Zero-penalty truncated/saturated fits can be rank deficient because a
  # random sample omits otherwise supported profiles.  Return the unique fitted
  # values induced by the Moore-Penrose solution rather than failing on an
  # arbitrary coefficient representation.
  decomp <- eigen((a + t(a)) / 2, symmetric = TRUE)
  tolerance <- max(dim(a)) * max(abs(decomp$values), 1) * .Machine$double.eps
  keep <- decomp$values > tolerance
  if (!any(keep)) return(numeric(p))
  as.vector(decomp$vectors[, keep, drop = FALSE] %*%
              (crossprod(decomp$vectors[, keep, drop = FALSE], b) /
                 decomp$values[keep]))
}

.solve_ridge_system <- function(system, penalty) {
  p <- length(system$score)
  if (length(penalty) != p || anyNA(penalty) || any(penalty < 0)) {
    stop("Invalid coefficient-level penalty vector.", call. = FALSE)
  }
  keep <- is.finite(penalty)
  beta <- numeric(p)
  if (any(keep)) {
    a <- system$gram[keep, keep, drop = FALSE]
    diag(a) <- diag(a) + penalty[keep]
    beta[keep] <- .solve_spd(a, system$score[keep])
  }
  residual_w <- system$yw - system$zw %*% beta
  alpha <- stats::lm.fit(system$uw, residual_w, singular.ok = TRUE)$coefficients
  alpha[is.na(alpha)] <- 0
  fitted <- as.vector(system$u %*% alpha + system$z %*% beta)
  list(alpha = unname(alpha), beta = beta, fitted = fitted)
}

.softplus <- function(x) ifelse(x > 30, x, log1p(exp(x)))
.inv_softplus <- function(x) ifelse(x > 30, x, log(pmax(expm1(x), exp(-25))))

.decode_lambda <- function(par, structure, k, lower, upper) {
  if (identical(structure, "common")) {
    logs <- rep(lower + (upper - lower) * stats::plogis(par[1L]), k)
  } else if (identical(structure, "unrestricted")) {
    logs <- lower + (upper - lower) * stats::plogis(par[seq_len(k)])
  } else {
    first <- lower + (upper - lower) * stats::plogis(par[1L])
    if (k == 1L) {
      logs <- first
    } else {
      increments <- .softplus(par[2L:k])
      available <- max(upper - first, 1e-8)
      if (sum(increments) > available) increments <- increments * available / sum(increments)
      logs <- c(first, first + cumsum(increments))
    }
  }
  exp(pmin(upper, pmax(lower, logs)))
}

.encode_lambda <- function(lambda, structure, lower, upper) {
  logs <- log(pmin(exp(upper) * (1 - 1e-8), pmax(exp(lower) * (1 + 1e-8), lambda)))
  scaled_logit <- function(x) stats::qlogis(pmin(1 - 1e-7, pmax(1e-7, (x - lower) / (upper - lower))))
  if (identical(structure, "common")) return(scaled_logit(logs[1L]))
  if (identical(structure, "unrestricted")) return(vapply(logs, scaled_logit, numeric(1)))
  c(scaled_logit(logs[1L]), if (length(logs) > 1L) .inv_softplus(pmax(diff(logs), exp(-20))) else NULL)
}

.lambda_to_coefficients <- function(lambda, order) {
  ans <- lambda[match(order, seq_along(lambda))]
  if (anyNA(ans)) stop("Basis contains an unsupported interaction order.", call. = FALSE)
  ans
}

.expand_order_lambda <- function(lambda, represented_orders, all_orders,
                                 structure, common_lambda) {
  out <- rep(as.numeric(common_lambda), all_orders)
  out[represented_orders] <- lambda
  if (identical(structure, "monotone") && length(represented_orders)) {
    known <- stats::setNames(log(lambda), represented_orders)
    for (k in seq_len(all_orders)) {
      if (k %in% represented_orders) next
      below <- represented_orders[represented_orders < k]
      above <- represented_orders[represented_orders > k]
      if (!length(below)) out[k] <- lambda[1L]
      else if (!length(above)) out[k] <- lambda[length(lambda)]
      else {
        lo <- max(below)
        hi <- min(above)
        fraction <- (k - lo) / (hi - lo)
        out[k] <- exp((1 - fraction) * known[as.character(lo)] +
                        fraction * known[as.character(hi)])
      }
    }
    out <- cummax(out)
  }
  out
}

.cv_systems <- function(z, y, u, weights, fold_id) {
  folds <- sort(unique(fold_id))
  lapply(folds, function(f) {
    train <- fold_id != f
    valid <- !train
    wt <- .normalize_weights(weights[train], sum(train))
    list(
      train = .prepare_ridge_system(z[train, , drop = FALSE], y[train], u[train, , drop = FALSE], wt),
      z_valid = z[valid, , drop = FALSE],
      u_valid = u[valid, , drop = FALSE],
      y_valid = y[valid],
      w_valid = weights[valid]
    )
  })
}

.balanced_sure_start <- function(z, y, u, weights, order, structure,
                                 lower, upper, tolerance = 1e-8) {
  n <- nrow(z)
  if (n <= ncol(z) + ncol(u) || !length(weights) ||
      max(abs(weights - weights[1L])) > tolerance * max(abs(weights[1L]), 1)) {
    return(NULL)
  }
  system <- .prepare_ridge_system(z, y, u, .normalize_weights(weights, n))
  gram <- system$gram
  gram_scale <- mean(diag(gram))
  if (!is.finite(gram_scale) || gram_scale <= 0 ||
      max(abs(gram - diag(gram_scale, ncol(gram)))) >
        tolerance * max(gram_scale, 1)) {
    return(NULL)
  }
  model_rank <- qr(cbind(u, z), tol = 1e-10)$rank
  residual_df <- n - model_rank
  if (residual_df <= 0L) return(NULL)
  co <- .solve_ridge_system(system, rep(0, ncol(z)))
  sigma2 <- sum((y - co$fitted)^2) / residual_df
  coefficient_variance <- sigma2 / (n * gram_scale)
  sure <- balanced_order_sure(
    co$beta, order, coefficient_variance,
    monotone = identical(structure, "monotone")
  )
  lambda <- gram_scale * unname(sure$penalty)
  lambda <- pmin(exp(upper), pmax(exp(lower), lambda))
  if (length(lambda) != max(order) || any(!is.finite(lambda))) return(NULL)
  lambda
}

.cv_score_factory <- function(systems, order, structure, lower, upper, trace = FALSE) {
  history <- new.env(parent = emptyenv())
  history$lambda <- list()
  history$score <- numeric()
  score_lambda <- function(lambda) {
    penalty <- .lambda_to_coefficients(lambda, order)
    numerator <- denominator <- 0
    for (fold in systems) {
      co <- .solve_ridge_system(fold$train, penalty)
      pred <- as.vector(fold$u_valid %*% co$alpha + fold$z_valid %*% co$beta)
      numerator <- numerator + sum(fold$w_valid * (fold$y_valid - pred)^2)
      denominator <- denominator + sum(fold$w_valid)
    }
    value <- numerator / denominator
    history$lambda[[length(history$lambda) + 1L]] <- lambda
    history$score <- c(history$score, value)
    if (isTRUE(trace)) message(sprintf("CV %.6g: %s", value, paste(signif(lambda, 3), collapse = ", ")))
    value
  }
  objective <- function(par) {
    score_lambda(.decode_lambda(par, structure, max(order), lower, upper))
  }
  objective_log <- function(log_lambda) {
    score_lambda(exp(pmin(upper, pmax(lower, log_lambda))))
  }
  list(objective = objective, objective_log = objective_log,
       score_lambda = score_lambda, history = history)
}

.tune_order_penalties <- function(systems, order, structure, control) {
  k <- max(order)
  lower <- as.numeric(control$log_lambda_lower %||% -10)
  upper <- as.numeric(control$log_lambda_upper %||% 8)
  grid_n <- control$common_grid_size %||% 15L
  common_grid <- exp(seq(lower, upper, length.out = grid_n))
  common_factory <- .cv_score_factory(systems, order, "common", lower, upper)
  common_scores <- vapply(common_grid, function(lam) {
    common_factory$score_lambda(rep(lam, k))
  }, numeric(1))
  best_common <- common_grid[which.min(common_scores)]
  start_lambda <- rep(best_common, k)

  if (k == 1L) {
    return(list(
      lambda = best_common,
      cv_error = min(common_scores),
      convergence = 0L,
      solver_status = 0L,
      solver_message = "common-grid search",
      solver_evaluations = length(common_scores),
      used_common_fallback = FALSE,
      common_lambda = best_common,
      common_cv_error = min(common_scores),
      evaluations = data.frame(score = common_scores, lambda_1 = common_grid)
    ))
  }

  if (!identical(structure, "common") && !is.null(control$start_lambda)) {
    supplied <- rep_len(as.numeric(control$start_lambda), k)
    if (all(is.finite(supplied)) && all(supplied > 0)) start_lambda <- supplied
  }
  if (identical(structure, "monotone")) {
    if (is.null(control$start_lambda)) {
      start_lambda <- best_common * exp(seq(0, 1, length.out = k))
      start_lambda <- pmin(exp(upper) * (1 - 1e-6), start_lambda)
    }
    start_lambda <- cummax(start_lambda)
  }
  factory <- .cv_score_factory(systems, order, structure, lower, upper,
                               trace = isTRUE(control$trace))
  if (identical(structure, "common")) {
    chosen <- rep(best_common, k)
    opt <- list(par = .encode_lambda(chosen, "common", lower, upper),
                value = min(common_scores), convergence = 0L,
                message = "common-grid search", counts = length(common_scores))
    used_common_fallback <- FALSE
  } else {
    log_start <- pmin(upper, pmax(lower, log(start_lambda)))
    nloptr_args <- list(
      x0 = log_start,
      eval_f = factory$objective_log,
      lb = rep(lower, k),
      ub = rep(upper, k),
      opts = list(
        algorithm = "NLOPT_LN_COBYLA",
        maxeval = as.integer(control$maxit %||% 80L),
        xtol_rel = control$reltol %||% 1e-5,
        print_level = if (isTRUE(control$optim_trace)) 1L else 0L
      )
    )
    if (identical(structure, "monotone")) {
      nloptr_args$eval_g_ineq <- function(log_lambda) {
        log_lambda[-length(log_lambda)] - log_lambda[-1L]
      }
    }
    nlopt <- do.call(nloptr::nloptr, nloptr_args)
    chosen <- exp(pmin(upper, pmax(lower, nlopt$solution)))
    opt <- list(par = nlopt$solution, value = nlopt$objective,
                convergence = nlopt$status, message = nlopt$message,
                counts = nlopt$iterations)
    used_common_fallback <- FALSE
    if (identical(structure, "monotone")) {
      # Enforce the public postcondition even if the constrained optimizer
      # returns a solution with a sub-tolerance numerical violation.
      chosen <- cummax(chosen)
      opt$value <- factory$score_lambda(chosen)
    }
    # Never return a worse result than the nested ordinary-ridge model.
    if (!is.finite(opt$value) || opt$value > min(common_scores)) {
      chosen <- rep(best_common, k)
      opt$value <- min(common_scores)
      used_common_fallback <- TRUE
    }
  }
  history_lambda <- if (identical(structure, "common")) {
    lapply(common_grid, function(v) rep(v, k))
  } else factory$history$lambda
  history_score <- if (identical(structure, "common")) common_scores else factory$history$score
  if (identical(structure, "monotone") && length(history_lambda)) {
    # COBYLA may evaluate infeasible trial points while enforcing its nonlinear
    # constraints.  Keep the public tuning trace on the feasible parameter
    # space so every recorded row is an admissible fitted model.
    feasible <- vapply(history_lambda, function(v) {
      all(is.finite(v)) && all(diff(v) >= 0)
    }, logical(1))
    history_lambda <- history_lambda[feasible]
    history_score <- history_score[feasible]
  }
  if (!length(history_lambda) ||
      !any(vapply(history_lambda, function(v) {
        isTRUE(all.equal(as.numeric(v), as.numeric(chosen), tolerance = 1e-10))
      }, logical(1)))) {
    history_lambda <- c(history_lambda, list(chosen))
    history_score <- c(history_score, opt$value)
  }
  list(
    lambda = chosen,
    cv_error = opt$value,
    convergence = opt$convergence,
    solver_status = opt$convergence,
    solver_message = opt$message,
    solver_evaluations = opt$counts,
    used_common_fallback = used_common_fallback,
    common_lambda = best_common,
    common_cv_error = min(common_scores),
    evaluations = data.frame(
      score = history_score,
      do.call(rbind, lapply(history_lambda, function(v) stats::setNames(v, paste0("lambda_", seq_len(k)))))
    )
  )
}

`%||%` <- function(x, y) if (is.null(x)) y else x

#' Fit order-specific ridge regression on a factorial function basis
#'
#' @param x Data frame containing the randomized factors.
#' @param y Numeric outcome.
#' @param basis A `factorial_basis` object.
#' @param weights Optional nonnegative observation weights.
#' @param cluster Optional independent-cluster identifier used to construct
#'   leakage-free tuning folds. Defaults to one cluster per row.
#' @param unpenalized `NULL`, a one-sided formula evaluated in `x`, or a numeric
#'   matrix of nuisance regressors. An intercept is always included for `NULL`
#'   and numeric matrices.
#' @param monotone Logical; enforce nondecreasing penalties by order. Ignored
#'   when `penalty_structure` is supplied.
#' @param penalty_structure One of `"monotone"`, `"unrestricted"`, or `"common"`.
#' @param lambda Optional fixed vector containing one penalty per fitted order.
#' @param nfolds Number of clustered tuning folds.
#' @param fold_id Optional precomputed row-level fold labels.
#' @param seed Fold-assignment seed.
#' @param active_orders Orders to include. Omitted orders receive infinite penalty.
#' @param control Tuning controls: `maxit` (maximum NLopt objective
#'   evaluations), `reltol` (relative parameter tolerance),
#'   `common_grid_size`, `log_lambda_lower`, `log_lambda_upper`, and optional
#'   `start_lambda`. Exact balanced designs use `balanced_order_sure()` as the
#'   default adaptive-optimizer initializer; set `balanced_sure_initializer =
#'   FALSE` to disable it.
#' @return An object of class `order_ridge`.
#' @export
fit_order_ridge <- function(x, y, basis, weights = NULL, cluster = NULL,
                            unpenalized = NULL, monotone = TRUE,
                            penalty_structure = NULL, lambda = NULL,
                            nfolds = 5L, fold_id = NULL, seed = 1L,
                            active_orders = NULL, control = list()) {
  if (!inherits(basis, "factorial_basis")) stop("`basis` must be a factorial_basis object.", call. = FALSE)
  x <- as.data.frame(x)
  y <- as.numeric(y)
  n <- nrow(x)
  if (length(y) != n || any(!is.finite(y))) stop("`y` must be finite with one value per row of `x`.", call. = FALSE)
  w_raw <- if (is.null(weights)) rep(1, n) else as.numeric(weights)
  w <- .normalize_weights(w_raw, n)
  z <- factorial_design(basis, x, include_intercept = FALSE)
  order <- as.integer(attr(z, "order") %||% basis$order)
  if (length(order) != ncol(z) || any(order < 1L)) stop("Invalid order metadata in `basis`.", call. = FALSE)
  k <- basis$max_order
  if (is.null(active_orders)) active_orders <- seq_len(k)
  active_orders <- sort(unique(as.integer(active_orders)))
  if (!length(active_orders) || any(!active_orders %in% seq_len(k))) stop("Invalid `active_orders`.", call. = FALSE)
  active <- order %in% active_orders
  u_info <- .training_unpenalized(x, unpenalized)
  u <- u_info$matrix
  structure <- penalty_structure %||% if (isTRUE(monotone)) "monotone" else "unrestricted"
  structure <- match.arg(structure, c("monotone", "unrestricted", "common"))

  tuning <- NULL
  if (is.null(lambda)) {
    if (is.null(fold_id)) {
      if (is.null(cluster)) cluster <- seq_len(n)
      fold_id <- cluster_folds(cluster, nfolds = nfolds, seed = seed)
    }
    if (length(fold_id) != n || anyNA(fold_id) || length(unique(fold_id)) < 2L) {
      stop("`fold_id` must supply at least two non-missing folds.", call. = FALSE)
    }
    if (!is.null(cluster)) {
      if (length(cluster) != n || anyNA(cluster)) {
        stop("`cluster` must have one non-missing value per observation.", call. = FALSE)
      }
      cluster_fold_count <- vapply(split(fold_id, as.character(cluster)),
                                   function(v) length(unique(v)), integer(1))
      if (any(cluster_fold_count != 1L)) {
        stop("`fold_id` splits at least one cluster across tuning folds.", call. = FALSE)
      }
    }
    systems <- .cv_systems(z[, active, drop = FALSE], y, u, w_raw, fold_id)
    active_order <- order[active]
    represented_orders <- sort(unique(active_order))
    compressed_order <- match(active_order, represented_orders)
    initializer <- if (!is.null(control$start_lambda)) "user" else "common-grid"
    if (!identical(structure, "common") && is.null(control$start_lambda) &&
        !identical(control$balanced_sure_initializer, FALSE)) {
      lower <- as.numeric(control$log_lambda_lower %||% -10)
      upper <- as.numeric(control$log_lambda_upper %||% 8)
      sure_start <- .balanced_sure_start(
        z[, active, drop = FALSE], y, u, w_raw, compressed_order,
        structure, lower, upper
      )
      if (!is.null(sure_start)) {
        control$start_lambda <- sure_start
        initializer <- "balanced-sure"
      }
    }
    tuning <- .tune_order_penalties(systems, compressed_order, structure, control)
    tuning$initializer <- initializer
    lambda_represented <- tuning$lambda
    lambda <- .expand_order_lambda(
      lambda_represented, represented_orders, k, structure,
      common_lambda = tuning$common_lambda
    )
    tuning$represented_orders <- represented_orders
    tuning$lambda_represented <- stats::setNames(lambda_represented, represented_orders)
    tuning$lambda <- stats::setNames(lambda, seq_len(k))
  } else {
    lambda <- as.numeric(lambda)
    if (length(lambda) == 1L) lambda <- rep(lambda, k)
    if (length(lambda) != k || anyNA(lambda) || any(lambda < 0)) {
      stop("`lambda` must be nonnegative with one value per interaction order.", call. = FALSE)
    }
    if (identical(structure, "monotone") && is.unsorted(lambda, strictly = FALSE)) {
      stop("Monotone fits require nondecreasing `lambda`.", call. = FALSE)
    }
  }
  if (length(lambda) < k) {
    lambda <- c(lambda, rep(lambda[length(lambda)], k - length(lambda)))
  }
  penalty <- .lambda_to_coefficients(lambda, order)
  penalty[!active] <- Inf
  final_system <- .prepare_ridge_system(z, y, u, w)
  co <- .solve_ridge_system(final_system, penalty)
  names(co$alpha) <- u_info$spec$columns
  names(co$beta) <- colnames(z)
  residuals <- y - co$fitted
  effective_n <- sum(w_raw)^2 / sum(w_raw^2)
  object <- list(
    call = match.call(), basis = basis, coefficients = co$beta,
    unpenalized_coefficients = co$alpha, lambda = stats::setNames(lambda, seq_len(k)),
    coefficient_penalty = penalty, order = order, active_orders = active_orders,
    penalty_structure = structure, tuning = tuning, fold_id = fold_id,
    fitted.values = co$fitted, residuals = residuals,
    weighted_mse = sum(w * residuals^2), weights = w_raw,
    effective_n = effective_n, unpenalized_spec = u_info$spec,
    training_unpenalized = if (identical(u_info$spec$type, "matrix")) u else NULL
  )
  class(object) <- "order_ridge"
  object
}

#' @export
predict.order_ridge <- function(object, newdata = NULL, new_unpenalized = NULL, ...) {
  if (is.null(newdata)) return(object$fitted.values)
  newdata <- as.data.frame(newdata)
  z <- factorial_design(object$basis, newdata, include_intercept = FALSE)
  u <- .prediction_unpenalized(object, newdata, new_unpenalized)
  as.vector(u %*% object$unpenalized_coefficients + z %*% object$coefficients)
}

#' @export
coef.order_ridge <- function(object, ...) {
  c(object$unpenalized_coefficients, object$coefficients)
}

#' @export
print.order_ridge <- function(x, ...) {
  cat("Order-adaptive ridge fit\n")
  cat("  penalty structure:", x$penalty_structure, "\n")
  cat("  observations:", length(x$residuals), "(effective n", signif(x$effective_n, 5), ")\n")
  cat("  basis coefficients:", length(x$coefficients), "\n")
  cat("  lambda by order:", paste(signif(x$lambda, 5), collapse = ", "), "\n")
  cat("  weighted training MSE:", signif(x$weighted_mse, 6), "\n")
  invisible(x)
}

#' Summarize fitted energy by interaction order
#'
#' @param object An `order_ridge` fit.
#' @return Data frame with total and per-degree-of-freedom target-space energy.
#' @export
order_energy <- function(object) {
  if (!inherits(object, "order_ridge")) stop("`object` must be an order_ridge fit.", call. = FALSE)
  groups <- sort(unique(object$order))
  do.call(rbind, lapply(groups, function(k) {
    b <- object$coefficients[object$order == k]
    data.frame(order = k, df = length(b), energy = sum(b^2),
               energy_per_df = if (length(b)) mean(b^2) else NA_real_)
  }))
}

#' Rank a finite set of profiles by fitted response
#'
#' @param object An `order_ridge` fit.
#' @param profiles Data frame of profiles; defaults to the basis target support.
#' @param new_unpenalized Optional nuisance matrix for numeric-matrix fits.
#' @param decreasing Rank larger fitted outcomes first.
#' @return `profiles` with `.prediction` and `.rank` columns.
#' @export
rank_profiles <- function(object, profiles = object$basis$target,
                          new_unpenalized = NULL, decreasing = TRUE) {
  profiles <- as.data.frame(profiles)
  pred <- predict(object, profiles, new_unpenalized = new_unpenalized)
  ord <- order(pred, decreasing = decreasing, na.last = NA)
  out <- profiles[ord, , drop = FALSE]
  out$.prediction <- pred[ord]
  out$.rank <- seq_len(nrow(out))
  rownames(out) <- NULL
  out
}
