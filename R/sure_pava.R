.pava_increasing <- function(y, w = rep(1, length(y))) {
  if (length(y) != length(w) || any(!is.finite(y)) || any(!is.finite(w)) || any(w <= 0)) {
    stop("PAVA inputs must be finite and have positive weights.", call. = FALSE)
  }
  means <- as.numeric(y)
  weights <- as.numeric(w)
  starts <- ends <- seq_along(y)
  m <- length(means)
  i <- 1L
  while (i < m) {
    if (means[i] <= means[i + 1L] + 1e-14) {
      i <- i + 1L
      next
    }
    new_w <- weights[i] + weights[i + 1L]
    means[i] <- (weights[i] * means[i] + weights[i + 1L] * means[i + 1L]) / new_w
    weights[i] <- new_w
    ends[i] <- ends[i + 1L]
    if (i + 1L < m) {
      destination <- (i + 1L):(m - 1L)
      source <- (i + 2L):m
      means[destination] <- means[source]
      weights[destination] <- weights[source]
      starts[destination] <- starts[source]
      ends[destination] <- ends[source]
    }
    m <- m - 1L
    means <- means[seq_len(m)]
    weights <- weights[seq_len(m)]
    starts <- starts[seq_len(m)]
    ends <- ends[seq_len(m)]
    if (i > 1L) i <- i - 1L
  }
  out <- numeric(length(y))
  for (j in seq_len(m)) out[starts[j]:ends[j]] <- means[j]
  out
}

#' Balanced-design SURE/PAVA order shrinkage
#'
#' Computes the orderwise linear shrinkage factors minimizing Stein's unbiased
#' risk estimate in an orthogonal balanced design, subject to nonincreasing
#' shrinkage by interaction order.  This is an initializer and diagnostic; the
#' general estimator tunes penalties using clustered cross-validation.
#'
#' @param z Estimated orthonormal-basis coefficients, excluding the intercept.
#' @param order Positive interaction order for each coefficient.
#' @param noise_variance Sampling variance of each element of `z`, either a
#'   scalar or a vector of the same length as `z`.
#' @param monotone Enforce nonincreasing shrinkage factors.
#' @return A list with orderwise shrinkage factors, penalties, and SURE.
#' @export
balanced_order_sure <- function(z, order, noise_variance, monotone = TRUE) {
  z <- as.numeric(z)
  order_raw <- as.numeric(order)
  if (!length(z) || length(z) != length(order_raw) || any(!is.finite(z)) ||
      any(!is.finite(order_raw)) || any(order_raw != as.integer(order_raw)) ||
      any(order_raw < 1L)) {
    stop("`z` and positive integer `order` must have equal nonzero length.", call. = FALSE)
  }
  order <- as.integer(order_raw)
  noise_variance <- as.numeric(noise_variance)
  if (!length(noise_variance) || !length(noise_variance) %in% c(1L, length(z))) {
    stop("`noise_variance` must be scalar or have the same length as `z`.", call. = FALSE)
  }
  v <- rep(noise_variance, length.out = length(z))
  if (any(!is.finite(v)) || any(v < 0)) {
    stop("`noise_variance` must be nonnegative and conformable with `z`.", call. = FALSE)
  }
  groups <- sort(unique(order))
  if (!identical(groups, seq_len(max(groups)))) {
    stop("Orders must be consecutive starting at one.", call. = FALSE)
  }
  ss <- vapply(groups, function(k) sum(z[order == k]^2), numeric(1))
  vv <- vapply(groups, function(k) sum(v[order == k]), numeric(1))
  raw <- ifelse(ss > 0, 1 - vv / ss, 0)
  raw <- pmin(1, pmax(0, raw))
  shrinkage <- raw
  if (isTRUE(monotone) && length(groups) > 1L) {
    # a_1 >= ... >= a_K is increasing after changing sign.
    shrinkage <- -.pava_increasing(-raw, pmax(ss, .Machine$double.eps))
    shrinkage <- pmin(1, pmax(0, shrinkage))
  }
  names(shrinkage) <- names(raw) <- groups
  penalty <- ifelse(shrinkage <= .Machine$double.eps, Inf, 1 / shrinkage - 1)
  sure_by_order <- vapply(seq_along(groups), function(j) {
    idx <- order == groups[j]
    a <- shrinkage[j]
    sum((1 - a)^2 * z[idx]^2 + 2 * a * v[idx] - v[idx])
  }, numeric(1))
  names(penalty) <- names(sure_by_order) <- groups
  structure(list(
    order = groups,
    raw_shrinkage = raw,
    shrinkage = shrinkage,
    penalty = penalty,
    sure_by_order = sure_by_order,
    sure = sum(sure_by_order)
  ), class = "order_sure")
}
