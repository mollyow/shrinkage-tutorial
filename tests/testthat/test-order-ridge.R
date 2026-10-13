if (!exists("fit_order_ridge", mode = "function") ||
    !exists("balanced_order_sure", mode = "function")) {
  source_candidates <- function(filename) {
    candidates <- c(
      file.path("R", filename),
      file.path("..", "..", "R", filename)
    )
    found <- candidates[file.exists(candidates)][1L]
    if (is.na(found)) stop(sprintf("Cannot locate R/%s", filename))
    source(found)
  }
  source_candidates("factorial_basis.R")
  source_candidates("cluster_folds.R")
  source_candidates("order_ridge.R")
  source_candidates("sure_pava.R")
  rm(source_candidates)
}

order_ridge_binary_grid <- function(n_factors = 2L) {
  values <- stats::setNames(
    replicate(n_factors, factor(c("0", "1"), levels = c("0", "1")),
              simplify = FALSE),
    LETTERS[seq_len(n_factors)]
  )
  out <- expand.grid(values, KEEP.OUT.ATTRS = FALSE)
  out[] <- lapply(out, factor)
  out
}

testthat::test_that("fixed order penalties equal direct generalized ridge", {
  target <- order_ridge_binary_grid(2)
  rows <- rep(seq_len(nrow(target)), times = 4L)
  x <- target[rows, , drop = FALSE]
  y <- c(0.4, 1.8, -0.7, 2.9) [rows] + rep(c(-0.2, 0.1, 0.3, -0.1), each = 4L)
  weights <- seq_len(nrow(x)) + 0.5
  basis <- factorial_basis(x, target = target, max_order = 2)
  lambda <- c(0.35, 1.7)

  fit <- fit_order_ridge(
    x, y, basis, weights = weights, lambda = lambda
  )

  z <- factorial_design(basis, x, include_intercept = FALSE)
  design <- cbind("(Intercept)" = 1, z)
  normalized_weights <- weights / sum(weights)
  weighted_design <- sweep(design, 1L, sqrt(normalized_weights), `*`)
  weighted_y <- y * sqrt(normalized_weights)
  penalty <- diag(c(0, lambda[attr(z, "order")]))
  direct <- solve(
    crossprod(weighted_design) + penalty,
    crossprod(weighted_design, weighted_y)
  )

  testthat::expect_equal(unname(coef(fit)), as.vector(direct), tolerance = 2e-10)
  testthat::expect_equal(
    unname(predict(fit)), as.vector(design %*% direct), tolerance = 2e-10
  )
  testthat::expect_equal(
    unname(fit$coefficient_penalty), lambda[attr(z, "order")]
  )
})

testthat::test_that("positive penalties give a finite fit when p exceeds n", {
  target <- expand.grid(
    A = factor(c("a0", "a1", "a2"), levels = c("a0", "a1", "a2")),
    B = factor(c("b0", "b1", "b2"), levels = c("b0", "b1", "b2")),
    C = factor(c("c0", "c1"), levels = c("c0", "c1")),
    KEEP.OUT.ATTRS = FALSE
  )
  target[] <- lapply(target, factor)
  x <- target[c(1, 3, 7, 12, 18), , drop = FALSE]
  basis <- factorial_basis(x, target = target, max_order = 3)
  y <- c(-1.2, 0.7, 2.1, -0.4, 1.3)

  testthat::expect_gt(ncol(factorial_design(basis, x)), nrow(x))
  fit <- fit_order_ridge(x, y, basis, lambda = c(0.1, 0.8, 4))

  testthat::expect_true(all(is.finite(coef(fit))))
  testthat::expect_true(all(is.finite(predict(fit, target))))
  testthat::expect_length(fit$coefficients, basis$rank)
})

testthat::test_that("positive penalties are stable at the p approximately n boundary", {
  target <- order_ridge_binary_grid(3)
  x <- target[-nrow(target), , drop = FALSE]
  basis <- factorial_basis(x, target = target, max_order = 3)
  y <- seq(-1, 1, length.out = nrow(x))

  testthat::expect_equal(ncol(factorial_design(basis, x)), nrow(x))
  fit <- fit_order_ridge(x, y, basis, lambda = c(0.1, 0.7, 3))
  testthat::expect_true(all(is.finite(predict(fit, target))))
})

testthat::test_that("zero-penalty truncated fits tolerate sample rank deficiency", {
  target <- order_ridge_binary_grid(3)
  x <- target[c(1, 1, 2, 2, 5, 5), , drop = FALSE]
  basis <- factorial_basis(x, target = target, max_order = 3)
  y <- c(0, 0.2, 1, 1.2, -0.5, -0.3)

  fit <- fit_order_ridge(
    x, y, basis, lambda = c(0, 0, 0), active_orders = 1:2,
    penalty_structure = "unrestricted"
  )

  testthat::expect_true(all(is.finite(predict(fit))))
  testthat::expect_true(all(is.finite(predict(fit, target))))
})

testthat::test_that("monotone penalties validate and tuning records ordered paths", {
  target <- order_ridge_binary_grid(3)
  rows <- rep(seq_len(nrow(target)), times = 6L)
  x <- target[rows, , drop = FALSE]
  cluster <- rep(sprintf("respondent-%02d", 1:6), each = nrow(target))
  aa <- as.integer(x$A) - 1
  bb <- as.integer(x$B) - 1
  cc <- as.integer(x$C) - 1
  respondent_shift <- rep(c(-0.4, -0.1, 0, 0.15, 0.25, 0.45), each = nrow(target))
  y <- 1 + aa + 0.5 * bb - 0.3 * cc + 0.8 * aa * bb + respondent_shift
  basis <- factorial_basis(x, target = target, max_order = 3)

  testthat::expect_error(
    fit_order_ridge(x, y, basis, lambda = c(1, 0.5, 2)),
    "nondecreasing"
  )
  unrestricted <- fit_order_ridge(
    x, y, basis, lambda = c(1, 0.5, 2), monotone = FALSE
  )
  testthat::expect_equal(unname(unrestricted$lambda), c(1, 0.5, 2))

  fit <- fit_order_ridge(
    x, y, basis, cluster = cluster, nfolds = 3L, seed = 29L,
    control = list(common_grid_size = 5L, maxit = 25L, reltol = 1e-7)
  )
  evaluations <- fit$tuning$evaluations

  testthat::expect_length(fit$lambda, 3L)
  testthat::expect_true(all(diff(fit$lambda) >= -1e-12))
  testthat::expect_identical(
    names(evaluations), c("score", "lambda_1", "lambda_2", "lambda_3")
  )
  testthat::expect_gt(nrow(evaluations), 0L)
  testthat::expect_true(all(apply(
    evaluations[paste0("lambda_", 1:3)], 1L,
    function(lambda_row) all(diff(lambda_row) >= -1e-12)
  )))
  testthat::expect_lte(fit$tuning$cv_error, fit$tuning$common_cv_error + 1e-12)

  bounded_start <- fit_order_ridge(
    x, y, basis, cluster = cluster, nfolds = 3L, seed = 29L,
    control = list(
      common_grid_size = 5L, maxit = 10L,
      log_lambda_lower = -4, log_lambda_upper = 2,
      start_lambda = c(1e-20, 1, 1e20)
    )
  )
  testthat::expect_true(all(is.finite(bounded_start$lambda)))
  testthat::expect_true(all(diff(bounded_start$lambda) >= -1e-12))

  integer_yaml_bounds <- fit_order_ridge(
    x, y, basis, cluster = cluster, nfolds = 3L, seed = 29L,
    penalty_structure = "unrestricted",
    control = list(
      common_grid_size = 3L, maxit = 5L,
      log_lambda_lower = -10L, log_lambda_upper = 8L
    )
  )
  testthat::expect_true(all(is.finite(integer_yaml_bounds$lambda)))
})

testthat::test_that("cluster folds isolate clusters and preserve caller RNG", {
  cluster <- rep(letters[1:8], times = c(1, 2, 3, 4, 2, 5, 1, 3))
  set.seed(8143)
  rng_before <- .Random.seed
  folds <- cluster_folds(cluster, nfolds = 4L, seed = 91L)

  testthat::expect_identical(.Random.seed, rng_before)
  testthat::expect_identical(
    folds, cluster_folds(cluster, nfolds = 4L, seed = 91L)
  )
  testthat::expect_equal(sort(unique(folds)), 1:4)
  testthat::expect_true(all(vapply(
    split(folds, cluster), function(values) length(unique(values)) == 1L,
    logical(1)
  )))
})

testthat::test_that("supplied tuning folds may not split a cluster", {
  target <- order_ridge_binary_grid(2)
  x <- target[rep(seq_len(4), times = 4L), , drop = FALSE]
  y <- seq_len(nrow(x)) / 10
  cluster <- rep(paste0("r", 1:4), each = 4L)
  leaky_fold <- rep(c(1L, 2L), length.out = nrow(x))
  basis <- factorial_basis(x, target = target, max_order = 2)

  testthat::expect_error(
    fit_order_ridge(
      x, y, basis, cluster = cluster, fold_id = leaky_fold,
      penalty_structure = "common",
      control = list(common_grid_size = 3L)
    ),
    "cluster|fold|isolat", ignore.case = TRUE
  )
})

testthat::test_that("balanced SURE uses the analytic PAVA solution", {
  result <- balanced_order_sure(
    z = c(3, 1, 2), order = 1:3, noise_variance = 1
  )
  unconstrained <- balanced_order_sure(
    z = c(3, 1, 2), order = 1:3, noise_variance = 1,
    monotone = FALSE
  )

  testthat::expect_equal(
    unname(result$raw_shrinkage), c(8 / 9, 0, 3 / 4), tolerance = 1e-12
  )
  testthat::expect_equal(
    unname(result$shrinkage), c(8 / 9, 3 / 5, 3 / 5), tolerance = 1e-12
  )
  testthat::expect_equal(
    unname(result$penalty), c(1 / 8, 2 / 3, 2 / 3), tolerance = 1e-12
  )
  testthat::expect_equal(
    unname(result$sure_by_order), c(8 / 9, 0.36, 0.84), tolerance = 1e-12
  )
  testthat::expect_equal(result$sure, 8 / 9 + 0.36 + 0.84, tolerance = 1e-12)
  testthat::expect_equal(
    unname(unconstrained$shrinkage), c(8 / 9, 0, 3 / 4), tolerance = 1e-12
  )
  testthat::expect_error(
    balanced_order_sure(c(1, 2, 3), 1:3, c(1, 2)),
    "conformable|same length|scalar", ignore.case = TRUE
  )
  testthat::expect_error(
    balanced_order_sure(c(1, 2), c(1, 3), 1), "consecutive"
  )
  testthat::expect_error(
    balanced_order_sure(c(1, 2), 1:2, c(1, -1)), "nonnegative"
  )
  testthat::expect_error(
    balanced_order_sure(c(Inf, 1), 1:2, 1, monotone = FALSE), "z|finite"
  )
})

testthat::test_that("balanced designs use SURE/PAVA as the adaptive initializer", {
  target <- order_ridge_binary_grid(3)
  x <- target[rep(seq_len(nrow(target)), times = 5L), , drop = FALSE]
  fold_id <- rep(seq_len(5L), each = nrow(target))
  surface <- c(-0.8, 0.1, 0.4, 1.0, -0.3, 0.7, 1.2, 1.9)
  noise <- rep(c(-0.2, 0.1, 0.25, -0.1, -0.05), each = nrow(target)) +
    rep(seq(-0.15, 0.15, length.out = nrow(target)), times = 5L)
  y <- rep(surface, times = 5L) + noise
  basis <- factorial_basis(x, target = target, max_order = 3)

  fit <- fit_order_ridge(
    x, y, basis, fold_id = fold_id, penalty_structure = "monotone",
    control = list(common_grid_size = 3L, maxit = 8L)
  )
  without_sure <- fit_order_ridge(
    x, y, basis, fold_id = fold_id, penalty_structure = "unrestricted",
    control = list(
      common_grid_size = 3L, maxit = 5L,
      balanced_sure_initializer = FALSE
    )
  )

  testthat::expect_identical(fit$tuning$initializer, "balanced-sure")
  testthat::expect_identical(without_sure$tuning$initializer, "common-grid")
  testthat::expect_true(all(diff(fit$lambda) >= 0))
})

testthat::test_that("predictions are invariant to factor reference coding", {
  target_one <- expand.grid(
    A = factor(c("a", "b", "c"), levels = c("a", "b", "c")),
    B = factor(c("no", "yes"), levels = c("no", "yes")),
    KEEP.OUT.ATTRS = FALSE
  )
  target_one[] <- lapply(target_one, factor)
  target_weights <- c(1, 3, 2, 5, 4, 7)
  rows <- rep(seq_len(nrow(target_one)), times = 3L)
  x_one <- target_one[rows, , drop = FALSE]
  y <- c(-0.2, 0.4, 1.7, 0.9, 2.8, -1.1)[rows] +
    rep(c(-0.1, 0.2, 0.05), each = nrow(target_one))

  target_two <- target_one
  target_two$A <- stats::relevel(target_two$A, ref = "c")
  target_two$B <- factor(as.character(target_two$B), levels = c("yes", "no"))
  x_two <- target_two[rows, , drop = FALSE]
  basis_one <- factorial_basis(
    x_one, target = target_one, target_weights = target_weights, max_order = 2
  )
  basis_two <- factorial_basis(
    x_two, target = target_two, target_weights = target_weights, max_order = 2
  )
  fit_one <- fit_order_ridge(x_one, y, basis_one, lambda = c(0.35, 2.1))
  fit_two <- fit_order_ridge(x_two, y, basis_two, lambda = c(0.35, 2.1))

  testthat::expect_equal(
    predict(fit_one, target_one), predict(fit_two, target_two),
    tolerance = 2e-8
  )
  testthat::expect_equal(predict(fit_one), predict(fit_two), tolerance = 2e-8)
})

testthat::test_that("fitted functions and contrasts are invariant to order-basis rotations", {
  target <- order_ridge_binary_grid(3)
  x <- target[rep(seq_len(nrow(target)), times = 5L), , drop = FALSE]
  y <- rep(c(-0.3, 0.4, 1.1, 1.8, -0.6, 0.9, 1.5, 2.7), times = 5L) +
    rep(seq(-0.2, 0.2, length.out = 5L), each = nrow(target))
  basis <- factorial_basis(x, target = target, max_order = 3)

  set.seed(9301)
  rotated <- basis
  lower_transform <- matrix(1, 1L, 1L)
  for (k in seq_len(basis$max_order)) {
    spec <- rotated$order_specs[[k]]
    spec$lower_coef <- solve(lower_transform, spec$lower_coef)
    d <- spec$rank
    rotation <- if (d == 1L) {
      matrix(sample(c(-1, 1), 1L), 1L, 1L)
    } else {
      qr.Q(qr(matrix(stats::rnorm(d * d), d, d)))
    }
    spec$raw_to_basis <- spec$raw_to_basis %*% rotation
    rotated$order_specs[[k]] <- spec
    expanded <- matrix(0, nrow(lower_transform) + d, ncol(lower_transform) + d)
    expanded[seq_len(nrow(lower_transform)), seq_len(ncol(lower_transform))] <-
      lower_transform
    expanded[nrow(lower_transform) + seq_len(d),
             ncol(lower_transform) + seq_len(d)] <- rotation
    lower_transform <- expanded
  }
  rotated$target_design <- factorial_design(rotated, target)
  rotated$design <- factorial_design(rotated, x)

  lambda <- c(0.2, 0.9, 3.5)
  original_fit <- fit_order_ridge(x, y, basis, lambda = lambda)
  rotated_fit <- fit_order_ridge(x, y, rotated, lambda = lambda)
  contrast <- c(1, -1, -1, 1, -1, 1, 1, -1)

  testthat::expect_equal(
    predict(original_fit, target), predict(rotated_fit, target),
    tolerance = 2e-8
  )
  testthat::expect_equal(
    sum(contrast * predict(original_fit, target)),
    sum(contrast * predict(rotated_fit, target)), tolerance = 2e-8
  )
  testthat::expect_equal(
    order_energy(original_fit)$energy,
    order_energy(rotated_fit)$energy, tolerance = 2e-8
  )
})

testthat::test_that("order-ridge inputs fail clearly", {
  target <- order_ridge_binary_grid(2)
  basis <- factorial_basis(target, max_order = 2)

  testthat::expect_error(
    fit_order_ridge(target, 1:3, basis, lambda = c(1, 2)), "one value per row"
  )
  testthat::expect_error(
    fit_order_ridge(target, 1:4, basis, lambda = c(1, 2, 3)),
    "one value per interaction order"
  )
  testthat::expect_error(
    fit_order_ridge(target, 1:4, basis, lambda = c(1, -1)), "nonnegative"
  )
  testthat::expect_error(
    fit_order_ridge(target, 1:4, basis, lambda = c(1, 2), active_orders = 3),
    "active_orders"
  )
  testthat::expect_error(cluster_folds(rep("only-one", 4)), "two independent")
  testthat::expect_error(cluster_folds(1:4, nfolds = 1), "at least two")
})
