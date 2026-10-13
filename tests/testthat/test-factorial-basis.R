if (!exists("factorial_basis", mode = "function")) {
  candidates <- c(
    file.path("R", "factorial_basis.R"),
    file.path("..", "..", "R", "factorial_basis.R")
  )
  basis_file <- candidates[file.exists(candidates)][1L]
  if (is.na(basis_file)) stop("Cannot locate R/factorial_basis.R")
  source(basis_file)
}

make_grid <- function() {
  grid <- expand.grid(
    A = factor(c("a1", "a2", "a3"), levels = c("a1", "a2", "a3")),
    B = factor(c("b1", "b2"), levels = c("b1", "b2")),
    C = factor(c("c1", "c2", "c3", "c4"),
               levels = c("c1", "c2", "c3", "c4")),
    KEEP.OUT.ATTRS = FALSE
  )
  grid[] <- lapply(grid, factor)
  grid
}

product_weights <- function(grid) {
  pa <- c(a1 = 0.15, a2 = 0.35, a3 = 0.50)
  pb <- c(b1 = 0.70, b2 = 0.30)
  pc <- c(c1 = 0.10, c2 = 0.20, c3 = 0.30, c4 = 0.40)
  unname(pa[as.character(grid$A)] * pb[as.character(grid$B)] *
           pc[as.character(grid$C)])
}

weighted_gram <- function(x, w) crossprod(x, sweep(x, 1L, w, `*`))

testthat::test_that("product-target tensor basis is exactly weighted orthonormal", {
  grid <- make_grid()
  w <- product_weights(grid)
  basis <- factorial_basis(grid, target_weights = w, max_order = 3)
  x <- factorial_design(basis, grid)

  testthat::expect_equal(
    unname(weighted_gram(x, basis$target_weights)), diag(ncol(x)),
    tolerance = 1e-10
  )
  testthat::expect_equal(
    as.vector(crossprod(basis$target_weights, x)),
    rep(0, ncol(x)), tolerance = 1e-10
  )
  testthat::expect_identical(attr(x, "order"), basis$order)
  testthat::expect_identical(as.integer(basis$df_by_order), c(6L, 11L, 6L))
  testthat::expect_equal(ncol(x), 23L)
  testthat::expect_equal(basis$rank, 23L)
  testthat::expect_length(basis$order, ncol(x))
  testthat::expect_length(attr(x, "scope"), ncol(x))
})

testthat::test_that("a separate full target grid controls geometry", {
  grid <- make_grid()
  w <- product_weights(grid)
  observed <- grid[c(1, 2, 7, 11, 18, 24), , drop = FALSE]
  basis <- factorial_basis(
    observed, target = grid, target_weights = w, max_order = 3
  )

  testthat::expect_equal(nrow(basis$design), nrow(observed))
  testthat::expect_equal(nrow(basis$target_design), nrow(grid))
  testthat::expect_equal(basis$rank, 23L)
  testthat::expect_lt(basis$diagnostics$observed_rank, basis$rank)
  testthat::expect_true(basis$diagnostics$full_joint_support)
})

testthat::test_that("dependent target weights produce orthogonal order spaces", {
  grid <- make_grid()
  index <- seq_len(nrow(grid))
  w <- 0.2 + (index %% 7) + 0.5 * (index %% 3)
  basis <- factorial_basis(grid, target_weights = w, max_order = 3)
  x <- basis$target_design

  testthat::expect_equal(
    unname(weighted_gram(x, basis$target_weights)), diag(ncol(x)),
    tolerance = 2e-9
  )
  testthat::expect_equal(
    as.vector(crossprod(basis$target_weights, x)),
    rep(0, ncol(x)), tolerance = 2e-9
  )
  for (k in 2:3) {
    current <- x[, basis$order == k, drop = FALSE]
    lower <- cbind(1, x[, basis$order < k, drop = FALSE])
    testthat::expect_equal(
      unname(crossprod(lower, sweep(current, 1L, basis$target_weights, `*`))),
      matrix(0, ncol(lower), ncol(current)), tolerance = 2e-9
    )
  }
  testthat::expect_true(length(basis$diagnostics$mixed_orders) > 0L)
})

testthat::test_that("reference and level permutations preserve weighted projections", {
  grid <- make_grid()
  w <- 0.5 + (seq_len(nrow(grid)) %% 11)
  first <- factorial_basis(grid, target_weights = w, max_order = 3)

  permuted <- grid
  permuted$A <- stats::relevel(permuted$A, ref = "a3")
  permuted$B <- factor(as.character(permuted$B), levels = rev(levels(permuted$B)))
  permuted$C <- factor(as.character(permuted$C),
                       levels = c("c3", "c1", "c4", "c2"))
  second <- factorial_basis(permuted, target_weights = w, max_order = 3)

  root_w <- sqrt(first$target_weights)
  for (k in 1:3) {
    x1 <- sweep(first$target_design[, first$order == k, drop = FALSE],
                1L, root_w, `*`)
    x2 <- sweep(second$target_design[, second$order == k, drop = FALSE],
                1L, root_w, `*`)
    testthat::expect_equal(tcrossprod(x1), tcrossprod(x2), tolerance = 2e-8)
  }
})

testthat::test_that("support restrictions and rank loss are diagnosed", {
  full <- expand.grid(
    A = factor(c("a0", "a1"), levels = c("a0", "a1")),
    B = factor(c("b0", "b1"), levels = c("b0", "b1")),
    KEEP.OUT.ATTRS = FALSE
  )
  full[] <- lapply(full, factor)
  target <- full[c(1, 4), , drop = FALSE]
  basis <- factorial_basis(
    full, target = target, target_weights = c(0.5, 0.5), max_order = 2
  )

  testthat::expect_false(basis$diagnostics$full_joint_support)
  testthat::expect_true(basis$diagnostics$rank_deficient)
  testthat::expect_equal(unname(basis$nominal_df_by_order), c(2L, 1L))
  testthat::expect_equal(unname(basis$df_by_order), c(1L, 0L))
  testthat::expect_equal(basis$rank, 1L)
})

testthat::test_that("unseen, missing, and invalid factor levels fail clearly", {
  grid <- make_grid()
  basis <- factorial_basis(grid, max_order = 2)

  unseen <- grid[1:2, , drop = FALSE]
  unseen$A <- factor(c("a1", "new"), levels = c(levels(grid$A), "new"))
  testthat::expect_error(factorial_design(basis, unseen), "Unseen levels")

  invalid <- grid[1:2, , drop = FALSE]
  invalid$B <- as.character(invalid$B)
  testthat::expect_error(factorial_design(basis, invalid), "must be factors")

  missing_column <- grid[1:2, c("A", "B"), drop = FALSE]
  testthat::expect_error(factorial_design(basis, missing_column), "missing factor columns")

  missing_value <- grid[1:2, , drop = FALSE]
  missing_value$C[1] <- NA
  testthat::expect_error(factorial_design(basis, missing_value), "missing values")

  zero_marginal <- grid[grid$A != "a3", , drop = FALSE]
  testthat::expect_error(
    factorial_basis(grid, target = zero_marginal, max_order = 2),
    "positive marginal probability"
  )
})

testthat::test_that("predict and transform delegate to factorial_design", {
  grid <- make_grid()
  basis <- factorial_basis(grid, max_order = 2)
  subset <- grid[c(2, 8, 19), , drop = FALSE]

  expected <- factorial_design(basis, subset)
  testthat::expect_equal(predict(basis, subset), expected)
  testthat::expect_equal(transform(basis, newdata = subset), expected)
  with_intercept <- factorial_design(basis, subset, include_intercept = TRUE)
  testthat::expect_equal(unname(with_intercept[, 1]), rep(1, nrow(subset)))
  testthat::expect_identical(attr(with_intercept, "order")[1], 0L)
})
