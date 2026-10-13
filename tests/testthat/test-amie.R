if (!exists("estimate_amie", mode = "function")) {
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
  source_candidates("amie.R")
  rm(source_candidates)
}

amie_binary_grid <- function() {
  out <- expand.grid(
    A = factor(c("a0", "a1"), levels = c("a0", "a1")),
    B = factor(c("b0", "b1"), levels = c("b0", "b1")),
    KEEP.OUT.ATTRS = FALSE
  )
  out[] <- lapply(out, factor)
  out
}

testthat::test_that("DID contrasts are direct target-profile weights", {
  target <- amie_binary_grid()
  basis <- factorial_basis(target, target_weights = rep(1 / 4, 4), max_order = 2)
  did <- amie_contrast(
    basis, factors = c("A", "B"), type = "did",
    levels = list(A = c("a0", "a1"), B = c("b0", "b1"))
  )
  cell_means <- c(1, 3, 4, 11)

  testthat::expect_equal(as.vector(did), c(1, -1, -1, 1))
  testthat::expect_equal(sum(did), 0)
  testthat::expect_equal(as.numeric(crossprod(did, cell_means)), 5)
  testthat::expect_identical(attr(did, "definition"), "direct target-profile weights")
  testthat::expect_error(
    amie_contrast(basis, "A", type = "did", levels = list(A = c("a0", "a1"))),
    "exactly two"
  )
  testthat::expect_error(
    amie_contrast(
      basis, c("A", "B"), type = "did",
      levels = list(A = "a0", B = c("b0", "b1"))
    ),
    "baseline and comparison"
  )
  testthat::expect_error(
    amie_contrast(
      basis, c("A", "B"), type = "did",
      levels = list(A = c("a0", "a0"), B = c("b0", "b1"))
    ),
    "distinct"
  )
})

testthat::test_that("balanced 2x2 cross-fitting is exactly unbiased for prespecified AMIEs", {
  target <- amie_binary_grid()
  rows <- rep(seq_len(nrow(target)), times = 6L)
  x <- target[rows, , drop = FALSE]
  cluster <- rep(sprintf("r%d", 1:6), each = nrow(target))
  fold_id <- rep(c(1L, 1L, 2L, 2L, 3L, 3L), each = nrow(target))
  cell_means <- c(1, 3, 4, 11)
  y <- cell_means[rows]
  basis <- factorial_basis(x, target = target, max_order = 2)
  did <- amie_contrast(
    basis, c("A", "B"), type = "did",
    levels = list(A = c("a0", "a1"), B = c("b0", "b1"))
  )
  contrasts <- cbind(
    did = as.vector(did),
    A_marginal = c(-0.5, 0.5, -0.5, 0.5)
  )

  result <- estimate_amie(
    x = x, y = y, cluster = cluster, basis = basis,
    contrasts = contrasts, assignment_prob = rep(0.25, 4),
    fold_id = fold_id, fit_args = list(lambda = c(0.35, 2)),
    simultaneous = FALSE
  )

  testthat::expect_equal(result$estimates$estimate, c(5, 4.5), tolerance = 2e-10)
  testthat::expect_equal(
    result$scores,
    matrix(
      rep(c(5, 4.5), each = 6L), nrow = 6L,
      dimnames = list(sprintf("r%d", 1:6), c("did", "A_marginal"))
    ),
    tolerance = 2e-10
  )
  testthat::expect_identical(dim(result$covariance), c(2L, 2L))
  testthat::expect_identical(
    dimnames(result$covariance), list(c("did", "A_marginal"), c("did", "A_marginal"))
  )
  testthat::expect_equal(unname(result$covariance), matrix(0, 2, 2), tolerance = 2e-20)
  testthat::expect_true(all(vapply(
    split(result$outer_fold_id, cluster),
    function(values) length(unique(values)) == 1L,
    logical(1)
  )))
})

testthat::test_that("AMIE outer folds preserve RNG and isolate whole respondents", {
  target <- amie_binary_grid()
  rows <- rep(seq_len(4), times = 8L)
  x <- target[rows, , drop = FALSE]
  cluster <- rep(sprintf("id-%02d", 1:8), each = 4L)
  y <- rep(c(0, 1, 2, 4), 8L) + rep(seq(0, 0.7, by = 0.1), each = 4L)
  basis <- factorial_basis(x, target = target, max_order = 2)
  contrast <- matrix(c(1, -1, -1, 1), ncol = 1,
                     dimnames = list(NULL, "did"))
  set.seed(783)
  rng_before <- .Random.seed

  result <- estimate_amie(
    x, y, cluster, basis, contrast, rep(0.25, 4),
    outer_folds = 4L, fit_args = list(lambda = c(0.4, 1.5)),
    simultaneous = FALSE, seed = 44L
  )

  testthat::expect_identical(.Random.seed, rng_before)
  testthat::expect_true(all(vapply(
    split(result$outer_fold_id, cluster),
    function(values) length(unique(values)) == 1L,
    logical(1)
  )))
  testthat::expect_equal(length(unique(result$outer_fold_id)), 4L)
})

testthat::test_that("cross-fitted AMIEs are invariant to reference and level order", {
  target_one <- amie_binary_grid()
  rows <- rep(seq_len(4L), times = 6L)
  x_one <- target_one[rows, , drop = FALSE]
  y <- rep(c(0.2, 1.1, -0.4, 2.3), times = 6L) +
    rep(seq(-0.15, 0.15, length.out = 6L), each = 4L)
  cluster <- rep(paste0("r", seq_len(6L)), each = 4L)
  fold_id <- rep(c(1L, 1L, 2L, 2L, 3L, 3L), each = 4L)

  target_two <- target_one
  target_two$A <- factor(as.character(target_two$A), levels = c("a1", "a0"))
  target_two$B <- factor(as.character(target_two$B), levels = c("b1", "b0"))
  x_two <- target_two[rows, , drop = FALSE]
  basis_one <- factorial_basis(x_one, target = target_one, max_order = 2)
  basis_two <- factorial_basis(x_two, target = target_two, max_order = 2)
  levels_spec <- list(A = c("a0", "a1"), B = c("b0", "b1"))
  contrast_one <- amie_contrast(
    basis_one, c("A", "B"), type = "did", levels = levels_spec
  )
  contrast_two <- amie_contrast(
    basis_two, c("A", "B"), type = "did", levels = levels_spec
  )

  first <- estimate_amie(
    x_one, y, cluster, basis_one, contrast_one, rep(0.25, 4L),
    fold_id = fold_id, fit_args = list(lambda = c(0.3, 1.7)),
    simultaneous = FALSE
  )
  second <- estimate_amie(
    x_two, y, cluster, basis_two, contrast_two, rep(0.25, 4L),
    fold_id = fold_id, fit_args = list(lambda = c(0.3, 1.7)),
    simultaneous = FALSE
  )

  testthat::expect_equal(
    first$estimates[c("estimate", "std_error", "conf_low", "conf_high")],
    second$estimates[c("estimate", "std_error", "conf_low", "conf_high")],
    tolerance = 2e-8
  )
})

testthat::test_that("AMIE validation rejects malformed inputs and leaky folds", {
  target <- amie_binary_grid()
  rows <- rep(seq_len(4), times = 4L)
  x <- target[rows, , drop = FALSE]
  y <- rep(c(1, 3, 4, 11), 4L)
  cluster <- rep(paste0("r", 1:4), each = 4L)
  fold_id <- rep(c(1L, 1L, 2L, 2L), each = 4L)
  basis <- factorial_basis(x, target = target, max_order = 2)
  did <- matrix(c(1, -1, -1, 1), ncol = 1,
                dimnames = list(NULL, "did"))
  fit_args <- list(lambda = c(0.4, 2))

  testthat::expect_error(
    estimate_amie(
      x, y, cluster, basis, matrix(1:4, ncol = 1), rep(0.25, 4),
      fold_id = fold_id, fit_args = fit_args
    ),
    "sum to zero"
  )
  testthat::expect_error(
    estimate_amie(
      x, y, cluster, basis, matrix(c(1, -1), ncol = 1), rep(0.25, 4),
      fold_id = fold_id, fit_args = fit_args
    ),
    "one row per target profile"
  )
  testthat::expect_error(
    estimate_amie(
      x, y, cluster, basis, did, c(0.25, 0.25, 0, 0.5),
      fold_id = fold_id, fit_args = fit_args
    ),
    "finite.*positive"
  )

  outside <- x
  outside$A <- factor(as.character(outside$A), levels = c("a0", "a1", "new"))
  outside$A[1] <- "new"
  testthat::expect_error(
    estimate_amie(
      outside, y, cluster, basis, did, rep(0.25, 4),
      fold_id = fold_id, fit_args = fit_args
    ),
    "outside the target support"
  )

  leaky_fold <- rep(c(1L, 2L), length.out = nrow(x))
  testthat::expect_error(
    estimate_amie(
      x, y, cluster, basis, did, rep(0.25, 4),
      fold_id = leaky_fold, fit_args = fit_args
    ),
    "isolated|outer fold"
  )
  testthat::expect_error(
    estimate_amie(
      x, y[-1], cluster, basis, did, rep(0.25, 4),
      fold_id = fold_id, fit_args = fit_args
    ),
    "equal length"
  )
  testthat::expect_error(
    estimate_amie(
      x, y, cluster, basis, did, rep(0.25, 4),
      fold_id = fold_id, fit_args = list(lambda = c(0.4, 2), y = y)
    ),
    "duplicates reserved arguments"
  )
  testthat::expect_error(
    estimate_amie(
      x, y, cluster, basis, did, rep(0.25, 4),
      fold_id = fold_id,
      fit_args = list(lambda = c(0.4, 2), unpenalized = ~ A)
    ),
    "profile-invariant|randomized factors", ignore.case = TRUE
  )
  testthat::expect_error(
    estimate_amie(
      x, y, cluster, basis, did, rep(0.25, 4),
      fold_id = rep(1L, nrow(x)), fit_args = fit_args
    ),
    "at least two.*fold", ignore.case = TRUE
  )
  testthat::expect_error(
    estimate_amie(
      x, y, cluster, basis, did, rep(0.25, 4),
      fold_id = fold_id, fit_args = fit_args, level = 1.2,
      simultaneous = FALSE
    ),
    "level", ignore.case = TRUE
  )
  testthat::expect_error(
    estimate_amie(
      x, y, cluster, basis, did, rep(0.25, 4),
      fold_id = fold_id, fit_args = fit_args, simultaneous_draws = 0
    ),
    "simultaneous_draws|draws", ignore.case = TRUE
  )
})

testthat::test_that("target-wide overlap is checked even for unobserved profiles", {
  target <- amie_binary_grid()
  observed_rows <- rep(1:3, times = 4L)
  x <- target[observed_rows, , drop = FALSE]
  y <- c(1, 3, 4)[observed_rows]
  cluster <- rep(paste0("r", 1:4), each = 3L)
  fold_id <- rep(c(1L, 1L, 2L, 2L), each = 3L)
  basis <- factorial_basis(x, target = target, max_order = 2)
  did <- matrix(c(1, -1, -1, 1), ncol = 1,
                dimnames = list(NULL, "did"))

  testthat::expect_error(
    estimate_amie(
      x, y, cluster, basis, did,
      assignment_prob = c(0.25, 0.25, 0.25, 0),
      fold_id = fold_id, fit_args = list(lambda = c(0.4, 2)),
      simultaneous = FALSE
    ),
    "finite.*positive|overlap", ignore.case = TRUE
  )
})

testthat::test_that("cluster covariance requires two positive-weight respondents", {
  target <- amie_binary_grid()
  rows <- rep(seq_len(4), times = 4L)
  x <- target[rows, , drop = FALSE]
  y <- rep(c(1, 3, 4, 11), 4L)
  cluster <- rep(paste0("r", 1:4), each = 4L)
  fold_id <- rep(c(1L, 1L, 2L, 2L), each = 4L)
  respondent_weights <- rep(c(1, 0, 0, 0), each = 4L)
  basis <- factorial_basis(x, target = target, max_order = 2)
  did <- matrix(c(1, -1, -1, 1), ncol = 1,
                dimnames = list(NULL, "did"))

  testthat::expect_error(
    estimate_amie(
      x, y, cluster, basis, did, rep(0.25, 4),
      weights = respondent_weights, fold_id = fold_id,
      fit_args = list(lambda = c(0.4, 2)), simultaneous = FALSE
    ),
    "two positive-weight|independent clusters|positive sum", ignore.case = TRUE
  )
})
