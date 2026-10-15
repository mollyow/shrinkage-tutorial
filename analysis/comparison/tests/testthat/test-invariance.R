# Gate C1: scoped coding and rotation invariance (comparison-v1.yml
# `invariance.exact_refit_scope`). Exact tolerance 1e-10 unless noted.

tol <- 1e-10

# Relabel the levels of one factor in a geometry's rows (focal and opponent).
cmp_relabel_rows <- function(geom, rows, factor, perm) {
  out <- rows
  out[[factor]] <- perm[rows[[factor]]]
  opp <- paste0("opp_", factor)
  if (opp %in% names(rows)) out[[opp]] <- perm[rows[[opp]]]
  out
}

cmp_small_dataset <- function(eng, cell_id, n, replication = 1L) {
  cell <- cmp_cell(eng, cell_id)
  geom <- cell$geometry_obj
  geom$respondents <- n
  s <- cmp_surface(eng, cell)
  seed <- cmp_seed(eng$contracts$simulation, "test", cell_id, replication, "data")
  d <- if (geom$name == "rating") cmp_rating_simulate(geom, s, seed) else cmp_choice_simulate(geom, s, seed)
  f <- cmp_respondent_folds(d$respondents$id, 5L, 5L, seed + 1L)
  d$rows$outer_fold <- f$outer_fold[match(d$rows$id, f$id)]
  d$rows$inner_fold <- f$inner_fold[match(d$rows$id, f$id)]
  list(geom = geom, surface = s, data = d, folds = f)
}

testthat::test_that("family bases are orthonormal, main-orthogonal, and give idempotent projectors", {
  eng <- cmp_test_engine()
  for (geom in eng$geoms) for (f in geom$families) {
    p <- f$cell_probability
    testthat::expect_lt(max(abs(crossprod(f$psi, f$psi * p) - diag(f$dimension))), tol)
    proj <- cmp_family_projector(f)
    testthat::expect_lt(max(abs(proj %*% proj - proj)), tol)
  }
})

testthat::test_that("family projectors are invariant to level permutations and baseline swaps", {
  eng <- cmp_test_engine()
  geom <- eng$geoms$choice
  f <- geom$families$religion_x_origin
  perms <- list(swap_baseline = c(2L, 1L, 3:7), permutation = c(5L, 3L, 7L, 1L, 6L, 2L, 4L))
  for (perm in perms) {
    # Basis for the relabeled factor, expressed back in the original cell order.
    hb <- cmp_helmert_basis(7L)[perm, , drop = FALSE]
    ha <- geom$bases$religion
    psi2 <- matrix(NA_real_, nrow(f$grid), f$dimension)
    cc <- 0L
    for (jb in seq_len(ncol(hb))) for (ja in seq_len(ncol(ha))) {
      cc <- cc + 1L
      psi2[, cc] <- ha[f$grid$a_code, ja] * hb[f$grid$b_code, jb]
    }
    proj2 <- psi2 %*% t(psi2) %*% diag(f$cell_probability)
    testthat::expect_lt(max(abs(proj2 - cmp_family_projector(f))), tol)
    # The two coordinate systems differ by an orthogonal rotation.
    q <- crossprod(f$psi, psi2 * f$cell_probability)
    testthat::expect_lt(max(abs(crossprod(q) - diag(f$dimension))), tol)
  }
})

testthat::test_that("family metrics are invariant to rotations of the registered coordinates", {
  eng <- cmp_test_engine()
  geom <- eng$geoms$choice
  set.seed(1)
  phi <- matrix(stats::rnorm(200 * nrow(geom$coordinates)), 200,
                dimnames = list(NULL, geom$coordinates$key))
  truth <- stats::setNames(stats::rnorm(ncol(phi), sd = 0.1), colnames(phi))
  fit <- cmp_aggregate(phi, rep(1, nrow(phi)))
  base <- cmp_family_inference(geom, fit, truth, 2000L, 1L)$family
  rot_phi <- phi
  rot_truth <- truth
  for (f in geom$families) {
    idx <- which(geom$coordinates$family == f$id)
    q <- qr.Q(qr(matrix(stats::rnorm(length(idx)^2), length(idx))))
    rot_phi[, idx] <- phi[, idx] %*% q
    rot_truth[idx] <- as.numeric(truth[idx] %*% q)
  }
  rot <- cmp_family_inference(geom, cmp_aggregate(rot_phi, rep(1, nrow(phi))), rot_truth, 2000L, 1L)$family
  for (col in c("squared_error", "normalized_risk", "variance_trace", "wald_statistic",
                "truth_wald_statistic")) {
    testthat::expect_lt(max(abs(rot[[col]] - base[[col]])), 1e-8)
  }
})

testthat::test_that("scores with identical fixed nuisance predictions have coding-invariant family metrics", {
  eng <- cmp_test_engine()
  ds <- cmp_small_dataset(eng, 9L, 60L)
  geom <- ds$geom
  rows <- ds$data$rows
  resp <- ds$data$respondents
  ids <- resp$id
  perm <- c(5L, 3L, 7L, 1L, 6L, 2L, 4L)
  inv <- order(perm)
  # Fixed nuisance defined on the original labels.
  nuis <- function(r) cmp_project_unit(0.5 + 0.03 * r$origin - 0.02 * r$religion * r$opp_origin +
                                         0.01 * r$z1 * r$vulnerability)
  geom2 <- geom
  ha <- geom$bases$religion
  hb <- geom$bases$origin
  # Relabeled data with the nuisance composed with the inverse relabeling.
  rows2 <- cmp_relabel_rows(geom, rows, "origin", perm)
  nuis2 <- function(r) nuis(cmp_relabel_rows(geom, r, "origin", inv))
  family_stats <- function(g, r, fun) {
    plan <- cmp_choice_plan(g, ids, 16L, 123L)
    plug <- cmp_choice_plugins(g, plan, fun, resp, ids)$plugins
    pred <- fun(r)
    phi <- plug + cmp_respondent_mean(cmp_riesz_rows(g, r) * (r$y - pred), r$id, ids)
    fit <- cmp_aggregate(phi, rep(1, length(ids)))
    vapply(g$families, function(f) {
      idx <- which(g$coordinates$family == f$id)
      c(norm = sum(fit$estimate[idx]^2), trace = sum(diag(fit$covariance[idx, idx])))
    }, numeric(2))
  }
  # The Monte Carlo draws are tied to factor codes, so compare only the
  # religion x origin family under exact enumeration of the rest.
  exact_stats <- function(r, fun) {
    f <- geom$families$religion_x_origin
    g1 <- geom
    g1$families <- geom$families["religion_x_origin"]
    g1$coordinates <- geom$coordinates[geom$coordinates$family == f$id, ]
    sub <- ids[1:6]
    plan <- cmp_choice_exact_plan(g1, sub)
    plug <- cmp_choice_plugins(g1, plan, fun, resp, sub)$plugins
    rr <- r[r$id %in% sub, ]
    phi <- plug + cmp_respondent_mean(cmp_riesz_rows(g1, rr) * (rr$y - fun(rr)), rr$id, sub)
    fit <- cmp_aggregate(phi, rep(1, length(sub)))
    c(norm = sum(fit$estimate^2), trace = sum(diag(fit$covariance)))
  }
  a <- exact_stats(rows, nuis)
  b <- exact_stats(rows2, nuis2)
  testthat::expect_lt(max(abs(a - b)), tol)
})

testthat::test_that("least-squares fitted functions are invariant to level permutations", {
  eng <- cmp_test_engine()
  # Full simulation sizes keep the two-way matrix inside its training row space.
  for (cell_id in c(5L, 9L)) {
    ds <- cmp_small_dataset(eng, cell_id, eng$geoms[[cmp_cell(eng, cell_id)$geometry]]$respondents)
    geom <- ds$geom
    fmap <- cmp_feature_map(geom)
    rows <- ds$data$rows
    factor <- geom$factors[[1]]
    perm <- rev(seq_len(geom$levels[[factor]]))
    rows2 <- cmp_relabel_rows(geom, rows, factor, perm)
    train <- rows$outer_fold != 1
    for (kind in c("additive", "two_way")) {
      ctx1 <- cmp_learner_context(geom, fmap, rows[train, ], rep(1, sum(train)),
                                  rows$inner_fold[train], 1L, eng$contracts)
      ctx2 <- cmp_learner_context(geom, fmap, rows2[train, ], rep(1, sum(train)),
                                  rows$inner_fold[train], 1L, eng$contracts)
      p1 <- cmp_fit_method(kind, ctx1)$predict(rows[!train, ])
      p2 <- cmp_fit_method(kind, ctx2)$predict(rows2[!train, ])
      testthat::expect_lt(max(abs(p1 - p2)), tol)
    }
  }
})

testthat::test_that("the least-squares guard rejects prediction outside the training row space", {
  eng <- cmp_test_engine()
  ds <- cmp_small_dataset(eng, 9L, 40L)
  geom <- ds$geom
  rows <- ds$data$rows
  train <- rows$outer_fold != 1
  ctx <- cmp_learner_context(geom, cmp_feature_map(geom), rows[train, ], rep(1, sum(train)),
                             rows$inner_fold[train], 1L, eng$contracts)
  fit <- cmp_fit_method("two_way", ctx)
  testthat::expect_error(fit$predict(rows[!train, ]), "outside the training column space")
})

testthat::test_that("order-ridge fitted functions are invariant to level permutations", {
  eng <- cmp_test_engine()
  ds <- cmp_small_dataset(eng, 5L, 80L)
  geom <- ds$geom
  fmap <- cmp_feature_map(geom)
  rows <- ds$data$rows
  perm <- c(3L, 5L, 1L, 4L, 2L)
  rows2 <- cmp_relabel_rows(geom, rows, "fundraising", perm)
  train <- rows$outer_fold != 1
  ctx1 <- cmp_learner_context(geom, fmap, rows[train, ], rep(1, sum(train)),
                              rows$inner_fold[train], 1L, eng$contracts, basis = geom$ridge_basis)
  ctx2 <- cmp_learner_context(geom, fmap, rows2[train, ], rep(1, sum(train)),
                              rows$inner_fold[train], 1L, eng$contracts, basis = geom$ridge_basis)
  prep1 <- cmp_fit_preprocess(fmap, ctx1$train)
  prep2 <- cmp_fit_preprocess(fmap, ctx2$train)
  in1 <- cmp_ridge_inputs(geom, fmap, prep1, ctx1$train)
  in2 <- cmp_ridge_inputs(geom, fmap, prep2, ctx2$train)
  out1 <- cmp_ridge_inputs(geom, fmap, prep1, rows[!train, ])
  out2 <- cmp_ridge_inputs(geom, fmap, prep2, rows2[!train, ])
  for (structure in c("common", "unrestricted", "monotone")) {
    # Exact invariance at fixed order penalties.
    lambda <- c(0.5, 2, 8)
    g1 <- orderridge::fit_order_ridge(in1$x, ctx1$train$y, geom$ridge_basis, unpenalized = in1$u,
                                      penalty_structure = structure, lambda = lambda)
    g2 <- orderridge::fit_order_ridge(in2$x, ctx2$train$y, geom$ridge_basis, unpenalized = in2$u,
                                      penalty_structure = structure, lambda = lambda)
    q1 <- stats::predict(g1, newdata = out1$x, new_unpenalized = out1$u)
    q2 <- stats::predict(g2, newdata = out2$x, new_unpenalized = out2$u)
    testthat::expect_lt(max(abs(q1 - q2)), tol)

    # Tuned fits: the penalty objective is invariant; the continuous optimizer
    # (relative tolerance 1e-5) reproduces it to its convergence tolerance.
    method <- paste0(structure, "_order_ridge")
    f1 <- cmp_fit_method(method, ctx1)
    f2 <- cmp_fit_method(method, ctx2)
    p1 <- f1$predict(rows[!train, ])
    p2 <- f2$predict(rows2[!train, ])
    testthat::expect_lt(max(abs(p1 - f1$predict_package(rows[!train, ]))), tol)
    bound <- if (structure == "common") tol else 1e-5 * max(abs(p1))
    testthat::expect_lt(max(abs(p1 - p2)), bound)
  }
})
