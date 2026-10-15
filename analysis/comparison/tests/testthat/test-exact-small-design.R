# Gate C1: exact small-design score and target-integration agreement, including
# paired-choice enumeration.

tol <- 1e-10

testthat::test_that("paired-choice exact integration matches brute-force enumeration", {
  cmp_test_engine()
  geom <- cmp_tiny_geometry("choice")
  resp <- data.frame(id = c("a", "b"), x1 = c(-0.7, 1.3), x2 = c(0, 1), stringsAsFactors = FALSE)
  fun <- function(rows) cmp_tiny_nuisance(geom, rows)
  plan <- cmp_choice_exact_plan(geom, resp$id)
  got <- cmp_choice_plugins(geom, plan, fun, resp, resp$id)$plugins
  for (i in seq_len(nrow(resp))) {
    want <- cmp_brute_theta(geom, fun, resp[i, c("x1", "x2")])
    testthat::expect_lt(max(abs(got[resp$id[[i]], ] - want)), tol)
  }
})

testthat::test_that("shared-rest Monte Carlo integration is unbiased for every cell", {
  cmp_test_engine()
  geom <- cmp_tiny_geometry("choice")
  resp <- data.frame(id = "a", x1 = 0.4, x2 = 1, stringsAsFactors = FALSE)
  # Shared rest draws cancel family main effects times the rest, and level
  # balancing makes a single rest component exact when its level count divides
  # the 8 draws (side, f3). The Monte Carlo error therefore comes from family
  # interactions with opp_f2 (3 levels) and with the joint rest component
  # f3 x opp_f1.
  fun <- function(rows) {
    cmp_tiny_nuisance(geom, rows) + 0.04 * rows$f1 * rows$opp_f2 - 0.03 * rows$f2 * rows$f3 +
      0.02 * rows$f1 * rows$side * rows$f2 + 0.04 * rows$f1 * rows$f2 * rows$opp_f2 +
      0.03 * rows$f1 * rows$f2 * rows$f3 * rows$opp_f1 - 0.05
  }
  exact <- cmp_brute_theta(geom, fun, resp[1, c("x1", "x2")])
  reps <- vapply(seq_len(400), function(s) {
    plan <- cmp_choice_plan(geom, resp$id, 8L, 9e7 + s)
    cmp_choice_plugins(geom, plan, fun, resp, resp$id)$plugins[1, ]
  }, numeric(length(exact)))
  sdv <- apply(reps, 1L, stats::sd)
  testthat::expect_true(all(sdv > 1e-6))
  z <- (rowMeans(reps) - exact) / (sdv / sqrt(ncol(reps)))
  testthat::expect_true(all(abs(z) < 4.5))
})

testthat::test_that("shared-rest draws are exact when cells and rest do not interact", {
  cmp_test_engine()
  geom <- cmp_tiny_geometry("choice")
  resp <- data.frame(id = "a", x1 = 0.4, x2 = 1, stringsAsFactors = FALSE)
  fun <- function(rows) cmp_tiny_nuisance(geom, rows)
  exact <- cmp_brute_theta(geom, fun, resp[1, c("x1", "x2")])
  plan <- cmp_choice_plan(geom, resp$id, 4L, 9e7 + 1)
  got <- cmp_choice_plugins(geom, plan, fun, resp, resp$id)$plugins[1, ]
  testthat::expect_lt(max(abs(got - exact)), tol)
})

testthat::test_that("the score is exactly unbiased for any nuisance under known randomization", {
  cmp_test_engine()
  for (kind in c("choice", "rating")) {
    geom <- cmp_tiny_geometry(kind)
    n <- nrow(geom$grid)
    # Rating variant shifts the target on the non-family factor f3.
    ratio <- if (kind == "rating") c(0.5, 1.5)[geom$grid[, "f3"]] else rep(1, n)
    mu <- ratio / n
    x <- list(x1 = 0.25, x2 = 1)
    truth_fun <- function(rows) cmp_project_unit(cmp_tiny_nuisance(geom, rows))
    wrong_fun <- function(rows) cmp_project_unit(0.6 - 0.5 * cmp_tiny_nuisance(geom, rows) + 0.07 * rows$f3)
    theta_true <- cmp_brute_theta(geom, truth_fun, x, mu)
    plugin_wrong <- cmp_brute_theta(geom, wrong_fun, x, mu)
    # Expected correction: enumerate the assignment law of one task row.
    corr <- numeric(length(theta_true))
    opps <- if (geom$has_opponent) seq_len(n) else NA_integer_
    count <- 0
    for (p in seq_len(n)) for (o in opps) for (t in seq_len(geom$tasks)) for (s in seq_len(geom$sides)) {
      row <- as.data.frame(as.list(geom$grid[p, ]))
      if (geom$has_opponent) {
        oc <- as.list(geom$grid[o, ])
        names(oc) <- geom$opponent_names
        row <- cbind(row, as.data.frame(oc))
      }
      row$task <- t
      row$side <- s
      row$x1 <- x$x1
      row$x2 <- x$x2
      row$target_ratio <- ratio[[p]]
      r <- cmp_riesz_rows(geom, row)
      corr <- corr + as.numeric(r) * (truth_fun(row) - wrong_fun(row))
      count <- count + 1
    }
    expected_phi <- plugin_wrong + corr / count
    testthat::expect_lt(max(abs(expected_phi - theta_true)), tol)
  }
})

testthat::test_that("rating exact integration reproduces the cached oracle plugins", {
  eng <- cmp_test_engine()
  geom <- eng$geoms$rating
  for (cell_id in c(5L, 6L)) {
    cell <- cmp_cell(eng, cell_id)
    s <- cmp_surface(eng, cell)
    mu <- s$target_ratio / nrow(geom$grid)
    ctx <- geom$contexts[seq_len(6), c("key", geom$context_names)]
    got <- cmp_rating_plugins(geom, mu, function(r) cmp_true_mean(geom, s, r), ctx,
                              geom$grid_rows)$plugins
    testthat::expect_lt(max(abs(got - s$oracle_plugins[seq_len(6), ])), tol)
    testthat::expect_lt(abs(sum(mu) - 1), tol)
  }
})

testthat::test_that("choice exact enumeration reproduces the 16-point truth", {
  eng <- cmp_test_engine()
  geom <- eng$geoms$choice
  s <- cmp_surface(eng, cmp_cell(eng, 9L))
  resp <- data.frame(id = "a", z1 = 0.3, z2 = -1.1, z3 = 0.8, z4 = 1.9, b1 = 1, b2 = 0,
                     b3 = 1, b4 = 0, stringsAsFactors = FALSE)
  exact <- cmp_choice_exact_plugins(geom, s, resp)
  # Enumerate the full rest for one family only (other factors, opponent, task, side).
  f <- geom$families[[1]]
  plan <- list()
  rest <- cmp_choice_rest_exact(geom, f)
  rest$id <- "a"
  rest$draw <- seq_len(nrow(rest))
  plan[[f$id]] <- rest
  geom1 <- geom
  geom1$families <- geom$families[1]
  geom1$coordinates <- geom$coordinates[geom$coordinates$family == f$id, ]
  got <- cmp_choice_plugins(geom1, plan, function(r) cmp_true_mean(geom, s, r), resp, "a")$plugins
  testthat::expect_lt(max(abs(got[1, ] - exact[1, geom1$coordinates$key])), tol)
})

testthat::test_that("profile indexing matches the expand.grid ordering", {
  eng <- cmp_test_engine()
  for (geom in eng$geoms) {
    rows <- as.data.frame(geom$grid)
    testthat::expect_identical(cmp_profile_index(geom, rows, geom$factors), seq_len(nrow(geom$grid)))
  }
})
