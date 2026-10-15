# Shared fixtures for the Gate C1 tests. Test data use the disjoint test seed
# phase and never touch calibration, smoke, or production streams.

cmp_test_root <- function() {
  root <- normalizePath(file.path(testthat::test_path(), "..", "..", "..", ".."))
  stopifnot(file.exists(file.path(root, "DESCRIPTION")))
  root
}

cmp_test_engine <- local({
  cache <- NULL
  function() {
    if (is.null(cache)) {
      old <- setwd(cmp_test_root())
      on.exit(setwd(old))
      source(file.path("analysis", "comparison", "load.R"), local = FALSE)
      cache <<- cmp_engine(getwd())
    }
    cache
  }
})

# A tiny geometry list with the fields the integration and score code reads.
cmp_tiny_geometry <- function(name = c("choice", "rating"), levels = c(f1 = 2L, f2 = 3L, f3 = 2L),
                              families = list(list(id = "f1_x_f2", attributes = c("f1", "f2"))),
                              tasks = 2L) {
  name <- match.arg(name)
  grid <- do.call(expand.grid, c(lapply(levels, seq_len), KEEP.OUT.ATTRS = FALSE))
  names(grid) <- names(levels)
  grid <- as.matrix(grid)
  storage.mode(grid) <- "integer"
  bases <- cmp_factor_bases(levels)
  fam <- cmp_build_families(bases, families)
  geom <- list(
    name = name, factors = names(levels), levels = levels, grid = grid, bases = bases,
    families = fam, coordinates = cmp_coordinate_table(fam), tasks = tasks,
    sides = if (name == "choice") 2L else 1L,
    has_opponent = name == "choice",
    opponent_names = paste0("opp_", names(levels)),
    context_names = c("x1", "x2"), x_numeric = "x1", x_binary = "x2",
    assignment_probability = 1 / nrow(grid)
  )
  geom
}

# An arbitrary smooth nuisance function of every row feature.
cmp_tiny_nuisance <- function(geom, rows, shift = 0) {
  f <- 0.55 + 0.05 * rows[[geom$factors[[1]]]] - 0.04 * rows[[geom$factors[[2]]]]^2 +
    0.03 * rows[[geom$factors[[1]]]] * rows[[geom$factors[[2]]]] +
    0.02 * rows$task + 0.1 * tanh(rows$x1) * rows[[geom$factors[[2]]]] / 3 + 0.05 * rows$x2
  if (isTRUE(geom$has_opponent)) {
    f <- f - 0.03 * rows[[geom$opponent_names[[1]]]] * rows[[geom$opponent_names[[2]]]] +
      0.01 * rows$side
  }
  f + shift
}

# Brute-force theta for one respondent: enumerate every focal profile, opponent,
# task, and side with explicit target weights, without any engine helper.
# Predictions are projected to [0, 1] exactly as the protocol requires.
cmp_brute_theta <- function(geom, fun, x, mu = NULL) {
  n <- nrow(geom$grid)
  if (is.null(mu)) mu <- rep(1 / n, n)
  out <- numeric(nrow(geom$coordinates))
  names(out) <- geom$coordinates$key
  for (p in seq_len(n)) {
    opps <- if (geom$has_opponent) seq_len(n) else NA_integer_
    acc <- 0
    for (o in opps) for (t in seq_len(geom$tasks)) for (s in seq_len(geom$sides)) {
      row <- as.data.frame(as.list(geom$grid[p, ]))
      if (geom$has_opponent) {
        oc <- as.list(geom$grid[o, ])
        names(oc) <- geom$opponent_names
        row <- cbind(row, as.data.frame(oc))
      }
      row$task <- t
      row$side <- s
      row$x1 <- x[["x1"]]
      row$x2 <- x[["x2"]]
      acc <- acc + cmp_project_unit(fun(row))
    }
    m_bar <- acc / (length(opps) * geom$tasks * geom$sides)
    for (f in geom$families) {
      cell <- cmp_family_cell(f, geom$grid[p, f$a], geom$grid[p, f$b])
      keys <- paste(f$id, colnames(f$psi), sep = "::")
      out[keys] <- out[keys] + mu[[p]] * f$psi[cell, ] * m_bar
    }
  }
  out
}
