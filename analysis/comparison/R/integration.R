# Target integration of held-out nuisance predictions.
#
# Rating geometry: exact. For each held-out covariate context the learner
# predicts the complete 1,440-profile x 8-task grid once, and all four families
# are read off the same task-averaged profile surface.
#
# Choice geometry: unbiased family-cell-stratified Monte Carlo. For each
# respondent and family, D draws of the "rest" (the other focal factors, the
# opponent profile, task, and side) are taken from the declared target, and every
# family cell is enumerated against each draw. Using the same rest draws for all
# cells of a family keeps each cell mean unbiased (the product target makes the
# rest independent of the cell) and cancels the additive part of the Monte Carlo
# noise in each interaction coordinate. Draws are respondent-specific, depend
# only on the target-integration seed, and are common to every method.

CMP_PREDICT_CHUNK <- 20000L

cmp_predict_chunked <- function(predict_fun, rows) {
  n <- nrow(rows)
  out <- numeric(n)
  starts <- seq(1L, n, by = CMP_PREDICT_CHUNK)
  for (s in starts) {
    e <- min(n, s + CMP_PREDICT_CHUNK - 1L)
    out[s:e] <- predict_fun(rows[s:e, , drop = FALSE])
  }
  cmp_assert(all(is.finite(out)), "A learner returned non-finite integration predictions.")
  out
}

# ---------------------------------------------------------------------------
# Rating: exact enumeration by covariate context
# ---------------------------------------------------------------------------

cmp_rating_grid_rows <- function(geom) {
  n_prof <- nrow(geom$grid)
  rows <- as.data.frame(geom$grid[rep(seq_len(n_prof), times = geom$tasks), , drop = FALSE])
  rows$task <- rep(seq_len(geom$tasks), each = n_prof)
  rows$side <- 1L
  rows$id <- "integration"
  rows
}

# contexts: data frame of distinct covariate contexts (columns = context names).
cmp_rating_plugins <- function(geom, mu, predict_fun, contexts, grid_rows = cmp_rating_grid_rows(geom)) {
  n_prof <- nrow(geom$grid)
  out <- matrix(NA_real_, nrow(contexts), nrow(geom$coordinates),
                dimnames = list(NULL, geom$coordinates$key))
  projected <- 0
  total <- 0
  for (i in seq_len(nrow(contexts))) {
    rows <- grid_rows
    for (v in geom$context_names) rows[[v]] <- contexts[[v]][[i]]
    raw <- cmp_predict_chunked(predict_fun, rows)
    projected <- projected + sum(raw < 0 | raw > 1)
    total <- total + length(raw)
    pred <- cmp_project_unit(raw)
    profile_mean <- rowMeans(matrix(pred, n_prof, geom$tasks))
    out[i, ] <- cmp_theta_from_profiles(geom, profile_mean, mu)
  }
  list(plugins = out, projected = projected, total = total)
}

# ---------------------------------------------------------------------------
# Choice: family-cell-stratified Monte Carlo (or exact enumeration for tests)
# ---------------------------------------------------------------------------

cmp_rest_factors <- function(geom, family) setdiff(geom$factors, c(family$a, family$b))

# Level-balanced draws of one discrete component with L uniform levels: each
# level appears floor(D / L) times, the remainder is a uniform draw without
# replacement, and the sequence is randomly permuted. Every single draw is
# exactly uniform, so the integration stays unbiased; balancing removes the
# component's main-effect contribution to the Monte Carlo error.
cmp_balanced_levels <- function(n_levels, draws) {
  base <- rep(seq_len(n_levels), times = draws %/% n_levels)
  extra <- sample.int(n_levels, draws %% n_levels, replace = FALSE)
  out <- c(base, extra)
  out[sample.int(draws)]
}

# Rest draws for every respondent and family. Each rest component (other focal
# factors, each opponent factor, task, side) is level-balanced independently,
# so each draw follows the product target exactly (Latin-hypercube sampling
# over discrete factors).
cmp_choice_plan <- function(geom, ids, draws, seed) {
  lapply(geom$families, function(f) {
    rest <- cmp_rest_factors(geom, f)
    per <- lapply(ids, function(id) {
      cmp_with_seed(cmp_hashed_seed(seed, "target_integration", id, f$id), {
        out <- data.frame(id = rep(id, draws), draw = seq_len(draws), stringsAsFactors = FALSE)
        for (r in rest) out[[r]] <- cmp_balanced_levels(geom$levels[[r]], draws)
        for (j in seq_along(geom$factors)) {
          out[[geom$opponent_names[[j]]]] <- cmp_balanced_levels(geom$levels[[j]], draws)
        }
        out$task <- cmp_balanced_levels(geom$tasks, draws)
        out$side <- cmp_balanced_levels(geom$sides, draws)
        out$weight <- 1 / draws
        out
      })
    })
    do.call(rbind, per)
  })
}

# Exact rest table: every combination of the rest, weighted by the product target.
cmp_choice_rest_exact <- function(geom, family) {
  rest <- cmp_rest_factors(geom, family)
  pieces <- c(lapply(geom$levels[rest], seq_len),
              list(.opp = seq_len(nrow(geom$grid)), task = seq_len(geom$tasks),
                   side = seq_len(geom$sides)))
  out <- do.call(expand.grid, c(pieces, KEEP.OUT.ATTRS = FALSE))
  oc <- geom$grid[out$.opp, , drop = FALSE]
  for (j in seq_along(geom$factors)) out[[geom$opponent_names[[j]]]] <- oc[, j]
  out$.opp <- NULL
  out$weight <- 1 / nrow(out)
  out
}

cmp_choice_exact_plan <- function(geom, ids) {
  lapply(geom$families, function(f) {
    rest <- cmp_choice_rest_exact(geom, f)
    out <- rest[rep(seq_len(nrow(rest)), times = length(ids)), , drop = FALSE]
    out$id <- rep(ids, each = nrow(rest))
    out$draw <- rep(seq_len(nrow(rest)), times = length(ids))
    out
  })
}

# Plugins for respondents `ids` (respondent x coordinate matrix).
cmp_choice_plugins <- function(geom, plan, predict_fun, respondents, ids,
                               chunk_respondents = 40L) {
  out <- matrix(NA_real_, length(ids), nrow(geom$coordinates),
                dimnames = list(ids, geom$coordinates$key))
  projected <- 0
  total <- 0
  x <- respondents[match(ids, respondents$id), geom$context_names, drop = FALSE]
  for (f in geom$families) {
    cols <- geom$coordinates$key[geom$coordinates$family == f$id]
    p <- plan[[f$id]]
    p <- p[p$id %in% ids, , drop = FALSE]
    n_cell <- nrow(f$grid)
    for (start in seq(1L, length(ids), by = chunk_respondents)) {
      chunk <- ids[start:min(length(ids), start + chunk_respondents - 1L)]
      pc <- p[p$id %in% chunk, , drop = FALSE]
      pc <- pc[order(match(pc$id, chunk), pc$draw), , drop = FALSE]
      n_rest <- nrow(pc)
      rows <- pc[rep(seq_len(n_rest), times = n_cell), , drop = FALSE]
      rows[[f$a]] <- rep(f$grid$a_code, each = n_rest)
      rows[[f$b]] <- rep(f$grid$b_code, each = n_rest)
      xi <- x[match(rows$id, ids), , drop = FALSE]
      for (v in geom$context_names) rows[[v]] <- xi[[v]]
      raw <- cmp_predict_chunked(predict_fun, rows)
      projected <- projected + sum(raw < 0 | raw > 1)
      total <- total + length(raw)
      pred <- cmp_project_unit(raw) * rows$weight
      # cell means for each respondent: sum over that respondent's rest rows
      key <- factor(rows$id, levels = chunk)
      cell <- rep(seq_len(n_cell), each = n_rest)
      cell_means <- tapply(pred, list(key, cell), sum)
      cell_means <- matrix(cell_means, nrow = length(chunk))
      theta <- cell_means %*% (f$psi * f$cell_probability)
      out[chunk, cols] <- theta
    }
  }
  list(plugins = out, projected = projected, total = total)
}
