# Target-integration draw count for the paired-choice simulations.
#
# The count is fixed in comparison-simulations-v1.yml
# (`choice_geometry.target_integration.draws`, 128). C1 measures the RMS
# integration error at every count in the v1 grid and records it; the table is
# reported, not tested against a tolerance. The application draw count is chosen
# separately in C2 by the v1 one-percent rule.
#
# Only the paired-choice geometry uses Monte Carlo integration; the rating
# geometry is integrated exactly. The known functions are the true conditional
# means of every choice cell that calibration and production will integrate:
# cells 8 and 9 and cell 10 at each of the seven calibration multipliers.

cmp_choice_exact_plugins <- function(geom, surface, respondents) {
  raw <- cmp_choice_features_raw(as.matrix(respondents[geom$context_names]))
  combo <- 1L + raw[, 1] + 2L * raw[, 2] + 4L * raw[, 3] + 8L * raw[, 4]
  out <- surface$combo_theta[combo, , drop = FALSE]
  rownames(out) <- respondents$id
  out
}

cmp_select_draw_count <- function(engine, phase = "test", replication = 1L) {
  sim <- engine$contracts$simulation
  grid <- as.integer(unlist(engine$contracts$v1$target_integration$large_product_designs$synthetic_gate_c_grid))
  geom <- engine$geoms$choice
  multipliers <- as.numeric(unlist(sim$positive_control_calibration$multiplier_grid))
  functions <- c(list(list(cell = 8L, multiplier = NULL), list(cell = 9L, multiplier = NULL)),
                 lapply(multipliers, function(k) list(cell = 10L, multiplier = k)))
  rows <- list()
  for (fn in functions) {
    cell <- cmp_cell(engine, fn$cell)
    surface <- cmp_surface(engine, cell, fn$multiplier)
    data <- cmp_choice_simulate(geom, surface, cmp_seed(sim, phase, cell$id, replication, "data"))
    resp <- data$respondents
    ids <- resp$id
    truth_fun <- function(r) cmp_true_mean(geom, surface, r)
    exact <- cmp_choice_exact_plugins(geom, surface, resp)
    riesz <- cmp_riesz_rows(geom, data$rows)
    corr <- cmp_respondent_mean(riesz * (data$rows$y - data$rows$true_mean), data$rows$id, ids)
    phi <- exact[ids, , drop = FALSE] + corr
    for (d in grid) {
      plan <- cmp_choice_plan(geom, ids, d,
                              cmp_seed(sim, phase, cell$id, replication, "target_integration"))
      mc <- cmp_choice_plugins(geom, plan, truth_fun, resp, ids)$plugins
      for (f in geom$families) {
        cols <- geom$coordinates$key[geom$coordinates$family == f$id]
        err <- sqrt(mean(rowSums((mc[, cols, drop = FALSE] - exact[ids, cols, drop = FALSE])^2)) /
                      f$dimension)
        sdv <- sqrt(sum(diag(stats::cov(phi[, cols, drop = FALSE]))) / f$dimension)
        rows[[length(rows) + 1L]] <- data.frame(
          cell_id = cell$id, multiplier = fn$multiplier %||% NA_real_, draws = d,
          family = f$id, rms_integration_error = err, exact_score_sd = sdv,
          ratio = err / sdv, below_one_percent = err < 0.01 * sdv,
          stringsAsFactors = FALSE)
      }
    }
  }
  table <- do.call(rbind, rows)
  fixed <- as.integer(sim$choice_geometry$target_integration$draws)
  complete <- nrow(table) == length(functions) * length(grid) * length(geom$families) &&
    all(is.finite(table$ratio))
  list(table = table, selected = fixed, passed = complete && fixed %in% grid,
       worst_ratio_at_selected = max(table$ratio[table$draws == fixed]),
       grid = grid, rule = sim$choice_geometry$target_integration$rule)
}
