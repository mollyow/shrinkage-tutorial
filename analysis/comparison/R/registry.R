# Expected-job registries and the pre-calibration seed-collision check.

CMP_SEED_PURPOSES <- c("data", "folds", "target_integration", "learner", "oracle", "bootstrap")

cmp_calibration_cells <- function(sim) {
  c(rating = 7L, choice = 10L)
}

# Calibration stage 1: additive and oracle at every multiplier for both positive
# controls. Stage 2 (after selection): elastic net, forest, and boosting at the
# selected multiplier on the same datasets.
cmp_calibration_registry <- function(sim, stage = 1L, selected = NULL, draws) {
  reps <- seq_len(as.integer(sim$replication_schedule$calibration$replications_per_positive_control_candidate))
  multipliers <- as.numeric(unlist(sim$positive_control_calibration$multiplier_grid))
  cells <- cmp_calibration_cells(sim)
  out <- list()
  for (g in names(cells)) {
    ks <- if (stage == 1L) multipliers else selected[[g]]
    if (stage == 2L && (is.null(ks) || is.na(ks))) next
    methods <- if (stage == 1L) c("additive", "oracle") else
      as.character(unlist(sim$positive_control_calibration$feasible_recovery_gate$eligible_learners))
    methods <- sub("^elastic_net$", "elastic_net", methods)
    for (k in ks) for (r in reps) {
      out[[length(out) + 1L]] <- list(
        phase = "calibration", stage = stage, geometry = g, cell_id = cells[[g]],
        multiplier = k, replication = r, methods = methods,
        draws = if (g == "choice") draws else NA_integer_)
    }
  }
  out
}

cmp_phase_registry <- function(sim, phase, methods, draws, multipliers) {
  cells <- sim$cells
  sched <- sim$replication_schedule
  out <- list()
  for (cell in cells) {
    n <- if (phase == "smoke") sched$smoke$replications_per_cell else
      if (cell$name %in% unlist(sched$production$headline_cells))
        sched$production$headline_cells_total_replications else
          sched$production$replications_per_all_cells
    k <- if (grepl("calibrated", cell$treatment_heterogeneity)) multipliers[[cell$geometry]] else NULL
    for (r in seq_len(n)) {
      out[[length(out) + 1L]] <- list(
        phase = phase, geometry = cell$geometry, cell_id = as.integer(cell$id),
        multiplier = k, replication = r, methods = methods,
        draws = if (cell$geometry == "choice") draws else NA_integer_)
    }
  }
  out
}

# Every (phase, cell, replication, purpose) seed must be distinct. The only
# shared seeds are the intentional common random numbers across calibration
# multipliers, which share a cell and replication by construction.
cmp_seed_collision_check <- function(sim) {
  sched <- sim$replication_schedule
  keys <- list()
  add <- function(phase, cell, reps) {
    for (p in CMP_SEED_PURPOSES) {
      keys[[length(keys) + 1L]] <<- data.frame(
        phase = phase, cell_id = cell, replication = reps, purpose = p,
        seed = vapply(reps, function(r) cmp_seed(sim, phase, cell, r, p), integer(1)),
        stringsAsFactors = FALSE)
    }
  }
  for (cell in cmp_calibration_cells(sim)) {
    add("calibration", cell, seq_len(sched$calibration$replications_per_positive_control_candidate))
  }
  for (cell in sim$cells) {
    add("smoke", cell$id, seq_len(sched$smoke$replications_per_cell))
    n <- if (cell$name %in% unlist(sched$production$headline_cells))
      sched$production$headline_cells_total_replications else sched$production$replications_per_all_cells
    add("production", cell$id, seq_len(n))
    add("test", cell$id, seq_len(5L))
  }
  table <- do.call(rbind, keys)
  production_datasets <- sum(table$phase == "production" & table$purpose == "data")
  list(
    seeds = nrow(table),
    distinct = length(unique(table$seed)),
    passed = !anyDuplicated(table$seed),
    production_datasets = production_datasets,
    production_matches_contract = production_datasets == sched$production$total_production_datasets,
    calibration_common_random_numbers_across_multipliers = TRUE,
    test_phase_offset = CMP_TEST_PHASE_OFFSET
  )
}
