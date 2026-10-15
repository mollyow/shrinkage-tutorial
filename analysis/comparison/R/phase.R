# Phase execution and Gate D evaluation, shared by run_simulations.R and
# evaluate_gate_d.R and covered by the Gate C1 tests.

cmp_job_path <- function(out_root, phase, stage, job) {
  m <- if (is.null(job$multiplier)) "fixed" else sprintf("m%05.2f", job$multiplier)
  file.path(out_root, sprintf("stage-%d", if (phase == "calibration") stage else 0L),
            job$geometry, sprintf("cell_%02d", job$cell_id), m,
            sprintf("rep_%04d.rds", job$replication))
}

# Drop calibration candidates whose generator is invalid; return them for the record.
cmp_filter_valid <- function(engine, jobs) {
  keep <- logical(length(jobs))
  invalid <- list()
  seen <- character()
  for (i in seq_along(jobs)) {
    j <- jobs[[i]]
    s <- cmp_surface(engine, cmp_cell(engine, j$cell_id), j$multiplier)
    keep[[i]] <- isTRUE(s$valid)
    key <- paste(j$geometry, j$multiplier %||% "fixed")
    if (!keep[[i]] && !key %in% seen) {
      seen <- c(seen, key)
      invalid[[length(invalid) + 1L]] <- list(geometry = j$geometry, multiplier = j$multiplier,
                                              conditional_mean_range = as.numeric(s$range))
    }
  }
  list(jobs = jobs[keep], invalid = invalid)
}

cmp_run_job <- function(engine, job) {
  tryCatch({
    res <- cmp_conform_dataset(cmp_run_dataset(engine, job))
    cmp_check_complete(res, engine$geoms[[job$geometry]], job$methods, 5L)
    cmp_save_rds(res, job$path)
    sprintf("%s cell %d rep %d: %.0fs", job$geometry, job$cell_id, job$replication,
            res$meta$runtime_seconds)
  }, error = function(e) paste("ERROR:", job$path, conditionMessage(e)))
}

# Collect every table per geometry/cell/multiplier into parts of 100 replications.
cmp_collect <- function(jobs, collect_dir) {
  groups <- split(jobs, vapply(jobs, function(j) paste(j$geometry, j$cell_id,
                                                        j$multiplier %||% "fixed", sep = "__"), ""))
  written <- character()
  for (g in names(groups)) {
    parts <- split(groups[[g]], ceiling(seq_along(groups[[g]]) / 100))
    for (p in names(parts)) {
      res <- lapply(parts[[p]], function(j) readRDS(j$path))
      stem <- sprintf("%s__part_%02d", g, as.integer(p))
      for (name in names(CMP_SCHEMAS)) {
        tab <- do.call(rbind, lapply(res, `[[`, name))
        path <- if (name %in% c("fold_predictions", "respondent_scores", "tuning_history")) {
          cmp_write_parquet(tab, file.path(collect_dir, name, paste0(stem, ".parquet")))
        } else {
          cmp_write_csv(tab, file.path(collect_dir, name, paste0(stem, ".csv")))
        }
        written <- c(written, path)
      }
    }
  }
  invisible(written)
}

# Family-balanced risk by geometry, multiplier, and method.
cmp_risk_by <- function(fm, methods) {
  fm <- fm[fm$method %in% methods, , drop = FALSE]
  per <- stats::aggregate(normalized_risk ~ geometry + multiplier + method + family, fm, mean)
  stats::aggregate(normalized_risk ~ geometry + multiplier + method, per, mean)
}

# Stage 1: oracle/additive ratio per candidate and the frozen selection rule.
cmp_gate_d_selection <- function(fm1, pc) {
  r <- cmp_risk_by(fm1, c("additive", "oracle"))
  add <- r[r$method == "additive", ]
  orc <- r[r$method == "oracle", ]
  wide <- merge(add, orc, by = c("geometry", "multiplier"), suffixes = c("_additive", "_oracle"))
  wide <- wide[, c("geometry", "multiplier", "normalized_risk_additive", "normalized_risk_oracle")]
  wide$ratio <- wide$normalized_risk_oracle / wide$normalized_risk_additive
  reps <- stats::aggregate(replication ~ geometry + multiplier,
                           unique(fm1[c("geometry", "multiplier", "replication")]), length)
  names(reps)[3] <- "replications"
  wide <- merge(wide, reps, by = c("geometry", "multiplier"))
  target <- as.numeric(pc$target_ratio)
  cutoff <- as.numeric(pc$passing_ratio_at_most)
  selected <- list()
  for (g in c("rating", "choice")) {
    w <- wide[wide$geometry == g & wide$ratio <= cutoff, , drop = FALSE]
    selected[[g]] <- if (nrow(w)) w$multiplier[order(abs(w$ratio - target), w$multiplier)[[1]]] else NA_real_
  }
  list(table = wide[order(wide$geometry, wide$multiplier), ], selected = selected)
}

# Stage 2: gain recovered at the selected multipliers, and the Gate D decision.
cmp_gate_d_recovery <- function(fm1, fm2, selected, pc) {
  eligible <- as.character(unlist(pc$feasible_recovery_gate$eligible_learners))
  minimum <- as.numeric(pc$feasible_recovery_gate$minimum_fraction_oracle_gain_recovered)
  cutoff <- as.numeric(pc$passing_ratio_at_most)
  out <- list()
  for (g in c("rating", "choice")) {
    k <- selected[[g]]
    if (is.null(k) || is.na(k)) {
      out[[g]] <- list(selected_multiplier = NA_real_, passed = FALSE,
                       reason = "no candidate reached the oracle/additive cutoff")
      next
    }
    both <- rbind(fm1[fm1$geometry == g & fm1$multiplier == k, , drop = FALSE],
                  fm2[fm2$geometry == g & fm2$multiplier == k, , drop = FALSE])
    reps1 <- unique(both$replication[both$method == "additive"])
    reps2 <- unique(both$replication[both$method %in% eligible])
    cmp_assert(setequal(reps1, reps2), "Stage 2 must use exactly the stage-1 calibration datasets.")
    r <- cmp_risk_by(both, c("additive", "oracle", eligible))
    risk <- stats::setNames(r$normalized_risk, r$method)
    cmp_assert(all(c("additive", "oracle", eligible) %in% names(risk)), "Missing calibration methods.")
    gain <- (risk[["additive"]] - risk[eligible]) / (risk[["additive"]] - risk[["oracle"]])
    ratio <- risk[["oracle"]] / risk[["additive"]]
    out[[g]] <- list(selected_multiplier = k, oracle_over_additive = ratio,
                     risks = as.list(risk), gain_recovered = as.list(gain),
                     best_learner = names(gain)[which.max(gain)],
                     passed = ratio <= cutoff && max(gain) >= minimum)
  }
  list(geometries = out,
       passed = all(vapply(out, function(x) isTRUE(x$passed), logical(1))),
       rule = list(oracle_over_additive_at_most = cutoff, minimum_gain_recovered = minimum,
                   eligible_learners = eligible, both_geometries_must_pass = TRUE))
}
