#!/usr/bin/env Rscript
# Gate C1: the shared engine and the simulations.
#
#   Rscript analysis/comparison/run_gate_c1.R [--workers=2]
#
# Runs every C1 check on synthetic and design-only inputs and writes
# results/comparison-v1/gate-c1/gate-c1-checks.yml. Calibration is authorized
# when that file records all_passed. The file also records the commit and the
# hashes of the code and configs the checks ran on. It never reads a fresh
# endpoint and never uses a calibration, smoke, or production seed.

args <- commandArgs(trailingOnly = TRUE)
workers_arg <- grep("^--workers=", args, value = TRUE)
workers <- if (length(workers_arg)) as.integer(sub("^--workers=", "", workers_arg[[1]])) else 2L

source(file.path("analysis", "comparison", "load.R"))
root <- getwd()
out_dir <- file.path(root, "results", "comparison-v1", "gate-c1")
checks_path <- file.path(root, CMP_GATE_PATHS$c1)

# Code, configs, and artifacts the C1 checks depend on, hashed for provenance.
cmp_c1_files <- function(root) {
  rel <- function(x) sub(paste0("^", root, "/"), "", x)
  pick <- function(dir, pattern) rel(sort(list.files(file.path(root, dir), pattern = pattern,
                                                     recursive = TRUE, full.names = TRUE)))
  c(
    pick("analysis/comparison", "\\.R$"),
    "analysis/candidate/build_comparison_context_v4.R",
    "analysis/candidate/derived/comparison_v1_context_distribution.rds",
    "analysis/candidate/estimands.yml",
    "DESCRIPTION", "NAMESPACE", pick("R", "\\.R$"), pick("tests", "\\.R$"),
    "renv.lock",
    "config/comparison-v1.yml", "config/comparison-simulations-v1.yml"
  )
}

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
session <- cmp_session_manifest(root)
checks <- list(
  created_utc = cmp_utc_now(),
  git_head = session$git_head,
  worktree_dirty = session$worktree_dirty,
  files = lapply(cmp_c1_files(root), function(f) list(path = f, sha256 = cmp_sha256(file.path(root, f))))
)
record <- function(name, passed, details = list()) {
  checks[[name]] <<- c(list(passed = isTRUE(passed)), details[names(details) != "passed"])
  cat(sprintf("[%s] %s\n", if (isTRUE(passed)) "PASS" else "FAIL", name))
}

contracts <- cmp_load_contracts(root)
sim <- contracts$simulation

# 1. Frozen learner implementations are installed and locked.
frozen <- c(glmnet = "4.1.8", ranger = "0.18.0", xgboost = "3.2.1.1")
lock <- renv::lockfile_read(file.path(root, "renv.lock"))
installed <- vapply(names(frozen), function(p) as.character(utils::packageVersion(p)), "")
locked <- vapply(names(frozen), function(p) gsub("-", ".", lock$Packages[[p]]$Version %||% "-"), "")
record("frozen_learner_versions", all(installed == frozen) && all(locked == frozen),
       list(frozen = as.list(frozen), installed = as.list(installed), renv_lock = as.list(locked),
            nanoparquet_locked = lock$Packages$nanoparquet$Version %||% NA_character_))

# 2. Outcome-free version-4 context artifact.
art <- yaml::read_yaml(file.path(root, "analysis/candidate/derived/comparison_v1_context_distribution-manifest.yml"))
art_ok <- identical(art$sha256, cmp_sha256(file.path(root, art$artifact))) &&
  identical(as.character(unlist(art$columns_read)), as.character(unlist(sim$rating_geometry$context_artifact$selected_columns))) &&
  !("interest" %in% unlist(art$columns_read)) && art$respondents == 3000L
record("candidate_v4_context_artifact", art_ok,
       list(sha256 = art$sha256, distinct_contexts = art$distinct_contexts))

# 3. Seed collisions.
seed_check <- cmp_seed_collision_check(sim)
record("seed_collision_check", seed_check$passed && seed_check$production_matches_contract, seed_check)

# 4. Generator validity across every cell and calibration candidate.
engine <- cmp_engine(root)
validity <- list()
for (cell in sim$cells) {
  ks <- if (grepl("calibrated", cell$treatment_heterogeneity))
    as.numeric(unlist(sim$positive_control_calibration$multiplier_grid)) else list(NULL)
  for (k in ks) {
    s <- cmp_surface(engine, cmp_cell(engine, cell$id), k)
    validity[[length(validity) + 1L]] <- list(
      cell_id = cell$id, cell = cell$name, multiplier = k %||% NA_real_, valid = isTRUE(s$valid),
      conditional_mean_range = if (is.null(s$range)) NA else as.numeric(s$range))
  }
}
fixed_valid <- all(vapply(validity, function(x) x$valid || !is.na(x$multiplier), logical(1)))
record("generator_validity", fixed_valid, list(
  candidates = validity,
  invalid_calibration_candidates = Filter(function(x) !x$valid, validity)))

# 5. Simulation draw count: fixed in the simulation config; the error table is recorded.
dc <- cmp_select_draw_count(engine)
cmp_write_yaml(list(rule = dc$rule, grid = dc$grid, selected = dc$selected,
                    worst_ratio_at_selected = dc$worst_ratio_at_selected,
                    table = lapply(split(dc$table, seq_len(nrow(dc$table))), as.list)),
               file.path(out_dir, "draw-count-selection.yml"))
record("simulation_draw_count", dc$passed, list(selected = dc$selected,
       worst_ratio_at_selected = dc$worst_ratio_at_selected))
checks$draw_count <- list(selected = dc$selected)
syn_draws <- dc$selected

# 6-7. Test suites.
run_suite <- function(dir) {
  res <- as.data.frame(testthat::test_dir(dir, reporter = testthat::SilentReporter$new(),
                                          stop_on_failure = FALSE, stop_on_warning = FALSE))
  list(tests = nrow(res), expectations = sum(res$nb), failed = sum(res$failed),
       errors = sum(res$error), skipped = sum(res$skipped))
}
cmp_suite <- run_suite(file.path(root, "analysis", "comparison", "tests", "testthat"))
record("comparison_test_suite", cmp_suite$failed == 0 && cmp_suite$errors == 0, cmp_suite)
pkg_suite <- run_suite(file.path(root, "tests", "testthat"))
record("package_test_suite", pkg_suite$failed == 0 && pkg_suite$errors == 0, pkg_suite)

# 8. Full-size synthetic run of every simulation method, one dataset per geometry.
syn_jobs <- list(
  list(phase = "test", cell_id = 5L, replication = 1L, multiplier = NULL,
       methods = CMP_SIMULATION_METHODS, draws = syn_draws),
  list(phase = "test", cell_id = 9L, replication = 1L, multiplier = NULL,
       methods = CMP_SIMULATION_METHODS, draws = syn_draws)
)
syn_dir <- file.path(out_dir, "synthetic")
run_syn <- function(job) {
  res <- cmp_conform_dataset(cmp_run_dataset(.cmp_engine, job))
  cmp_check_complete(res, .cmp_engine$geoms[[cmp_cell(.cmp_engine, job$cell_id)$geometry]],
                     job$methods, 5L)
  res
}
cl <- parallel::makePSOCKcluster(min(workers, length(syn_jobs)))
parallel::clusterExport(cl, c("root", "run_syn"))
invisible(parallel::clusterEvalQ(cl, {
  setwd(root)
  source(file.path("analysis", "comparison", "load.R"))
  .cmp_engine <- cmp_engine(root)
  NULL
}))
syn <- parallel::clusterApply(cl, syn_jobs, function(job) {
  tryCatch(run_syn(job), error = function(e) structure(conditionMessage(e), class = "cmp_error"))
})
parallel::stopCluster(cl)
syn_failed <- vapply(syn, inherits, logical(1), "cmp_error")
syn_details <- list(errors = as.character(unlist(syn[syn_failed])))
if (!any(syn_failed)) {
  for (i in seq_along(syn)) {
    stem <- sprintf("%s_cell_%02d", syn[[i]]$meta$geometry, syn[[i]]$meta$cell_id)
    for (name in names(CMP_SCHEMAS)) {
      if (name %in% c("fold_predictions", "respondent_scores", "tuning_history")) {
        cmp_write_parquet(syn[[i]][[name]], file.path(syn_dir, name, paste0(stem, ".parquet")))
      } else {
        cmp_write_csv(syn[[i]][[name]], file.path(syn_dir, name, paste0(stem, ".csv")))
      }
    }
  }
  session <- cmp_session_manifest(root, list(purpose = "gate_c1_synthetic_render"))
  cmp_write_yaml(session, file.path(syn_dir, "session_manifest.yml"))
  sections <- lapply(syn, function(r) {
    jl <- r$job_ledger
    rt <- stats::aggregate(elapsed_seconds ~ method, jl, sum)
    st <- stats::aggregate(status == "ok" ~ method, jl, mean)
    tab <- data.frame(method = rt$method, job_seconds = cmp_fmt(rt$elapsed_seconds, 1),
                      ok_fraction = cmp_fmt(st[[2]][match(rt$method, st$method)], 2))
    list(title = sprintf("%s geometry, cell %d (%s), one test-phase dataset",
                         r$meta$geometry, r$meta$cell_id, r$meta$cell_name),
         note = paste("Synthetic render for Gate C1. One dataset is not evidence about any",
                      "method; the table shows that every method ran at full size and how long",
                      "it took on one core."),
         table = tab)
  })
  cmp_render_report(file.path(out_dir, "synthetic-report.md"),
                    "Gate C1 synthetic simulation render",
                    paste("Every simulation method on one full-size test-phase dataset per geometry,",
                          "using the disjoint test seed stream. Rendered from schema-validated outputs."),
                    sections, session)
  syn_details$runtime_seconds <- lapply(syn, function(r) r$meta$runtime_seconds)
  syn_details$ledger_all_ok <- all(vapply(syn, function(r) all(r$job_ledger$status == "ok"), logical(1)))
}
record("synthetic_full_size_run_and_report", !any(syn_failed) && isTRUE(syn_details$ledger_all_ok),
       syn_details)

gate_checks <- !names(checks) %in% c("files", "draw_count") & vapply(checks, is.list, logical(1))
checks$all_passed <- all(vapply(checks[gate_checks],
                                function(x) isTRUE(x$passed), logical(1)))
cmp_write_yaml(checks, checks_path)
cat("Gate C1 checks", if (checks$all_passed) "ALL PASSED" else "FAILED", "\n")
