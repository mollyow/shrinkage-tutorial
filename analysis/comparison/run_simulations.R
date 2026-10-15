#!/usr/bin/env Rscript
# Run one simulation phase of comparison v1.
#
#   Rscript analysis/comparison/run_simulations.R --phase=calibration --stage=1 --workers=6
#   Rscript analysis/comparison/run_simulations.R --phase=calibration --stage=2 --workers=6
#   Rscript analysis/comparison/run_simulations.R --phase=smoke --workers=6
#   Rscript analysis/comparison/run_simulations.R --phase=production --workers=6
#
# Each phase is refused until the gates before it pass (C1 for calibration,
# Gate D for smoke, C2 as well for production). Every dataset is checkpointed
# atomically, so an interrupted run resumes where it stopped.

args <- commandArgs(trailingOnly = TRUE)
arg <- function(name, default = NULL) {
  hit <- grep(paste0("^--", name, "="), args, value = TRUE)
  if (length(hit)) sub(paste0("^--", name, "="), "", hit[[1]]) else default
}
phase <- arg("phase")
stage <- as.integer(arg("stage", "1"))
workers <- as.integer(arg("workers", "1"))
if (is.null(phase) || !phase %in% c("calibration", "smoke", "production")) {
  stop("Use --phase=calibration|smoke|production.", call. = FALSE)
}

source(file.path("analysis", "comparison", "load.R"))
root <- getwd()
cmp_require_phase(root, phase)
contracts <- cmp_load_contracts(root)
sim <- contracts$simulation
draws <- as.integer(sim$choice_geometry$target_integration$draws)
out_root <- file.path(root, "results", "comparison-v1", phase)
stage_used <- if (phase == "calibration") stage else 0L

jobs <- switch(
  phase,
  calibration = {
    selected <- NULL
    if (stage == 2L) {
      sel <- yaml::read_yaml(file.path(out_root, "stage-1-selection.yml"))
      selected <- sel$selected_multiplier
    }
    cmp_calibration_registry(sim, stage, selected, draws)
  },
  smoke = ,
  production = {
    gd <- yaml::read_yaml(file.path(root, CMP_GATE_PATHS$gate_d))
    cmp_phase_registry(sim, phase, CMP_SIMULATION_METHODS, draws, gd$selected_multiplier)
  }
)
jobs <- lapply(jobs, function(j) {
  j$path <- cmp_job_path(out_root, phase, stage_used, j)
  j
})

# A candidate whose generator is invalid is recorded, never run, and never
# selectable; the multiplier grid is not extended.
probe <- cmp_engine(root)
filtered <- cmp_filter_valid(probe, jobs)
jobs <- filtered$jobs
if (phase == "calibration" && stage == 1L) {
  cmp_write_yaml(list(invalid = filtered$invalid),
                 file.path(out_root, "stage-1-invalid-candidates.yml"))
}
rm(probe)

pending <- Filter(function(j) !file.exists(j$path), jobs)
cat(sprintf("%s stage %d: %d jobs, %d pending, %d workers, draws %d\n",
            phase, stage_used, length(jobs), length(pending), workers, draws))

if (length(pending)) {
  if (workers > 1L) {
    cl <- parallel::makePSOCKcluster(workers)
    on.exit(parallel::stopCluster(cl), add = TRUE)
    parallel::clusterExport(cl, "root")
    invisible(parallel::clusterEvalQ(cl, {
      setwd(root)
      source(file.path("analysis", "comparison", "load.R"))
      .cmp_engine <- cmp_engine(root)
      NULL
    }))
    status <- parallel::clusterApplyLB(cl, pending, function(job) cmp_run_job(.cmp_engine, job))
  } else {
    engine <- cmp_engine(root)
    status <- lapply(pending, function(job) cmp_run_job(engine, job))
  }
  errors <- grep("^ERROR", unlist(status), value = TRUE)
  cat(sprintf("Completed %d jobs; %d dataset-level errors.\n", length(status), length(errors)))
  if (length(errors)) {
    writeLines(errors, file.path(out_root, sprintf("stage-%d-dataset-errors.txt", stage_used)))
    stop("Dataset-level errors are hard failures; see the error file.", call. = FALSE)
  }
}

collect_dir <- file.path(out_root, sprintf("stage-%d-collected", stage_used))
cmp_collect(jobs, collect_dir)
cmp_write_yaml(cmp_session_manifest(root, list(phase = phase, stage = stage_used,
                                               jobs = length(jobs), draws = draws,
                                               workers = workers)),
               file.path(collect_dir, "session_manifest.yml"))
cat("Collected outputs in", collect_dir, "\n")
