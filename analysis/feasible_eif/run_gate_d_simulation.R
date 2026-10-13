#!/usr/bin/env Rscript

script_path <- function() {
  flag <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (!length(flag)) {
    return(normalizePath("analysis/feasible_eif/run_gate_d_simulation.R"))
  }
  normalizePath(sub("^--file=", "", flag[[1L]]), mustWork = TRUE)
}

script <- script_path()
analysis_dir <- dirname(script)
repo_root <- normalizePath(file.path(analysis_dir, "..", ".."), mustWork = TRUE)
source(file.path(repo_root, "analysis", "covariance_headroom",
                 "calibration_solver.R"), local = TRUE)
source(file.path(analysis_dir, "feasible_estimator.R"), local = TRUE)
source(file.path(analysis_dir, "simulation_functions.R"), local = TRUE)

args <- commandArgs(trailingOnly = TRUE)
value_arg <- function(prefix, default = NULL) {
  hit <- grep(paste0("^", prefix, "="), args, value = TRUE)
  if (length(hit)) sub(paste0("^", prefix, "="), "", hit[[1L]]) else default
}
config_arg <- value_arg("--config", "gate-d-smoke.yml")
config_path <- if (grepl("^/", config_arg)) config_arg else
  file.path(analysis_dir, config_arg)
config <- read_gate_d_config(config_path)
cores <- as.integer(value_arg("--cores", "1"))
if (!is.finite(cores) || cores < 1L) stop("--cores must be positive.", call. = FALSE)
output_arg <- value_arg("--output-dir", NULL)
output_dir <- if (is.null(output_arg)) {
  file.path(repo_root, "results", "feasible-eif", config$mode)
} else if (grepl("^/", output_arg)) output_arg else file.path(repo_root, output_arg)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

geometry <- gate_d_geometry(config)
scenario_results <- list()
selector_results <- list()
failure_results <- list()
replications <- seq_len(config$replications_per_scenario)

for (scenario in config$simulation$scenarios) {
  message("Gate D scenario ", scenario$id, ": ", length(replications),
          " replications on ", cores, " core(s)...")
  worker <- function(replication) {
    tryCatch(
      run_gate_d_replication(scenario, replication, config, geometry),
      error = function(e) list(
        results = NULL, selector = NULL,
        error = data.frame(
          scenario = scenario$id,
          replication = replication,
          message = conditionMessage(e),
          stringsAsFactors = FALSE
        )
      )
    )
  }
  pieces <- if (cores > 1L && .Platform$OS.type != "windows") {
    parallel::mclapply(replications, worker, mc.cores = cores,
                       mc.preschedule = TRUE, mc.set.seed = FALSE)
  } else {
    lapply(replications, worker)
  }
  scenario_result <- do.call(rbind, lapply(pieces, `[[`, "results"))
  scenario_selector <- do.call(rbind, lapply(pieces, `[[`, "selector"))
  scenario_failure <- do.call(rbind, lapply(pieces, `[[`, "error"))
  scenario_results[[scenario$id]] <- scenario_result
  selector_results[[scenario$id]] <- scenario_selector
  failure_results[[scenario$id]] <- scenario_failure
  saveRDS(
    list(results = scenario_result, selector = scenario_selector,
         failures = scenario_failure, scenario = scenario, config = config),
    file.path(output_dir, paste0("scenario_", scenario$id, ".rds")),
    version = 3
  )
  message("  completed; failures: ", if (is.null(scenario_failure)) 0L else
            nrow(scenario_failure))
}

results <- do.call(rbind, scenario_results)
selector <- do.call(rbind, selector_results)
failures <- do.call(rbind, failure_results)
utils::write.csv(results, file.path(output_dir, "replication_results.csv"),
                 row.names = FALSE)
utils::write.csv(selector, file.path(output_dir, "selector_results.csv"),
                 row.names = FALSE)
if (!is.null(failures)) {
  utils::write.csv(failures, file.path(output_dir, "failures.csv"), row.names = FALSE)
}
saveRDS(
  list(results = results, selector = selector, failures = failures,
       config = config, geometry = geometry,
       config_md5 = unname(tools::md5sum(config$.path))),
  file.path(output_dir, "gate_d_simulation_results.rds"), version = 3
)
cat("Saved Gate-D ", config$mode, " results to ", output_dir, "\n", sep = "")
