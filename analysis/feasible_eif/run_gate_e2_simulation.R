#!/usr/bin/env Rscript

script_path <- function() {
  flag <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (!length(flag)) {
    return(normalizePath("analysis/feasible_eif/run_gate_e2_simulation.R"))
  }
  normalizePath(sub("^--file=", "", flag[[1L]]), mustWork = TRUE)
}

script <- script_path()
analysis_dir <- dirname(script)
repo_root <- normalizePath(file.path(analysis_dir, "..", ".."), mustWork = TRUE)
source_paths <- c(
  calibration_solver = file.path(repo_root, "analysis", "covariance_headroom",
                                 "calibration_solver.R"),
  feasible_estimator = file.path(analysis_dir, "feasible_estimator.R"),
  simulation_functions = file.path(analysis_dir, "simulation_functions.R"),
  sensitivity_functions = file.path(analysis_dir, "sensitivity_functions.R"),
  e2_simulation_functions = file.path(analysis_dir,
                                      "e2_simulation_functions.R"),
  runner = script,
  summarizer = file.path(analysis_dir, "summarize_gate_e2.R")
)
source_paths <- vapply(source_paths, normalizePath, character(1),
                       mustWork = TRUE)
source_md5 <- unname(tools::md5sum(source_paths))
names(source_md5) <- names(source_paths)
for (path in source_paths[c(
  "calibration_solver", "feasible_estimator", "simulation_functions",
  "sensitivity_functions", "e2_simulation_functions"
)]) source(path, local = TRUE)

args <- commandArgs(trailingOnly = TRUE)
value_arg <- function(prefix, default = NULL) {
  hit <- grep(paste0("^", prefix, "="), args, value = TRUE)
  if (length(hit)) sub(paste0("^", prefix, "="), "", hit[[1L]]) else default
}
config_arg <- value_arg("--config", "gate-e2-smoke.yml")
config_path <- if (grepl("^/", config_arg)) config_arg else
  file.path(analysis_dir, config_arg)
config <- read_gate_e2_config(config_path)
cores <- as.integer(value_arg("--cores", "1"))
if (!is.finite(cores) || cores < 1L) stop("--cores must be positive.", call. = FALSE)
output_arg <- value_arg("--output-dir", NULL)
output_dir <- if (is.null(output_arg)) {
  file.path(repo_root, "results", "feasible-eif", paste0("gate-e2-", config$mode))
} else if (grepl("^/", output_arg)) output_arg else file.path(repo_root, output_arg)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

geometry <- gate_d_geometry(config)
scenario_results <- list()
diagnostic_results <- list()
selector_results <- list()
tau_results <- list()
departure_results <- list()
failure_results <- list()
replications <- seq_len(config$replications_per_scenario)

for (scenario in config$simulation$scenarios) {
  message("Gate E2 scenario ", scenario$id, ": ", length(replications),
          " replications on ", cores, " core(s)...")
  worker <- function(replication) {
    tryCatch(
      run_gate_e2_replication(scenario, replication, config, geometry),
      error = function(e) list(
        results = NULL, diagnostics = NULL, selector = NULL, tau = NULL,
        departure = NULL,
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
  bind_piece <- function(name) {
    values <- lapply(pieces, `[[`, name)
    values <- values[!vapply(values, is.null, logical(1))]
    if (length(values)) do.call(rbind, values) else NULL
  }
  scenario_result <- bind_piece("results")
  scenario_diagnostic <- bind_piece("diagnostics")
  scenario_selector <- bind_piece("selector")
  scenario_tau <- bind_piece("tau")
  scenario_departure <- bind_piece("departure")
  scenario_failure <- bind_piece("error")
  scenario_results[[scenario$id]] <- scenario_result
  diagnostic_results[[scenario$id]] <- scenario_diagnostic
  selector_results[[scenario$id]] <- scenario_selector
  tau_results[[scenario$id]] <- scenario_tau
  departure_results[[scenario$id]] <- scenario_departure
  failure_results[[scenario$id]] <- scenario_failure
  saveRDS(
    list(
      results = scenario_result, diagnostics = scenario_diagnostic,
      selector = scenario_selector, tau = scenario_tau,
      departure = scenario_departure, failures = scenario_failure,
      scenario = scenario, config = config, source_md5 = source_md5
    ),
    file.path(output_dir, paste0("scenario_", scenario$id, ".rds")),
    version = 3
  )
  message("  completed; failures: ", if (is.null(scenario_failure)) 0L else
            nrow(scenario_failure))
}

bind_all <- function(x) {
  x <- x[!vapply(x, is.null, logical(1))]
  if (length(x)) do.call(rbind, x) else NULL
}
results <- bind_all(scenario_results)
diagnostics <- bind_all(diagnostic_results)
selector <- bind_all(selector_results)
tau <- bind_all(tau_results)
departure <- bind_all(departure_results)
failures <- bind_all(failure_results)
if (!is.null(results)) {
  utils::write.csv(results, file.path(output_dir, "replication_results.csv"),
                   row.names = FALSE)
}
if (!is.null(diagnostics)) {
  utils::write.csv(diagnostics, file.path(output_dir, "diagnostic_results.csv"),
                   row.names = FALSE)
}
if (!is.null(selector)) {
  utils::write.csv(selector, file.path(output_dir, "selector_results.csv"),
                   row.names = FALSE)
}
if (!is.null(tau)) {
  utils::write.csv(tau, file.path(output_dir, "tau_results.csv"),
                   row.names = FALSE)
}
if (!is.null(departure)) {
  utils::write.csv(departure, file.path(output_dir, "departure_results.csv"),
                   row.names = FALSE)
}
if (!is.null(failures)) {
  utils::write.csv(failures, file.path(output_dir, "failures.csv"),
                   row.names = FALSE)
}
saveRDS(
  list(
    results = results, diagnostics = diagnostics, selector = selector,
    tau = tau, departure = departure, failures = failures,
    config = config, geometry = geometry,
    config_md5 = unname(tools::md5sum(config$.path)),
    source_md5 = source_md5
  ),
  file.path(output_dir, "gate_e2_simulation_results.rds"), version = 3
)
cat("Saved Gate-E2 ", config$mode, " results to ", output_dir, "\n", sep = "")
