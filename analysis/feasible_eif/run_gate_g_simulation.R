#!/usr/bin/env Rscript

# Gate G proof-of-concept runner. This file is new; it sources but never
# modifies the frozen Gate E2 implementation.

script_path <- function() {
  flag <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (!length(flag)) {
    return(normalizePath(
      "analysis/feasible_eif/run_gate_g_simulation.R", mustWork = TRUE
    ))
  }
  normalizePath(sub("^--file=", "", flag[[1L]]), mustWork = TRUE)
}

script <- script_path()
analysis_dir <- dirname(script)
repo_root <- normalizePath(file.path(analysis_dir, "..", ".."),
                           mustWork = TRUE)

args <- commandArgs(trailingOnly = TRUE)
value_arg <- function(prefix, default = NULL) {
  hit <- grep(paste0("^", prefix, "="), args, value = TRUE)
  if (length(hit) > 1L) {
    stop("Argument supplied more than once: ", prefix, call. = FALSE)
  }
  if (length(hit)) sub(paste0("^", prefix, "="), "", hit[[1L]]) else default
}

config_arg <- value_arg("--config", "gate-g-screen.yml")
config_path <- if (grepl("^/", config_arg)) {
  config_arg
} else {
  file.path(analysis_dir, config_arg)
}
config_path <- normalizePath(config_path, mustWork = TRUE)

if (!requireNamespace("yaml", quietly = TRUE)) {
  stop("Package `yaml` is required.", call. = FALSE)
}

child_config <- yaml::read_yaml(config_path)
if (is.null(child_config$inherits)) {
  stop("Gate G requires a frozen inherited configuration.", call. = FALSE)
}
inherited_config_path <- normalizePath(
  file.path(dirname(config_path), child_config$inherits), mustWork = TRUE
)

source_paths <- c(
  calibration_solver = file.path(
    repo_root, "analysis", "covariance_headroom", "calibration_solver.R"
  ),
  feasible_estimator = file.path(analysis_dir, "feasible_estimator.R"),
  simulation_functions = file.path(analysis_dir, "simulation_functions.R"),
  sensitivity_functions = file.path(analysis_dir, "sensitivity_functions.R"),
  e2_simulation_functions = file.path(
    analysis_dir, "e2_simulation_functions.R"
  ),
  gate_g_corrections = file.path(analysis_dir, "gate_g_corrections.R"),
  gate_g_simulation_functions = file.path(
    analysis_dir, "gate_g_simulation_functions.R"
  ),
  gate_g_theory_checks = file.path(analysis_dir, "check_gate_g_theory.R"),
  runner = script,
  summarizer = file.path(analysis_dir, "summarize_gate_g.R")
)
source_paths <- vapply(source_paths, normalizePath, character(1),
                       mustWork = TRUE)
source_md5 <- unname(tools::md5sum(source_paths))
names(source_md5) <- names(source_paths)
config_paths <- c(
  gate_g = config_path,
  inherited_gate_e2 = inherited_config_path
)
config_md5 <- unname(tools::md5sum(config_paths))
names(config_md5) <- names(config_paths)

for (path in source_paths[c(
  "calibration_solver", "feasible_estimator", "simulation_functions",
  "sensitivity_functions", "e2_simulation_functions",
  "gate_g_corrections", "gate_g_simulation_functions"
)]) {
  source(path, local = TRUE)
}

config <- read_gate_g_config(config_path)
cores <- suppressWarnings(as.integer(value_arg("--cores", "1")))
if (length(cores) != 1L || !is.finite(cores) || cores < 1L) {
  stop("--cores must be a positive integer.", call. = FALSE)
}

planned_replications <- as.integer(config$replications_per_scenario)
replication_arg <- value_arg("--replications", NULL)
effective_replications <- if (is.null(replication_arg)) {
  planned_replications
} else {
  suppressWarnings(as.integer(replication_arg))
}
if (length(effective_replications) != 1L ||
    !is.finite(effective_replications) || effective_replications < 1L) {
  stop("--replications must be a positive integer.", call. = FALSE)
}
if (effective_replications > planned_replications) {
  stop("--replications cannot exceed the frozen planned count (",
       planned_replications, ").", call. = FALSE)
}

output_arg <- value_arg("--output-dir", NULL)
output_dir <- if (is.null(output_arg)) {
  file.path(repo_root, "results", "feasible-eif", "gate-g-screen")
} else if (grepl("^/", output_arg)) {
  output_arg
} else {
  file.path(repo_root, output_arg)
}
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
output_dir <- normalizePath(output_dir, mustWork = TRUE)

# The deterministic identities are a mandatory preflight under the protocol.
old_wd <- getwd()
theory_output <- tryCatch({
  setwd(repo_root)
  system2(
    file.path(R.home("bin"), "Rscript"),
    shQuote(source_paths[["gate_g_theory_checks"]]),
    stdout = TRUE, stderr = TRUE
  )
}, finally = setwd(old_wd))
theory_status <- attr(theory_output, "status")
if (is.null(theory_status)) theory_status <- 0L
writeLines(as.character(theory_output),
           file.path(output_dir, "gate_g_theory_checks.log"))
if (!identical(as.integer(theory_status), 0L)) {
  stop("Gate-G deterministic theory checks failed; see ",
       file.path(output_dir, "gate_g_theory_checks.log"), call. = FALSE)
}

geometry <- gate_d_geometry(config)
replications <- seq_len(effective_replications)
scenario_results <- list()
covariance_results <- list()
fold_results <- list()
support_diagnostic_results <- list()
split_diagnostic_results <- list()
failure_results <- list()

empty_failures <- function() {
  data.frame(
    scenario = character(), replication = integer(), seed = integer(),
    message = character(), stringsAsFactors = FALSE
  )
}

bind_piece <- function(pieces, name) {
  values <- lapply(pieces, `[[`, name)
  values <- values[!vapply(values, is.null, logical(1))]
  if (length(values)) do.call(rbind, values) else NULL
}

for (scenario in config$simulation$scenarios) {
  message("Gate G scenario ", scenario$id, ": ", length(replications),
          " replications on ", cores, " core(s)...")
  worker <- function(replication) {
    seed <- tryCatch(
      gate_e2_scenario_seed(scenario, replication, config),
      error = function(e) NA_integer_
    )
    tryCatch(
      run_gate_g_replication(scenario, replication, config, geometry),
      error = function(e) list(
        results = NULL,
        covariance = NULL,
        folds = NULL,
        support_diagnostics = NULL,
        split_diagnostics = NULL,
        error = data.frame(
          scenario = scenario$id,
          replication = replication,
          seed = seed,
          message = conditionMessage(e),
          stringsAsFactors = FALSE
        )
      )
    )
  }
  pieces <- if (cores > 1L && .Platform$OS.type != "windows") {
    parallel::mclapply(
      replications, worker, mc.cores = cores,
      mc.preschedule = TRUE, mc.set.seed = FALSE
    )
  } else {
    lapply(replications, worker)
  }

  scenario_result <- bind_piece(pieces, "results")
  scenario_covariance_rows <- bind_piece(pieces, "covariance")
  scenario_folds <- bind_piece(pieces, "folds")
  scenario_support_diagnostics <- bind_piece(pieces, "support_diagnostics")
  scenario_split_diagnostics <- bind_piece(pieces, "split_diagnostics")
  scenario_failures <- bind_piece(pieces, "error")
  if (!is.null(scenario_result)) {
    scenario_result$method <- gsub(
      "^oracle_tau_([0-9]+)\\.([0-9]+)_crossfit$",
      "oracle_tau_\\1_\\2_crossfit",
      scenario_result$method
    )
  }
  if (is.null(scenario_failures)) scenario_failures <- empty_failures()

  scenario_results[[scenario$id]] <- scenario_result
  covariance_results[[scenario$id]] <- scenario_covariance_rows
  fold_results[[scenario$id]] <- scenario_folds
  support_diagnostic_results[[scenario$id]] <- scenario_support_diagnostics
  split_diagnostic_results[[scenario$id]] <- scenario_split_diagnostics
  failure_results[[scenario$id]] <- scenario_failures

  saveRDS(
    list(
      results = scenario_result,
      covariance = scenario_covariance_rows,
      folds = scenario_folds,
      support_diagnostics = scenario_support_diagnostics,
      split_diagnostics = scenario_split_diagnostics,
      failures = scenario_failures,
      scenario = scenario,
      config = config,
      planned_replications = planned_replications,
      effective_replications = effective_replications,
      source_paths = source_paths,
      source_md5 = source_md5,
      config_paths = config_paths,
      config_md5 = config_md5,
      theory_checks_passed = TRUE
    ),
    file.path(output_dir, paste0("scenario_", scenario$id, ".rds")),
    version = 3
  )
  message("  completed; failures: ", nrow(scenario_failures))
}

bind_all <- function(values) {
  values <- values[!vapply(values, function(x) is.null(x) || !nrow(x),
                           logical(1))]
  if (length(values)) do.call(rbind, values) else NULL
}

results <- bind_all(scenario_results)
covariance <- bind_all(covariance_results)
folds <- bind_all(fold_results)
support_diagnostics <- bind_all(support_diagnostic_results)
split_diagnostics <- bind_all(split_diagnostic_results)
failures <- bind_all(failure_results)
if (is.null(failures)) failures <- empty_failures()

write_table <- function(data, filename) {
  if (!is.null(data)) {
    utils::write.csv(data, file.path(output_dir, filename), row.names = FALSE)
  }
}
write_table(results, "replication_results.csv")
write_table(covariance, "covariance_results.csv")
write_table(folds, "outer_fold_assignments.csv")
write_table(support_diagnostics, "outer_support_diagnostics.csv")
write_table(split_diagnostics, "split_gls_diagnostics.csv")
write_table(failures, "failures.csv")

hash_table <- function(paths, hashes) {
  data.frame(
    name = names(paths), path = unname(paths), md5 = unname(hashes),
    stringsAsFactors = FALSE
  )
}
utils::write.csv(
  hash_table(source_paths, source_md5),
  file.path(output_dir, "source_hashes.csv"), row.names = FALSE
)
utils::write.csv(
  hash_table(config_paths, config_md5),
  file.path(output_dir, "config_hashes.csv"), row.names = FALSE
)

run_metadata <- list(
  gate = "G_finite_sample_nuisance_remainder",
  mode = as.character(config$mode),
  started_from_frozen_config = identical(
    as.character(config$status), "frozen_before_fresh_gate_g_screen"
  ),
  completed_at_utc = format(Sys.time(), tz = "UTC", usetz = TRUE),
  planned_replications_per_scenario = planned_replications,
  effective_replications_per_scenario = effective_replications,
  replication_override = !is.null(replication_arg),
  scenario_count = length(config$simulation$scenarios),
  cores = cores,
  theory_checks_passed = TRUE,
  output_directory = output_dir,
  r_version = R.version.string,
  platform = R.version$platform
)
yaml::write_yaml(run_metadata, file.path(output_dir, "run_metadata.yml"))
utils::write.csv(
  data.frame(
    key = names(run_metadata),
    value = vapply(run_metadata, as.character, character(1)),
    stringsAsFactors = FALSE
  ),
  file.path(output_dir, "run_metadata.csv"), row.names = FALSE
)

bundle <- list(
  results = results,
  covariance = covariance,
  folds = folds,
  support_diagnostics = support_diagnostics,
  split_diagnostics = split_diagnostics,
  failures = failures,
  config = config,
  geometry = geometry,
  run_metadata = run_metadata,
  planned_replications = planned_replications,
  effective_replications = effective_replications,
  source_paths = source_paths,
  source_md5 = source_md5,
  config_paths = config_paths,
  config_md5 = config_md5,
  theory_checks_passed = TRUE
)
saveRDS(bundle, file.path(output_dir, "gate_g_simulation_results.rds"),
        version = 3)

cat("Saved Gate-G screen results to ", output_dir, "\n", sep = "")
