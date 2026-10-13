#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
arg_value <- function(prefix, default = NULL) {
  hit <- grep(paste0("^", prefix, "="), args, value = TRUE)
  if (!length(hit)) return(default)
  sub(paste0("^", prefix, "="), "", hit[[1L]])
}

repo_root <- normalizePath(getwd(), mustWork = TRUE)
config_path <- arg_value(
  "--config",
  file.path(repo_root, "analysis", "narrow_ml", "benchmark-production.yml")
)
config_path <- normalizePath(config_path, mustWork = TRUE)
functions_path <- file.path(
  repo_root, "analysis", "narrow_ml", "benchmark_functions.R"
)
runner_path <- file.path(
  repo_root, "analysis", "narrow_ml", "run_candidate_geometry_benchmark.R"
)
source(functions_path, local = globalenv())
config <- read_benchmark_config(config_path)

workers <- as.integer(arg_value("--workers", max(1L, min(6L, parallel::detectCores() - 1L))))
benchmark_assert(is.finite(workers) && workers >= 1L, "`--workers` must be positive.")
output_dir <- arg_value(
  "--output-dir",
  file.path(repo_root, "results", "narrow-ml", paste0("benchmark-", config$mode))
)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
checkpoint_dir <- file.path(output_dir, "checkpoints")
dir.create(checkpoint_dir, recursive = TRUE, showWarnings = FALSE)

if (identical(config$mode, "production")) {
  benchmark_assert(
    identical(config$status, "locked_before_benchmark") &&
      is.character(config$locked_on) && nzchar(config$locked_on),
    "Production execution requires a dated locked_before_benchmark configuration."
  )
}

required_packages <- c("readr", "yaml")
if ("ranger" %in% config$models) required_packages <- c(required_packages, "ranger")
if ("elastic_net" %in% config$models) required_packages <- c(required_packages, "glmnet")
missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1L), quietly = TRUE)
]
benchmark_assert(
  !length(missing_packages),
  paste("Install required packages:", paste(missing_packages, collapse = ", "))
)

source_files <- c(config_path, functions_path, runner_path)
source_md5 <- unname(tools::md5sum(source_files))
names(source_md5) <- basename(source_files)
input_files <- c(
  file.path(repo_root, "analysis", "candidate", "derived", "candidate_prepared.rds"),
  file.path(repo_root, "Candidate recruitment", "replication", "PSRM_replication",
            "data", "CleanedData.csv")
)
input_md5 <- unname(tools::md5sum(input_files))
names(input_md5) <- basename(input_files)
run_fingerprint <- paste(c(source_md5, input_md5), collapse = "-")

design <- load_candidate_benchmark_design(repo_root)
message("Calibrating scenario-specific rating means...")
design <- prepare_scenario_calibration(design, config)
jobs <- expand.grid(
  scenario = config$scenarios,
  replication = seq_len(config$replications),
  stringsAsFactors = FALSE
)

checkpoint_path <- function(scenario, replication) {
  file.path(
    checkpoint_dir,
    sprintf("%s-rep-%04d.rds", scenario, as.integer(replication))
  )
}

read_checkpoint <- function(path) {
  if (!file.exists(path)) return(NULL)
  object <- readRDS(path)
  benchmark_assert(
    identical(object$run_fingerprint, run_fingerprint),
    paste("Stale checkpoint has a different code/config fingerprint:", path)
  )
  object$result
}

write_checkpoint <- function(result, path) {
  temporary <- paste0(path, ".tmp-", Sys.getpid())
  saveRDS(
    list(run_fingerprint = run_fingerprint, result = result),
    temporary, version = 3
  )
  benchmark_assert(file.rename(temporary, path),
                   paste("Could not atomically save checkpoint", path))
  invisible(path)
}

run_job <- function(index) {
  scenario <- jobs$scenario[[index]]
  replication <- jobs$replication[[index]]
  path <- checkpoint_path(scenario, replication)
  existing <- read_checkpoint(path)
  if (!is.null(existing)) {
    message("resume ", scenario, " replication ", replication)
    return(existing)
  }
  message("run ", scenario, " replication ", replication)
  result <- tryCatch(
    run_benchmark_replication(design, scenario, replication, config),
    error = function(error) list(
      estimates = NULL,
      families = NULL,
      details = NULL,
      runtime_seconds = NA_real_,
      truth = NULL,
      scenario = scenario,
      replication = replication,
      seed = benchmark_dataset_seed(scenario, replication, config),
      error = conditionMessage(error)
    )
  )
  result$error <- result$error %||% NA_character_
  write_checkpoint(result, path)
  result
}

results <- if (workers > 1L && .Platform$OS.type != "windows") {
  parallel::mclapply(
    seq_len(nrow(jobs)), run_job, mc.cores = workers,
    mc.preschedule = FALSE, mc.set.seed = FALSE
  )
} else {
  lapply(seq_len(nrow(jobs)), run_job)
}

failed <- vapply(results, function(x) {
  is.null(x$estimates) || (length(x$error) && !is.na(x$error))
}, logical(1L))
failure_rows <- do.call(rbind, lapply(results[failed], function(x) {
  data.frame(
    scenario = x$scenario,
    replication = x$replication,
    seed = x$seed,
    error = x$error %||% "missing results",
    stringsAsFactors = FALSE
  )
}))
if (is.null(failure_rows)) {
  failure_rows <- data.frame(
    scenario = character(), replication = integer(), seed = integer(),
    error = character(), stringsAsFactors = FALSE
  )
}
utils::write.csv(failure_rows, file.path(output_dir, "failures.csv"), row.names = FALSE)

successful <- results[!failed]
if (length(successful)) {
  estimates <- do.call(rbind, lapply(successful, `[[`, "estimates"))
  families <- do.call(rbind, lapply(successful, `[[`, "families"))
  runtimes <- do.call(rbind, lapply(successful, function(x) {
    data.frame(
      scenario = x$scenario,
      replication = x$replication,
      seed = x$seed,
      runtime_seconds = x$runtime_seconds,
      stringsAsFactors = FALSE
    )
  }))
  utils::write.csv(estimates, file.path(output_dir, "estimates.csv"), row.names = FALSE)
  utils::write.csv(families, file.path(output_dir, "family_tests.csv"), row.names = FALSE)
  utils::write.csv(runtimes, file.path(output_dir, "runtimes.csv"), row.names = FALSE)

  if (!any(failed)) {
    expected_estimate_keys <- expand.grid(
    scenario = config$scenarios,
    replication = seq_len(config$replications),
    model = config$models,
    estimand = colnames(design$uniform_contrast),
    stringsAsFactors = FALSE
  )
    observed_key <- do.call(paste, c(estimates[c(
      "scenario", "replication", "model", "estimand"
    )], sep = "\u001f"))
    expected_key <- do.call(paste, c(expected_estimate_keys, sep = "\u001f"))
    benchmark_assert(!anyDuplicated(observed_key),
                     "The reduced estimates contain duplicated result keys.")
    benchmark_assert(setequal(observed_key, expected_key),
                     "The reduced estimates do not match the complete expected key set.")
  }
}

git_head <- tryCatch(
  system2("git", c("rev-parse", "HEAD"), stdout = TRUE, stderr = FALSE),
  error = function(error) NA_character_
)
git_status <- tryCatch(
  system2("git", c("status", "--short"), stdout = TRUE, stderr = FALSE),
  error = function(error) NA_character_
)

manifest <- list(
  created_at_utc = format(Sys.time(), tz = "UTC", usetz = TRUE),
  mode = config$mode,
  status = config$status,
  config = normalizePath(config_path),
  source_md5 = as.list(source_md5),
  input_md5 = as.list(input_md5),
  run_fingerprint = run_fingerprint,
  git_head = paste(git_head, collapse = ""),
  git_dirty = length(git_status) > 0L,
  git_status = as.list(git_status),
  renv_lock_md5 = if (file.exists(file.path(repo_root, "renv.lock"))) {
    unname(tools::md5sum(file.path(repo_root, "renv.lock")))
  } else NA_character_,
  requested_jobs = nrow(jobs),
  successful_jobs = length(successful),
  failed_jobs = sum(failed),
  workers = workers,
  r_version = R.version.string,
  package_versions = as.list(vapply(required_packages, function(package) {
    as.character(utils::packageVersion(package))
  }, character(1L))),
  design = list(
    profiles = design$profile_count,
    coordinates = ncol(design$uniform_contrast),
    families = length(unique(design$coordinate_meta$family)),
    context_strata = nrow(design$contexts),
    profile_sampling = config$profile_sampling,
    marginal_assignment_probability = design$assignment_probability,
    calibration = design$calibration
  )
)
if (requireNamespace("jsonlite", quietly = TRUE)) {
  jsonlite::write_json(
    manifest, file.path(output_dir, "run-manifest.json"),
    auto_unbox = TRUE, pretty = TRUE, null = "null"
  )
} else {
  dput(manifest, file = file.path(output_dir, "run-manifest.R"))
}

message(
  "Completed ", length(successful), "/", nrow(jobs), " jobs; results in ",
  normalizePath(output_dir, mustWork = TRUE)
)
if (any(failed)) quit(save = "no", status = 1L)
