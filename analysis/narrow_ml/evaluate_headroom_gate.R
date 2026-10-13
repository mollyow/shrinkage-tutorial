#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
arg_value <- function(prefix, default = NULL) {
  hit <- grep(paste0("^", prefix, "="), args, value = TRUE)
  if (!length(hit)) return(default)
  sub(paste0("^", prefix, "="), "", hit[[1L]])
}

repo_root <- normalizePath(getwd(), mustWork = TRUE)
config_path <- normalizePath(arg_value(
  "--config",
  file.path(repo_root, "analysis", "narrow_ml", "benchmark-headroom.yml")
), mustWork = TRUE)
functions_path <- file.path(repo_root, "analysis", "narrow_ml", "benchmark_functions.R")
source(functions_path, local = globalenv())
config <- read_benchmark_config(config_path)
benchmark_assert(identical(config$models, c("additive_lm", "oracle")),
                 "The headroom gate must compare only additive adjustment and the oracle.")
results_dir <- normalizePath(arg_value(
  "--results-dir",
  file.path(repo_root, "results", "narrow-ml", "benchmark-headroom")
), mustWork = TRUE)

estimate_path <- file.path(results_dir, "estimates.csv")
failure_path <- file.path(results_dir, "failures.csv")
manifest_path <- file.path(results_dir, "run-manifest.json")
benchmark_assert(all(file.exists(c(estimate_path, failure_path, manifest_path))),
                 "The headroom run output is incomplete.")
estimates <- utils::read.csv(estimate_path, stringsAsFactors = FALSE)
failures <- utils::read.csv(failure_path, stringsAsFactors = FALSE)
manifest <- jsonlite::read_json(manifest_path, simplifyVector = TRUE)
benchmark_assert(nrow(failures) == 0L, "At least one headroom job failed.")
benchmark_assert(identical(manifest$mode, config$mode) &&
                   manifest$successful_jobs ==
                     length(config$scenarios) * config$replications,
                 "The headroom run manifest is incomplete or from another mode.")

runner_path <- file.path(
  repo_root, "analysis", "narrow_ml", "run_candidate_geometry_benchmark.R"
)
bundle_path <- file.path(
  repo_root, "analysis", "candidate", "derived", "candidate_prepared.rds"
)
raw_path <- file.path(
  repo_root, "Candidate recruitment", "replication", "PSRM_replication",
  "data", "CleanedData.csv"
)
current_fingerprint <- paste(unname(tools::md5sum(c(
  config_path, functions_path, runner_path, bundle_path, raw_path
))), collapse = "-")
benchmark_assert(identical(manifest$run_fingerprint, current_fingerprint),
                 "The config, code, or candidate input changed after the headroom run.")

design <- load_candidate_benchmark_design(repo_root)
expected_estimands <- colnames(design$uniform_contrast)
benchmark_assert(setequal(unique(estimates$estimand), expected_estimands) &&
                   length(unique(estimates$estimand)) == 22L,
                 "The headroom run does not contain the exact 22 locked coordinates.")
benchmark_assert(setequal(unique(estimates$family),
                         unique(design$coordinate_meta$family)),
                 "The headroom run does not contain the four locked families.")
expected <- expand.grid(
  scenario = config$scenarios,
  replication = seq_len(config$replications),
  model = config$models,
  estimand = expected_estimands,
  stringsAsFactors = FALSE
)
key <- function(data) do.call(paste, c(data[c(
  "scenario", "replication", "model", "estimand"
)], sep = "\u001f"))
benchmark_assert(!anyDuplicated(key(estimates)) && setequal(key(estimates), key(expected)),
                 "Headroom result keys are duplicated, missing, or unexpected.")
expected_seed <- mapply(
  benchmark_dataset_seed, estimates$scenario, estimates$replication,
  MoreArgs = list(config = config)
)
benchmark_assert(all(estimates$seed == expected_seed), "A headroom seed changed.")
benchmark_assert(max(abs(estimates$error - (estimates$estimate - estimates$truth))) < 1e-12,
                 "The saved error column is inconsistent.")
benchmark_assert(all(is.finite(as.matrix(estimates[c(
  "truth", "estimate", "error", "std_error", "prediction_mse"
)]))) && all(estimates$std_error > 0),
                 "Headroom estimates are non-finite or have nonpositive standard errors.")
expected_ratio <- ifelse(grepl("target_shift$", estimates$scenario), 1.5, 1)
benchmark_assert(max(abs(estimates$max_target_assignment_ratio - expected_ratio)) < 1e-12,
                 "The headroom target/assignment leverage changed.")

family_path <- file.path(results_dir, "family_tests.csv")
benchmark_assert(file.exists(family_path), "Headroom family results are missing.")
families <- utils::read.csv(family_path, stringsAsFactors = FALSE)
expected_family <- expand.grid(
  scenario = config$scenarios,
  replication = seq_len(config$replications),
  model = config$models,
  family = unique(design$coordinate_meta$family),
  stringsAsFactors = FALSE
)
family_key <- function(data) do.call(paste, c(data[c(
  "scenario", "replication", "model", "family"
)], sep = "\u001f"))
benchmark_assert(!anyDuplicated(family_key(families)) &&
                   setequal(family_key(families), family_key(expected_family)) &&
                   all(families$covariance_rank == families$dimension),
                 "Headroom family keys or covariance ranks are invalid.")

family_loss <- aggregate(
  squared_error ~ scenario + replication + model + family,
  transform(estimates, squared_error = error^2),
  mean
)
replication_loss <- aggregate(
  squared_error ~ scenario + replication + model,
  family_loss,
  mean
)

ratio_bootstrap <- function(scenarios, draws, seed) {
  point_parts <- vapply(scenarios, function(scenario) {
    local <- replication_loss[replication_loss$scenario == scenario, ]
    c(
      oracle = mean(local$squared_error[local$model == "oracle"]),
      additive = mean(local$squared_error[local$model == "additive_lm"])
    )
  }, numeric(2L))
  point <- mean(point_parts["oracle", ]) / mean(point_parts["additive", ])
  set.seed(as.integer(seed))
  boot <- replicate(draws, {
    parts <- vapply(scenarios, function(scenario) {
      local <- replication_loss[replication_loss$scenario == scenario, ]
      reps <- sort(unique(local$replication))
      selected <- sample(reps, length(reps), replace = TRUE)
      oracle_index <- match(
        paste(selected, "oracle"), paste(local$replication, local$model)
      )
      additive_index <- match(
        paste(selected, "additive_lm"), paste(local$replication, local$model)
      )
      c(
        oracle = mean(local$squared_error[oracle_index]),
        additive = mean(local$squared_error[additive_index])
      )
    }, numeric(2L))
    mean(parts["oracle", ]) / mean(parts["additive", ])
  })
  c(
    ratio = point,
    lower = unname(stats::quantile(boot, 0.025, names = FALSE)),
    upper = unname(stats::quantile(boot, 0.975, names = FALSE))
  )
}

pooled <- ratio_bootstrap(config$scenarios, config$bootstrap_draws,
                          config$seed + 8100L)
by_scenario <- do.call(rbind, lapply(seq_along(config$scenarios), function(i) {
  value <- ratio_bootstrap(config$scenarios[[i]], config$bootstrap_draws,
                           config$seed + 8200L + i)
  data.frame(
    scenario = config$scenarios[[i]],
    ratio = value[["ratio"]],
    lower = value[["lower"]],
    upper = value[["upper"]],
    stringsAsFactors = FALSE
  )
}))

threshold <- config$thresholds
headroom_gate_passed <-
  pooled[["ratio"]] <= threshold$oracle_headroom_mse_ratio &&
  pooled[["upper"]] < threshold$oracle_headroom_upper &&
  all(by_scenario$ratio <= threshold$oracle_headroom_mse_ratio) &&
  all(by_scenario$upper < threshold$oracle_headroom_upper)
scientific_futility <- pooled[["lower"]] >
  threshold$nonlinear_mse_ratio_to_additive

status <- list(
  evaluated_at_utc = format(Sys.time(), tz = "UTC", usetz = TRUE),
  mode = config$mode,
  replications_per_scenario = config$replications,
  weighting = "coordinate squared error averaged within family, then equally across families",
  resampling_unit = "replication, stratified by scenario",
  pooled_oracle_to_additive = as.list(pooled),
  by_scenario = by_scenario,
  thresholds = list(
    required_pooled_ratio = threshold$oracle_headroom_mse_ratio,
    required_pooled_upper = threshold$oracle_headroom_upper,
    downstream_ml_ratio = threshold$nonlinear_mse_ratio_to_additive
  ),
  headroom_gate_passed = headroom_gate_passed,
  scientific_futility = scientific_futility,
  evaluator_md5 = unname(tools::md5sum(file.path(
    repo_root, "analysis", "narrow_ml", "evaluate_headroom_gate.R"
  ))),
  input_md5 = as.list(tools::md5sum(c(
    estimate_path, family_path, failure_path, manifest_path, config_path,
    functions_path
  )))
)
jsonlite::write_json(
  status, file.path(results_dir, "headroom-gate-status.json"),
  auto_unbox = TRUE, pretty = TRUE, digits = 12
)
utils::write.csv(by_scenario, file.path(results_dir, "headroom-by-scenario.csv"),
                 row.names = FALSE)

fmt <- function(x) formatC(x, digits = 3, format = "f")
lines <- c(
  "# Oracle-headroom gate",
  "",
  paste0("- Gate: **", if (headroom_gate_passed) "PASS" else "FAIL", "**"),
  paste0("- Scientific futility for the 0.85 downstream ML target: **",
         if (scientific_futility) "YES" else "NO", "**"),
  paste0("- Pooled oracle/additive MSE ratio: ", fmt(pooled[["ratio"]]),
         " (bootstrap 95% CI ", fmt(pooled[["lower"]]), "–",
         fmt(pooled[["upper"]]), ")."),
  "",
  "The bootstrap resamples complete replications within scenario. Loss is",
  "averaged within each prespecified family and then equally across families."
)
writeLines(lines, file.path(results_dir, "headroom-gate-report.md"), useBytes = TRUE)
cat(paste(lines, collapse = "\n"), "\n")
