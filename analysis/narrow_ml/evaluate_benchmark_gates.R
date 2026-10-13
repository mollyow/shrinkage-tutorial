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
  file.path(repo_root, "analysis", "narrow_ml", "benchmark-production.yml")
), mustWork = TRUE)
functions_path <- file.path(repo_root, "analysis", "narrow_ml", "benchmark_functions.R")
source(functions_path, local = globalenv())
config <- read_benchmark_config(config_path)
results_dir <- normalizePath(arg_value(
  "--results-dir",
  file.path(repo_root, "results", "narrow-ml", paste0("benchmark-", config$mode))
), mustWork = TRUE)
application_path <- normalizePath(arg_value(
  "--application-rds",
  file.path(repo_root, "results", "narrow-ml", "candidate_precision_screen.rds")
), mustWork = TRUE)

if (!requireNamespace("jsonlite", quietly = TRUE)) {
  stop("Package `jsonlite` is required.", call. = FALSE)
}

estimate_path <- file.path(results_dir, "estimates.csv")
family_path <- file.path(results_dir, "family_tests.csv")
failure_path <- file.path(results_dir, "failures.csv")
manifest_path <- file.path(results_dir, "run-manifest.json")
benchmark_assert(all(file.exists(c(
  estimate_path, family_path, failure_path, manifest_path
))), "The benchmark output is incomplete.")

estimates <- utils::read.csv(estimate_path, stringsAsFactors = FALSE)
families <- utils::read.csv(family_path, stringsAsFactors = FALSE)
failures <- utils::read.csv(failure_path, stringsAsFactors = FALSE)
manifest <- jsonlite::read_json(manifest_path, simplifyVector = TRUE)
benchmark_assert(nrow(failures) == 0L, "The benchmark contains failed jobs.")
benchmark_assert(identical(manifest$mode, config$mode),
                 "The run manifest and gate configuration modes differ.")
benchmark_assert(manifest$successful_jobs == length(config$scenarios) * config$replications,
                 "The run manifest is not complete.")

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
                 "The config, code, or candidate input changed after the run.")

models <- config$models
benchmark_assert(identical(models, c(
  "unadjusted", "additive_lm", "interacted_lm", "elastic_net", "ranger",
  "plugin_ranger", "oracle"
)), "The full gate requires the complete locked comparator set.")
design <- load_candidate_benchmark_design(repo_root)
estimands <- colnames(design$uniform_contrast)
expected_families <- unique(design$coordinate_meta$family)
benchmark_assert(length(estimands) == 22L &&
                   setequal(unique(estimates$estimand), estimands),
                 "The result does not contain the exact 22 locked coordinates.")
benchmark_assert(length(expected_families) == 4L &&
                   setequal(unique(estimates$family), expected_families),
                 "The result does not contain the exact four locked families.")
expected <- expand.grid(
  scenario = config$scenarios,
  replication = seq_len(config$replications),
  model = models,
  estimand = estimands,
  stringsAsFactors = FALSE
)
key <- function(data, columns) do.call(paste, c(data[columns], sep = "\u001f"))
observed_key <- key(estimates, c("scenario", "replication", "model", "estimand"))
expected_key <- key(expected, c("scenario", "replication", "model", "estimand"))
benchmark_assert(!anyDuplicated(observed_key), "Benchmark estimate keys are duplicated.")
benchmark_assert(setequal(observed_key, expected_key),
                 "Benchmark estimate keys are missing or unexpected.")

expected_seed <- mapply(
  benchmark_dataset_seed, estimates$scenario, estimates$replication,
  MoreArgs = list(config = config)
)
benchmark_assert(all(estimates$seed == expected_seed),
                 "At least one result has an unexpected dataset seed.")
expected_ratio <- ifelse(grepl("target_shift$", estimates$scenario), 1.5, 1)
benchmark_assert(max(abs(estimates$max_target_assignment_ratio - expected_ratio)) < 1e-12,
                 "The assignment/target leverage does not match the lock.")
benchmark_assert(all(is.finite(as.matrix(estimates[c(
  "truth", "estimate", "error", "std_error", "prediction_mse"
)]))), "Benchmark estimates contain non-finite values.")
benchmark_assert(all(estimates$std_error > 0),
                 "Every reported coordinate standard error must be positive.")
benchmark_assert(max(abs(estimates$error -
                           (estimates$estimate - estimates$truth))) < 1e-12,
                 "The saved estimation errors are algebraically inconsistent.")
benchmark_assert(all(estimates$covered_95 ==
                       (abs(estimates$error) <= 1.96 * estimates$std_error)),
                 "The saved pointwise-coverage indicators are inconsistent.")

expected_family_keys <- expand.grid(
  scenario = config$scenarios,
  replication = seq_len(config$replications),
  model = models,
  family = expected_families,
  stringsAsFactors = FALSE
)
observed_family_key <- key(families, c(
  "scenario", "replication", "model", "family"
))
expected_family_key <- key(expected_family_keys, c(
  "scenario", "replication", "model", "family"
))
benchmark_assert(!anyDuplicated(observed_family_key) &&
                   setequal(observed_family_key, expected_family_key),
                 "Family-result keys are duplicated, missing, or unexpected.")
benchmark_assert(all(families$covariance_rank == families$dimension),
                 "At least one family covariance is rank deficient.")
benchmark_assert(all(is.finite(families$truth_statistic)) &&
                   all(families$truth_covered_95 %in% c(TRUE, FALSE)),
                 "Family confidence-region results are incomplete.")

truth_spread <- aggregate(
  truth ~ scenario + estimand, estimates,
  function(x) max(x) - min(x)
)
benchmark_assert(max(truth_spread$truth) < 1e-12,
                 "Truth varies across models or replications.")

# Equal-family weighting prevents the 12-dimensional family from dominating
# the two one-dimensional families. Resampling is always by replication.
replication_family <- aggregate(
  cbind(squared_error = estimates$error^2,
        reported_variance = estimates$std_error^2,
        covered = as.numeric(estimates$covered_95),
        interval_width = 2 * 1.96 * estimates$std_error) ~
    scenario + replication + model + family,
  estimates,
  mean
)
replication_loss <- aggregate(
  cbind(squared_error, reported_variance, covered, interval_width) ~
    scenario + replication + model,
  replication_family,
  mean
)

coordinate_error <- aggregate(
  error ~ scenario + model + family + estimand, estimates, mean
)
names(coordinate_error)[names(coordinate_error) == "error"] <- "mean_error"
coordinate_variance <- aggregate(
  estimate ~ scenario + model + family + estimand, estimates, stats::var
)
names(coordinate_variance)[names(coordinate_variance) == "estimate"] <-
  "empirical_variance"
coordinate_reported <- aggregate(
  I(std_error^2) ~ scenario + model + family + estimand, estimates, mean
)
names(coordinate_reported)[names(coordinate_reported) == "I(std_error^2)"] <-
  "mean_reported_variance"
coordinate_long <- Reduce(function(x, y) {
  merge(x, y, by = c("scenario", "model", "family", "estimand"), sort = FALSE)
}, list(coordinate_error, coordinate_variance, coordinate_reported))
coordinate_long$squared_mean_error <- coordinate_long$mean_error^2

family_variance <- aggregate(
  cbind(empirical_variance, mean_reported_variance, squared_mean_error) ~
    scenario + model + family,
  coordinate_long, mean
)
family_variance$standardized_bias_norm <- sqrt(
  family_variance$squared_mean_error / family_variance$mean_reported_variance
)
scenario_variance <- aggregate(
  cbind(empirical_variance, mean_reported_variance) ~ scenario + model,
  family_variance,
  mean
)
scenario_variance$max_family_standardized_bias <- vapply(
  seq_len(nrow(scenario_variance)), function(i) {
    max(family_variance$standardized_bias_norm[
      family_variance$scenario == scenario_variance$scenario[[i]] &
        family_variance$model == scenario_variance$model[[i]]
    ])
  }, numeric(1)
)

scenario_summary <- aggregate(
  cbind(squared_error, reported_variance, covered, interval_width) ~
    scenario + model,
  replication_loss,
  mean
)
scenario_summary$coefficient_rmse <- sqrt(scenario_summary$squared_error)
scenario_summary$prediction_mse <- vapply(seq_len(nrow(scenario_summary)), function(i) {
  rows <- estimates$scenario == scenario_summary$scenario[[i]] &
    estimates$model == scenario_summary$model[[i]]
  mean(estimates$prediction_mse[rows])
}, numeric(1))
scenario_summary <- merge(
  scenario_summary,
  scenario_variance[c(
    "scenario", "model", "empirical_variance", "mean_reported_variance",
    "max_family_standardized_bias"
  )],
  by = c("scenario", "model"), all.x = TRUE, sort = FALSE
)

bootstrap_loss_ratio <- function(data, scenarios, numerator, denominator,
                                 draws, seed) {
  subset <- data[
    data$scenario %in% scenarios & data$model %in% c(numerator, denominator),
    c("scenario", "replication", "model", "squared_error"),
    drop = FALSE
  ]
  point_values <- vapply(scenarios, function(scenario) {
    local <- subset[subset$scenario == scenario, , drop = FALSE]
    c(
      numerator = mean(local$squared_error[local$model == numerator]),
      denominator = mean(local$squared_error[local$model == denominator])
    )
  }, numeric(2L))
  point <- mean(point_values["numerator", ]) /
    mean(point_values["denominator", ])
  set.seed(as.integer(seed))
  boot <- replicate(draws, {
    values <- vapply(scenarios, function(scenario) {
      local <- subset[subset$scenario == scenario, , drop = FALSE]
      reps <- sort(unique(local$replication))
      chosen <- sample(reps, length(reps), replace = TRUE)
      numerator_rows <- match(
        paste(chosen, numerator), paste(local$replication, local$model)
      )
      denominator_rows <- match(
        paste(chosen, denominator), paste(local$replication, local$model)
      )
      c(
        numerator = mean(local$squared_error[numerator_rows]),
        denominator = mean(local$squared_error[denominator_rows])
      )
    }, numeric(2L))
    mean(values["numerator", ]) / mean(values["denominator", ])
  })
  c(
    ratio = point,
    lower = unname(stats::quantile(boot, 0.025, names = FALSE)),
    upper = unname(stats::quantile(boot, 0.975, names = FALSE))
  )
}

coverage_bootstrap <- function(data, scenario, draws, seed) {
  local <- data[data$scenario == scenario & data$model == "ranger", , drop = FALSE]
  per_rep <- aggregate(
    as.numeric(truth_covered_95) ~ replication, local, mean
  )
  names(per_rep)[[2L]] <- "covered"
  set.seed(as.integer(seed))
  boot <- replicate(draws, {
    chosen <- sample(seq_len(nrow(per_rep)), nrow(per_rep), replace = TRUE)
    mean(per_rep$covered[chosen])
  })
  c(
    coverage = mean(per_rep$covered),
    lower = unname(stats::quantile(boot, 0.025, names = FALSE)),
    upper = unname(stats::quantile(boot, 0.975, names = FALSE))
  )
}

nonlinear_scenarios <- as.character(unlist(
  config$confirmatory_scenarios$nonlinear_benefit, use.names = FALSE
))
benefit_additive <- bootstrap_loss_ratio(
  replication_loss, nonlinear_scenarios, "ranger", "additive_lm",
  config$bootstrap_draws, config$seed + 7001L
)
benefit_interacted <- bootstrap_loss_ratio(
  replication_loss, nonlinear_scenarios, "ranger", "interacted_lm",
  config$bootstrap_draws, config$seed + 7002L
)
benefit_oracle <- bootstrap_loss_ratio(
  replication_loss, nonlinear_scenarios, "oracle", "additive_lm",
  config$bootstrap_draws, config$seed + 7003L
)
benefit_by_scenario <- do.call(rbind, lapply(seq_along(nonlinear_scenarios), function(i) {
  scenario <- nonlinear_scenarios[[i]]
  additive <- bootstrap_loss_ratio(
    replication_loss, scenario, "ranger", "additive_lm",
    config$bootstrap_draws, config$seed + 7200L + 3L * i
  )
  interacted <- bootstrap_loss_ratio(
    replication_loss, scenario, "ranger", "interacted_lm",
    config$bootstrap_draws, config$seed + 7201L + 3L * i
  )
  oracle <- bootstrap_loss_ratio(
    replication_loss, scenario, "oracle", "additive_lm",
    config$bootstrap_draws, config$seed + 7202L + 3L * i
  )
  data.frame(
    scenario = scenario,
    ranger_additive_ratio = additive[["ratio"]],
    ranger_additive_lower = additive[["lower"]],
    ranger_additive_upper = additive[["upper"]],
    ranger_interacted_ratio = interacted[["ratio"]],
    ranger_interacted_lower = interacted[["lower"]],
    ranger_interacted_upper = interacted[["upper"]],
    oracle_additive_ratio = oracle[["ratio"]],
    oracle_additive_lower = oracle[["lower"]],
    oracle_additive_upper = oracle[["upper"]],
    recovered_oracle_gain = (1 - additive[["ratio"]]) /
      pmax(1 - oracle[["ratio"]], .Machine$double.eps),
    stringsAsFactors = FALSE
  )
}))

coverage_scenarios <- as.character(unlist(
  config$confirmatory_scenarios$coverage, use.names = FALSE
))
coverage_rows <- do.call(rbind, lapply(seq_along(coverage_scenarios), function(i) {
  value <- coverage_bootstrap(
    families, coverage_scenarios[[i]], config$bootstrap_draws,
    config$seed + 7100L + i
  )
  data.frame(
    scenario = coverage_scenarios[[i]],
    coverage = value[["coverage"]],
    mc_lower = value[["lower"]],
    mc_upper = value[["upper"]],
    stringsAsFactors = FALSE
  )
}))
coverage_rows$max_family_standardized_bias <- vapply(coverage_rows$scenario, function(scenario) {
  scenario_variance$max_family_standardized_bias[
    scenario_variance$scenario == scenario & scenario_variance$model == "ranger"
  ]
}, numeric(1))

variance_ratio_for <- function(scenario, numerator = "ranger",
                               denominator = "additive_lm") {
  numerator_value <- scenario_variance$empirical_variance[
    scenario_variance$scenario == scenario & scenario_variance$model == numerator
  ]
  denominator_value <- scenario_variance$empirical_variance[
    scenario_variance$scenario == scenario & scenario_variance$model == denominator
  ]
  numerator_value / denominator_value
}
additive_scenarios <- as.character(unlist(
  config$confirmatory_scenarios$additive_safety, use.names = FALSE
))
additive_variance_rows <- data.frame(
  scenario = additive_scenarios,
  variance_ratio = vapply(additive_scenarios, variance_ratio_for, numeric(1)),
  stringsAsFactors = FALSE
)
aggregate_additive_variance_ratio <- mean(additive_variance_rows$variance_ratio)

family_summary <- aggregate(
  reject_05 ~ scenario + model + null_family,
  families,
  mean
)
names(family_summary)[names(family_summary) == "reject_05"] <- "rejection_rate"
null_size_rows <- families[
  families$model == "ranger" & as.logical(families$null_family), , drop = FALSE
]
benchmark_assert(nrow(null_size_rows) > 0L,
                 "No true-null ranger family tests are available for size.")
null_size_by_scenario <- aggregate(
  as.numeric(reject_05) ~ scenario, null_size_rows, mean
)
names(null_size_by_scenario)[[2L]] <- "rejection_rate"

application <- readRDS(application_path)
benchmark_assert(application$num_trees == config$forest$trees,
                 "The application forest does not use the locked tree count.")
application_overall <- application$overall_results
application_coordinates <- application$coordinate_results
application_model_metrics <- function(model, baseline = "additive_lm") {
  model_mse <- application$prediction_metrics$weighted_mse[
    application$prediction_metrics$model == model
  ]
  baseline_mse <- application$prediction_metrics$weighted_mse[
    application$prediction_metrics$model == baseline
  ]
  model_rows <- application_coordinates[application_coordinates$model == model, ]
  baseline_rows <- application_coordinates[
    application_coordinates$model == baseline,
  ]
  merged <- merge(
    model_rows[c("estimand", "family", "std_error")],
    baseline_rows[c("estimand", "std_error")],
    by = "estimand", suffixes = c("_model", "_baseline")
  )
  family_ratio <- aggregate(
    cbind(model_variance = merged$std_error_model^2,
          baseline_variance = merged$std_error_baseline^2) ~ family,
    merged,
    mean
  )
  family_ratio$ratio <- family_ratio$model_variance / family_ratio$baseline_variance
  c(
    prediction_mse_ratio = model_mse / baseline_mse,
    score_variance_ratio = mean(family_ratio$model_variance) /
      mean(family_ratio$baseline_variance),
    fraction_coordinates_improved = mean(
      merged$std_error_model^2 < merged$std_error_baseline^2
    ),
    max_family_variance_ratio = max(family_ratio$ratio)
  )
}
application_ranger <- application_model_metrics("ranger")
application_elastic <- application_model_metrics("elastic_net")

threshold <- config$thresholds
criteria <- c(
  oracle_headroom =
    benefit_oracle[["ratio"]] <= threshold$oracle_headroom_mse_ratio &&
    benefit_oracle[["upper"]] < threshold$oracle_headroom_upper &&
    all(benefit_by_scenario$oracle_additive_ratio <=
          threshold$oracle_headroom_mse_ratio) &&
    all(benefit_by_scenario$oracle_additive_upper < threshold$oracle_headroom_upper),
  nonlinear_benefit_vs_additive =
    benefit_additive[["ratio"]] <= threshold$nonlinear_mse_ratio_to_additive &&
    benefit_additive[["upper"]] < threshold$nonlinear_mse_ratio_upper_to_additive &&
    all(benefit_by_scenario$ranger_additive_ratio <=
          threshold$nonlinear_mse_ratio_to_additive) &&
    all(benefit_by_scenario$ranger_additive_upper <
          threshold$nonlinear_mse_ratio_upper_to_additive),
  nonlinear_benefit_vs_interacted =
    benefit_interacted[["ratio"]] <= threshold$nonlinear_mse_ratio_to_interacted &&
    benefit_interacted[["upper"]] < threshold$nonlinear_mse_ratio_upper_to_interacted &&
    all(benefit_by_scenario$ranger_interacted_ratio <=
          threshold$nonlinear_mse_ratio_to_interacted) &&
    all(benefit_by_scenario$ranger_interacted_upper <
          threshold$nonlinear_mse_ratio_upper_to_interacted),
  recovered_oracle_gain =
    all(benefit_by_scenario$recovered_oracle_gain >=
          threshold$minimum_oracle_gain_recovered),
  coverage = all(coverage_rows$coverage >= threshold$coverage_lower &
                   coverage_rows$coverage <= threshold$coverage_upper),
  bias = all(coverage_rows$max_family_standardized_bias <=
               threshold$rms_standardized_bias),
  family_test_size = all(
    null_size_by_scenario$rejection_rate >= threshold$family_size_lower &
      null_size_by_scenario$rejection_rate <= threshold$family_size_upper
  ),
  additive_safety_aggregate =
    aggregate_additive_variance_ratio <= threshold$additive_variance_ratio,
  additive_safety_each_cell =
    all(additive_variance_rows$variance_ratio <= threshold$additive_cell_variance_ratio),
  application_prediction =
    application_ranger[["prediction_mse_ratio"]] <=
      threshold$application_prediction_mse_ratio,
  application_variance =
    application_ranger[["score_variance_ratio"]] <=
      threshold$application_score_variance_ratio,
  legacy_coordinate_audit =
    application_ranger[["fraction_coordinates_improved"]] >=
      threshold$application_fraction_coordinates_improved,
  application_family_safety =
    application_ranger[["max_family_variance_ratio"]] <=
      threshold$application_max_family_variance_ratio
)

simulation_gate_passed <- all(criteria[c(
  "oracle_headroom", "nonlinear_benefit_vs_additive",
  "nonlinear_benefit_vs_interacted", "recovered_oracle_gain", "coverage",
  "bias", "family_test_size", "additive_safety_aggregate",
  "additive_safety_each_cell"
)])
application_gate_passed <- all(criteria[c(
  "application_prediction", "application_variance", "application_family_safety"
)])
full_paper_gate_passed <- simulation_gate_passed && application_gate_passed

utils::write.csv(scenario_summary, file.path(results_dir, "gate-scenario-summary.csv"),
                 row.names = FALSE)
utils::write.csv(coordinate_long, file.path(results_dir, "gate-coordinate-summary.csv"),
                 row.names = FALSE)
utils::write.csv(coverage_rows, file.path(results_dir, "gate-coverage.csv"),
                 row.names = FALSE)
utils::write.csv(additive_variance_rows,
                 file.path(results_dir, "gate-additive-safety.csv"), row.names = FALSE)
utils::write.csv(family_summary, file.path(results_dir, "gate-family-tests.csv"),
                 row.names = FALSE)
utils::write.csv(null_size_by_scenario,
                 file.path(results_dir, "gate-family-size.csv"), row.names = FALSE)
utils::write.csv(benefit_by_scenario,
                 file.path(results_dir, "gate-benefit-by-scenario.csv"),
                 row.names = FALSE)

gate <- list(
  evaluated_at_utc = format(Sys.time(), tz = "UTC", usetz = TRUE),
  mode = config$mode,
  replications_per_scenario = config$replications,
  family_weighting = "equal weight for each of four prespecified families",
  monte_carlo_resampling_unit = "replication, stratified by scenario",
  nonlinear_benefit = list(
    scenarios = nonlinear_scenarios,
    ranger_vs_additive = as.list(benefit_additive),
    ranger_vs_interacted = as.list(benefit_interacted),
    oracle_vs_additive = as.list(benefit_oracle),
    by_scenario = benefit_by_scenario
  ),
  coverage = coverage_rows,
  null_family_size = null_size_by_scenario,
  additive_safety = list(
    aggregate_variance_ratio = aggregate_additive_variance_ratio,
    by_scenario = additive_variance_rows
  ),
  application = list(
    primary_ranger_vs_additive = as.list(application_ranger),
    secondary_elastic_net_vs_additive = as.list(application_elastic),
    note = paste(
      "The 70% coordinate criterion is retained only as a legacy audit because",
      "coordinate-win fractions depend on the chosen within-family basis."
    )
  ),
  criteria = as.list(criteria),
  simulation_gate_passed = simulation_gate_passed,
  application_gate_passed = application_gate_passed,
  full_paper_gate_passed = full_paper_gate_passed,
  evaluator_md5 = unname(tools::md5sum(file.path(
    repo_root, "analysis", "narrow_ml", "evaluate_benchmark_gates.R"
  ))),
  input_results_md5 = as.list(tools::md5sum(c(
    estimate_path, family_path, failure_path, manifest_path, application_path
  )))
)
jsonlite::write_json(
  gate, file.path(results_dir, "gate-status.json"),
  auto_unbox = TRUE, pretty = TRUE, digits = 12, null = "null"
)

format_number <- function(x, digits = 3L) formatC(x, digits = digits, format = "f")
lines <- c(
  paste0("# Narrow-ML gate report (", config$mode, ")"),
  "",
  paste0("- Simulation gate: **", if (simulation_gate_passed) "PASS" else "FAIL", "**"),
  paste0("- Application gate: **", if (application_gate_passed) "PASS" else "FAIL", "**"),
  paste0("- Full narrow-paper gate: **", if (full_paper_gate_passed) "PASS" else "FAIL", "**"),
  "",
  "## Primary nonlinear benefit",
  "",
  paste0(
    "- Ranger/additive family-balanced MSE ratio: ",
    format_number(benefit_additive[["ratio"]]), " (bootstrap 95% CI ",
    format_number(benefit_additive[["lower"]]), "–",
    format_number(benefit_additive[["upper"]]), ")."
  ),
  paste0(
    "- Ranger/interacted-linear family-balanced MSE ratio: ",
    format_number(benefit_interacted[["ratio"]]), " (bootstrap 95% CI ",
    format_number(benefit_interacted[["lower"]]), "–",
    format_number(benefit_interacted[["upper"]]), ")."
  ),
  "",
  "## Machine-checked criteria",
  "",
  "| Criterion | Result |",
  "|---|---:|",
  vapply(names(criteria), function(name) {
    paste0("| ", name, " | ", if (criteria[[name]]) "PASS" else "FAIL", " |")
  }, character(1)),
  "",
  paste0("The evaluator used ", config$replications,
         " independent replications per scenario and resampled whole replications."),
  "Coordinate losses were averaged within family and then equally across the four families."
)
writeLines(lines, file.path(results_dir, "gate-report.md"), useBytes = TRUE)
cat(paste(lines, collapse = "\n"), "\n")
