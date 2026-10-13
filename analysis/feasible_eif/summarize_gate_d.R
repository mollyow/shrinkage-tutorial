#!/usr/bin/env Rscript

script_path <- function() {
  flag <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (!length(flag)) {
    return(normalizePath("analysis/feasible_eif/summarize_gate_d.R"))
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
mode <- value_arg("--mode", "smoke")
result_dir <- file.path(repo_root, "results", "feasible-eif", mode)
result_path <- file.path(result_dir, "gate_d_simulation_results.rds")
if (!file.exists(result_path)) stop("Gate-D simulation results are missing.", call. = FALSE)
bundle <- readRDS(result_path)
config <- bundle$config
results <- as.data.frame(bundle$results)
selector <- as.data.frame(bundle$selector)
failures <- bundle$failures
rules <- config$pass_rules
production <- identical(config$mode, "production")

methods_for_validity <- c("ordinary", "selected")
coordinate_summary <- do.call(rbind, lapply(
  split(results, interaction(results$scenario, results$method,
                             results$coordinate, drop = TRUE)),
  function(x) {
    mc_variance <- stats::var(x$estimate)
    bias <- mean(x$estimate - x$truth)
    data.frame(
      scenario = x$scenario[[1L]],
      scenario_class = x$scenario_class[[1L]],
      method = x$method[[1L]],
      coordinate = x$coordinate[[1L]],
      replications = nrow(x),
      bias = bias,
      mc_variance = mc_variance,
      mean_sandwich_variance = mean(x$standard_error^2),
      sandwich_variance_calibration = mean(x$standard_error^2) / mc_variance,
      standardized_bias = abs(bias) / sqrt(mc_variance),
      coverage = mean(x$covered),
      stringsAsFactors = FALSE
    )
  }
))
rownames(coordinate_summary) <- NULL

method_summary <- do.call(rbind, lapply(
  split(coordinate_summary,
        interaction(coordinate_summary$scenario, coordinate_summary$method,
                    drop = TRUE)),
  function(x) data.frame(
    scenario = x$scenario[[1L]],
    scenario_class = x$scenario_class[[1L]],
    method = x$method[[1L]],
    trace_variance_per_coordinate = mean(x$mc_variance),
    rms_standardized_bias = sqrt(mean(x$standardized_bias^2)),
    minimum_variance_calibration = min(x$sandwich_variance_calibration),
    maximum_variance_calibration = max(x$sandwich_variance_calibration),
    minimum_coordinate_coverage = min(x$coverage),
    maximum_coordinate_coverage = max(x$coverage),
    stringsAsFactors = FALSE
  )
))

# Pooled coordinate coverage is computed from the underlying binary events, not
# by averaging rounded coordinate summaries.
pooled_coverage <- aggregate(
  covered ~ scenario + scenario_class + method,
  data = results[results$method %in% methods_for_validity, , drop = FALSE],
  FUN = mean
)
names(pooled_coverage)[names(pooled_coverage) == "covered"] <- "pooled_coverage"
method_summary <- merge(method_summary, pooled_coverage,
                        by = c("scenario", "scenario_class", "method"),
                        all.x = TRUE, sort = FALSE)

scenario_ids <- vapply(config$simulation$scenarios, `[[`, character(1), "id")
bootstrap_draws <- as.integer(config$monte_carlo_bootstrap_draws)
set.seed(as.integer(config$monte_carlo_bootstrap_seed))
ratio_rows <- list()
bootstrap_store <- list()

matrix_cache <- new.env(parent = emptyenv())
scenario_matrix <- function(scenario, method) {
  key <- paste(scenario, method, sep = "::")
  if (exists(key, envir = matrix_cache, inherits = FALSE)) {
    return(get(key, envir = matrix_cache, inherits = FALSE))
  }
  x <- results[results$scenario == scenario & results$method == method,
               c("replication", "coordinate", "estimate"), drop = FALSE]
  wide <- reshape(x, idvar = "replication", timevar = "coordinate",
                  direction = "wide")
  out <- list(
    replication = wide$replication,
    value = as.matrix(wide[, setdiff(names(wide), "replication"), drop = FALSE])
  )
  assign(key, out, envir = matrix_cache)
  out
}

trace_variance <- function(x) {
  centered <- sweep(x, 2L, colMeans(x), "-")
  sum(centered^2) / ((nrow(x) - 1L) * ncol(x))
}

for (scenario in scenario_ids) {
  ordinary <- scenario_matrix(scenario, "ordinary")
  selected <- scenario_matrix(scenario, "selected")
  oracle <- scenario_matrix(scenario, "oracle")
  stopifnot(identical(ordinary$replication, selected$replication),
            identical(ordinary$replication, oracle$replication))
  v0 <- trace_variance(ordinary$value)
  vs <- trace_variance(selected$value)
  vo <- trace_variance(oracle$value)
  ratio <- vs / v0
  oracle_ratio <- vo / v0
  gain_defined <- abs(1 - oracle_ratio) > 1e-8
  gain <- if (gain_defined) (1 - ratio) / (1 - oracle_ratio) else NA_real_
  boot_ratio <- numeric(bootstrap_draws)
  boot_oracle_ratio <- numeric(bootstrap_draws)
  boot_gain <- numeric(bootstrap_draws)
  replication_count <- nrow(ordinary$value)
  for (bb in seq_len(bootstrap_draws)) {
    index <- sample.int(replication_count, replication_count, replace = TRUE)
    b0 <- trace_variance(ordinary$value[index, , drop = FALSE])
    bs <- trace_variance(selected$value[index, , drop = FALSE])
    bo <- trace_variance(oracle$value[index, , drop = FALSE])
    boot_ratio[[bb]] <- bs / b0
    boot_oracle_ratio[[bb]] <- bo / b0
    boot_gain[[bb]] <- if (abs(1 - boot_oracle_ratio[[bb]]) > 1e-8) {
      (1 - boot_ratio[[bb]]) / (1 - boot_oracle_ratio[[bb]])
    } else NA_real_
  }
  class <- results$scenario_class[match(scenario, results$scenario)]
  ratio_rows[[length(ratio_rows) + 1L]] <- data.frame(
    scenario = scenario,
    scenario_class = class,
    selected_to_ordinary_variance_ratio = ratio,
    ratio_lower_2_5 = unname(stats::quantile(boot_ratio, 0.025, type = 8)),
    ratio_upper_97_5 = unname(stats::quantile(boot_ratio, 0.975, type = 8)),
    oracle_to_ordinary_variance_ratio = oracle_ratio,
    oracle_gain_recovered = gain,
    gain_lower_2_5 = if (any(is.finite(boot_gain))) unname(stats::quantile(
      boot_gain[is.finite(boot_gain)], 0.025, type = 8
    )) else NA_real_,
    gain_upper_97_5 = if (any(is.finite(boot_gain))) unname(stats::quantile(
      boot_gain[is.finite(boot_gain)], 0.975, type = 8
    )) else NA_real_,
    stringsAsFactors = FALSE
  )
  bootstrap_store[[scenario]] <- list(
    selected_ratio = boot_ratio,
    oracle_ratio = boot_oracle_ratio,
    gain = boot_gain
  )
}
ratio_summary <- do.call(rbind, ratio_rows)

weak_scenarios <- ratio_summary$scenario[
  ratio_summary$scenario_class == "weak_headroom_safety"
]
aggregate_ratio <- function(scenarios, index_by_scenario = NULL) {
  numerator <- denominator <- numeric(length(scenarios))
  for (j in seq_along(scenarios)) {
    ordinary <- scenario_matrix(scenarios[[j]], "ordinary")
    selected <- scenario_matrix(scenarios[[j]], "selected")
    if (!is.null(index_by_scenario)) {
      index <- index_by_scenario[[j]]
      ordinary$value <- ordinary$value[index, , drop = FALSE]
      selected$value <- selected$value[index, , drop = FALSE]
    }
    denominator[[j]] <- trace_variance(ordinary$value)
    numerator[[j]] <- trace_variance(selected$value)
  }
  mean(numerator) / mean(denominator)
}
weak_aggregate_ratio <- aggregate_ratio(weak_scenarios)
weak_bootstrap <- numeric(bootstrap_draws)
if (length(weak_scenarios)) {
  for (bb in seq_len(bootstrap_draws)) {
    indices <- lapply(weak_scenarios, function(scenario) {
      n <- nrow(scenario_matrix(scenario, "ordinary")$value)
      sample.int(n, n, replace = TRUE)
    })
    weak_bootstrap[[bb]] <- aggregate_ratio(weak_scenarios, indices)
  }
}

expected_replications <- as.integer(config$replications_per_scenario)
completed <- aggregate(replication ~ scenario + method, results, function(x) {
  length(unique(x))
})
failure_count <- if (is.null(failures)) 0L else nrow(failures)
failure_rate <- failure_count /
  (length(config$simulation$scenarios) * expected_replications)
maximum_calibration <- max(results$calibration_residual)

strong <- ratio_summary[ratio_summary$scenario_class == "strong_headroom", ]
safety <- ratio_summary[ratio_summary$scenario_class %in%
                          c("weak_headroom_safety", "covariance_robustness"), ]
validity <- method_summary[
  method_summary$method %in% methods_for_validity,
  , drop = FALSE
]

strong_point_pass <- all(strong$selected_to_ordinary_variance_ratio <=
                           rules$strong_each_point_variance_ratio_max)
strong_upper_pass <- all(strong$ratio_upper_97_5 <
                           rules$strong_each_upper_97_5_max)
strong_gain_pass <- all(strong$oracle_gain_recovered >=
                          rules$strong_each_oracle_gain_recovered_min)
weak_pass <- weak_aggregate_ratio <= rules$weak_aggregate_variance_ratio_max
safety_pass <- all(safety$selected_to_ordinary_variance_ratio <=
                     rules$safety_each_point_variance_ratio_max)
coverage_pass <- all(
  validity$pooled_coverage >= rules$selected_coverage_min &
    validity$pooled_coverage <= rules$selected_coverage_max
)
bias_pass <- all(validity$rms_standardized_bias <=
                   rules$rms_standardized_bias_max)
variance_calibration_pass <- all(
  validity$minimum_variance_calibration >=
    rules$sandwich_variance_calibration_min &
    validity$maximum_variance_calibration <=
      rules$sandwich_variance_calibration_max
)
calibration_pass <- maximum_calibration <= rules$maximum_calibration_residual
failure_pass <- failure_rate <= rules$maximum_numerical_failure_rate &&
  all(completed$replication == expected_replications)

correctness_path <- file.path(repo_root, "results", "feasible-eif",
                              "correctness-tests.yml")
correctness <- if (file.exists(correctness_path)) yaml::read_yaml(correctness_path) else
  list(passed = FALSE)
correctness_pass <- isTRUE(correctness$passed)

component <- data.frame(
  component = c(
    "strong_point", "strong_upper", "oracle_gain_recovered",
    "weak_aggregate_safety", "each_safety_cell", "coverage",
    "standardized_bias", "sandwich_variance_calibration",
    "exact_calibration", "numerical_completion", "correctness_invariance"
  ),
  passed = c(
    strong_point_pass, strong_upper_pass, strong_gain_pass,
    weak_pass, safety_pass, coverage_pass, bias_pass,
    variance_calibration_pass, calibration_pass, failure_pass,
    correctness_pass
  ),
  stringsAsFactors = FALSE
)
gate_passed <- production && all(component$passed)

selector_summary <- as.data.frame(prop.table(table(
  selector$scenario, selector$selected_candidate
), margin = 1L))
names(selector_summary) <- c("scenario", "selected_candidate", "fraction")

utils::write.csv(coordinate_summary,
                 file.path(result_dir, "coordinate_summary.csv"), row.names = FALSE)
utils::write.csv(method_summary,
                 file.path(result_dir, "method_summary.csv"), row.names = FALSE)
utils::write.csv(ratio_summary,
                 file.path(result_dir, "variance_ratio_summary.csv"), row.names = FALSE)
utils::write.csv(selector_summary,
                 file.path(result_dir, "selector_summary.csv"), row.names = FALSE)
utils::write.csv(component,
                 file.path(result_dir, "gate_components.csv"), row.names = FALSE)

decision <- list(
  gate = "D_feasible_covariance_adaptive_estimator",
  mode = config$mode,
  production_gate_evaluated = production,
  passed = gate_passed,
  decision = if (!production) "smoke_only_no_gate_decision" else if (gate_passed) {
    "proceed_to_gate_E_assumption_sensitivity"
  } else "gate_D_failed_reframe_or_redesign",
  weak_aggregate_variance_ratio = weak_aggregate_ratio,
  weak_aggregate_upper_97_5 = if (length(weak_bootstrap)) {
    unname(stats::quantile(weak_bootstrap, 0.975, type = 8))
  } else NA_real_,
  worst_strong_point_ratio = max(strong$selected_to_ordinary_variance_ratio),
  worst_strong_upper_97_5 = max(strong$ratio_upper_97_5),
  minimum_strong_oracle_gain_recovered = min(strong$oracle_gain_recovered),
  maximum_safety_point_ratio = max(safety$selected_to_ordinary_variance_ratio),
  minimum_pooled_coverage = min(validity$pooled_coverage),
  maximum_pooled_coverage = max(validity$pooled_coverage),
  maximum_rms_standardized_bias = max(validity$rms_standardized_bias),
  minimum_variance_calibration = min(validity$minimum_variance_calibration),
  maximum_variance_calibration = max(validity$maximum_variance_calibration),
  maximum_calibration_residual = maximum_calibration,
  numerical_failure_rate = failure_rate,
  correctness_tests_passed = correctness_pass,
  config_md5 = unname(tools::md5sum(config$.path)),
  generated_at_utc = format(Sys.time(), tz = "UTC", usetz = TRUE)
)
yaml::write_yaml(decision, file.path(result_dir, "gate_d_decision.yml"))
saveRDS(list(
  decision = decision, components = component,
  coordinate_summary = coordinate_summary, method_summary = method_summary,
  ratio_summary = ratio_summary, selector_summary = selector_summary,
  bootstrap = bootstrap_store, weak_bootstrap = weak_bootstrap,
  config = config
), file.path(result_dir, "gate_d_summary.rds"), version = 3)

print(ratio_summary[c(
  "scenario", "scenario_class", "selected_to_ordinary_variance_ratio",
  "ratio_upper_97_5", "oracle_gain_recovered"
)], row.names = FALSE, digits = 4)
cat("\nGate components:\n")
print(component, row.names = FALSE)
cat("\nDecision: ", decision$decision, "\n", sep = "")
