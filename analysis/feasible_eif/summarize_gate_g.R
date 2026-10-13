#!/usr/bin/env Rscript

# Gate G proof-of-concept summarizer. Decisions use only the prespecified
# x1_x2 contrast and the frozen rules in gate-g-screen.yml.

script_path <- function() {
  flag <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (!length(flag)) {
    return(normalizePath(
      "analysis/feasible_eif/summarize_gate_g.R", mustWork = TRUE
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

default_dir <- file.path(repo_root, "results", "feasible-eif",
                         "gate-g-screen")
input_path <- value_arg(
  "--input", file.path(default_dir, "gate_g_simulation_results.rds")
)
if (!grepl("^/", input_path)) input_path <- file.path(repo_root, input_path)
input_path <- normalizePath(input_path, mustWork = TRUE)
output_arg <- value_arg("--output-dir", dirname(input_path))
output_dir <- if (grepl("^/", output_arg)) output_arg else
  file.path(repo_root, output_arg)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
output_dir <- normalizePath(output_dir, mustWork = TRUE)

if (!requireNamespace("yaml", quietly = TRUE)) {
  stop("Package `yaml` is required.", call. = FALSE)
}

bundle <- readRDS(input_path)
if (!is.list(bundle) || is.null(bundle$results) || is.null(bundle$config)) {
  stop("Gate-G input must be a runner bundle with results and config.",
       call. = FALSE)
}

results <- bundle$results
covariance <- bundle$covariance
folds <- bundle$folds
support_diagnostics <- bundle$support_diagnostics
split_diagnostics <- bundle$split_diagnostics
failures <- bundle$failures
config <- bundle$config
rules <- config$gate_g_pass_rules
if (is.null(rules)) stop("Gate-G pass rules are missing.", call. = FALSE)
`%||%` <- function(x, y) if (is.null(x)) y else x

# Accept the pre-audit aliases but write summaries using the audited names.
rename_alias <- function(data, canonical, aliases) {
  if (canonical %in% names(data)) return(data)
  hit <- aliases[aliases %in% names(data)]
  if (length(hit)) names(data)[match(hit[[1L]], names(data))] <- canonical
  data
}
aliases <- list(
  frozen_operator_mean = c("conditional_mean"),
  frozen_operator_bias = c("conditional_bias"),
  propagated_variance = c("exact_variance"),
  plugin_propagated_variance = c("model_variance"),
  predicted_mse_propagated = c("predicted_mse_count_aware"),
  covered_propagated = c("covered_exact"),
  covered_plugin = c("covered_model"),
  propagated_variance_is_exact = c("exact_for_realized_operator")
)
for (canonical in names(aliases)) {
  results <- rename_alias(results, canonical, aliases[[canonical]])
}

required_result_columns <- c(
  "scenario", "scenario_class", "replication", "respondents", "method",
  "coordinate", "truth", "estimate", "error", "propagated_variance",
  "predicted_mse_propagated", "covered_propagated",
  "propagated_variance_is_exact", "first_order_scaled_variance",
  "calibration_residual"
)
missing_result_columns <- setdiff(required_result_columns, names(results))
if (length(missing_result_columns)) {
  stop("Gate-G result schema is missing: ",
       paste(missing_result_columns, collapse = ", "), call. = FALSE)
}
required_covariance_columns <- c(
  "scenario", "replication", "outer_fold", "respondents",
  "raw_frobenius_squared", "count_frobenius_squared"
)
if (is.null(covariance) ||
    length(setdiff(required_covariance_columns, names(covariance)))) {
  stop("Gate-G covariance diagnostics are missing or malformed.",
       call. = FALSE)
}
required_fold_columns <- c(
  "scenario", "replication", "respondent", "outer_fold"
)
if (is.null(folds) || length(setdiff(required_fold_columns, names(folds)))) {
  stop("Gate-G outer-fold diagnostics are missing or malformed.",
       call. = FALSE)
}
if (is.null(failures)) {
  failures <- data.frame(
    scenario = character(), replication = integer(), seed = integer(),
    message = character(), stringsAsFactors = FALSE
  )
}

primary_coordinate <- "x1_x2"
primary_available <- primary_coordinate %in% unique(results$coordinate)
primary <- results[results$coordinate == primary_coordinate, , drop = FALSE]

scenario_registry <- do.call(rbind, lapply(
  config$simulation$scenarios,
  function(scenario) data.frame(
    scenario = as.character(scenario$id),
    scenario_class = as.character(scenario$class),
    respondents = as.integer(scenario$respondents),
    local_scaling_pair = as.character(
      if (is.null(scenario$local_scaling_pair)) NA else
        scenario$local_scaling_pair
    ),
    stringsAsFactors = FALSE
  )
))
scenario_ids <- scenario_registry$scenario
null_registry <- scenario_registry[scenario_registry$scenario_class == "null",
                                   , drop = FALSE]
lf_registry <- scenario_registry[
  scenario_registry$scenario_class == "local_least_favourable",
  , drop = FALSE
]
required_methods <- as.character(config$estimator$gate_g_methods)
planned_replications <- as.integer(config$replications_per_scenario)
outer_folds <- as.integer(config$estimator$outer_folds)
expected_coordinates <- if (!is.null(bundle$geometry$truth)) {
  names(bundle$geometry$truth)
} else {
  sort(unique(results$coordinate))
}

# ---- Provenance and completion -------------------------------------------

hashes_match <- function(paths, expected) {
  if (is.null(paths) || is.null(expected) || !length(paths) ||
      !all(file.exists(paths))) return(FALSE)
  observed <- unname(tools::md5sum(paths))
  if (!is.null(names(expected)) && !is.null(names(paths))) {
    expected <- expected[names(paths)]
  }
  length(observed) == length(expected) &&
    all(!is.na(expected)) && identical(observed, unname(expected))
}
source_integrity <- hashes_match(bundle$source_paths, bundle$source_md5)
config_integrity <- hashes_match(bundle$config_paths, bundle$config_md5)
theory_checks_passed <- isTRUE(bundle$theory_checks_passed) &&
  isTRUE(bundle$run_metadata$theory_checks_passed)

result_key <- paste(
  results$scenario, results$replication, results$method, results$coordinate,
  sep = "\r"
)
result_keys_unique <- !anyDuplicated(result_key)
expected_result_rows <- length(scenario_ids) * planned_replications *
  length(required_methods) * length(expected_coordinates)
result_registry_valid <- nrow(results) == expected_result_rows &&
  all(results$scenario %in% scenario_ids) &&
  all(results$method %in% required_methods) &&
  all(results$coordinate %in% expected_coordinates) &&
  all(results$replication %in% seq_len(planned_replications))

replication_registry_valid <- all(vapply(scenario_ids, function(id) {
  setequal(
    unique(results$replication[results$scenario == id]),
    seq_len(planned_replications)
  )
}, logical(1)))

covariance_key <- paste(
  covariance$scenario, covariance$replication, covariance$outer_fold,
  sep = "\r"
)
covariance_complete <- !anyDuplicated(covariance_key) &&
  nrow(covariance) == length(scenario_ids) * planned_replications * outer_folds

fold_key <- paste(folds$scenario, folds$replication, folds$respondent,
                  sep = "\r")
expected_fold_rows <- planned_replications * sum(scenario_registry$respondents)
fold_registry_complete <- !anyDuplicated(fold_key) &&
  nrow(folds) == expected_fold_rows &&
  all(folds$outer_fold %in% seq_len(outer_folds)) &&
  all(vapply(scenario_ids, function(id) {
    expected_n <- scenario_registry$respondents[
      scenario_registry$scenario == id
    ][[1L]]
    pieces <- split(folds[folds$scenario == id, , drop = FALSE],
                    folds$replication[folds$scenario == id])
    length(pieces) == planned_replications && all(vapply(pieces, function(x) {
      nrow(x) == expected_n &&
        setequal(x$respondent, seq_len(expected_n)) &&
        setequal(x$outer_fold, seq_len(outer_folds))
    }, logical(1)))
  }, logical(1)))

effective_replications <- as.integer(
  bundle$effective_replications %||% bundle$run_metadata$effective_replications_per_scenario
)
replication_count_complete <- length(effective_replications) == 1L &&
  is.finite(effective_replications) &&
  effective_replications == planned_replications
complete_evidence <- primary_available && result_keys_unique &&
  result_registry_valid && replication_registry_valid &&
  covariance_complete && fold_registry_complete &&
  replication_count_complete

# ---- Gate summaries -------------------------------------------------------

safe_var <- function(x) if (sum(is.finite(x)) >= 2L) stats::var(x) else NA_real_
logical_mean <- function(x) mean(as.logical(x), na.rm = TRUE)

oracle_method <- "oracle_hstar_crossfit"
null_oracle_summary <- do.call(rbind, lapply(seq_len(nrow(null_registry)),
  function(index) {
    meta <- null_registry[index, ]
    x <- primary[
      primary$scenario == meta$scenario & primary$method == oracle_method,
      , drop = FALSE
    ]
    empirical_variance <- safe_var(x$error)
    propagated_variance <- mean(x$propagated_variance)
    data.frame(
      scenario = meta$scenario,
      respondents = meta$respondents,
      replications = nrow(x),
      empirical_variance = empirical_variance,
      mean_propagated_variance = propagated_variance,
      empirical_to_propagated_variance =
        empirical_variance / propagated_variance,
      propagated_coverage = logical_mean(x$covered_propagated),
      all_propagated_variances_exact =
        all(as.logical(x$propagated_variance_is_exact)),
      stringsAsFactors = FALSE
    )
  }
))

exact_variance_pass <- nrow(null_oracle_summary) == nrow(null_registry) &&
  all(null_oracle_summary$replications == planned_replications) &&
  all(null_oracle_summary$all_propagated_variances_exact) &&
  all(null_oracle_summary$empirical_to_propagated_variance >=
        as.numeric(rules$exact_variance_ratio_min) &
      null_oracle_summary$empirical_to_propagated_variance <=
        as.numeric(rules$exact_variance_ratio_max))
coverage_pass <- nrow(null_oracle_summary) == nrow(null_registry) &&
  all(null_oracle_summary$propagated_coverage >=
        as.numeric(rules$exact_coverage_min) &
      null_oracle_summary$propagated_coverage <=
        as.numeric(rules$exact_coverage_max))

scaling_rows <- do.call(rbind, lapply(seq_len(nrow(lf_registry)),
  function(index) {
    meta <- lf_registry[index, ]
    x <- primary[
      primary$scenario == meta$scenario & primary$method == oracle_method,
      , drop = FALSE
    ]
    empirical_scaled <- meta$respondents * safe_var(x$error)
    predicted_scaled <- meta$respondents * mean(x$propagated_variance)
    first_order <- mean(x$first_order_scaled_variance)
    finite_sample_remainder <- predicted_scaled - first_order
    data.frame(
      scenario = meta$scenario,
      pair = meta$local_scaling_pair,
      respondents = meta$respondents,
      empirical_scaled_variance = empirical_scaled,
      predicted_scaled_variance = predicted_scaled,
      first_order_scaled_variance = first_order,
      predicted_finite_sample_remainder = finite_sample_remainder,
      corrected_empirical_scaled_variance =
        empirical_scaled - finite_sample_remainder,
      stringsAsFactors = FALSE
    )
  }
))

scaling_pair_valid <- nrow(scaling_rows) == 2L &&
  setequal(scaling_rows$respondents, c(250L, 1000L)) &&
  length(unique(scaling_rows$pair)) == 1L
if (scaling_pair_valid) {
  scaling_rows <- scaling_rows[order(scaling_rows$respondents), ]
  first_order_reference <- mean(scaling_rows$first_order_scaled_variance)
  original_scaling_discrepancy <- abs(diff(
    scaling_rows$empirical_scaled_variance
  )) / first_order_reference
  corrected_scaling_discrepancy <- abs(diff(
    scaling_rows$corrected_empirical_scaled_variance
  )) / first_order_reference
} else {
  first_order_reference <- original_scaling_discrepancy <-
    corrected_scaling_discrepancy <- NA_real_
}
scaling_summary <- data.frame(
  pair = if (nrow(scaling_rows)) scaling_rows$pair[[1L]] else NA_character_,
  first_order_reference = first_order_reference,
  original_scaled_variance_discrepancy = original_scaling_discrepancy,
  corrected_scaled_variance_discrepancy = corrected_scaling_discrepancy,
  threshold = as.numeric(rules$corrected_scaled_variance_discrepancy_max),
  stringsAsFactors = FALSE
)
scaling_pass <- scaling_pair_valid && is.finite(corrected_scaling_discrepancy) &&
  corrected_scaling_discrepancy <=
    as.numeric(rules$corrected_scaled_variance_discrepancy_max)

amse_methods <- c(
  "oracle_hstar_crossfit", "oracle_tau_0.25_crossfit",
  "oracle_tau_0.50_crossfit", "oracle_tau_0.75_crossfit"
)
# Runner method names use decimal points; tolerate underscore aliases.
available_methods <- unique(primary$method)
amse_methods <- vapply(amse_methods, function(method) {
  if (method %in% available_methods) return(method)
  alias <- gsub("\\.", "_", method)
  if (alias %in% available_methods) alias else method
}, character(1))
amse_pieces <- lapply(seq_len(nrow(lf_registry)), function(index) {
  do.call(rbind, lapply(amse_methods, function(method) {
    meta <- lf_registry[index, ]
    x <- primary[
      primary$scenario == meta$scenario & primary$method == method,
      , drop = FALSE
    ]
    empirical_mse <- mean(x$error^2)
    predicted_mse <- mean(x$predicted_mse_propagated)
    data.frame(
      scenario = meta$scenario,
      respondents = meta$respondents,
      method = method,
      replications = nrow(x),
      empirical_mse = empirical_mse,
      mean_predicted_mse = predicted_mse,
      empirical_to_predicted_mse = empirical_mse / predicted_mse,
      all_propagated_variances_exact =
        length(x$propagated_variance_is_exact) > 0L &&
        all(as.logical(x$propagated_variance_is_exact)),
      stringsAsFactors = FALSE
    )
  }))
})
amse_summary <- do.call(rbind, amse_pieces)
amse_pass <- nrow(amse_summary) == nrow(lf_registry) * length(amse_methods) &&
  all(amse_summary$replications == planned_replications) &&
  all(amse_summary$all_propagated_variances_exact) &&
  all(amse_summary$empirical_to_predicted_mse >=
        as.numeric(rules$corrected_amse_ratio_min) &
      amse_summary$empirical_to_predicted_mse <=
        as.numeric(rules$corrected_amse_ratio_max))

covariance_gate_registry <- null_registry[
  null_registry$respondents %in% c(250L, 500L), , drop = FALSE
]
covariance_summary <- do.call(rbind, lapply(
  seq_len(nrow(covariance_gate_registry)), function(index) {
    meta <- covariance_gate_registry[index, ]
    x <- covariance[covariance$scenario == meta$scenario, , drop = FALSE]
    raw_mse <- mean(x$raw_frobenius_squared)
    count_mse <- mean(x$count_frobenius_squared)
    data.frame(
      scenario = meta$scenario,
      respondents = meta$respondents,
      fold_replications = nrow(x),
      raw_mean_frobenius_squared = raw_mse,
      count_mean_frobenius_squared = count_mse,
      relative_mse_reduction = 1 - count_mse / raw_mse,
      stringsAsFactors = FALSE
    )
  }
))
covariance_pass <- nrow(covariance_summary) == 2L &&
  all(covariance_summary$relative_mse_reduction >=
        as.numeric(rules$covariance_frobenius_mse_reduction_min))

feasible_summary <- do.call(rbind, lapply(seq_len(nrow(null_registry)),
  function(index) {
    meta <- null_registry[index, ]
    raw <- primary[
      primary$scenario == meta$scenario &
        primary$method == "feasible_raw_hstar_crossfit", , drop = FALSE
    ]
    count <- primary[
      primary$scenario == meta$scenario &
        primary$method == "feasible_count_hstar_crossfit", , drop = FALSE
    ]
    oracle <- primary[
      primary$scenario == meta$scenario & primary$method == oracle_method,
      , drop = FALSE
    ]
    raw_variance <- safe_var(raw$error)
    count_variance <- safe_var(count$error)
    oracle_bound <- mean(
      oracle$first_order_scaled_variance / oracle$respondents
    )
    data.frame(
      scenario = meta$scenario,
      respondents = meta$respondents,
      raw_variance = raw_variance,
      count_variance = count_variance,
      count_to_raw_variance = count_variance / raw_variance,
      oracle_first_order_bound = oracle_bound,
      raw_to_oracle = raw_variance / oracle_bound,
      count_to_oracle = count_variance / oracle_bound,
      stringsAsFactors = FALSE
    )
  }
))
raw_geometric_mean <- exp(mean(log(feasible_summary$raw_to_oracle)))
count_geometric_mean <- exp(mean(log(feasible_summary$count_to_oracle)))
feasible_summary$raw_geometric_mean_to_oracle <- raw_geometric_mean
feasible_summary$count_geometric_mean_to_oracle <- count_geometric_mean
feasible_pass <- nrow(feasible_summary) == nrow(null_registry) &&
  all(feasible_summary$count_to_raw_variance <=
        as.numeric(rules$feasible_count_variance_inflation_max)) &&
  is.finite(raw_geometric_mean) && is.finite(count_geometric_mean) &&
  count_geometric_mean < raw_geometric_mean

support_summary <- data.frame(
  diagnostic = c(
    "outer_training_empty_cells",
    "auxiliary_covariance_df_fallback_rows",
    "honest_split_ordinary_fallback_rows"
  ),
  count = c(
    if (is.null(support_diagnostics)) NA_integer_ else
      sum(support_diagnostics$empty_training_cells),
    if (!"covariance_correction_strategy" %in% names(covariance)) NA_integer_ else
      sum(covariance$covariance_correction_strategy ==
            "full_residual_df_fallback"),
    if (is.null(split_diagnostics) ||
        !"strategy" %in% names(split_diagnostics)) NA_integer_ else
      sum(split_diagnostics$strategy == "ordinary_support_fallback")
  ),
  denominator = c(
    if (is.null(support_diagnostics)) NA_integer_ else nrow(support_diagnostics),
    nrow(covariance),
    if (is.null(split_diagnostics)) NA_integer_ else nrow(split_diagnostics)
  ),
  stringsAsFactors = FALSE
)
support_summary$rate <- support_summary$count / support_summary$denominator

planned_dataset_count <- length(scenario_ids) * planned_replications
failure_rate <- nrow(failures) / planned_dataset_count
maximum_calibration <- if (nrow(results)) {
  max(abs(results$calibration_residual), na.rm = TRUE)
} else {
  Inf
}
outer_isolation <- fold_registry_complete
numerics_pass <- nrow(failures) == 0L &&
  failure_rate <= as.numeric(rules$maximum_failure_rate) &&
  outer_isolation && is.finite(maximum_calibration) &&
  maximum_calibration < as.numeric(rules$maximum_calibration_residual)

component <- function(name, passed, observed, rule, prerequisite = FALSE) {
  data.frame(
    component = name,
    prerequisite = prerequisite,
    passed = isTRUE(passed),
    observed = as.character(observed),
    rule = as.character(rule),
    stringsAsFactors = FALSE
  )
}
components <- do.call(rbind, list(
  component(
    "source_and_config_integrity", source_integrity && config_integrity,
    paste(source_integrity, config_integrity, sep = "/"),
    "both source and config hashes match", TRUE
  ),
  component(
    "deterministic_theory_checks", theory_checks_passed,
    theory_checks_passed, "all required exact identities pass", TRUE
  ),
  component(
    "complete_frozen_screen", complete_evidence,
    paste(nrow(results), expected_result_rows, sep = "/"),
    paste0(planned_replications, " replications in every planned cell"), TRUE
  ),
  component(
    "exact_fixed_operator_variance", exact_variance_pass,
    if (nrow(null_oracle_summary)) paste(range(
      null_oracle_summary$empirical_to_propagated_variance
    ), collapse = "--") else NA,
    paste0("ratio in [", rules$exact_variance_ratio_min, ", ",
           rules$exact_variance_ratio_max, "]")
  ),
  component(
    "count_aware_coverage", coverage_pass,
    if (nrow(null_oracle_summary)) paste(range(
      null_oracle_summary$propagated_coverage
    ), collapse = "--") else NA,
    paste0("coverage in [", rules$exact_coverage_min, ", ",
           rules$exact_coverage_max, "]")
  ),
  component(
    "scaled_variance_repair", scaling_pass,
    corrected_scaling_discrepancy,
    paste0("discrepancy <= ",
           rules$corrected_scaled_variance_discrepancy_max)
  ),
  component(
    "local_amse_repair", amse_pass,
    if (nrow(amse_summary)) paste(range(
      amse_summary$empirical_to_predicted_mse
    ), collapse = "--") else NA,
    paste0("ratio in [", rules$corrected_amse_ratio_min, ", ",
           rules$corrected_amse_ratio_max, "]")
  ),
  component(
    "covariance_count_correction", covariance_pass,
    if (nrow(covariance_summary)) paste(range(
      covariance_summary$relative_mse_reduction
    ), collapse = "--") else NA,
    paste0("reduction >= ",
           rules$covariance_frobenius_mse_reduction_min)
  ),
  component(
    "feasible_score_safety", feasible_pass,
    paste(count_geometric_mean, raw_geometric_mean, sep = "/"),
    paste0("each count/raw <= ",
           rules$feasible_count_variance_inflation_max,
           " and count geometric mean improves")
  ),
  component(
    "numerics_and_leakage", numerics_pass,
    paste(nrow(failures), maximum_calibration, outer_isolation, sep = "/"),
    paste0("zero failures; isolated outer folds; calibration < ",
           rules$maximum_calibration_residual)
  )
))

formal_decision <- complete_evidence && source_integrity && config_integrity &&
  theory_checks_passed
required_pass <- all(components$passed)
decision_label <- if (!formal_decision) {
  "incomplete_no_gate_decision"
} else if (required_pass) {
  "gate_G_screen_pass_proceed_to_production_gate"
} else {
  "gate_G_screen_failed"
}

write_csv <- function(data, filename) {
  utils::write.csv(data, file.path(output_dir, filename), row.names = FALSE)
}
write_csv(null_oracle_summary, "gate_g_oracle_null_summary.csv")
write_csv(scaling_rows, "gate_g_scaling_cells.csv")
write_csv(scaling_summary, "gate_g_scaling_summary.csv")
write_csv(amse_summary, "gate_g_local_amse_summary.csv")
write_csv(covariance_summary, "gate_g_covariance_correction_summary.csv")
write_csv(feasible_summary, "gate_g_feasible_safety_summary.csv")
write_csv(support_summary, "gate_g_support_summary.csv")
write_csv(components, "gate_g_components.csv")

records <- function(data) {
  lapply(seq_len(nrow(data)), function(index) as.list(data[index, , drop = FALSE]))
}
decision <- list(
  gate = "G_finite_sample_nuisance_remainder",
  protocol = "docs/gate-g-protocol.md",
  primary_coordinate = primary_coordinate,
  formal_decision = formal_decision,
  decision = decision_label,
  passed = formal_decision && required_pass,
  planned_replications_per_scenario = planned_replications,
  effective_replications_per_scenario = effective_replications,
  completed_datasets = length(unique(paste(
    results$scenario, results$replication, sep = "\r"
  ))),
  planned_datasets = planned_dataset_count,
  numerical_failures = nrow(failures),
  failure_rate = failure_rate,
  maximum_calibration_residual = maximum_calibration,
  outer_respondent_isolation = outer_isolation,
  source_integrity = source_integrity,
  config_integrity = config_integrity,
  deterministic_theory_checks_passed = theory_checks_passed,
  support_diagnostics = records(support_summary),
  original_scaled_variance_discrepancy = original_scaling_discrepancy,
  corrected_scaled_variance_discrepancy = corrected_scaling_discrepancy,
  failed_required_components = as.list(
    components$component[!components$passed]
  ),
  components = records(components),
  generated_at_utc = format(Sys.time(), tz = "UTC", usetz = TRUE),
  input_bundle = input_path
)
yaml::write_yaml(decision, file.path(output_dir, "gate_g_decision.yml"))

cat("Gate G decision: ", decision_label, "\n", sep = "")
cat("Component table: ", file.path(output_dir, "gate_g_components.csv"),
    "\n", sep = "")
