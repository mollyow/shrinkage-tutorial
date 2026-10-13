#!/usr/bin/env Rscript

# Gate E2: computational assessment of local-misspecification robustness.
#
# The simulation runner is intentionally decoupled from this file.  This
# summarizer accepts a small set of aliases (documented below), but it never
# converts missing production evidence into a pass.

script_path <- function() {
  flag <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (!length(flag)) {
    return(normalizePath("analysis/feasible_eif/summarize_gate_e2.R",
                         mustWork = FALSE))
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
  if (length(hit)) sub(paste0("^", prefix, "="), "", hit[[1L]]) else default
}

mode_arg <- value_arg("--mode", "smoke")
default_result_dir <- file.path(repo_root, "results", "feasible-eif",
                                paste0("gate-e2-", mode_arg))
result_path <- value_arg(
  "--input",
  file.path(default_result_dir, "gate_e2_simulation_results.rds")
)
result_dir <- value_arg("--output-dir", dirname(result_path))

if (!file.exists(result_path)) {
  stop("Gate-E2 simulation results are missing: ", result_path, call. = FALSE)
}
if (!requireNamespace("yaml", quietly = TRUE)) {
  stop("The yaml package is required to write the Gate-E2 decision.",
       call. = FALSE)
}
dir.create(result_dir, recursive = TRUE, showWarnings = FALSE)

bundle <- readRDS(result_path)
if (is.data.frame(bundle)) bundle <- list(results = bundle)
if (is.null(bundle$results)) {
  stop("The input RDS must contain bundle$results (or itself be a data frame).",
       call. = FALSE)
}

# Prefer the runner's embedded configuration.  A separately stored protocol is
# a fallback for early runner builds, not a reason to overwrite embedded rules.
config_path <- NULL
config <- bundle$config
if (is.null(config)) {
  candidates <- c(
    file.path(analysis_dir, paste0("gate-e2-", mode_arg, ".yml")),
    file.path(analysis_dir, "gate-e2-production.yml"),
    file.path(analysis_dir, "gate-e2-protocol.yml")
  )
  candidates <- candidates[file.exists(candidates)]
  if (length(candidates)) {
    config_path <- normalizePath(candidates[[1L]], mustWork = TRUE)
    config <- yaml::read_yaml(config_path)
  } else {
    config <- list()
  }
} else if (!is.null(config$.path) && file.exists(config$.path)) {
  config_path <- normalizePath(config$.path, mustWork = TRUE)
}

mode <- if (!is.null(config$mode)) as.character(config$mode) else mode_arg
production <- identical(mode, "production")

`%||%` <- function(x, y) if (is.null(x)) y else x
hard_protocol <- config$hard_correctness_pass_rules %||%
  config$pass_rules$hard_correctness
practical_protocol <- config$practical_value_pass_rules %||%
  config$pass_rules$practical_value
frozen_protocol <- !is.null(hard_protocol) && !is.null(practical_protocol)

default_rules <- list(
  maximum_calibration_residual = 1e-10,
  maximum_numerical_failure_rate = 0,
  bias_agreement_ci_level = 0.99,
  bias_prediction_paired_mc_z_max = 3.50,
  bias_prediction_standardized_error_max = 0.10,
  assignment_bias_standardized_error_max = 0.10,
  mse_theory_relative_error_max = 0.10,
  mse_theory_ratio_min = 0.90,
  mse_theory_ratio_max = 1.10,
  sharpness_oracle_min = 0.95,
  sharpness_oracle_max = 1.05,
  sharpness_feasible_min = 0.90,
  sharpness_feasible_max = 1.10,
  ordinary_coordinate_coverage_min = 0.93,
  ordinary_coordinate_coverage_max = 0.97,
  ordinary_pooled_coverage_min = 0.94,
  ordinary_pooled_coverage_max = 0.96,
  null_restricted_coordinate_coverage_min = 0.93,
  null_restricted_coordinate_coverage_max = 0.97,
  ordinary_rms_standardized_bias_max = 0.10,
  sandwich_variance_calibration_min = 0.90,
  sandwich_variance_calibration_max = 1.10,
  bias_aware_coverage_lcb_min = 0.93,
  hausman_null_size_min = 0.035,
  hausman_null_size_max = 0.065,
  hausman_power_absolute_error_oracle_max = 0.03,
  hausman_power_absolute_error_feasible_max = 0.04,
  hausman_blind_rejection_max = 0.07,
  oracle_blend_mse_relative_error_max = 0.02,
  feasible_blend_mse_relative_error_max = 0.10,
  strong_radius_max = 1,
  strong_oracle_gain_recovered_min = 0.50,
  strong_aggregate_risk_ratio_max = 0.98,
  strong_aggregate_risk_ratio_upper_97_5_max = 1,
  large_departure_radius_min = 2,
  large_departure_aggregate_risk_ratio_max = 1.02,
  large_departure_each_risk_ratio_max = 1.05,
  weak_safety_each_risk_ratio_max = 1.05,
  half_radius_length_ratio_max = 0.90,
  half_radius_length_ratio_upper_97_5_max = 0.98,
  unit_radius_length_ratio_max = 1.10,
  unit_radius_each_length_ratio_max = 1.25,
  monte_carlo_bootstrap_draws = 4000L,
  monte_carlo_bootstrap_seed = 52027L,
  require_family_coverage = TRUE,
  require_assignment_check = TRUE,
  require_calibration_check = TRUE,
  require_hausman_check = TRUE,
  require_sharpness_check = TRUE,
  require_blend_check = TRUE
)
rules <- utils::modifyList(default_rules, config$pass_rules %||% list(),
                           keep.null = TRUE)
if (frozen_protocol) {
  rules$maximum_calibration_residual <-
    hard_protocol$maximum_calibration_residual
  rules$maximum_numerical_failure_rate <-
    hard_protocol$maximum_numerical_failure_rate
  rules$bias_prediction_standardized_error_max <-
    hard_protocol$exact_bias$maximum_absolute_discrepancy_in_mc_standard_deviations
  rules$assignment_bias_standardized_error_max <-
    hard_protocol$exact_bias$maximum_absolute_discrepancy_in_mc_standard_deviations
  rules$bias_prediction_paired_mc_z_max <-
    hard_protocol$exact_bias$maximum_absolute_paired_mc_z
  rules$mse_theory_ratio_min <-
    hard_protocol$local_amse$empirical_to_theory_ratio_min
  rules$mse_theory_ratio_max <-
    hard_protocol$local_amse$empirical_to_theory_ratio_max
  rules$ordinary_coordinate_coverage_min <-
    hard_protocol$ordinary_inference$primary_coverage_min
  rules$ordinary_coordinate_coverage_max <-
    hard_protocol$ordinary_inference$primary_coverage_max
  rules$ordinary_pooled_coverage_min <-
    hard_protocol$ordinary_inference$primary_coverage_min
  rules$ordinary_pooled_coverage_max <-
    hard_protocol$ordinary_inference$primary_coverage_max
  rules$ordinary_rms_standardized_bias_max <-
    hard_protocol$ordinary_inference$maximum_absolute_standardized_bias
  rules$sandwich_variance_calibration_min <-
    hard_protocol$ordinary_inference$sandwich_to_mc_variance_min
  rules$sandwich_variance_calibration_max <-
    hard_protocol$ordinary_inference$sandwich_to_mc_variance_max
  rules$null_restricted_coordinate_coverage_min <-
    hard_protocol$selected_null_and_blind_inference$primary_coverage_min
  rules$null_restricted_coordinate_coverage_max <-
    hard_protocol$selected_null_and_blind_inference$primary_coverage_max
  rules$bias_aware_coverage_lcb_min <-
    hard_protocol$oracle_bias_aware_primary_coverage_min
  rules$hausman_null_size_min <-
    hard_protocol$hausman_directional$null_and_blind_rejection_min
  rules$hausman_null_size_max <-
    hard_protocol$hausman_directional$null_and_blind_rejection_max
  rules$hausman_power_absolute_error_oracle_max <-
    hard_protocol$hausman_directional$oracle_power_absolute_error_max
  rules$hausman_power_absolute_error_feasible_max <-
    hard_protocol$hausman_directional$selected_power_absolute_error_max
  rules$require_family_coverage <- FALSE
  rules$require_sharpness_check <- FALSE
  # The frozen protocol evaluates blend performance in the practical gate;
  # it does not impose an additional generic line-MSE hard gate.
  rules$require_blend_check <- FALSE
}

# ---- Schema normalization -------------------------------------------------

rename_first <- function(data, canonical, aliases) {
  if (canonical %in% names(data)) return(data)
  hit <- aliases[aliases %in% names(data)]
  if (length(hit)) names(data)[match(hit[[1L]], names(data))] <- canonical
  data
}

normalize_aliases <- function(data) {
  aliases <- list(
    scenario = c("scenario_id"),
    scenario_class = c("class"),
    replication = c("replicate", "rep"),
    coordinate = c("target", "contrast", "estimand"),
    method = c("estimator"),
    estimate = c("theta_hat", "estimate_hat"),
    truth = c("true_value", "theta", "estimand_truth"),
    standard_error = c("se", "sandwich_se"),
    naive_covered = c("covered", "coverage", "ci_covered", "covered_naive"),
    naive_length = c("interval_length", "ci_length", "width_naive"),
    bias_aware_covered = c("sensitivity_covered", "robust_covered",
                           "covered_bias_additive"),
    bias_aware_length = c("sensitivity_length", "robust_length",
                          "width_bias_additive"),
    predicted_bias = c("theory_bias", "expected_bias", "analytic_bias"),
    bias_bound = c("sensitivity_bias_bound", "worst_case_bias"),
    predicted_mse = c("theory_mse", "expected_mse", "analytic_mse"),
    hausman_reject = c("hausman_rejected", "pretest_reject"),
    hausman_variant = c("hausman_type", "diagnostic_variant"),
    theoretical_hausman_power = c("hausman_power_theory", "predicted_power"),
    assignment_predicted_bias = c("assignment_theory_bias",
                                  "assignment_expected_bias"),
    calibration_residual = c("solver_residual", "calibration_error"),
    blend_tau = c("tau_hat", "blend_weight"),
    theoretical_tau = c("tau_theory", "optimal_tau"),
    line_oracle_mse = c("oracle_line_mse", "minimum_line_mse"),
    radius_ratio = c("radius", "departure_radius", "r", "normalized_r",
                     "radius_multiplier_r"),
    departure_pattern = c("pattern", "misspecification_pattern")
  )
  for (canonical in names(aliases)) {
    data <- rename_first(data, canonical, aliases[[canonical]])
  }
  data
}

as_logical_safe <- function(x) {
  if (is.logical(x)) return(x)
  if (is.numeric(x)) return(ifelse(is.na(x), NA, x != 0))
  y <- tolower(trimws(as.character(x)))
  out <- rep(NA, length(y))
  out[y %in% c("true", "t", "yes", "y", "1", "pass", "passed")] <- TRUE
  out[y %in% c("false", "f", "no", "n", "0", "fail", "failed")] <- FALSE
  out
}

derive_interval_fields <- function(data) {
  derive_covered <- function(current, lower, upper) {
    if (!current %in% names(data) && all(c(lower, upper, "truth") %in% names(data))) {
      data[[current]] <<- data[[lower]] <= data$truth & data$truth <= data[[upper]]
    }
  }
  derive_length <- function(current, lower, upper) {
    if (!current %in% names(data) && all(c(lower, upper) %in% names(data))) {
      data[[current]] <<- data[[upper]] - data[[lower]]
    }
  }
  naive_pairs <- list(c("naive_ci_lower", "naive_ci_upper"),
                      c("ci_lower", "ci_upper"),
                      c("lower", "upper"))
  for (pair in naive_pairs) {
    if (all(pair %in% names(data))) {
      derive_covered("naive_covered", pair[[1L]], pair[[2L]])
      derive_length("naive_length", pair[[1L]], pair[[2L]])
      break
    }
  }
  aware_pairs <- list(c("bias_aware_lower", "bias_aware_upper"),
                      c("sensitivity_lower", "sensitivity_upper"),
                      c("robust_lower", "robust_upper"))
  for (pair in aware_pairs) {
    if (all(pair %in% names(data))) {
      derive_covered("bias_aware_covered", pair[[1L]], pair[[2L]])
      derive_length("bias_aware_length", pair[[1L]], pair[[2L]])
      break
    }
  }
  if (!"naive_length" %in% names(data) && "standard_error" %in% names(data)) {
    data$naive_length <- 2 * stats::qnorm(0.975) * data$standard_error
  }
  data
}

scenario_frame <- function(config) {
  scenarios <- config$simulation$scenarios %||% config$scenarios
  if (is.null(scenarios) || !length(scenarios)) return(data.frame())
  rows <- lapply(scenarios, function(x) {
    if (!is.list(x)) return(data.frame(scenario = as.character(x)))
    flat <- x[vapply(x, function(z) length(z) == 1L &&
                       (is.atomic(z) || is.factor(z)), logical(1))]
    if (!length(flat)) return(data.frame())
    out <- as.data.frame(flat, stringsAsFactors = FALSE)
    if (!"scenario" %in% names(out) && "id" %in% names(out)) {
      names(out)[names(out) == "id"] <- "scenario"
    }
    out
  })
  fields <- unique(unlist(lapply(rows, names)))
  rows <- lapply(rows, function(x) {
    for (field in setdiff(fields, names(x))) x[[field]] <- NA
    x[fields]
  })
  if (!length(rows)) data.frame() else do.call(rbind, rows)
}

results <- normalize_aliases(as.data.frame(bundle$results))
numeric_like <- unique(c(
  "estimate", "truth", "standard_error", "naive_length",
  "bias_aware_length", "predicted_bias", "bias_bound", "predicted_mse",
  "theoretical_hausman_power", "assignment_predicted_bias",
  "calibration_residual", "blend_tau", "theoretical_tau", "line_oracle_mse",
  "radius_ratio", "hausman_ncp", "hausman_df", "respondents",
  "theoretical_scaled_mse", "sensitivity_margin_true",
  "sensitivity_margin_hat", "naive_ci_lower", "naive_ci_upper", "ci_lower",
  "ci_upper", "lower", "upper", "bias_aware_lower", "bias_aware_upper",
  "sensitivity_lower", "sensitivity_upper", "robust_lower", "robust_upper",
  "width_naive", "width_bias_additive", "width_bias_additive_oracle_k"
))
for (field in intersect(numeric_like, names(results))) {
  original <- results[[field]]
  converted <- suppressWarnings(as.numeric(as.character(original)))
  newly_missing <- !is.na(original) & is.na(converted)
  if (any(newly_missing)) {
    stop("Column `", field, "` contains non-numeric values.", call. = FALSE)
  }
  results[[field]] <- converted
}
results <- derive_interval_fields(results)
# Gate-E2 v1 stores the AMSE on the respondent-local scale.  Convert it to the
# estimator's MSE scale before comparing it with mean squared estimation error.
if (!"predicted_mse" %in% names(results) &&
    all(c("theoretical_scaled_mse", "respondents") %in% names(results))) {
  results$predicted_mse <- results$theoretical_scaled_mse / results$respondents
}
# Oracle-K intervals are the relevant oracle benchmark; feasible methods retain
# the direct estimated-K interval.  Preserve both source columns in `results`.
if ("covered_bias_additive_oracle_k" %in% names(results)) {
  oracle_interval <- grepl("^oracle_", results$method)
  results$bias_aware_covered[oracle_interval] <-
    results$covered_bias_additive_oracle_k[oracle_interval]
}
if ("width_bias_additive_oracle_k" %in% names(results)) {
  oracle_interval <- grepl("^oracle_", results$method)
  results$bias_aware_length[oracle_interval] <-
    results$width_bias_additive_oracle_k[oracle_interval]
}
if (!"bias_aware_family_covered" %in% names(results) &&
    "family_covered_bias_additive" %in% names(results)) {
  results$bias_aware_family_covered <- results$family_covered_bias_additive
  if ("family_covered_bias_additive_oracle_k" %in% names(results)) {
    oracle_interval <- grepl("^oracle_", results$method)
    results$bias_aware_family_covered[oracle_interval] <-
      results$family_covered_bias_additive_oracle_k[oracle_interval]
  }
}
required_key <- c("scenario", "replication", "coordinate", "method",
                  "estimate", "truth")
missing_key <- setdiff(required_key, names(results))
if (length(missing_key)) {
  stop("Gate-E2 results are missing required columns: ",
       paste(missing_key, collapse = ", "), call. = FALSE)
}

metadata <- scenario_frame(config)
metadata <- normalize_aliases(metadata)
if (nrow(metadata) && "scenario" %in% names(metadata)) {
  metadata$scenario <- as.character(metadata$scenario)
  new_fields <- setdiff(names(metadata), names(results))
  if (length(new_fields)) {
    results <- merge(results, metadata[c("scenario", new_fields)],
                     by = "scenario", all.x = TRUE, sort = FALSE)
  } else {
    # Fill row-level missing metadata from configuration without overwriting it.
    common <- setdiff(intersect(names(metadata), names(results)), "scenario")
    if (length(common)) {
      lookup <- metadata[match(as.character(results$scenario), metadata$scenario), ]
      for (field in common) {
        blank <- is.na(results[[field]]) |
          (is.character(results[[field]]) & !nzchar(results[[field]]))
        results[[field]][blank] <- lookup[[field]][blank]
      }
    }
  }
}

results$scenario <- as.character(results$scenario)
results$coordinate <- as.character(results$coordinate)
results$method <- as.character(results$method)
results$replication <- as.character(results$replication)
for (field in c("estimate", "truth", "standard_error", "naive_length",
                "bias_aware_length", "predicted_bias", "bias_bound",
                "predicted_mse", "theoretical_hausman_power",
                "assignment_predicted_bias", "calibration_residual",
                "blend_tau", "theoretical_tau", "line_oracle_mse",
                "radius_ratio", "hausman_ncp", "hausman_df")) {
  if (field %in% names(results)) {
    results[[field]] <- suppressWarnings(as.numeric(as.character(results[[field]])))
  }
}
for (field in c("naive_covered", "bias_aware_covered", "hausman_reject",
                "within_sensitivity_ball", "is_null", "is_least_favourable",
                "is_blind", "is_assignment_error")) {
  if (field %in% names(results)) results[[field]] <- as_logical_safe(results[[field]])
}

if (!"scenario_class" %in% names(results)) results$scenario_class <- "unspecified"
results$scenario_class[is.na(results$scenario_class)] <- "unspecified"
if (!"departure_pattern" %in% names(results)) results$departure_pattern <- "unspecified"
results$departure_pattern[is.na(results$departure_pattern)] <- "unspecified"
if (!"radius_ratio" %in% names(results)) results$radius_ratio <- NA_real_

pattern_text <- tolower(paste(results$scenario, results$scenario_class,
                              results$departure_pattern))
if (!"is_null" %in% names(results)) {
  results$is_null <- (!is.na(results$radius_ratio) & results$radius_ratio == 0) |
    grepl("(^|[_ -])(null|none|no.carryover)([_ -]|$)", pattern_text)
}
if (!"is_least_favourable" %in% names(results)) {
  results$is_least_favourable <- grepl("least|worst", pattern_text)
}
if (!"is_blind" %in% names(results)) {
  results$is_blind <- grepl("blind|undetect", pattern_text)
}
if (!"is_assignment_error" %in% names(results)) {
  results$is_assignment_error <- grepl("assignment|wrong.q|misspecified.q",
                                       pattern_text)
}
if (!"within_sensitivity_ball" %in% names(results)) {
  results$within_sensitivity_ball <- if ("sensitivity_margin_true" %in%
                                         names(results)) {
    is.finite(results$sensitivity_margin_true)
  } else NA
}
if (!"assignment_predicted_bias" %in% names(results) &&
    "predicted_bias" %in% names(results)) {
  results$assignment_predicted_bias <- ifelse(
    results$is_assignment_error, results$predicted_bias, NA_real_
  )
}

if (!"theoretical_hausman_power" %in% names(results) &&
    all(c("hausman_ncp", "hausman_df") %in% names(results))) {
  critical <- stats::qchisq(0.95, df = results$hausman_df)
  results$theoretical_hausman_power <- 1 - stats::pchisq(
    critical, df = results$hausman_df, ncp = results$hausman_ncp
  )
}
if (!"hausman_reject" %in% names(results) && "hausman_p_value" %in% names(results)) {
  results$hausman_reject <- as.numeric(results$hausman_p_value) < 0.05
}
if (!"hausman_variant" %in% names(results)) {
  results$hausman_variant <- "feasible"
}

duplicate_key <- duplicated(results[required_key[1:4]])
if (any(duplicate_key)) {
  stop("Results contain duplicate scenario/replication/coordinate/method rows.",
       call. = FALSE)
}

# ---- Method roles ---------------------------------------------------------

configured_roles <- config$method_roles %||% list()
role_candidates <- list(
  ordinary = c("ordinary_taskwise", "ordinary", "unrestricted",
               "taskwise_aipw", "robust"),
  restricted = c("oracle_adaptive", "restricted", "restricted_oracle",
                 "covariance_oracle"),
  selected = c("selected_adaptive", "selected", "covariance_adaptive",
               "restricted_feasible"),
  oracle_blend = c("oracle_tau_mm", "oracle_blend",
                   "sensitivity_oracle_blend"),
  feasible_blend = c("feasible_tau_generic", "feasible_blend",
                     "sensitivity_blend", "generic_blend")
)
pick_role <- function(role) {
  explicit <- configured_roles[[role]]
  if (!is.null(explicit) && as.character(explicit) %in% unique(results$method)) {
    return(as.character(explicit))
  }
  hit <- role_candidates[[role]][role_candidates[[role]] %in% unique(results$method)]
  if (length(hit)) hit[[1L]] else NA_character_
}
method_roles <- vapply(names(role_candidates), pick_role, character(1))
method_role_frame <- data.frame(role = names(method_roles),
                                method = unname(method_roles),
                                available = !is.na(method_roles),
                                stringsAsFactors = FALSE)

# ---- Stable scalar helpers ------------------------------------------------

mean_or_na <- function(x) if (length(x) && any(is.finite(x))) {
  mean(x[is.finite(x)])
} else NA_real_
var_or_na <- function(x) if (sum(is.finite(x)) >= 2L) {
  stats::var(x[is.finite(x)])
} else NA_real_
max_or_na <- function(x) if (length(x) && any(is.finite(x))) {
  max(x[is.finite(x)])
} else NA_real_
min_or_na <- function(x) if (length(x) && any(is.finite(x))) {
  min(x[is.finite(x)])
} else NA_real_
all_or_false <- function(x) length(x) > 0L && all(!is.na(x)) && all(x)
geometric_mean <- function(x) {
  x <- x[is.finite(x) & x > 0]
  if (length(x)) exp(mean(log(x))) else NA_real_
}
coverage_lcb <- function(x, alpha = 0.05) {
  x <- x[!is.na(x)]
  if (!length(x)) return(NA_real_)
  successes <- sum(x)
  if (successes == 0L) return(0)
  stats::qbeta(alpha, successes, length(x) - successes + 1L)
}

ci_alpha <- (1 - rules$bias_agreement_ci_level) / 2
z_bias <- stats::qnorm(1 - ci_alpha)
meta_fields <- c("scenario", "scenario_class", "departure_pattern",
                 "radius_ratio", "is_null", "is_least_favourable",
                 "is_blind", "is_assignment_error",
                 "within_sensitivity_ball", "method", "coordinate")

summarize_coordinate <- function(x) {
  error <- x$estimate - x$truth
  mc_variance <- var_or_na(x$estimate)
  empirical_bias <- mean_or_na(error)
  predicted_bias <- if ("predicted_bias" %in% names(x))
    mean_or_na(x$predicted_bias) else NA_real_
  residual <- if (is.finite(predicted_bias)) error - x$predicted_bias else NA_real_
  residual_se <- if (sum(is.finite(residual)) >= 2L)
    stats::sd(residual[is.finite(residual)]) / sqrt(sum(is.finite(residual))) else NA_real_
  assignment_predicted <- if ("assignment_predicted_bias" %in% names(x))
    mean_or_na(x$assignment_predicted_bias) else NA_real_
  assignment_residual <- if (is.finite(assignment_predicted))
    error - x$assignment_predicted_bias else NA_real_
  assignment_se <- if (sum(is.finite(assignment_residual)) >= 2L)
    stats::sd(assignment_residual[is.finite(assignment_residual)]) /
      sqrt(sum(is.finite(assignment_residual))) else NA_real_
  se2 <- if ("standard_error" %in% names(x)) x$standard_error^2 else NA_real_
  naive <- if ("naive_covered" %in% names(x)) x$naive_covered else NA
  aware <- if ("bias_aware_covered" %in% names(x)) x$bias_aware_covered else NA
  bias_bound <- if ("bias_bound" %in% names(x)) mean_or_na(x$bias_bound) else NA_real_
  predicted_mse <- if ("predicted_mse" %in% names(x))
    mean_or_na(x$predicted_mse) else NA_real_
  empirical_mse <- mean_or_na(error^2)
  first <- function(field, default = NA) {
    if (!field %in% names(x)) return(default)
    good <- which(!is.na(x[[field]]))
    if (length(good)) x[[field]][good[[1L]]] else default
  }
  data.frame(
    scenario = first("scenario", NA_character_),
    scenario_class = first("scenario_class", NA_character_),
    departure_pattern = first("departure_pattern", NA_character_),
    radius_ratio = as.numeric(first("radius_ratio", NA_real_)),
    is_null = as.logical(first("is_null", NA)),
    is_least_favourable = as.logical(first("is_least_favourable", NA)),
    is_blind = as.logical(first("is_blind", NA)),
    is_assignment_error = as.logical(first("is_assignment_error", NA)),
    within_sensitivity_ball = as.logical(first("within_sensitivity_ball", NA)),
    method = first("method", NA_character_),
    coordinate = first("coordinate", NA_character_),
    replications = length(unique(x$replication)),
    empirical_bias = empirical_bias,
    mc_variance = mc_variance,
    empirical_mse = empirical_mse,
    mean_sandwich_variance = mean_or_na(se2),
    sandwich_variance_calibration = mean_or_na(se2) / mc_variance,
    standardized_bias = abs(empirical_bias) / sqrt(mc_variance),
    naive_coverage = mean_or_na(as.numeric(naive)),
    naive_interval_length = if ("naive_length" %in% names(x))
      mean_or_na(x$naive_length) else NA_real_,
    bias_aware_coverage = mean_or_na(as.numeric(aware)),
    bias_aware_coverage_lcb_95 = coverage_lcb(aware),
    bias_aware_interval_length = if ("bias_aware_length" %in% names(x))
      mean_or_na(x$bias_aware_length) else NA_real_,
    predicted_bias = predicted_bias,
    bias_prediction_error = empirical_bias - predicted_bias,
    bias_prediction_mc_se = residual_se,
    bias_prediction_mc_z = abs(mean_or_na(residual)) / residual_se,
    bias_prediction_ci_99_agreement = is.finite(residual_se) &&
      abs(mean_or_na(residual)) <= z_bias * residual_se,
    bias_prediction_standardized_error = abs(empirical_bias - predicted_bias) /
      sqrt(mc_variance),
    bias_bound = bias_bound,
    bound_sharpness_ratio = abs(empirical_bias) / bias_bound,
    predicted_mse = predicted_mse,
    mse_to_theory_ratio = empirical_mse / predicted_mse,
    mse_theory_relative_error = abs(empirical_mse - predicted_mse) /
      predicted_mse,
    assignment_predicted_bias = assignment_predicted,
    assignment_bias_error = empirical_bias - assignment_predicted,
    assignment_bias_mc_se = assignment_se,
    assignment_bias_mc_z = abs(mean_or_na(assignment_residual)) / assignment_se,
    assignment_bias_ci_99_agreement = is.finite(assignment_se) &&
      abs(mean_or_na(assignment_residual)) <= z_bias * assignment_se,
    assignment_bias_standardized_error =
      abs(empirical_bias - assignment_predicted) / sqrt(mc_variance),
    mean_blend_tau = if ("blend_tau" %in% names(x))
      mean_or_na(x$blend_tau) else NA_real_,
    theoretical_blend_tau = if ("theoretical_tau" %in% names(x))
      mean_or_na(x$theoretical_tau) else NA_real_,
    mean_line_oracle_mse = if ("line_oracle_mse" %in% names(x))
      mean_or_na(x$line_oracle_mse) else NA_real_,
    stringsAsFactors = FALSE
  )
}

coordinate_groups <- split(
  results,
  interaction(results$scenario, results$method, results$coordinate,
              drop = TRUE, lex.order = TRUE)
)
coordinate_summary <- do.call(rbind, lapply(coordinate_groups,
                                             summarize_coordinate))
rownames(coordinate_summary) <- NULL

# Pool binary coverage from replication-level events, not rounded summaries.
method_summary <- do.call(rbind, lapply(
  split(coordinate_summary,
        interaction(coordinate_summary$scenario, coordinate_summary$method,
                    drop = TRUE, lex.order = TRUE)),
  function(x) data.frame(
    scenario = x$scenario[[1L]],
    scenario_class = x$scenario_class[[1L]],
    departure_pattern = x$departure_pattern[[1L]],
    radius_ratio = x$radius_ratio[[1L]],
    is_null = x$is_null[[1L]],
    is_least_favourable = x$is_least_favourable[[1L]],
    is_blind = x$is_blind[[1L]],
    is_assignment_error = x$is_assignment_error[[1L]],
    within_sensitivity_ball = x$within_sensitivity_ball[[1L]],
    method = x$method[[1L]],
    coordinates = nrow(x),
    trace_mse_per_coordinate = mean_or_na(x$empirical_mse),
    rms_standardized_bias = sqrt(mean_or_na(x$standardized_bias^2)),
    minimum_variance_calibration = min_or_na(x$sandwich_variance_calibration),
    maximum_variance_calibration = max_or_na(x$sandwich_variance_calibration),
    minimum_coordinate_naive_coverage = min_or_na(x$naive_coverage),
    maximum_coordinate_naive_coverage = max_or_na(x$naive_coverage),
    minimum_bias_aware_coverage_lcb = min_or_na(x$bias_aware_coverage_lcb_95),
    stringsAsFactors = FALSE
  )
))
rownames(method_summary) <- NULL

if ("naive_covered" %in% names(results)) {
  pooled <- aggregate(
    as.numeric(naive_covered) ~ scenario + method,
    data = results,
    FUN = function(x) mean(x, na.rm = TRUE)
  )
  names(pooled)[[3L]] <- "pooled_naive_coverage"
  method_summary <- merge(method_summary, pooled,
                          by = c("scenario", "method"), all.x = TRUE,
                          sort = FALSE)
} else {
  method_summary$pooled_naive_coverage <- NA_real_
}

# ---- Hausman/pretest and family summaries --------------------------------

hausman_summary <- data.frame()
hausman <- data.frame()
hausman_source <- results
if (!is.null(bundle$diagnostics)) {
  hausman_source <- as.data.frame(bundle$diagnostics)
  hausman_source <- rename_first(hausman_source, "hausman_variant",
                                 c("diagnostic", "hausman_type"))
  hausman_source <- rename_first(hausman_source, "hausman_reject",
                                 c("rejected_0_05", "hausman_rejected"))
  hausman_source <- rename_first(hausman_source,
                                 "theoretical_hausman_power",
                                 c("predicted_power", "hausman_power_theory"))
  hausman_source$replication <- as.character(hausman_source$replication)
  hausman_source$coordinate <- as.character(hausman_source$coordinate)
  hausman_source$hausman_variant <- as.character(hausman_source$hausman_variant)
  hausman_source$hausman_reject <-
    as_logical_safe(hausman_source$hausman_reject)
  hausman_source$theoretical_hausman_power <- suppressWarnings(as.numeric(
    as.character(hausman_source$theoretical_hausman_power)
  ))
  if ("stable_rank" %in% names(hausman_source)) {
    hausman_source$stable_rank <- as_logical_safe(hausman_source$stable_rank)
  }
  scenario_meta <- unique(results[c(
    "scenario", "scenario_class", "departure_pattern", "radius_ratio",
    "is_null", "is_blind"
  )])
  diagnostic_new_fields <- setdiff(names(scenario_meta), names(hausman_source))
  if (length(diagnostic_new_fields)) {
    hausman_source <- merge(
      hausman_source, scenario_meta[c("scenario", diagnostic_new_fields)],
      by = "scenario", all.x = TRUE, sort = FALSE
    )
  }
}
if ("hausman_reject" %in% names(hausman_source) &&
    any(!is.na(hausman_source$hausman_reject))) {
  hcols <- intersect(c("scenario", "scenario_class", "departure_pattern",
                       "radius_ratio", "is_null", "is_blind", "replication",
                       "coordinate", "hausman_reject",
                       "theoretical_hausman_power", "hausman_rank",
                       "hausman_population_rank", "hausman_variant",
                       "stable_rank"), names(hausman_source))
  hausman <- unique(hausman_source[hcols])
  hkey <- c("scenario", "replication", "coordinate", "hausman_variant")
  hkey <- hkey[hkey %in% names(hausman)]
  hdup <- duplicated(hausman[hkey]) | duplicated(hausman[hkey], fromLast = TRUE)
  if (any(hdup)) {
    duplicated_groups <- split(
      hausman[hdup, , drop = FALSE],
      do.call(interaction, c(hausman[hdup, hkey, drop = FALSE],
                             list(drop = TRUE)))
    )
    inconsistent <- vapply(duplicated_groups, function(x) {
      reject_conflict <- length(unique(x$hausman_reject[!is.na(x$hausman_reject)])) > 1L
      power_conflict <- "theoretical_hausman_power" %in% names(x) &&
        length(unique(x$theoretical_hausman_power[
          !is.na(x$theoretical_hausman_power)])) > 1L
      reject_conflict || power_conflict
    }, logical(1))
    if (any(inconsistent)) {
      stop("Conflicting Hausman diagnostics are repeated across method rows. ",
           "Supply hausman_variant to distinguish them.", call. = FALSE)
    }
    hausman <- hausman[!duplicated(hausman[hkey]), , drop = FALSE]
  }
  hausman_groups <- split(
    hausman,
    interaction(hausman$scenario, hausman$coordinate, hausman$hausman_variant,
                drop = TRUE, lex.order = TRUE)
  )
  hausman_summary <- do.call(rbind, lapply(hausman_groups, function(x) {
    power <- if ("theoretical_hausman_power" %in% names(x))
      mean_or_na(x$theoretical_hausman_power) else NA_real_
    rejection <- mean_or_na(as.numeric(x$hausman_reject))
    rank_stability <- if (all(c("hausman_rank", "hausman_population_rank") %in%
                              names(x))) {
      mean_or_na(as.numeric(x$hausman_rank == x$hausman_population_rank))
    } else if ("stable_rank" %in% names(x)) {
      mean_or_na(as.numeric(as_logical_safe(x$stable_rank)))
    } else NA_real_
    data.frame(
      scenario = x$scenario[[1L]],
      scenario_class = x$scenario_class[[1L]],
      departure_pattern = x$departure_pattern[[1L]],
      radius_ratio = x$radius_ratio[[1L]],
      is_null = x$is_null[[1L]],
      is_blind = x$is_blind[[1L]],
      coordinate = x$coordinate[[1L]],
      hausman_variant = x$hausman_variant[[1L]],
      replications = length(unique(x$replication)),
      rejection_rate = rejection,
      theoretical_power = power,
      absolute_power_error = abs(rejection - power),
      rank_stability = rank_stability,
      stringsAsFactors = FALSE
    )
  }))
  rownames(hausman_summary) <- NULL
}

family_data <- bundle$family_results
if (is.null(family_data) && "bias_aware_family_covered" %in% names(results)) {
  family_data <- results[intersect(
    c("scenario", "scenario_class", "radius_ratio", "within_sensitivity_ball",
      "replication", "method", "bias_aware_family_covered"), names(results)
  )]
}
family_summary <- data.frame()
if (!is.null(family_data)) {
  family_data <- normalize_aliases(as.data.frame(family_data))
  family_data <- rename_first(family_data, "bias_aware_family_covered",
                              c("family_covered", "simultaneous_covered"))
  if (all(c("scenario", "replication", "method",
            "bias_aware_family_covered") %in% names(family_data))) {
    family_data$bias_aware_family_covered <-
      as_logical_safe(family_data$bias_aware_family_covered)
    family_data <- unique(family_data[c("scenario", "replication", "method",
                                       "bias_aware_family_covered")])
    family_groups <- split(
      family_data,
      interaction(family_data$scenario, family_data$method,
                  drop = TRUE, lex.order = TRUE)
    )
    family_summary <- do.call(rbind, lapply(family_groups, function(x) {
      covered <- x$bias_aware_family_covered
      data.frame(
        scenario = as.character(x$scenario[[1L]]),
        method = as.character(x$method[[1L]]),
        replications = length(unique(x$replication)),
        family_coverage = mean_or_na(as.numeric(covered)),
        family_coverage_lcb_95 = coverage_lcb(covered),
        stringsAsFactors = FALSE
      )
    }))
    scenario_meta <- unique(results[c("scenario", "scenario_class",
                                      "radius_ratio", "within_sensitivity_ball")])
    family_summary <- merge(family_summary, scenario_meta, by = "scenario",
                            all.x = TRUE, sort = FALSE)
  }
}

# ---- Role comparisons and bootstrap --------------------------------------

role_method <- function(role) unname(method_roles[[role]])
get_role_rows <- function(role) {
  method <- role_method(role)
  if (is.na(method)) return(coordinate_summary[FALSE, ])
  coordinate_summary[coordinate_summary$method == method, , drop = FALSE]
}

ordinary_rows <- get_role_rows("ordinary")
comparison_base <- if (nrow(ordinary_rows)) ordinary_rows[c(
  "scenario", "scenario_class", "departure_pattern", "radius_ratio",
  "is_null", "is_least_favourable", "is_blind", "is_assignment_error",
  "within_sensitivity_ball", "coordinate", "empirical_mse",
  "naive_interval_length"
)] else data.frame()
if (nrow(comparison_base)) {
  names(comparison_base)[names(comparison_base) == "empirical_mse"] <-
    "ordinary_mse"
  names(comparison_base)[names(comparison_base) == "naive_interval_length"] <-
    "ordinary_interval_length"
}

for (role in c("restricted", "selected", "oracle_blend", "feasible_blend")) {
  x <- get_role_rows(role)
  if (!nrow(x) || !nrow(comparison_base)) next
  keep <- x[c("scenario", "coordinate", "empirical_mse",
              "bias_aware_interval_length", "mean_line_oracle_mse",
              "mse_theory_relative_error")]
  names(keep)[-(1:2)] <- paste0(role, "_", names(keep)[-(1:2)])
  comparison_base <- merge(comparison_base, keep,
                           by = c("scenario", "coordinate"), all.x = TRUE,
                           sort = FALSE)
}
blend_summary <- comparison_base
if (nrow(blend_summary)) {
  for (role in c("restricted", "selected", "oracle_blend", "feasible_blend")) {
    mse <- paste0(role, "_empirical_mse")
    if (mse %in% names(blend_summary)) {
      blend_summary[[paste0(role, "_to_ordinary_risk_ratio")]] <-
        blend_summary[[mse]] / blend_summary$ordinary_mse
    }
  }
  if (all(c("feasible_blend_empirical_mse", "oracle_blend_mean_line_oracle_mse") %in%
          names(blend_summary))) {
    denominator <- blend_summary$ordinary_mse -
      blend_summary$oracle_blend_mean_line_oracle_mse
    blend_summary$feasible_oracle_gain_recovered <-
      (blend_summary$ordinary_mse - blend_summary$feasible_blend_empirical_mse) /
      denominator
    blend_summary$feasible_oracle_gain_recovered[abs(denominator) < 1e-12] <- NA
  }
  if ("oracle_blend_mean_line_oracle_mse" %in% names(blend_summary)) {
    blend_summary$oracle_blend_line_mse_relative_error <- abs(
      blend_summary$oracle_blend_empirical_mse -
        blend_summary$oracle_blend_mean_line_oracle_mse
    ) / blend_summary$oracle_blend_mean_line_oracle_mse
  }
  if ("feasible_blend_mean_line_oracle_mse" %in% names(blend_summary)) {
    blend_summary$feasible_blend_line_mse_relative_error <- abs(
      blend_summary$feasible_blend_empirical_mse -
        blend_summary$feasible_blend_mean_line_oracle_mse
    ) / blend_summary$feasible_blend_mean_line_oracle_mse
  }
  if ("feasible_blend_bias_aware_interval_length" %in% names(blend_summary)) {
    blend_summary$feasible_bias_aware_to_ordinary_length_ratio <-
      blend_summary$feasible_blend_bias_aware_interval_length /
      blend_summary$ordinary_interval_length
  }
}

bootstrap_draws <- as.integer(config$monte_carlo_bootstrap_draws %||%
                                rules$monte_carlo_bootstrap_draws)
bootstrap_seed <- as.integer(config$monte_carlo_bootstrap_seed %||%
                               rules$monte_carlo_bootstrap_seed)
set.seed(bootstrap_seed)

paired_replication_data <- function(numerator_method, denominator_method,
                                    numerator_field, denominator_field,
                                    scenario_filter) {
  if (is.na(numerator_method) || is.na(denominator_method)) return(data.frame())
  fields <- c("scenario", "replication", "coordinate", "method",
              "estimate", "truth", numerator_field, denominator_field)
  fields <- unique(fields[fields %in% names(results)])
  num <- results[results$method == numerator_method & scenario_filter,
                 fields, drop = FALSE]
  den <- results[results$method == denominator_method & scenario_filter,
                 fields, drop = FALSE]
  if (!nrow(num) || !nrow(den)) return(data.frame())
  num_value <- num[[numerator_field]]
  den_value <- den[[denominator_field]]
  if (identical(numerator_field, "squared_error"))
    num_value <- (num$estimate - num$truth)^2
  if (identical(denominator_field, "squared_error"))
    den_value <- (den$estimate - den$truth)^2
  num <- data.frame(num[c("scenario", "replication", "coordinate")],
                    numerator = num_value)
  den <- data.frame(den[c("scenario", "replication", "coordinate")],
                    denominator = den_value)
  merge(num, den, by = c("scenario", "replication", "coordinate"),
        all = FALSE, sort = FALSE)
}

bootstrap_ratio <- function(data, draws) {
  data <- data[is.finite(data$numerator) & is.finite(data$denominator), ]
  if (!nrow(data) || mean(data$denominator) <= 0) {
    return(list(point = NA_real_, lower = NA_real_, upper = NA_real_,
                draws = numeric()))
  }
  scenarios <- unique(data$scenario)
  point <- mean(data$numerator) / mean(data$denominator)
  # Collapse coordinates before resampling.  This is equivalent to a
  # respondent/replication bootstrap when every replication contains the same
  # coordinate family, and avoids repeatedly rebuilding a large long table.
  collapsed <- aggregate(cbind(numerator, denominator) ~ scenario + replication,
                         data, sum)
  boot <- numeric(draws)
  for (bb in seq_len(draws)) {
    totals <- lapply(scenarios, function(scenario) {
      x <- collapsed[collapsed$scenario == scenario, , drop = FALSE]
      index <- sample.int(nrow(x), nrow(x), replace = TRUE)
      c(sum(x$numerator[index]), sum(x$denominator[index]))
    })
    totals <- do.call(rbind, totals)
    boot[[bb]] <- sum(totals[, 1L]) / sum(totals[, 2L])
  }
  list(
    point = point,
    lower = unname(stats::quantile(boot, 0.025, type = 8, na.rm = TRUE)),
    upper = unname(stats::quantile(boot, 0.975, type = 8, na.rm = TRUE)),
    draws = boot
  )
}

feasible_blend_method <- role_method("feasible_blend")
ordinary_method <- role_method("ordinary")
strong_filter <- !is.na(results$radius_ratio) &
  results$radius_ratio > 0 & results$radius_ratio <= rules$strong_radius_max &
  !results$is_assignment_error
large_filter <- !is.na(results$radius_ratio) &
  results$radius_ratio >= rules$large_departure_radius_min &
  !results$is_assignment_error
half_filter <- !is.na(results$radius_ratio) &
  abs(results$radius_ratio - 0.5) < 1e-8 & !results$is_assignment_error
unit_filter <- !is.na(results$radius_ratio) &
  abs(results$radius_ratio - 1) < 1e-8 & !results$is_assignment_error

# `squared_error` is a virtual field computed by paired_replication_data().
strong_risk_boot <- bootstrap_ratio(paired_replication_data(
  feasible_blend_method, ordinary_method, "squared_error", "squared_error",
  strong_filter
), bootstrap_draws)
large_risk_boot <- bootstrap_ratio(paired_replication_data(
  feasible_blend_method, ordinary_method, "squared_error", "squared_error",
  large_filter
), bootstrap_draws)
half_length_boot <- bootstrap_ratio(paired_replication_data(
  feasible_blend_method, ordinary_method, "bias_aware_length", "naive_length",
  half_filter
), bootstrap_draws)
unit_length_boot <- bootstrap_ratio(paired_replication_data(
  feasible_blend_method, ordinary_method, "bias_aware_length", "naive_length",
  unit_filter
), bootstrap_draws)

# ---- Completion and calibration ------------------------------------------

expected_replications <- config$replications_per_scenario %||%
  config$simulation$replications_per_scenario %||% NA_integer_
expected_replications <- as.integer(expected_replications)
completed_observed <- aggregate(replication ~ scenario + method + coordinate,
                                results, function(x) length(unique(x)))
names(completed_observed)[names(completed_observed) == "replication"] <-
  "completed_replications"

configured_scenarios <- config$expected_scenarios %||%
  config$simulation$expected_scenarios
scenarios_declared <- !is.null(configured_scenarios) ||
  (nrow(metadata) && "scenario" %in% names(metadata))
expected_scenarios <- if (!is.null(configured_scenarios)) {
  unique(as.character(unlist(configured_scenarios, use.names = FALSE)))
} else if (nrow(metadata) && "scenario" %in% names(metadata)) {
  unique(as.character(metadata$scenario))
} else unique(results$scenario)
configured_methods <- config$expected_methods %||%
  config$simulation$expected_methods
runner_method_alias <- c(
  oracle_tau_1_00 = "oracle_adaptive",
  selected_adaptive_wrong_assignment = "selected_adaptive"
)
reported_method_id <- function(x) {
  x <- as.character(x)
  replace <- x %in% names(runner_method_alias)
  x[replace] <- unname(runner_method_alias[x[replace]])
  x
}
if (is.null(configured_methods) && !is.null(config$methods)) {
  configured_methods <- vapply(config$methods, function(x) {
    if (is.list(x) && !is.null(x$id)) as.character(x$id) else as.character(x)
  }, character(1))
  configured_methods <- reported_method_id(configured_methods)
  configured_methods <- unique(configured_methods)
}
methods_declared <- !is.null(configured_methods) ||
  any(!is.na(method_roles))
expected_methods <- if (!is.null(configured_methods)) {
  unique(as.character(unlist(configured_methods, use.names = FALSE)))
} else {
  roles <- unname(method_roles[!is.na(method_roles)])
  if (length(roles)) unique(roles) else unique(results$method)
}
configured_coordinates <- config$expected_coordinates %||%
  config$simulation$expected_coordinates %||% config$estimand$coordinates
coordinates_declared <- !is.null(configured_coordinates)
expected_coordinates <- if (!is.null(configured_coordinates)) {
  unique(as.character(unlist(configured_coordinates, use.names = FALSE)))
} else unique(results$coordinate)
# Respect the frozen method applicability declarations.  In particular, the
# assignment-law controls and rescue estimators are not required in ordinary
# correct-law cells merely because the runner also emits them there.
expected_grid <- data.frame()
if (!is.null(config$methods) && nrow(metadata) &&
    all(c("scenario", "scenario_class") %in% names(metadata))) {
  assignment_scenarios <- as.character(metadata$scenario[
    grepl("^assignment_error", as.character(metadata$scenario_class))
  ])
  method_grids <- lapply(config$methods, function(definition) {
    method_id <- if (is.list(definition)) definition$id else definition
    method_id <- reported_method_id(method_id)
    applies_to <- if (is.list(definition)) definition$applies_to else NULL
    applicable_scenarios <- if (!is.null(applies_to) &&
                                identical(as.character(applies_to),
                                          "assignment_error")) {
      assignment_scenarios
    } else {
      expected_scenarios
    }
    expand.grid(
      scenario = applicable_scenarios,
      method = method_id,
      coordinate = expected_coordinates,
      stringsAsFactors = FALSE
    )
  })
  expected_grid <- unique(do.call(rbind, method_grids))
} else {
  expected_grid <- expand.grid(
    scenario = expected_scenarios,
    method = expected_methods,
    coordinate = expected_coordinates,
    stringsAsFactors = FALSE
  )
}
completed <- merge(expected_grid, completed_observed,
                   by = c("scenario", "method", "coordinate"),
                   all.x = TRUE, sort = FALSE)
completed$completed_replications[is.na(completed$completed_replications)] <- 0L
completed$expected_replications <- expected_replications
completed$complete <- if (is.finite(expected_replications))
  completed$completed_replications == expected_replications else NA

failures <- bundle$failures
failure_count <- if (is.null(failures)) 0L else if (is.data.frame(failures))
  nrow(failures) else length(failures)
scenario_count <- length(expected_scenarios)
failure_denominator <- if (is.finite(expected_replications))
  scenario_count * expected_replications else NA_real_
failure_rate <- failure_count / failure_denominator
# Limit integrity checks to protocol-applicable rows; extra descriptive methods
# emitted by the runner cannot create or conceal a gate result.
required_results <- merge(
  results, transform(expected_grid, protocol_required_row = TRUE),
  by = c("scenario", "method", "coordinate"), all = FALSE, sort = FALSE
)
maximum_calibration <- if ("calibration_residual" %in% names(required_results))
  max_or_na(abs(required_results$calibration_residual)) else NA_real_
essential_numeric <- c("estimate", "truth", "standard_error",
                       "predicted_bias", "predicted_mse",
                       "calibration_residual")
essential_schema_available <- all(essential_numeric %in% names(results)) &&
  "naive_covered" %in% names(results)
invalid_result_row <- rep(FALSE, nrow(required_results))
if (essential_schema_available) {
  for (field in essential_numeric) {
    invalid_result_row <- invalid_result_row |
      !is.finite(required_results[[field]])
  }
  invalid_result_row <- invalid_result_row |
    is.na(required_results$naive_covered)
} else {
  invalid_result_row[] <- TRUE
}
invalid_required_result_rows <- sum(invalid_result_row)

expected_replication_ids <- if (is.finite(expected_replications)) {
  seq_len(expected_replications)
} else integer()
replication_id_groups <- split(
  required_results$replication,
  interaction(required_results$scenario, required_results$method,
              required_results$coordinate, drop = TRUE)
)
replication_ids_complete <- length(replication_id_groups) == nrow(expected_grid) &&
  length(expected_replication_ids) > 0L && all(vapply(
    replication_id_groups,
    function(x) identical(sort(unique(as.integer(as.character(x)))),
                          expected_replication_ids),
    logical(1)
  ))
common_seed_complete <- "seed" %in% names(required_results) &&
  all(vapply(
    split(required_results$seed,
          interaction(required_results$scenario, required_results$replication,
                      drop = TRUE)),
    function(x) length(unique(x[!is.na(x)])) == 1L,
    logical(1)
  ))

completion_summary <- completed
completion_summary$reported_numerical_failures <- failure_count
completion_summary$reported_numerical_failure_rate <- failure_rate
completion_summary$maximum_calibration_residual <- maximum_calibration
completion_summary$invalid_required_result_rows <- invalid_required_result_rows
completion_summary$replication_ids_complete <- replication_ids_complete
completion_summary$common_seed_complete <- common_seed_complete

# ---- Gate components ------------------------------------------------------

components <- list()
add_component <- function(gate, component, required, available, passed,
                          value = NA_real_, criterion = "") {
  components[[length(components) + 1L]] <<- data.frame(
    gate = gate,
    component = component,
    required = isTRUE(required),
    available = isTRUE(available),
    passed = isTRUE(available) && isTRUE(passed),
    value = if (length(value)) as.character(value[[1L]]) else NA_character_,
    criterion = criterion,
    stringsAsFactors = FALSE
  )
}

# Exact-bias and MSE theory agreement are evaluated only where predictions
# were supplied.  In production at least one row from each family is required.
bias_rows <- coordinate_summary[is.finite(coordinate_summary$predicted_bias), ]
bias_available <- nrow(bias_rows) > 0L
bias_pass <- bias_available && all_or_false(
  bias_rows$bias_prediction_ci_99_agreement &
    bias_rows$bias_prediction_standardized_error <=
      rules$bias_prediction_standardized_error_max
)
add_component("hard", "exact_bias_agreement", TRUE, bias_available, bias_pass,
              max_or_na(bias_rows$bias_prediction_standardized_error),
              paste0("all analytic biases inside ",
                     100 * rules$bias_agreement_ci_level,
                     "% MC intervals and max standardized error <= ",
                     rules$bias_prediction_standardized_error_max))

mse_rows <- coordinate_summary[is.finite(coordinate_summary$predicted_mse), ]
mse_available <- nrow(mse_rows) > 0L
mse_pass <- mse_available && all(mse_rows$mse_theory_relative_error <=
                                  rules$mse_theory_relative_error_max)
add_component("hard", "mse_theory_agreement", TRUE, mse_available, mse_pass,
              max_or_na(mse_rows$mse_theory_relative_error),
              paste0("max relative MSE error <= ",
                     rules$mse_theory_relative_error_max))

sharpness_required <- isTRUE(rules$require_sharpness_check)
oracle_sharpness_rows <- coordinate_summary[
  FALSE, ]
if (!is.na(role_method("restricted"))) {
  oracle_sharpness_rows <- coordinate_summary[
    coordinate_summary$is_least_favourable %in% TRUE &
      coordinate_summary$method == role_method("restricted") &
      is.finite(coordinate_summary$bound_sharpness_ratio), ]
}
oracle_sharpness_available <- nrow(oracle_sharpness_rows) > 0L
oracle_sharpness_pass <- oracle_sharpness_available && all(
  oracle_sharpness_rows$bound_sharpness_ratio >= rules$sharpness_oracle_min &
    oracle_sharpness_rows$bound_sharpness_ratio <= rules$sharpness_oracle_max
)
add_component("hard", "oracle_bound_sharpness", sharpness_required,
              oracle_sharpness_available, oracle_sharpness_pass,
              paste(min_or_na(oracle_sharpness_rows$bound_sharpness_ratio),
                    max_or_na(oracle_sharpness_rows$bound_sharpness_ratio),
                    sep = " to "),
              paste0("least-favourable |bias|/bound in [",
                     rules$sharpness_oracle_min, ", ",
                     rules$sharpness_oracle_max, "]"))

feasible_sharpness_methods <- na.omit(c(role_method("selected"),
                                        role_method("feasible_blend")))
feasible_sharpness_rows <- coordinate_summary[
  coordinate_summary$is_least_favourable %in% TRUE &
    coordinate_summary$method %in% feasible_sharpness_methods &
    is.finite(coordinate_summary$bound_sharpness_ratio), ]
feasible_sharpness_available <- nrow(feasible_sharpness_rows) > 0L
feasible_sharpness_pass <- feasible_sharpness_available && all(
  feasible_sharpness_rows$bound_sharpness_ratio >=
    rules$sharpness_feasible_min &
    feasible_sharpness_rows$bound_sharpness_ratio <=
      rules$sharpness_feasible_max
)
add_component("hard", "feasible_bound_sharpness", sharpness_required,
              feasible_sharpness_available, feasible_sharpness_pass,
              paste(min_or_na(feasible_sharpness_rows$bound_sharpness_ratio),
                    max_or_na(feasible_sharpness_rows$bound_sharpness_ratio),
                    sep = " to "),
              paste0("least-favourable |bias|/bound in [",
                     rules$sharpness_feasible_min, ", ",
                     rules$sharpness_feasible_max, "]"))

ordinary_method_rows <- method_summary[FALSE, ]
ordinary_coord_rows <- coordinate_summary[FALSE, ]
if (!is.na(ordinary_method)) {
  ordinary_method_rows <- method_summary[
    method_summary$method == ordinary_method &
      !(method_summary$is_assignment_error %in% TRUE), ]
  ordinary_coord_rows <- coordinate_summary[
    coordinate_summary$method == ordinary_method &
      !coordinate_summary$is_assignment_error, ]
}
ordinary_coverage_available <- nrow(ordinary_coord_rows) > 0L &&
  any(is.finite(ordinary_coord_rows$naive_coverage)) &&
  any(is.finite(ordinary_method_rows$pooled_naive_coverage))
ordinary_coverage_pass <- ordinary_coverage_available && all(
  ordinary_coord_rows$naive_coverage >= rules$ordinary_coordinate_coverage_min &
    ordinary_coord_rows$naive_coverage <= rules$ordinary_coordinate_coverage_max
) && all(
  ordinary_method_rows$pooled_naive_coverage >=
    rules$ordinary_pooled_coverage_min &
    ordinary_method_rows$pooled_naive_coverage <=
      rules$ordinary_pooled_coverage_max
)
add_component("hard", "ordinary_coverage", TRUE,
              ordinary_coverage_available, ordinary_coverage_pass,
              paste(min_or_na(ordinary_coord_rows$naive_coverage),
                    max_or_na(ordinary_coord_rows$naive_coverage), sep = " to "),
              paste0("coordinate [", rules$ordinary_coordinate_coverage_min,
                     ", ", rules$ordinary_coordinate_coverage_max,
                     "], pooled [", rules$ordinary_pooled_coverage_min, ", ",
                     rules$ordinary_pooled_coverage_max, "]"))

ordinary_validity_available <- nrow(ordinary_method_rows) > 0L &&
  all(is.finite(ordinary_method_rows$rms_standardized_bias)) &&
  all(is.finite(ordinary_method_rows$minimum_variance_calibration))
ordinary_validity_pass <- ordinary_validity_available && all(
  ordinary_method_rows$rms_standardized_bias <=
    rules$ordinary_rms_standardized_bias_max &
  ordinary_method_rows$minimum_variance_calibration >=
    rules$sandwich_variance_calibration_min &
  ordinary_method_rows$maximum_variance_calibration <=
    rules$sandwich_variance_calibration_max
)
add_component("hard", "ordinary_bias_and_variance_calibration", TRUE,
              ordinary_validity_available, ordinary_validity_pass,
              max_or_na(ordinary_method_rows$rms_standardized_bias),
              paste0("RMS standardized bias <= ",
                     rules$ordinary_rms_standardized_bias_max,
                     "; sandwich/MC variance in [",
                     rules$sandwich_variance_calibration_min, ", ",
                     rules$sandwich_variance_calibration_max, "]"))

null_methods <- na.omit(c(role_method("restricted"), role_method("selected")))
null_rows <- coordinate_summary[
  coordinate_summary$method %in% null_methods &
    coordinate_summary$is_null %in% TRUE, ]
null_coverage_available <- length(null_methods) > 0L && nrow(null_rows) > 0L &&
  all(is.finite(null_rows$naive_coverage))
null_coverage_pass <- null_coverage_available && all(
  null_rows$naive_coverage >= rules$null_restricted_coordinate_coverage_min &
    null_rows$naive_coverage <= rules$null_restricted_coordinate_coverage_max
)
add_component("hard", "restricted_null_coverage", TRUE,
              null_coverage_available, null_coverage_pass,
              paste(min_or_na(null_rows$naive_coverage),
                    max_or_na(null_rows$naive_coverage), sep = " to "),
              paste0("null coordinate coverage in [",
                     rules$null_restricted_coordinate_coverage_min, ", ",
                     rules$null_restricted_coordinate_coverage_max, "]"))

aware_methods <- na.omit(c(role_method("restricted"), role_method("selected"),
                           role_method("oracle_blend"),
                           role_method("feasible_blend")))
aware_rows <- coordinate_summary[
  coordinate_summary$method %in% aware_methods &
    coordinate_summary$within_sensitivity_ball %in% TRUE, ]
aware_available <- length(aware_methods) > 0L && nrow(aware_rows) > 0L &&
  all(is.finite(aware_rows$bias_aware_coverage_lcb_95))
aware_pass <- aware_available && all(
  aware_rows$bias_aware_coverage_lcb_95 >= rules$bias_aware_coverage_lcb_min
)
add_component("hard", "bias_aware_pointwise_coverage", TRUE,
              aware_available, aware_pass,
              min_or_na(aware_rows$bias_aware_coverage_lcb_95),
              paste0("minimum one-sided 95% coverage LCB >= ",
                     rules$bias_aware_coverage_lcb_min))

family_required <- isTRUE(rules$require_family_coverage)
family_in_ball <- if (nrow(family_summary)) family_summary[
  family_summary$within_sensitivity_ball %in% TRUE &
    family_summary$method %in% aware_methods, ] else data.frame()
family_available <- nrow(family_in_ball) > 0L &&
  all(is.finite(family_in_ball$family_coverage_lcb_95))
family_pass <- family_available && all(
  family_in_ball$family_coverage_lcb_95 >= rules$bias_aware_coverage_lcb_min
)
add_component("hard", "bias_aware_family_coverage", family_required,
              family_available, family_pass,
              if (family_available) min(family_in_ball$family_coverage_lcb_95)
              else NA_real_,
              paste0("minimum one-sided 95% family-coverage LCB >= ",
                     rules$bias_aware_coverage_lcb_min))

hausman_required <- isTRUE(rules$require_hausman_check)
hnull <- if (nrow(hausman_summary)) hausman_summary[
  hausman_summary$is_null %in% TRUE, ] else data.frame()
hpower <- if (nrow(hausman_summary)) hausman_summary[
  !(hausman_summary$is_null %in% TRUE) & !(hausman_summary$is_blind %in% TRUE),
  ] else data.frame()
hblind <- if (nrow(hausman_summary)) hausman_summary[
  hausman_summary$is_blind %in% TRUE, ] else data.frame()
hnull_available <- nrow(hnull) > 0L
hnull_pass <- hnull_available && all(
  hnull$rejection_rate >= rules$hausman_null_size_min &
    hnull$rejection_rate <= rules$hausman_null_size_max
)
add_component("hard", "hausman_null_size", hausman_required,
              hnull_available, hnull_pass,
              paste(min_or_na(hnull$rejection_rate),
                    max_or_na(hnull$rejection_rate), sep = " to "),
              paste0("null rejection in [", rules$hausman_null_size_min,
                     ", ", rules$hausman_null_size_max, "]"))

hpower_available <- nrow(hpower) > 0L &&
  all(is.finite(hpower$absolute_power_error))
hpower_threshold <- if (nrow(hpower)) ifelse(
  grepl("oracle", tolower(hpower$hausman_variant)),
  rules$hausman_power_absolute_error_oracle_max,
  rules$hausman_power_absolute_error_feasible_max
) else numeric()
hpower_pass <- hpower_available && all(hpower$absolute_power_error <=
                                         hpower_threshold)
add_component("hard", "hausman_power_calibration", hausman_required,
              hpower_available, hpower_pass,
              max_or_na(hpower$absolute_power_error),
              paste0("max absolute power error <= ",
                     rules$hausman_power_absolute_error_oracle_max,
                     " (oracle) or ",
                     rules$hausman_power_absolute_error_feasible_max,
                     " (feasible)"))

hblind_available <- nrow(hblind) > 0L
hblind_pass <- hblind_available && all(
  hblind$rejection_rate <= rules$hausman_blind_rejection_max
)
add_component("hard", "hausman_blind_direction", hausman_required,
              hblind_available, hblind_pass,
              max_or_na(hblind$rejection_rate),
              paste0("max rejection <= ", rules$hausman_blind_rejection_max))

assignment_required <- isTRUE(rules$require_assignment_check)
assignment_rows <- coordinate_summary[
  coordinate_summary$is_assignment_error %in% TRUE &
    is.finite(coordinate_summary$assignment_predicted_bias), ]
assignment_available <- nrow(assignment_rows) > 0L
assignment_pass <- assignment_available && all_or_false(
  assignment_rows$assignment_bias_ci_99_agreement &
    assignment_rows$assignment_bias_standardized_error <=
      rules$assignment_bias_standardized_error_max
)
add_component("hard", "assignment_error_formula", assignment_required,
              assignment_available, assignment_pass,
              max_or_na(assignment_rows$assignment_bias_standardized_error),
              paste0("all assignment biases inside ",
                     100 * rules$bias_agreement_ci_level,
                     "% MC intervals and standardized error <= ",
                     rules$assignment_bias_standardized_error_max))

calibration_required <- isTRUE(rules$require_calibration_check)
calibration_available <- is.finite(maximum_calibration)
calibration_pass <- calibration_available &&
  maximum_calibration <= rules$maximum_calibration_residual
add_component("hard", "exact_calibration", calibration_required,
              calibration_available, calibration_pass, maximum_calibration,
              paste0("max residual <= ", rules$maximum_calibration_residual))

completion_available <- is.finite(expected_replications) &&
  is.finite(failure_rate) && scenarios_declared && methods_declared &&
  coordinates_declared && essential_schema_available && nrow(required_results) > 0L
completion_pass <- completion_available && all(completed$complete %in% TRUE) &&
  failure_rate <= rules$maximum_numerical_failure_rate &&
  invalid_required_result_rows == 0L && replication_ids_complete &&
  common_seed_complete
add_component("hard", "numerical_completion", TRUE,
              completion_available, completion_pass, failure_rate,
              paste0("scenario/method/coordinate grid declared; all cells ",
                     "complete with common seeds/replication IDs; failure rate <= ",
                     rules$maximum_numerical_failure_rate))

blend_required <- isTRUE(rules$require_blend_check)
oracle_blend_rows <- get_role_rows("oracle_blend")
oracle_blend_available <- nrow(oracle_blend_rows) > 0L &&
  any(is.finite(oracle_blend_rows$mean_line_oracle_mse))
oracle_blend_error <- if (oracle_blend_available) max_or_na(abs(
  oracle_blend_rows$empirical_mse - oracle_blend_rows$mean_line_oracle_mse
) / oracle_blend_rows$mean_line_oracle_mse) else NA_real_
add_component("hard", "oracle_blend_theory", blend_required,
              oracle_blend_available,
              oracle_blend_available &&
                oracle_blend_error <= rules$oracle_blend_mse_relative_error_max,
              oracle_blend_error,
              paste0("max relative line-oracle MSE error <= ",
                     rules$oracle_blend_mse_relative_error_max))

feasible_blend_rows <- get_role_rows("feasible_blend")
feasible_blend_available <- nrow(feasible_blend_rows) > 0L &&
  any(is.finite(feasible_blend_rows$predicted_mse))
feasible_blend_error <- if (feasible_blend_available)
  max_or_na(feasible_blend_rows$mse_theory_relative_error) else NA_real_
add_component("hard", "feasible_blend_theory", blend_required,
              feasible_blend_available,
              feasible_blend_available && feasible_blend_error <=
                rules$feasible_blend_mse_relative_error_max,
              feasible_blend_error,
              paste0("max relative feasible-blend MSE error <= ",
                     rules$feasible_blend_mse_relative_error_max))

# Practical gates answer whether robustness buys enough precision to anchor a
# full methods paper.  These do not weaken any hard validity requirement.
strong_cells <- if (nrow(blend_summary)) blend_summary[
  is.finite(blend_summary$radius_ratio) & blend_summary$radius_ratio > 0 &
    blend_summary$radius_ratio <= rules$strong_radius_max &
    !(blend_summary$is_assignment_error %in% TRUE), ] else data.frame()
gain_available <- nrow(strong_cells) > 0L &&
  "feasible_oracle_gain_recovered" %in% names(strong_cells) &&
  any(is.finite(strong_cells$feasible_oracle_gain_recovered))
gain_pass <- gain_available && all(
  strong_cells$feasible_oracle_gain_recovered[
    is.finite(strong_cells$feasible_oracle_gain_recovered)
  ] >= rules$strong_oracle_gain_recovered_min
)
add_component("practical", "strong_oracle_gain_recovered", TRUE,
              gain_available, gain_pass,
              if (gain_available) min_or_na(
                strong_cells$feasible_oracle_gain_recovered) else NA_real_,
              paste0("each strong cell >= ",
                     rules$strong_oracle_gain_recovered_min))

strong_ratio_available <- is.finite(strong_risk_boot$point) &&
  is.finite(strong_risk_boot$upper)
strong_ratio_pass <- strong_ratio_available &&
  strong_risk_boot$point <= rules$strong_aggregate_risk_ratio_max &&
  strong_risk_boot$upper < rules$strong_aggregate_risk_ratio_upper_97_5_max
add_component("practical", "strong_aggregate_blend_risk", TRUE,
              strong_ratio_available, strong_ratio_pass,
              paste(strong_risk_boot$point, strong_risk_boot$upper, sep = "; UCB="),
              paste0("point <= ", rules$strong_aggregate_risk_ratio_max,
                     "; bootstrap UCB < ",
                     rules$strong_aggregate_risk_ratio_upper_97_5_max))

large_cells <- if (nrow(blend_summary) &&
                   "feasible_blend_to_ordinary_risk_ratio" %in%
                     names(blend_summary)) blend_summary[
  is.finite(blend_summary$radius_ratio) &
    blend_summary$radius_ratio >= rules$large_departure_radius_min &
    !(blend_summary$is_assignment_error %in% TRUE), ] else data.frame()
large_available <- nrow(large_cells) > 0L && is.finite(large_risk_boot$point)
large_pass <- large_available &&
  large_risk_boot$point <= rules$large_departure_aggregate_risk_ratio_max &&
  all(large_cells$feasible_blend_to_ordinary_risk_ratio <=
        rules$large_departure_each_risk_ratio_max)
add_component("practical", "large_departure_safety", TRUE,
              large_available, large_pass,
              if (large_available) max_or_na(
                large_cells$feasible_blend_to_ordinary_risk_ratio) else NA_real_,
              paste0("aggregate <= ",
                     rules$large_departure_aggregate_risk_ratio_max,
                     "; each <= ", rules$large_departure_each_risk_ratio_max))

weak_cells <- if (nrow(blend_summary) &&
                  "feasible_blend_to_ordinary_risk_ratio" %in%
                    names(blend_summary)) blend_summary[
  grepl("weak|safety", tolower(blend_summary$scenario_class)), ] else data.frame()
weak_available <- nrow(weak_cells) > 0L
weak_pass <- weak_available && all(
  weak_cells$feasible_blend_to_ordinary_risk_ratio <=
    rules$weak_safety_each_risk_ratio_max
)
add_component("practical", "weak_cell_safety", TRUE,
              weak_available, weak_pass,
              max_or_na(weak_cells$feasible_blend_to_ordinary_risk_ratio),
              paste0("each weak/safety cell <= ",
                     rules$weak_safety_each_risk_ratio_max))

half_cells <- if (nrow(blend_summary) &&
                  "feasible_bias_aware_to_ordinary_length_ratio" %in%
                    names(blend_summary)) blend_summary[
  is.finite(blend_summary$radius_ratio) &
    abs(blend_summary$radius_ratio - 0.5) < 1e-8, ] else data.frame()
half_available <- nrow(half_cells) > 0L && is.finite(half_length_boot$point) &&
  is.finite(half_length_boot$upper)
half_pass <- half_available &&
  geometric_mean(half_cells$feasible_bias_aware_to_ordinary_length_ratio) <=
    rules$half_radius_length_ratio_max &&
  half_length_boot$upper < rules$half_radius_length_ratio_upper_97_5_max
add_component("practical", "half_radius_interval_length", TRUE,
              half_available, half_pass,
              paste(geometric_mean(
                half_cells$feasible_bias_aware_to_ordinary_length_ratio),
                half_length_boot$upper, sep = "; UCB="),
              paste0("geometric mean <= ", rules$half_radius_length_ratio_max,
                     "; bootstrap UCB < ",
                     rules$half_radius_length_ratio_upper_97_5_max))

unit_cells <- if (nrow(blend_summary) &&
                  "feasible_bias_aware_to_ordinary_length_ratio" %in%
                    names(blend_summary)) blend_summary[
  is.finite(blend_summary$radius_ratio) &
    abs(blend_summary$radius_ratio - 1) < 1e-8, ] else data.frame()
unit_available <- nrow(unit_cells) > 0L && is.finite(unit_length_boot$point)
unit_pass <- unit_available && geometric_mean(
  unit_cells$feasible_bias_aware_to_ordinary_length_ratio
) <= rules$unit_radius_length_ratio_max && all(
  unit_cells$feasible_bias_aware_to_ordinary_length_ratio <=
    rules$unit_radius_each_length_ratio_max
)
add_component("practical", "unit_radius_interval_length", TRUE,
              unit_available, unit_pass,
              if (unit_available) max_or_na(
                unit_cells$feasible_bias_aware_to_ordinary_length_ratio)
              else NA_real_,
              paste0("geometric mean <= ", rules$unit_radius_length_ratio_max,
                     "; each <= ", rules$unit_radius_each_length_ratio_max))

# The repository's frozen v1 protocol predates the generic fallback criteria
# above and is more specific.  Rebuild the component list from those frozen
# rules so a production decision cannot silently drift from the protocol.
protocol_risk_summary <- data.frame()
protocol_structured_summary <- data.frame()
if (frozen_protocol) {
  components <- list()
  primary_coordinate <- config$estimand$primary_direction$coordinate %||%
    config$estimand$coordinates[[1L]]

  make_protocol_grid <- function(scenarios, methods,
                                 coordinate = primary_coordinate) {
    expand.grid(
      scenario = unique(as.character(scenarios)),
      method = unique(as.character(methods)),
      coordinate = unique(as.character(coordinate)),
      stringsAsFactors = FALSE
    )
  }
  protocol_summary_cells <- function(summary, grid, metrics) {
    keys <- c("scenario", "method", "coordinate")
    if (!nrow(grid) || !all(c(keys, metrics, "replications") %in%
                            names(summary))) {
      return(list(available = FALSE, rows = data.frame()))
    }
    rows <- merge(grid, summary, by = keys, all.x = TRUE, sort = FALSE)
    complete <- nrow(rows) == nrow(grid) && !anyDuplicated(rows[keys]) &&
      all(rows$replications == expected_replications) &&
      all(vapply(metrics, function(field) {
        all(is.finite(rows[[field]]))
      }, logical(1)))
    list(available = isTRUE(complete), rows = rows)
  }
  protocol_raw_complete <- function(source, grid, fields) {
    keys <- c("scenario", "method", "coordinate")
    required <- c(keys, "replication", fields)
    if (!nrow(grid) || !all(required %in% names(source))) return(FALSE)
    rows <- merge(grid, source, by = keys, all = FALSE, sort = FALSE)
    if (nrow(rows) != nrow(grid) * expected_replications ||
        anyDuplicated(rows[c(keys, "replication")])) return(FALSE)
    valid <- vapply(fields, function(field) {
      value <- rows[[field]]
      if (is.numeric(value) || is.integer(value) || is.logical(value)) {
        all(!is.na(value) & is.finite(value))
      } else {
        all(!is.na(value) & nzchar(as.character(value)))
      }
    }, logical(1))
    id_groups <- split(
      rows$replication,
      interaction(rows$scenario, rows$method, rows$coordinate, drop = TRUE)
    )
    all(valid) && length(id_groups) == nrow(grid) && all(vapply(
      id_groups,
      function(x) identical(sort(unique(as.integer(as.character(x)))),
                            expected_replication_ids),
      logical(1)
    ))
  }
  scenario_ids <- function(condition) {
    as.character(metadata$scenario[which(condition)])
  }
  correct_law <- as.character(metadata$true_assignment) ==
    as.character(metadata$analysis_assignment)
  has_carryover <- as.character(metadata$carryover) != "none"
  assignment_cell <- grepl("^assignment_error",
                           as.character(metadata$scenario_class))

  canonical_path <- file.path(analysis_dir, "gate-e2-production.yml")
  canonical_config <- if (file.exists(canonical_path))
    yaml::read_yaml(canonical_path) else NULL
  project_to_reference <- function(candidate, reference) {
    if (!is.list(reference)) return(candidate)
    if (!is.list(candidate)) return(candidate)
    if (is.null(names(reference))) {
      if (length(candidate) != length(reference)) return(candidate)
      return(Map(project_to_reference, candidate, reference))
    }
    stats::setNames(lapply(names(reference), function(field) {
      project_to_reference(candidate[[field]], reference[[field]])
    }), names(reference))
  }
  embedded_config_matches <- !is.null(canonical_config) && identical(
    project_to_reference(config, canonical_config), canonical_config
  )
  embedded_scenarios_match <- FALSE
  if (!is.null(canonical_config) && !is.null(config$simulation$scenarios)) {
    embedded_by_id <- stats::setNames(
      config$simulation$scenarios,
      vapply(config$simulation$scenarios, `[[`, character(1), "id")
    )
    embedded_scenarios_match <- all(vapply(
      canonical_config$simulation$scenarios,
      function(reference) {
        candidate <- embedded_by_id[[reference$id]]
        !is.null(candidate) && identical(candidate[names(reference)], reference)
      }, logical(1)
    )) && length(embedded_by_id) ==
      length(canonical_config$simulation$scenarios)
  }
  embedded_methods <- if (!is.null(config$methods)) vapply(
    config$methods, function(x) as.character(x$id), character(1)
  ) else character()
  canonical_methods <- if (!is.null(canonical_config$methods)) vapply(
    canonical_config$methods, function(x) as.character(x$id), character(1)
  ) else character()
  bundle_md5 <- if (!is.null(bundle$config_md5))
    unname(as.character(bundle$config_md5[[1L]])) else NA_character_
  canonical_md5 <- if (file.exists(canonical_path))
    unname(tools::md5sum(canonical_path)) else NA_character_
  canonical_source_paths <- c(
    calibration_solver = file.path(repo_root, "analysis",
                                   "covariance_headroom",
                                   "calibration_solver.R"),
    feasible_estimator = file.path(analysis_dir, "feasible_estimator.R"),
    simulation_functions = file.path(analysis_dir, "simulation_functions.R"),
    sensitivity_functions = file.path(analysis_dir, "sensitivity_functions.R"),
    e2_simulation_functions = file.path(analysis_dir,
                                        "e2_simulation_functions.R"),
    runner = file.path(analysis_dir, "run_gate_e2_simulation.R"),
    summarizer = file.path(analysis_dir, "summarize_gate_e2.R")
  )
  source_files_available <- all(file.exists(canonical_source_paths))
  canonical_source_md5 <- if (source_files_available) {
    value <- unname(tools::md5sum(canonical_source_paths))
    names(value) <- names(canonical_source_paths)
    value
  } else stats::setNames(rep(NA_character_, length(canonical_source_paths)),
                         names(canonical_source_paths))
  bundle_source_md5 <- if (!is.null(bundle$source_md5)) {
    value <- as.character(bundle$source_md5)
    names(value) <- names(bundle$source_md5)
    value
  } else character()
  source_integrity_pass <- identical(bundle_source_md5,
                                     canonical_source_md5)
  embedded_path_is_canonical <- !is.null(config$.path) &&
    file.exists(config$.path) && identical(
      normalizePath(config$.path, mustWork = TRUE),
      normalizePath(canonical_path, mustWork = TRUE)
    )
  config_integrity_available <- !production || (
    !is.null(canonical_config) && is.finite(expected_replications) &&
      !is.na(bundle_md5) && !is.na(canonical_md5) &&
      source_files_available && length(bundle_source_md5) > 0L
  )
  config_integrity_pass <- !production || (
      identical(mode, "production") &&
      identical(as.character(config$status),
                "frozen_after_preproduction_and_completion_audit") &&
      identical(expected_replications,
                as.integer(canonical_config$replications_per_scenario)) &&
      identical(as.integer(config$seed), as.integer(canonical_config$seed)) &&
      identical(as.integer(bootstrap_draws),
                as.integer(canonical_config$monte_carlo_bootstrap_draws)) &&
      identical(as.integer(bootstrap_seed),
                as.integer(canonical_config$monte_carlo_bootstrap_seed)) &&
      identical(config$hard_correctness_pass_rules,
                canonical_config$hard_correctness_pass_rules) &&
      identical(config$practical_value_pass_rules,
                canonical_config$practical_value_pass_rules) &&
      identical(embedded_methods, canonical_methods) &&
      embedded_scenarios_match && embedded_config_matches &&
      embedded_path_is_canonical &&
      identical(bundle_md5, canonical_md5) && source_integrity_pass
  )
  add_component(
    "hard", "frozen_configuration_integrity", TRUE,
    config_integrity_available, config_integrity_pass,
    if (production) paste0(
      bundle_md5, "; canonical=", canonical_md5,
      "; sources=", if (source_integrity_pass) "match" else "mismatch"
    )
    else "smoke (not evaluated)",
    "production bundle, embedded protocol, and canonical frozen YAML agree"
  )

  base_exact_methods <- unique(c(
    role_method("ordinary"), role_method("selected"),
    role_method("restricted"), "oracle_tau_0_25", "oracle_tau_0_50",
    "oracle_tau_0_75", role_method("oracle_blend"),
    role_method("feasible_blend")
  ))
  base_exact_methods <- base_exact_methods[!is.na(base_exact_methods)]
  assignment_exact_methods <- c(
    "selected_adaptive_correct_assignment",
    "wrong_assignment_saturated_mean", "wrong_assignment_oracle_mean"
  )
  exact_grid <- rbind(
    make_protocol_grid(expected_scenarios, base_exact_methods),
    make_protocol_grid(scenario_ids(assignment_cell), assignment_exact_methods)
  )
  exact_grid <- unique(exact_grid)
  exact_check <- protocol_summary_cells(
    coordinate_summary, exact_grid,
    c("bias_prediction_standardized_error", "bias_prediction_mc_z")
  )
  exact_rows <- exact_check$rows
  exact_available <- exact_check$available && protocol_raw_complete(
    results, exact_grid, "predicted_bias"
  )
  exact_pass <- exact_available && all(
    exact_rows$bias_prediction_standardized_error <=
      rules$bias_prediction_standardized_error_max &
      exact_rows$bias_prediction_mc_z <=
        rules$bias_prediction_paired_mc_z_max
  )
  add_component(
    "hard", "exact_bias_agreement", TRUE, exact_available, exact_pass,
    paste(max_or_na(exact_rows$bias_prediction_standardized_error),
          max_or_na(exact_rows$bias_prediction_mc_z), sep = "; max z="),
    paste0("max discrepancy/MC SD <= ",
           rules$bias_prediction_standardized_error_max,
           "; max paired MC z <= ", rules$bias_prediction_paired_mc_z_max)
  )

  least_favourable_local_scenarios <- scenario_ids(
    as.character(metadata$scenario_class) == "local_least_favourable"
  )
  fixed_operator_methods <- unique(c(
    role_method("ordinary"), role_method("restricted"),
    "oracle_tau_0_25", "oracle_tau_0_50", "oracle_tau_0_75"
  ))
  fixed_operator_methods <- fixed_operator_methods[
    !is.na(fixed_operator_methods)
  ]
  local_mse_grid <- make_protocol_grid(
    least_favourable_local_scenarios, fixed_operator_methods
  )
  local_mse_check <- protocol_summary_cells(
    coordinate_summary, local_mse_grid, "mse_to_theory_ratio"
  )
  local_mse_rows <- local_mse_check$rows
  local_mse_available <- local_mse_check$available &&
    protocol_raw_complete(results, local_mse_grid,
                          c("predicted_mse", "estimate", "truth"))
  local_mse_pass <- local_mse_available && all(
    local_mse_rows$mse_to_theory_ratio >= rules$mse_theory_ratio_min &
      local_mse_rows$mse_to_theory_ratio <= rules$mse_theory_ratio_max
  )
  add_component(
    "hard", "local_amse_theory", TRUE, local_mse_available, local_mse_pass,
    paste(min_or_na(local_mse_rows$mse_to_theory_ratio),
          max_or_na(local_mse_rows$mse_to_theory_ratio), sep = " to "),
    paste0("each empirical/theoretical local AMSE ratio in [",
           rules$mse_theory_ratio_min, ", ", rules$mse_theory_ratio_max, "]")
  )

  scaling_bias_difference <- numeric()
  scaling_variance_difference <- numeric()
  if (nrow(metadata) && "local_scaling_pair" %in% names(metadata) &&
      "respondents" %in% names(metadata)) {
    pair_meta <- metadata[
      !is.na(metadata$local_scaling_pair) & nzchar(metadata$local_scaling_pair),
      c("scenario", "local_scaling_pair", "respondents"), drop = FALSE
    ]
    pair_rows <- merge(
      coordinate_summary[c("scenario", "method", "coordinate",
                           "empirical_bias", "mc_variance", "predicted_bias",
                           "predicted_mse")], pair_meta,
      by = "scenario", all = FALSE, sort = FALSE
    )
    pair_rows <- pair_rows[
      pair_rows$method %in% fixed_operator_methods &
        pair_rows$coordinate == primary_coordinate, , drop = FALSE
    ]
    if (nrow(pair_rows)) {
      pair_rows$scaled_bias <- sqrt(pair_rows$respondents) *
        pair_rows$empirical_bias
      pair_rows$scaled_variance <- pair_rows$respondents *
        pair_rows$mc_variance
      pair_rows$theory_scaled_bias <- sqrt(pair_rows$respondents) *
        pair_rows$predicted_bias
      pair_rows$theory_scaled_variance <- pmax(
        pair_rows$respondents * pair_rows$predicted_mse -
          pair_rows$theory_scaled_bias^2,
        0
      )
      pair_groups <- split(
        pair_rows,
        interaction(pair_rows$local_scaling_pair, pair_rows$method,
                    pair_rows$coordinate, drop = TRUE)
      )
      scaling_bias_difference <- vapply(pair_groups, function(x) {
        required <- c("scaled_bias", "theory_scaled_bias",
                      "theory_scaled_variance")
        if (nrow(x) != 2L || any(!is.finite(as.matrix(x[required])))) {
          return(NA_real_)
        }
        # A zero theoretical bias (notably H0) has no meaningful relative
        # denominator.  Normalize the pair difference by the common
        # theoretical root-variance scale in that case.
        bias_scale <- max(
          abs(mean(x$theory_scaled_bias)),
          sqrt(mean(x$theory_scaled_variance)),
          .Machine$double.eps
        )
        abs(diff(x$scaled_bias)) / bias_scale
      }, numeric(1))
      scaling_variance_difference <- vapply(pair_groups, function(x) {
        required <- c("scaled_variance", "theory_scaled_variance")
        if (nrow(x) != 2L || any(!is.finite(as.matrix(x[required])))) {
          return(NA_real_)
        }
        variance_scale <- max(mean(x$theory_scaled_variance),
                              .Machine$double.eps)
        abs(diff(x$scaled_variance)) / variance_scale
      }, numeric(1))
    }
  }
  scaling_available <- length(scaling_bias_difference) ==
    length(fixed_operator_methods) &&
    length(scaling_variance_difference) == length(fixed_operator_methods) &&
    all(is.finite(scaling_bias_difference)) &&
    all(is.finite(scaling_variance_difference))
  scaling_limit <- hard_protocol$local_amse$local_scaling_pair_max_relative_difference
  add_component(
    "hard", "local_scaling_pair", TRUE, scaling_available,
    scaling_available && all(scaling_bias_difference <= scaling_limit) &&
      all(scaling_variance_difference <= scaling_limit),
    paste(max_or_na(scaling_bias_difference),
          max_or_na(scaling_variance_difference),
          sep = "; variance max="),
    paste0("max paired scaled-bias and G-scaled-variance difference <= ",
           scaling_limit, " relative to their common theoretical scales")
  )

  lf_rows <- blend_summary[
    grepl("local_least", blend_summary$scenario_class) &
      is.finite(blend_summary$radius_ratio) &
      blend_summary$coordinate == primary_coordinate &
      "restricted_to_ordinary_risk_ratio" %in% names(blend_summary),
    , drop = FALSE
  ]
  lf_ratio <- if (nrow(lf_rows)) lf_rows$restricted_to_ordinary_risk_ratio else
    numeric()
  lf_available <- nrow(lf_rows) ==
    length(least_favourable_local_scenarios) &&
    setequal(lf_rows$scenario, least_favourable_local_scenarios) &&
    all(is.finite(lf_ratio)) &&
    all(c(0.5, 1, 2, 3) %in% unique(lf_rows$radius_ratio))
  lf_rules <- hard_protocol$least_favourable_oracle_adaptive_to_ordinary_mse
  lf_pass <- lf_available &&
    all(lf_rows$restricted_to_ordinary_risk_ratio[
      abs(lf_rows$radius_ratio - 0.5) < 1e-8] < lf_rules$r_0_5_max) &&
    all(lf_rows$restricted_to_ordinary_risk_ratio[
      abs(lf_rows$radius_ratio - 1) < 1e-8] >= lf_rules$r_1_min &
      lf_rows$restricted_to_ordinary_risk_ratio[
        abs(lf_rows$radius_ratio - 1) < 1e-8] <= lf_rules$r_1_max) &&
    all(lf_rows$restricted_to_ordinary_risk_ratio[
      abs(lf_rows$radius_ratio - 2) < 1e-8] > lf_rules$r_2_min) &&
    all(lf_rows$restricted_to_ordinary_risk_ratio[
      abs(lf_rows$radius_ratio - 3) < 1e-8] > lf_rules$r_3_min)
  add_component(
    "hard", "least_favourable_risk_shape", TRUE, lf_available, lf_pass,
    paste(min_or_na(lf_ratio), max_or_na(lf_ratio), sep = " to "),
    paste0("oracle-adaptive/ordinary MSE: r=.5 < ", lf_rules$r_0_5_max,
           "; r=1 in [", lf_rules$r_1_min, ", ", lf_rules$r_1_max,
           "]; r=2 > ", lf_rules$r_2_min, "; r=3 > ", lf_rules$r_3_min)
  )

  ordinary_grid <- make_protocol_grid(expected_scenarios,
                                      role_method("ordinary"))
  ordinary_check <- protocol_summary_cells(
    coordinate_summary, ordinary_grid,
    c("naive_coverage", "standardized_bias",
      "sandwich_variance_calibration")
  )
  ordinary_primary <- ordinary_check$rows
  ordinary_rules <- hard_protocol$ordinary_inference
  ordinary_available <- ordinary_check$available && protocol_raw_complete(
    results, ordinary_grid, c("standard_error", "naive_covered")
  )
  ordinary_pass <- ordinary_available && all(
    ordinary_primary$naive_coverage >= ordinary_rules$primary_coverage_min &
      ordinary_primary$naive_coverage <= ordinary_rules$primary_coverage_max &
      ordinary_primary$standardized_bias <=
        ordinary_rules$maximum_absolute_standardized_bias &
      ordinary_primary$sandwich_variance_calibration >=
        ordinary_rules$sandwich_to_mc_variance_min &
      ordinary_primary$sandwich_variance_calibration <=
        ordinary_rules$sandwich_to_mc_variance_max
  )
  add_component(
    "hard", "ordinary_primary_inference", TRUE, ordinary_available,
    ordinary_pass,
    paste(min_or_na(ordinary_primary$naive_coverage),
          max_or_na(ordinary_primary$naive_coverage), sep = " to "),
    paste0("coverage [", ordinary_rules$primary_coverage_min, ", ",
           ordinary_rules$primary_coverage_max, "], standardized bias <= ",
           ordinary_rules$maximum_absolute_standardized_bias,
           ", sandwich/MC [", ordinary_rules$sandwich_to_mc_variance_min,
           ", ", ordinary_rules$sandwich_to_mc_variance_max, "]")
  )

  selected_nb_scenarios <- scenario_ids(
    correct_law & as.character(metadata$scenario_class) %in%
      c("null", "local_blind")
  )
  selected_nb_grid <- make_protocol_grid(
    selected_nb_scenarios, role_method("selected")
  )
  selected_nb_check <- protocol_summary_cells(
    coordinate_summary, selected_nb_grid,
    c("naive_coverage", "standardized_bias",
      "sandwich_variance_calibration")
  )
  selected_nb <- selected_nb_check$rows
  selected_rules <- hard_protocol$selected_null_and_blind_inference
  selected_available <- length(selected_nb_scenarios) == 2L &&
    selected_nb_check$available && protocol_raw_complete(
      results, selected_nb_grid, c("standard_error", "naive_covered")
    )
  selected_pass <- selected_available && all(
    selected_nb$naive_coverage >= selected_rules$primary_coverage_min &
      selected_nb$naive_coverage <= selected_rules$primary_coverage_max &
      selected_nb$standardized_bias <=
        selected_rules$maximum_absolute_standardized_bias &
      selected_nb$sandwich_variance_calibration >=
        selected_rules$sandwich_to_mc_variance_min &
      selected_nb$sandwich_variance_calibration <=
        selected_rules$sandwich_to_mc_variance_max
  )
  add_component(
    "hard", "selected_null_blind_inference", TRUE, selected_available,
    selected_pass,
    paste(min_or_na(selected_nb$naive_coverage),
          max_or_na(selected_nb$naive_coverage), sep = " to "),
    paste0("null/blind primary coverage [",
           selected_rules$primary_coverage_min, ", ",
           selected_rules$primary_coverage_max,
           "], standardized bias and sandwich calibration within limits")
  )

  correct_carryover_scenarios <- scenario_ids(correct_law & has_carryover)
  oracle_aware_grid <- make_protocol_grid(
    correct_carryover_scenarios, role_method("restricted")
  )
  oracle_aware_check <- protocol_summary_cells(
    coordinate_summary, oracle_aware_grid, "bias_aware_coverage"
  )
  oracle_aware <- oracle_aware_check$rows
  oracle_aware_available <- oracle_aware_check$available &&
    protocol_raw_complete(
      results, oracle_aware_grid, "covered_bias_additive_oracle_k"
    )
  oracle_aware_min <- hard_protocol$oracle_bias_aware_primary_coverage_min
  add_component(
    "hard", "oracle_bias_aware_primary_coverage", TRUE,
    oracle_aware_available,
    oracle_aware_available && all(oracle_aware$bias_aware_coverage >=
                                    oracle_aware_min),
    min_or_na(oracle_aware$bias_aware_coverage),
    paste0("each correct-assignment primary coverage >= ", oracle_aware_min)
  )

  expected_h_variants <- na.omit(c(role_method("selected"),
                                    role_method("restricted")))
  expected_h_grid <- expand.grid(
    scenario = expected_scenarios,
    coordinate = expected_coordinates,
    hausman_variant = expected_h_variants,
    stringsAsFactors = FALSE
  )
  observed_h_counts <- if (nrow(hausman_summary)) hausman_summary[
    c("scenario", "coordinate", "hausman_variant", "replications")
  ] else data.frame()
  hausman_complete <- FALSE
  if (nrow(expected_h_grid) && nrow(observed_h_counts)) {
    h_completion <- merge(
      expected_h_grid, observed_h_counts,
      by = c("scenario", "coordinate", "hausman_variant"),
      all.x = TRUE, sort = FALSE
    )
    raw_h <- merge(expected_h_grid, hausman,
                   by = c("scenario", "coordinate", "hausman_variant"),
                   all = FALSE, sort = FALSE)
    raw_h_groups <- split(
      raw_h$replication,
      interaction(raw_h$scenario, raw_h$coordinate, raw_h$hausman_variant,
                  drop = TRUE)
    )
    hausman_complete <- nrow(h_completion) == nrow(expected_h_grid) &&
      all(!is.na(h_completion$replications)) &&
      all(h_completion$replications == expected_replications) &&
      nrow(raw_h) == nrow(expected_h_grid) * expected_replications &&
      !anyDuplicated(raw_h[c("scenario", "coordinate", "hausman_variant",
                            "replication")]) &&
      all(!is.na(raw_h$hausman_reject)) &&
      all(is.finite(raw_h$theoretical_hausman_power)) &&
      length(raw_h_groups) == nrow(expected_h_grid) && all(vapply(
        raw_h_groups,
        function(x) identical(sort(unique(as.integer(as.character(x)))),
                              expected_replication_ids),
        logical(1)
      ))
  }
  hausman_primary <- hausman_summary[
    hausman_summary$coordinate == primary_coordinate, ]
  hausman_size_scenarios <- scenario_ids(
    correct_law & as.character(metadata$scenario_class) %in%
      c("null", "local_blind")
  )
  hausman_nb <- hausman_primary[
    hausman_primary$scenario %in% hausman_size_scenarios &
      hausman_primary$hausman_variant == role_method("restricted"), ]
  h_rules <- hard_protocol$hausman_directional
  h_size_available <- hausman_complete &&
    nrow(hausman_nb) == length(hausman_size_scenarios) &&
    setequal(hausman_nb$scenario, hausman_size_scenarios)
  add_component(
    "hard", "hausman_null_blind_size", TRUE, h_size_available,
    h_size_available && all(
      hausman_nb$rejection_rate >= h_rules$null_and_blind_rejection_min &
        hausman_nb$rejection_rate <= h_rules$null_and_blind_rejection_max
    ),
    paste(min_or_na(hausman_nb$rejection_rate),
          max_or_na(hausman_nb$rejection_rate), sep = " to "),
    paste0("null/blind rejection in [",
           h_rules$null_and_blind_rejection_min, ", ",
           h_rules$null_and_blind_rejection_max, "]")
  )
  hausman_power_scenarios <- scenario_ids(
    correct_law & has_carryover &
      grepl("^local_", as.character(metadata$scenario_class)) &
      as.character(metadata$scenario_class) != "local_blind"
  )
  hausman_power <- hausman_primary[
    hausman_primary$scenario %in% hausman_power_scenarios &
      hausman_primary$hausman_variant %in% expected_h_variants, ]
  h_threshold <- if (nrow(hausman_power)) ifelse(
    grepl("oracle", hausman_power$hausman_variant),
    h_rules$oracle_power_absolute_error_max,
    h_rules$selected_power_absolute_error_max
  ) else numeric()
  h_power_available <- hausman_complete &&
    nrow(hausman_power) == length(hausman_power_scenarios) *
      length(expected_h_variants) &&
    all(is.finite(hausman_power$absolute_power_error))
  add_component(
    "hard", "hausman_power_calibration", TRUE, h_power_available,
    h_power_available && all(hausman_power$absolute_power_error <= h_threshold),
    max_or_na(hausman_power$absolute_power_error),
    paste0("absolute error <= ", h_rules$oracle_power_absolute_error_max,
           " oracle and <= ", h_rules$selected_power_absolute_error_max,
           " selected")
  )
  h_oracle_power <- hausman_power[
    hausman_power$hausman_variant == role_method("restricted"), ]
  h_r1 <- h_oracle_power[abs(h_oracle_power$radius_ratio - 1) < 1e-8, ]
  h_r2 <- h_oracle_power[abs(h_oracle_power$radius_ratio - 2) < 1e-8, ]
  expected_r1 <- metadata$scenario[
    metadata$scenario %in% hausman_power_scenarios &
      is.finite(as.numeric(metadata$radius_ratio)) &
      abs(as.numeric(metadata$radius_ratio) - 1) < 1e-8
  ]
  expected_r2 <- metadata$scenario[
    metadata$scenario %in% hausman_power_scenarios &
      is.finite(as.numeric(metadata$radius_ratio)) &
      abs(as.numeric(metadata$radius_ratio) - 2) < 1e-8
  ]
  h_shape_available <- hausman_complete &&
    nrow(h_r1) == length(expected_r1) && setequal(h_r1$scenario, expected_r1) &&
    nrow(h_r2) == length(expected_r2) && setequal(h_r2$scenario, expected_r2)
  add_component(
    "hard", "hausman_directional_power_shape", TRUE, h_shape_available,
    h_shape_available && all(h_r1$rejection_rate <= h_rules$r_1_power_max) &&
      all(h_r2$rejection_rate >= h_rules$r_2_power_min),
    paste(max_or_na(h_r1$rejection_rate), min_or_na(h_r2$rejection_rate),
          sep = "; r2 min="),
    paste0("r=1 power <= ", h_rules$r_1_power_max,
           "; r=2 power >= ", h_rules$r_2_power_min)
  )

  assignment_metadata <- metadata[assignment_cell, , drop = FALSE]
  both_directions <- nrow(assignment_metadata) > 0L &&
    any(assignment_metadata$true_assignment == "product" &
          assignment_metadata$analysis_assignment ==
            "uniform_without_replacement") &&
    any(assignment_metadata$true_assignment ==
          "uniform_without_replacement" &
          assignment_metadata$analysis_assignment == "product")
  combined_scenarios <- as.character(assignment_metadata$scenario[
    as.character(assignment_metadata$scenario_class) ==
      "assignment_error_with_carryover"
  ])
  assignment_grid <- exact_grid[
    exact_grid$scenario %in% assignment_metadata$scenario, , drop = FALSE
  ]
  assignment_check <- protocol_summary_cells(
    coordinate_summary, assignment_grid,
    c("bias_prediction_standardized_error", "bias_prediction_mc_z")
  )
  assignment_exact <- assignment_check$rows
  assignment_exact_available <- assignment_check$available && both_directions &&
    length(combined_scenarios) == 1L && protocol_raw_complete(
      results, assignment_grid, "predicted_bias"
    )
  assignment_exact_pass <- assignment_exact_available && all(
    assignment_exact$bias_prediction_standardized_error <=
      rules$assignment_bias_standardized_error_max &
      assignment_exact$bias_prediction_mc_z <=
        rules$bias_prediction_paired_mc_z_max
  )
  add_component(
    "hard", "assignment_error_exact_decomposition", TRUE,
    assignment_exact_available, assignment_exact_pass,
    paste(max_or_na(assignment_exact$bias_prediction_standardized_error),
          max_or_na(assignment_exact$bias_prediction_mc_z), sep = "; max z="),
    "both assignment directions plus combined carryover; exact total-bias agreement"
  )

  combined_grid <- make_protocol_grid(
    combined_scenarios,
    c("wrong_assignment_oracle_mean", "wrong_assignment_saturated_mean")
  )
  combined_check <- protocol_summary_cells(
    coordinate_summary, combined_grid,
    c("bias_prediction_standardized_error", "bias_prediction_mc_z")
  )
  combined_rows <- combined_check$rows
  combined_available <- combined_check$available && protocol_raw_complete(
    results, combined_grid, "predicted_bias"
  )
  combined_pass <- combined_available && all(
    combined_rows$bias_prediction_standardized_error <=
      rules$assignment_bias_standardized_error_max &
      combined_rows$bias_prediction_mc_z <=
        rules$bias_prediction_paired_mc_z_max
  )
  add_component(
    "hard", "assignment_error_combined_carryover", TRUE,
    combined_available, combined_pass,
    paste(max_or_na(combined_rows$bias_prediction_standardized_error),
          max_or_na(combined_rows$bias_prediction_mc_z), sep = "; max z="),
    paste0("combined-cell oracle-mean predicted bias is the direct carryover ",
           "term; saturated mean satisfies the full exact decomposition")
  )

  assignment_rules <- hard_protocol$assignment_error
  rescue_specs <- list(
    wrong_assignment_oracle_mean = list(
      bias = assignment_rules$oracle_mean_rescue_maximum_absolute_standardized_bias,
      lower = assignment_rules$oracle_mean_rescue_coverage_min,
      upper = assignment_rules$oracle_mean_rescue_coverage_max
    ),
    wrong_assignment_saturated_mean = list(
      bias = assignment_rules$saturated_mean_rescue_maximum_absolute_standardized_bias,
      lower = assignment_rules$saturated_mean_rescue_coverage_min,
      upper = assignment_rules$saturated_mean_rescue_coverage_max
    )
  )
  assignment_null_scenarios <- as.character(assignment_metadata$scenario[
    as.character(assignment_metadata$carryover) == "none"
  ])
  rescue_checks <- lapply(names(rescue_specs), function(method) {
    grid <- make_protocol_grid(assignment_null_scenarios, method)
    checked <- protocol_summary_cells(
      coordinate_summary, grid, c("naive_coverage", "standardized_bias")
    )
    x <- checked$rows
    spec <- rescue_specs[[method]]
    list(
      available = checked$available && protocol_raw_complete(
        results, grid, c("standard_error", "naive_covered")
      ),
      passed = checked$available && all(x$standardized_bias <= spec$bias &
        x$naive_coverage >= spec$lower & x$naive_coverage <= spec$upper),
      value = max_or_na(x$standardized_bias)
    )
  })
  rescue_available <- all(vapply(rescue_checks, `[[`, logical(1), "available"))
  rescue_pass <- rescue_available &&
    all(vapply(rescue_checks, `[[`, logical(1), "passed"))
  add_component(
    "hard", "assignment_error_rescue_methods", TRUE, rescue_available,
    rescue_pass,
    max_or_na(vapply(rescue_checks, `[[`, numeric(1), "value")),
    paste0("oracle-mean and saturated-mean rescues satisfy frozen bias/coverage ",
           "limits in both no-carryover assignment-error cells")
  )

  add_component(
    "hard", "exact_calibration", TRUE, is.finite(maximum_calibration),
    is.finite(maximum_calibration) && maximum_calibration <=
      hard_protocol$maximum_calibration_residual,
    maximum_calibration,
    paste0("max residual <= ", hard_protocol$maximum_calibration_residual)
  )

  expected_outer_folds <- as.integer(config$estimator$outer_folds)
  selector_complete <- FALSE
  if (!is.null(bundle$selector)) {
    selector_raw <- as.data.frame(bundle$selector)
    selector_required <- c(
      "scenario", "replication", "outer_fold", "selected_candidate",
      "selected_blend", "selected_covariance_class", "calibration_residual"
    )
    if (all(selector_required %in% names(selector_raw))) {
      selector_complete <- nrow(selector_raw) ==
        length(expected_scenarios) * expected_replications *
          expected_outer_folds &&
        !anyDuplicated(selector_raw[c("scenario", "replication",
                                     "outer_fold")]) &&
        setequal(unique(as.character(selector_raw$scenario)),
                 expected_scenarios) &&
        all(!is.na(selector_raw$selected_candidate)) &&
        all(!is.na(selector_raw$selected_covariance_class)) &&
        all(is.finite(selector_raw$selected_blend)) &&
        all(is.finite(selector_raw$calibration_residual))
    }
  }
  tau_complete <- FALSE
  if (!is.null(bundle$tau)) {
    tau_raw <- as.data.frame(bundle$tau)
    tau_required <- c(
      "scenario", "replication", "outer_fold", "coordinate",
      "feasible_tau", "oracle_unrestricted_tau", "oracle_pattern_tau",
      "selected_blend", "k_hat_selected"
    )
    if (all(tau_required %in% names(tau_raw))) {
      tau_complete <- nrow(tau_raw) ==
        length(expected_scenarios) * expected_replications *
          expected_outer_folds * length(expected_coordinates) &&
        !anyDuplicated(tau_raw[c("scenario", "replication", "outer_fold",
                                "coordinate")]) &&
        setequal(unique(as.character(tau_raw$scenario)), expected_scenarios) &&
        setequal(unique(as.character(tau_raw$coordinate)),
                 expected_coordinates) &&
        all(vapply(tau_required[5:9], function(field) {
          all(is.finite(tau_raw[[field]]))
        }, logical(1)))
    }
  }
  departure_complete <- FALSE
  if (!is.null(bundle$departure)) {
    departure_raw <- as.data.frame(bundle$departure)
    departure_required <- c(
      "scenario", "replication", "pattern", "epsilon", "local_radius",
      "normalized_r", "unit_epsilon", "unit_bias_oracle",
      "projection_residual"
    )
    if (all(departure_required %in% names(departure_raw))) {
      departure_complete <- nrow(departure_raw) ==
        length(expected_scenarios) * expected_replications &&
        !anyDuplicated(departure_raw[c("scenario", "replication")]) &&
        setequal(unique(as.character(departure_raw$scenario)),
                 expected_scenarios) &&
        all(!is.na(departure_raw$pattern)) &&
        all(vapply(departure_required[4:9], function(field) {
          all(is.finite(departure_raw[[field]]))
        }, logical(1)))
    }
  }
  auxiliary_records_complete <- selector_complete && tau_complete &&
    departure_complete && hausman_complete
  add_component(
    "hard", "recorded_diagnostics_completion", TRUE, TRUE,
    auxiliary_records_complete,
    paste(selector_complete, tau_complete, departure_complete,
          hausman_complete, sep = ";"),
    paste0("selector, tuning/sensitivity, departure, and Hausman records are ",
           "complete and nonmissing for the frozen grid")
  )

  add_component(
    "hard", "numerical_completion", TRUE, completion_available,
    completion_available && all(completed$complete %in% TRUE) &&
      failure_rate <= hard_protocol$maximum_numerical_failure_rate &&
      invalid_required_result_rows == 0L && replication_ids_complete &&
      common_seed_complete,
    paste(failure_rate, invalid_required_result_rows,
          sep = "; invalid rows="),
    paste0("applicable scenario/method/coordinate cells complete with no ",
           "dropped replications, mismatched IDs/seeds, or missing essential values")
  )

  # Paired, cellwise bootstrap for a method relative to the better of two
  # prespecified benchmarks.  The minimum is recomputed inside every draw.
  bootstrap_min_mse_cells <- function(numerator_method, benchmark_methods,
                                      keep, draws) {
    methods <- c(numerator_method, benchmark_methods)
    x <- results[keep & results$method %in% methods,
                 c("scenario", "replication", "coordinate", "method",
                   "estimate", "truth"), drop = FALSE]
    if (!nrow(x)) return(data.frame())
    x$squared_error <- (x$estimate - x$truth)^2
    groups <- split(x, interaction(x$scenario, x$coordinate, drop = TRUE))
    rows <- lapply(groups, function(z) {
      wide <- reshape(z[c("scenario", "replication", "coordinate", "method",
                          "squared_error")],
                      idvar = c("scenario", "replication", "coordinate"),
                      timevar = "method", direction = "wide")
      required <- paste0("squared_error.", methods)
      if (!all(required %in% names(wide))) return(NULL)
      wide <- wide[stats::complete.cases(wide[required]), ]
      if (nrow(wide) < 2L) return(NULL)
      num <- wide[[required[[1L]]]]
      b1 <- wide[[required[[2L]]]]
      b2 <- wide[[required[[3L]]]]
      ratio <- mean(num) / min(mean(b1), mean(b2))
      boot <- replicate(draws, {
        index <- sample.int(length(num), length(num), replace = TRUE)
        mean(num[index]) / min(mean(b1[index]), mean(b2[index]))
      })
      data.frame(
        scenario = wide$scenario[[1L]],
        coordinate = wide$coordinate[[1L]],
        numerator_method = numerator_method,
        benchmark_1 = benchmark_methods[[1L]],
        benchmark_2 = benchmark_methods[[2L]],
        risk_ratio = ratio,
        lower_2_5 = unname(stats::quantile(boot, .025, type = 8)),
        upper_97_5 = unname(stats::quantile(boot, .975, type = 8)),
        stringsAsFactors = FALSE
      )
    })
    rows <- rows[!vapply(rows, is.null, logical(1))]
    if (length(rows)) do.call(rbind, rows) else data.frame()
  }

  correct_assignment <- results$true_assignment == results$analysis_assignment
  carryover <- results$departure_pattern != "none"
  product_lf <- correct_assignment &
    grepl("local_least", results$scenario_class) &
    results$true_assignment == "product" &
    results$coordinate == primary_coordinate
  oracle_tau_cells <- bootstrap_min_mse_cells(
    role_method("oracle_blend"),
    c(role_method("ordinary"), role_method("restricted")),
    product_lf, bootstrap_draws
  )
  if (nrow(oracle_tau_cells)) oracle_tau_cells$gate_family <- "oracle_tau_mm"
  expected_oracle_tau_scenarios <- scenario_ids(
    correct_law &
      as.character(metadata$scenario_class) == "local_least_favourable" &
      as.character(metadata$true_assignment) == "product"
  )
  oracle_tau_available <- nrow(oracle_tau_cells) ==
    length(expected_oracle_tau_scenarios) &&
    setequal(oracle_tau_cells$scenario, expected_oracle_tau_scenarios) &&
    all(is.finite(oracle_tau_cells$risk_ratio)) &&
    all(is.finite(oracle_tau_cells$upper_97_5))
  oracle_tau_limit <- practical_protocol$oracle_tau_mm$paired_bootstrap_upper_97_5_max
  add_component(
    "practical", "oracle_tau_mm_risk", TRUE, oracle_tau_available,
    oracle_tau_available &&
      all(oracle_tau_cells$upper_97_5 < oracle_tau_limit),
    max_or_na(oracle_tau_cells$upper_97_5),
    paste0("each paired-bootstrap risk-ratio UCB < ", oracle_tau_limit,
           " versus min(ordinary, oracle adaptive)"))

  feasible_cells <- bootstrap_min_mse_cells(
    role_method("feasible_blend"),
    c(role_method("ordinary"), role_method("selected")),
    correct_assignment & carryover & results$coordinate == primary_coordinate,
    bootstrap_draws
  )
  if (nrow(feasible_cells)) {
    feasible_cells$gate_family <- "feasible_tau_generic"
    feasible_meta <- unique(results[c("scenario", "scenario_class",
                                      "radius_ratio")])
    feasible_cells <- merge(feasible_cells, feasible_meta, by = "scenario",
                            all.x = TRUE, sort = FALSE)
  }
  feasible_rules <- practical_protocol$feasible_tau_generic
  feasible_available <- nrow(feasible_cells) ==
    length(correct_carryover_scenarios) &&
    setequal(feasible_cells$scenario, correct_carryover_scenarios) &&
    all(is.finite(feasible_cells$risk_ratio)) &&
    all(is.finite(feasible_cells$upper_97_5))
  feasible_point_pass <- feasible_available &&
    all(feasible_cells$risk_ratio <= feasible_rules$each_point_ratio_max)
  feasible_r1 <- if (feasible_available) feasible_cells[
    grepl("local_least", feasible_cells$scenario_class) &
      abs(feasible_cells$radius_ratio - 1) < 1e-8, ] else data.frame()
  expected_feasible_r1 <- scenario_ids(
    correct_law &
      as.character(metadata$scenario_class) == "local_least_favourable" &
      is.finite(as.numeric(metadata$radius_ratio)) &
      abs(as.numeric(metadata$radius_ratio) - 1) < 1e-8
  )
  feasible_r1_available <- nrow(feasible_r1) ==
    length(expected_feasible_r1) &&
    setequal(feasible_r1$scenario, expected_feasible_r1)
  add_component(
    "practical", "feasible_tau_generic_risk", TRUE,
    feasible_available && feasible_r1_available,
    feasible_point_pass && all(feasible_r1$upper_97_5 <
      feasible_rules$r_1_least_favourable_upper_97_5_max),
    paste(max_or_na(feasible_cells$risk_ratio),
          max_or_na(feasible_r1$upper_97_5), sep = "; r1 UCB="),
    paste0("each point <= ", feasible_rules$each_point_ratio_max,
           "; r=1 LF UCB < ",
           feasible_rules$r_1_least_favourable_upper_97_5_max)
  )

  null_efficiency_rows <- results[
    results$is_null %in% TRUE & !(results$is_assignment_error %in% TRUE) &
      results$coordinate == primary_coordinate &
      results$method %in% c(role_method("ordinary"), role_method("selected")),
    c("scenario", "replication", "method", "estimate"), drop = FALSE
  ]
  null_efficiency <- data.frame()
  if (nrow(null_efficiency_rows)) {
    null_groups <- split(null_efficiency_rows, null_efficiency_rows$scenario)
    null_efficiency <- do.call(rbind, lapply(null_groups, function(z) {
      wide <- reshape(z, idvar = c("scenario", "replication"),
                      timevar = "method", direction = "wide")
      ordinary_name <- paste0("estimate.", role_method("ordinary"))
      selected_name <- paste0("estimate.", role_method("selected"))
      if (!all(c(ordinary_name, selected_name) %in% names(wide))) return(NULL)
      ordinary_value <- wide[[ordinary_name]]
      selected_value <- wide[[selected_name]]
      point <- stats::var(selected_value) / stats::var(ordinary_value)
      boot <- replicate(bootstrap_draws, {
        index <- sample.int(nrow(wide), nrow(wide), replace = TRUE)
        stats::var(selected_value[index]) / stats::var(ordinary_value[index])
      })
      data.frame(
        scenario = wide$scenario[[1L]], ratio = point,
        upper_97_5 = unname(stats::quantile(boot, .975, type = 8)),
        stringsAsFactors = FALSE
      )
    }))
  }
  null_eff_rules <- practical_protocol$null_selected_efficiency
  expected_null_scenarios <- scenario_ids(
    correct_law & as.character(metadata$scenario_class) == "null"
  )
  null_eff_available <- nrow(null_efficiency) ==
    length(expected_null_scenarios) &&
    setequal(null_efficiency$scenario, expected_null_scenarios) &&
    all(is.finite(null_efficiency$ratio)) &&
    all(is.finite(null_efficiency$upper_97_5))
  add_component(
    "practical", "null_selected_efficiency", TRUE, null_eff_available,
    null_eff_available && all(
      null_efficiency$ratio <=
        null_eff_rules$selected_to_ordinary_variance_ratio_max &
      null_efficiency$upper_97_5 <
        null_eff_rules$paired_bootstrap_upper_97_5_max
    ),
    paste(max_or_na(null_efficiency$ratio),
          max_or_na(null_efficiency$upper_97_5), sep = "; UCB="),
    paste0("variance ratio <= ",
           null_eff_rules$selected_to_ordinary_variance_ratio_max,
           "; UCB < ", null_eff_rules$paired_bootstrap_upper_97_5_max)
  )

  feasible_aware_grid <- make_protocol_grid(
    correct_carryover_scenarios, role_method("selected")
  )
  feasible_aware_check <- protocol_summary_cells(
    coordinate_summary, feasible_aware_grid, "bias_aware_coverage"
  )
  feasible_aware <- feasible_aware_check$rows
  feasible_coverage_min <-
    practical_protocol$feasible_bias_aware_primary_coverage_min
  feasible_aware_available <- feasible_aware_check$available &&
    protocol_raw_complete(
      results, feasible_aware_grid, "bias_aware_covered"
    )
  add_component(
    "practical", "feasible_bias_aware_primary_coverage", TRUE,
    feasible_aware_available,
    feasible_aware_available && all(feasible_aware$bias_aware_coverage >=
                                      feasible_coverage_min),
    min_or_na(feasible_aware$bias_aware_coverage),
    paste0("each feasible primary coverage >= ", feasible_coverage_min)
  )

  lag_scenarios <- scenario_ids(
    correct_law & as.character(metadata$scenario_class) == "local_lag"
  )
  structured_grid <- make_protocol_grid(
    lag_scenarios, role_method("selected")
  )
  structured <- merge(
    structured_grid, results,
    by = c("scenario", "method", "coordinate"),
    all = FALSE, sort = FALSE
  )
  if (nrow(structured)) {
    structured_groups <- split(structured, structured$scenario)
    protocol_structured_summary <- do.call(rbind, lapply(
      structured_groups, function(x) data.frame(
        scenario = x$scenario[[1L]],
        coverage = mean(x$covered_structured_additive),
        median_length_ratio = stats::median(x$width_structured_additive) /
          stats::median(x$bias_aware_length),
        stringsAsFactors = FALSE
      )
    ))
  }
  structured_rules <- practical_protocol$structured_lag
  structured_available <- length(lag_scenarios) == 2L &&
    protocol_raw_complete(
      results, structured_grid,
      c("covered_structured_additive", "width_structured_additive",
        "bias_aware_length")
    ) && nrow(protocol_structured_summary) == length(lag_scenarios) &&
    setequal(protocol_structured_summary$scenario, lag_scenarios) &&
    all(is.finite(protocol_structured_summary$coverage)) &&
    all(is.finite(protocol_structured_summary$median_length_ratio))
  add_component(
    "practical", "structured_lag_intervals", TRUE, structured_available,
    structured_available && all(
      protocol_structured_summary$coverage >= structured_rules$primary_coverage_min &
      protocol_structured_summary$median_length_ratio <=
        structured_rules$median_length_ratio_to_unrestricted_max
    ),
    if (structured_available) paste(
      min(protocol_structured_summary$coverage),
      max(protocol_structured_summary$median_length_ratio),
      sep = "; max length ratio=") else NA_real_,
    paste0("coverage >= ", structured_rules$primary_coverage_min,
           "; median structured/unrestricted length <= ",
           structured_rules$median_length_ratio_to_unrestricted_max)
  )

  protocol_risk_summary <- rbind(
    oracle_tau_cells,
    if (nrow(feasible_cells)) feasible_cells[intersect(
      names(oracle_tau_cells), names(feasible_cells))] else data.frame()
  )
  if (nrow(null_efficiency)) {
    null_rows_out <- data.frame(
      scenario = null_efficiency$scenario,
      coordinate = primary_coordinate,
      numerator_method = role_method("selected"),
      benchmark_1 = role_method("ordinary"),
      benchmark_2 = NA_character_,
      risk_ratio = null_efficiency$ratio,
      lower_2_5 = NA_real_,
      upper_97_5 = null_efficiency$upper_97_5,
      gate_family = "null_selected_efficiency",
      stringsAsFactors = FALSE
    )
    common <- union(names(protocol_risk_summary), names(null_rows_out))
    for (field in setdiff(common, names(protocol_risk_summary)))
      protocol_risk_summary[[field]] <- NA
    for (field in setdiff(common, names(null_rows_out)))
      null_rows_out[[field]] <- NA
    protocol_risk_summary <- rbind(protocol_risk_summary[common],
                                   null_rows_out[common])
  }
}

component <- do.call(rbind, components)
hard_required <- component$gate == "hard" & component$required
practical_required <- component$gate == "practical" & component$required
hard_pass <- production && all(component$available[hard_required]) &&
  all(component$passed[hard_required])
practical_evidence_pass <- production &&
  all(component$available[practical_required]) &&
  all(component$passed[practical_required])
practical_evaluated <- production && hard_pass
practical_pass <- practical_evaluated && practical_evidence_pass
gate_passed <- hard_pass && practical_pass

# ---- Outputs --------------------------------------------------------------

bias_agreement <- coordinate_summary[c(
  "scenario", "scenario_class", "departure_pattern", "radius_ratio", "method",
  "coordinate", "empirical_bias", "predicted_bias", "bias_prediction_error",
  "bias_prediction_mc_se", "bias_prediction_mc_z",
  "bias_prediction_ci_99_agreement",
  "bias_prediction_standardized_error", "bias_bound", "bound_sharpness_ratio"
)]
mse_summary <- coordinate_summary[c(
  "scenario", "scenario_class", "departure_pattern", "radius_ratio", "method",
  "coordinate", "empirical_mse", "predicted_mse", "mse_to_theory_ratio",
  "mse_theory_relative_error"
)]
coverage_summary <- coordinate_summary[c(
  "scenario", "scenario_class", "departure_pattern", "radius_ratio", "method",
  "coordinate", "naive_coverage", "naive_interval_length",
  "bias_aware_coverage", "bias_aware_coverage_lcb_95",
  "bias_aware_interval_length"
)]
assignment_summary <- coordinate_summary[
  coordinate_summary$is_assignment_error %in% TRUE,
  c("scenario", "scenario_class", "method", "coordinate", "empirical_bias",
    "assignment_predicted_bias", "assignment_bias_error",
    "assignment_bias_mc_se", "assignment_bias_mc_z",
    "assignment_bias_ci_99_agreement",
    "assignment_bias_standardized_error"), drop = FALSE
]

selector_summary <- data.frame()
if (!is.null(bundle$selector)) {
  selector_data <- as.data.frame(bundle$selector)
  if (all(c("scenario", "selected_candidate") %in% names(selector_data))) {
    selector_summary <- as.data.frame(table(
      selector_data$scenario, selector_data$selected_candidate
    ), stringsAsFactors = FALSE)
    names(selector_summary) <- c("scenario", "selected_candidate", "count")
    selector_summary <- selector_summary[selector_summary$count > 0, ]
    selector_summary$fraction <- ave(
      selector_summary$count, selector_summary$scenario,
      FUN = function(x) x / sum(x)
    )
  }
}

tau_summary <- data.frame()
if (!is.null(bundle$tau)) {
  tau_data <- as.data.frame(bundle$tau)
  tau_fields <- intersect(c("feasible_tau", "oracle_unrestricted_tau",
                            "oracle_pattern_tau", "selected_blend",
                            "k_hat_selected"), names(tau_data))
  if (all(c("scenario", "coordinate") %in% names(tau_data)) &&
      length(tau_fields)) {
    tau_summary <- aggregate(
      tau_data[tau_fields], by = tau_data[c("scenario", "coordinate")],
      FUN = mean_or_na
    )
  }
}

departure_summary <- data.frame()
if (!is.null(bundle$departure)) {
  departure_data <- as.data.frame(bundle$departure)
  departure_fields <- intersect(c("epsilon", "local_radius", "normalized_r",
                                  "unit_epsilon", "unit_bias_oracle",
                                  "projection_residual"), names(departure_data))
  if ("scenario" %in% names(departure_data) && length(departure_fields)) {
    departure_summary <- aggregate(
      departure_data[departure_fields], by = departure_data["scenario"],
      FUN = mean_or_na
    )
  }
}

write_csv <- function(x, name) utils::write.csv(
  x, file.path(result_dir, name), row.names = FALSE, na = ""
)
write_csv(coordinate_summary, "gate_e2_coordinate_summary.csv")
write_csv(method_summary, "gate_e2_method_summary.csv")
write_csv(bias_agreement, "gate_e2_bias_agreement.csv")
write_csv(mse_summary, "gate_e2_mse_summary.csv")
write_csv(coverage_summary, "gate_e2_coverage_summary.csv")
write_csv(hausman_summary, "gate_e2_hausman_summary.csv")
write_csv(family_summary, "gate_e2_family_coverage_summary.csv")
write_csv(blend_summary, "gate_e2_blend_summary.csv")
write_csv(assignment_summary, "gate_e2_assignment_summary.csv")
write_csv(completion_summary, "gate_e2_completion_summary.csv")
write_csv(method_role_frame, "gate_e2_method_roles.csv")
write_csv(component, "gate_e2_components.csv")
write_csv(protocol_risk_summary, "gate_e2_protocol_risk_summary.csv")
write_csv(protocol_structured_summary,
          "gate_e2_protocol_structured_interval_summary.csv")
write_csv(selector_summary, "gate_e2_selector_summary.csv")
write_csv(tau_summary, "gate_e2_tau_summary.csv")
write_csv(departure_summary, "gate_e2_departure_summary.csv")

required_result_columns <- list(
  keys = required_key,
  estimation_and_inference = c("standard_error", "naive_covered or naive bounds",
                               "naive_length or naive bounds",
                               "bias_aware_covered or bias-aware bounds",
                               "bias_aware_length or bias-aware bounds"),
  exact_theory = c("predicted_bias", "bias_bound", "predicted_mse"),
  hausman = c("hausman_reject or hausman_p_value",
              "theoretical_hausman_power or hausman_ncp plus hausman_df",
              "hausman_variant when more than one diagnostic is reported"),
  blending = c("blend_tau", "theoretical_tau", "line_oracle_mse"),
  assignment = "assignment_predicted_bias",
  integrity = "calibration_residual",
  config_integrity = c("replications_per_scenario",
                       "expected_scenarios or simulation.scenarios",
                       "expected_methods or method_roles",
                       "expected_coordinates"),
  scenario_metadata = c("scenario_class", "departure_pattern", "radius_ratio",
                        "within_sensitivity_ball", "is_null",
                        "is_least_favourable", "is_blind",
                        "is_assignment_error")
)
if (frozen_protocol) {
  required_result_columns$runner_v1_results <- c(
    "scenario", "scenario_class", "replication", "method", "coordinate",
    "truth", "estimate", "standard_error", "covered_naive", "width_naive",
    "predicted_bias", "theoretical_scaled_mse", "respondents",
    "sensitivity_margin_true", "sensitivity_margin_hat",
    "covered_bias_additive", "covered_bias_additive_oracle_k",
    "width_bias_additive", "width_bias_additive_oracle_k",
    "family_covered_bias_additive",
    "family_covered_bias_additive_oracle_k",
    "covered_structured_additive", "width_structured_additive",
    "true_assignment", "analysis_assignment", "departure_pattern",
    "normalized_r", "calibration_residual"
  )
  required_result_columns$runner_v1_diagnostics <- c(
    "scenario", "replication", "diagnostic", "coordinate",
    "predicted_power", "rejected_0_05", "stable_rank"
  )
  required_result_columns$runner_v1_config <- c(
    "replications_per_scenario", "simulation.scenarios",
    "estimand.coordinates", "estimand.primary_direction.coordinate",
    "hard_correctness_pass_rules", "practical_value_pass_rules"
  )
}

decision_bias_rows <- if (frozen_protocol) exact_rows else bias_rows
decision_mse_error <- if (frozen_protocol) {
  max_or_na(abs(local_mse_rows$mse_to_theory_ratio - 1))
} else {
  max_or_na(mse_rows$mse_theory_relative_error)
}
decision_aware_coverage <- if (frozen_protocol) {
  min_or_na(oracle_aware$bias_aware_coverage)
} else {
  min_or_na(aware_rows$bias_aware_coverage_lcb_95)
}
decision_hausman_power <- if (frozen_protocol) hausman_power else hpower
decision_assignment_error <- if (frozen_protocol) {
  max_or_na(assignment_exact$bias_prediction_standardized_error)
} else {
  max_or_na(assignment_rows$assignment_bias_standardized_error)
}

decision_name <- if (!production) {
  "smoke_only_no_gate_decision"
} else if (gate_passed) {
  "gate_E2_pass_full_methods_evidence"
} else if (hard_pass) {
  "gate_E2_hard_pass_practical_value_failed"
} else {
  "gate_E2_hard_failed"
}

config_md5 <- if (!is.null(config_path) && file.exists(config_path))
  unname(tools::md5sum(config_path)) else NA_character_
decision <- list(
  gate = "E2_local_misspecification_computation",
  mode = mode,
  frozen_protocol_applied = frozen_protocol,
  production_gate_evaluated = production,
  hard_passed = hard_pass,
  practical_evaluated = practical_evaluated,
  practical_passed = practical_pass,
  passed = gate_passed,
  decision = decision_name,
  strong_aggregate_risk_ratio = strong_risk_boot$point,
  strong_aggregate_risk_ratio_upper_97_5 = strong_risk_boot$upper,
  large_departure_aggregate_risk_ratio = large_risk_boot$point,
  half_radius_length_ratio = half_length_boot$point,
  half_radius_length_ratio_upper_97_5 = half_length_boot$upper,
  unit_radius_length_ratio = unit_length_boot$point,
  maximum_bias_prediction_standardized_error =
    max_or_na(decision_bias_rows$bias_prediction_standardized_error),
  maximum_mse_theory_relative_error =
    decision_mse_error,
  minimum_bias_aware_coverage_lcb =
    decision_aware_coverage,
  minimum_protocol_bias_aware_coverage = decision_aware_coverage,
  minimum_family_coverage_lcb = if (family_available)
    min_or_na(family_in_ball$family_coverage_lcb_95) else NA_real_,
  maximum_hausman_power_error =
    max_or_na(decision_hausman_power$absolute_power_error),
  maximum_assignment_bias_standardized_error =
    decision_assignment_error,
  maximum_calibration_residual = maximum_calibration,
  numerical_failure_rate = failure_rate,
  unavailable_required_components = component$component[
    component$required & !component$available
  ],
  failed_required_components = component$component[
    component$required & component$available & !component$passed
  ],
  method_roles = as.list(method_roles),
  config_md5 = config_md5,
  source_md5 = as.list(if (!is.null(bundle$source_md5))
    bundle$source_md5 else character()),
  generated_at_utc = format(Sys.time(), tz = "UTC", usetz = TRUE)
)
yaml::write_yaml(decision, file.path(result_dir, "gate_e2_decision.yml"))

saveRDS(list(
  decision = decision,
  components = component,
  coordinate_summary = coordinate_summary,
  method_summary = method_summary,
  bias_agreement = bias_agreement,
  mse_summary = mse_summary,
  coverage_summary = coverage_summary,
  hausman_summary = hausman_summary,
  family_summary = family_summary,
  blend_summary = blend_summary,
  protocol_risk_summary = protocol_risk_summary,
  protocol_structured_summary = protocol_structured_summary,
  selector_summary = selector_summary,
  tau_summary = tau_summary,
  departure_summary = departure_summary,
  assignment_summary = assignment_summary,
  completion_summary = completion_summary,
  method_roles = method_role_frame,
  bootstrap = list(
    strong_risk = strong_risk_boot,
    large_departure_risk = large_risk_boot,
    half_radius_length = half_length_boot,
    unit_radius_length = unit_length_boot
  ),
  required_result_columns = required_result_columns,
  rules = rules,
  config = config
), file.path(result_dir, "gate_e2_summary.rds"), version = 3)

cat("Gate E2 components:\n")
print(component, row.names = FALSE)
cat("\nHard pass: ", hard_pass,
    "\nPractical evaluated: ", practical_evaluated,
    "\nPractical pass: ", practical_pass,
    "\nDecision: ", decision_name, "\n", sep = "")
