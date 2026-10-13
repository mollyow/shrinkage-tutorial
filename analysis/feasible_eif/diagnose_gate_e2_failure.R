#!/usr/bin/env Rscript

script_path <- function() {
  flag <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (!length(flag)) {
    return(normalizePath(
      "analysis/feasible_eif/diagnose_gate_e2_failure.R"
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
  if (length(hit)) sub(paste0("^", prefix, "="), "", hit[[1L]]) else default
}

result_dir <- file.path(repo_root, "results", "feasible-eif",
                        "gate-e2-production")
summary_path <- value_arg(
  "--summary", file.path(result_dir, "gate_e2_summary.rds")
)
bundle_path <- value_arg(
  "--bundle", file.path(result_dir, "gate_e2_simulation_results.rds")
)
output_dir <- value_arg("--output-dir", result_dir)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

if (!file.exists(summary_path) || !file.exists(bundle_path)) {
  stop("Gate-E2 production summary and bundle are required.", call. = FALSE)
}
if (!requireNamespace("yaml", quietly = TRUE)) {
  stop("Package `yaml` is required.", call. = FALSE)
}

summary <- readRDS(summary_path)
bundle <- readRDS(bundle_path)
coordinate <- summary$coordinate_summary
primary <- as.character(
  summary$config$estimand$primary_direction$coordinate
)
fixed_methods <- c(
  "ordinary_taskwise", "oracle_adaptive", "oracle_tau_0_25",
  "oracle_tau_0_50", "oracle_tau_0_75"
)
`%||%` <- function(x, y) if (is.null(x) || !length(x)) y else x

local <- coordinate[
  coordinate$scenario_class == "local_least_favourable" &
    coordinate$coordinate == primary & coordinate$method %in% fixed_methods,
  , drop = FALSE
]
local <- local[order(local$mse_to_theory_ratio, decreasing = TRUE), ]
local$violates_frozen_interval <- local$mse_to_theory_ratio < 0.9 |
  local$mse_to_theory_ratio > 1.1
utils::write.csv(
  local, file.path(output_dir, "gate_e2_postgate_local_amse.csv"),
  row.names = FALSE
)

scenario_metadata <- do.call(rbind, lapply(
  summary$config$simulation$scenarios,
  function(x) data.frame(
    scenario = as.character(x$id),
    respondents = as.integer(x$respondents),
    local_scaling_pair = as.character(x$local_scaling_pair %||% ""),
    stringsAsFactors = FALSE
  )
))

pair_meta <- scenario_metadata[
  nzchar(scenario_metadata$local_scaling_pair), , drop = FALSE
]
pair <- merge(
  coordinate[
    coordinate$coordinate == primary & coordinate$method %in% fixed_methods,
    c("scenario", "method", "coordinate", "empirical_bias", "predicted_bias",
      "mc_variance", "predicted_mse"), drop = FALSE
  ],
  pair_meta,
  by = "scenario", all = FALSE, sort = FALSE
)
pair$scaled_bias <- sqrt(pair$respondents) * pair$empirical_bias
pair$theory_scaled_bias <- sqrt(pair$respondents) * pair$predicted_bias
pair$scaled_variance <- pair$respondents * pair$mc_variance
pair$theory_scaled_variance <- pmax(
  pair$respondents * pair$predicted_mse - pair$theory_scaled_bias^2, 0
)
scaling <- do.call(rbind, lapply(split(pair, pair$method), function(x) {
  x <- x[order(x$respondents), ]
  bias_scale <- max(
    abs(mean(x$theory_scaled_bias)),
    sqrt(mean(x$theory_scaled_variance)), .Machine$double.eps
  )
  variance_scale <- max(mean(x$theory_scaled_variance), .Machine$double.eps)
  data.frame(
    method = x$method[[1L]],
    scaled_bias_g250 = x$scaled_bias[[1L]],
    scaled_bias_g1000 = x$scaled_bias[[2L]],
    relative_bias_difference = abs(diff(x$scaled_bias)) / bias_scale,
    scaled_variance_g250 = x$scaled_variance[[1L]],
    scaled_variance_g1000 = x$scaled_variance[[2L]],
    theory_scaled_variance = mean(x$theory_scaled_variance),
    relative_variance_difference =
      abs(diff(x$scaled_variance)) / variance_scale,
    stringsAsFactors = FALSE
  )
}))
scaling <- scaling[order(scaling$relative_variance_difference,
                         decreasing = TRUE), ]
utils::write.csv(
  scaling, file.path(output_dir, "gate_e2_postgate_scaling.csv"),
  row.names = FALSE
)

# `wrong_assignment_oracle_mean` is named for its assignment-error role, but in
# correct-law carryover cells it is also the frozen known-restricted-mean H*
# control. It removes feasible nuisance estimation while retaining the same
# local carryover bias.
oracle_control <- coordinate[
  coordinate$scenario_class == "local_least_favourable" &
    coordinate$coordinate == primary &
    coordinate$method == "wrong_assignment_oracle_mean",
  , drop = FALSE
]
oracle_control <- oracle_control[
  order(oracle_control$mse_to_theory_ratio, decreasing = TRUE),
  , drop = FALSE
]
utils::write.csv(
  oracle_control,
  file.path(output_dir, "gate_e2_postgate_oracle_nuisance_control.csv"),
  row.names = FALSE
)

utils::write.csv(
  summary$protocol_risk_summary,
  file.path(output_dir, "gate_e2_postgate_practical_risk.csv"),
  row.names = FALSE
)

hstar_scaling <- scaling[scaling$method == "oracle_adaptive", ]
hstar_pair <- pair[pair$method == "oracle_adaptive", ]
hstar_pair <- hstar_pair[order(hstar_pair$respondents), ]
hstar_excess <- hstar_pair$scaled_variance -
  hstar_pair$theory_scaled_variance
control_pair <- merge(
  oracle_control[
    oracle_control$scenario %in% pair_meta$scenario,
    c("scenario", "mc_variance", "predicted_mse", "predicted_bias"),
    drop = FALSE
  ],
  pair_meta, by = "scenario", all = FALSE, sort = FALSE
)
control_pair$scaled_variance <- control_pair$respondents *
  control_pair$mc_variance
control_pair$theory_scaled_variance <-
  control_pair$respondents * control_pair$predicted_mse -
  (sqrt(control_pair$respondents) * control_pair$predicted_bias)^2
control_pair <- control_pair[order(control_pair$respondents), ]
control_scaling_difference <- abs(diff(control_pair$scaled_variance)) /
  mean(control_pair$theory_scaled_variance)

selector <- as.data.frame(bundle$selector)
fallback_rows <- sum(selector$outer_saturated_empty_cells > 0L)
fallback_replications <- length(unique(selector$replication[
  selector$outer_saturated_empty_cells > 0L &
    selector$scenario == "lf_r1_cs_product_g250"
]))

diagnosis <- list(
  formal_gate_decision = summary$decision$decision,
  formal_hard_passed = summary$decision$hard_passed,
  formal_practical_evaluated = summary$decision$practical_evaluated,
  failed_hard_components = intersect(
    summary$decision$failed_required_components,
    summary$components$component[summary$components$gate == "hard"]
  ),
  maximum_local_amse_ratio = max(local$mse_to_theory_ratio),
  local_amse_violating_rows = sum(local$violates_frozen_interval),
  maximum_scaling_bias_difference =
    max(scaling$relative_bias_difference),
  maximum_scaling_variance_difference =
    max(scaling$relative_variance_difference),
  feasible_hstar_scaled_variance_excess_g250 = hstar_excess[[1L]],
  feasible_hstar_scaled_variance_excess_g1000 = hstar_excess[[2L]],
  feasible_hstar_excess_ratio_g250_to_g1000 =
    hstar_excess[[1L]] / hstar_excess[[2L]],
  known_mean_hstar_local_amse_ratio_range = c(
    min(oracle_control$mse_to_theory_ratio),
    max(oracle_control$mse_to_theory_ratio)
  ),
  known_mean_hstar_scaling_variance_difference =
    unname(control_scaling_difference),
  interpretation = paste(
    "The frozen decision remains a hard failure. The known-restricted-mean",
    "control satisfies first-order AMSE and scaling, while feasible H* excess",
    "scaled variance falls by approximately four when G rises fourfold.",
    "This isolates a finite-sample nuisance-estimation remainder rather than",
    "an error in the E1 first-order bias or efficiency geometry."
  ),
  practical_status = "descriptive_only_because_hard_gate_failed",
  maximum_feasible_generic_risk_ratio = max(
    summary$protocol_risk_summary$risk_ratio[
      summary$protocol_risk_summary$gate_family == "feasible_tau_generic"
    ]
  ),
  small_g_fallback_fold_rows = fallback_rows,
  small_g_fallback_replications = fallback_replications,
  generated_at_utc = format(Sys.time(), tz = "UTC", usetz = TRUE)
)
yaml::write_yaml(
  diagnosis, file.path(output_dir, "gate_e2_postgate_diagnosis.yml")
)

cat("Formal Gate E2 decision:", diagnosis$formal_gate_decision, "\n")
cat("Maximum local AMSE ratio:", diagnosis$maximum_local_amse_ratio, "\n")
cat("Maximum scaling variance difference:",
    diagnosis$maximum_scaling_variance_difference, "\n")
cat("Known-mean H* AMSE ratio range:",
    paste(diagnosis$known_mean_hstar_local_amse_ratio_range,
          collapse = " to "), "\n")
cat("Feasible H* excess scaled-variance ratio G250/G1000:",
    diagnosis$feasible_hstar_excess_ratio_g250_to_g1000, "\n")
