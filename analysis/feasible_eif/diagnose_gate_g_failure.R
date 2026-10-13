#!/usr/bin/env Rscript

# Post-decision Gate G diagnostic. This script cannot change the frozen gate;
# it quantifies Monte Carlo uncertainty and practical method differences.

script_path <- function() {
  flag <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (!length(flag)) {
    return(normalizePath(
      "analysis/feasible_eif/diagnose_gate_g_failure.R", mustWork = TRUE
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
input <- value_arg(
  "--input",
  file.path(repo_root, "results", "feasible-eif", "gate-g-screen",
            "gate_g_simulation_results.rds")
)
if (!grepl("^/", input)) input <- file.path(repo_root, input)
input <- normalizePath(input, mustWork = TRUE)
output_arg <- value_arg("--output-dir", dirname(input))
output_dir <- if (grepl("^/", output_arg)) output_arg else
  file.path(repo_root, output_arg)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
output_dir <- normalizePath(output_dir, mustWork = TRUE)

if (!requireNamespace("yaml", quietly = TRUE)) {
  stop("Package `yaml` is required.", call. = FALSE)
}
bundle <- readRDS(input)
results <- subset(bundle$results, coordinate == "x1_x2")
config <- bundle$config
draws <- as.integer(config$monte_carlo_bootstrap_draws)
base_seed <- as.integer(config$monte_carlo_bootstrap_seed)

bootstrap_indices <- function(n, statistic, seed) {
  set.seed(seed)
  replicate(draws, statistic(sample.int(n, n, replace = TRUE)))
}

interval_row <- function(component, scenario, method, estimate, values,
                         target = NA_real_) {
  data.frame(
    component = component,
    scenario = scenario,
    method = method,
    estimate = estimate,
    bootstrap_lower = unname(stats::quantile(values, 0.025, na.rm = TRUE)),
    bootstrap_upper = unname(stats::quantile(values, 0.975, na.rm = TRUE)),
    target = target,
    bootstrap_probability_at_or_below_target = if (is.finite(target)) {
      mean(values <= target, na.rm = TRUE)
    } else {
      NA_real_
    },
    stringsAsFactors = FALSE
  )
}

# Fixed-oracle variance and local AMSE uncertainty.
uncertainty <- list()
counter <- 0L
null_ids <- vapply(
  Filter(function(s) identical(s$class, "null"), config$simulation$scenarios),
  `[[`, character(1), "id"
)
for (scenario_id in null_ids) {
  x <- results[
    results$scenario == scenario_id &
      results$method == "oracle_hstar_crossfit",
    , drop = FALSE
  ]
  statistic <- function(index) stats::var(x$error[index]) /
    mean(x$propagated_variance[index])
  estimate <- stats::var(x$error) / mean(x$propagated_variance)
  counter <- counter + 1L
  values <- bootstrap_indices(nrow(x), statistic, base_seed + counter)
  uncertainty[[length(uncertainty) + 1L]] <- interval_row(
    "null_empirical_to_propagated_variance", scenario_id,
    "oracle_hstar_crossfit", estimate, values, 1
  )
}

lf_ids <- vapply(
  Filter(function(s) identical(s$class, "local_least_favourable"),
         config$simulation$scenarios),
  `[[`, character(1), "id"
)
amse_methods <- c(
  "oracle_hstar_crossfit", "oracle_tau_0_25_crossfit",
  "oracle_tau_0_50_crossfit", "oracle_tau_0_75_crossfit"
)
for (scenario_id in lf_ids) {
  for (method_id in amse_methods) {
    x <- results[
      results$scenario == scenario_id & results$method == method_id,
      , drop = FALSE
    ]
    statistic <- function(index) mean(x$error[index]^2) /
      mean(x$predicted_mse_propagated[index])
    estimate <- mean(x$error^2) / mean(x$predicted_mse_propagated)
    counter <- counter + 1L
    values <- bootstrap_indices(nrow(x), statistic, base_seed + counter)
    uncertainty[[length(uncertainty) + 1L]] <- interval_row(
      "local_empirical_to_predicted_amse", scenario_id, method_id,
      estimate, values, 1
    )
  }
}

# Signed and absolute corrected scaling discrepancy. Independent scenario
# streams are resampled independently, as in the frozen screen.
lf_250 <- subset(results, scenario == "gate_g_lf_r1_g250" &
                   method == "oracle_hstar_crossfit")
lf_1000 <- subset(results, scenario == "gate_g_lf_r1_g1000" &
                    method == "oracle_hstar_crossfit")
first_order <- mean(c(lf_250$first_order_scaled_variance,
                      lf_1000$first_order_scaled_variance))
corrected_scaled <- function(data) {
  unique(data$respondents) * stats::var(data$error) -
    (unique(data$respondents) * mean(data$propagated_variance) -
       mean(data$first_order_scaled_variance))
}
observed_signed <- (corrected_scaled(lf_1000) -
                      corrected_scaled(lf_250)) / first_order
set.seed(base_seed + 1000L)
signed_draws <- replicate(draws, {
  ia <- sample.int(nrow(lf_250), nrow(lf_250), replace = TRUE)
  ib <- sample.int(nrow(lf_1000), nrow(lf_1000), replace = TRUE)
  corrected_a <- 250 * stats::var(lf_250$error[ia]) -
    (250 * mean(lf_250$propagated_variance[ia]) -
       mean(lf_250$first_order_scaled_variance[ia]))
  corrected_b <- 1000 * stats::var(lf_1000$error[ib]) -
    (1000 * mean(lf_1000$propagated_variance[ib]) -
       mean(lf_1000$first_order_scaled_variance[ib]))
  (corrected_b - corrected_a) / first_order
})
uncertainty[[length(uncertainty) + 1L]] <- interval_row(
  "corrected_scaled_variance_signed_difference",
  "gate_g_lf_r1_g250_vs_g1000", "oracle_hstar_crossfit",
  observed_signed, signed_draws, 0
)
absolute_row <- interval_row(
  "corrected_scaled_variance_absolute_discrepancy",
  "gate_g_lf_r1_g250_vs_g1000", "oracle_hstar_crossfit",
  abs(observed_signed), abs(signed_draws),
  as.numeric(config$gate_g_pass_rules$corrected_scaled_variance_discrepancy_max)
)
uncertainty[[length(uncertainty) + 1L]] <- absolute_row
uncertainty <- do.call(rbind, uncertainty)

# Complete primary-coordinate performance table.
group_keys <- unique(results[c("scenario", "scenario_class", "respondents",
                               "method")])
performance <- do.call(rbind, lapply(seq_len(nrow(group_keys)), function(i) {
  key <- group_keys[i, ]
  x <- results[
    results$scenario == key$scenario & results$method == key$method,
    , drop = FALSE
  ]
  data.frame(
    key,
    replications = nrow(x),
    bias = mean(x$error),
    variance = stats::var(x$error),
    mse = mean(x$error^2),
    scaled_variance = key$respondents * stats::var(x$error),
    scaled_mse = key$respondents * mean(x$error^2),
    mean_propagated_variance = mean(x$propagated_variance),
    propagated_coverage = mean(x$covered_propagated, na.rm = TRUE),
    plugin_coverage = mean(x$covered_plugin, na.rm = TRUE),
    sandwich_coverage = mean(x$covered_sandwich, na.rm = TRUE),
    stringsAsFactors = FALSE
  )
}))
reference <- performance[
  performance$method %in% c("ordinary_crossfit", "oracle_hstar_crossfit"),
  c("scenario", "method", "variance", "mse")
]
ordinary <- reference[reference$method == "ordinary_crossfit", ]
oracle <- reference[reference$method == "oracle_hstar_crossfit", ]
performance$variance_to_ordinary <- performance$variance /
  ordinary$variance[match(performance$scenario, ordinary$scenario)]
performance$variance_to_oracle_hstar <- performance$variance /
  oracle$variance[match(performance$scenario, oracle$scenario)]
performance$mse_to_ordinary <- performance$mse /
  ordinary$mse[match(performance$scenario, ordinary$scenario)]
performance$mse_to_oracle_hstar <- performance$mse /
  oracle$mse[match(performance$scenario, oracle$scenario)]

utils::write.csv(
  uncertainty,
  file.path(output_dir, "gate_g_posthoc_monte_carlo_uncertainty.csv"),
  row.names = FALSE
)
utils::write.csv(
  performance,
  file.path(output_dir, "gate_g_method_performance.csv"),
  row.names = FALSE
)

diagnostic <- list(
  gate_decision_unchanged = "gate_G_screen_failed",
  purpose = "post-decision Monte Carlo uncertainty and practical-value audit",
  bootstrap_draws = draws,
  bootstrap_seed = base_seed,
  every_fixed_operator_interval_contains_one = all(
    uncertainty$bootstrap_lower[
      uncertainty$component %in% c(
        "null_empirical_to_propagated_variance",
        "local_empirical_to_predicted_amse"
      )
    ] <= 1 &
      uncertainty$bootstrap_upper[
        uncertainty$component %in% c(
          "null_empirical_to_propagated_variance",
          "local_empirical_to_predicted_amse"
        )
      ] >= 1
  ),
  signed_scaling_interval_contains_zero =
    uncertainty$bootstrap_lower[
      uncertainty$component ==
        "corrected_scaled_variance_signed_difference"
    ] <= 0 &&
    uncertainty$bootstrap_upper[
      uncertainty$component ==
        "corrected_scaled_variance_signed_difference"
    ] >= 0,
  formal_gate_may_not_be_overturned = TRUE,
  input = input,
  generated_at_utc = format(Sys.time(), tz = "UTC", usetz = TRUE)
)
yaml::write_yaml(
  diagnostic,
  file.path(output_dir, "gate_g_posthoc_diagnostic.yml")
)

cat("Wrote post-decision Gate-G diagnostic to ", output_dir, "\n", sep = "")
