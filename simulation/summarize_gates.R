#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly = TRUE)
input <- if (length(args)) args[1L] else "results/simulations/production/results.csv"
output <- if (length(args) > 1L) args[2L] else "results/simulations/production/gate-status.json"
config_path <- if (length(args) > 2L) args[3L] else "config/simulation-production.yml"
if (!requireNamespace("jsonlite", quietly = TRUE)) stop("Install the suggested package `jsonlite`.")
if (!requireNamespace("yaml", quietly = TRUE)) stop("Install the suggested package `yaml`.")
if (!file.exists(input)) stop("Simulation result file not found: ", input)
if (!file.exists(config_path)) stop("Simulation configuration not found: ", config_path)

source("simulation/simulation_functions.R")
config <- read_simulation_config(config_path)
expected_scenarios <- expand_simulation_scenarios(config)
raw <- utils::read.csv(input, stringsAsFactors = FALSE)
required_columns <- c(
  "scenario_id", "replication", "status", "method", "risk", "design",
  "n_over_p", "snr", "scale", "within_order"
)
missing_columns <- setdiff(required_columns, names(raw))
if (length(missing_columns)) stop("Result file is missing: ", paste(missing_columns, collapse = ", "))

methods <- c("monotone_order_ridge", "unrestricted_order_ridge")
key_method <- c("scenario_id", "replication", "method")
target_rows <- raw[raw$method %in% methods, , drop = FALSE]
if (any(!is.finite(target_rows$replication)) ||
    any(target_rows$replication < 1L) ||
    any(target_rows$replication != floor(target_rows$replication))) {
  stop("Replication identifiers for gate methods must be positive integers.")
}
if (anyDuplicated(target_rows[key_method])) {
  stop("Duplicate scenario-replication-method rows make paired gate evaluation ambiguous.")
}

m <- target_rows[target_rows$method == methods[1L], c("scenario_id", "replication", "status", "risk")]
u <- target_rows[target_rows$method == methods[2L], c("scenario_id", "replication", "status", "risk")]
names(m)[3:4] <- c("status_monotone", "risk_monotone")
names(u)[3:4] <- c("status_unrestricted", "risk_unrestricted")
pairs_all <- merge(m, u, by = c("scenario_id", "replication"), all = TRUE)
valid_pair <- !is.na(pairs_all$status_monotone) & !is.na(pairs_all$status_unrestricted) &
  pairs_all$status_monotone == "ok" & pairs_all$status_unrestricted == "ok" &
  is.finite(pairs_all$risk_monotone) & is.finite(pairs_all$risk_unrestricted) &
  pairs_all$risk_monotone > 0 & pairs_all$risk_unrestricted > 0
pairs <- pairs_all[valid_pair, , drop = FALSE]

meta_columns <- c("scenario_id", "design", "n_over_p", "snr", "scale", "within_order")
meta <- unique(raw[meta_columns])
if (anyDuplicated(meta$scenario_id)) stop("Scenario metadata are inconsistent within the result file.")
expected_meta <- expected_scenarios[meta_columns]
common_scenarios <- intersect(expected_meta$scenario_id, meta$scenario_id)
metadata_mismatches <- common_scenarios[vapply(common_scenarios, function(id) {
  observed <- meta[meta$scenario_id == id, , drop = FALSE]
  expected <- expected_meta[expected_meta$scenario_id == id, , drop = FALSE]
  fields <- setdiff(meta_columns, "scenario_id")
  any(vapply(fields, function(field) {
    if (is.numeric(expected[[field]]) || is.numeric(observed[[field]])) {
      !isTRUE(all.equal(as.numeric(observed[[field]]),
                        as.numeric(expected[[field]]), tolerance = 1e-12))
    } else {
      !identical(as.character(observed[[field]]), as.character(expected[[field]]))
    }
  }, logical(1)))
}, logical(1))]
pairs <- merge(pairs, meta, by = "scenario_id", all.x = TRUE)
pairs$log_ratio <- log(pairs$risk_monotone / pairs$risk_unrestricted)

summarize_rows <- function(z) {
  n <- nrow(z)
  if (!n) {
    return(c(n = 0, geometric_mean_ratio = NA, lower_95 = NA, upper_95 = NA,
             replication_win_rate = NA, mean_log_ratio = NA,
             variance_log_ratio = NA))
  }
  avg <- mean(z$log_ratio)
  if (n < 2L) {
    lower <- upper <- NA_real_
  } else {
    se <- stats::sd(z$log_ratio) / sqrt(n)
    critical <- stats::qt(.975, n - 1L)
    lower <- exp(avg - critical * se)
    upper <- exp(avg + critical * se)
  }
  c(n = n, geometric_mean_ratio = exp(avg), lower_95 = lower, upper_95 = upper,
    replication_win_rate = mean(z$log_ratio < 0), mean_log_ratio = avg,
    variance_log_ratio = if (n > 1L) stats::var(z$log_ratio) else NA_real_)
}

summarize_equal_cells <- function(z, expected_ids) {
  pieces <- split(z, z$scenario_id)
  if (!length(pieces)) {
    return(c(cells = 0, n = nrow(z),
             geometric_mean_ratio = NA, lower_95 = NA, upper_95 = NA,
             replication_win_rate = NA))
  }
  means <- vapply(pieces, function(x) mean(x$log_ratio), numeric(1))
  counts <- vapply(pieces, nrow, integer(1))
  variances <- vapply(pieces, function(x) if (nrow(x) > 1L) stats::var(x$log_ratio) else NA_real_, numeric(1))
  avg <- mean(means)
  if (any(counts < 2L) || any(!is.finite(variances))) {
    lower <- upper <- NA_real_
  } else {
    components <- variances / counts / length(pieces)^2
    se <- sqrt(sum(components))
    if (se <= sqrt(.Machine$double.eps)) {
      lower <- upper <- exp(avg)
    } else {
      df <- sum(components)^2 / sum(components^2 / (counts - 1L))
      critical <- stats::qt(.975, df = df)
      lower <- exp(avg - critical * se)
      upper <- exp(avg + critical * se)
    }
  }
  c(cells = length(pieces), n = sum(counts), geometric_mean_ratio = exp(avg),
    lower_95 = lower, upper_95 = upper,
    replication_win_rate = mean(unlist(lapply(pieces, function(x) x$log_ratio < 0))))
}

cell_key <- function(x) interaction(x$design, x$n_over_p, x$snr, x$scale,
                                    x$within_order, drop = TRUE, lex.order = TRUE)
cell_split <- split(pairs, cell_key(pairs))
cells <- if (length(cell_split)) {
  do.call(rbind, lapply(cell_split, function(z) {
    cbind(z[1L, c("scenario_id", "design", "n_over_p", "snr", "scale", "within_order")],
          as.data.frame(as.list(summarize_rows(z))))
  }))
} else {
  data.frame(scenario_id = integer(), design = character(), n_over_p = numeric(),
             snr = numeric(), scale = character(), within_order = character(),
             n = integer(), geometric_mean_ratio = numeric(), lower_95 = numeric(),
             upper_95 = numeric(), replication_win_rate = numeric(),
             mean_log_ratio = numeric(), variance_log_ratio = numeric())
}
rownames(cells) <- NULL

expected_counts <- vapply(seq_len(nrow(expected_scenarios)), function(i) {
  scenario_replications(expected_scenarios[i, , drop = FALSE], config)
}, integer(1))
names(expected_counts) <- expected_scenarios$scenario_id
observed_counts <- table(factor(pairs$scenario_id, levels = expected_scenarios$scenario_id))
valid_replications <- split(as.integer(pairs$replication), as.character(pairs$scenario_id))
complete_cells <- vapply(expected_scenarios$scenario_id, function(id) {
  observed <- sort(valid_replications[[as.character(id)]] %||% integer())
  identical(observed, seq_len(expected_counts[as.character(id)]))
}, logical(1))
missing_scenarios <- expected_scenarios$scenario_id[as.integer(observed_counts) == 0L]
incomplete_scenarios <- expected_scenarios$scenario_id[!complete_cells]
unexpected_scenarios <- setdiff(unique(raw$scenario_id), expected_scenarios$scenario_id)

is_primary <- function(x) x$within_order == "dense" & x$scale %in% c("steep", "mild")
is_misspec <- function(x) x$within_order == "dense" & x$scale %in% c("flat", "mild_reversal")
monotone_cells <- cells[is_primary(cells), , drop = FALSE]
monotone_rows <- pairs[is_primary(pairs), , drop = FALSE]
misspec_cells <- cells[is_misspec(cells), , drop = FALSE]
misspec_rows <- pairs[is_misspec(pairs), , drop = FALSE]
expected_primary <- expected_scenarios$scenario_id[is_primary(expected_scenarios)]
expected_misspec <- expected_scenarios$scenario_id[is_misspec(expected_scenarios)]
primary <- summarize_equal_cells(monotone_rows, expected_primary)
robust <- summarize_equal_cells(misspec_rows, expected_misspec)

enough_primary_cells <- length(expected_primary) > 0L &&
  setequal(monotone_cells$scenario_id, expected_primary) &&
  all(complete_cells[match(expected_primary, expected_scenarios$scenario_id)])
enough_misspec_cells <- length(expected_misspec) > 0L &&
  setequal(misspec_cells$scenario_id, expected_misspec) &&
  all(complete_cells[match(expected_misspec, expected_scenarios$scenario_id)])
eligible <- identical(as.character(config$mode), "production") &&
  enough_primary_cells && enough_misspec_cells && all(complete_cells) &&
  !length(unexpected_scenarios) && !length(metadata_mismatches)

fraction_cells_won <- if (nrow(monotone_cells)) {
  mean(monotone_cells$geometric_mean_ratio < 1, na.rm = TRUE)
} else NA_real_
maximum_misspec <- if (nrow(misspec_cells)) {
  max(misspec_cells$geometric_mean_ratio, na.rm = TRUE)
} else NA_real_
criteria <- list(
  primary_geometric_mean_at_most_0_95 = isTRUE(primary["geometric_mean_ratio"] <= .95),
  primary_upper_95_below_0_98 = isTRUE(primary["upper_95"] < .98),
  wins_at_least_75_percent_cells = isTRUE(fraction_cells_won >= .75),
  misspec_aggregate_at_most_1_05 = isTRUE(robust["geometric_mean_ratio"] <= 1.05),
  no_misspec_cell_above_1_10 = isTRUE(maximum_misspec <= 1.10)
)
issues <- character()
if (!identical(as.character(config$mode), "production")) issues <- c(issues, "configuration is not production mode")
if (length(missing_scenarios)) issues <- c(issues, paste0("missing scenarios: ", paste(missing_scenarios, collapse = ",")))
if (length(incomplete_scenarios)) issues <- c(issues, paste0("incomplete scenarios: ", paste(incomplete_scenarios, collapse = ",")))
if (length(unexpected_scenarios)) issues <- c(issues, paste0("unexpected scenarios: ", paste(unexpected_scenarios, collapse = ",")))
if (length(metadata_mismatches)) issues <- c(issues, paste0("scenario metadata mismatch: ", paste(metadata_mismatches, collapse = ",")))
if (any(!valid_pair)) issues <- c(issues, paste0(sum(!valid_pair), " missing, failed, or nonpositive method pairs"))

status <- list(
  generated_at = format(Sys.time(), tz = "UTC", usetz = TRUE),
  input = input, configuration = config_path,
  eligible_for_methods_verdict = eligible,
  expected_replications_by_scenario = as.list(expected_counts),
  primary = as.list(primary), robustness = as.list(robust),
  fraction_primary_cells_won = fraction_cells_won,
  maximum_misspec_cell_ratio = maximum_misspec,
  criteria = criteria,
  full_methods_gate_passed = isTRUE(eligible && all(unlist(criteria))),
  issues = issues,
  metric_definition = paste(
    "Within each scenario cell, the risk ratio is exp(mean(log(MSE_monotone /",
    "MSE_unrestricted))) over paired replications; scenario-cell log ratios",
    "receive equal weight in aggregate criteria."
  ),
  caveat = "Fast, incomplete, failed, duplicated, or non-production results cannot pass the methods gate.",
  cell_results = cells
)
dir.create(dirname(output), recursive = TRUE, showWarnings = FALSE)
jsonlite::write_json(status, output, pretty = TRUE, auto_unbox = TRUE,
                     na = "null", digits = 16)
message("Eligible for methods verdict: ", status$eligible_for_methods_verdict)
message("Full-methods gate passed: ", status$full_methods_gate_passed)
