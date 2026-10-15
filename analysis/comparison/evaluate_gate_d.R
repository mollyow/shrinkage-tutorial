#!/usr/bin/env Rscript
# Evaluate the positive-control calibration (Gate D).
#
#   Rscript analysis/comparison/evaluate_gate_d.R --stage=1   # select multipliers
#   Rscript analysis/comparison/evaluate_gate_d.R --stage=2   # recovery and decision
#
# Stage 1: oracle/additive family-balanced risk for each valid candidate over its
# 100 calibration datasets; among ratios <= 0.75 choose the one closest to 0.70
# (ties to the smaller multiplier). Stage 2: elastic net, forest, and boosting at
# the selected multiplier on the same datasets; a geometry passes if one recovers
# >= 50% of the oracle gain. Gate D passes only if both geometries pass.

args <- commandArgs(trailingOnly = TRUE)
stage_arg <- grep("^--stage=", args, value = TRUE)
stage <- if (length(stage_arg)) as.integer(sub("^--stage=", "", stage_arg[[1]])) else 1L

source(file.path("analysis", "comparison", "load.R"))
root <- getwd()
cmp_require_phase(root, "calibration")
pc <- cmp_load_contracts(root)$simulation$positive_control_calibration
base <- file.path(root, "results", "comparison-v1", "calibration")

read_stage <- function(s) {
  files <- list.files(file.path(base, sprintf("stage-%d", s)), pattern = "^rep_.*\\.rds$",
                      recursive = TRUE, full.names = TRUE)
  cmp_assert(length(files) > 0L, "No stage-%d calibration results.", s)
  do.call(rbind, lapply(files, function(f) readRDS(f)$family_metrics))
}

fm1 <- read_stage(1L)
sel <- cmp_gate_d_selection(fm1, pc)

if (stage == 1L) {
  invalid <- yaml::read_yaml(file.path(base, "stage-1-invalid-candidates.yml"))$invalid
  cmp_write_yaml(list(
    created_utc = cmp_utc_now(),
    rule = pc$selection,
    candidates = lapply(split(sel$table, seq_len(nrow(sel$table))), as.list),
    invalid_candidates = invalid,
    selected_multiplier = sel$selected,
    selection_failed = names(sel$selected)[is.na(unlist(sel$selected))]
  ), file.path(base, "stage-1-selection.yml"))
  print(sel$table)
  cat("Selected:", paste(names(sel$selected), unlist(sel$selected), sep = "=", collapse = ", "), "\n")
  quit(save = "no")
}

dec <- cmp_gate_d_recovery(fm1, read_stage(2L), sel$selected, pc)
# The decision reports the C1 integration-error table at the simulation draw count.
dc_path <- file.path(root, "results", "comparison-v1", "gate-c1", "draw-count-selection.yml")
dc <- yaml::read_yaml(dc_path)
at <- do.call(rbind, lapply(dc$table, as.data.frame))
at <- at[at$draws == dc$selected, ]
fn <- ifelse(is.na(at$multiplier), sprintf("cell_%d", at$cell_id),
             sprintf("cell_%d_multiplier_%s", at$cell_id, at$multiplier))
dc$worst_by_function <- as.list(tapply(at$ratio, fn, max))
collected <- list.files(base, pattern = "\\.(csv|parquet)$", recursive = TRUE, full.names = TRUE)
cmp_write_yaml(list(
  decision_utc = cmp_utc_now(),
  protocol = "factorial_ml_comparison_v1",
  gate = "D",
  gate_d_passed = dec$passed,
  rule = dec$rule,
  selected_multiplier = sel$selected,
  geometries = dec$geometries,
  simulation_integration = list(
    draws = dc$selected,
    worst_family_ratio_by_function = dc$worst_by_function,
    added_score_variance_fraction = "ratio_squared",
    source = sub(paste0("^", root, "/"), "", dc_path)),
  calibration_output_hashes = lapply(collected, function(f) list(
    path = sub(paste0("^", root, "/"), "", f), sha256 = cmp_sha256(f))),
  next_step = if (dec$passed) "hash_calibration_results_then_run_smoke" else
    "stop_v1_benchmark_and_take_software_tutorial_route"
), file.path(root, CMP_GATE_PATHS$gate_d))
cat("Gate D", if (dec$passed) "PASSED" else "FAILED", "\n")
