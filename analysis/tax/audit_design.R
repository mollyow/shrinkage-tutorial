#!/usr/bin/env Rscript

options(stringsAsFactors = FALSE)

abort <- function(...) stop(sprintf(...), call. = FALSE)
assert <- function(ok, ...) if (!isTRUE(ok)) abort(...)
`%||%` <- function(x, y) if (is.null(x)) y else x

arg_value <- function(flag, default = NULL) {
  args <- commandArgs(trailingOnly = TRUE)
  hit <- which(args == flag)
  if (!length(hit)) return(default)
  assert(hit[[1]] < length(args), "Missing value after %s", flag)
  args[[hit[[1]] + 1L]]
}

script_path <- function() {
  hit <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (!length(hit)) return(NA_character_)
  normalizePath(sub("^--file=", "", hit[[1]]), mustWork = TRUE)
}

find_repo_root <- function() {
  candidates <- c(
    normalizePath(getwd(), mustWork = TRUE),
    if (!is.na(script_path())) normalizePath(file.path(dirname(script_path()), "..", ".."), mustWork = TRUE)
  )
  for (candidate in unique(candidates)) {
    if (file.exists(file.path(candidate, "config", "datasets", "income-tax-plans.yml"))) return(candidate)
  }
  abort("Could not locate the repository root containing config/datasets/income-tax-plans.yml")
}

resolve_path <- function(path, root) if (grepl("^/", path)) path else file.path(root, path)
write_csv <- function(x, path) utils::write.csv(x, path, row.names = FALSE, na = "")
respondent_invariant <- function(values, ids) {
  all(vapply(split(values, ids), function(x) {
    token <- ifelse(is.na(x), "<NA>", as.character(x))
    length(unique(token)) == 1L
  }, logical(1)))
}

repo_root <- find_repo_root()
contract_arg <- arg_value("--contract", "config/datasets/income-tax-plans.yml")
output_arg <- arg_value("--output-dir", "analysis/tax/derived")
contract_path <- resolve_path(contract_arg, repo_root)
output_dir <- resolve_path(output_arg, repo_root)

assert(requireNamespace("haven", quietly = TRUE), "Package 'haven' is required")
assert(requireNamespace("tidyselect", quietly = TRUE), "Package 'tidyselect' is required")
assert(requireNamespace("yaml", quietly = TRUE), "Package 'yaml' is required")

contract <- yaml::read_yaml(contract_path)
assert(identical(contract$status, "gate_b_green_for_six_rate_total_schedule_effects_outcomes_prohibited"),
       "Unexpected tax contract status")
assert(identical(contract$gate$result, "green_for_narrow_total_schedule_effects"),
       "Unexpected Gate B result")
assert(identical(contract$estimand$eligible_scope, "six_randomized_rate_total_effects_only"),
       "The eligible estimand must remain the total effect of the six randomized rates")

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
expected_csv_files <- c(
  "all_pair_support_summary.csv",
  "archive_checksums.csv",
  "assignment_diagnostics.csv",
  "contrast_coordinates.csv",
  "covariate_audit.csv",
  "covariate_category_levels.csv",
  "covariate_feature_lock.csv",
  "design_validation.csv",
  "family_identification.csv",
  "family_projectors.csv",
  "level_lock.csv",
  "loaded_columns.csv",
  "opponent_law_diagnostics.csv",
  "profile_probability_strata.csv",
  "profile_uniformity_diagnostics.csv",
  "respondent_folds.csv",
  "revenue_descendant_diagnostics.csv",
  "sample_summary.csv",
  "target_weight_diagnostics.csv"
)
existing_generated <- list.files(output_dir, pattern = "\\.(csv|yml)$", full.names = FALSE)
unexpected_generated <- setdiff(existing_generated, c(expected_csv_files, "design_audit_manifest.yml"))
assert(!length(unexpected_generated),
       "Output directory contains unexpected generated files: %s",
       paste(unexpected_generated, collapse = ", "))
manifest_path <- file.path(output_dir, "design_audit_manifest.yml")
if (file.exists(manifest_path)) unlink(manifest_path)

# Lock every archived input before any design column is read.
file_specs <- contract$provenance$files
archive_checks <- do.call(rbind, lapply(names(file_specs), function(id) {
  spec <- file_specs[[id]]
  path <- resolve_path(spec$path, repo_root)
  assert(file.exists(path), "Missing locked %s file: %s", id, path)
  observed_bytes <- unname(file.info(path)$size)
  observed_md5 <- unname(tools::md5sum(path))
  data.frame(
    file_id = id,
    relative_path = spec$path,
    expected_bytes = as.numeric(spec$bytes),
    observed_bytes = observed_bytes,
    expected_md5 = spec$md5,
    observed_md5 = observed_md5,
    pass = observed_bytes == as.numeric(spec$bytes) && identical(observed_md5, spec$md5)
  )
}))
assert(all(archive_checks$pass), "At least one archive checksum or byte-size lock failed")
write_csv(archive_checks, file.path(output_dir, "archive_checksums.csv"))

# This exact selected-column read is the enforceable outcome barrier. The
# 443-column Stata file is never broad-read and trimmed after the fact.
allowed_columns <- unlist(contract$read_barrier$allowed_columns, use.names = FALSE)
forbidden_outcomes <- unlist(contract$read_barrier$forbidden_outcomes, use.names = FALSE)
forbidden_adjacent <- unlist(contract$read_barrier$forbidden_post_choice_or_outcome_adjacent,
                             use.names = FALSE)
canonical_choice_fields <- c(paste0("pref_plan_", 1:8), "pref_plan", "chose_plan")
assert(all(canonical_choice_fields %in% forbidden_outcomes),
       "All raw and long forced-choice fields must be hard-forbidden")
assert(!any(allowed_columns %in% c(forbidden_outcomes, forbidden_adjacent)),
       "An outcome or outcome-adjacent field appears in the allowlist")
assert(!anyDuplicated(allowed_columns), "The tax allowlist contains duplicate columns")

data_path <- resolve_path(file_specs$data$path, repo_root)
raw_archive <- haven::read_dta(data_path, col_select = tidyselect::all_of(allowed_columns))
assert(setequal(names(raw_archive), allowed_columns), "Selected columns differ from the contract allowlist")
assert(!any(names(raw_archive) %in% c(forbidden_outcomes, forbidden_adjacent)),
       "A forbidden response or response-adjacent field was loaded")
raw_archive$.source_row <- seq_len(nrow(raw_archive))

attribute_ids <- names(contract$attributes)
attribute_columns <- vapply(contract$attributes, `[[`, character(1), "column")
covariate_columns <- unlist(contract$covariates$primary_features, use.names = FALSE)
source_value_labels <- lapply(
  raw_archive[c("plan", "saw_revenue", attribute_columns, contract$revenue_display$column,
                covariate_columns)],
  attr, which = "labels"
)

role_for_column <- function(nm) {
  if (nm == "ID") return("cluster_id")
  if (nm == "table") return("task_id")
  if (nm == "plan") return("profile_side")
  if (nm == "weight") return("survey_weight")
  if (nm == "saw_revenue") return("author_defined_experiment_version")
  if (nm %in% attribute_columns) return("randomized_rate_factor")
  if (nm == contract$revenue_display$column) return("randomized_downstream_mediator_excluded_from_primary_nuisance")
  if (nm %in% covariate_columns) return("primary_preexperiment_feature")
  "audited_excluded"
}
loaded_columns <- data.frame(
  column = allowed_columns,
  role = vapply(allowed_columns, role_for_column, character(1)),
  outcome_loaded = FALSE
)
write_csv(loaded_columns, file.path(output_dir, "loaded_columns.csv"))

numeric_columns <- allowed_columns
for (nm in numeric_columns) raw_archive[[nm]] <- as.numeric(raw_archive[[nm]])

validation <- data.frame(
  check = character(), observed = character(), expected = character(),
  pass = logical(), detail = character()
)
add_check <- function(check, observed, expected, pass, detail = "") {
  validation <<- rbind(validation, data.frame(
    check = check,
    observed = as.character(observed),
    expected = as.character(expected),
    pass = isTRUE(pass),
    detail = detail
  ))
}

expected <- contract$sample$expected
archive_ids <- as.character(raw_archive$ID)
archive_pair_key <- paste(archive_ids, raw_archive$table, sep = "::")
archive_rows_by_id <- table(archive_ids)
archive_tasks_by_id <- tapply(raw_archive$table, archive_ids, function(x) length(unique(x)))
archive_pair_rows <- table(archive_pair_key)

respondent_sources <- c("weight", "saw_revenue", covariate_columns)
source_invariance <- vapply(respondent_sources, function(nm) {
  respondent_invariant(raw_archive[[nm]], raw_archive$ID)
}, logical(1))
assert(all(source_invariance), "At least one respondent-level source variable varies within ID")

primary <- as.data.frame(raw_archive[raw_archive$saw_revenue == 1, , drop = FALSE])
held <- as.data.frame(raw_archive[raw_archive$saw_revenue == 0, , drop = FALSE])
assert(!anyNA(primary[c("ID", "table", "plan", "weight", attribute_columns,
                       contract$revenue_display$column)]),
       "The primary design contains missing identifiers, weights, rates, or revenue categories")

id_chr <- as.character(primary$ID)
pair_key <- paste(id_chr, primary$table, sep = "::")
id_rows <- table(id_chr)
id_tasks <- tapply(primary$table, id_chr, function(x) length(unique(x)))
pair_rows <- table(pair_key)
side_sets <- tapply(primary$plan, pair_key, function(x) identical(sort(as.integer(x)), 1:2))

add_check("archive_profile_rows", nrow(raw_archive), expected$archive_profile_rows,
          nrow(raw_archive) == expected$archive_profile_rows)
add_check("archive_respondents", length(unique(raw_archive$ID)), expected$archive_respondents,
          length(unique(raw_archive$ID)) == expected$archive_respondents)
add_check("archive_tasks", length(archive_pair_rows), expected$archive_tasks,
          length(archive_pair_rows) == expected$archive_tasks)
add_check("archive_rows_per_respondent", paste(range(archive_rows_by_id), collapse = ":"),
          expected$tasks_per_respondent * expected$profiles_per_task,
          all(archive_rows_by_id == expected$tasks_per_respondent * expected$profiles_per_task))
add_check("archive_tasks_per_respondent", paste(range(archive_tasks_by_id), collapse = ":"),
          expected$tasks_per_respondent, all(archive_tasks_by_id == expected$tasks_per_respondent))
add_check("archive_profiles_per_task", paste(range(archive_pair_rows), collapse = ":"),
          expected$profiles_per_task, all(archive_pair_rows == expected$profiles_per_task))
add_check("primary_profile_rows", nrow(primary), expected$primary_profile_rows,
          nrow(primary) == expected$primary_profile_rows)
add_check("primary_respondents", length(unique(primary$ID)), expected$primary_respondents,
          length(unique(primary$ID)) == expected$primary_respondents)
add_check("primary_tasks", length(pair_rows), expected$primary_tasks,
          length(pair_rows) == expected$primary_tasks)
add_check("held_no_revenue_profile_rows", nrow(held), expected$held_no_revenue_profile_rows,
          nrow(held) == expected$held_no_revenue_profile_rows)
add_check("held_no_revenue_respondents", length(unique(held$ID)), expected$held_no_revenue_respondents,
          length(unique(held$ID)) == expected$held_no_revenue_respondents)
add_check("held_no_revenue_tasks", length(unique(paste(held$ID, held$table))),
          expected$held_no_revenue_tasks,
          length(unique(paste(held$ID, held$table))) == expected$held_no_revenue_tasks)
add_check("primary_rows_per_respondent", paste(range(id_rows), collapse = ":"), 16,
          all(id_rows == 16L))
add_check("primary_tasks_per_respondent", paste(range(id_tasks), collapse = ":"), 8,
          all(id_tasks == 8L))
add_check("task_levels", paste(sort(unique(primary$table)), collapse = ","),
          paste(1:8, collapse = ","), identical(sort(as.integer(unique(primary$table))), 1:8))
add_check("profiles_per_task", paste(range(pair_rows), collapse = ":"), 2,
          all(pair_rows == 2L))
add_check("profile_sides_A_and_B", sum(side_sets), length(side_sets), all(side_sets))
add_check("unique_ID_task_side_keys", anyDuplicated(primary[c("ID", "table", "plan")]), 0,
          anyDuplicated(primary[c("ID", "table", "plan")]) == 0L)
add_check("respondent_invariant_sources", sum(source_invariance), length(source_invariance),
          all(source_invariance))
add_check("positive_finite_primary_weights",
          sum(is.finite(primary$weight) & primary$weight > 0), nrow(primary),
          all(is.finite(primary$weight) & primary$weight > 0))

sample_summary <- data.frame(
  archive_respondents = length(unique(raw_archive$ID)),
  archive_tasks = length(archive_pair_rows),
  archive_profile_rows = nrow(raw_archive),
  primary_respondents = length(unique(primary$ID)),
  primary_tasks = length(pair_rows),
  primary_profile_rows = nrow(primary),
  held_no_revenue_respondents = length(unique(held$ID)),
  held_no_revenue_tasks = length(unique(paste(held$ID, held$table))),
  held_no_revenue_profile_rows = nrow(held),
  tasks_per_primary_respondent_min = min(id_tasks),
  tasks_per_primary_respondent_max = max(id_tasks),
  profiles_per_task_min = min(pair_rows),
  profiles_per_task_max = max(pair_rows),
  all_primary_design_columns_complete = !anyNA(primary[c("ID", "table", "plan", "weight",
                                                          attribute_columns,
                                                          contract$revenue_display$column)])
)
write_csv(sample_summary, file.path(output_dir, "sample_summary.csv"))

# Lock the six randomized rate-factor levels and the downstream displayed
# revenue categories separately. Revenue is not added to attribute_ids.
level_lock <- data.frame()
for (id in attribute_ids) {
  spec <- contract$attributes[[id]]
  codes <- as.numeric(unlist(spec$codes, use.names = FALSE))
  labels <- as.character(unlist(spec$labels, use.names = FALSE))
  observed_codes <- sort(unique(primary[[spec$column]]))
  assert(identical(observed_codes, codes), "Level-code lock failed for %s", id)
  source_labels <- source_value_labels[[spec$column]]
  source_by_code <- setNames(names(source_labels), as.character(as.numeric(source_labels)))
  counts <- table(factor(primary[[spec$column]], levels = codes))
  level_lock <- rbind(level_lock, data.frame(
    component_type = "randomized_rate_factor",
    component = id,
    column = spec$column,
    code = codes,
    contract_label = labels,
    source_label = unname(source_by_code[as.character(codes)]),
    observed_count_primary = as.integer(counts),
    target_probability = 1 / length(codes)
  ))
}
revenue_codes <- as.numeric(unlist(contract$revenue_display$codes, use.names = FALSE))
revenue_labels <- as.character(unlist(contract$revenue_display$labels, use.names = FALSE))
revenue_column <- contract$revenue_display$column
assert(identical(sort(unique(primary[[revenue_column]])), revenue_codes),
       "Revenue category code lock failed")
revenue_source_labels <- source_value_labels[[revenue_column]]
revenue_source_by_code <- setNames(names(revenue_source_labels),
                                   as.character(as.numeric(revenue_source_labels)))
revenue_counts <- table(factor(primary[[revenue_column]], levels = revenue_codes))
level_lock <- rbind(level_lock, data.frame(
  component_type = "randomized_downstream_mediator",
  component = "revenue_display",
  column = revenue_column,
  code = revenue_codes,
  contract_label = revenue_labels,
  source_label = unname(revenue_source_by_code[as.character(revenue_codes)]),
  observed_count_primary = as.integer(revenue_counts),
  target_probability = NA_real_
))
write_csv(level_lock, file.path(output_dir, "level_lock.csv"))

# Every pair of rate factors must have complete Cartesian support, pooled and
# within each task and side. Full 7,680-schedule empirical coverage is not
# required for this randomized-design estimand.
support_strata <- c(
  list(pooled = rep(TRUE, nrow(primary))),
  setNames(lapply(1:8, function(task) primary$table == task), paste0("task::", 1:8)),
  setNames(lapply(1:2, function(side) primary$plan == side), paste0("side::", c("A", "B")))
)
all_pair_support <- data.frame()
for (ids in utils::combn(attribute_ids, 2L, simplify = FALSE)) {
  a <- contract$attributes[[ids[[1]]]]
  b <- contract$attributes[[ids[[2]]]]
  a_codes <- as.numeric(unlist(a$codes, use.names = FALSE))
  b_codes <- as.numeric(unlist(b$codes, use.names = FALSE))
  for (stratum in names(support_strata)) {
    idx <- support_strata[[stratum]]
    tab <- table(factor(primary[[a$column]][idx], levels = a_codes),
                 factor(primary[[b$column]][idx], levels = b_codes))
    all_pair_support <- rbind(all_pair_support, data.frame(
      stratum = stratum,
      attribute_a = ids[[1]],
      attribute_b = ids[[2]],
      expected_cartesian_cells = length(a_codes) * length(b_codes),
      observed_cells = sum(tab > 0),
      minimum_cell_count = min(tab),
      maximum_cell_count = max(tab),
      complete_cartesian = all(tab > 0)
    ))
  }
}
assert(all(all_pair_support$complete_cartesian), "At least one factor-pair support is incomplete")
add_check("all_factor_pair_supports_complete", sum(all_pair_support$complete_cartesian),
          nrow(all_pair_support), all(all_pair_support$complete_cartesian))
write_csv(all_pair_support, file.path(output_dir, "all_pair_support_summary.csv"))

# Uniform one-way assignment diagnostics by pooled sample, task, and side.
# These diagnose conflicts; source documentation and the prospective lock,
# not realized balance, supply the assignment law.
assignment_diagnostics <- data.frame()
for (id in attribute_ids) {
  spec <- contract$attributes[[id]]
  codes <- as.numeric(unlist(spec$codes, use.names = FALSE))
  for (stratum in names(support_strata)) {
    idx <- support_strata[[stratum]]
    counts <- table(factor(primary[[spec$column]][idx], levels = codes))
    n <- sum(counts)
    expected_count <- n / length(codes)
    assignment_diagnostics <- rbind(assignment_diagnostics, data.frame(
      component = id,
      column = spec$column,
      stratum = stratum,
      level_code = codes,
      observed_count = as.integer(counts),
      expected_count = expected_count,
      observed_probability = as.integer(counts) / n,
      expected_probability = 1 / length(codes),
      pearson_residual = (as.integer(counts) - expected_count) / sqrt(expected_count)
    ))
  }
}
max_assignment_residual <- max(abs(assignment_diagnostics$pearson_residual))
add_check("maximum_absolute_uniform_assignment_residual",
          format(max_assignment_residual, digits = 8), "<4", max_assignment_residual < 4,
          "Prespecified diagnostic; not a substitute for randomization documentation.")
write_csv(assignment_diagnostics, file.path(output_dir, "assignment_diagnostics.csv"))

# Complete-schedule occupancy under the locked q over all 7,680 schedules.
factor_levels <- as.integer(unlist(contract$support$factor_levels, use.names = FALSE))
number_schedules <- as.integer(contract$support$schedule_profiles)
assert(prod(factor_levels) == number_schedules, "Factor-level product does not equal schedule support")
schedule_key <- do.call(paste, c(lapply(primary[attribute_columns], as.integer), sep = "|"))
schedule_grid <- expand.grid(lapply(factor_levels, seq_len), KEEP.OUT.ATTRS = FALSE)
schedule_universe <- do.call(paste, c(schedule_grid, sep = "|"))
schedule_counts <- table(factor(schedule_key, levels = schedule_universe))
expected_schedule_count <- nrow(primary) / number_schedules
schedule_chisq <- sum((as.numeric(schedule_counts) - expected_schedule_count)^2 /
                        expected_schedule_count)
schedule_df <- number_schedules - 1L
schedule_p <- stats::pchisq(schedule_chisq, schedule_df, lower.tail = FALSE)
expected_occupied <- number_schedules * (1 - (1 - 1 / number_schedules)^nrow(primary))
occupied_probability <- 1 - (1 - 1 / number_schedules)^nrow(primary)
both_occupied_probability <- 1 - 2 * (1 - 1 / number_schedules)^nrow(primary) +
  (1 - 2 / number_schedules)^nrow(primary)
occupied_variance <- number_schedules * occupied_probability * (1 - occupied_probability) +
  number_schedules * (number_schedules - 1) *
  (both_occupied_probability - occupied_probability^2)
occupied_sd <- sqrt(occupied_variance)
observed_occupied <- sum(schedule_counts > 0)
occupied_z <- (observed_occupied - expected_occupied) / occupied_sd
assert(observed_occupied == as.integer(contract$support$observed_distinct_schedules_primary),
       "Observed complete-schedule occupancy differs from the lock")
profile_uniformity_diagnostics <- data.frame(
  profile_rows = nrow(primary),
  schedule_support_size = number_schedules,
  observed_distinct_schedules = observed_occupied,
  unobserved_schedules = sum(schedule_counts == 0),
  expected_occupied_under_iid_rows = expected_occupied,
  occupied_standard_deviation_under_iid_rows = occupied_sd,
  occupied_z_score = occupied_z,
  expected_count_per_schedule = expected_schedule_count,
  minimum_count_including_zeros = min(schedule_counts),
  maximum_count = max(schedule_counts),
  pearson_chisq = schedule_chisq,
  degrees_of_freedom = schedule_df,
  iid_row_reference_p_value = schedule_p,
  interpretation = "Descriptive iid-row compatibility diagnostic; source-supported prospective q supplies identification."
)
add_check("observed_distinct_schedules", observed_occupied,
          contract$support$observed_distinct_schedules_primary,
          observed_occupied == as.integer(contract$support$observed_distinct_schedules_primary))
add_check("complete_schedule_occupancy_z_score", format(occupied_z, digits = 8),
          "absolute value <4", abs(occupied_z) < 4,
          "Exact first two occupancy moments under iid q; source lock, not this diagnostic, supplies q.")
write_csv(profile_uniformity_diagnostics,
          file.path(output_dir, "profile_uniformity_diagnostics.csv"))

# q x q opponent diagnostics. Explicit side labels permit direct cross-side
# tables. Zero identical pairs is plausible here and does not contradict the
# source-supported independent-pair reconstruction.
side_a <- primary[primary$plan == 1, , drop = FALSE]
side_b <- primary[primary$plan == 2, , drop = FALSE]
side_a <- side_a[order(side_a$ID, side_a$table), , drop = FALSE]
side_b <- side_b[order(side_b$ID, side_b$table), , drop = FALSE]
assert(nrow(side_a) == nrow(side_b) &&
         identical(as.numeric(side_a$ID), as.numeric(side_b$ID)) &&
         identical(as.numeric(side_a$table), as.numeric(side_b$table)),
       "Plan A and Plan B task keys do not align")
schedule_a <- do.call(paste, c(lapply(side_a[attribute_columns], as.integer), sep = "|"))
schedule_b <- do.call(paste, c(lapply(side_b[attribute_columns], as.integer), sep = "|"))
number_tasks <- nrow(side_a)
identical_pairs <- sum(schedule_a == schedule_b)
expected_identical <- number_tasks / number_schedules
probability_zero_identical <- (1 - 1 / number_schedules)^number_tasks

opponent_law_diagnostics <- data.frame(
  diagnostic = "identical_complete_schedule_pairs",
  component = "complete_schedule",
  level_a = NA_integer_,
  level_b = NA_integer_,
  observed = identical_pairs,
  expected_under_q_times_q = expected_identical,
  pearson_residual = (identical_pairs - expected_identical) /
    sqrt(expected_identical * (1 - 1 / number_schedules)),
  probability_or_candidate_value = probability_zero_identical,
  status = "compatible_with_source_supported_q_times_q_reconstruction"
)

for (ja in seq_along(attribute_ids)) {
  id_a <- attribute_ids[[ja]]
  spec_a <- contract$attributes[[id_a]]
  codes_a <- as.numeric(unlist(spec_a$codes, use.names = FALSE))
  for (jb in seq_along(attribute_ids)) {
    id_b <- attribute_ids[[jb]]
    spec_b <- contract$attributes[[id_b]]
    codes_b <- as.numeric(unlist(spec_b$codes, use.names = FALSE))
    tab <- table(factor(side_a[[spec_a$column]], levels = codes_a),
                 factor(side_b[[spec_b$column]], levels = codes_b))
    expected_cell <- number_tasks / (length(codes_a) * length(codes_b))
    cell_grid <- expand.grid(level_a = codes_a, level_b = codes_b,
                             KEEP.OUT.ATTRS = FALSE)
    opponent_law_diagnostics <- rbind(opponent_law_diagnostics, data.frame(
      diagnostic = "cross_side_level_cell",
      component = paste(id_a, id_b, sep = "::"),
      level_a = cell_grid$level_a,
      level_b = cell_grid$level_b,
      observed = as.integer(tab),
      expected_under_q_times_q = expected_cell,
      pearson_residual = (as.integer(tab) - expected_cell) / sqrt(expected_cell),
      probability_or_candidate_value = NA_real_,
      status = "compatible_with_independent_uniform_side_draws"
    ))
  }
}

for (j in seq_along(attribute_ids)) {
  id <- attribute_ids[[j]]
  spec <- contract$attributes[[id]]
  codes <- as.numeric(unlist(spec$codes, use.names = FALSE))
  matches <- sum(side_a[[spec$column]] == side_b[[spec$column]])
  p_match <- 1 / length(codes)
  opponent_law_diagnostics <- rbind(opponent_law_diagnostics, data.frame(
    diagnostic = "same_factor_level_across_sides",
    component = id,
    level_a = NA_integer_,
    level_b = NA_integer_,
    observed = matches,
    expected_under_q_times_q = number_tasks * p_match,
    pearson_residual = (matches - number_tasks * p_match) /
      sqrt(number_tasks * p_match * (1 - p_match)),
    probability_or_candidate_value = NA_real_,
    status = "compatible_with_independent_uniform_side_draws"
  ))
}

repeated_schedule_rows_within_respondent <- sum(vapply(
  split(schedule_key, primary$ID),
  function(x) length(x) - length(unique(x)),
  integer(1)
))
opponent_law_diagnostics <- rbind(opponent_law_diagnostics, data.frame(
  diagnostic = "repeated_schedule_rows_across_tasks_within_respondent",
  component = "complete_schedule",
  level_a = NA_integer_, level_b = NA_integer_,
  observed = repeated_schedule_rows_within_respondent,
  expected_under_q_times_q = NA_real_,
  pearson_residual = NA_real_,
  probability_or_candidate_value = NA_real_,
  status = "confirms_no_blanket_respondent_level_no_repeat_rule"
))

cross_side_rows <- opponent_law_diagnostics$diagnostic == "cross_side_level_cell"
max_cross_side_residual <- max(abs(opponent_law_diagnostics$pearson_residual[cross_side_rows]))
add_check("identical_complete_schedule_pairs", identical_pairs, 0, identical_pairs == 0,
          sprintf("Under q x q, E=%.6f and P(zero)=%.6f.",
                  expected_identical, probability_zero_identical))
add_check("zero_identical_pairs_compatible_with_q_times_q", probability_zero_identical,
          ">0.05", probability_zero_identical > 0.05)
add_check("maximum_absolute_cross_side_cell_residual",
          format(max_cross_side_residual, digits = 8), "<4", max_cross_side_residual < 4)
add_check("schedule_repeats_across_tasks_exist", repeated_schedule_rows_within_respondent,
          ">0", repeated_schedule_rows_within_respondent > 0)
write_csv(opponent_law_diagnostics, file.path(output_dir, "opponent_law_diagnostics.csv"))

profile_probability_strata <- data.frame(
  law = c("focal_uniform_product_q", "within_task_ordered_pair_q_times_q"),
  reconstruction_status = c(
    "source_supported_prospective_generator_not_released",
    "source_supported_prospective_generator_not_released"
  ),
  schedule_support_size = c(number_schedules, number_schedules),
  ordered_pair_support_size = c(NA_real_, number_schedules^2),
  q_schedule = c(1 / number_schedules, 1 / number_schedules),
  ordered_pair_probability = c(NA_real_, 1 / number_schedules^2),
  structural_zeros = c(0L, 0L)
)
write_csv(profile_probability_strata, file.path(output_dir, "profile_probability_strata.csv"))

# Revenue is a noisy descendant of the six assigned rates. Variation in the
# displayed category among repeated identical schedules verifies that it is
# neither deterministic in the archive nor an independently randomized
# seventh factor. The headline nuisance model must not condition on it.
schedule_groups <- split(seq_len(nrow(primary)), schedule_key)
schedule_group_n <- vapply(schedule_groups, length, integer(1))
schedule_revenue_levels <- vapply(schedule_groups, function(ix) {
  length(unique(primary[[revenue_column]][ix]))
}, integer(1))
repeated_schedule <- schedule_group_n > 1L
varying_revenue <- schedule_revenue_levels > 1L
revenue_descendant_diagnostics <- data.frame(
  diagnostic = c(
    "observed_revenue_categories",
    "revenue_missing_rows",
    "observed_distinct_rate_schedules",
    "rate_schedules_observed_more_than_once",
    "repeated_rate_schedules_with_multiple_revenue_categories",
    "rows_in_rate_schedules_with_multiple_revenue_categories",
    "maximum_revenue_categories_within_one_rate_schedule"
  ),
  observed = c(
    length(unique(primary[[revenue_column]])),
    sum(is.na(primary[[revenue_column]])),
    length(schedule_groups),
    sum(repeated_schedule),
    sum(varying_revenue),
    sum(schedule_group_n[varying_revenue]),
    max(schedule_revenue_levels)
  ),
  expected_or_rule = c(
    "5", "0", "7554", ">0", ">0", ">0", ">1"
  ),
  pass = c(
    length(unique(primary[[revenue_column]])) == 5L,
    !anyNA(primary[[revenue_column]]),
    length(schedule_groups) == as.integer(contract$support$observed_distinct_schedules_primary),
    sum(repeated_schedule) > 0,
    sum(varying_revenue) > 0,
    sum(schedule_group_n[varying_revenue]) > 0,
    max(schedule_revenue_levels) > 1L
  ),
  implication = c(
    "All displayed bins occur.",
    "The downstream display is observed for every primary-arm profile.",
    "Schedule occupancy matches the design lock.",
    "The same randomized schedule is replicated in the realized design.",
    "Revenue includes schedule-conditional noise and is not a deterministic seventh factor.",
    "These rows directly expose the noisy descendant mechanism.",
    "At least one identical schedule maps to multiple displayed bins."
  )
)
assert(all(revenue_descendant_diagnostics$pass), "A revenue-descendant diagnostic failed")
add_check("same_schedule_multiple_revenue_categories", sum(varying_revenue), ">0",
          sum(varying_revenue) > 0,
          "Primary estimand is the total schedule effect; do not condition on taxrev.")
write_csv(revenue_descendant_diagnostics,
          file.path(output_dir, "revenue_descendant_diagnostics.csv"))

# One row per primary-arm respondent for target weights, covariates, and folds.
respondents <- primary[!duplicated(primary$ID), c("ID", "weight", covariate_columns), drop = FALSE]
respondents <- as.data.frame(respondents)
assert(nrow(respondents) == expected$primary_respondents, "Primary respondent extraction failed")

feature_semantics <- contract$covariates$feature_semantics
assert(setequal(names(feature_semantics), covariate_columns),
       "Feature semantics must cover every and only primary features")
covariate_feature_lock <- do.call(rbind, lapply(covariate_columns, function(nm) {
  spec <- feature_semantics[[nm]]
  data.frame(
    variable = nm,
    semantic_type = spec$semantic_type,
    primary_encoding = spec$encoding %||% "numeric",
    missing_value_rule = spec$missing_value_rule,
    scaling_rule = spec$scaling_rule %||% "none"
  )
}))
write_csv(covariate_feature_lock, file.path(output_dir, "covariate_feature_lock.csv"))

categorical_features <- setdiff(covariate_columns, "age")
covariate_category_levels <- do.call(rbind, lapply(categorical_features, function(nm) {
  observed_codes <- sort(unique(respondents[[nm]][!is.na(respondents[[nm]])]))
  source_labels <- source_value_labels[[nm]]
  source_by_code <- if (length(source_labels)) {
    setNames(names(source_labels), as.character(as.numeric(source_labels)))
  } else {
    setNames(character(), character())
  }
  counts <- table(factor(respondents[[nm]], levels = observed_codes))
  labels <- unname(source_by_code[as.character(observed_codes)])
  labels[is.na(labels)] <- as.character(observed_codes[is.na(labels)])
  rbind(
    data.frame(
      variable = nm,
      level_order = seq_along(observed_codes),
      source_code = as.character(observed_codes),
      source_label = labels,
      enrolled_count = as.integer(counts),
      is_missing_level = FALSE
    ),
    data.frame(
      variable = nm,
      level_order = length(observed_codes) + 1L,
      source_code = "__MISSING__",
      source_label = "__MISSING__",
      enrolled_count = sum(is.na(respondents[[nm]])),
      is_missing_level = TRUE
    )
  )
}))
write_csv(covariate_category_levels,
          file.path(output_dir, "covariate_category_levels.csv"))

locked_missing <- as.integer(unlist(contract$covariates$expected_missing_primary,
                                    use.names = FALSE))
names(locked_missing) <- names(contract$covariates$expected_missing_primary)
covariate_audit <- do.call(rbind, lapply(covariate_columns, function(nm) {
  x <- respondents[[nm]]
  data.frame(
    variable = nm,
    classification = "primary_feature",
    respondents = nrow(respondents),
    missing_n = sum(is.na(x)),
    missing_fraction = mean(is.na(x)),
    unique_nonmissing = length(unique(x[!is.na(x)])),
    minimum = if (all(is.na(x))) NA_real_ else min(x, na.rm = TRUE),
    maximum = if (all(is.na(x))) NA_real_ else max(x, na.rm = TRUE),
    respondent_invariant = source_invariance[[nm]]
  )
}))
for (nm in names(locked_missing)) {
  observed_missing <- covariate_audit$missing_n[covariate_audit$variable == nm]
  add_check(paste0("covariate_missing_", nm), observed_missing, locked_missing[[nm]],
            length(observed_missing) == 1L && observed_missing == locked_missing[[nm]])
}
add_check("primary_feature_count", length(covariate_columns), 7,
          length(covariate_columns) == 7L)
add_check("age_range", paste(range(respondents$age), collapse = ":"), "19:91",
          identical(as.numeric(range(respondents$age)), c(19, 91)))
write_csv(covariate_audit, file.path(output_dir, "covariate_audit.csv"))

respondents$target_weight <- respondents$weight / sum(respondents$weight)
decile_breaks <- stats::quantile(respondents$weight, probs = 0:10 / 10, type = 8,
                                 names = FALSE)
assert(length(unique(decile_breaks)) == 11L, "Released-weight deciles do not have unique cut points")
respondents$weight_decile <- as.integer(cut(
  respondents$weight, breaks = decile_breaks, include.lowest = TRUE, labels = FALSE
))
assert(identical(sort(unique(respondents$weight_decile)), 1:10),
       "Weight-decile construction did not produce ten strata")
target_weight_diagnostics <- do.call(rbind, lapply(1:10, function(decile) {
  idx <- respondents$weight_decile == decile
  w <- respondents$weight[idx]
  data.frame(
    weight_decile = decile,
    respondents = sum(idx),
    raw_weight_sum = sum(w),
    target_weight_sum = sum(respondents$target_weight[idx]),
    effective_sample_size = sum(w)^2 / sum(w^2),
    minimum_raw_weight = min(w),
    maximum_raw_weight = max(w)
  )
}))
add_check("target_weight_total", sum(respondents$target_weight), 1,
          abs(sum(respondents$target_weight) - 1) < 1e-12)
write_csv(target_weight_diagnostics,
          file.path(output_dir, "target_weight_diagnostics.csv"))

# Stable respondent-cluster folds, balanced within released-weight decile.
stable_string_hash <- function(x, salt, multiplier = 131, reverse_codes = FALSE) {
  vapply(x, function(token) {
    h <- as.numeric(salt)
    codes <- utf8ToInt(enc2utf8(token))
    if (reverse_codes) codes <- rev(codes)
    for (code in codes) h <- (h * multiplier + code) %% 2147483629
    h
  }, numeric(1))
}
assign_balanced_folds <- function(ids, k, salt, multiplier = 131, reverse_codes = FALSE) {
  ord <- order(stable_string_hash(ids, salt, multiplier, reverse_codes), ids)
  out <- integer(length(ids))
  out[ord] <- rep(seq_len(k), length.out = length(ids))
  out
}
outer_k <- as.integer(contract$folds$outer)
inner_k <- as.integer(contract$folds$inner_labels)
fold_seed <- as.integer(contract$folds$deterministic_seed)
assert(length(fold_seed) == 1L && is.finite(fold_seed), "A deterministic fold seed is required")
respondents$outer_fold <- ave(
  seq_len(nrow(respondents)), respondents$weight_decile,
  FUN = function(ix) assign_balanced_folds(as.character(respondents$ID[ix]), outer_k, fold_seed)
)
respondents$inner_fold_label <- ave(
  seq_len(nrow(respondents)), respondents$weight_decile,
  FUN = function(ix) assign_balanced_folds(as.character(respondents$ID[ix]), inner_k,
                                           fold_seed + 104729L, 257, TRUE)
)
respondents$outer_fold <- as.integer(respondents$outer_fold)
respondents$inner_fold_label <- as.integer(respondents$inner_fold_label)
outer_balance <- vapply(split(respondents$outer_fold, respondents$weight_decile),
                        function(x) diff(range(tabulate(x, nbins = outer_k))) <= 1L,
                        logical(1))
inner_balance <- vapply(split(respondents$inner_fold_label, respondents$weight_decile),
                        function(x) diff(range(tabulate(x, nbins = inner_k))) <= 1L,
                        logical(1))
nested_coverage <- unlist(lapply(seq_len(outer_k), function(held_out) {
  vapply(split(respondents, respondents$weight_decile), function(x) {
    train <- x[x$outer_fold != held_out, , drop = FALSE]
    length(unique(train$inner_fold_label)) == inner_k
  }, logical(1))
}))
row_fold_match <- match(as.character(primary$ID), as.character(respondents$ID))
add_check("all_primary_rows_map_to_one_respondent_fold", sum(!is.na(row_fold_match)),
          nrow(primary), !anyNA(row_fold_match))
add_check("outer_fold_weight_decile_balance", sum(outer_balance), length(outer_balance),
          all(outer_balance))
add_check("inner_fold_weight_decile_balance", sum(inner_balance), length(inner_balance),
          all(inner_balance))
add_check("inner_fold_coverage_in_every_outer_training_set", sum(nested_coverage),
          length(nested_coverage), all(nested_coverage))
respondent_folds <- respondents[, c(
  "ID", "weight", "target_weight", "weight_decile", "outer_fold", "inner_fold_label"
)]
write_csv(respondent_folds, file.path(output_dir, "respondent_folds.csv"))

# Uniform-target, coding-invariant bases and projectors for all coordinates in
# each prospectively registered two-rate interaction family.
helmert_function_basis <- function(n_levels) {
  h <- stats::contr.helmert(n_levels)
  sweep(h, 2L, sqrt(colSums(h^2) / n_levels), "/")
}

family_specs <- Filter(
  function(x) is.list(x) && !is.null(x$attributes),
  contract$interaction_families
)
family_identification <- data.frame()
contrast_coordinates <- data.frame()
family_projectors <- data.frame()

for (family_id in names(family_specs)) {
  family <- family_specs[[family_id]]
  ids <- unlist(family$attributes, use.names = FALSE)
  assert(length(ids) == 2L && all(ids %in% attribute_ids),
         "Family %s must name exactly two locked rate factors", family_id)
  a <- contract$attributes[[ids[[1]]]]
  b <- contract$attributes[[ids[[2]]]]
  a_codes <- as.numeric(unlist(a$codes, use.names = FALSE))
  b_codes <- as.numeric(unlist(b$codes, use.names = FALSE))
  grid <- expand.grid(a_code = a_codes, b_code = b_codes,
                      KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE)
  grid$cell_index <- seq_len(nrow(grid))
  observed <- table(factor(primary[[a$column]], levels = a_codes),
                    factor(primary[[b$column]], levels = b_codes))
  observed_count <- as.integer(observed)

  ua <- helmert_function_basis(length(a_codes))
  ub <- helmert_function_basis(length(b_codes))
  basis <- matrix(NA_real_, nrow = nrow(grid), ncol = ncol(ua) * ncol(ub))
  coordinate_labels <- character(ncol(basis))
  cc <- 0L
  for (jb in seq_len(ncol(ub))) {
    for (ja in seq_len(ncol(ua))) {
      cc <- cc + 1L
      basis[, cc] <- ua[match(grid$a_code, a_codes), ja] *
        ub[match(grid$b_code, b_codes), jb]
      coordinate_labels[[cc]] <- sprintf("h%02d_x_h%02d", ja, jb)
    }
  }
  p <- rep(1 / nrow(grid), nrow(grid))
  w <- diag(p)
  main <- cbind(
    intercept = 1,
    ua[match(grid$a_code, a_codes), , drop = FALSE],
    ub[match(grid$b_code, b_codes), , drop = FALSE]
  )
  gram_error <- max(abs(crossprod(basis, w %*% basis) - diag(ncol(basis))))
  main_error <- max(abs(crossprod(main, w %*% basis)))
  projector <- basis %*% t(basis) %*% w
  idempotence_error <- max(abs(projector %*% projector - projector))
  observed_rank <- qr(basis, tol = 1e-10)$rank
  expected_rank <- as.integer(family$dimension)

  task_min <- Inf
  side_min <- Inf
  for (task in 1:8) {
    idx <- primary$table == task
    tab <- table(factor(primary[[a$column]][idx], levels = a_codes),
                 factor(primary[[b$column]][idx], levels = b_codes))
    assert(all(tab > 0), "Family %s is incomplete in task %s", family_id, task)
    task_min <- min(task_min, tab)
  }
  for (side in 1:2) {
    idx <- primary$plan == side
    tab <- table(factor(primary[[a$column]][idx], levels = a_codes),
                 factor(primary[[b$column]][idx], levels = b_codes))
    assert(all(tab > 0), "Family %s is incomplete on side %s", family_id, side)
    side_min <- min(side_min, tab)
  }

  assert(nrow(grid) == as.integer(family$cells), "Cell lock failed for %s", family_id)
  assert(observed_rank == expected_rank, "Interaction rank lock failed for %s", family_id)
  assert(all(observed_count > 0), "An interaction cell is empty for %s", family_id)
  assert(gram_error < 1e-10 && main_error < 1e-10 && idempotence_error < 1e-10,
         "Invariant basis checks failed for %s", family_id)

  family_identification <- rbind(family_identification, data.frame(
    family = family_id,
    attribute_a = ids[[1]],
    attribute_b = ids[[2]],
    cells = nrow(grid),
    observed_cells = sum(observed_count > 0),
    minimum_cell_count = min(observed_count),
    minimum_task_cell_count = as.integer(task_min),
    minimum_side_cell_count = as.integer(side_min),
    additive_rank = 1L + length(a_codes) - 1L + length(b_codes) - 1L,
    expected_interaction_rank = expected_rank,
    observed_interaction_rank = observed_rank,
    orthonormality_error = gram_error,
    main_effect_orthogonality_error = main_error,
    projector_idempotence_error = idempotence_error,
    identified = observed_rank == expected_rank && all(observed_count > 0)
  ))
  for (j in seq_len(ncol(basis))) {
    contrast_coordinates <- rbind(contrast_coordinates, data.frame(
      family = family_id,
      coordinate = coordinate_labels[[j]],
      cell_index = grid$cell_index,
      a_code = grid$a_code,
      b_code = grid$b_code,
      target_probability = p,
      basis_value = basis[, j],
      direct_contrast_weight = p * basis[, j]
    ))
  }
  projector_grid <- expand.grid(
    row_cell_index = seq_len(nrow(grid)),
    column_cell_index = seq_len(nrow(grid)),
    KEEP.OUT.ATTRS = FALSE
  )
  family_projectors <- rbind(family_projectors, data.frame(
    family = family_id,
    projector_grid,
    value = as.vector(projector)
  ))
}

add_check("registered_family_ranks", sum(family_identification$identified),
          nrow(family_identification), all(family_identification$identified))
add_check("total_registered_dimension", sum(family_identification$observed_interaction_rank),
          50, sum(family_identification$observed_interaction_rank) == 50L)
add_check("registered_family_task_support", min(family_identification$minimum_task_cell_count),
          ">0", all(family_identification$minimum_task_cell_count > 0))
add_check("registered_family_side_support", min(family_identification$minimum_side_cell_count),
          ">0", all(family_identification$minimum_side_cell_count > 0))
write_csv(family_identification, file.path(output_dir, "family_identification.csv"))
write_csv(contrast_coordinates, file.path(output_dir, "contrast_coordinates.csv"))
write_csv(family_projectors, file.path(output_dir, "family_projectors.csv"))

write_csv(validation, file.path(output_dir, "design_validation.csv"))
assert(all(validation$pass), "At least one tax design-contract validation failed; inspect design_validation.csv")

generated_paths <- file.path(output_dir, expected_csv_files)
assert(all(file.exists(generated_paths)), "At least one expected audit output was not generated")
generated_md5 <- unname(tools::md5sum(generated_paths))
names(generated_md5) <- expected_csv_files
manifest <- list(
  schema_version = 1,
  dataset_id = contract$dataset_id,
  contract_schema_version = contract$schema_version,
  contract_md5 = unname(tools::md5sum(contract_path)),
  completed_utc = format(Sys.time(), tz = "UTC", usetz = TRUE),
  status = "passed",
  gate = "B",
  gate_result = "green_for_narrow_total_schedule_effects",
  portfolio_eligible_scope = "six_randomized_rate_total_effects_only",
  outcome_blind = TRUE,
  outcomes_loaded = FALSE,
  outcome_models_fit = FALSE,
  read_method = "haven::read_dta(col_select=locked_allowlist)",
  primary_respondents = nrow(respondents),
  primary_tasks = number_tasks,
  primary_profile_rows = nrow(primary),
  schedule_support = number_schedules,
  registered_family_dimensions = as.list(setNames(
    family_identification$observed_interaction_rank,
    family_identification$family
  )),
  assignment_law = list(
    q_times_q_locked = TRUE,
    source_supported_reconstruction = TRUE,
    original_generator_released = FALSE,
    observed_identical_pairs = identical_pairs,
    expected_identical_pairs_under_q_times_q = expected_identical,
    probability_zero_under_iid_task_pairs = probability_zero_identical,
    maximum_absolute_cross_side_cell_residual = max_cross_side_residual,
    across_task_dependence = "arbitrary_within_respondent_subject_to_declared_mean_no_carryover"
  ),
  revenue = list(
    causal_role = "randomized_downstream_mediator_not_seventh_factor",
    repeated_schedules_with_multiple_revenue_categories = sum(varying_revenue),
    rows_in_schedules_with_multiple_revenue_categories = sum(schedule_group_n[varying_revenue]),
    primary_nuisance_action = "do_not_condition_on_taxrev"
  ),
  generated_files_md5 = as.list(generated_md5),
  note = paste(
    "No pref_plan_1--pref_plan_8, pref_plan, chose_plan, ideal-rate outcome,",
    "post-choice justification, timing, or quality-control value was loaded, summarized, or modeled."
  )
)
yaml::write_yaml(manifest, manifest_path)

cat(sprintf(
  paste0(
    "Tax design audit passed outcome-blind: %s respondents, %s paired tasks, %s profiles, ",
    "q over 7,680 schedules, q x q opponent diagnostics, and family ranks 15/15/20. ",
    "Eligibility is narrow: total effects of the six rates under the natural noisy revenue display.\n"
  ),
  format(nrow(respondents), big.mark = ","),
  format(number_tasks, big.mark = ","),
  format(nrow(primary), big.mark = ",")
))
