#!/usr/bin/env Rscript

options(stringsAsFactors = FALSE)

abort <- function(...) stop(sprintf(...), call. = FALSE)
assert <- function(ok, ...) if (!isTRUE(ok)) abort(...)

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
    if (file.exists(file.path(candidate, "config", "datasets", "graham-svolik.yml"))) return(candidate)
  }
  abort("Could not locate the repository root containing config/datasets/graham-svolik.yml")
}

resolve_path <- function(path, root) if (grepl("^/", path)) path else file.path(root, path)
write_csv <- function(x, path) utils::write.csv(as.data.frame(x), path, row.names = FALSE, na = "")
same_values <- function(x, y) {
  all((is.na(x) & is.na(y)) |
        (!is.na(x) & !is.na(y) & as.character(x) == as.character(y)))
}
respondent_invariant <- function(values, ids) {
  all(vapply(split(values, ids), function(x) {
    token <- ifelse(is.na(x), "<NA>", as.character(x))
    length(unique(token)) == 1L
  }, logical(1)))
}

repo_root <- find_repo_root()
contract_arg <- arg_value("--contract", "config/datasets/graham-svolik.yml")
output_arg <- arg_value("--output-dir", "analysis/graham_svolik/derived")
contract_path <- resolve_path(contract_arg, repo_root)
output_dir <- resolve_path(output_arg, repo_root)

assert(requireNamespace("data.table", quietly = TRUE), "Package 'data.table' is required")
assert(requireNamespace("yaml", quietly = TRUE), "Package 'yaml' is required")

contract <- yaml::read_yaml(contract_path)
assert(identical(contract$status, "gate_b_green_narrow_coarsened_democracy_outcomes_prohibited"),
       "Unexpected Graham-Svolik contract status")
assert(isTRUE(contract$decision$outcome_blind_audit), "The contract must require an outcome-blind audit")
assert(identical(contract$gate$result, "green_for_narrow_coarsened_democracy_families"),
       "Unexpected narrow Gate-B result")
assert(isTRUE(contract$protocol$pre_audit_deviation$occurred),
       "The one-row pre-audit outcome-display deviation must remain disclosed")

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
expected_csv_files <- c(
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
  "nuisance_context_audit.csv",
  "respondent_folds.csv",
  "sample_summary.csv",
  "target_cells.csv",
  "target_weight_diagnostics.csv"
)
existing_generated <- list.files(output_dir, pattern = "\\.(csv|yml)$", full.names = FALSE)
unexpected_generated <- setdiff(existing_generated, c(expected_csv_files, "design_audit_manifest.yml"))
assert(!length(unexpected_generated),
       "Output directory contains unexpected generated files: %s",
       paste(unexpected_generated, collapse = ", "))
manifest_path <- file.path(output_dir, "design_audit_manifest.yml")
if (file.exists(manifest_path)) unlink(manifest_path)

# Verify original-format archive bytes before reading any design values.
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

# The schema-only reads expose names but no record values. Every value-bearing
# read below uses an explicit allowlist. In particular, no response column is
# ever materialized by this audit.
experiment_path <- resolve_path(file_specs$experiment$path, repo_root)
covariate_path <- resolve_path(file_specs$covariates$path, repo_root)
weights_path <- resolve_path(file_specs$weights$path, repo_root)

experiment_allowed <- unlist(contract$read_barrier$experiment_allowed_columns, use.names = FALSE)
covariate_allowed <- unlist(contract$read_barrier$covariate_allowed_columns, use.names = FALSE)
weights_allowed <- unlist(contract$read_barrier$weights_allowed_columns, use.names = FALSE)
forbidden_outcomes <- unlist(contract$read_barrier$forbidden_experiment_outcomes, use.names = FALSE)
forbidden_derived <- unlist(contract$read_barrier$forbidden_derived_or_post_choice, use.names = FALSE)

experiment_header <- names(data.table::fread(experiment_path, nrows = 0L, showProgress = FALSE))
covariate_header <- names(data.table::fread(covariate_path, nrows = 0L, showProgress = FALSE))
weights_header <- names(data.table::fread(weights_path, nrows = 0L, showProgress = FALSE))
assert(all(experiment_allowed %in% experiment_header), "An experiment allowlist column is absent")
assert(all(covariate_allowed %in% covariate_header), "A covariate allowlist column is absent")
assert(all(weights_allowed %in% weights_header), "A weight allowlist column is absent")
assert(all(forbidden_outcomes %in% experiment_header), "A canonical forbidden outcome is absent from the schema")
assert(!any(experiment_allowed %in% c(forbidden_outcomes, forbidden_derived)),
       "A forbidden response, response derivative, or post-choice field appears in the experiment allowlist")
assert(!any(grepl("(^|_)(win|vote)($|_)", experiment_allowed, ignore.case = TRUE)),
       "A win/vote field appears in the experiment allowlist")
assert(!anyDuplicated(experiment_allowed) && !anyDuplicated(covariate_allowed) &&
         !anyDuplicated(weights_allowed), "At least one read allowlist contains duplicates")

experiment <- data.table::fread(
  experiment_path, select = experiment_allowed, showProgress = FALSE
)
covariates <- data.table::fread(
  covariate_path, select = covariate_allowed, showProgress = FALSE
)
weights_source <- data.table::fread(
  weights_path, select = weights_allowed, showProgress = FALSE
)
assert(setequal(names(experiment), experiment_allowed), "Experiment selected columns differ from the lock")
assert(setequal(names(covariates), covariate_allowed), "Covariate selected columns differ from the lock")
assert(setequal(names(weights_source), weights_allowed), "Weight selected columns differ from the lock")
assert(!any(names(experiment) %in% c(forbidden_outcomes, forbidden_derived)),
       "A forbidden experiment field was loaded")
experiment[, .source_row := .I]

experiment_roles <- function(nm) {
  if (nm == "id") return("cluster_id")
  if (nm == "weight") return("survey_weight")
  if (nm == "matchNum") return("released_scenario_id_not_actual_display_order")
  if (nm == "candNum") return("mirrored_candidate_orientation")
  if (nm %in% c("c_party", "o_party", "c_dem_code", "o_dem_code", "c_dem_type",
                "o_dem_type", "c_sex", "o_sex")) return("primary_factor_source_or_opponent_context")
  "randomized_current_task_context_empirically_standardized"
}
loaded_columns <- rbind(
  data.frame(
    source_file = "experiment",
    column = experiment_allowed,
    role = vapply(experiment_allowed, experiment_roles, character(1)),
    response_value_loaded = FALSE
  ),
  data.frame(
    source_file = "covariates",
    column = covariate_allowed,
    role = ifelse(covariate_allowed %in% c("id", "rid"), "link_key", "pre_choice_feature"),
    response_value_loaded = FALSE
  ),
  data.frame(
    source_file = "weights",
    column = weights_allowed,
    role = c("link_key", "survey_weight"),
    response_value_loaded = FALSE
  )
)
write_csv(loaded_columns, file.path(output_dir, "loaded_columns.csv"))

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
design_required <- setdiff(experiment_allowed, character())
id_chr <- as.character(experiment$id)
task_key <- paste(id_chr, experiment$matchNum, sep = "::")
row_key <- paste(task_key, experiment$candNum, sep = "::")
rows_by_id <- table(id_chr)
tasks_by_id <- tapply(experiment$matchNum, id_chr, function(x) length(unique(x)))
rows_by_task <- table(task_key)
candidates_by_task <- tapply(experiment$candNum, task_key, function(x) {
  identical(sort(as.integer(x)), 1:2)
})

add_check("released_rows", nrow(experiment), expected$released_task_rows,
          nrow(experiment) == expected$released_task_rows)
add_check("released_respondents", data.table::uniqueN(experiment$id), expected$released_respondents,
          data.table::uniqueN(experiment$id) == expected$released_respondents)
add_check("released_ids", paste(range(experiment$id), collapse = ":"), "1:1682",
          identical(sort(unique(as.integer(experiment$id))), seq_len(expected$released_respondents)))
add_check("released_tasks", length(rows_by_task), expected$released_tasks,
          length(rows_by_task) == expected$released_tasks)
add_check("unique_id_task_candidate_key", data.table::uniqueN(row_key), nrow(experiment),
          data.table::uniqueN(row_key) == nrow(experiment))
add_check("rows_per_respondent", paste(range(rows_by_id), collapse = ":"), 26,
          all(rows_by_id == 26L))
add_check("tasks_per_respondent", paste(range(tasks_by_id), collapse = ":"),
          expected$released_tasks_per_respondent,
          all(tasks_by_id == expected$released_tasks_per_respondent))
add_check("scenario_levels", paste(sort(unique(experiment$matchNum)), collapse = ","),
          paste(1:13, collapse = ","),
          identical(as.integer(sort(unique(experiment$matchNum))), 1:13))
add_check("mirrored_rows_per_task", paste(range(rows_by_task), collapse = ":"),
          expected$mirrored_rows_per_task, all(rows_by_task == expected$mirrored_rows_per_task))
add_check("candidate_orientations_per_task", sum(candidates_by_task), length(candidates_by_task),
          all(candidates_by_task))
add_check("complete_design_allowlist", sum(is.na(experiment[, ..design_required])), 0,
          !anyNA(experiment[, ..design_required]))
add_check("positive_finite_survey_weights", sum(is.finite(experiment$weight) & experiment$weight > 0),
          nrow(experiment), all(is.finite(experiment$weight) & experiment$weight > 0))
add_check("respondent_invariant_survey_weight", respondent_invariant(experiment$weight, experiment$id),
          TRUE, respondent_invariant(experiment$weight, experiment$id))

# Verify that the two released rows in a task are exact focal/opponent mirrors.
row1 <- experiment[candNum == 1][order(id, matchNum)]
row2 <- experiment[candNum == 2][order(id, matchNum)]
assert(nrow(row1) == nrow(row2) && same_values(row1$id, row2$id) &&
         same_values(row1$matchNum, row2$matchNum), "Mirrored task keys do not align")
mirror_stems <- c(
  "party", "Party", "p1_code", "p1_area", "p1_num", "p2_code", "p2_area", "p2_num",
  "dem_code", "dem_type", "sex", "race", "pro", "age", "exper", "onLeft", "respParty"
)
mirror_pass <- vapply(mirror_stems, function(stem) {
  same_values(row1[[paste0("c_", stem)]], row2[[paste0("o_", stem)]]) &&
    same_values(row1[[paste0("o_", stem)]], row2[[paste0("c_", stem)]])
}, logical(1))
add_check("all_randomized_fields_mirror_across_candidate_rows", sum(mirror_pass),
          length(mirror_pass), all(mirror_pass), paste(names(mirror_pass)[!mirror_pass], collapse = ","))
add_check("focal_and_opponent_left_right_are_complements", sum(row1$c_onLeft + row1$o_onLeft == 1),
          nrow(row1), all(row1$c_onLeft + row1$o_onLeft == 1))

# Work in the source-defined candNum==1 orientation and remove the two
# randomized valence tasks without consulting a response.
cand1_all <- data.table::copy(row1)
cand1_all[, contest_relation := ifelse(c_party == o_party, "same", "cross")]
cand1_all[, contest_code := paste0(c_party, o_party)]
primary <- data.table::copy(cand1_all[c_dem_type != "v" & o_dem_type != "v"])
primary[, democracy_behavior := ifelse(c_dem_type == "g", "generic", c_dem_code)]
primary[, contest_relation := ifelse(c_party == o_party, "same", "cross")]
primary[, focal_gender := c_sex]

primary_rows_by_id <- table(primary$id)
primary_task_key <- paste(primary$id, primary$matchNum, sep = "::")
add_check("cand1_oriented_tasks", nrow(cand1_all), expected$released_tasks,
          nrow(cand1_all) == expected$released_tasks)
add_check("omitted_valence_tasks", nrow(cand1_all) - nrow(primary), expected$omitted_valence_tasks,
          nrow(cand1_all) - nrow(primary) == expected$omitted_valence_tasks)
add_check("primary_tasks", nrow(primary), expected$primary_tasks,
          nrow(primary) == expected$primary_tasks)
add_check("primary_respondents", data.table::uniqueN(primary$id), expected$primary_respondents,
          data.table::uniqueN(primary$id) == expected$primary_respondents)
add_check("primary_tasks_per_respondent", paste(range(primary_rows_by_id), collapse = ":"),
          expected$primary_tasks_per_respondent,
          all(primary_rows_by_id == expected$primary_tasks_per_respondent))
add_check("primary_unique_respondent_task", data.table::uniqueN(primary_task_key), nrow(primary),
          data.table::uniqueN(primary_task_key) == nrow(primary))
add_check("primary_democracy_types", paste(sort(unique(primary$c_dem_type)), collapse = ","), "g,u",
          identical(sort(unique(primary$c_dem_type)), c("g", "u")))
add_check("primary_opponent_is_always_generic", sum(primary$o_dem_type == "g"), nrow(primary),
          all(primary$o_dem_type == "g"))

democracy_codes <- unlist(contract$factors$democracy_behavior$codes, use.names = FALSE)
undemocratic_codes <- democracy_codes[democracy_codes != "generic"]
exact_block_counts <- function(data, value_column, levels, expected_counts) {
  by_id <- split(data[[value_column]], data$id)
  all(vapply(by_id, function(x) {
    observed <- table(factor(x, levels = levels))
    identical(as.integer(observed), as.integer(expected_counts))
  }, logical(1)))
}
democracy_block_ok <- exact_block_counts(
  primary, "democracy_behavior", democracy_codes,
  c(4L, rep(1L, length(undemocratic_codes)))
)
# DR/RD orientations vary, so separately lock the exact same/cross schedule.
relation_block_ok <- exact_block_counts(cand1_all, "contest_relation", c("same", "cross"), c(6L, 7L))
same_party_composition_ok <- all(vapply(split(cand1_all$contest_code, cand1_all$id), function(x) {
  sum(x == "DD") == 3L && sum(x == "RR") == 3L && sum(x %in% c("DR", "RD")) == 7L
}, logical(1)))
valence_block_ok <- all(vapply(split(cand1_all$c_dem_code, cand1_all$id), function(x) {
  sum(x == "v_tax") == 1L && sum(x == "v_affair") == 1L
}, logical(1)))
add_check("four_generic_plus_each_of_seven_U_once_per_respondent", democracy_block_ok, TRUE,
          democracy_block_ok)
add_check("six_same_and_seven_cross_party_tasks_per_respondent", relation_block_ok, TRUE,
          relation_block_ok)
add_check("three_DD_three_RR_seven_cross_tasks_per_respondent", same_party_composition_ok, TRUE,
          same_party_composition_ok)
add_check("each_valence_behavior_once_per_respondent_before_exclusion", valence_block_ok, TRUE,
          valence_block_ok)

# Check the separate respondent and weight keys without opening any endpoint.
add_check("covariate_rows", nrow(covariates), expected$covariate_rows,
          nrow(covariates) == expected$covariate_rows)
add_check("covariate_rows_missing_experiment_id", sum(is.na(covariates$id)),
          expected$covariate_rows_with_missing_experiment_id,
          sum(is.na(covariates$id)) == expected$covariate_rows_with_missing_experiment_id)
linked_covariates <- data.table::copy(covariates[!is.na(id)])
add_check("linked_covariate_ids", data.table::uniqueN(linked_covariates$id), expected$linked_covariate_ids,
          data.table::uniqueN(linked_covariates$id) == expected$linked_covariate_ids)
add_check("one_covariate_row_per_linked_id", nrow(linked_covariates), expected$linked_covariate_ids,
          nrow(linked_covariates) == expected$linked_covariate_ids && !anyDuplicated(linked_covariates$id))
experiment_ids <- as.character(unique(experiment$id))
covariate_ids <- as.character(unique(linked_covariates$id))
id_symmetric_difference <- c(setdiff(experiment_ids, covariate_ids),
                             setdiff(covariate_ids, experiment_ids))
add_check("experiment_and_covariate_id_sets_agree", length(id_symmetric_difference),
          0, setequal(experiment_ids, covariate_ids))
add_check("weight_rows", nrow(weights_source), expected$covariate_rows,
          nrow(weights_source) == expected$covariate_rows)
add_check("unique_weight_rid", data.table::uniqueN(weights_source$rid), nrow(weights_source),
          !anyDuplicated(weights_source$rid))
add_check("unique_covariate_rid", data.table::uniqueN(covariates$rid), nrow(covariates),
          !anyDuplicated(covariates$rid))

cov_weight <- merge(covariates[, .(id, rid)], weights_source, by = "rid", all.x = TRUE, sort = FALSE)
assert(nrow(cov_weight) == nrow(covariates) && !anyNA(cov_weight$weight),
       "The separate weight file does not map one-to-one to covariate rows")
respondents <- unique(cand1_all[, .(id, weight)])
respondents <- merge(respondents, cov_weight[!is.na(id), .(id, source_weight = weight)],
                     by = "id", all.x = TRUE, sort = FALSE)
add_check("experiment_weights_match_separate_weight_file",
          max(abs(respondents$weight - respondents$source_weight)), 0,
          max(abs(respondents$weight - respondents$source_weight)) < 1e-12)
respondents[, source_weight := NULL]
respondents[, target_weight := weight / sum(weight)]

sample_summary <- data.frame(
  released_experiment_rows = nrow(experiment),
  released_respondents = data.table::uniqueN(experiment$id),
  released_tasks = length(rows_by_task),
  released_tasks_per_respondent_min = min(tasks_by_id),
  released_tasks_per_respondent_max = max(tasks_by_id),
  primary_respondents = data.table::uniqueN(primary$id),
  primary_tasks = nrow(primary),
  primary_tasks_per_respondent_min = min(primary_rows_by_id),
  primary_tasks_per_respondent_max = max(primary_rows_by_id),
  omitted_valence_tasks = nrow(cand1_all) - nrow(primary),
  covariate_rows = nrow(covariates),
  linked_covariate_ids = data.table::uniqueN(linked_covariates$id),
  covariate_rows_missing_id = sum(is.na(covariates$id)),
  supplement_wave2_completers = contract$sample$source_count_discrepancy$supplement_wave2_completers,
  article_respondents = contract$sample$source_count_discrepancy$article_respondents,
  article_choices = contract$sample$source_count_discrepancy$article_choices
)
write_csv(sample_summary, file.path(output_dir, "sample_summary.csv"))

# Lock factor levels, the exact current-task q, and the uniform 64-cell target.
factor_columns <- c(
  democracy_behavior = "democracy_behavior",
  focal_party = "c_party",
  contest_relation = "contest_relation",
  focal_gender = "focal_gender"
)
assignment_probabilities <- list(
  democracy_behavior = setNames(c(4 / 11, rep(1 / 11, 7)), democracy_codes),
  focal_party = c(D = 1 / 2, R = 1 / 2),
  contest_relation = c(same = 6 / 13, cross = 7 / 13),
  focal_gender = c(Female = 3 / 8, Male = 5 / 8)
)

level_lock <- data.frame()
for (id in names(factor_columns)) {
  spec <- contract$factors[[id]]
  codes <- as.character(unlist(spec$codes, use.names = FALSE))
  labels <- as.character(unlist(spec$labels, use.names = FALSE))
  values <- as.character(primary[[factor_columns[[id]]]])
  observed_codes <- unique(values)
  assert(setequal(observed_codes, codes), "Observed level set differs for %s", id)
  counts <- table(factor(values, levels = codes))
  level_lock <- rbind(level_lock, data.frame(
    factor = id,
    source_or_derived_column = factor_columns[[id]],
    code = codes,
    label = labels,
    observed_count = as.integer(counts),
    assignment_probability = unname(assignment_probabilities[[id]][codes]),
    target_probability = 1 / length(codes)
  ))
}
write_csv(level_lock, file.path(output_dir, "level_lock.csv"))

target_cells <- data.table::as.data.table(expand.grid(
  democracy_behavior = democracy_codes,
  focal_party = unlist(contract$factors$focal_party$codes, use.names = FALSE),
  contest_relation = unlist(contract$factors$contest_relation$codes, use.names = FALSE),
  focal_gender = unlist(contract$factors$focal_gender$codes, use.names = FALSE),
  KEEP.OUT.ATTRS = FALSE,
  stringsAsFactors = FALSE
))
target_cells[, assignment_probability :=
               assignment_probabilities$democracy_behavior[democracy_behavior] *
               assignment_probabilities$focal_party[focal_party] *
               assignment_probabilities$contest_relation[contest_relation] *
               assignment_probabilities$focal_gender[focal_gender]]
target_cells[, target_probability := 1 / .N]
target_cells[, mu_over_q := target_probability / assignment_probability]
observed_cells <- primary[, .N, by = .(democracy_behavior, focal_party = c_party,
                                        contest_relation, focal_gender)]
target_cells[observed_cells, observed_count := i.N,
             on = .(democracy_behavior, focal_party, contest_relation, focal_gender)]
target_cells[is.na(observed_count), observed_count := 0L]
target_cells[, cell_id := sprintf("D=%s|P=%s|R=%s|S=%s", democracy_behavior, focal_party,
                                  contest_relation, focal_gender)]
data.table::setcolorder(target_cells, c("cell_id", "democracy_behavior", "focal_party",
                                       "contest_relation", "focal_gender", "observed_count",
                                       "assignment_probability", "target_probability", "mu_over_q"))
assert(nrow(target_cells) == 64L && all(target_cells$observed_count > 0),
       "The supported 64-cell primary target is incomplete")
assert(abs(sum(target_cells$assignment_probability) - 1) < 1e-12 &&
         abs(sum(target_cells$target_probability) - 1) < 1e-12,
       "Assignment or target probabilities do not sum to one")
write_csv(target_cells, file.path(output_dir, "target_cells.csv"))
add_check("all_64_target_cells_observed", sum(target_cells$observed_count > 0), 64,
          all(target_cells$observed_count > 0))

assignment_diagnostics <- data.frame()
add_assignment_component <- function(component, values, probabilities) {
  levels <- names(probabilities)
  observed <- table(factor(as.character(values), levels = levels))
  expected_count <- length(values) * probabilities
  assignment_diagnostics <<- rbind(assignment_diagnostics, data.frame(
    component = component,
    level = levels,
    observed_count = as.integer(observed),
    expected_count = unname(expected_count),
    observed_probability = as.integer(observed) / length(values),
    expected_probability = unname(probabilities),
    pearson_residual = (as.integer(observed) - expected_count) / sqrt(expected_count),
    interpretation = "Descriptive compatibility check; source randomization, not realized balance, identifies q."
  ))
}
add_assignment_component("democracy_behavior", primary$democracy_behavior,
                         assignment_probabilities$democracy_behavior)
add_assignment_component("focal_party", primary$c_party, assignment_probabilities$focal_party)
add_assignment_component("contest_relation", primary$contest_relation,
                         assignment_probabilities$contest_relation)
add_assignment_component("focal_gender", primary$focal_gender,
                         assignment_probabilities$focal_gender)
assignment_diagnostics <- rbind(assignment_diagnostics, data.frame(
  component = "joint_64_cell",
  level = target_cells$cell_id,
  observed_count = target_cells$observed_count,
  expected_count = nrow(primary) * target_cells$assignment_probability,
  observed_probability = target_cells$observed_count / nrow(primary),
  expected_probability = target_cells$assignment_probability,
  pearson_residual = (target_cells$observed_count - nrow(primary) * target_cells$assignment_probability) /
    sqrt(nrow(primary) * target_cells$assignment_probability),
  interpretation = "IID-row reference residual only; respondent assignments are blocked without replacement."
))
max_assignment_residual <- max(abs(assignment_diagnostics$pearson_residual))
add_check("maximum_absolute_assignment_reference_residual", format(max_assignment_residual, digits = 8),
          "<4", max_assignment_residual < 4,
          "A compatibility diagnostic only; block documentation supplies the probabilities.")
write_csv(assignment_diagnostics, file.path(output_dir, "assignment_diagnostics.csv"))

primary_cell_key <- sprintf("D=%s|P=%s|R=%s|S=%s", primary$democracy_behavior, primary$c_party,
                            primary$contest_relation, primary$focal_gender)
cell_match <- match(primary_cell_key, target_cells$cell_id)
assert(!anyNA(cell_match), "A primary row does not map to a locked target cell")
primary[, mu_over_q := target_cells$mu_over_q[cell_match]]
survey_total <- sum(respondents$weight)
primary[, analysis_measure_weight := (weight / survey_total) * mu_over_q /
          as.numeric(expected$primary_tasks_per_respondent)]
target_weight_diagnostics <- data.frame(
  respondents = nrow(respondents),
  raw_survey_weight_sum = survey_total,
  normalized_respondent_weight_sum = sum(respondents$target_weight),
  respondent_weight_effective_sample_size = survey_total^2 / sum(respondents$weight^2),
  supported_target_cells = nrow(target_cells),
  minimum_assignment_probability = min(target_cells$assignment_probability),
  maximum_assignment_probability = max(target_cells$assignment_probability),
  target_cell_probability = unique(target_cells$target_probability),
  minimum_mu_over_q = min(target_cells$mu_over_q),
  median_mu_over_q = stats::median(target_cells$mu_over_q),
  maximum_mu_over_q = max(target_cells$mu_over_q),
  empirical_row_mean_mu_over_q = mean(primary$mu_over_q),
  survey_weighted_analysis_measure_sum = sum(primary$analysis_measure_weight)
)
add_check("normalized_respondent_target_weight", sum(respondents$target_weight), 1,
          abs(sum(respondents$target_weight) - 1) < 1e-12)
add_check("finite_bounded_target_ratio", max(target_cells$mu_over_q), "<2",
          all(is.finite(target_cells$mu_over_q)) && max(target_cells$mu_over_q) < 2)
write_csv(target_weight_diagnostics, file.path(output_dir, "target_weight_diagnostics.csv"))

# Audit but do not promote incompletely documented complete-profile fields.
nuisance_columns <- setdiff(
  experiment_allowed,
  c("id", "weight", "matchNum", "candNum", "c_party", "o_party", "c_dem_code",
    "o_dem_code", "c_dem_type", "o_dem_type", "c_sex")
)
nuisance_context_audit <- do.call(rbind, lapply(nuisance_columns, function(nm) {
  x <- primary[[nm]]
  numeric_x <- is.numeric(x) || is.integer(x)
  data.frame(
    column = nm,
    role = if (nm %in% c("c_age", "o_age", "c_exper", "o_exper", "c_pro", "o_pro"))
      "excluded_complete_profile_factor_empirically_standardized" else
        "randomized_current_task_context_empirically_standardized",
    storage_type = class(x)[[1]],
    missing_count = sum(is.na(x)),
    unique_nonmissing = data.table::uniqueN(x, na.rm = TRUE),
    minimum = if (numeric_x) min(x, na.rm = TRUE) else NA_real_,
    maximum = if (numeric_x) max(x, na.rm = TRUE) else NA_real_
  )
}))
age_experience_differences <- sort(unique(c(primary$c_age - primary$c_exper,
                                            primary$o_age - primary$o_exper)))
profession_levels <- sort(unique(c(primary$c_pro, primary$o_pro)))
add_check("candidate_age_range", paste(range(c(primary$c_age, primary$o_age)), collapse = ":"),
          "35:65", identical(as.numeric(range(c(primary$c_age, primary$o_age))), c(35, 65)))
add_check("released_age_minus_experience_values", paste(age_experience_differences, collapse = ","),
          paste(23:33, collapse = ","), identical(as.integer(age_experience_differences), 23:33),
          "Conflicts with the supplement's stated 20:30 subtraction rule; excluded from causal target.")
add_check("profession_levels_include_omitted_police_category", "Police officer" %in% profession_levels,
          TRUE, "Police officer" %in% profession_levels,
          "The supplement probability list omits this released category and sums to 0.90.")
add_check("all_nuisance_context_fields_complete", sum(nuisance_context_audit$missing_count), 0,
          all(nuisance_context_audit$missing_count == 0))
write_csv(nuisance_context_audit, file.path(output_dir, "nuisance_context_audit.csv"))

# Freeze the 28 pre-choice features without deriving any fold-global moments.
primary_features <- unlist(contract$covariates$primary_features, use.names = FALSE)
numeric_features <- unlist(contract$covariates$numeric_features, use.names = FALSE)
categorical_features <- unlist(contract$covariates$categorical_features, use.names = FALSE)
assert(length(primary_features) == 28L && !anyDuplicated(primary_features),
       "The primary X lock must contain 28 unique raw features")
assert(setequal(primary_features, c(numeric_features, categorical_features)) &&
         !length(intersect(numeric_features, categorical_features)),
       "Numeric and categorical feature partitions do not match the primary X lock")
assert(all(primary_features %in% names(linked_covariates)), "A primary feature is absent after the strict read")

covariate_audit <- do.call(rbind, lapply(primary_features, function(nm) {
  x <- linked_covariates[[nm]]
  numeric_x <- nm %in% numeric_features
  data.frame(
    feature = nm,
    semantic_type = if (numeric_x) "numeric_score_or_continuous" else "categorical",
    storage_type = class(x)[[1]],
    respondents = length(x),
    missing_count = sum(is.na(x)),
    missing_proportion = mean(is.na(x)),
    unique_nonmissing = data.table::uniqueN(x, na.rm = TRUE),
    minimum = if (numeric_x) min(as.numeric(x), na.rm = TRUE) else NA_real_,
    maximum = if (numeric_x) max(as.numeric(x), na.rm = TRUE) else NA_real_
  )
}))
assert(all(covariate_audit$respondents == expected$linked_covariate_ids),
       "A linked feature has the wrong respondent count")
write_csv(covariate_audit, file.path(output_dir, "covariate_audit.csv"))

covariate_category_levels <- do.call(rbind, lapply(categorical_features, function(nm) {
  x <- as.character(linked_covariates[[nm]])
  observed_levels <- sort(unique(x[!is.na(x)]))
  levels <- c(observed_levels, "__MISSING__")
  counts <- c(vapply(observed_levels, function(level) sum(x == level, na.rm = TRUE), integer(1)),
              sum(is.na(x)))
  data.frame(
    feature = nm,
    level = levels,
    observed_count = counts,
    level_role = ifelse(levels == "__MISSING__", "explicit_foldwise_missing_level", "locked_outcome_blind_universe")
  )
}))
write_csv(covariate_category_levels, file.path(output_dir, "covariate_category_levels.csv"))

covariate_feature_lock <- data.frame(
  feature = primary_features,
  semantic_type = ifelse(primary_features %in% numeric_features, "numeric", "categorical"),
  missing_count = covariate_audit$missing_count[match(primary_features, covariate_audit$feature)],
  imputation = ifelse(primary_features %in% numeric_features,
                      "outer_training_fold_median_plus_missing_indicator",
                      "outer_training_fold_explicit_missing_level"),
  encoding_or_scaling = ifelse(primary_features %in% numeric_features,
                               "outer_training_fold_center_and_scale",
                               "full_one_hot_over_locked_outcome_blind_level_universe"),
  distributional_preprocessing_source = "outer_training_fold_only"
)
write_csv(covariate_feature_lock, file.path(output_dir, "covariate_feature_lock.csv"))
add_check("primary_prechoice_features", nrow(covariate_feature_lock), 28,
          nrow(covariate_feature_lock) == 28L)

# Deterministic respondent-level five-fold assignment. No row or outcome enters
# the hash, and all 11 current tasks remain with their respondent.
stable_string_hash <- function(x, salt, multiplier = 131) {
  vapply(x, function(token) {
    h <- as.numeric(salt)
    for (code in utf8ToInt(enc2utf8(token))) h <- (h * multiplier + code) %% 2147483629
    h
  }, numeric(1))
}
assign_balanced_folds <- function(ids, k, salt) {
  ord <- order(stable_string_hash(ids, salt), ids)
  out <- integer(length(ids))
  out[ord] <- rep(seq_len(k), length.out = length(ids))
  out
}
outer_k <- as.integer(contract$folds$outer_folds)
fold_seed <- as.integer(contract$folds$deterministic_seed)
respondents[, outer_fold := assign_balanced_folds(as.character(id), outer_k, fold_seed)]
fold_sizes <- tabulate(respondents$outer_fold, nbins = outer_k)
row_fold <- respondents$outer_fold[match(primary$id, respondents$id)]
add_check("all_primary_rows_map_to_a_fold", sum(!is.na(row_fold)), nrow(primary), !anyNA(row_fold))
add_check("respondent_fold_sizes_balanced", paste(fold_sizes, collapse = ","), "difference<=1",
          diff(range(fold_sizes)) <= 1L)
add_check("all_tasks_for_each_respondent_share_one_fold",
          all(vapply(split(row_fold, primary$id), function(x) length(unique(x)) == 1L, logical(1))),
          TRUE, all(vapply(split(row_fold, primary$id), function(x) length(unique(x)) == 1L, logical(1))))
add_check("deterministic_fold_recalculation",
          identical(respondents$outer_fold,
                    assign_balanced_folds(as.character(respondents$id), outer_k, fold_seed)),
          TRUE,
          identical(respondents$outer_fold,
                    assign_balanced_folds(as.character(respondents$id), outer_k, fold_seed)))
respondent_folds <- respondents[, .(id, survey_weight = weight, target_weight, outer_fold)]
write_csv(respondent_folds, file.path(output_dir, "respondent_folds.csv"))

# Construct whole-family function-space projectors under the uniform target.
# Invariance is checked against reference-baseline swaps, random level
# permutations, and deterministic full-rank random contrast rotations.
helmert_function_basis <- function(n_levels) {
  h <- stats::contr.helmert(n_levels)
  sweep(h, 2L, sqrt(colSums(h^2) / n_levels), "/")
}
weighted_crossprod <- function(a, b, p) crossprod(a, sweep(b, 1L, p, "*"))
weighted_projector <- function(basis, p) {
  gram <- weighted_crossprod(basis, basis, p)
  basis %*% solve(gram, sweep(t(basis), 2L, p, "*"))
}
interaction_basis <- function(a_index, b_index, a_basis, b_basis) {
  out <- matrix(NA_real_, nrow = length(a_index), ncol = ncol(a_basis) * ncol(b_basis))
  cc <- 0L
  for (jb in seq_len(ncol(b_basis))) {
    for (ja in seq_len(ncol(a_basis))) {
      cc <- cc + 1L
      out[, cc] <- a_basis[a_index, ja] * b_basis[b_index, jb]
    }
  }
  out
}
reference_residual_basis <- function(a_index, b_index, n_a, n_b, base_a, base_b, p) {
  ca <- stats::contr.treatment(n_a, base = base_a)
  cb <- stats::contr.treatment(n_b, base = base_b)
  a_eval <- ca[a_index, , drop = FALSE]
  b_eval <- cb[b_index, , drop = FALSE]
  raw_interaction <- interaction_basis(seq_along(a_index), seq_along(b_index), a_eval, b_eval)
  main <- cbind(intercept = 1, a_eval, b_eval)
  coefficient <- solve(weighted_crossprod(main, main, p),
                       weighted_crossprod(main, raw_interaction, p))
  raw_interaction - main %*% coefficient
}

family_identification <- data.frame()
contrast_coordinates <- data.frame()
family_projectors <- data.frame()
for (family_index in seq_along(contract$estimands$families)) {
  family <- contract$estimands$families[[family_index]]
  family_id <- family$id
  ids <- unlist(family$attributes, use.names = FALSE)
  a_id <- ids[[1]]
  b_id <- ids[[2]]
  a_codes <- as.character(unlist(contract$factors[[a_id]]$codes, use.names = FALSE))
  b_codes <- as.character(unlist(contract$factors[[b_id]]$codes, use.names = FALSE))
  a_col <- factor_columns[[a_id]]
  b_col <- factor_columns[[b_id]]
  grid <- expand.grid(
    a_code = a_codes, b_code = b_codes,
    KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE
  )
  grid$cell_index <- seq_len(nrow(grid))
  a_index <- match(grid$a_code, a_codes)
  b_index <- match(grid$b_code, b_codes)
  ua <- helmert_function_basis(length(a_codes))
  ub <- helmert_function_basis(length(b_codes))
  basis <- interaction_basis(a_index, b_index, ua, ub)
  coordinate_labels <- sprintf("h%02d_x_h%02d",
                               rep(seq_len(ncol(ua)), times = ncol(ub)),
                               rep(seq_len(ncol(ub)), each = ncol(ua)))
  p <- rep(1 / nrow(grid), nrow(grid))
  main <- cbind(intercept = 1, ua[a_index, , drop = FALSE], ub[b_index, , drop = FALSE])
  gram_error <- max(abs(weighted_crossprod(basis, basis, p) - diag(ncol(basis))))
  main_error <- max(abs(weighted_crossprod(main, basis, p)))
  projector <- weighted_projector(basis, p)
  idempotence_error <- max(abs(projector %*% projector - projector))

  # Reference-coded interaction columns are residualized from their associated
  # intercept/main-effect span. Swapping both baselines must recover the same
  # target interaction projector.
  ref_first <- reference_residual_basis(a_index, b_index, length(a_codes), length(b_codes),
                                        1L, 1L, p)
  ref_last <- reference_residual_basis(a_index, b_index, length(a_codes), length(b_codes),
                                       length(a_codes), length(b_codes), p)
  baseline_first_error <- max(abs(weighted_projector(ref_first, p) - projector))
  baseline_last_error <- max(abs(weighted_projector(ref_last, p) - projector))

  set.seed(fold_seed + 1000L * family_index)
  perm_a <- sample(a_codes)
  perm_b <- sample(b_codes)
  ua_perm <- helmert_function_basis(length(a_codes))
  ub_perm <- helmert_function_basis(length(b_codes))
  perm_basis <- interaction_basis(match(grid$a_code, perm_a), match(grid$b_code, perm_b),
                                  ua_perm, ub_perm)
  permutation_error <- max(abs(weighted_projector(perm_basis, p) - projector))

  random_matrix <- matrix(stats::rnorm(ncol(basis)^2), ncol(basis), ncol(basis))
  rotation <- qr.Q(qr(random_matrix))
  rotated_basis <- basis %*% rotation
  rotation_error <- max(abs(weighted_projector(rotated_basis, p) - projector))

  observed <- table(
    factor(as.character(primary[[a_col]]), levels = a_codes),
    factor(as.character(primary[[b_col]]), levels = b_codes)
  )
  observed_count <- as.integer(observed)
  observed_rank <- qr(basis[observed_count > 0, , drop = FALSE], tol = 1e-10)$rank
  expected_rank <- as.integer(family$interaction_rank)
  task_minimum <- Inf
  for (task in sort(unique(primary$matchNum))) {
    index <- primary$matchNum == task
    task_table <- table(
      factor(as.character(primary[[a_col]][index]), levels = a_codes),
      factor(as.character(primary[[b_col]][index]), levels = b_codes)
    )
    task_minimum <- min(task_minimum, task_table)
  }
  assert(nrow(grid) == as.integer(family$cells), "Cell lock failed for %s", family_id)
  assert(all(observed_count > 0), "A registered family cell is empty for %s", family_id)
  assert(min(observed_count) == as.integer(family$expected_minimum_cell_count),
         "Minimum cell-count lock failed for %s", family_id)
  assert(observed_rank == expected_rank, "Interaction rank lock failed for %s", family_id)
  invariant_errors <- c(gram_error, main_error, idempotence_error, baseline_first_error,
                        baseline_last_error, permutation_error, rotation_error)
  assert(all(invariant_errors < 1e-10), "A coding-invariance check failed for %s", family_id)

  family_identification <- rbind(family_identification, data.frame(
    family = family_id,
    factor_a = a_id,
    factor_b = b_id,
    cells = nrow(grid),
    observed_cells = sum(observed_count > 0),
    minimum_cell_count = min(observed_count),
    minimum_released_scenario_cell_count_diagnostic = as.integer(task_minimum),
    expected_interaction_rank = expected_rank,
    observed_interaction_rank = observed_rank,
    orthonormality_error = gram_error,
    main_effect_orthogonality_error = main_error,
    projector_idempotence_error = idempotence_error,
    first_baseline_projector_error = baseline_first_error,
    last_baseline_projector_error = baseline_last_error,
    random_level_permutation_projector_error = permutation_error,
    random_contrast_rotation_projector_error = rotation_error,
    identified_and_invariant = observed_rank == expected_rank && all(invariant_errors < 1e-10)
  ))
  for (j in seq_len(ncol(basis))) {
    contrast_coordinates <- rbind(contrast_coordinates, data.frame(
      family = family_id,
      coordinate = coordinate_labels[[j]],
      cell_index = grid$cell_index,
      factor_a_code = grid$a_code,
      factor_b_code = grid$b_code,
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
add_check("registered_family_ranks", sum(family_identification$identified_and_invariant),
          nrow(family_identification), all(family_identification$identified_and_invariant))
add_check("registered_family_total_dimension", sum(family_identification$observed_interaction_rank),
          contract$estimands$total_dimension,
          sum(family_identification$observed_interaction_rank) ==
            as.integer(contract$estimands$total_dimension))
add_check("registered_family_pooled_support",
          min(family_identification$observed_cells), 16,
          all(family_identification$observed_cells == 16L),
          "Scenario-specific zeros are permitted by the blocked design; matchNum is empirically standardized, not a separately targeted surface.")
write_csv(family_identification, file.path(output_dir, "family_identification.csv"))
write_csv(contrast_coordinates, file.path(output_dir, "contrast_coordinates.csv"))
write_csv(family_projectors, file.path(output_dir, "family_projectors.csv"))

write_csv(validation, file.path(output_dir, "design_validation.csv"))
assert(all(validation$pass),
       "At least one Graham-Svolik design-contract validation failed; inspect design_validation.csv")

generated_paths <- file.path(output_dir, expected_csv_files)
assert(all(file.exists(generated_paths)), "At least one expected audit output was not generated")
generated_md5 <- unname(tools::md5sum(generated_paths))
names(generated_md5) <- expected_csv_files
manifest <- list(
  schema_version = 1,
  study_id = contract$study_id,
  contract_version = contract$contract_version,
  contract_md5 = unname(tools::md5sum(contract_path)),
  completed_utc = format(Sys.time(), tz = "UTC", usetz = TRUE),
  status = "passed",
  gate = "B",
  gate_result = "green_for_narrow_coarsened_democracy_families",
  complete_profile_result = "failed_assignment_law_not_certified",
  portfolio_eligible = TRUE,
  outcome_blind_audit_run = TRUE,
  new_outcome_values_loaded_by_audit = FALSE,
  outcome_models_fit = FALSE,
  pre_audit_protocol_deviation = list(
    occurred = TRUE,
    description = contract$protocol$pre_audit_deviation$description,
    quarantine = contract$protocol$pre_audit_deviation$quarantine
  ),
  read_method = list(
    schema = "data.table::fread(nrows=0)",
    values = "data.table::fread(select=locked_allowlist)",
    experiment_loaded_columns = length(experiment_allowed),
    covariate_loaded_columns = length(covariate_allowed),
    forbidden_response_values_loaded = FALSE
  ),
  sample = list(
    respondents = data.table::uniqueN(primary$id),
    retained_tasks = nrow(primary),
    tasks_per_respondent = as.integer(expected$primary_tasks_per_respondent),
    excluded_valence_tasks = nrow(cand1_all) - nrow(primary)
  ),
  assignment = list(
    target_cells = nrow(target_cells),
    minimum_cell_count = min(target_cells$observed_count),
    maximum_mu_over_q = max(target_cells$mu_over_q),
    mean_no_carryover_required = TRUE,
    actual_display_sequence_released = FALSE
  ),
  registered_family_dimensions = as.list(setNames(
    family_identification$observed_interaction_rank,
    family_identification$family
  )),
  folds = list(
    cluster = "id",
    outer_folds = outer_k,
    deterministic_seed = fold_seed,
    sizes = as.list(fold_sizes),
    row_leakage = FALSE
  ),
  generated_files_md5 = as.list(generated_md5),
  next_step = contract$gate$next_step,
  note = "The executable audit loaded no choice/win value. The separately disclosed one-row pre-audit display remains quarantined and was not used."
)
yaml::write_yaml(manifest, manifest_path)

cat(sprintf(
  paste0(
    "Graham-Svolik narrow Gate-B audit passed: %s respondents, %s nonvalence tasks, ",
    "64/64 target cells, and three invariant rank-7 families. No response value was loaded ",
    "by the audit; the disclosed one-row pre-audit display remains quarantined.\n"
  ),
  format(data.table::uniqueN(primary$id), big.mark = ","),
  format(nrow(primary), big.mark = ",")
))
