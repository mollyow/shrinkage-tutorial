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
    if (file.exists(file.path(candidate, "config", "datasets", "climate-agreements.yml"))) return(candidate)
  }
  abort("Could not locate the repository root containing config/datasets/climate-agreements.yml")
}

write_csv <- function(x, path) utils::write.csv(x, path, row.names = FALSE, na = "")
resolve_path <- function(path, root) if (grepl("^/", path)) path else file.path(root, path)
respondent_invariant <- function(values, ids) {
  all(vapply(split(values, ids), function(x) {
    token <- ifelse(is.na(x), "<NA>", as.character(x))
    length(unique(token)) == 1L
  }, logical(1)))
}

repo_root <- find_repo_root()
contract_arg <- arg_value("--contract", "config/datasets/climate-agreements.yml")
output_arg <- arg_value("--output-dir", "analysis/climate/derived")
contract_path <- resolve_path(contract_arg, repo_root)
output_dir <- resolve_path(output_arg, repo_root)

assert(requireNamespace("haven", quietly = TRUE), "Package 'haven' is required")
assert(requireNamespace("tidyselect", quietly = TRUE), "Package 'tidyselect' is required")
assert(requireNamespace("yaml", quietly = TRUE), "Package 'yaml' is required")

contract <- yaml::read_yaml(contract_path)
assert(identical(contract$status, "conditional_design_lock_outcomes_prohibited"), "Unexpected contract status")
assert(isTRUE(contract$decision$outcome_blind), "Contract must require an outcome-blind audit")

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
expected_csv_files <- c(
  "all_pair_support_summary.csv",
  "archive_checksums.csv",
  "assignment_diagnostics.csv",
  "contrast_coordinates.csv",
  "country_counts.csv",
  "covariate_audit.csv",
  "covariate_category_levels.csv",
  "covariate_feature_lock.csv",
  "design_validation.csv",
  "family_identification.csv",
  "family_projectors.csv",
  "level_lock.csv",
  "loaded_columns.csv",
  "opponent_law_diagnostics.csv",
  "profile_uniformity_diagnostics.csv",
  "profile_probability_strata.csv",
  "respondent_folds.csv",
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

# Check every raw and documentary source before any design column is read.
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

allowed_columns <- unlist(contract$read_barrier$allowed_columns, use.names = FALSE)
forbidden_outcomes <- unlist(contract$read_barrier$forbidden_outcomes, use.names = FALSE)
canonical_forbidden <- c("choice_cj", "rating_cj", "incon")
assert(setequal(forbidden_outcomes, canonical_forbidden),
       "The contract must hard-forbid choice_cj, rating_cj, and incon")
assert(!any(allowed_columns %in% forbidden_outcomes), "An outcome or outcome-derived field appears in the allowlist")
assert(!anyDuplicated(allowed_columns), "The design-only allowlist contains duplicate columns")

# This selected-column call is the enforceable outcome read barrier. In
# particular, choice_cj, rating_cj, and incon never enter memory.
data_path <- resolve_path(file_specs$data$path, repo_root)
raw <- haven::read_dta(data_path, col_select = tidyselect::all_of(allowed_columns))
assert(setequal(names(raw), allowed_columns), "Selected columns differ from the contract allowlist")
assert(!any(names(raw) %in% forbidden_outcomes), "A forbidden response field was loaded")
raw$.source_row <- seq_len(nrow(raw))

attribute_ids <- names(contract$attributes)
attribute_columns <- vapply(contract$attributes, `[[`, character(1), "column")
source_value_labels <- lapply(raw[c("country", attribute_columns)], attr, which = "labels")
design_columns <- c("ID", "weight", "conjoint", "cj_order", "country", attribute_columns)
primary_source_columns <- c(
  "age", "female", "income", "support_iec_high_group",
  unlist(contract$covariates$education_source_by_country, use.names = FALSE)
)
role_for_column <- function(nm) {
  if (nm == "ID") return("cluster_id")
  if (nm == "weight") return("survey_weight")
  if (nm == "conjoint") return("task_id")
  if (nm == "cj_order") return("randomized_display_context")
  if (nm == "country") return("target_stratum_and_primary_feature")
  if (nm %in% attribute_columns) return("randomized_attribute")
  if (nm %in% primary_source_columns) return("primary_feature_source")
  "audited_but_excluded_feature"
}
loaded_columns <- data.frame(
  column = allowed_columns,
  role = vapply(allowed_columns, role_for_column, character(1)),
  outcome_loaded = FALSE
)
write_csv(loaded_columns, file.path(output_dir, "loaded_columns.csv"))

for (nm in c("ID", "weight", "conjoint", "cj_order", "country", attribute_columns)) {
  raw[[nm]] <- as.numeric(raw[[nm]])
}

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
id_chr <- as.character(raw$ID)
pair_key <- paste(id_chr, raw$conjoint, sep = "::")
id_rows <- table(id_chr)
id_tasks <- tapply(raw$conjoint, id_chr, function(x) length(unique(x)))
pair_rows <- table(pair_key)
pair_source_rows <- split(raw$.source_row, pair_key)
pair_rows_contiguous <- vapply(
  pair_source_rows,
  function(x) length(x) == 2L && diff(sort(x)) == 1L,
  logical(1)
)
raw$profile_index <- ave(raw$.source_row, pair_key, FUN = function(z) rank(z, ties.method = "first"))

add_check("profile_rows", nrow(raw), expected$profile_rows, nrow(raw) == expected$profile_rows)
add_check("respondents", length(unique(raw$ID)), expected$respondents,
          length(unique(raw$ID)) == expected$respondents)
add_check("tasks", length(pair_rows), expected$tasks, length(pair_rows) == expected$tasks)
add_check("rows_per_respondent", paste(range(id_rows), collapse = ":"),
          expected$tasks_per_respondent * expected$profiles_per_task,
          all(id_rows == expected$tasks_per_respondent * expected$profiles_per_task))
add_check("tasks_per_respondent", paste(range(id_tasks), collapse = ":"),
          expected$tasks_per_respondent, all(id_tasks == expected$tasks_per_respondent))
add_check("task_levels", paste(sort(unique(raw$conjoint)), collapse = ","), "1,2,3,4",
          identical(as.integer(sort(unique(raw$conjoint))), 1:4))
add_check("profiles_per_task", paste(range(pair_rows), collapse = ":"),
          expected$profiles_per_task, all(pair_rows == expected$profiles_per_task))
add_check("pair_rows_adjacent_in_source", sum(pair_rows_contiguous), length(pair_rows_contiguous),
          all(pair_rows_contiguous), "The source has no semantic side field; source order defines a neutral index.")
add_check("derived_profile_index", paste(sort(unique(raw$profile_index)), collapse = ","), "1,2",
          identical(sort(unique(as.integer(raw$profile_index))), 1:2))
add_check("complete_design_columns", sum(is.na(raw[design_columns])), 0,
          !anyNA(raw[design_columns]))

weight_invariant <- respondent_invariant(raw$weight, raw$ID)
country_invariant <- respondent_invariant(raw$country, raw$ID)
order_invariant <- respondent_invariant(raw$cj_order, raw$ID)
add_check("positive_finite_weights", sum(is.finite(raw$weight) & raw$weight > 0), nrow(raw),
          all(is.finite(raw$weight) & raw$weight > 0))
add_check("respondent_invariant_weights", weight_invariant, TRUE, weight_invariant)
add_check("respondent_invariant_country", country_invariant, TRUE, country_invariant)
add_check("respondent_invariant_display_order", order_invariant, TRUE, order_invariant)
add_check("display_order_codes", length(unique(raw$cj_order)), 696,
          length(unique(raw$cj_order)) == 696,
          "Codes are opaque because the archive does not map them to the 720 possible permutations.")

country_source_labels <- source_value_labels$country
country_label_by_code <- setNames(names(country_source_labels), as.character(as.numeric(country_source_labels)))
country_names <- unname(country_label_by_code[as.character(1:4)])
assert(!anyNA(country_names), "Country labels are not fully available")

first_respondent <- !duplicated(raw$ID)
respondents <- as.data.frame(raw[first_respondent, , drop = FALSE])
country_counts <- do.call(rbind, lapply(seq_along(country_names), function(i) {
  profile_idx <- raw$country == i
  respondent_idx <- respondents$country == i
  data.frame(
    country_code = i,
    country = country_names[[i]],
    respondents = sum(respondent_idx),
    profile_rows = sum(profile_idx),
    raw_weight_sum = sum(respondents$weight[respondent_idx]),
    weight_minimum = min(respondents$weight[respondent_idx]),
    weight_maximum = max(respondents$weight[respondent_idx])
  )
}))
locked_country_counts <- as.integer(unlist(expected$country_respondents, use.names = FALSE))
assert(identical(country_counts$respondents, locked_country_counts), "Country respondent counts differ from the lock")
add_check("country_counts", paste(country_counts$respondents, collapse = ","),
          paste(locked_country_counts, collapse = ","),
          identical(country_counts$respondents, locked_country_counts))
add_check("country_weight_sums", paste(round(country_counts$raw_weight_sum, 6), collapse = ","),
          paste(locked_country_counts, collapse = ","),
          max(abs(country_counts$raw_weight_sum - locked_country_counts)) < 1e-6)
write_csv(country_counts, file.path(output_dir, "country_counts.csv"))

sample_summary <- data.frame(
  respondents = length(unique(raw$ID)),
  tasks = length(pair_rows),
  profile_rows = nrow(raw),
  tasks_per_respondent_min = min(id_tasks),
  tasks_per_respondent_max = max(id_tasks),
  profiles_per_task_min = min(pair_rows),
  profiles_per_task_max = max(pair_rows),
  countries = length(unique(raw$country)),
  all_design_columns_complete = !anyNA(raw[design_columns])
)
write_csv(sample_summary, file.path(output_dir, "sample_summary.csv"))

# Attribute level and profile-support locks use numeric design codes. Display
# labels for cost and sanctions varied by country, so the policy meaning is
# taken from the article/SI rather than requiring equality to the DTA labels.
level_lock <- data.frame()
for (id in attribute_ids) {
  spec <- contract$attributes[[id]]
  codes <- as.numeric(unlist(spec$codes, use.names = FALSE))
  labels <- as.character(unlist(spec$labels, use.names = FALSE))
  observed_codes <- sort(unique(raw[[spec$column]]))
  assert(identical(observed_codes, codes), "Level-code lock failed for %s", id)
  source_labels <- source_value_labels[[spec$column]]
  source_by_code <- setNames(names(source_labels), as.character(as.numeric(source_labels)))
  source_text <- unname(source_by_code[as.character(codes)])
  counts <- table(factor(raw[[spec$column]], levels = codes))
  level_lock <- rbind(level_lock, data.frame(
    attribute = id,
    column = spec$column,
    code = codes,
    contract_label = labels,
    source_label = source_text,
    observed_count = as.integer(counts)
  ))
}
write_csv(level_lock, file.path(output_dir, "level_lock.csv"))

profile_key <- do.call(paste, c(lapply(raw[attribute_columns], as.integer), sep = "|"))
profile_counts <- table(profile_key)
add_check("complete_profile_support", length(profile_counts), contract$support$profiles,
          length(profile_counts) == as.integer(contract$support$profiles))
add_check("minimum_complete_profile_count", min(profile_counts), ">0", min(profile_counts) > 0)
uniform_profile_expected <- nrow(raw) / as.integer(contract$support$profiles)
uniform_profile_chisq <- sum((as.numeric(profile_counts) - uniform_profile_expected)^2 /
                               uniform_profile_expected)
uniform_profile_df <- as.integer(contract$support$profiles) - 1L
uniform_profile_p <- stats::pchisq(uniform_profile_chisq, df = uniform_profile_df,
                                  lower.tail = FALSE)
profile_uniformity_diagnostics <- data.frame(
  profile_rows = nrow(raw),
  profile_support_size = length(profile_counts),
  expected_count_per_profile = uniform_profile_expected,
  minimum_profile_count = min(profile_counts),
  median_profile_count = stats::median(as.numeric(profile_counts)),
  maximum_profile_count = max(profile_counts),
  pearson_chisq = uniform_profile_chisq,
  degrees_of_freedom = uniform_profile_df,
  iid_reference_p_value = uniform_profile_p,
  interpretation = "IID-row reference only; compatible diagnostic, not proof of the declared uniform-product reconstruction."
)
add_check("pooled_complete_profile_uniformity_iid_reference_p_value",
          format(uniform_profile_p, digits = 8), ">0.01", uniform_profile_p > 0.01,
          "A descriptive compatibility diagnostic, not certification of the generator.")
write_csv(profile_uniformity_diagnostics, file.path(output_dir, "profile_uniformity_diagnostics.csv"))

all_pair_support <- data.frame()
pair_list <- utils::combn(attribute_ids, 2, simplify = FALSE)
support_strata <- c("pooled", country_names)
for (ids in pair_list) {
  a <- contract$attributes[[ids[[1]]]]
  b <- contract$attributes[[ids[[2]]]]
  expected_cells <- length(a$codes) * length(b$codes)
  for (stratum in support_strata) {
    idx <- if (stratum == "pooled") rep(TRUE, nrow(raw)) else country_names[raw$country] == stratum
    observed_cells <- nrow(unique(raw[idx, c(a$column, b$column), drop = FALSE]))
    all_pair_support <- rbind(all_pair_support, data.frame(
      stratum = stratum,
      attribute_a = ids[[1]],
      attribute_b = ids[[2]],
      expected_cartesian_cells = expected_cells,
      observed_cells = observed_cells,
      complete_cartesian = observed_cells == expected_cells
    ))
  }
}
assert(all(all_pair_support$complete_cartesian), "At least one two-factor support is incomplete")
add_check("all_pair_support_strata", sum(all_pair_support$complete_cartesian), nrow(all_pair_support),
          all(all_pair_support$complete_cartesian))
write_csv(all_pair_support, file.path(output_dir, "all_pair_support_summary.csv"))

# Coarse assignment diagnostics cannot prove randomization, but can reveal a
# material conflict with the prospectively reconstructed equal marginal law.
assignment_diagnostics <- data.frame()
assignment_strata <- c(
  list(pooled = rep(TRUE, nrow(raw))),
  setNames(lapply(seq_along(country_names), function(i) raw$country == i),
           paste0("country::", country_names)),
  setNames(lapply(1:2, function(i) raw$profile_index == i), paste0("profile_index::", 1:2)),
  setNames(lapply(1:4, function(i) raw$conjoint == i), paste0("task::", 1:4))
)
for (id in attribute_ids) {
  spec <- contract$attributes[[id]]
  codes <- as.numeric(unlist(spec$codes, use.names = FALSE))
  for (stratum in names(assignment_strata)) {
    idx <- assignment_strata[[stratum]]
    counts <- table(factor(raw[[spec$column]][idx], levels = codes))
    n <- sum(counts)
    expected_count <- n / length(codes)
    assignment_diagnostics <- rbind(assignment_diagnostics, data.frame(
      component = id,
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
          "Prespecified diagnostic only; documentation, not realized balance, identifies the row law.")
write_csv(assignment_diagnostics, file.path(output_dir, "assignment_diagnostics.csv"))

# Diagnose, rather than silently overwrite, the source/realization conflict in
# the joint paired-profile law.
pair_indices <- split(seq_len(nrow(raw)), pair_key)
identical_complete_pair <- vapply(pair_indices, function(ix) profile_key[ix[[1]]] == profile_key[ix[[2]]], logical(1))
number_pairs <- length(pair_indices)
number_profiles <- as.integer(contract$support$profiles)
iid_expected_identical <- number_pairs / number_profiles
iid_probability_zero <- (1 - 1 / number_profiles)^number_pairs
respondent_cluster_upper_probability_zero <- (1 - 1 / number_profiles)^nrow(respondents)

pair_attribute_matches <- do.call(rbind, lapply(attribute_ids, function(id) {
  spec <- contract$attributes[[id]]
  matches <- vapply(pair_indices, function(ix) raw[[spec$column]][ix[[1]]] == raw[[spec$column]][ix[[2]]], logical(1))
  levels_n <- length(spec$codes)
  expected_matches <- number_pairs / levels_n
  data.frame(
    diagnostic = "attribute_level_match",
    component = id,
    observed = sum(matches),
    expected_under_iid = expected_matches,
    pearson_residual = (sum(matches) - expected_matches) / sqrt(expected_matches * (1 - 1 / levels_n)),
    probability_or_candidate_value = NA_real_,
    status = "compatible_with_independent_level_draws"
  )
}))
repeated_profile_rows_within_respondent <- sum(vapply(split(profile_key, raw$ID), function(x) length(x) - length(unique(x)), integer(1)))
opponent_law_diagnostics <- rbind(
  data.frame(
    diagnostic = "identical_complete_profile_pairs",
    component = "complete_profile",
    observed = sum(identical_complete_pair),
    expected_under_iid = iid_expected_identical,
    pearson_residual = (sum(identical_complete_pair) - iid_expected_identical) / sqrt(iid_expected_identical * (1 - 1 / number_profiles)),
    probability_or_candidate_value = iid_probability_zero,
    status = "global_iid_task_pair_model_strongly_disfavored_exact_pair_law_uncertified"
  ),
  pair_attribute_matches,
  data.frame(
    diagnostic = "upper_probability_zero_with_arbitrary_four_task_dependence",
    component = "complete_profile",
    observed = sum(identical_complete_pair),
    expected_under_iid = iid_expected_identical,
    pearson_residual = NA_real_,
    probability_or_candidate_value = respondent_cluster_upper_probability_zero,
    status = "q_times_q_per_task_not_logically_rejected_by_zero_count_alone"
  ),
  data.frame(
    diagnostic = "repeated_profile_rows_across_tasks_within_respondent",
    component = "complete_profile",
    observed = repeated_profile_rows_within_respondent,
    expected_under_iid = NA_real_,
    pearson_residual = NA_real_,
    probability_or_candidate_value = NA_real_,
    status = "rules_out_blanket_respondent_no_repeat_exact_law_still_unknown"
  ),
  data.frame(
    diagnostic = "candidate_uniform_ordered_distinct_probability",
    component = "ordered_pair",
    observed = NA_real_,
    expected_under_iid = NA_real_,
    pearson_residual = NA_real_,
    probability_or_candidate_value = 1 / (number_profiles * (number_profiles - 1)),
    status = "sensitivity_assumption_not_source_certified"
  )
)
add_check("identical_complete_profile_pairs", sum(identical_complete_pair), 0,
          sum(identical_complete_pair) == 0,
          sprintf("Under q x q the expectation is %.6f and P(zero)=%.8g; q x q is not encoded as known.",
                  iid_expected_identical, iid_probability_zero))
add_check("global_iid_task_pair_model_strongly_disfavored", iid_probability_zero, "<0.00001",
          iid_probability_zero < 1e-5,
          sprintf("With arbitrary dependence among four tasks, a conservative respondent-cluster upper probability is %.6f; the exact pair law remains uncertified.",
                  respondent_cluster_upper_probability_zero))
add_check("profile_repeats_across_different_tasks_exist", repeated_profile_rows_within_respondent,
          ">0", repeated_profile_rows_within_respondent > 0,
          "The apparent restriction is not a blanket respondent-level no-repeat rule.")
write_csv(opponent_law_diagnostics, file.path(output_dir, "opponent_law_diagnostics.csv"))

profile_probability_strata <- data.frame(
  law = c("row_marginal_uniform_product", "candidate_uniform_ordered_distinct_pair"),
  certification = c("declared_reconstruction_source_and_frequency_supported_not_generator_certified", "sensitivity_only_not_source_certified"),
  profile_support_size = c(number_profiles, number_profiles),
  ordered_pair_support_size = c(NA_real_, number_profiles * (number_profiles - 1)),
  q_profile = c(1 / number_profiles, 1 / number_profiles),
  ordered_pair_probability = c(NA_real_, 1 / (number_profiles * (number_profiles - 1))),
  identical_pair_probability = c(NA_real_, 0)
)
write_csv(profile_probability_strata, file.path(output_dir, "profile_probability_strata.csv"))

# Lock primary respondent features without consulting either response. The
# country-specific education variables are used only in their named country.
respondent_source_columns <- setdiff(allowed_columns, c("conjoint", attribute_columns))
source_invariance <- vapply(respondent_source_columns, function(nm) respondent_invariant(raw[[nm]], raw$ID), logical(1))
assert(all(source_invariance), "At least one respondent-level source variable varies within respondent")

respondents$country_name <- country_names[respondents$country]
education_source <- c("1" = "educgr_fr", "2" = "educgr_ge", "3" = "educgr_uk", "4" = "educgr_us")
respondents$education_category <- vapply(seq_len(nrow(respondents)), function(i) {
  nm <- education_source[[as.character(respondents$country[[i]])]]
  value <- respondents[[nm]][[i]]
  if (is.na(value)) return(NA_character_)
  paste0(respondents$country[[i]], "::", as.numeric(value))
}, character(1))
respondents$income_category <- ifelse(
  is.na(respondents$income), NA_character_,
  paste0(respondents$country, "::", as.numeric(respondents$income))
)
respondents$support_iec_high_group <- as.numeric(respondents$support_iec_high_group)
respondents$female <- as.numeric(respondents$female)
respondents$age <- as.numeric(respondents$age)

feature_names <- unlist(contract$covariates$primary_features, use.names = FALSE)
feature_semantics <- contract$covariates$feature_semantics
assert(setequal(names(feature_semantics), feature_names), "Feature semantics do not exactly cover primary features")
covariate_feature_lock <- do.call(rbind, lapply(feature_names, function(nm) {
  spec <- feature_semantics[[nm]]
  data.frame(
    variable = nm,
    semantic_type = spec$semantic_type,
    primary_encoding = spec$primary_encoding,
    missing_value_rule = spec$missing_value_rule,
    scaling_rule = spec$scaling_rule
  )
}))
write_csv(covariate_feature_lock, file.path(output_dir, "covariate_feature_lock.csv"))

categorical_primary <- c("country", "female", "education_category", "income_category", "support_iec_high_group")
covariate_category_levels <- do.call(rbind, lapply(categorical_primary, function(nm) {
  x <- if (nm == "country") respondents$country_name else respondents[[nm]]
  observed_levels <- sort(unique(as.character(x[!is.na(x)])))
  counts <- table(factor(as.character(x), levels = observed_levels))
  rbind(
    data.frame(
      variable = nm,
      level_order = seq_along(observed_levels),
      source_code = observed_levels,
      enrolled_count = as.integer(counts),
      is_missing_level = FALSE
    ),
    data.frame(
      variable = nm,
      level_order = length(observed_levels) + 1L,
      source_code = "__MISSING__",
      enrolled_count = sum(is.na(x)),
      is_missing_level = TRUE
    )
  )
}))
write_csv(covariate_category_levels, file.path(output_dir, "covariate_category_levels.csv"))

audit_names <- unique(c(
  setdiff(allowed_columns, c("ID", "weight", "conjoint", "cj_order", "country", attribute_columns)),
  "country", "education_category", "income_category"
))
covariate_audit <- do.call(rbind, lapply(audit_names, function(nm) {
  x <- if (nm == "country") respondents$country_name else respondents[[nm]]
  classification <- if (nm %in% feature_names) {
    "primary_feature"
  } else if (nm %in% unlist(contract$covariates$education_source_by_country, use.names = FALSE)) {
    "primary_derivation_source_country_specific"
  } else {
    "audited_excluded"
  }
  numeric_x <- suppressWarnings(as.numeric(x))
  data.frame(
    variable = nm,
    classification = classification,
    respondents = nrow(respondents),
    missing_n = sum(is.na(x)),
    missing_fraction = mean(is.na(x)),
    unique_nonmissing = length(unique(x[!is.na(x)])),
    minimum = if (all(is.na(numeric_x))) NA_real_ else min(numeric_x, na.rm = TRUE),
    maximum = if (all(is.na(numeric_x))) NA_real_ else max(numeric_x, na.rm = TRUE),
    respondent_invariant = TRUE
  )
}))

locked_missing <- c(age = 0L, female = 0L, education_category = 1L,
                    income_category = 301L, support_iec_high_group = 1L)
for (nm in names(locked_missing)) {
  observed_missing <- covariate_audit$missing_n[covariate_audit$variable == nm]
  add_check(paste0("covariate_missing_", nm), observed_missing, locked_missing[[nm]],
            length(observed_missing) == 1L && observed_missing == locked_missing[[nm]])
}
add_check("primary_feature_count", length(feature_names), 6, length(feature_names) == 6)
add_check("age_range", paste(range(respondents$age), collapse = ":"), "18:99",
          identical(as.numeric(range(respondents$age)), c(18, 99)))
write_csv(covariate_audit, file.path(output_dir, "covariate_audit.csv"))

# Equal-country target: normalize released weights within country, then give
# each country one quarter of the respondent mass.
respondents$target_weight <- ave(
  respondents$weight, respondents$country,
  FUN = function(x) x / sum(x) / length(country_names)
)
target_weight_diagnostics <- do.call(rbind, lapply(seq_along(country_names), function(i) {
  idx <- respondents$country == i
  w <- respondents$weight[idx]
  tw <- respondents$target_weight[idx]
  data.frame(
    country_code = i,
    country = country_names[[i]],
    respondents = sum(idx),
    raw_weight_sum = sum(w),
    target_weight_sum = sum(tw),
    effective_sample_size = sum(w)^2 / sum(w^2),
    minimum_raw_weight = min(w),
    maximum_raw_weight = max(w)
  )
}))
assert(max(abs(target_weight_diagnostics$target_weight_sum - 0.25)) < 1e-12,
       "Equal-country target weights do not sum to one quarter within country")
add_check("target_weight_total", sum(respondents$target_weight), 1,
          abs(sum(respondents$target_weight) - 1) < 1e-12)
write_csv(target_weight_diagnostics, file.path(output_dir, "target_weight_diagnostics.csv"))

# Freeze respondent-level folds using identifiers and countries only.
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
outer_k <- as.integer(contract$folds$outer_folds)
inner_k <- as.integer(contract$folds$inner_folds)
fold_seed <- as.integer(contract$folds$deterministic_seed)
respondents$outer_fold <- ave(
  seq_len(nrow(respondents)), respondents$country,
  FUN = function(ix) assign_balanced_folds(as.character(respondents$ID[ix]), outer_k, fold_seed)
)
respondents$inner_fold_label <- ave(
  seq_len(nrow(respondents)), respondents$country,
  FUN = function(ix) assign_balanced_folds(as.character(respondents$ID[ix]), inner_k,
                                           fold_seed + 104729L, 257, TRUE)
)
respondents$outer_fold <- as.integer(respondents$outer_fold)
respondents$inner_fold_label <- as.integer(respondents$inner_fold_label)
outer_balance <- vapply(split(respondents$outer_fold, respondents$country),
                        function(x) diff(range(tabulate(x, nbins = outer_k))) <= 1L, logical(1))
inner_balance <- vapply(split(respondents$inner_fold_label, respondents$country),
                        function(x) diff(range(tabulate(x, nbins = inner_k))) <= 1L, logical(1))
nested_coverage <- unlist(lapply(seq_len(outer_k), function(held_out) {
  vapply(split(respondents, respondents$country), function(x) {
    train <- x[x$outer_fold != held_out, , drop = FALSE]
    length(unique(train$inner_fold_label)) == inner_k
  }, logical(1))
}))
row_fold_match <- match(as.character(raw$ID), as.character(respondents$ID))
add_check("all_rows_map_to_one_respondent_fold", sum(!is.na(row_fold_match)), nrow(raw), !anyNA(row_fold_match))
add_check("outer_fold_country_balance", sum(outer_balance), length(outer_balance), all(outer_balance))
add_check("inner_fold_country_balance", sum(inner_balance), length(inner_balance), all(inner_balance))
add_check("inner_fold_coverage_in_every_outer_training_set", sum(nested_coverage),
          length(nested_coverage), all(nested_coverage))
respondent_folds <- respondents[, c(
  "ID", "country", "country_name", "target_weight", "outer_fold", "inner_fold_label"
)]
write_csv(respondent_folds, file.path(output_dir, "respondent_folds.csv"))

# Uniform-target, coding-invariant whole-family bases and projectors.
helmert_function_basis <- function(n_levels) {
  h <- stats::contr.helmert(n_levels)
  sweep(h, 2L, sqrt(colSums(h^2) / n_levels), "/")
}

family_identification <- data.frame()
contrast_coordinates <- data.frame()
family_projectors <- data.frame()
for (family in contract$estimands$families) {
  family_id <- family$id
  ids <- unlist(family$attributes, use.names = FALSE)
  a <- contract$attributes[[ids[[1]]]]
  b <- contract$attributes[[ids[[2]]]]
  a_codes <- as.numeric(unlist(a$codes, use.names = FALSE))
  b_codes <- as.numeric(unlist(b$codes, use.names = FALSE))
  grid <- expand.grid(a_code = a_codes, b_code = b_codes,
                      KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE)
  grid$cell_index <- seq_len(nrow(grid))
  observed <- table(factor(raw[[a$column]], levels = a_codes), factor(raw[[b$column]], levels = b_codes))
  observed_count <- as.integer(observed)
  ua <- helmert_function_basis(length(a_codes))
  ub <- helmert_function_basis(length(b_codes))
  basis <- matrix(NA_real_, nrow = nrow(grid), ncol = ncol(ua) * ncol(ub))
  coordinate_labels <- character(ncol(basis))
  cc <- 0L
  for (jb in seq_len(ncol(ub))) {
    for (ja in seq_len(ncol(ua))) {
      cc <- cc + 1L
      basis[, cc] <- ua[match(grid$a_code, a_codes), ja] * ub[match(grid$b_code, b_codes), jb]
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
  expected_rank <- as.integer(family$interaction_rank)
  country_min <- Inf
  task_min <- Inf
  for (country_code in 1:4) {
    idx <- raw$country == country_code
    tab <- table(factor(raw[[a$column]][idx], levels = a_codes),
                 factor(raw[[b$column]][idx], levels = b_codes))
    assert(all(tab > 0), "Family %s is incomplete in country %s", family_id, country_names[[country_code]])
    country_min <- min(country_min, tab)
  }
  for (task in 1:4) {
    idx <- raw$conjoint == task
    tab <- table(factor(raw[[a$column]][idx], levels = a_codes),
                 factor(raw[[b$column]][idx], levels = b_codes))
    assert(all(tab > 0), "Family %s is incomplete in task %s", family_id, task)
    task_min <- min(task_min, tab)
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
    minimum_country_cell_count = as.integer(country_min),
    minimum_task_cell_count = as.integer(task_min),
    additive_rank = 1 + length(a_codes) - 1 + length(b_codes) - 1,
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
          contract$estimands$total_dimension,
          sum(family_identification$observed_interaction_rank) == as.integer(contract$estimands$total_dimension))
add_check("registered_family_country_support", min(family_identification$minimum_country_cell_count),
          ">0", all(family_identification$minimum_country_cell_count > 0))
add_check("registered_family_task_support", min(family_identification$minimum_task_cell_count),
          ">0", all(family_identification$minimum_task_cell_count > 0))
write_csv(family_identification, file.path(output_dir, "family_identification.csv"))
write_csv(contrast_coordinates, file.path(output_dir, "contrast_coordinates.csv"))
write_csv(family_projectors, file.path(output_dir, "family_projectors.csv"))

write_csv(validation, file.path(output_dir, "design_validation.csv"))
assert(all(validation$pass), "At least one climate design-contract validation failed; inspect design_validation.csv")

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
  gate_result = "split_rating_green_paired_choice_amber",
  portfolio_eligible_scope = "rating_only_under_mean_no_opponent_and_no_carryover_restrictions",
  outcome_blind = TRUE,
  outcomes_loaded = FALSE,
  outcome_models_fit = FALSE,
  read_method = "haven::read_dta(col_select=locked_allowlist)",
  respondents = nrow(respondents),
  profile_rows = nrow(raw),
  profile_support = number_profiles,
  registered_family_dimensions = as.list(setNames(
    family_identification$observed_interaction_rank,
    family_identification$family
  )),
  pair_law = list(
    q_times_q_certified = FALSE,
    observed_identical_pairs = sum(identical_complete_pair),
    expected_under_q_times_q = iid_expected_identical,
    probability_zero_under_iid_task_pairs = iid_probability_zero,
    upper_probability_zero_with_arbitrary_four_task_dependence = respondent_cluster_upper_probability_zero,
    next_unblocker = "Obtain the original randomizer/generator or author confirmation of the no-identical-profile rule and exact conditional opponent probabilities; otherwise paired choice stays out."
  ),
  generated_files_md5 = as.list(generated_md5),
  note = "No choice_cj, rating_cj, or incon value was loaded, summarized, or modeled."
)
yaml::write_yaml(manifest, manifest_path)

cat(sprintf(
  paste0(
    "Climate design audit passed outcome-blind: %s respondents, %s profiles, all 2,880 profile cells, ",
    "and family ranks 12/9/4. Rating is conditionally eligible under mean no-opponent and no-carryover; ",
    "paired choice remains amber because the opponent law is uncertified.\n"
  ),
  format(nrow(respondents), big.mark = ","),
  format(nrow(raw), big.mark = ",")
))
