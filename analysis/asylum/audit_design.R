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
    if (file.exists(file.path(candidate, "config", "datasets", "asylum-seekers.yml"))) return(candidate)
  }
  abort("Could not locate the repository root containing config/datasets/asylum-seekers.yml")
}

write_csv <- function(x, path) utils::write.csv(x, path, row.names = FALSE, na = "")
resolve_path <- function(path, root) if (grepl("^/", path)) path else file.path(root, path)

repo_root <- find_repo_root()
contract_arg <- arg_value("--contract", "config/datasets/asylum-seekers.yml")
output_arg <- arg_value("--output-dir", "analysis/asylum/derived")
contract_path <- resolve_path(contract_arg, repo_root)
output_dir <- resolve_path(output_arg, repo_root)

suppressPackageStartupMessages(library("data.table", character.only = TRUE))
assert(requireNamespace("yaml", quietly = TRUE), "Package 'yaml' is required")
data.table::setDTthreads(1L)

contract <- yaml::read_yaml(contract_path)
assert(identical(contract$status, "design_lock_outcomes_prohibited"), "Unexpected contract status")
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
  "frame_counts.csv",
  "level_lock.csv",
  "loaded_columns.csv",
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
old_manifest <- file.path(output_dir, "design_audit_manifest.yml")
if (file.exists(old_manifest)) unlink(old_manifest)

# Lock every raw or documentary source before reading a design column.
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

forbidden_outcomes <- unlist(contract$read_barrier$forbidden_outcomes, use.names = FALSE)
conjoint_allowed <- unlist(contract$read_barrier$conjoint_allowed_columns, use.names = FALSE)
respondent_common <- unlist(contract$read_barrier$respondent_allowed_columns$common, use.names = FALSE)
respondent_2022_extra <- unlist(contract$read_barrier$respondent_allowed_columns$wave_2022, use.names = FALSE)
assert(!any(conjoint_allowed %in% forbidden_outcomes), "An outcome appears in the conjoint allowlist")
assert(!any(respondent_common %in% forbidden_outcomes), "An outcome appears in the respondent allowlist")
assert(!anyDuplicated(conjoint_allowed), "The conjoint allowlist has duplicate columns")
assert(!anyDuplicated(c(respondent_common, respondent_2022_extra)), "The respondent allowlist has duplicate columns")

wave_ids <- c("wave_2016", "wave_2022")
wave_years <- c(wave_2016 = 2016L, wave_2022 = 2022L)
conjoint_file_ids <- c(wave_2016 = "conjoint_2016", wave_2022 = "conjoint_2022")
respondent_file_ids <- c(wave_2016 = "respondents_2016", wave_2022 = "respondents_2022")

attribute_ids <- names(contract$attributes)
attribute_columns <- vapply(contract$attributes, `[[`, character(1), "column")
levels_for <- function(attribute_id, wave_id) {
  spec <- contract$attributes[[attribute_id]]
  if (!is.null(spec$levels_by_wave)) {
    unlist(spec$levels_by_wave[[wave_id]], use.names = FALSE)
  } else {
    unlist(spec$levels, use.names = FALSE)
  }
}

validation <- data.frame(
  wave = character(), check = character(), observed = character(), expected = character(),
  pass = logical(), detail = character()
)
add_check <- function(wave, check, observed, expected, pass, detail = "") {
  validation <<- rbind(validation, data.frame(
    wave = wave, check = check, observed = as.character(observed), expected = as.character(expected),
    pass = isTRUE(pass), detail = detail
  ))
}

loaded_columns <- data.frame()
sample_summary <- data.frame()
country_counts <- data.frame()
level_lock <- data.frame()
all_pair_support <- data.frame()
assignment_diagnostics <- data.frame()
frame_counts <- data.frame()
covariate_audit <- data.frame()
covariate_category_levels <- data.frame()
target_weight_diagnostics <- data.frame()
wave_data <- list()
fold_source <- data.frame()

role_for_respondent_column <- function(nm) {
  if (nm == "ResponseId") return("cluster_id")
  if (nm == "frame") return("randomized_preconjoint_context")
  if (nm == "weight") return("primary_survey_weight")
  if (nm == "alt_weight") return("sensitivity_survey_weight")
  if (nm %in% unlist(contract$covariates$primary_raw, use.names = FALSE)) return("primary_nuisance_feature")
  if (nm %in% unlist(contract$covariates$post_conjoint_sensitivity_only, use.names = FALSE)) return("post_conjoint_sensitivity_only")
  if (nm %in% unlist(contract$covariates$exclude_post_conjoint, use.names = FALSE)) return("excluded_post_conjoint")
  "audited_excluded_or_timing_unresolved"
}

for (wave_id in wave_ids) {
  year <- wave_years[[wave_id]]
  expected <- contract$sample$waves[[wave_id]]
  conjoint_path <- resolve_path(file_specs[[conjoint_file_ids[[wave_id]]]]$path, repo_root)
  respondent_path <- resolve_path(file_specs[[respondent_file_ids[[wave_id]]]]$path, repo_root)
  respondent_allowed <- c(respondent_common, if (wave_id == "wave_2022") respondent_2022_extra)

  # fread(select=...) is the enforceable outcome read barrier.
  profiles <- as.data.frame(data.table::fread(
    conjoint_path, select = conjoint_allowed, na.strings = c("", "NA"),
    check.names = FALSE, showProgress = FALSE
  ))
  respondents <- as.data.frame(data.table::fread(
    respondent_path, select = respondent_allowed, na.strings = c("", "NA"),
    check.names = FALSE, showProgress = FALSE
  ))
  assert(identical(names(profiles), conjoint_allowed), "Wave %s conjoint allowlist did not match", year)
  assert(identical(names(respondents), respondent_allowed), "Wave %s respondent allowlist did not match", year)
  assert(!any(c(names(profiles), names(respondents)) %in% forbidden_outcomes),
         "Wave %s loaded a forbidden outcome", year)

  loaded_columns <- rbind(
    loaded_columns,
    data.frame(wave = year, source = "conjoint", column = names(profiles),
               role = ifelse(names(profiles) %in% attribute_columns, "randomized_attribute", "design_identifier"),
               outcome_loaded = FALSE),
    data.frame(wave = year, source = "respondent", column = names(respondents),
               role = vapply(names(respondents), role_for_respondent_column, character(1)),
               outcome_loaded = FALSE)
  )

  profiles$.source_row <- seq_len(nrow(profiles))
  profile_id <- as.character(profiles$ResponseId)
  respondent_id <- as.character(respondents$ResponseId)
  assert(!anyDuplicated(respondent_id), "Wave %s respondent file has duplicate IDs", year)
  assert(setequal(profile_id, respondent_id), "Wave %s respondent and conjoint ID sets differ", year)
  assert(!anyDuplicated(profiles[c("ResponseId", "task", "prof")]),
         "Wave %s has duplicate respondent-task-profile keys", year)

  rows_per_id_raw <- table(profile_id)
  tasks_per_id_raw <- tapply(as.character(profiles$task), profile_id, function(x) length(unique(x)))
  design_complete_row <- Reduce(`&`, lapply(profiles[attribute_columns], function(x) !is.na(x)))
  complete_by_id <- tapply(design_complete_row, profile_id, sum)
  assert(all(complete_by_id == 0L | complete_by_id == rows_per_id_raw),
         "Wave %s contains a partially missing profile block", year)

  design <- profiles[design_complete_row, , drop = FALSE]
  design_id <- as.character(design$ResponseId)
  respondent_match <- match(design_id, respondent_id)
  assert(!anyNA(respondent_match), "Wave %s respondent join failed", year)
  respondent_join <- respondents[respondent_match, , drop = FALSE]
  for (nm in setdiff(names(respondent_join), "ResponseId")) design[[nm]] <- respondent_join[[nm]]

  pair_key <- paste(design_id, design$task, sep = "::")
  respondent_rows <- table(design_id)
  respondent_tasks <- tapply(as.character(design$task), design_id, function(x) length(unique(x)))
  pair_rows <- table(pair_key)
  # Direct string checks avoid assigning a meaning beyond the released first/second profile label.
  prof_task_match <- substr(as.character(design$prof), 1L, 1L) == as.character(design$task)
  prof_side <- substr(as.character(design$prof), 2L, 2L)
  sides_by_pair <- tapply(prof_side, pair_key,
                          function(x) length(x) == 2L && identical(sort(unname(x)), c("1", "2")))

  weight_ok <- is.finite(as.numeric(design$weight)) & as.numeric(design$weight) > 0
  eligible_id <- unique(design_id[weight_ok])
  weight_complete_by_id <- vapply(split(weight_ok, design_id), function(x) all(x), logical(1))
  assert(all(vapply(split(as.numeric(design$weight), design_id), function(x) {
    token <- ifelse(is.na(x), "<NA>", format(x, digits = 17))
    length(unique(token)) == 1L
  }, logical(1))), "Wave %s primary weight varies within respondent", year)
  assert(all(weight_complete_by_id | vapply(split(weight_ok, design_id), function(x) !any(x), logical(1))),
         "Wave %s has partial within-respondent primary-weight availability", year)

  design_complete_ids <- unique(design_id)
  respondent_design <- respondents[respondent_id %in% design_complete_ids, , drop = FALSE]
  respondent_weighted <- respondent_design[
    is.finite(as.numeric(respondent_design$weight)) & as.numeric(respondent_design$weight) > 0, , drop = FALSE
  ]

  add_check(year, "archive_profile_rows", nrow(profiles), expected$archive_profile_rows,
            nrow(profiles) == expected$archive_profile_rows)
  add_check(year, "archive_respondents", length(unique(profile_id)), expected$archive_respondents,
            length(unique(profile_id)) == expected$archive_respondents)
  add_check(year, "archive_tasks", length(unique(paste(profile_id, profiles$task))), expected$archive_tasks,
            length(unique(paste(profile_id, profiles$task))) == expected$archive_tasks)
  add_check(year, "raw_rows_per_respondent", paste(range(rows_per_id_raw), collapse = ":"), 10,
            all(rows_per_id_raw == 10L))
  add_check(year, "raw_tasks_per_respondent", paste(range(tasks_per_id_raw), collapse = ":"), 5,
            all(tasks_per_id_raw == 5L))
  add_check(year, "design_complete_profile_rows", nrow(design), expected$design_complete_profile_rows,
            nrow(design) == expected$design_complete_profile_rows)
  add_check(year, "design_complete_respondents", length(design_complete_ids), expected$design_complete_respondents,
            length(design_complete_ids) == expected$design_complete_respondents)
  add_check(year, "design_complete_tasks", length(pair_rows), expected$design_complete_tasks,
            length(pair_rows) == expected$design_complete_tasks)
  add_check(year, "all_profile_missing_respondents", sum(complete_by_id == 0L),
            expected$all_profile_missing_respondents, sum(complete_by_id == 0L) == expected$all_profile_missing_respondents)
  add_check(year, "rows_per_design_respondent", paste(range(respondent_rows), collapse = ":"), 10,
            all(respondent_rows == 10L))
  add_check(year, "tasks_per_design_respondent", paste(range(respondent_tasks), collapse = ":"), 5,
            all(respondent_tasks == 5L))
  add_check(year, "task_levels", paste(sort(unique(design$task)), collapse = ","), "A,B,C,D,E",
            identical(sort(unique(as.character(design$task))), LETTERS[1:5]))
  add_check(year, "profiles_per_task", paste(range(pair_rows), collapse = ":"), 2,
            all(pair_rows == 2L))
  add_check(year, "profile_task_letters", sum(prof_task_match), nrow(design), all(prof_task_match))
  add_check(year, "profile_sides_1_and_2", sum(sides_by_pair), length(sides_by_pair), all(sides_by_pair))
  add_check(year, "respondent_conjoint_id_set", length(intersect(unique(profile_id), respondent_id)),
            expected$archive_respondents, setequal(profile_id, respondent_id))
  add_check(year, "primary_weight_eligible_respondents", nrow(respondent_weighted),
            expected$primary_weight_eligible_respondents,
            nrow(respondent_weighted) == expected$primary_weight_eligible_respondents)
  add_check(year, "primary_weight_eligible_profile_rows", sum(weight_ok),
            expected$primary_weight_eligible_profile_rows,
            sum(weight_ok) == expected$primary_weight_eligible_profile_rows)
  add_check(year, "primary_weight_eligible_tasks", sum(weight_complete_by_id) * 5L,
            expected$primary_weight_eligible_tasks,
            sum(weight_complete_by_id) * 5L == expected$primary_weight_eligible_tasks)
  add_check(year, "missing_primary_weight_among_design_complete", sum(!weight_complete_by_id),
            expected$missing_primary_weight_among_design_complete,
            sum(!weight_complete_by_id) == expected$missing_primary_weight_among_design_complete)
  add_check(year, "observed_primary_weights_positive", sum(is.finite(as.numeric(design$weight)) & as.numeric(design$weight) > 0),
            sum(!is.na(design$weight)), all(is.na(design$weight) | (is.finite(as.numeric(design$weight)) & as.numeric(design$weight) > 0)))
  add_check(year, "primary_weight_trim_bound", max(as.numeric(design$weight), na.rm = TRUE), 6,
            max(as.numeric(design$weight), na.rm = TRUE) <= 6 + 1e-12)

  sample_summary <- rbind(sample_summary, data.frame(
    wave = year,
    archive_respondents = length(unique(profile_id)),
    archive_tasks = length(unique(paste(profile_id, profiles$task))),
    archive_profile_rows = nrow(profiles),
    all_profile_missing_respondents = sum(complete_by_id == 0L),
    design_complete_respondents = length(design_complete_ids),
    design_complete_tasks = length(pair_rows),
    design_complete_profile_rows = nrow(design),
    primary_weight_eligible_respondents = nrow(respondent_weighted),
    primary_weight_eligible_tasks = nrow(respondent_weighted) * 5L,
    primary_weight_eligible_profile_rows = nrow(respondent_weighted) * 10L
  ))

  observed_country <- merge(
    as.data.frame(table(country = respondents$cty), stringsAsFactors = FALSE),
    as.data.frame(table(country = respondent_design$cty), stringsAsFactors = FALSE),
    by = "country", all = TRUE, suffixes = c("_archive", "_design")
  )
  observed_country <- merge(
    observed_country,
    as.data.frame(table(country = respondent_weighted$cty), stringsAsFactors = FALSE),
    by = "country", all = TRUE
  )
  names(observed_country) <- c("country", "archive_respondents", "design_complete_respondents",
                               "primary_weight_eligible_respondents")
  observed_country$wave <- year
  observed_country <- observed_country[, c("wave", "country", "archive_respondents",
                                            "design_complete_respondents", "primary_weight_eligible_respondents")]
  country_lock <- contract$sample$country_count_lock[[wave_id]]
  for (country in names(country_lock)) {
    expected_counts <- as.integer(unlist(country_lock[[country]], use.names = FALSE))
    got <- observed_country[observed_country$country == country, 3:5, drop = TRUE]
    add_check(year, paste0("country_counts_", country), paste(got, collapse = ":"),
              paste(expected_counts, collapse = ":"), identical(as.integer(got), expected_counts))
  }
  country_counts <- rbind(country_counts, observed_country)

  # Exact level sets, pair support, and marginal assignment diagnostics.
  for (attribute_id in attribute_ids) {
    column <- contract$attributes[[attribute_id]]$column
    locked_levels <- levels_for(attribute_id, wave_id)
    observed_levels <- unique(as.character(design[[column]]))
    counts <- table(factor(as.character(design[[column]]), levels = locked_levels))
    add_check(year, paste0("level_set_", attribute_id), paste(sort(observed_levels), collapse = " | "),
              paste(sort(locked_levels), collapse = " | "), setequal(observed_levels, locked_levels))
    level_lock <- rbind(level_lock, data.frame(
      wave = year, attribute = attribute_id, column = column,
      level_index = seq_along(locked_levels), level = locked_levels,
      observed_count = as.integer(counts), expected_probability = 1 / length(locked_levels)
    ))
    n <- sum(counts)
    expected_count <- n / length(locked_levels)
    assignment_diagnostics <- rbind(assignment_diagnostics, data.frame(
      wave = year, component = column, stratum = "unconditional", level = locked_levels,
      observed_count = as.integer(counts), expected_count = expected_count,
      observed_probability = as.integer(counts) / n,
      expected_probability = 1 / length(locked_levels),
      pearson_residual = (as.integer(counts) - expected_count) / sqrt(expected_count)
    ))
  }

  pair_list <- utils::combn(attribute_ids, 2L, simplify = FALSE)
  for (ids in pair_list) {
    a_levels <- levels_for(ids[[1]], wave_id)
    b_levels <- levels_for(ids[[2]], wave_id)
    a_col <- contract$attributes[[ids[[1]]]]$column
    b_col <- contract$attributes[[ids[[2]]]]$column
    tab <- table(factor(as.character(design[[a_col]]), levels = a_levels),
                 factor(as.character(design[[b_col]]), levels = b_levels))
    observed_cells <- sum(tab > 0)
    expected_cells <- length(a_levels) * length(b_levels)
    all_pair_support <- rbind(all_pair_support, data.frame(
      wave = year, attribute_a = ids[[1]], attribute_b = ids[[2]],
      cartesian_cells = expected_cells, observed_cells = observed_cells,
      minimum_cell_count = min(tab), maximum_cell_count = max(tab),
      complete_cartesian = observed_cells == expected_cells
    ))
  }
  add_check(year, "all_36_pair_supports_complete",
            sum(all_pair_support$wave == year & all_pair_support$complete_cartesian), 36,
            sum(all_pair_support$wave == year & all_pair_support$complete_cartesian) == 36L)

  profile_count <- prod(vapply(attribute_ids, function(id) length(levels_for(id, wave_id)), integer(1)))
  expected_profile_count <- as.numeric(contract$support$profiles[[wave_id]])
  add_check(year, "cartesian_profile_count", profile_count, expected_profile_count,
            profile_count == expected_profile_count)

  # Frame was randomized before the conjoint and is a 2022 design stratum.
  if (wave_id == "wave_2022") {
    locked_frames <- unlist(contract$assignment$frame_2022$levels, use.names = FALSE)
    raw_frame <- table(factor(as.character(respondents$frame), levels = locked_frames))
    design_frame <- table(factor(as.character(respondent_design$frame), levels = locked_frames))
    weighted_frame <- table(factor(as.character(respondent_weighted$frame), levels = locked_frames))
    frame_counts <- data.frame(
      wave = year, frame = locked_frames,
      archive_respondents = as.integer(raw_frame),
      design_complete_respondents = as.integer(design_frame),
      primary_weight_eligible_respondents = as.integer(weighted_frame),
      target_probability = 0.25
    )
    expected_raw_frame <- unlist(contract$assignment$frame_2022$realized_counts$archive[locked_frames], use.names = FALSE)
    expected_design_frame <- unlist(contract$assignment$frame_2022$realized_counts$design_complete[locked_frames], use.names = FALSE)
    add_check(year, "frame_levels", paste(sort(unique(respondents$frame)), collapse = ","),
              paste(sort(locked_frames), collapse = ","), setequal(respondents$frame, locked_frames))
    add_check(year, "frame_archive_counts", paste(raw_frame, collapse = ":"),
              paste(expected_raw_frame, collapse = ":"), identical(as.integer(raw_frame), as.integer(expected_raw_frame)))
    add_check(year, "frame_design_counts", paste(design_frame, collapse = ":"),
              paste(expected_design_frame, collapse = ":"), identical(as.integer(design_frame), as.integer(expected_design_frame)))
    frame_expected <- sum(raw_frame) / 4
    assignment_diagnostics <- rbind(assignment_diagnostics, data.frame(
      wave = year, component = "frame", stratum = "respondent_randomized_context", level = locked_frames,
      observed_count = as.integer(raw_frame), expected_count = frame_expected,
      observed_probability = as.integer(raw_frame) / sum(raw_frame), expected_probability = 0.25,
      pearson_residual = (as.integer(raw_frame) - frame_expected) / sqrt(frame_expected)
    ))
  }

  # Covariate audit uses one row per respondent and never an outcome.
  primary_features <- unlist(contract$covariates$primary_raw, use.names = FALSE)
  feature_semantics <- contract$covariates$feature_semantics
  for (nm in setdiff(names(respondents), "ResponseId")) {
    x <- respondents[[nm]]
    numeric_x <- suppressWarnings(as.numeric(as.character(x)))
    numeric_ok <- all(is.na(x) | !is.na(numeric_x))
    covariate_audit <- rbind(covariate_audit, data.frame(
      wave = year, variable = nm, role = role_for_respondent_column(nm),
      primary_feature = nm %in% primary_features,
      n = length(x), nonmissing = sum(!is.na(x)), missing = sum(is.na(x)),
      unique_nonmissing = length(unique(x[!is.na(x)])),
      minimum = if (numeric_ok && any(!is.na(numeric_x))) min(numeric_x, na.rm = TRUE) else NA_real_,
      maximum = if (numeric_ok && any(!is.na(numeric_x))) max(numeric_x, na.rm = TRUE) else NA_real_
    ))
  }
  categorical_primary <- setdiff(primary_features, "age")
  for (nm in categorical_primary) {
    vals <- sort(unique(as.character(respondents[[nm]][!is.na(respondents[[nm]])])))
    vals <- c(vals, "__MISSING__")
    covariate_category_levels <- rbind(covariate_category_levels, data.frame(
      wave = year, variable = nm, level_index = seq_along(vals), level = vals,
      observed_nonmissing = vals != "__MISSING__"
    ))
  }
  add_check(year, "primary_covariate_columns_present", sum(primary_features %in% names(respondents)),
            length(primary_features), all(primary_features %in% names(respondents)))
  add_check(year, "age_missing", sum(is.na(respondents$age)), 0, !anyNA(respondents$age))
  add_check(year, "age_over_110", sum(as.numeric(respondents$age) > 110), if (year == 2016) 1 else 0,
            sum(as.numeric(respondents$age) > 110) == if (year == 2016) 1 else 0,
            "The contract keeps enrollment, flags values above 110, and imputes age within folds.")

  # Equal-country target: normalize author weights within each country, then give each country 1/15.
  target_country_levels <- sort(unique(as.character(respondent_weighted$cty)))
  assert(length(target_country_levels) == 15L, "Wave %s does not contain 15 target countries", year)
  for (country in target_country_levels) {
    idx <- as.character(respondent_weighted$cty) == country
    w <- as.numeric(respondent_weighted$weight[idx])
    normalized <- w / sum(w) / length(target_country_levels)
    target_weight_diagnostics <- rbind(target_weight_diagnostics, data.frame(
      wave = year, country = country, respondents = length(w),
      raw_weight_sum = sum(w), normalized_target_weight_sum = sum(normalized),
      effective_sample_size = sum(w)^2 / sum(w^2), maximum_raw_weight = max(w)
    ))
  }
  add_check(year, "equal_country_target_sums_to_one",
            sum(target_weight_diagnostics$normalized_target_weight_sum[target_weight_diagnostics$wave == year]), 1,
            abs(sum(target_weight_diagnostics$normalized_target_weight_sum[target_weight_diagnostics$wave == year]) - 1) < 1e-12)

  fold_source <- rbind(fold_source, data.frame(
    wave = year,
    ResponseId = as.character(respondent_design$ResponseId),
    country = as.character(respondent_design$cty),
    frame = if (wave_id == "wave_2022") as.character(respondent_design$frame) else "not_applicable",
    primary_weight_eligible = is.finite(as.numeric(respondent_design$weight)) &
      as.numeric(respondent_design$weight) > 0,
    stringsAsFactors = FALSE
  ))

  wave_data[[wave_id]] <- list(design = design, respondents = respondents)
}

add_check("both", "respondent_ids_disjoint_across_waves",
          length(intersect(as.character(wave_data$wave_2016$respondents$ResponseId),
                           as.character(wave_data$wave_2022$respondents$ResponseId))),
          0,
          !length(intersect(as.character(wave_data$wave_2016$respondents$ResponseId),
                            as.character(wave_data$wave_2022$respondents$ResponseId))),
          "The waves cannot be analyzed as a respondent panel.")

add_check("both", "maximum_absolute_uniform_assignment_residual",
          format(max(abs(assignment_diagnostics$pearson_residual)), digits = 6), "<5",
          max(abs(assignment_diagnostics$pearson_residual)) < 5,
          "A prespecified coarse diagnostic, not evidence that realized frequencies prove randomization.")

# Freeze respondent-level folds using only identifiers and design strata. Two
# stable string hashes keep global inner labels from duplicating the outer
# partition; inner labels are used only inside each outer training set.
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
fold_source$stratum <- paste(fold_source$wave, fold_source$country, fold_source$frame, sep = "::")
fold_source$outer_fold <- ave(
  seq_len(nrow(fold_source)), fold_source$stratum,
  FUN = function(ix) assign_balanced_folds(fold_source$ResponseId[ix], outer_k, fold_seed)
)
fold_source$inner_fold_label <- ave(
  seq_len(nrow(fold_source)), fold_source$stratum,
  FUN = function(ix) assign_balanced_folds(fold_source$ResponseId[ix], inner_k, fold_seed + 104729L,
                                           multiplier = 257, reverse_codes = TRUE)
)
fold_source$outer_fold <- as.integer(fold_source$outer_fold)
fold_source$inner_fold_label <- as.integer(fold_source$inner_fold_label)
assert(!anyDuplicated(fold_source[c("wave", "ResponseId")]), "Fold file has duplicate wave-respondent keys")
outer_balance <- vapply(
  split(fold_source$outer_fold, fold_source$stratum),
  function(x) diff(range(tabulate(x, nbins = outer_k))) <= 1L,
  logical(1)
)
inner_balance <- vapply(
  split(fold_source$inner_fold_label, fold_source$stratum),
  function(x) diff(range(tabulate(x, nbins = inner_k))) <= 1L,
  logical(1)
)
nested_coverage <- unlist(lapply(seq_len(outer_k), function(held_out) {
  vapply(split(fold_source, fold_source$stratum), function(x) {
    train <- x[x$outer_fold != held_out, , drop = FALSE]
    length(unique(train$inner_fold_label)) == inner_k
  }, logical(1))
}))
for (wave_id in wave_ids) {
  year <- wave_years[[wave_id]]
  design <- wave_data[[wave_id]]$design
  wave_folds <- fold_source[fold_source$wave == year, , drop = FALSE]
  fold_match <- match(as.character(design$ResponseId), wave_folds$ResponseId)
  add_check(year, "all_profile_rows_map_to_one_respondent_fold", sum(!is.na(fold_match)), nrow(design),
            !anyNA(fold_match))
}
add_check("both", "outer_fold_stratum_balance", sum(outer_balance), length(outer_balance), all(outer_balance))
add_check("both", "inner_fold_label_stratum_balance", sum(inner_balance), length(inner_balance), all(inner_balance))
add_check("both", "inner_fold_coverage_in_every_outer_training_set", sum(nested_coverage),
          length(nested_coverage), all(nested_coverage))
write_csv(
  fold_source[, c("wave", "ResponseId", "country", "frame", "primary_weight_eligible",
                  "outer_fold", "inner_fold_label")],
  file.path(output_dir, "respondent_folds.csv")
)

write_csv(loaded_columns, file.path(output_dir, "loaded_columns.csv"))
write_csv(sample_summary, file.path(output_dir, "sample_summary.csv"))
write_csv(country_counts, file.path(output_dir, "country_counts.csv"))
write_csv(level_lock, file.path(output_dir, "level_lock.csv"))
write_csv(all_pair_support, file.path(output_dir, "all_pair_support_summary.csv"))
write_csv(assignment_diagnostics, file.path(output_dir, "assignment_diagnostics.csv"))
write_csv(frame_counts, file.path(output_dir, "frame_counts.csv"))
write_csv(covariate_audit, file.path(output_dir, "covariate_audit.csv"))
write_csv(covariate_category_levels, file.path(output_dir, "covariate_category_levels.csv"))
write_csv(target_weight_diagnostics, file.path(output_dir, "target_weight_diagnostics.csv"))

feature_semantics <- contract$covariates$feature_semantics
primary_features <- unlist(contract$covariates$primary_raw, use.names = FALSE)
assert(setequal(names(feature_semantics), primary_features),
       "Feature semantics must cover every and only primary features")
covariate_feature_lock <- do.call(rbind, lapply(names(feature_semantics), function(nm) {
  spec <- feature_semantics[[nm]]
  data.frame(
    variable = nm, semantic_type = spec$semantic_type,
    primary_encoding = spec$primary_encoding,
    missing_value_rule = spec$missing_value_rule,
    scaling_rule = spec$scaling_rule
  )
}))
write_csv(covariate_feature_lock, file.path(output_dir, "covariate_feature_lock.csv"))

# Uniform-target, coding-invariant whole-family bases and projectors.
helmert_function_basis <- function(n_levels) {
  if (n_levels < 2L) return(matrix(numeric(), nrow = n_levels, ncol = 0L))
  h <- stats::contr.helmert(n_levels)
  sweep(h, 2L, sqrt(colSums(h^2) / n_levels), "/")
}

family_identification <- data.frame()
contrast_coordinates <- data.frame()
family_projectors <- data.frame()

for (wave_id in wave_ids) {
  year <- wave_years[[wave_id]]
  design <- wave_data[[wave_id]]$design
  for (family in contract$estimands$families) {
    family_id <- family$id
    ids <- unlist(family$attributes, use.names = FALSE)
    a_id <- ids[[1]]
    b_id <- ids[[2]]
    a_col <- contract$attributes[[a_id]]$column
    b_col <- contract$attributes[[b_id]]$column
    a_levels <- levels_for(a_id, wave_id)
    b_levels <- levels_for(b_id, wave_id)
    grid <- expand.grid(
      a_level = a_levels, b_level = b_levels,
      KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE
    )
    grid$cell_index <- seq_len(nrow(grid))
    observed <- table(factor(as.character(design[[a_col]]), levels = a_levels),
                      factor(as.character(design[[b_col]]), levels = b_levels))
    observed_count <- as.integer(observed)
    assert(length(observed_count) == nrow(grid), "Family grid count mismatch for %s", family_id)
    # table() and expand.grid() both vary the first factor fastest.
    ua <- helmert_function_basis(length(a_levels))
    ub <- helmert_function_basis(length(b_levels))
    basis <- matrix(NA_real_, nrow = nrow(grid), ncol = ncol(ua) * ncol(ub))
    coordinate_labels <- character(ncol(basis))
    cc <- 0L
    for (jb in seq_len(ncol(ub))) {
      for (ja in seq_len(ncol(ua))) {
        cc <- cc + 1L
        basis[, cc] <- ua[match(grid$a_level, a_levels), ja] * ub[match(grid$b_level, b_levels), jb]
        coordinate_labels[[cc]] <- sprintf("h%02d_x_h%02d", ja, jb)
      }
    }
    p <- rep(1 / nrow(grid), nrow(grid))
    w <- diag(p)
    intercept <- rep(1, nrow(grid))
    a_main <- ua[match(grid$a_level, a_levels), , drop = FALSE]
    b_main <- ub[match(grid$b_level, b_levels), , drop = FALSE]
    main <- cbind(intercept, a_main, b_main)
    gram <- crossprod(basis, w %*% basis)
    mean_error <- max(abs(crossprod(intercept * p, basis)))
    main_error <- max(abs(crossprod(main, w %*% basis)))
    orthonormal_error <- max(abs(gram - diag(ncol(basis))))
    projector <- basis %*% t(basis) %*% w
    idempotence_error <- max(abs(projector %*% projector - projector))
    expected_dimension <- as.integer(family$dimension_by_wave[[wave_id]])
    observed_rank <- qr(basis, tol = 1e-10)$rank
    assert(observed_rank == expected_dimension, "Interaction rank mismatch for %s in %s", family_id, year)
    assert(all(observed_count > 0), "A registered family cell is unobserved for %s in %s", family_id, year)
    assert(orthonormal_error < 1e-10 && main_error < 1e-10 && idempotence_error < 1e-10,
           "Invariant basis numerical check failed for %s in %s", family_id, year)

    # Verify each registered family is complete within all 15 countries.
    country_min <- Inf
    for (country in unique(as.character(design$cty))) {
      idx <- as.character(design$cty) == country
      country_tab <- table(factor(as.character(design[[a_col]][idx]), levels = a_levels),
                           factor(as.character(design[[b_col]][idx]), levels = b_levels))
      assert(all(country_tab > 0), "Family %s lacks support in %s, %s", family_id, country, year)
      country_min <- min(country_min, country_tab)
    }

    family_identification <- rbind(family_identification, data.frame(
      wave = year, family = family_id, attribute_a = a_id, attribute_b = b_id,
      levels_a = length(a_levels), levels_b = length(b_levels), cells = nrow(grid),
      observed_cells = sum(observed_count > 0), minimum_cell_count = min(observed_count),
      minimum_country_cell_count = as.integer(country_min),
      additive_rank = 1L + (length(a_levels) - 1L) + (length(b_levels) - 1L),
      expected_interaction_rank = expected_dimension, observed_interaction_rank = observed_rank,
      weighted_mean_error = mean_error, main_effect_orthogonality_error = main_error,
      orthonormality_error = orthonormal_error, projector_idempotence_error = idempotence_error,
      identified = observed_rank == expected_dimension && all(observed_count > 0)
    ))
    for (j in seq_len(ncol(basis))) {
      contrast_coordinates <- rbind(contrast_coordinates, data.frame(
        wave = year, family = family_id, coordinate = coordinate_labels[[j]],
        cell_index = grid$cell_index, a_level = grid$a_level, b_level = grid$b_level,
        target_probability = p, basis_value = basis[, j]
      ))
    }
    projector_grid <- expand.grid(
      row_cell_index = seq_len(nrow(grid)), column_cell_index = seq_len(nrow(grid)),
      KEEP.OUT.ATTRS = FALSE
    )
    family_projectors <- rbind(family_projectors, data.frame(
      wave = year, family = family_id, projector_grid,
      value = as.vector(projector)
    ))
  }
}

add_check("both", "registered_family_ranks",
          sum(family_identification$identified), nrow(family_identification),
          all(family_identification$identified))
add_check("both", "registered_family_country_support",
          min(family_identification$minimum_country_cell_count), ">0",
          all(family_identification$minimum_country_cell_count > 0))
add_check("both", "basis_orthonormality", max(family_identification$orthonormality_error), "<1e-10",
          max(family_identification$orthonormality_error) < 1e-10)
add_check("both", "basis_main_effect_orthogonality", max(family_identification$main_effect_orthogonality_error), "<1e-10",
          max(family_identification$main_effect_orthogonality_error) < 1e-10)

write_csv(family_identification, file.path(output_dir, "family_identification.csv"))
write_csv(contrast_coordinates, file.path(output_dir, "contrast_coordinates.csv"))
write_csv(family_projectors, file.path(output_dir, "family_projectors.csv"))

profile_probability_strata <- data.frame(
  wave = c(2016L, 2022L),
  profile_support_size = c(as.numeric(contract$support$profiles$wave_2016),
                           as.numeric(contract$support$profiles$wave_2022))
)
profile_probability_strata$q_profile <- 1 / profile_probability_strata$profile_support_size
profile_probability_strata$ordered_pair_support_size <- profile_probability_strata$profile_support_size^2
profile_probability_strata$q_ordered_pair <- profile_probability_strata$q_profile^2
profile_probability_strata$structural_zeros <- 0L
write_csv(profile_probability_strata, file.path(output_dir, "profile_probability_strata.csv"))

write_csv(validation, file.path(output_dir, "design_validation.csv"))
assert(all(validation$pass), "At least one design-contract validation failed; inspect design_validation.csv")

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
  gate_result = "green",
  outcome_blind = TRUE,
  outcomes_loaded = FALSE,
  outcome_models_fit = FALSE,
  read_method = "data.table::fread(select=locked_allowlist)",
  waves = list(
    wave_2016 = list(design_complete_respondents = 18021, primary_weight_eligible_respondents = 17874),
    wave_2022 = list(design_complete_respondents = 14966, primary_weight_eligible_respondents = 14846)
  ),
  registered_family_dimensions = list(wave_2016 = 34, wave_2022 = 38, common_support = 34),
  generated_files_md5 = as.list(generated_md5),
  note = "No pref, rate, or ratebin value was loaded, summarized, or modeled."
)
yaml::write_yaml(manifest, old_manifest)

cat(sprintf(
  "Asylum design audit passed: 2016 %s design-complete respondents; 2022 %s; six wave-family bases identified; no outcome loaded.\n",
  format(sample_summary$design_complete_respondents[sample_summary$wave == 2016], big.mark = ","),
  format(sample_summary$design_complete_respondents[sample_summary$wave == 2022], big.mark = ",")
))
